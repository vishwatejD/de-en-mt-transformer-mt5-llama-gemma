"""Train the basic encoder-decoder Transformer from scratch (joint 10k BPE vocabulary).

    python 01_train_transformer.py                 # ~1.5-2.5 h on an RTX 3080 laptop
    python 01_train_transformer.py --resume        # continue after an interruption
Saves checkpoints/transformer/{best,last}.pt and outputs/transformer_train.json
"""
import argparse
import random
import time
import warnings

import sacrebleu
import sentencepiece as spm
import torch
import torch.nn as nn

from common import (CKPT, DATA, OUT, SRC, TGT, N_TRAIN_TRANSFORMER, count_params, gpu_name,
                    load_split, peak_mem_gb, save_json, set_seed)
from transformer_model import BOS, EOS, PAD, Seq2SeqTransformer, translate_batch

warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
DIR = CKPT / "transformer"
SPM_PREFIX = DATA / "spm_joint"
MAX_LEN = 128


def train_spm(src, tgt, vocab_size):
    if SPM_PREFIX.with_suffix(".model").exists():
        return
    corpus = DATA / "spm_corpus.txt"
    corpus.write_text("\n".join(src + tgt) + "\n", encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(SPM_PREFIX), vocab_size=vocab_size, model_type="bpe",
        character_coverage=1.0, pad_id=PAD, unk_id=1, bos_id=BOS, eos_id=EOS,
        input_sentence_size=1_000_000, shuffle_input_sentence=True, num_threads=8)
    corpus.unlink()


def make_batches(pairs, max_tokens):
    """Group length-sorted examples into batches of ~max_tokens (padded) tokens."""
    pairs = sorted(pairs, key=lambda p: (len(p[0]), len(p[1])))
    batches, cur, longest = [], [], 0
    for p in pairs:
        L = max(len(p[0]), len(p[1]) + 1)
        if cur and max(longest, L) * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, longest = [], 0
        cur.append(p)
        longest = max(longest, L)
    if cur:
        batches.append(cur)
    return batches


def collate(batch, device):
    S = max(len(s) for s, _ in batch)
    T = max(len(t) for _, t in batch) + 1
    src = torch.full((len(batch), S), PAD, dtype=torch.long)
    tin = torch.full((len(batch), T), PAD, dtype=torch.long)
    tout = torch.full((len(batch), T), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src[i, : len(s)] = torch.tensor(s)
        tin[i, : len(t) + 1] = torch.tensor([BOS] + t)
        tout[i, : len(t) + 1] = torch.tensor(t + [EOS])
    return src.to(device, non_blocking=True), tin.to(device, non_blocking=True), tout.to(device, non_blocking=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max_hours", type=float, default=2.5, help="stop after this much training time")
    ap.add_argument("--patience", type=int, default=6, help="early stop: epochs without dev BLEU gain")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--vocab", type=int, default=10000)
    ap.add_argument("--n_train", type=int, default=N_TRAIN_TRANSFORMER)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    set_seed()
    DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    tr_src, tr_tgt = load_split("train", args.n_train)
    dev_src, dev_tgt = load_split("dev")
    train_spm(tr_src, tr_tgt, args.vocab)
    sp = spm.SentencePieceProcessor(model_file=str(SPM_PREFIX.with_suffix(".model")))

    print("Encoding training data ...")
    enc_s, enc_t = sp.encode(tr_src), sp.encode(tr_tgt)
    pairs = [(s[: MAX_LEN - 1] + [EOS], t[: MAX_LEN - 1]) for s, t in zip(enc_s, enc_t)]
    batches = make_batches(pairs, args.max_tokens)
    print(f"{len(pairs):,} pairs -> {len(batches):,} batches/epoch")

    model = Seq2SeqTransformer(sp.get_piece_size()).to(device)
    total, _ = count_params(model)
    print(f"Parameters: {total/1e6:.1f}M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / args.warmup, (args.warmup / (s + 1)) ** 0.5))
    crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    state = {"epoch": 0, "step": 0, "best_bleu": -1.0, "bad_epochs": 0, "train_seconds": 0.0, "history": []}
    if args.resume and (DIR / "last.pt").exists():
        ck = torch.load(DIR / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        state = ck["state"]
        print(f"Resumed from epoch {state['epoch']} (best dev BLEU {state['best_bleu']:.2f})")

    rng = random.Random(state["epoch"])
    while state["epoch"] < args.epochs:
        model.train()
        rng.shuffle(batches)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tot_loss, tot_tok = 0.0, 0
        for bi, batch in enumerate(batches):
            src, tin, tout = collate(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(src, tin)
            loss = crit(logits.float().view(-1, logits.size(-1)), tout.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            state["step"] += 1
            n = tout.ne(PAD).sum().item()
            tot_loss += loss.item() * n
            tot_tok += n
            if bi % 200 == 0:
                print(f"  ep {state['epoch']+1} batch {bi}/{len(batches)} loss {loss.item():.3f} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        torch.cuda.synchronize()
        state["train_seconds"] += time.perf_counter() - t0
        state["epoch"] += 1

        model.eval()
        hyps = translate_batch(model, sp, dev_src, device)
        bleu = sacrebleu.corpus_bleu(hyps, [dev_tgt]).score
        state["history"].append({"epoch": state["epoch"], "train_loss": round(tot_loss / tot_tok, 4),
                                 "dev_bleu": round(bleu, 2), "train_hours": round(state["train_seconds"] / 3600, 3)})
        print(f"Epoch {state['epoch']}: train loss {tot_loss/tot_tok:.3f}  dev BLEU {bleu:.2f}  "
              f"({state['train_seconds']/60:.1f} min trained)", flush=True)

        if bleu > state["best_bleu"]:
            state["best_bleu"], state["bad_epochs"] = bleu, 0
            torch.save({"config": model.config, "model": model.state_dict()}, DIR / "best.pt")
        else:
            state["bad_epochs"] += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "state": state}, DIR / "last.pt")
        if state["bad_epochs"] >= args.patience:
            print("Early stopping.")
            break
        if state["train_seconds"] / 3600 >= args.max_hours:
            print("Reached --max_hours.")
            break

    save_json(OUT / "transformer_train.json", {
        "model": "Transformer (from scratch)", "gpu": gpu_name(), "params_total": total, "params_trainable": total,
        "train_pairs": len(pairs), "epochs": state["epoch"], "steps": state["step"],
        "train_seconds": round(state["train_seconds"], 1), "best_dev_bleu": round(state["best_bleu"], 2),
        "peak_train_mem_gb": peak_mem_gb(), "vocab": sp.get_piece_size(), "history": state["history"],
        "adaptation": "full training from random init"})
    print(f"Done. Best dev BLEU {state['best_bleu']:.2f}")


if __name__ == "__main__":
    main()
