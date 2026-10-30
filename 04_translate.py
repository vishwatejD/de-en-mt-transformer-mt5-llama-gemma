"""Translate the test set with one model under identical settings (greedy decoding, same GPU).

    python 04_translate.py --model transformer
    python 04_translate.py --model mt5
    python 04_translate.py --model llama --zeroshot      # prompting baseline (no fine-tuning)
    python 04_translate.py --model llama                 # QLoRA adapter
    python 04_translate.py --model gemma --zeroshot
    python 04_translate.py --model gemma
Writes outputs/<run>.<split>.en and outputs/<run>_infer.json
"""
import argparse

import torch

from common import CKPT, DATA, MT5_ID, MT5_PREFIX, OUT, SRC, TGT, count_params, gpu_name, peak_mem_gb, \
    read_lines, save_json, timer, write_lines

MAX_NEW = 256
WARMUP = 4  # untimed sentences to initialise CUDA kernels


def run_transformer(srcs, bs):
    import sentencepiece as spm
    from transformer_model import load_transformer, translate_batch
    sp = spm.SentencePieceProcessor(model_file=str(DATA / "spm_joint.model"))
    model = load_transformer(CKPT / "transformer" / "best.pt")
    return model, lambda xs: translate_batch(model, sp, xs, batch_size=bs)


def run_mt5(srcs, bs):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    path = CKPT / "mt5" / "final"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()

    def fn(xs):
        order = sorted(range(len(xs)), key=lambda i: -len(xs[i]))
        res = [None] * len(xs)
        for k in range(0, len(order), bs):
            idx = order[k:k + bs]
            enc = tok([MT5_PREFIX + xs[i] for i in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, num_beams=1, do_sample=False, max_new_tokens=MAX_NEW)
            for i, g in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                res[i] = g.strip()
        return res
    return model, fn


def run_llm(name, zeroshot, bs):
    from llm_utils import clean_output, load_llm, prompt_text, stop_token_ids
    model, tok = load_llm(name, None if zeroshot else CKPT / name / "adapter")
    model.eval()
    tok.padding_side = "left"
    stops = stop_token_ids(tok)

    def fn(xs):
        order = sorted(range(len(xs)), key=lambda i: -len(xs[i]))
        res = [None] * len(xs)
        for k in range(0, len(order), bs):
            idx = order[k:k + bs]
            enc = tok([prompt_text(tok, xs[i]) for i in idx], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None,
                                     max_new_tokens=MAX_NEW, eos_token_id=stops, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, t in zip(idx, texts):
                res[i] = clean_output(t)
            print(f"  {min(k + bs, len(xs))}/{len(xs)}", flush=True)
        return res
    return model, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["transformer", "mt5", "llama", "gemma"], required=True)
    ap.add_argument("--zeroshot", action="store_true", help="LLMs only: no adapter, prompting only")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()
    is_llm = args.model in ("llama", "gemma")
    run = args.model + (("_zeroshot" if args.zeroshot else "_qlora") if is_llm else "")
    bs = args.batch_size or {"transformer": 64, "mt5": 32}.get(args.model, 16)

    srcs = read_lines(DATA / f"{args.split}.{SRC}")
    info = {}
    with timer(info, "load_seconds"):
        if args.model == "transformer":
            model, fn = run_transformer(srcs, bs)
        elif args.model == "mt5":
            model, fn = run_mt5(srcs, bs)
        else:
            model, fn = run_llm(args.model, args.zeroshot, bs)
    fn(srcs[:WARMUP])
    torch.cuda.reset_peak_memory_stats()
    with timer(info, "infer_seconds"):
        hyps = fn(srcs)

    total, trainable = count_params(model)
    if is_llm:  # bitsandbytes stores two 4-bit weights per uint8 element
        total = sum(p.numel() * (2 if p.dtype == torch.uint8 else 1) for p in model.parameters())
    write_lines(OUT / f"{run}.{args.split}.{TGT}", hyps)
    save_json(OUT / f"{run}_infer.json", {
        "run": run, "split": args.split, "gpu": gpu_name(), "sentences": len(srcs), "batch_size": bs,
        "decoding": "greedy", **info,
        "sentences_per_second": round(len(srcs) / info["infer_seconds"], 2),
        "ms_per_sentence": round(1000 * info["infer_seconds"] / len(srcs), 1),
        "peak_infer_mem_gb": peak_mem_gb(), "params_total": total})
    print(f"{run}: {len(srcs)} sentences in {info['infer_seconds']:.1f}s")
    for s, h in list(zip(srcs, hyps))[:3]:
        print(f"  {SRC}: {s}\n  {TGT}: {h}")


if __name__ == "__main__":
    main()
