"""Fully fine-tune mT5-base (encoder-decoder, 580M params) on a 60k-pair subset.

Adafactor + bf16 so it fits in 8 GB VRAM (AdamW would need ~9 GB).
    python 02_finetune_mt5.py            # ~2 h on an RTX 3080 laptop
    python 02_finetune_mt5.py --resume
Saves checkpoints/mt5/final and outputs/mt5_train.json
"""
import argparse
import json

import torch
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

from common import (CKPT, MT5_ID, MT5_PREFIX, N_TRAIN_MT5, OUT, count_params, gpu_name,
                    load_split, peak_mem_gb, save_json, set_seed, timer)

DIR = CKPT / "mt5"
MAX_LEN = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=N_TRAIN_MT5)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    set_seed()

    tok = AutoTokenizer.from_pretrained(MT5_ID)
    # bf16 weights: fp32 weights+grads (4.6 GB) overflow 8 GB VRAM into shared memory and run ~10x slower
    model = AutoModelForSeq2SeqLM.from_pretrained(MT5_ID, dtype=torch.bfloat16)
    total, _ = count_params(model)

    def encode(batch):
        enc = tok([MT5_PREFIX + s for s in batch["src"]], max_length=MAX_LEN, truncation=True)
        enc["labels"] = tok(text_target=batch["tgt"], max_length=MAX_LEN, truncation=True)["input_ids"]
        return enc

    tr_src, tr_tgt = load_split("train", args.n_train)
    dev_src, dev_tgt = load_split("dev")
    train_ds = Dataset.from_dict({"src": tr_src, "tgt": tr_tgt}).map(encode, batched=True, remove_columns=["src", "tgt"])
    dev_ds = Dataset.from_dict({"src": dev_src, "tgt": dev_tgt}).map(encode, batched=True, remove_columns=["src", "tgt"])

    targs = Seq2SeqTrainingArguments(
        output_dir=str(DIR), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=16,
        gradient_accumulation_steps=args.accum, optim="adafactor", lr_scheduler_type="linear",
        warmup_steps=300, weight_decay=0.0, max_grad_norm=1.0, bf16=True,
        gradient_checkpointing=True, train_sampling_strategy="group_by_length",
        eval_strategy="steps", eval_steps=500, save_strategy="steps", save_steps=500, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        logging_steps=50, report_to="none", dataloader_num_workers=0, seed=42)
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                             data_collator=DataCollatorForSeq2Seq(tok, model=model), processing_class=tok)

    prev = 0.0
    log_path = OUT / "mt5_train.json"
    if args.resume and log_path.exists():
        prev = json.loads(log_path.read_text(encoding="utf-8")).get("train_seconds", 0.0)
    from transformers.trainer_utils import get_last_checkpoint
    last = get_last_checkpoint(str(DIR)) if args.resume else None
    start_step = int(last.rsplit("-", 1)[1]) if last else 0
    t = {}
    torch.cuda.reset_peak_memory_stats()
    with timer(t, "train_seconds"):
        trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(DIR / "final"))
    tok.save_pretrained(str(DIR / "final"))

    save_json(log_path, {
        "model": "mT5-base (full fine-tune)", "hf_id": MT5_ID, "gpu": gpu_name(),
        "params_total": total, "params_trainable": total, "train_pairs": len(train_ds),
        "epochs": args.epochs, "steps": trainer.state.global_step,
        "train_seconds": round(prev + t["train_seconds"], 1), "peak_train_mem_gb": peak_mem_gb(),
        # steps before a resume may have run under different conditions; extrapolate from the timed steps
        "steps_timed": trainer.state.global_step - start_step,
        "est_full_train_seconds": round(t["train_seconds"] / max(1, trainer.state.global_step - start_step)
                                        * trainer.state.global_step, 1),
        "best_dev_loss": trainer.state.best_metric, "adaptation": "full fine-tuning (Adafactor, bf16)"})
    print("Done:", trainer.state.best_metric)


if __name__ == "__main__":
    main()
