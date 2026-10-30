"""QLoRA fine-tuning of a decoder-only LLM (Llama-3.2-3B-Instruct or Gemma-2-2B-it).

Identical recipe for both: 4-bit NF4 base, LoRA r=16 on all projections, 20k pairs, 1 epoch,
loss only on the translation tokens.
    python 03_finetune_llm.py --model llama      # ~1.5-2 h on an RTX 3080 laptop
    python 03_finetune_llm.py --model gemma
    python 03_finetune_llm.py --model llama --resume
Saves checkpoints/<model>/adapter and outputs/<model>_train.json
"""
import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

from common import CKPT, LLM_IDS, N_TRAIN_LLM, OUT, gpu_name, load_split, peak_mem_gb, save_json, set_seed, timer
from llm_utils import build_example, load_llm


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        L = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), L), self.pad_id, dtype=torch.long)
        labels = torch.full((len(feats), L), -100, dtype=torch.long)
        mask = torch.zeros((len(feats), L), dtype=torch.long)
        for i, f in enumerate(feats):
            n = len(f["input_ids"])
            ids[i, :n] = torch.tensor(f["input_ids"])
            labels[i, :n] = torch.tensor(f["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(LLM_IDS), required=True)
    ap.add_argument("--n_train", type=int, default=N_TRAIN_LLM)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=None, help="default: 4 for llama, 2 for gemma (256k vocab)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    set_seed()
    batch = args.batch or (4 if args.model == "llama" else 2)
    accum = 16 // batch  # effective batch 16 for both models
    out_dir = CKPT / args.model

    model, tok = load_llm(args.model)
    total_base = sum(p.numel() * (2 if p.dtype == torch.uint8 else 1) for p in model.parameters())  # 4-bit packs 2/byte
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.print_trainable_parameters()

    src, tgt = load_split("train", args.n_train)
    dev_src, dev_tgt = load_split("dev")
    train_ds = Dataset.from_list([build_example(tok, s, t) for s, t in zip(src, tgt)])
    dev_ds = Dataset.from_list([build_example(tok, s, t) for s, t in zip(dev_src[:300], dev_tgt[:300])])
    print(f"{len(train_ds):,} training examples; avg length "
          f"{sum(map(len, train_ds['input_ids'])) / len(train_ds):.0f} tokens")

    targs = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=batch, per_device_eval_batch_size=batch, gradient_accumulation_steps=accum,
        lr_scheduler_type="cosine", warmup_steps=0.03, optim="paged_adamw_8bit", bf16=True,
        train_sampling_strategy="group_by_length", eval_strategy="steps", eval_steps=250, save_strategy="steps", save_steps=250,
        save_total_limit=2, logging_steps=25, report_to="none", dataloader_num_workers=0,
        remove_unused_columns=False, seed=42)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                      data_collator=Collator(tok.pad_token_id))

    log_path = OUT / f"{args.model}_train.json"
    prev = json.loads(log_path.read_text(encoding="utf-8")).get("train_seconds", 0.0) \
        if args.resume and log_path.exists() else 0.0
    t = {}
    torch.cuda.reset_peak_memory_stats()
    with timer(t, "train_seconds"):
        trainer.train(resume_from_checkpoint=args.resume or None)
    model.save_pretrained(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))

    save_json(log_path, {
        "model": f"{args.model} (QLoRA)", "hf_id": LLM_IDS[args.model], "gpu": gpu_name(),
        "params_total": total_base, "params_trainable": trainable, "train_pairs": len(train_ds),
        "epochs": args.epochs, "steps": trainer.state.global_step,
        "train_seconds": round(prev + t["train_seconds"], 1), "peak_train_mem_gb": peak_mem_gb(),
        "final_dev_loss": trainer.evaluate().get("eval_loss"),
        "adaptation": f"QLoRA (4-bit NF4, LoRA r={args.rank}, all linear layers)"})
    print("Done.")


if __name__ == "__main__":
    main()
