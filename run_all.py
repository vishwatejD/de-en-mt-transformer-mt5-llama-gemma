"""Run the whole pipeline in order, skipping steps that already finished.

    .venv\\Scripts\\python run_all.py            # everything (~9-10 GPU-hours)
    .venv\\Scripts\\python run_all.py --from 4   # start at step 4
Each step logs to logs/<step>.log. Re-running after a crash resumes the step that failed.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import CKPT, OUT, ROOT

LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

# (name, command, "done" marker file, supports --resume)
STEPS = [
    ("01_transformer_train", ["01_train_transformer.py"], OUT / "transformer_train.json", True),
    ("02_transformer_translate", ["04_translate.py", "--model", "transformer"], OUT / "transformer_infer.json", False),
    ("03_mt5_train", ["02_finetune_mt5.py"], CKPT / "mt5" / "final" / "config.json", True),
    ("04_mt5_translate", ["04_translate.py", "--model", "mt5"], OUT / "mt5_infer.json", False),
    ("05_llama_zeroshot", ["04_translate.py", "--model", "llama", "--zeroshot"], OUT / "llama_zeroshot_infer.json", False),
    ("06_llama_train", ["03_finetune_llm.py", "--model", "llama"], CKPT / "llama" / "adapter" / "adapter_config.json", True),
    ("07_llama_translate", ["04_translate.py", "--model", "llama"], OUT / "llama_qlora_infer.json", False),
    ("08_gemma_zeroshot", ["04_translate.py", "--model", "gemma", "--zeroshot"], OUT / "gemma_zeroshot_infer.json", False),
    ("09_gemma_train", ["03_finetune_llm.py", "--model", "gemma"], CKPT / "gemma" / "adapter" / "adapter_config.json", True),
    ("10_gemma_translate", ["04_translate.py", "--model", "gemma"], OUT / "gemma_qlora_infer.json", False),
    ("11_evaluate", ["05_evaluate.py"], None, False),
    ("12_plots", ["06_plots.py"], None, False),
]


def has_checkpoint(name):
    d = {"01": CKPT / "transformer" / "last.pt", "03": CKPT / "mt5", "06": CKPT / "llama", "09": CKPT / "gemma"}.get(name[:2])
    return d is not None and d.exists() and (d.is_file() or any(d.glob("checkpoint-*")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1)
    args = ap.parse_args()
    for i, (name, cmd, marker, resumable) in enumerate(STEPS, 1):
        if i < args.start or (marker is not None and marker.exists()):
            print(f"[skip] {name}")
            continue
        full = [sys.executable, "-u", *cmd] + (["--resume"] if resumable and has_checkpoint(name) else [])
        print(f"[run ] {name}: {' '.join(full[2:])}  (log: logs/{name}.log)", flush=True)
        t0 = time.time()
        with open(LOGS / f"{name}.log", "a", encoding="utf-8") as log:
            proc = subprocess.Popen(full, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
            proc.wait()
        print(f"[{'done' if proc.returncode == 0 else 'FAIL'}] {name} in {(time.time()-t0)/60:.1f} min", flush=True)
        if proc.returncode != 0:
            sys.exit(f"Step {i} failed; fix it and re-run (finished steps are skipped).")


if __name__ == "__main__":
    main()
