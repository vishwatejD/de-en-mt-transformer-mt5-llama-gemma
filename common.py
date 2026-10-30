"""Shared paths, I/O helpers, timing and the LLM prompt used by every script."""
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"        # translations + timing JSONs
CKPT = ROOT / "checkpoints"   # model weights
RESULTS = ROOT / "results"    # final tables / plots
for d in (DATA, OUT, CKPT, RESULTS):
    d.mkdir(exist_ok=True)

SRC, TGT = "de", "en"
SRC_NAME, TGT_NAME = "German", "English"
SEED = 42

# Training-set sizes (nested prefixes of the shuffled cleaned train set)
N_TRAIN_TRANSFORMER = None   # all
N_TRAIN_MT5 = 60_000
N_TRAIN_LLM = 20_000

LLM_IDS = {
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma": "google/gemma-2-2b-it",
}
MT5_ID = "google/mt5-base"
MT5_PREFIX = f"translate {SRC_NAME} to {TGT_NAME}: "

# Windows consoles default to cp1252; force UTF-8 so German text prints.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def llm_messages(src, tgt=None):
    """Chat messages for the decoder-only models (same prompt for zero-shot and fine-tuning)."""
    msgs = [{"role": "user", "content":
             f"Translate the following {SRC_NAME} text into {TGT_NAME}. "
             f"Reply with the {TGT_NAME} translation only.\n\n{src}"}]
    if tgt is not None:
        msgs.append({"role": "assistant", "content": tgt})
    return msgs


# Fixed date so Llama-3's chat template renders identically on every day.
CHAT_TEMPLATE_KWARGS = {"date_string": "26 Jul 2024"}


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line.replace("\n", " ") + "\n")


def load_split(split, n=None):
    src = read_lines(DATA / f"{split}.{SRC}")
    tgt = read_lines(DATA / f"{split}.{TGT}")
    if n is not None:
        src, tgt = src[:n], tgt[:n]
    return src, tgt


def save_json(path, obj):
    path = Path(path)
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(obj)
    path.write_text(json.dumps(old, indent=2), encoding="utf-8")


def set_seed(seed=SEED):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def gpu_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


@contextmanager
def timer(result: dict, key: str):
    """Wall-clock timer that synchronises CUDA so GPU work is fully counted."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    result[key] = round(time.perf_counter() - t0, 2)


def peak_mem_gb():
    import torch
    return round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else 0.0
