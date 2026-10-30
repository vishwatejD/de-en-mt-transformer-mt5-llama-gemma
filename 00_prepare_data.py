"""Download IWSLT 2017 German-English, clean it, and write train/dev/test splits.

Outputs data/{train,dev,test}.{de,en} and data/stats.json (numbers for the report).
    python 00_prepare_data.py
"""
import random
import re
import statistics
import unicodedata

from common import DATA, SEED, SRC, TGT, write_lines, save_json

REPO = "IWSLT/iwslt2017"
CONFIG = f"iwslt2017-{SRC}-{TGT}"
N_TEST = 1000          # fixed random subset of the official test set (same for all models)
MAX_WORDS = 80
MAX_RATIO = 2.5


def load_raw():
    """Return {split: [(src, tgt), ...]} using HF's auto-converted Parquet (no loading script needed)."""
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    try:
        files = HfApi().list_repo_files(REPO, repo_type="dataset", revision="refs/convert/parquet")
        out = {}
        for split in ("train", "validation", "test"):
            parts = sorted(f for f in files if f.startswith(f"{CONFIG}/{split}/") and f.endswith(".parquet"))
            if not parts:
                raise FileNotFoundError(f"no parquet for {split}")
            df = pd.concat(pd.read_parquet(hf_hub_download(REPO, p, repo_type="dataset",
                                                           revision="refs/convert/parquet")) for p in parts)
            out[split] = [(t[SRC], t[TGT]) for t in df["translation"]]
        return out
    except Exception as e:
        print(f"Parquet route failed ({e}); falling back to the loading script.")
        from datasets import load_dataset
        ds = load_dataset(REPO, CONFIG, trust_remote_code=True)
        return {s: [(t[SRC], t[TGT]) for t in ds[s]["translation"]] for s in ("train", "validation", "test")}


def normalize(s):
    s = unicodedata.normalize("NFC", s)
    s = s.replace(" ", " ").replace("​", "")
    return re.sub(r"\s+", " ", s).strip()


def keep(src, tgt):
    if not src or not tgt:
        return "empty"
    ls, lt = len(src.split()), len(tgt.split())
    if ls > MAX_WORDS or lt > MAX_WORDS:
        return "too_long"
    if min(ls, lt) > 3 and max(ls, lt) / min(ls, lt) > MAX_RATIO:
        return "bad_ratio"
    return None


def length_stats(pairs):
    s = [len(a.split()) for a, _ in pairs]
    t = [len(b.split()) for _, b in pairs]
    return {"pairs": len(pairs),
            f"{SRC}_words_mean": round(statistics.mean(s), 1), f"{TGT}_words_mean": round(statistics.mean(t), 1),
            f"{SRC}_words_median": statistics.median(s), f"{TGT}_words_median": statistics.median(t),
            f"{SRC}_vocab": len({w for a, _ in pairs for w in a.split()}),
            f"{TGT}_vocab": len({w for _, b in pairs for w in b.split()})}


def main():
    raw = load_raw()
    stats = {"raw": {k: len(v) for k, v in raw.items()}}

    dev = [(normalize(a), normalize(b)) for a, b in raw["validation"]]
    test_full = [(normalize(a), normalize(b)) for a, b in raw["test"]]
    test_full = [p for p in test_full if keep(*p) is None]
    rng = random.Random(SEED)
    test = rng.sample(test_full, N_TEST)
    held_out_src = {a for a, _ in dev} | {a for a, _ in test_full}

    drops, seen, train = {}, set(), []
    for a, b in raw["train"]:
        a, b = normalize(a), normalize(b)
        reason = keep(a, b)
        if reason is None and (a, b) in seen:
            reason = "duplicate"
        if reason is None and a in held_out_src:
            reason = "overlaps_dev_test"
        if reason:
            drops[reason] = drops.get(reason, 0) + 1
            continue
        seen.add((a, b))
        train.append((a, b))
    rng.shuffle(train)  # shuffled once; smaller training sets are prefixes of this

    for name, pairs in (("train", train), ("dev", dev), ("test", test)):
        write_lines(DATA / f"{name}.{SRC}", [a for a, _ in pairs])
        write_lines(DATA / f"{name}.{TGT}", [b for _, b in pairs])
        stats[name] = length_stats(pairs)
    stats["train_dropped"] = drops
    stats["test_source"] = f"random {N_TEST} of {len(test_full)} official test pairs (seed {SEED})"
    save_json(DATA / "stats.json", stats)

    print(f"train {len(train):,}  dev {len(dev):,}  test {len(test):,}")
    print("dropped from train:", drops)
    for a, b in train[:3]:
        print(f"  {SRC}: {a}\n  {TGT}: {b}")


if __name__ == "__main__":
    main()
