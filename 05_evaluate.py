"""Score every run on the test set and build the comparison tables + plots.

    python 05_evaluate.py
Writes results/metrics.csv, results/metrics.md, results/fertility.md, results/significance.txt, results/*.png
"""
import json
import subprocess
import sys

import sacrebleu

from common import DATA, LLM_IDS, MT5_ID, OUT, RESULTS, SRC, TGT, read_lines

RUNS = [  # (run name, architecture, training-log prefix)
    ("transformer", "encoder-decoder", "transformer"),
    ("mt5", "encoder-decoder", "mt5"),
    ("llama_zeroshot", "decoder-only", None),
    ("llama_qlora", "decoder-only", "llama"),
    ("gemma_zeroshot", "decoder-only", None),
    ("gemma_qlora", "decoder-only", "gemma"),
]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fertility():
    """Average subword tokens per sentence: shows how each tokenizer segments German/English."""
    src, tgt = read_lines(DATA / f"test.{SRC}"), read_lines(DATA / f"test.{TGT}")
    rows = []
    toks = {}
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(DATA / "spm_joint.model"))
        toks["Transformer (joint BPE)"] = (sp.get_piece_size(), lambda s: len(sp.encode(s)))
    except Exception as e:
        print("spm:", e)
    from transformers import AutoTokenizer
    for label, hf_id in [("mT5", MT5_ID), ("Llama-3.2", LLM_IDS["llama"]), ("Gemma-2", LLM_IDS["gemma"])]:
        try:
            t = AutoTokenizer.from_pretrained(hf_id)
            toks[label] = (len(t), lambda s, t=t: len(t(s, add_special_tokens=False)["input_ids"]))
        except Exception as e:
            print(f"{label} tokenizer:", e)
    words_s = sum(len(s.split()) for s in src)
    words_t = sum(len(s.split()) for s in tgt)
    for label, (vocab, f) in toks.items():
        ns, nt = sum(map(f, src)), sum(map(f, tgt))
        rows.append(f"| {label} | {vocab:,} | {ns/len(src):.1f} | {ns/words_s:.2f} | {nt/len(tgt):.1f} | {nt/words_t:.2f} |")
    md = ["| Tokenizer | Vocab | {0} tok/sent | {0} tok/word | {1} tok/sent | {1} tok/word |".format(SRC, TGT),
          "|---|---|---|---|---|---|"] + rows
    (RESULTS / "fertility.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


BLEU_M, CHRF_M = sacrebleu.BLEU(), sacrebleu.CHRF()  # signatures need a scored metric


def main():
    refs = read_lines(DATA / f"test.{TGT}")
    rows, present = [], []
    for run, arch, train_key in RUNS:
        hyp_path = OUT / f"{run}.test.{TGT}"
        if not hyp_path.exists():
            print(f"skip {run}: no outputs yet")
            continue
        hyps = read_lines(hyp_path)
        assert len(hyps) == len(refs), f"{run}: {len(hyps)} lines vs {len(refs)} refs"
        present.append(hyp_path)
        bleu = BLEU_M.corpus_score(hyps, [refs])
        chrf = CHRF_M.corpus_score(hyps, [refs])
        chrfpp = sacrebleu.CHRF(word_order=2).corpus_score(hyps, [refs])
        tr = load_json(OUT / f"{train_key}_train.json") if train_key else {}
        inf = load_json(OUT / f"{run}_infer.json")
        rows.append({
            "run": run, "architecture": arch, "BLEU": round(bleu.score, 2), "chrF": round(chrf.score, 2),
            "chrF++": round(chrfpp.score, 2),
            "params_total_M": round((tr.get("params_total") or inf.get("params_total", 0)) / 1e6, 1),
            "params_trainable_M": round(tr.get("params_trainable", 0) / 1e6, 2),
            "train_pairs": tr.get("train_pairs", 0),
            # mT5's first steps ran under memory spill; use its extrapolated clean-run time
            "train_min": round((tr.get("est_full_train_seconds") or tr.get("train_seconds", 0)) / 60, 1),
            "train_peak_gb": tr.get("peak_train_mem_gb", ""),
            "infer_s": inf.get("infer_seconds", ""), "ms_per_sent": inf.get("ms_per_sentence", ""),
            "infer_peak_gb": inf.get("peak_infer_mem_gb", ""),
            "adaptation": tr.get("adaptation", "zero-shot prompting" if "zeroshot" in run else ""),
        })
    if not rows:
        print("No outputs found in outputs/. Run 04_translate.py first.")
        return

    cols = list(rows[0])
    with open(RESULTS / "metrics.csv", "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f'"{r[c]}"' if "," in str(r[c]) else str(r[c]) for c in cols) + "\n")
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    sig = f"\nBLEU signature: {BLEU_M.get_signature()}\nchrF signature: {CHRF_M.get_signature()}\n"
    (RESULTS / "metrics.md").write_text("\n".join(md) + "\n" + sig, encoding="utf-8")
    print("\n".join(md))

    if len(present) > 1:  # paired bootstrap resampling, first system = baseline
        cmd = [sys.executable, "-m", "sacrebleu", str(DATA / f"test.{TGT}"), "-i", *map(str, present),
               "-m", "bleu", "chrf", "--paired-bs", "-f", "text"]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        (RESULTS / "significance.txt").write_text(res.stdout + res.stderr, encoding="utf-8")
        print(res.stdout)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [r["run"] for r in rows]
        colors = ["#4C72B0" if r["architecture"] == "encoder-decoder" else "#DD8452" for r in rows]
        for key, ylabel, fname in [("BLEU", "BLEU", "bleu.png"), ("chrF", "chrF", "chrf.png"),
                                   ("ms_per_sent", "ms per sentence (greedy)", "inference_time.png"),
                                   ("params_total_M", "parameters (M)", "params.png")]:
            vals = [float(r[key] or 0) for r in rows]
            plt.figure(figsize=(8, 4))
            plt.bar(names, vals, color=colors)
            plt.ylabel(ylabel)
            plt.xticks(rotation=20, ha="right")
            for i, v in enumerate(vals):
                plt.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=8)
            plt.title(f"{ylabel}  (blue = encoder-decoder, orange = decoder-only)")
            plt.tight_layout()
            plt.savefig(RESULTS / fname, dpi=150)
            plt.close()
    except ImportError:
        print("matplotlib not installed; skipping plots")

    fertility()


if __name__ == "__main__":
    main()
