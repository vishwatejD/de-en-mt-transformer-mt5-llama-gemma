"""Extra comparison charts built from the files 05_evaluate.py and the training scripts already wrote.

    python 06_plots.py
Reads results/metrics.csv, results/fertility.md and outputs/transformer_train.json; writes results/*.png
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT, RESULTS

ENC_DEC, DEC_ONLY = "#4C72B0", "#DD8452"
LABELS = {"transformer": "Transformer", "mt5": "mT5-base", "llama_zeroshot": "Llama zero-shot",
          "llama_qlora": "Llama QLoRA", "gemma_zeroshot": "Gemma zero-shot", "gemma_qlora": "Gemma QLoRA"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS / name, dpi=150)
    plt.close(fig)


def quality_vs_speed(rows):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows:
        x, y = float(r["ms_per_sent"]), float(r["BLEU"])
        size = 40 + float(r["params_total_M"]) / 10  # marker area grows with model size
        ax.scatter(x, y, s=size, alpha=0.8, edgecolor="black",
                   color=ENC_DEC if r["architecture"] == "encoder-decoder" else DEC_ONLY)
        offset = {"llama_qlora": (14, -22), "gemma_qlora": (-105, 6)}.get(r["run"], (12, -4))
        ax.annotate(f"{LABELS[r['run']]}\n{y:.1f} BLEU, {x:g} ms", (x, y), textcoords="offset points",
                    xytext=offset, fontsize=8,
                    arrowprops={"arrowstyle": "-", "lw": 0.6} if r["run"] == "gemma_qlora" else None)
    ax.set_xscale("log")
    ax.set_xlabel("inference time per sentence (ms, log scale) — lower is faster")
    ax.set_ylabel("BLEU — higher is better")
    ax.set_title("Quality vs speed (marker size ∝ total parameters)\nblue = encoder-decoder, orange = decoder-only")
    ax.set_xlim(5, 900)
    ax.set_ylim(23, 42.5)
    ax.grid(alpha=0.3)
    save(fig, "quality_vs_speed.png")


def trainable_params(rows):
    rows = [r for r in rows if "zeroshot" not in r["run"]]
    names = [LABELS[r["run"]] for r in rows]
    total = [float(r["params_total_M"]) for r in rows]
    train = [float(r["params_trainable_M"]) for r in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - 0.2 for i in x], total, 0.4, label="total parameters", color="#B0B0B0")
    ax.bar([i + 0.2 for i in x], train, 0.4, label="trainable parameters", color="#55A868")
    for i, (t, p) in enumerate(zip(total, train)):
        ax.text(i - 0.2, t, f"{t:,.0f}M", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, p, f"{p:,.1f}M\n({min(100, round(100 * p / t, 2)) if p < 0.99 * t else 100:g}%)", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(top=max(total) * 8)
    ax.set_xticks(list(x), names)
    ax.set_ylabel("parameters (millions, log scale)")
    ax.set_title("Total vs trainable parameters by adaptation method")
    ax.legend(loc="upper left")
    save(fig, "trainable_params.png")


def transformer_curve():
    hist = json.loads((OUT / "transformer_train.json").read_text(encoding="utf-8"))["history"]
    ep = [h["epoch"] for h in hist]
    bleu = [h["dev_bleu"] for h in hist]
    best = max(range(len(hist)), key=lambda i: bleu[i])
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(ep, bleu, "o-", color=ENC_DEC, label="dev BLEU")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("dev BLEU", color=ENC_DEC)
    ax1.annotate(f"best: {bleu[best]} (epoch {ep[best]})", (ep[best], bleu[best]),
                 xytext=(ep[best] - 7, bleu[best] - 6), arrowprops={"arrowstyle": "->"}, fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["train_loss"] for h in hist], "s--", color="#C44E52", label="train loss")
    ax2.set_ylabel("train loss (label-smoothed)", color="#C44E52")
    ax1.axvline(ep[-1], color="gray", ls=":", lw=1)
    ax1.text(ep[-1] - 0.3, 2, "early stop", rotation=90, ha="right", fontsize=8, color="gray")
    ax1.grid(alpha=0.3)
    ax1.set_title("From-scratch Transformer: training curve")
    save(fig, "transformer_training_curve.png")


def fertility():
    lines = [l for l in (RESULTS / "fertility.md").read_text(encoding="utf-8").splitlines() if l.startswith("|")][2:]
    cells = [[c.strip() for c in l.strip("|").split("|")] for l in lines]
    names = [c[0] for c in cells]
    de, en = [float(c[3]) for c in cells], [float(c[5]) for c in cells]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - 0.2 for i in x], de, 0.4, label="German", color="#8172B2")
    ax.bar([i + 0.2 for i in x], en, 0.4, label="English", color="#64B5CD")
    for i, (d, e) in enumerate(zip(de, en)):
        ax.text(i - 0.2, d, f"{d:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, e, f"{e:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x), [f"{n}\n({c[1]} vocab)" for n, c in zip(names, cells)], fontsize=8)
    ax.set_ylabel("subword tokens per word (lower = more efficient)")
    ax.set_ylim(1, max(de + en) * 1.12)
    ax.set_title("Tokenizer fertility on the test set")
    ax.legend()
    save(fig, "tokenizer_fertility.png")


def adaptation_gain(rows):
    by = {r["run"]: r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, metric in zip(axes, ["BLEU", "chrF"]):
        for i, m in enumerate(["llama", "gemma"]):
            zs, ft = float(by[f"{m}_zeroshot"][metric]), float(by[f"{m}_qlora"][metric])
            ax.bar(i - 0.2, zs, 0.4, color="#CCB974", label="zero-shot" if i == 0 else None)
            ax.bar(i + 0.2, ft, 0.4, color=DEC_ONLY, label="QLoRA (20k pairs)" if i == 0 else None)
            ax.text(i - 0.2, zs, f"{zs:.1f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + 0.2, ft, f"{ft:.1f}\n(+{ft - zs:.1f})", ha="center", va="bottom", fontsize=8)
        for base, color in [("transformer", "#4C72B0"), ("mt5", "#55A868")]:
            ax.axhline(float(by[base][metric]), color=color, ls="--", lw=1, label=f"{LABELS[base]} ({by[base][metric]})")
        ax.set_xticks([0, 1], ["Llama-3.2-3B", "Gemma-2-2B"])
        ax.set_title(metric)
        ax.set_ylim(0, max(float(r[metric]) for r in rows) * 1.18)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Effect of QLoRA fine-tuning over zero-shot prompting")
    save(fig, "zeroshot_vs_qlora.png")


def main():
    with open(RESULTS / "metrics.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    quality_vs_speed(rows)
    trainable_params(rows)
    transformer_curve()
    fertility()
    adaptation_gain(rows)
    print("wrote 5 charts to", RESULTS)


if __name__ == "__main__":
    main()
