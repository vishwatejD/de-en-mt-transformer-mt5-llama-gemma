| run | architecture | BLEU | chrF | chrF++ | params_total_M | params_trainable_M | train_pairs | train_min | train_peak_gb | infer_s | ms_per_sent | infer_peak_gb | adaptation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| transformer | encoder-decoder | 24.49 | 49.06 | 47.85 | 36.7 | 36.67 | 204044 | 48.1 | 2.12 | 8.96 | 9.0 | 0.39 | full training from random init |
| mt5 | encoder-decoder | 31.25 | 53.48 | 52.54 | 582.4 | 582.4 | 60000 | 104.1 | 5.11 | 47.77 | 47.8 | 1.61 | full fine-tuning (Adafactor, bf16) |
| llama_zeroshot | decoder-only | 32.88 | 56.99 | 55.5 | 3212.7 | 0.0 | 0 | 0.0 |  | 122.75 | 122.8 | 2.63 | zero-shot prompting |
| llama_qlora | decoder-only | 40.0 | 60.99 | 60.01 | 3212.7 | 24.31 | 20000 | 67.7 | 3.96 | 235.92 | 235.9 | 2.88 | QLoRA (4-bit NF4, LoRA r=16, all linear layers) |
| gemma_zeroshot | decoder-only | 34.87 | 58.32 | 56.92 | 2614.3 | 0.0 | 0 | 0.0 |  | 197.93 | 197.9 | 2.68 | zero-shot prompting |
| gemma_qlora | decoder-only | 40.64 | 61.76 | 60.79 | 2614.3 | 20.77 | 20000 | 158.6 | 3.96 | 255.89 | 255.9 | 2.67 | QLoRA (4-bit NF4, LoRA r=16, all linear layers) |

BLEU signature: nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
chrF signature: nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0
