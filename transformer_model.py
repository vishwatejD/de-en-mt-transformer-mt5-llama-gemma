"""Encoder-decoder Transformer (Vaswani et al., 2017) implemented from PyTorch building blocks."""
import math

import torch
import torch.nn as nn

PAD, UNK, BOS, EOS = 0, 1, 2, 3


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Seq2SeqTransformer(nn.Module):
    """Pre-LN Transformer with one shared (joint-vocabulary) embedding tied to the output layer."""

    def __init__(self, vocab_size, d_model=512, nhead=4, num_layers=6, dim_ff=1024,
                 dropout=0.3, max_len=512):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                           num_layers=num_layers, dim_ff=dim_ff, dropout=dropout, max_len=max_len)
        self.scale = math.sqrt(d_model)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        nn.init.normal_(self.embed.weight, 0.0, d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[PAD].zero_()
        self.pos = PositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True, norm_first=True)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.embed.weight

    def _embed(self, ids):
        return self.drop(self.pos(self.embed(ids) * self.scale))

    def encode(self, src):
        src_pad = src.eq(PAD)
        memory = self.transformer.encoder(self._embed(src), src_key_padding_mask=src_pad)
        return memory, src_pad

    def decode(self, tgt_in, memory, src_pad, tgt_pad=None):
        T = tgt_in.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_in.device), diagonal=1)
        y = self.transformer.decoder(self._embed(tgt_in), memory, tgt_mask=causal,
                                     tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=src_pad)
        return self.out(y)

    def forward(self, src, tgt_in):
        memory, src_pad = self.encode(src)
        return self.decode(tgt_in, memory, src_pad, tgt_in.eq(PAD))

    @torch.no_grad()
    def greedy_decode(self, src, max_len=256):
        memory, src_pad = self.encode(src)
        B = src.size(0)
        limit = min(max_len, int(src.size(1) * 1.5) + 10)
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(limit):
            nxt = self.decode(ys, memory, src_pad)[:, -1].argmax(-1)
            nxt = nxt.masked_fill(done, PAD)
            ys = torch.cat([ys, nxt.unsqueeze(1)], 1)
            done |= nxt.eq(EOS)
            if done.all():
                break
        out = []
        for row in ys[:, 1:].tolist():
            out.append(row[: row.index(EOS)] if EOS in row else [t for t in row if t != PAD])
        return out


def load_transformer(path, device="cuda"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Seq2SeqTransformer(**ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def translate_batch(model, sp, sentences, device="cuda", batch_size=64, max_src_len=256):
    """Length-sorted batched greedy translation; returns outputs in input order."""
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    results = [None] * len(sentences)
    for k in range(0, len(order), batch_size):
        idx = order[k: k + batch_size]
        ids = [sp.encode(sentences[i])[: max_src_len - 1] + [EOS] for i in idx]
        L = max(map(len, ids))
        src = torch.tensor([x + [PAD] * (L - len(x)) for x in ids], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hyps = model.greedy_decode(src)
        for i, h in zip(idx, hyps):
            results[i] = sp.decode(h)
    return results
