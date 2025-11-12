from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _shape(x, B, T, n_heads, d_head):
    return x.view(B, T, n_heads, d_head).transpose(1, 2)


class MultiHeadSelfAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.nh = n_heads
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, H, key_padding_mask=None, causal=True):
        B, T, D = H.shape
        q = _shape(self.q(H), B, T, self.nh, self.dh)
        k = _shape(self.k(H), B, T, self.nh, self.dh)
        v = _shape(self.v(H), B, T, self.nh, self.dh)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        if causal:
            idx = torch.arange(T, device=H.device)
            causal_mask = (idx[None, :] <= idx[:, None]).float()
            scores = scores + (causal_mask[None, None, :, :] - 1) * 1e9
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].bool()
            scores = scores.masked_fill(mask, float('-inf'))
        A = torch.softmax(scores, dim=-1)
        O = (A @ v).transpose(1, 2).contiguous().view(B, T, D)
        O = self.o(O)
        return O


class DualHeadEmoAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_emo: int, dropout: float = 0.1):
        super().__init__()
        self.sem = MultiHeadSelfAttn(d_model, n_heads)
        self.proj_u = nn.Linear(d_emo, d_model)
        self.emo = MultiHeadSelfAttn(d_model, n_heads)
        self.Wg_h = nn.Linear(d_model, d_model)
        self.Wg_e = nn.Linear(d_model, d_model)
        self.Wg_g = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, H, U, g, key_padding_mask=None, return_attn=False):
        O_sem = self.sem(H, key_padding_mask=key_padding_mask, causal=True)
        U_proj = self.proj_u(U)
        O_emo = self.emo(U_proj, key_padding_mask=key_padding_mask, causal=True)
        g_proj = self.Wg_g(self.proj_u(g)).expand(H.size(0), H.size(1), -1)
        # La compuerta aprende cuánto peso dar a la pista semántica (O_sem) frente a la emocional (O_emo).
        G = torch.sigmoid(self.Wg_h(O_sem) + self.Wg_e(O_emo) + g_proj)
        mix = (1 - G) * O_sem + G * O_emo
        out = self.norm(self.out(self.drop(mix)) + H)
        if return_attn:
            return out, {"G": G}
        return out, None


class EmoAdapter(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_emo: int, dropout: float = 0.1):
        super().__init__()
        self.dual = DualHeadEmoAttention(d_model, n_heads, d_emo, dropout)
        self.proj = nn.Linear(d_model, d_emo)

    def forward(self, H, g_in, key_padding_mask=None):
        U = self.proj(H)
        return self.dual(H, U, g_in, key_padding_mask=key_padding_mask, return_attn=True)


class BaseBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.attn = MultiHeadSelfAttn(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * d_model), d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, H, pad_mask=None):
        attn_out = self.attn(H, key_padding_mask=pad_mask, causal=True)
        H = self.norm1(H + attn_out)
        H = self.norm2(H + self.mlp(H))
        return H


class EmoDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_emo: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(512, d_model)
        self.blocks = nn.ModuleList([BaseBlock(d_model, n_heads) for _ in range(n_layers)])
        self.adapters = nn.ModuleList([EmoAdapter(d_model, n_heads, d_emo, dropout) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.emo_in_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_emo))
        self.emo_out_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_emo))

    def freeze_base(self):
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False
        for emb in (self.tok, self.pos):
            for p in emb.parameters():
                p.requires_grad = False

    def forward(self, X, pad_mask, in_mask, out_mask, return_attn=False):
        B, T = X.shape
        pos = torch.arange(T, device=X.device)[None, :].expand(B, T)
        H = self.tok(X) + self.pos(pos)
        g_in = self._pool(H, in_mask, self.emo_in_head)  # (B, d_emo)
        attn_info = []
        for block, adapter in zip(self.blocks, self.adapters):
            H = block(H, pad_mask)
            H, attn = adapter(H, g_in.unsqueeze(1), key_padding_mask=pad_mask)
            if return_attn:
                attn_info.append(attn)
        logits = self.lm_head(H)
        g_out = self._pool(H, out_mask, self.emo_out_head)
        if return_attn:
            return logits, g_in, g_out, attn_info
        return logits, g_in, g_out, None

    def _pool(self, H, mask, head):
        H_masked = H.masked_fill(~mask[..., None], 0.0)
        denom = mask.sum(1).clamp(min=1).view(H.size(0), 1).float()
        # Promediamos únicamente las posiciones activas del enmascarado para obtener una representación compacta.
        pooled = H_masked.sum(1) / denom
        return head(pooled)
