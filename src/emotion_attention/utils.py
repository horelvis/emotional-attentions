from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F


def shift_targets(X: torch.Tensor) -> torch.Tensor:
    """Left-rotate sequence for next-token prediction."""
    return torch.roll(X, shifts=-1, dims=1)


def cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (1 - (a * b).sum(-1)).mean()


def sample_topk(logits_row: torch.Tensor, k: int = 5, temp: float = 0.9) -> int:
    probs = F.softmax(logits_row / temp, dim=-1)
    topk = torch.topk(probs, k)
    idx = torch.multinomial(topk.values, 1)
    return topk.indices[idx].item()


def make_masks(X: torch.Tensor, sep_id: int, pad_id: int) -> tuple[torch.Tensor, ...]:
    B, T = X.shape
    pad = (X == pad_id)
    in_mask = torch.zeros_like(X, dtype=torch.bool)
    out_mask = torch.zeros_like(X, dtype=torch.bool)
    for b in range(B):
        ids = X[b].tolist()
        sep_pos = ids.index(sep_id) if sep_id in ids else max(1, len(ids) - 1)
        in_mask[b, :sep_pos] = True
        out_mask[b, sep_pos + 1 :] = X[b, sep_pos + 1 :] != pad_id
    return pad, in_mask, out_mask


def decode(ids: List[int], itos: dict[int, str]) -> str:
    special = {"<pad>", "<bos>"}
    toks = [itos[int(i)] for i in ids if itos[int(i)] not in special]
    return " ".join(toks)
