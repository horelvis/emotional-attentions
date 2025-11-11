"""Emotional attention package with Plutchik-aligned adapters."""

from .data import ensure_special_tokens, create_dataloader, SpecialTokenIds
from .model import EmoDecoder, DualHeadEmoAttention, EmoAdapter
from .utils import shift_targets, cosine_loss

__all__ = [
    "ensure_special_tokens",
    "create_dataloader",
    "SpecialTokenIds",
    "EmoDecoder",
    "DualHeadEmoAttention",
    "EmoAdapter",
    "shift_targets",
    "cosine_loss",
]
