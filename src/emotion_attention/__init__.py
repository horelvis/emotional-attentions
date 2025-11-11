"""Emotional attention package with Plutchik-aligned adapters."""

from .data import get_vocab_and_pairs, build_batch
from .model import EmoDecoder, DualHeadEmoAttention, EmoAdapter
from .utils import shift_targets, cosine_loss

__all__ = [
    "get_vocab_and_pairs",
    "build_batch",
    "EmoDecoder",
    "DualHeadEmoAttention",
    "EmoAdapter",
    "shift_targets",
    "cosine_loss",
]
