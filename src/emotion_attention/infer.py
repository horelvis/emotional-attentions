from __future__ import annotations

import argparse
import json
from typing import List

import torch

from .data import build_batch, encode
from .model import EmoDecoder
from .utils import decode, make_masks, sample_topk


def parse_args():
    p = argparse.ArgumentParser(description="Generate responses and alignment scores")
    p.add_argument('--checkpoint', type=str, default='artifacts/emo_decoder.pt')
    p.add_argument('--input', type=str, required=True, help='Entrada separada por espacios')
    p.add_argument('--max-new', type=int, default=12)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--temp', type=float, default=0.9)
    p.add_argument('--ema-alpha', type=float, default=0.15)
    return p.parse_args()


def load_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    config = ckpt['config']
    model = EmoDecoder(**config).to(device)
    model.load_state_dict(ckpt['model'])
    stoi = ckpt['stoi']
    itos = {i: tok for i, tok in enumerate(ckpt['itos'])}
    model.eval()
    return model, stoi, itos


def infer_g_in(model, words: List[str], stoi, device):
    ids = [stoi['<bos>']] + encode(words, stoi) + [stoi['<sep>']]
    X = torch.tensor([ids], device=device)
    pad, in_mask, out_mask = make_masks(X, stoi['<sep>'], stoi['<pad>'])
    _, g_in, _, _ = model(X, pad, in_mask, out_mask, return_attn=True)
    return g_in.unsqueeze(1)


def generate(model, words, stoi, itos, device, max_new, k, temp, ema_alpha):
    ids = [stoi['<bos>']] + encode(words, stoi) + [stoi['<sep>']]
    X = torch.tensor([ids], device=device)
    g_t = infer_g_in(model, words, stoi, device)
    for _ in range(max_new):
        pad, in_mask, out_mask = make_masks(X, stoi['<sep>'], stoi['<pad>'])
        logits, _, _, _ = model(X, pad, in_mask, out_mask, return_attn=False)
        next_id = sample_topk(logits[0, -1], k=k, temp=temp)
        X = torch.cat([X, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == stoi['<eos>']:
            break
        pad, in_mask, out_mask = make_masks(X, stoi['<sep>'], stoi['<pad>'])
        _, _, g_out_step, _ = model(X, pad, in_mask, out_mask, return_attn=False)
        g_t = (1 - ema_alpha) * g_t + ema_alpha * g_out_step.unsqueeze(1)
    return X[0].tolist(), decode(X[0].tolist(), itos)


def emo_alignment_score(model, words, gen_ids, stoi, device):
    g_in = infer_g_in(model, words, stoi, device).squeeze()
    toks = gen_ids
    sep_id = stoi['<sep>']
    eos_id = stoi['<eos>']
    start = toks.index(sep_id) + 1 if sep_id in toks else max(1, len(toks) // 2)
    end = toks.index(eos_id) + 1 if eos_id in toks else len(toks)
    X = torch.tensor([toks[:end]], device=device)
    pad = torch.zeros_like(X).bool()
    in_mask = torch.zeros_like(X).bool()
    out_mask = torch.zeros_like(X).bool()
    out_mask[:, start:end] = True
    _, _, g_out, _ = model(X, pad, in_mask, out_mask, return_attn=False)
    g_out = g_out.squeeze(0)
    gi = torch.nn.functional.normalize(g_in, dim=-1)
    go = torch.nn.functional.normalize(g_out, dim=-1)
    return float((gi * go).sum().clamp(-1, 1))


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stoi, itos = load_model(args.checkpoint, device)
    words = args.input.strip().split()
    gen_ids, decoded = generate(
        model,
        words,
        stoi,
        itos,
        device,
        max_new=args.max_new,
        k=args.k,
        temp=args.temp,
        ema_alpha=args.ema_alpha,
    )
    score = emo_alignment_score(model, words, gen_ids, stoi, device)
    print(json.dumps({'input': words, 'output': decoded, 'alignment': score}, ensure_ascii=False))


if __name__ == '__main__':
    main()
