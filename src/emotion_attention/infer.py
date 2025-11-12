from __future__ import annotations

import argparse
import json
import sys
import torch
from transformers import AutoTokenizer

from .data import ensure_special_tokens, SpecialTokenIds
from .model import EmoDecoder
from .utils import make_masks, sample_topk


def parse_args():
    p = argparse.ArgumentParser(description="Generate emotional responses with adapters")
    p.add_argument('--checkpoint', type=str, default='artifacts/emo_decoder.pt')
    p.add_argument('--tokenizer-dir', type=str, default='artifacts/tokenizer')
    p.add_argument('--input', type=str, help='Entrada del usuario')
    p.add_argument('--max-new', type=int, default=40)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--temp', type=float, default=0.8)
    p.add_argument('--ema-alpha', type=float, default=0.1)
    p.add_argument('--max-length', type=int, default=256)
    return p.parse_args()


DEFAULT_PROMPT = (
    "USER: hola necesito ayuda con un problema familiar muy complicado BOT: claro dime más "
    "USER: llevo semanas intentando hablar pero no me entienden y me siento aislado BOT: entiendo debe ser duro "
    "USER: cada conversación termina en discusiones y estoy agotado"
)


def load_model(path: str, tokenizer_dir: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    tokenizer, special_ids = ensure_special_tokens(tokenizer)
    config = ckpt['config']
    model = EmoDecoder(**config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    saved_special = ckpt.get('special_ids', None)
    if saved_special:
        special_ids = SpecialTokenIds(**saved_special)
    return model, tokenizer, special_ids


def encode_prompt(tokenizer, special_ids, text: str, max_length: int):
    ids = tokenizer.encode(text.strip(), add_special_tokens=False, truncation=True, max_length=max_length - 2)
    seq = [special_ids.bos] + ids + [special_ids.sep]
    return seq


def infer_g_in(model, X, special_ids, device):
    pad, in_mask, out_mask = make_masks(X, special_ids.sep, special_ids.pad)
    _, g_in, _, _ = model(X.to(device), pad.to(device), in_mask.to(device), out_mask.to(device), return_attn=True)
    return g_in.unsqueeze(1)


def generate(model, tokenizer, special_ids, prompt_text, device, max_new, k, temp, ema_alpha, max_length):
    ids = encode_prompt(tokenizer, special_ids, prompt_text, max_length)
    X = torch.tensor([ids], device=device)
    g_t = infer_g_in(model, X, special_ids, device)
    for _ in range(max_new):
        pad, in_mask, out_mask = make_masks(X, special_ids.sep, special_ids.pad)
        logits, _, _, _ = model(X, pad.to(device), in_mask.to(device), out_mask.to(device), return_attn=False)
        next_id = sample_topk(logits[0, -1], k=k, temp=temp)
        X = torch.cat([X, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == special_ids.eos or X.size(1) >= max_length:
            break
        pad, in_mask, out_mask = make_masks(X, special_ids.sep, special_ids.pad)
        _, _, g_out_step, _ = model(X, pad.to(device), in_mask.to(device), out_mask.to(device), return_attn=False)
        g_t = (1 - ema_alpha) * g_t + ema_alpha * g_out_step.unsqueeze(1)
    decoded = tokenizer.decode(X[0].tolist(), skip_special_tokens=True)
    return X[0].tolist(), decoded


def emo_alignment_score(model, special_ids, tokenizer, prompt_text, gen_ids, device):
    prompt_ids = encode_prompt(tokenizer, special_ids, prompt_text, len(gen_ids))
    X_prompt = torch.tensor([prompt_ids], device=device)
    g_in = infer_g_in(model, X_prompt, special_ids, device).squeeze(0)
    toks = gen_ids
    if special_ids.sep in toks:
        start = toks.index(special_ids.sep) + 1
    else:
        start = max(1, len(toks) // 2)
    if special_ids.eos in toks:
        end = toks.index(special_ids.eos) + 1
    else:
        end = len(toks)
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


def resolve_user_input(cli_input: str | None, stdin_stream=None) -> str:
    if cli_input and cli_input.strip():
        return cli_input.strip()
    if stdin_stream is None:
        stdin_stream = sys.stdin
    is_tty = bool(getattr(stdin_stream, 'isatty', lambda: False)())
    if not is_tty:
        data = stdin_stream.read().strip()
        if data:
            return data
    if stdin_stream is sys.stdin:
        print("Introduce el historial o mensaje del usuario:", file=sys.stderr)
        try:
            line = input().strip()
        except EOFError:
            line = ""
        if line:
            return line
    return DEFAULT_PROMPT


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, special_ids = load_model(args.checkpoint, args.tokenizer_dir, device)
    user_text = resolve_user_input(args.input)
    gen_ids, decoded = generate(
        model,
        tokenizer,
        special_ids,
        user_text,
        device,
        max_new=args.max_new,
        k=args.k,
        temp=args.temp,
        ema_alpha=args.ema_alpha,
        max_length=args.max_length,
    )
    score = emo_alignment_score(model, special_ids, tokenizer, user_text, gen_ids, device)
    print(json.dumps({'input': user_text, 'output': decoded, 'alignment': score}, ensure_ascii=False))


if __name__ == '__main__':
    main()
