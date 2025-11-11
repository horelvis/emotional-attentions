from __future__ import annotations

import argparse
import math
import os
from typing import List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .data import build_batch, get_vocab_and_pairs
from .model import EmoDecoder
from .utils import cosine_loss, shift_targets


def parse_args():
    p = argparse.ArgumentParser(description="Train EmoDecoder with adapters")
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lambda-prop', type=float, default=0.8)
    p.add_argument('--lambda-distil', type=float, default=0.3)
    p.add_argument('--lambda-sparse', type=float, default=1e-4)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--d-emo', type=int, default=64)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--use-distil', action='store_true')
    p.add_argument('--hf-name', type=str, default='pysentimiento/robertuito-sentiment-analysis')
    p.add_argument('--freeze-base', action='store_true', help='Freeze backbone, train adapters/heads only')
    p.add_argument('--save-path', type=str, default='artifacts/emo_decoder.pt')
    return p.parse_args()


def load_teacher(hf_name: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
    model = AutoModel.from_pretrained(hf_name).to(device)
    for p in model.parameters():
        p.requires_grad = False
    return tok, model


def teacher_vec(words: Sequence[str], tok, model, proj, device):
    text = " ".join(words)
    enc = tok(text, return_tensors='pt', truncation=True, max_length=128).to(device)
    hs = model(**enc).last_hidden_state
    h_pool = hs.mean(dim=1)
    return proj(h_pool)[0]


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab, stoi, itos, train_pairs, valid_pairs = get_vocab_and_pairs(seed=args.seed)

    config = {
        'vocab_size': len(vocab),
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'd_emo': args.d_emo,
        'n_layers': args.n_layers,
    }
    model = EmoDecoder(**config).to(device)
    if args.freeze_base:
        model.freeze_base()

    train_batch = build_batch(train_pairs, stoi, device)
    valid_batch = build_batch(valid_pairs, stoi, device)
    Xtr, Mtr, INtr, OUTtr = train_batch
    Xva, Mva, INva, OUTva = valid_batch
    targets_tr = shift_targets(Xtr)
    targets_va = shift_targets(Xva)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    if args.use_distil:
        tok_teacher, teacher = load_teacher(args.hf_name, device)
        teacher_proj = torch.nn.Sequential(
            torch.nn.Linear(teacher.config.hidden_size, teacher.config.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(teacher.config.hidden_size, args.d_emo),
        ).to(device)
        g_in_hat = torch.stack([teacher_vec(inp, tok_teacher, teacher, teacher_proj, device) for inp, _ in train_pairs])
        g_out_hat = torch.stack([teacher_vec(out, tok_teacher, teacher, teacher_proj, device) for _, out in train_pairs])
    else:
        tok_teacher = teacher = teacher_proj = None
        g_in_hat = g_out_hat = None

    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits, g_in, g_out, attn = model(Xtr, Mtr, INtr, OUTtr, return_attn=True)
        L_lm = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets_tr.view(-1),
            ignore_index=stoi['<pad>'],
        )
        L_prop = cosine_loss(g_in, g_out)
        if args.use_distil:
            L_distil = F.mse_loss(g_in, g_in_hat.to(device)) + F.mse_loss(g_out, g_out_hat.to(device))
        else:
            L_distil = torch.tensor(0.0, device=device)
        if attn:
            sparsity_terms = [a['G'].abs().mean() for a in attn]
            L_sparse = torch.stack(sparsity_terms).mean()
        else:
            L_sparse = torch.tensor(0.0, device=device)
        loss = L_lm + args.lambda_prop * L_prop + args.lambda_distil * L_distil + args.lambda_sparse * L_sparse
        loss.backward()
        opt.step()

        if ep % 5 == 0 or ep == args.epochs:
            model.eval()
            with torch.no_grad():
                logits_v, gi_v, go_v, _ = model(Xva, Mva, INva, OUTva, return_attn=False)
                L_lm_v = F.cross_entropy(
                    logits_v.view(-1, logits_v.size(-1)),
                    targets_va.view(-1),
                    ignore_index=stoi['<pad>'],
                ).item()
                L_prop_v = cosine_loss(gi_v, go_v).item()
                ppl = math.exp(min(10, L_lm_v))
            print(
                f"Ep{ep:02d} | loss={loss.item():.3f} | LM={L_lm.item():.3f} | "
                f"PROP={L_prop.item():.3f} | DISTIL={L_distil.item():.3f} | SP={L_sparse.item():.4f} | PPL={ppl:.2f}"
            )

    save_path = args.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'model': model.state_dict(), 'stoi': stoi, 'itos': vocab, 'config': config}, save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == '__main__':
    main()
