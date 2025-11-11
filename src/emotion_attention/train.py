from __future__ import annotations

import argparse
import math
import os
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .data import create_dataloader, ensure_special_tokens
from .model import EmoDecoder
from .utils import cosine_loss, shift_targets


def parse_args():
    p = argparse.ArgumentParser(description="Train EmoDecoder adapters on conversational emotions")
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--lambda-prop', type=float, default=0.8)
    p.add_argument('--lambda-distil', type=float, default=0.3)
    p.add_argument('--lambda-sparse', type=float, default=1e-4)
    p.add_argument('--d-model', type=int, default=512)
    p.add_argument('--d-emo', type=int, default=64)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--n-layers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--use-distil', action='store_true')
    p.add_argument('--hf-name', type=str, default='pysentimiento/robertuito-sentiment-analysis')
    p.add_argument('--freeze-base', action='store_true')
    p.add_argument('--save-path', type=str, default='artifacts/emo_decoder.pt')
    p.add_argument('--tokenizer-name', type=str, default='distilroberta-base')
    p.add_argument('--tokenizer-dir', type=str, default='artifacts/tokenizer')
    p.add_argument('--dataset-name', type=str, default='daily_dialog')
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--history-turns', type=int, default=3)
    p.add_argument('--include-neutral', action='store_true')
    return p.parse_args()


def load_teacher(hf_name: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
    model = AutoModel.from_pretrained(hf_name).to(device)
    for p in model.parameters():
        p.requires_grad = False
    return tok, model


def teacher_vec_batch(texts: Sequence[str], tok, model, proj, device, max_length: int = 128):
    enc = tok(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        hs = model(**enc).last_hidden_state
    h_pool = hs.mean(dim=1)
    return proj(h_pool)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    tokenizer, special_ids = ensure_special_tokens(tokenizer)
    os.makedirs(args.tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(args.tokenizer_dir)

    train_loader = create_dataloader(
        tokenizer=tokenizer,
        special_ids=special_ids,
        split='train',
        dataset_name=args.dataset_name,
        history_turns=args.history_turns,
        max_length=args.max_length,
        include_neutral=args.include_neutral,
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = create_dataloader(
        tokenizer=tokenizer,
        special_ids=special_ids,
        split='validation',
        dataset_name=args.dataset_name,
        history_turns=args.history_turns,
        max_length=args.max_length,
        include_neutral=args.include_neutral,
        batch_size=args.batch_size,
        shuffle=False,
    )

    config = {
        'vocab_size': len(tokenizer),
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'd_emo': args.d_emo,
        'n_layers': args.n_layers,
    }
    model = EmoDecoder(**config).to(device)
    if args.freeze_base:
        model.freeze_base()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    if args.use_distil:
        tok_teacher, teacher = load_teacher(args.hf_name, device)
        teacher_proj = torch.nn.Sequential(
            torch.nn.Linear(teacher.config.hidden_size, teacher.config.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(teacher.config.hidden_size, args.d_emo),
        ).to(device)
    else:
        tok_teacher = teacher = teacher_proj = None

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = total_lm = total_prop = total_distil = total_sparse = 0.0
        for batch in train_loader:
            X, pad_mask, in_mask, out_mask, user_texts, target_texts = batch
            X = X.to(device)
            pad_mask = pad_mask.to(device)
            in_mask = in_mask.to(device)
            out_mask = out_mask.to(device)
            targets = shift_targets(X)

            logits, g_in, g_out, attn = model(X, pad_mask, in_mask, out_mask, return_attn=True)
            L_lm = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=special_ids.pad,
            )
            L_prop = cosine_loss(g_in, g_out)
            if args.use_distil and tok_teacher is not None:
                g_in_hat = teacher_vec_batch(user_texts, tok_teacher, teacher, teacher_proj, device)
                g_out_hat = teacher_vec_batch(target_texts, tok_teacher, teacher, teacher_proj, device)
                L_distil = F.mse_loss(g_in, g_in_hat) + F.mse_loss(g_out, g_out_hat)
            else:
                L_distil = torch.tensor(0.0, device=device)
            if attn:
                sparsity_terms = [a['G'].abs().mean() for a in attn]
                L_sparse = torch.stack(sparsity_terms).mean()
            else:
                L_sparse = torch.tensor(0.0, device=device)
            loss = L_lm + args.lambda_prop * L_prop + args.lambda_distil * L_distil + args.lambda_sparse * L_sparse

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_lm += L_lm.item()
            total_prop += L_prop.item()
            total_distil += L_distil.item()
            total_sparse += L_sparse.item()

        steps = len(train_loader)
        print(
            f"Ep{ep:02d} | loss={total_loss/steps:.3f} | LM={total_lm/steps:.3f} | "
            f"PROP={total_prop/steps:.3f} | DISTIL={total_distil/steps:.3f} | SP={total_sparse/steps:.4f}"
        )

        model.eval()
        with torch.no_grad():
            val_lm = val_prop = 0.0
            val_steps = 0
            for batch in valid_loader:
                X, pad_mask, in_mask, out_mask, _, _ = batch
                X = X.to(device)
                pad_mask = pad_mask.to(device)
                in_mask = in_mask.to(device)
                out_mask = out_mask.to(device)
                targets = shift_targets(X)
                logits, g_in, g_out, _ = model(X, pad_mask, in_mask, out_mask, return_attn=False)
                L_lm_v = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=special_ids.pad,
                )
                L_prop_v = cosine_loss(g_in, g_out)
                val_lm += L_lm_v.item()
                val_prop += L_prop_v.item()
                val_steps += 1
            ppl = math.exp(min(10, val_lm / max(val_steps, 1)))
            print(f"          | val_LM={val_lm/max(val_steps,1):.3f} | val_PROP={val_prop/max(val_steps,1):.3f} | PPL={ppl:.2f}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {
            'model': model.state_dict(),
            'config': config,
            'tokenizer_dir': args.tokenizer_dir,
            'special_ids': vars(special_ids),
        },
        args.save_path,
    )
    print(f"Saved checkpoint to {args.save_path}")


if __name__ == '__main__':
    main()
