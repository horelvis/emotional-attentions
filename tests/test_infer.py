import io
import unittest

import torch

from emotion_attention.data import SpecialTokenIds
from emotion_attention.infer import DEFAULT_PROMPT, generate, resolve_user_input


class FakeTokenizer:
    def __init__(self):
        self.token_to_id = {
            '<bos>': 0,
            '<pad>': 1,
            '<eos>': 2,
            '<sep>': 3,
        }
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

    def encode(self, text, add_special_tokens=False, truncation=True, max_length=None):
        tokens = text.strip().split()
        ids = []
        for token in tokens:
            tok = token.lower()
            idx = self.token_id(tok)
            ids.append(idx)
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        specials = {'<bos>', '<pad>', '<eos>', '<sep>'}
        words = []
        for idx in ids:
            token = self.id_to_token.get(idx, '')
            if skip_special_tokens and token in specials:
                continue
            words.append(token)
        return ' '.join(words).strip()

    def __len__(self):
        return len(self.token_to_id)

    def token_id(self, token: str) -> int:
        t = token.lower()
        if t not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[t] = idx
            self.id_to_token[idx] = t
        return self.token_to_id[t]


class FakeModel:
    def __init__(self, special: SpecialTokenIds, generation_ids):
        self.special = special
        self.generation_ids = generation_ids
        self.step = 0
        self.hidden = 4  # dimensionality of g vectors
        self.expect_g_out = False
        self.last_token = special.eos

    def __call__(self, X, pad, in_mask, out_mask, return_attn=False):
        B, T = X.shape
        vocab_size = max(self.generation_ids + [self.special.eos]) + 5
        logits = torch.zeros(B, T, vocab_size)
        g = torch.zeros(B, self.hidden)
        if return_attn:
            return logits, g, g, []
        if not self.expect_g_out:
            idx = self.generation_ids[self.step] if self.step < len(self.generation_ids) else self.special.eos
            self.step += 1
            self.last_token = idx
            self.expect_g_out = True
        else:
            idx = self.last_token
            self.expect_g_out = False
        logits[0, -1, idx] = 10.0
        return logits, g, g, []


class InferenceLongDialogTest(unittest.TestCase):
    def test_generate_from_long_dialog(self):
        tokenizer = FakeTokenizer()
        special = SpecialTokenIds(
            pad=tokenizer.token_id('<pad>'),
            bos=tokenizer.token_id('<bos>'),
            eos=tokenizer.token_id('<eos>'),
            sep=tokenizer.token_id('<sep>'),
        )
        respuesta_id = tokenizer.token_id('respuesta')
        amable_id = tokenizer.token_id('amable')
        generation = [respuesta_id, amable_id, special.eos]
        model = FakeModel(special, generation)

        prompt = (
            "USER: hola necesito ayuda con un problema familiar muy complicado BOT: claro dime más "
            "USER: llevo semanas intentando hablar pero no me entienden y me siento aislado BOT: entiendo debe ser duro "
            "USER: cada conversación termina en discusiones y estoy agotado BOT: estoy aquí para escucharte"
        )
        ids, decoded = generate(
            model,
            tokenizer,
            special,
            prompt_text=prompt,
            device=torch.device('cpu'),
            max_new=8,
            k=1,
            temp=1.0,
            ema_alpha=0.5,
            max_length=128,
        )

        self.assertIn('respuesta amable', decoded)
        self.assertTrue(ids[-1] == special.eos)
        self.assertGreater(len(ids), len(tokenizer.encode(prompt, max_length=128)))


class ResolveUserInputTest(unittest.TestCase):
    def test_cli_input_has_priority(self):
        captured = resolve_user_input("  Hola  ", io.StringIO("otro"))
        self.assertEqual(captured, "Hola")

    def test_reads_from_stdin_stream(self):
        captured = resolve_user_input(None, io.StringIO("   Texto desde pipe   "))
        self.assertEqual(captured, "Texto desde pipe")

    def test_defaults_when_empty(self):
        captured = resolve_user_input(None, io.StringIO(""))
        self.assertEqual(captured, DEFAULT_PROMPT)


if __name__ == '__main__':
    unittest.main()
