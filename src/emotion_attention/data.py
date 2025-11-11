from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import torch


VOCAB = [
    '<pad>', '<bos>', '<eos>', '<sep>',
    'te','quiero','mamá','hola','cariño','yo','tambien','estoy','triste','lo','siento','aqui',
    'no','odio','mucho','calma','hablemos','por','favor','feliz','me','alegra','porque',
    'gracias','contigo','gratitud','abrazarte','confio','en','ti','orgullosa','de','inspiras',
    'acompanas','tengo','miedo','ansiosa','preocupa','perderte','esperaba','que','sorpresa',
    'creo','solo','perdi','algo','importante','culpable','gusta','alejarme','enojada','soporto',
    'dudas','nerviosa','manana','se','pasara','respira','dejare','sola','entiendo','sorprende',
    'escucharlo','procesemos','juntas','lamento','decirlo','respeto','si','necesitas','tu',
    'rabia','escucho','enojo','pregunta','planifiquemos','tranquila','incertidumbre',
    'acompanio','siempre','confiar'
]

PAIRS = [
    (['te','quiero','mamá'],               ['yo','tambien','te','quiero']),
    (['hola','mamá'],                      ['hola','cariño']),
    (['estoy','triste'],                   ['lo','siento','aqui','estoy']),
    (['no','te','quiero'],                 ['lo','siento','hablemos']),
    (['te','odio','mucho'],                ['calma','hablemos','por','favor']),
    (['feliz'],                            ['me','alegra','mucho']),
    (['estoy','feliz'],                    ['me','alegra','contigo']),
    (['siento','gratitud'],                ['gracias','aqui','contigo']),
    (['quiero','abrazarte'],               ['yo','tambien','te','quiero']),
    (['confio','en','ti'],                 ['gracias','por','confiar']),
    (['estoy','orgullosa','de','ti'],      ['me','inspiras','mucho']),
    (['me','acompanas'],                   ['siempre','estoy','contigo']),
    (['tengo','miedo'],                    ['respira','aqui','estoy']),
    (['estoy','ansiosa'],                  ['calma','te','acompanio']),
    (['me','preocupa','perderte'],         ['no','te','dejare','sola']),
    (['no','lo','esperaba'],               ['entiendo','sorprende','mucho']),
    (['que','sorpresa'],                   ['me','alegra','escucharlo']),
    (['no','lo','creo'],                   ['respira','procesemos','juntas']),
    (['me','siento','solo'],               ['te','acompanio','siempre']),
    (['perdi','algo','importante'],        ['lamento','mucho','estoy','contigo']),
    (['me','siento','culpable'],           ['gracias','por','decirlo','calma']),
    (['no','me','gusta'],                  ['entiendo','hablemos']),
    (['quiero','alejarme'],                ['te','respeto','aqui','si','necesitas']),
    (['estoy','enojada'],                  ['entiendo','tu','rabia','calma']),
    (['te','odio'],                        ['lamento','mucho','hablemos']),
    (['no','te','soporto'],                ['escucho','tu','enojo']),
    (['tengo','dudas'],                    ['pregunta','estoy','aqui']),
    (['estoy','nerviosa','por','manana'],  ['planifiquemos','tranquila']),
    (['no','se','que','pasara'],           ['estoy','contigo','en','incertidumbre'])
]


def get_vocab_and_pairs(seed: int = 42, split_ratio: float = 0.8):
    vocab = VOCAB.copy()
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    pairs = PAIRS.copy()
    rng = random.Random(seed)
    rng.shuffle(pairs)
    split = int(split_ratio * len(pairs))
    train_pairs = pairs[:split]
    valid_pairs = pairs[split:]
    return vocab, stoi, itos, train_pairs, valid_pairs


def encode(words: Sequence[str], stoi: Dict[str, int]) -> List[int]:
    return [stoi[w] for w in words]


def build_batch(
    pairs: Sequence[Tuple[Sequence[str], Sequence[str]]],
    stoi: Dict[str, int],
    device: torch.device,
):
    Xs, pad_masks, in_masks, out_masks = [], [], [], []
    for inp, out in pairs:
        ids = [stoi['<bos>']] + encode(inp, stoi) + [stoi['<sep>']] + encode(out, stoi) + [stoi['<eos>']]
        Xs.append(ids)
    maxlen = max(len(x) for x in Xs)
    for ids in Xs:
        pad = [stoi['<pad>']] * (maxlen - len(ids))
        ids.extend(pad)
        pad_mask = [0] * (maxlen - len(pad)) + [1] * len(pad)
        sep_pos = ids.index(stoi['<sep>'])
        in_mask = [0] * maxlen
        out_mask = [0] * maxlen
        for t in range(sep_pos):
            in_mask[t] = 1
        for t in range(sep_pos + 1, len(ids)):
            if ids[t] != stoi['<pad>']:
                out_mask[t] = 1
        pad_masks.append(pad_mask)
        in_masks.append(in_mask)
        out_masks.append(out_mask)
    X = torch.tensor(Xs, device=device)
    PM = torch.tensor(pad_masks, device=device).bool()
    INM = torch.tensor(in_masks, device=device).bool()
    OUTM = torch.tensor(out_masks, device=device).bool()
    return X, PM, INM, OUTM
