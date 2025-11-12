from __future__ import annotations

import os
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

EMOTION_LABELS = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "joy",
    5: "sadness",
    6: "surprise",
    7: "other",
}

EMOTION_ALIAS = {
    # Neutral / balanced
    'neutral': 0,
    'content': 0,
    'prepared': 0,
    'confident': 0,
    'trusting': 0,
    'calm': 0,
    # Anger
    'angry': 1,
    'annoyed': 1,
    'furious': 1,
    'jealous': 1,
    # Disgust
    'disgusted': 2,
    # Fear
    'afraid': 3,
    'terrified': 3,
    'anxious': 3,
    'apprehensive': 3,
    'nervous': 3,
    'worried': 3,
    # Joy / positive affect
    'joyful': 4,
    'happy': 4,
    'excited': 4,
    'anticipating': 4,
    'hopeful': 4,
    'grateful': 4,
    'impressed': 4,
    'proud': 4,
    'caring': 4,
    'faithful': 4,
    # Sadness related
    'sad': 5,
    'devastated': 5,
    'disappointed': 5,
    'lonely': 5,
    'guilty': 5,
    'ashamed': 5,
    'embarrassed': 5,
    'sentimental': 5,
    'nostalgic': 5,
    # Surprise
    'surprised': 6,
}


def extract_emotion_field(row: Dict) -> str:
    value = row.get('emotion')
    if value is None:
        value = row.get('context')
    if isinstance(value, (list, tuple)):
        return value[0] if value else ''
    return value or ''


def map_emotion_label(label: str) -> Tuple[int, str]:
    emo_id = EMOTION_ALIAS.get(label.lower(), 7)
    return emo_id, EMOTION_LABELS.get(emo_id, 'other')


@dataclass
class SpecialTokenIds:
    pad: int
    bos: int
    eos: int
    sep: int


def ensure_special_tokens(tokenizer: PreTrainedTokenizerBase) -> Tuple[PreTrainedTokenizerBase, SpecialTokenIds]:
    added = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        added = True
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '<bos>'})
        added = True
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<eos>'})
        added = True
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '<sep>'})
        added = True
    if '<emo_sep>' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<emo_sep>']})
        added = True
    sep_token = '<emo_sep>'
    sep_id = tokenizer.convert_tokens_to_ids(sep_token)
    special = SpecialTokenIds(
        pad=tokenizer.pad_token_id,
        bos=tokenizer.bos_token_id,
        eos=tokenizer.eos_token_id,
        sep=sep_id,
    )
    return tokenizer, special


class EmotionConversationDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        special_ids: SpecialTokenIds,
        split: str,
        dataset_name: str = 'empathetic_dialogues',
        history_turns: int = 3,
        max_length: int = 256,
        include_neutral: bool = True,
    ):
        self.tokenizer = tokenizer
        self.special = special_ids
        self.max_length = max_length
        self.history_turns = history_turns
        token = os.environ.get('HF_TOKEN')
        load_kwargs = {'split': split}
        if dataset_name in {'daily_dialog', 'empathetic_dialogues'}:
            load_kwargs['trust_remote_code'] = True
        if token:
            load_kwargs['token'] = token
        try:
            raw = load_dataset(dataset_name, **load_kwargs)
        except TypeError:
            # Compatibilidad con versiones antiguas de datasets que no aceptan 'token' ni 'trust_remote_code'
            legacy_kwargs = {'split': split}
            if token:
                legacy_kwargs['use_auth_token'] = token
            raw = load_dataset(dataset_name, **legacy_kwargs)
        self.samples: List[Dict] = []
        if dataset_name == 'daily_dialog':
            for dialog, emotions in zip(raw['dialog'], raw['emotion']):
                self._ingest_daily_dialog(dialog, emotions, include_neutral)
        elif dataset_name == 'empathetic_dialogues':
            conversations: Dict[str, List[Dict]] = defaultdict(list)
            for row in raw:
                conversations[row['conv_id']].append(row)
            for convo in conversations.values():
                convo.sort(key=lambda x: x['utterance_idx'])
                self._ingest_empathetic_dialog(convo, include_neutral)
        else:
            raise ValueError(f'Dataset {dataset_name} no soportado')

    def _format_turn(self, idx: int, text: str) -> str:
        speaker = 'USER' if idx % 2 == 0 else 'BOT'
        return f"{speaker}: {text.strip()}"

    def _format_turn_with_speaker(self, speaker_idx: int, text: str) -> str:
        speaker = 'USER' if speaker_idx == 0 else 'BOT'
        return f"{speaker}: {text.strip()}"

    def _ingest_daily_dialog(self, dialog: Sequence[str], emotions: Sequence[int], include_neutral: bool):
        for turn_idx in range(1, len(dialog)):
            if turn_idx % 2 == 0:
                continue  # entrenamos solo respuestas del BOT (turnos impares en 0-index)
            user_idx = turn_idx - 1
            emotion_id = emotions[user_idx] if user_idx < len(emotions) else 0
            if not include_neutral and emotion_id == 0:
                continue
            emotion_label = EMOTION_LABELS.get(emotion_id, 'neutral')
            start_idx = max(0, turn_idx - self.history_turns * 2)
            context_turns = [self._format_turn(i, dialog[i]) for i in range(start_idx, turn_idx)]
            if not context_turns:
                continue
            history_text = ' '.join(context_turns)
            target_text = dialog[turn_idx].strip()
            sample = self._encode_pair(history_text, target_text)
            if sample is None:
                continue
            sample['emotion_id'] = emotion_id
            sample['emotion_label'] = emotion_label
            sample['user_text'] = dialog[user_idx].strip()
            sample['target_text'] = target_text
            self.samples.append(sample)

    def _ingest_empathetic_dialog(self, convo: Iterable[Dict], include_neutral: bool):
        dialog = [row['utterance'] for row in convo]
        speakers = [row['speaker_idx'] for row in convo]
        emotions = [extract_emotion_field(row) for row in convo]
        for turn_idx in range(1, len(dialog)):
            if speakers[turn_idx] != 1:
                continue  # consideramos respuestas del oyente como "BOT"
            user_idx = turn_idx - 1
            while user_idx >= 0 and speakers[user_idx] == 1:
                user_idx -= 1
            if user_idx < 0:
                continue
            emotion_id, emotion_label = map_emotion_label(emotions[user_idx])
            if not include_neutral and emotion_id == 0:
                continue
            start_idx = max(0, turn_idx - self.history_turns * 2)
            context_turns = [
                self._format_turn_with_speaker(speakers[i], dialog[i]) for i in range(start_idx, turn_idx)
            ]
            if not context_turns:
                continue
            history_text = ' '.join(context_turns)
            target_text = dialog[turn_idx].strip()
            sample = self._encode_pair(history_text, target_text)
            if sample is None:
                continue
            sample['emotion_id'] = emotion_id
            sample['emotion_label'] = emotion_label
            sample['user_text'] = dialog[user_idx].strip()
            sample['target_text'] = target_text
            self.samples.append(sample)

    def _encode_pair(self, history_text: str, target_text: str) -> Optional[Dict]:
        inp_ids = self.tokenizer.encode(history_text, add_special_tokens=False, truncation=True, max_length=self.max_length // 2)
        tgt_budget = self.max_length - len(inp_ids) - 3  # bos, sep, eos
        if tgt_budget <= 0:
            inp_ids = inp_ids[-(self.max_length - 3):]
            tgt_budget = self.max_length - len(inp_ids) - 3
        tgt_ids = self.tokenizer.encode(target_text, add_special_tokens=False, truncation=True, max_length=max(tgt_budget, 1))
        if not tgt_ids:
            return None
        ids = [self.special.bos] + inp_ids + [self.special.sep] + tgt_ids[:tgt_budget] + [self.special.eos]
        if len(ids) > self.max_length:
            ids = ids[: self.max_length - 1] + [self.special.eos]
        sep_pos = len(inp_ids) + 1
        return {'ids': ids, 'sep_pos': sep_pos, 'input_text': history_text, 'target_text': target_text}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class BatchCollator:
    def __init__(self, special: SpecialTokenIds):
        self.special = special

    def __call__(self, batch: Sequence[Dict]):
        lengths = [len(item['ids']) for item in batch]
        max_len = max(lengths)
        B = len(batch)
        X = torch.full((B, max_len), self.special.pad, dtype=torch.long)
        pad_mask = torch.ones((B, max_len), dtype=torch.bool)
        in_mask = torch.zeros((B, max_len), dtype=torch.bool)
        out_mask = torch.zeros((B, max_len), dtype=torch.bool)
        for i, item in enumerate(batch):
            ids = item['ids']
            L = len(ids)
            X[i, :L] = torch.tensor(ids, dtype=torch.long)
            pad_mask[i, :L] = False
            sep_pos = item['sep_pos']
            if sep_pos > 1:
                in_mask[i, 1:sep_pos] = True
            if L - (sep_pos + 1) > 1:
                out_mask[i, sep_pos + 1 : L - 1] = True
        input_texts = [item['user_text'] for item in batch]
        target_texts = [item['target_text'] for item in batch]
        return X, pad_mask, in_mask, out_mask, input_texts, target_texts


def create_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    special_ids: SpecialTokenIds,
    split: str,
    dataset_name: str = 'daily_dialog',
    history_turns: int = 3,
    max_length: int = 256,
    include_neutral: bool = True,
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    dataset = EmotionConversationDataset(
        tokenizer=tokenizer,
        special_ids=special_ids,
        split=split,
        dataset_name=dataset_name,
        history_turns=history_turns,
        max_length=max_length,
        include_neutral=include_neutral,
    )
    collate = BatchCollator(special_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
