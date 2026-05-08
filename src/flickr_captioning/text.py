from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"


TOKEN_PATTERN = re.compile(r"[^a-z\s]")


def clean_caption(caption: str) -> list[str]:
    caption = caption.lower()
    caption = TOKEN_PATTERN.sub(" ", caption)
    return [token for token in caption.split() if len(token) > 1 or token == "a"]


def prepare_caption(caption: str) -> list[str]:
    return [START_TOKEN, *clean_caption(caption), END_TOKEN]


@dataclass
class Vocabulary:
    min_freq: int = 5
    stoi: dict[str, int] = field(default_factory=dict)
    itos: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.stoi:
            for token in (PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN):
                self.add_token(token)

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK_TOKEN]

    @property
    def start_idx(self) -> int:
        return self.stoi[START_TOKEN]

    @property
    def end_idx(self) -> int:
        return self.stoi[END_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def add_token(self, token: str) -> int:
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
        return self.stoi[token]

    def fit(self, captions: list[str]) -> None:
        counts: Counter[str] = Counter()
        for caption in captions:
            counts.update(clean_caption(caption))
        for token, count in sorted(counts.items()):
            if count >= self.min_freq:
                self.add_token(token)

    def encode(self, caption: str, max_length: int | None = None) -> list[int]:
        tokens = prepare_caption(caption)
        if max_length is not None:
            tokens = tokens[:max_length]
            if tokens[-1] != END_TOKEN:
                tokens[-1] = END_TOKEN
        return [self.stoi.get(token, self.unk_idx) for token in tokens]

    def decode(self, indices: list[int], skip_special: bool = True) -> list[str]:
        special = {PAD_TOKEN, START_TOKEN, END_TOKEN} if skip_special else set()
        tokens = []
        for index in indices:
            token = self.itos[index] if 0 <= index < len(self.itos) else UNK_TOKEN
            if token == END_TOKEN and skip_special:
                break
            if token not in special:
                tokens.append(token)
        return tokens
