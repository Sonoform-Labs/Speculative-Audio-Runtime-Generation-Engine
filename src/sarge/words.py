from __future__ import annotations

import re

_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")
_SPLIT = re.compile(r"[^a-z0-9]+")


def split_words(text) -> list:
    return [w for w in _SPLIT.split(_BOUNDARY.sub(" ", str(text or "")).lower()) if w]


def phrase(text) -> str:
    return " ".join(split_words(text))


def inflects(word: str, stem: str) -> bool:
    return word == stem or (word.startswith(stem) and word[len(stem):] in ("s", "es", "ed", "ing"))


def names(stem_words, words) -> bool:
    k = len(stem_words)
    if not k:
        return False
    for i in range(len(words) - k + 1):
        if words[i:i + k - 1] == stem_words[:-1] and inflects(words[i + k - 1], stem_words[-1]):
            return True
    return False
