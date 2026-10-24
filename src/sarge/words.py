from __future__ import annotations

import re

_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")
_SPLIT = re.compile(r"[^a-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_TOKEN = re.compile(r"[^A-Za-z0-9]+")
_TAKE = re.compile(r"(?:v|ver|var|take|tk|alt)?[0-9]+")
_GLUED = re.compile(r"([a-z]{3,})[0-9]+")
FORMAT_WORDS = frozenset(("sfx", "sfxs", "fx", "snd", "sound", "sounds", "audio", "wav", "ogg", "mp3", "aif", "aiff",
                          "flac"))
ENGINE_WORDS = frozenset(("prefab", "clone", "instance", "lod", "mesh", "gameobject", "geo", "sm", "prop", "variant"))
SIDE_WORDS = frozenset(("l", "r"))
LEAD_WORDS = frozenset(("a", "play"))
TAKE_MARKS = frozenset(("v", "ver", "var", "take", "tk", "alt"))


def split_words(text) -> list:
    return [w for w in _SPLIT.split(_BOUNDARY.sub(" ", str(text or "")).lower()) if w]


def phrase(text) -> str:
    return " ".join(split_words(text))


def name_tokens(text) -> list:
    return [t.lower() for t in _TOKEN.split(_CAMEL.sub(" ", str(text or ""))) if t]


def clean_words(text) -> list:
    toks = name_tokens(text)
    while toks and (_TAKE.fullmatch(toks[-1]) or len(toks[-1]) == 1):
        last = toks.pop()
        if last.isdigit() and toks and toks[-1] in TAKE_MARKS:
            toks.pop()
    out = []
    for i, t in enumerate(toks):
        if t.isdigit() or t in FORMAT_WORDS or t in ENGINE_WORDS or t in SIDE_WORDS:
            continue
        if not out and t in LEAD_WORDS and i < len(toks) - 1:
            continue
        m = _GLUED.fullmatch(t)
        out.append(m.group(1) if m else t)
    return split_words(" ".join(out)) if out else split_words(text)


def clean_phrase(text) -> str:
    return " ".join(clean_words(text))


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
