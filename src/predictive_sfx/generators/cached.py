from __future__ import annotations

import hashlib
import os

from .base import Generator
from ..types import AudioClip, CacheKey


class CachedGenerator(Generator):
    def __init__(self, inner: Generator, cache_dir: str):
        self.inner = inner
        self.cache_dir = cache_dir
        self.name = inner.name
        self.sample_rate = inner.sample_rate
        self.latency_ms = inner.latency_ms
        os.makedirs(cache_dir, exist_ok=True)

    def _tag(self) -> str:
        return f"{self.inner.name}|{getattr(self.inner, 'steps', 0)}|{getattr(self.inner, 'cfg_scale', 0)}"

    def _path(self, key: CacheKey, variant: int) -> str:
        pr = getattr(self.inner, "prompts", {}).get(str(key), "")
        h = hashlib.md5(f"{self._tag()}|{key}|{pr}|{variant}".encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{h}.wav")

    def generate(self, key: CacheKey, *, variant: int = 0) -> AudioClip:
        path = self._path(key, variant)
        if os.path.exists(path):
            clip = AudioClip.from_wav(path)
            clip.variant_id = variant
            return clip
        clip = self.inner.generate(key, variant=variant)
        clip.to_wav(path)
        return clip

    def generate_batch(self, items: list) -> list:
        results = [None] * len(items)
        misses = []
        for i, (key, variant) in enumerate(items):
            path = self._path(key, variant)
            if os.path.exists(path):
                clip = AudioClip.from_wav(path)
                clip.variant_id = variant
                results[i] = clip
            else:
                misses.append(i)
        if misses:
            made = self.inner.generate_batch([items[i] for i in misses])
            for j, i in enumerate(misses):
                clip = made[j]
                clip.to_wav(self._path(*items[i]))
                results[i] = clip
        return results
