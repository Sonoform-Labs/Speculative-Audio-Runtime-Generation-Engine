from __future__ import annotations

import hashlib
import os

from .base import Generator
from ..types import AudioClip, CacheKey

CACHE_SCHEMA = "prompt_enrichment_v1"


class CachedGenerator(Generator):
    def __init__(self, inner: Generator, cache_dir: str):
        self.inner = inner
        self.cache_dir = cache_dir
        self.name = inner.name
        self.sample_rate = inner.sample_rate
        self.latency_ms = inner.latency_ms
        os.makedirs(cache_dir, exist_ok=True)

    def ensure_ready(self):
        r = getattr(self.inner, "ensure_ready", None)
        if r:
            r()

    def _key_prompt(self, key: CacheKey) -> str:
        fn = getattr(self.inner, "_prompt", None)
        if fn is not None:
            try:
                return fn(key)
            except Exception:
                pass
        return getattr(self.inner, "prompts", {}).get(str(key), "")

    def _key_steps(self, key: CacheKey):
        fn = getattr(self.inner, "_steps_for", None)
        if fn is not None:
            try:
                return fn(key)
            except Exception:
                pass
        return getattr(self.inner, "steps", 0)

    def _path(self, key: CacheKey, variant: int) -> str:
        g = self.inner
        sig = "|".join(str(x) for x in (
            CACHE_SCHEMA,
            getattr(g, "name", "?"), getattr(g, "model_name", ""),
            key,
            self._key_prompt(key),
            getattr(g, "sampler_type", ""),
            self._key_steps(key),
            getattr(g, "cfg_scale", ""),
            getattr(g, "gen_duration_s", ""), getattr(g, "duration_s", ""),
            getattr(g, "sample_rate", ""),
            getattr(g, "duration_padding_s", ""), getattr(g, "mid_noise", ""),
            variant,
        ))
        h = hashlib.md5(sig.encode()).hexdigest()[:16]
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
