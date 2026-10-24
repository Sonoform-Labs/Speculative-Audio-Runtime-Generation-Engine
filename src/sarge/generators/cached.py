from __future__ import annotations

import hashlib
import os
import queue
import threading

from .base import Generator
from ..types import AudioClip, CacheKey

CACHE_SCHEMA = "prompt_enrichment_v1"


class CachedGenerator(Generator):
    def __init__(self, inner: Generator, cache_dir: str, async_write: bool = False):
        self.inner = inner
        self.cache_dir = cache_dir
        self.name = inner.name
        self.sample_rate = inner.sample_rate
        self.latency_ms = inner.latency_ms
        os.makedirs(cache_dir, exist_ok=True)
        self.async_write = bool(async_write)
        self._pending = {}
        self._pending_lock = threading.Lock()
        self._writes = None
        if self.async_write:
            self._writes = queue.Queue()
            threading.Thread(target=self._writer, name="sfx-cache-writer", daemon=True).start()

    def _writer(self):
        while True:
            path, clip, done = self._writes.get()
            try:
                clip.to_wav(path)
            except Exception:
                pass
            finally:
                with self._pending_lock:
                    if self._pending.get(path) is done:
                        del self._pending[path]
                done.set()

    def _store(self, path: str, clip: AudioClip):
        if self._writes is None:
            clip.to_wav(path)
            return
        done = threading.Event()
        with self._pending_lock:
            self._pending[path] = done
        self._writes.put((path, clip, done))

    def _settled(self, path: str):
        with self._pending_lock:
            done = self._pending.get(path)
        if done is not None:
            done.wait()

    def flush(self, timeout: float | None = None) -> bool:
        with self._pending_lock:
            waits = list(self._pending.values())
        return all(d.wait(timeout) for d in waits)

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

    def _path(self, key: CacheKey, variant: int, prompt: str | None = None) -> str:
        g = self.inner
        tag = getattr(g, "post_tag", "")
        side = g.route_tag(key) if callable(getattr(g, "route_tag", None)) else ""
        tag = tag + "|" + side if side else tag
        sig = "|".join(str(x) for x in (
            CACHE_SCHEMA,
            getattr(g, "name", "?"), getattr(g, "model_name", ""),
            key,
            prompt if prompt else self._key_prompt(key),
            getattr(g, "sampler_type", ""),
            self._key_steps(key),
            getattr(g, "cfg_scale", ""),
            getattr(g, "gen_duration_s", ""), getattr(g, "duration_s", ""),
            getattr(g, "sample_rate", ""),
            getattr(g, "duration_padding_s", ""), getattr(g, "mid_noise", ""),
            getattr(g, "static_guard", ""),
            variant,
        ) + ((tag,) if tag else ()))
        h = hashlib.md5(sig.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{h}.wav")

    def generate(self, key: CacheKey, *, variant: int = 0, prompt: str | None = None) -> AudioClip:
        path = self._path(key, variant, prompt)
        self._settled(path)
        if os.path.exists(path):
            clip = AudioClip.from_wav(path, keep_channels=True)
            clip.variant_id = variant
            return clip
        clip = self.inner.generate(key, variant=variant, prompt=prompt) if prompt else self.inner.generate(
            key, variant=variant)
        self._store(path, clip)
        return clip

    def generate_batch(self, items: list) -> list:
        results = [None] * len(items)
        misses = []
        for i, (key, variant) in enumerate(items):
            path = self._path(key, variant)
            self._settled(path)
            if os.path.exists(path):
                clip = AudioClip.from_wav(path, keep_channels=True)
                clip.variant_id = variant
                results[i] = clip
            else:
                misses.append(i)
        if misses:
            made = self.inner.generate_batch([items[i] for i in misses])
            for j, i in enumerate(misses):
                clip = made[j]
                self._store(self._path(*items[i]), clip)
                results[i] = clip
        return results
