from __future__ import annotations

import threading

from ..amp import family_for
from ..types import AudioClip, CacheKey
from .base import Generator

GUN_FAMILIES = ("blast", "handling")


class RoutedGenerator(Generator):
    name = "hybrid"
    latency_ms = 260.0

    def __init__(self, woosh_host: str = "127.0.0.1", woosh_port: int = 8771, gun_families=GUN_FAMILIES,
                 guns=None, rest=None, warm_guns: bool = True, **sa3_kwargs):
        if guns is None:
            from .stable_audio3 import StableAudio3Generator
            guns = StableAudio3Generator(**sa3_kwargs)
        if rest is None:
            from .remote import RemoteGenerator
            rest = RemoteGenerator(host=woosh_host, port=int(woosh_port))
        self.guns = guns
        self.rest = rest
        self.gun_families = set(gun_families)
        self.sample_rate = int(getattr(guns, "sample_rate", self.sample_rate))
        self.prompts = getattr(guns, "prompts", {})
        self._rest_lock = threading.Lock()
        self.counts = {"guns": 0, "rest": 0}
        self.warm_guns = bool(warm_guns)

    def route(self, key: CacheKey) -> str:
        return "guns" if family_for(key.event_type) in self.gun_families else "rest"

    def ensure_ready(self):
        for g in (self.guns, self.rest):
            ready = getattr(g, "ensure_ready", None)
            if ready:
                ready()
        if self.warm_guns:
            self.guns.generate(CacheKey("shoot", "pistol", "normal"), variant=0)

    def generate(self, key: CacheKey, *, variant: int = 0) -> AudioClip:
        side = self.route(key)
        self.counts[side] += 1
        if side == "guns":
            return self.guns.generate(key, variant=variant)
        with self._rest_lock:
            return self.rest.generate(key, variant=variant)

    def generate_batch(self, items: list) -> list:
        out = [None] * len(items)
        guns = [i for i, (k, _v) in enumerate(items) if self.route(k) == "guns"]
        if guns:
            clips = self.guns.generate_batch([items[i] for i in guns])
            for i, c in zip(guns, clips):
                out[i] = c
            self.counts["guns"] += len(guns)
        for i, (k, v) in enumerate(items):
            if out[i] is None:
                out[i] = self.generate(k, variant=v)
        return out

    def fast_stats(self) -> dict:
        s = dict(getattr(self.guns, "fast_stats", lambda: {})())
        s.update({"routed_" + k: v for k, v in self.counts.items()})
        return s
