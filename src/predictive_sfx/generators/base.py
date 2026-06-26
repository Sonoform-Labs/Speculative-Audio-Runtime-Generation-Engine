from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import AudioClip, CacheKey, SAMPLE_RATE


class Generator(ABC):

    name: str = "base"
    latency_ms: float = 0.0
    sample_rate: int = SAMPLE_RATE

    @abstractmethod
    def generate(self, key: CacheKey, *, variant: int = 0) -> AudioClip:
        raise NotImplementedError

    def generate_batch(self, items: list) -> list:
        return [self.generate(key, variant=variant) for key, variant in items]

    def ensure_ready(self):
        pass
