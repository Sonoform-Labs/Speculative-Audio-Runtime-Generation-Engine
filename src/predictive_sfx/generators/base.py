"""The generator interface shared by every backend."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import AudioClip, CacheKey, SAMPLE_RATE


class Generator(ABC):
    """Turns a context key into an audio clip.

    Attributes:
        name: short identifier for reports.
        latency_ms: nominal wall-clock time to generate one short clip on the
            target hardware. The simulator uses this to model whether a
            prediction's audio is ready before its event fires. It is NOT the
            audio-callback budget — generation never runs on the audio thread.
        sample_rate: output sample rate of produced clips.
    """

    name: str = "base"
    latency_ms: float = 0.0
    sample_rate: int = SAMPLE_RATE

    @abstractmethod
    def generate(self, key: CacheKey, *, variant: int = 0) -> AudioClip:
        """Generate one variant of the sound for `key`.

        `variant` selects a distinct take of the same logical sound; backends
        should make different variants audibly different but consistent for a
        given (key, variant) pair so results are reproducible.
        """
        raise NotImplementedError
