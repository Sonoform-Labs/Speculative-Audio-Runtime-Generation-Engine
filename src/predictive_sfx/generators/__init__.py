"""Pluggable SFX generators.

All generators implement the same `Generator` interface, so the rest of the
pipeline never cares whether audio came from a mock synth, a cloud API, or a
local diffusion model. Start on `MockGenerator` (zero dependencies), then swap
in a real backend behind the same interface.
"""
from .base import Generator
from .mock import MockGenerator

__all__ = ["Generator", "MockGenerator", "get_generator"]


def get_generator(name: str, **kwargs) -> Generator:
    """Factory: resolve a generator by name (lazy-imports optional backends)."""
    name = name.lower()
    if name == "mock":
        return MockGenerator(**kwargs)
    if name == "elevenlabs":
        from .elevenlabs import ElevenLabsGenerator
        return ElevenLabsGenerator(**kwargs)
    if name in ("stableaudio", "stable_audio", "stable-audio"):
        from .stable_audio import StableAudioGenerator
        return StableAudioGenerator(**kwargs)
    raise ValueError(f"unknown generator {name!r} (choose: mock, elevenlabs, stableaudio)")
