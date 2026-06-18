from .base import Generator
from .mock import MockGenerator

__all__ = ["Generator", "MockGenerator", "get_generator"]


def get_generator(name: str, **kwargs) -> Generator:
    name = name.lower()
    if name == "mock":
        return MockGenerator(**kwargs)
    if name in ("stableaudio", "stable_audio", "stable-audio"):
        from .stable_audio import StableAudioGenerator
        return StableAudioGenerator(**kwargs)
    raise ValueError(f"unknown generator {name!r} (choose: mock, stableaudio)")
