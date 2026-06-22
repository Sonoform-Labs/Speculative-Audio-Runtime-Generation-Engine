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
    if name in ("stableaudio3", "sa3", "stable-audio-3", "stable_audio_3"):
        from .stable_audio3 import StableAudio3Generator
        return StableAudio3Generator(**kwargs)
    raise ValueError(f"unknown generator {name!r} (choose: mock, stableaudio, stableaudio3)")
