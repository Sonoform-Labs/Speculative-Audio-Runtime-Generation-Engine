from __future__ import annotations

import hashlib
import os
import wave
from dataclasses import dataclass
from enum import Enum

import numpy as np

SAMPLE_RATE = 48_000


def seed_for(key, variant: int = 0) -> int:
    s = str(key)
    cls = getattr(key, "cls", "")
    if cls:
        s = s[:-len("|class=" + cls)]
    base = int(hashlib.md5(s.encode()).hexdigest()[:8], 16)
    return (base ^ (variant * 2654435761)) & 0x7FFFFFFF


class Material(str, Enum):
    WOOD = "wood"
    STONE = "stone"
    GRASS = "grass"
    METAL = "metal"
    WATER = "water"
    DIRT = "dirt"


class Intensity(str, Enum):
    SOFT = "soft"
    NORMAL = "normal"
    HARD = "hard"


@dataclass(frozen=True, slots=True)
class CacheKey:
    event_type: str
    material: str
    intensity: str
    context: str = "default"
    phys: str = ""
    cls: str = ""

    def __str__(self) -> str:
        base = f"{self.event_type}|{self.material}|{self.intensity}|{self.context}"
        base = f"{base}|{self.phys}" if self.phys else base
        return f"{base}|class={self.cls}" if self.cls else base


@dataclass(slots=True)
class GameEvent:
    event_type: str
    material: str
    intensity: str
    time_s: float
    context: str = "default"

    @property
    def key(self) -> CacheKey:
        return CacheKey(self.event_type, self.material, self.intensity, self.context)

    @classmethod
    def from_dict(cls, d: dict) -> "GameEvent":
        return cls(
            event_type=d["event_type"],
            material=d["material"],
            intensity=d["intensity"],
            time_s=float(d["time_s"]),
            context=d.get("context", "default"),
        )

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "material": self.material,
            "intensity": self.intensity,
            "time_s": self.time_s,
            "context": self.context,
        }


@dataclass(slots=True)
class AudioClip:
    samples: np.ndarray
    sample_rate: int = SAMPLE_RATE
    variant_id: int = 0
    gen_ms: float | None = None

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else int(self.samples.shape[1])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def mono(self) -> np.ndarray:
        return self.samples if self.samples.ndim == 1 else self.samples.mean(axis=1)

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate

    @property
    def nbytes(self) -> int:
        return int(self.samples.nbytes)

    def to_wav(self, path: str | os.PathLike) -> None:
        path = os.fspath(path)
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        pcm16 = np.round(np.clip(self.samples, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(self.channels)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm16.tobytes())

    @classmethod
    def from_wav(cls, path: str | os.PathLike, keep_channels: bool = False) -> "AudioClip":
        with wave.open(os.fspath(path), "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            raw = w.readframes(w.getnframes())
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
        if ch > 1:
            a = a.reshape(-1, ch)
            if not keep_channels:
                a = a.mean(axis=1)
        return cls(np.ascontiguousarray(a, dtype=np.float32), sr)

    @classmethod
    def silence(cls, duration_s: float = 0.2, sample_rate: int = SAMPLE_RATE) -> "AudioClip":
        n = int(duration_s * sample_rate)
        return cls(np.zeros(n, dtype=np.float32), sample_rate)
