"""A dependency-free synthetic generator.

It produces a short, material-flavoured percussive burst so the whole
pipeline (predict -> cache -> fallback -> metrics) can be built and tested
with zero external setup. Swap in a real neural backend later behind the same
interface. Output is deterministic for a given (key, variant).
"""
from __future__ import annotations

import numpy as np

from .base import Generator
from ..types import AudioClip, CacheKey, SAMPLE_RATE

# Per-material synthesis profiles: a rough caricature, not a real foley model.
#   freq:  body resonance (Hz)
#   noise: how noise-dominated vs tonal the hit is (0..1)
#   decay: amplitude decay rate (higher = snappier)
#   ring:  amount of metallic overtone ring
_MATERIAL_PROFILE = {
    "wood":  dict(freq=220.0, noise=0.55, decay=24.0, ring=0.12),
    "stone": dict(freq=140.0, noise=0.72, decay=32.0, ring=0.05),
    "grass": dict(freq=95.0,  noise=0.95, decay=42.0, ring=0.00),
    "metal": dict(freq=440.0, noise=0.30, decay=8.0,  ring=0.60),
    "water": dict(freq=70.0,  noise=0.90, decay=18.0, ring=0.00),
    "dirt":  dict(freq=110.0, noise=0.85, decay=36.0, ring=0.02),
}
_INTENSITY_GAIN = {"soft": 0.40, "normal": 0.70, "hard": 1.0}


class MockGenerator(Generator):
    name = "mock"

    def __init__(self, latency_ms: float = 120.0, duration_s: float = 0.18,
                 sample_rate: int = SAMPLE_RATE):
        self.latency_ms = float(latency_ms)
        self.duration_s = float(duration_s)
        self.sample_rate = int(sample_rate)

    def generate(self, key: CacheKey, *, variant: int = 0) -> AudioClip:
        prof = _MATERIAL_PROFILE.get(key.material, _MATERIAL_PROFILE["dirt"])
        gain = _INTENSITY_GAIN.get(key.intensity, 0.70)

        # Deterministic per (key, variant) so caches/tests are reproducible.
        seed = (hash(str(key)) ^ (variant * 2654435761)) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        n = int(self.duration_s * self.sample_rate)
        t = np.arange(n) / self.sample_rate
        env = np.exp(-prof["decay"] * t)

        noise = rng.standard_normal(n) * prof["noise"]
        freq = prof["freq"] * (1.0 + rng.uniform(-0.06, 0.06))  # per-variant pitch jitter
        tone = np.sin(2 * np.pi * freq * t) * (1.0 - prof["noise"])
        ring = prof["ring"] * np.sin(2 * np.pi * freq * 2.0 * t) * np.exp(-6.0 * t)

        sig = (noise + tone + ring) * env
        peak = float(np.max(np.abs(sig))) or 1.0
        sig = (sig / peak) * gain
        return AudioClip(sig.astype(np.float32), self.sample_rate, variant_id=variant)
