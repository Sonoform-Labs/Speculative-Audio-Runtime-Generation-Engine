"""ElevenLabs text-to-SFX backend (optional).

Fastest way to get *real* neural SFX flowing: needs only an API key and the
`requests` package (`pip install 'predictive-neural-sfx[elevenlabs]'`).

Note on latency: this is a cloud round-trip (hundreds of ms to seconds) and is
billed per generation, so it is meant for offline/at-load baking and prototype
content, NOT the runtime hot path. For low, controllable latency use the local
`StableAudioGenerator` instead.
"""
from __future__ import annotations

import os

import numpy as np

from .base import Generator
from ..types import AudioClip, CacheKey

_PROMPT_TEMPLATES = {
    "footstep": "a single {intensity} footstep on {material}, close mic, dry, isolated game sound effect",
    "impact":   "a single {intensity} impact on {material}, close mic, dry, isolated game sound effect",
}
_DEFAULT_TEMPLATE = "a single {intensity} {event} on {material}, dry, isolated sound effect"


class ElevenLabsGenerator(Generator):
    name = "elevenlabs"
    latency_ms = 800.0  # nominal cloud round-trip; not the hot path

    def __init__(self, api_key: str | None = None,
                 model_id: str = "eleven_text_to_sound_v2",
                 duration_s: float = 0.5,
                 sample_rate: int = 44_100):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.model_id = model_id
        self.duration_s = float(duration_s)
        self.sample_rate = int(sample_rate)

    def _prompt(self, key: CacheKey) -> str:
        tmpl = _PROMPT_TEMPLATES.get(key.event_type, _DEFAULT_TEMPLATE)
        return tmpl.format(intensity=key.intensity, material=key.material, event=key.event_type)

    def generate(self, key: CacheKey, *, variant: int = 0) -> AudioClip:
        if not self.api_key:
            raise RuntimeError(
                "ElevenLabs API key not set. Pass api_key=... or set ELEVENLABS_API_KEY."
            )
        try:
            import requests
        except ImportError as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "requests not installed. Run: pip install 'predictive-neural-sfx[elevenlabs]'"
            ) from e

        # Request raw little-endian 16-bit PCM so we can decode with numpy alone.
        fmt = f"pcm_{self.sample_rate}"
        resp = requests.post(
            "https://api.elevenlabs.io/v1/sound-generation",
            params={"output_format": fmt},
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
            json={
                "text": self._prompt(key),
                "model_id": self.model_id,
                "duration_seconds": self.duration_s,
                # vary prompt determinism per variant for distinct takes
                "prompt_influence": 0.3,
            },
            timeout=60,
        )
        resp.raise_for_status()
        pcm = np.frombuffer(resp.content, dtype="<i2").astype(np.float32) / 32768.0
        return AudioClip(pcm, self.sample_rate, variant_id=variant)
