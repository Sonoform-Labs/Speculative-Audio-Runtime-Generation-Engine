from __future__ import annotations

import hashlib

import numpy as np

from .base import Generator
from ..types import AudioClip, CacheKey

_PROMPT_TEMPLATES = {
    "footstep": "a single {intensity} footstep on {material}, close mic, dry, isolated sound effect",
    "impact": "a single {intensity} impact on {material}, close mic, dry, isolated sound effect",
}
_DEFAULT_TEMPLATE = "a single {intensity} {event} on {material}, dry, isolated sound effect"


def _seed(key: CacheKey, variant: int) -> int:
    base = int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16)
    return (base ^ (variant * 2654435761)) & 0x7FFFFFFF


class StableAudioGenerator(Generator):
    name = "stableaudio"
    latency_ms = 300.0

    def __init__(self, model_name: str = "stabilityai/stable-audio-open-small",
                 device: str | None = None, steps: int = 8, cfg_scale: float = 1.0,
                 sampler_type: str = "pingpong", seconds_total: float | None = None,
                 duration_s: float = 0.6, sample_rate: int = 44_100):
        self.model_name = model_name
        self.device = device
        self.steps = int(steps)
        self.cfg_scale = float(cfg_scale)
        self.sampler_type = sampler_type
        self.seconds_total = seconds_total
        self.duration_s = float(duration_s)
        self.sample_rate = int(sample_rate)
        self._model = None
        self._sample_size = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from stable_audio_tools import get_pretrained_model
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, cfg = get_pretrained_model(self.model_name)
        self._model = model.to(self.device).eval()
        self.sample_rate = int(cfg["sample_rate"])
        self._sample_size = int(cfg["sample_size"])
        if self.seconds_total is None:
            self.seconds_total = self._sample_size / self.sample_rate

    def _prompt(self, key: CacheKey) -> str:
        tmpl = _PROMPT_TEMPLATES.get(key.event_type, _DEFAULT_TEMPLATE)
        return tmpl.format(intensity=key.intensity, material=key.material, event=key.event_type)

    def generate(self, key: CacheKey, *, variant: int = 0) -> AudioClip:
        self._ensure_loaded()
        import torch
        from einops import rearrange
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        conditioning = [{"prompt": self._prompt(key), "seconds_start": 0,
                         "seconds_total": self.seconds_total}]
        with torch.no_grad():
            audio = generate_diffusion_cond(
                self._model,
                steps=self.steps,
                cfg_scale=self.cfg_scale,
                conditioning=conditioning,
                sample_size=self._sample_size,
                sample_rate=self.sample_rate,
                seed=_seed(key, variant),
                device=self.device,
                sampler_type=self.sampler_type,
            )
        audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32)
        peak = torch.max(torch.abs(audio))
        if peak > 0:
            audio = audio / peak
        audio = audio.clamp(-1.0, 1.0).cpu().numpy()
        mono = audio.mean(axis=0) if audio.ndim > 1 else audio
        n = int(self.duration_s * self.sample_rate)
        if 0 < n < mono.shape[0]:
            mono = mono[:n]
        return AudioClip(mono.astype(np.float32), self.sample_rate, variant_id=variant)
