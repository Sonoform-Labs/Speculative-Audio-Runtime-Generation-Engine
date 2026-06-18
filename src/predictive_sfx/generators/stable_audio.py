from __future__ import annotations

import numpy as np

from .base import Generator
from ..types import AudioClip, CacheKey

_PROMPT_TEMPLATES = {
    "footstep": "a single {intensity} footstep on {material}, close mic, dry, isolated sound effect",
    "impact":   "a single {intensity} impact on {material}, close mic, dry, isolated sound effect",
}
_DEFAULT_TEMPLATE = "a single {intensity} {event} on {material}, dry, isolated sound effect"


class StableAudioGenerator(Generator):
    name = "stableaudio"
    latency_ms = 300.0

    def __init__(self, model_name: str = "stabilityai/stable-audio-open-small",
                 device: str | None = None, steps: int = 8, duration_s: float = 0.5,
                 sample_rate: int = 44_100):
        self.model_name = model_name
        self.device = device
        self.steps = int(steps)
        self.duration_s = float(duration_s)
        self.sample_rate = int(sample_rate)
        self._model = None
        self._cfg = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import torch
            from stable_audio_tools import get_pretrained_model
        except ImportError as e:
            raise RuntimeError(
                "Stable Audio deps missing. Run: "
                "pip install 'predictive-neural-sfx[stableaudio]' and `huggingface-cli login`."
            ) from e
        import torch
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, self._cfg = get_pretrained_model(self.model_name)
        self._model = self._model.to(self.device)

    def _prompt(self, key: CacheKey) -> str:
        tmpl = _PROMPT_TEMPLATES.get(key.event_type, _DEFAULT_TEMPLATE)
        return tmpl.format(intensity=key.intensity, material=key.material, event=key.event_type)

    def generate(self, key: CacheKey, *, variant: int = 0) -> AudioClip:
        self._ensure_loaded()
        import torch
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        sr = self._cfg["sample_rate"]
        conditioning = [{
            "prompt": self._prompt(key),
            "seconds_start": 0,
            "seconds_total": self.duration_s,
        }]
        with torch.no_grad():
            audio = generate_diffusion_cond(
                self._model,
                steps=self.steps,
                conditioning=conditioning,
                sample_size=int(sr * self.duration_s),
                device=self.device,
                seed=(hash(str(key)) ^ (variant * 2654435761)) & 0x7FFFFFFF,
            )
        wav = audio.squeeze(0).to(torch.float32).cpu().numpy()
        mono = wav.mean(axis=0) if wav.ndim > 1 else wav
        return AudioClip(mono.astype(np.float32), int(sr), variant_id=variant)
