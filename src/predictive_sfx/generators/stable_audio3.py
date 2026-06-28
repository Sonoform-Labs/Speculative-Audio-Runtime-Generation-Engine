from __future__ import annotations

import numpy as np

from .base import Generator
from ..loudness import normalize_loudness, target_for
from ..types import AudioClip, CacheKey
from .stable_audio import _extract, _seed, build_prompt


class StableAudio3Generator(Generator):
    name = "stableaudio3"
    latency_ms = 190.0

    def __init__(self, model_name: str = "small-sfx", device: str | None = None,
                 steps: int = 3, cfg_scale: float = 2.0, gen_duration_s: float = 1.0,
                 duration_s: float = 0.6, sample_rate: int = 44_100,
                 prompt_template: str | None = None, negative_prompt: str | None = None,
                 model_half: bool = True, sampler_type: str = "euler",
                 detail_steps: int = 8, detail_keys=None, **_):
        self.model_name = model_name
        self.device = device
        self.steps = int(steps)
        self.sampler_type = str(sampler_type)
        self.detail_steps = int(detail_steps)
        self.detail_keys = set(detail_keys) if detail_keys is not None else {"splash", "water", "liquid"}
        self.cfg_scale = float(cfg_scale)
        self.gen_duration_s = float(gen_duration_s)
        self.duration_s = float(duration_s)
        self.sample_rate = int(sample_rate)
        self.prompt_template = prompt_template
        self.negative_prompt = negative_prompt
        self.model_half = model_half
        self.prompts: dict = {}
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from stable_audio_3 import StableAudioModel
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = StableAudioModel.from_pretrained(
            self.model_name, device=self.device, model_half=self.model_half)
        sr = getattr(self._model, "sample_rate", None)
        if not sr:
            cfg = getattr(self._model, "model_config", None) or {}
            sr = cfg.get("sample_rate")
        if sr:
            self.sample_rate = int(sr)

    def ensure_ready(self):
        self._ensure_loaded()

    def _prompt(self, key: CacheKey) -> str:
        override = self.prompts.get(str(key))
        if override:
            return override
        if self.prompt_template:
            return self.prompt_template.format(intensity=key.intensity, material=key.material, event=key.event_type)
        return build_prompt(key.event_type, key.material, key.intensity)

    def _steps_for(self, key: CacheKey) -> int:
        if key.event_type in self.detail_keys or key.material in self.detail_keys:
            return self.detail_steps
        return self.steps

    def generate(self, key: CacheKey, *, variant: int = 0) -> AudioClip:
        self._ensure_loaded()
        import torch
        kw = {}
        if self.negative_prompt:
            kw["negative_prompt"] = self.negative_prompt
        kw["sampler_type"] = self.sampler_type
        with torch.no_grad():
            audio = self._model.generate(
                prompt=self._prompt(key),
                duration=self.gen_duration_s,
                steps=self._steps_for(key),
                cfg_scale=self.cfg_scale,
                seed=_seed(key, variant),
                **kw,
            )
        a = audio.detach().to(torch.float32).cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio, dtype=np.float32)
        while a.ndim > 2:
            a = a[0]
        mono = a.mean(axis=0) if a.ndim == 2 else a
        mono = _extract(mono, self.sample_rate, self.duration_s)
        mono = normalize_loudness(mono, target_rms=target_for(key.intensity))
        return AudioClip(mono.astype(np.float32), self.sample_rate, variant_id=variant)

    def generate_batch(self, items: list) -> list:
        if not items:
            return []
        self._ensure_loaded()
        import torch
        prompts = [self._prompt(key) for key, _v in items]
        kw = {}
        if self.negative_prompt:
            kw["negative_prompt"] = self.negative_prompt
        kw["sampler_type"] = self.sampler_type
        with torch.no_grad():
            audio = self._model.generate(
                prompt=prompts, duration=self.gen_duration_s, steps=self.steps,
                cfg_scale=self.cfg_scale, seed=_seed(items[0][0], items[0][1]),
                batch_size=len(prompts), **kw)
        a = audio.detach().to(torch.float32).cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio, dtype=np.float32)
        out = []
        for i, (key, variant) in enumerate(items):
            clip = a[i]
            while clip.ndim > 2:
                clip = clip[0]
            mono = clip.mean(axis=0) if clip.ndim == 2 else clip
            mono = _extract(mono, self.sample_rate, self.duration_s)
            mono = normalize_loudness(mono, target_rms=target_for(key.intensity))
            out.append(AudioClip(mono.astype(np.float32), self.sample_rate, variant_id=variant))
        return out
