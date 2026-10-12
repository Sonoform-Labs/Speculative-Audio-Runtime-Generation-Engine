from __future__ import annotations

import numpy as np

from .base import Generator
from ..loudness import normalize_loudness, target_for
from ..types import AudioClip, CacheKey, seed_for

_PROMPT_TEMPLATES = {
    "footstep": "a single {intensity} footstep on {material}, close mic, dry, isolated sound effect",
    "impact": "a single {intensity} impact on {material}, close mic, dry, isolated sound effect",
    "jump": "a single {intensity} jump take-off scuff on {material}, dry, isolated sound effect",
    "land": "a single {intensity} heavy landing thud on {material}, dry, isolated sound effect",
}
_DEFAULT_TEMPLATE = "a single {intensity} {event} on {material}, dry, isolated sound effect"

_MATERIAL_ADJ = {
    "metal": "metallic", "glass": "glassy", "water": "watery", "wood": "wooden",
    "stone": "stone", "gravel": "gravelly", "grass": "grassy", "dirt": "earthy",
    "ceramic": "ceramic", "plastic": "plastic", "cloth": "soft cloth",
}
_FOOTSTEP_PLAIN = {"grass"}
_IMPACT_EVENTS = {"hit", "impact", "strike", "clang", "shatter", "break", "smash", "land"}
_IMPACT_TAIL = {
    "metal": "a bright ringing resonant clang with a sustained metallic tail",
    "glass": "sharp shattering glass shards",
    "wood": "a solid wooden knock",
    "stone": "a hard stony thud",
    "ceramic": "a brittle ceramic shatter",
}
_SPLASHY = {"water", "liquid", "mud"}


def build_prompt(event: str, material: str, intensity: str = "normal") -> str:
    lead = "" if intensity in (None, "", "normal") else f"{intensity} "
    if event == "footstep" and material in _SPLASHY:
        return (f"a {intensity or 'normal'} footstep splashing through shallow {material}, a clear distinct wet step, "
                f"dry, isolated sound effect")
    if event in ("splash", "drip", "pour") or material in _SPLASHY:
        return f"a {lead}{event} in {material}, wet splashing water with droplets and bubbling, dry, isolated sound effect"
    if event == "footstep":
        adj = "" if material in _FOOTSTEP_PLAIN else _MATERIAL_ADJ.get(material, material) + " "
        return f"a {intensity or 'normal'} {adj}footstep on {material}, a clear distinct footstep, dry, isolated sound effect"
    if event in _IMPACT_EVENTS:
        tail = _IMPACT_TAIL.get(material)
        if tail:
            return f"a {lead}{event} on {material}, {tail}, dry, isolated sound effect"
    adj = _MATERIAL_ADJ.get(material, "")
    adj = f"{adj} " if adj else ""
    return f"a {lead}{adj}{event} on {material}, dry, isolated sound effect"


def _seed(key: CacheKey, variant: int) -> int:
    return seed_for(key, variant)


def _loudest(mono: np.ndarray, win: int) -> int:
    n = mono.shape[0]
    c = np.empty(n + 1)
    c[0] = 0.0
    np.cumsum(mono.astype(np.float64) ** 2, out=c[1:])
    lo = win - 1 - (win - 1) // 2
    hi = (win - 1) // 2 + 1
    up = np.empty(n)
    m = n + 1 - hi
    up[:m] = c[hi:]
    up[m:] = c[n]
    low = np.empty(n)
    low[:lo] = 0.0
    low[lo:] = c[:n - lo]
    return int(np.argmax(up - low))


def _extract(audio: np.ndarray, sr: int, duration_s: float, preroll_s: float = 0.03) -> np.ndarray:
    n = int(duration_s * sr)
    if n <= 0 or audio.shape[0] <= n:
        return audio
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    win = max(1, int(0.005 * sr))
    center = _loudest(mono, win)
    start = max(0, center - int(preroll_s * sr))
    end = min(mono.shape[0], start + n)
    start = max(0, end - n)
    seg = audio[start:end].copy()
    shape = (-1,) if seg.ndim == 1 else (-1, 1)
    fi = min(int(0.005 * sr), len(seg) // 2)
    fo = min(int(0.05 * sr), len(seg) // 2)
    if fi > 0:
        seg[:fi] *= np.linspace(0.0, 1.0, fi, dtype=seg.dtype).reshape(shape)
    if fo > 0:
        seg[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=seg.dtype).reshape(shape)
    return seg


def _natural(audio: np.ndarray, sr: int, floor_db: float = -55.0, onset_db: float = -30.0, preroll_s: float = 0.02,
             fade_s: float = 0.03) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    x = a if a.ndim == 1 else a.mean(axis=1)
    n = x.shape[0]
    win = max(1, int(0.01 * sr))
    if n <= win or not np.any(x):
        return a
    c = np.empty(n + 1)
    c[0] = 0.0
    np.cumsum(x.astype(np.float64) ** 2, out=c[1:])
    env = np.sqrt(np.maximum(c[win:] - c[:-win], 0.0) / win)
    top = float(env.max())
    on = np.flatnonzero(np.abs(x) >= float(np.max(np.abs(x))) * 10.0 ** (onset_db / 20.0))
    start = max(0, int(on[0]) - int(preroll_s * sr))
    alive = np.flatnonzero(env >= top * 10.0 ** (floor_db / 20.0))
    end = min(n, int(alive[-1]) + win + int(fade_s * sr))
    seg = a[start:end].copy()
    m = seg.shape[0]
    fi = min(int(0.002 * sr), m // 2)
    fo = min(int(fade_s * sr), m // 2)
    shape = (-1,) if seg.ndim == 1 else (-1, 1)
    if fi > 0:
        seg[:fi] *= np.linspace(0.0, 1.0, fi, dtype=seg.dtype).reshape(shape)
    if fo > 0:
        seg[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=seg.dtype).reshape(shape)
    return seg


class StableAudioGenerator(Generator):
    name = "stableaudio"
    latency_ms = 300.0

    def __init__(self, model_name: str = "stabilityai/stable-audio-open-small",
                 device: str | None = None, steps: int = 8, cfg_scale: float = 4.0,
                 sampler_type: str = "pingpong", seconds_total: float | None = 1.0,
                 duration_s: float = 0.6, sample_rate: int = 44_100,
                 prompt_template: str | None = None, negative_prompt: str | None = None):
        self.model_name = model_name
        self.device = device
        self.steps = int(steps)
        self.cfg_scale = float(cfg_scale)
        self.sampler_type = sampler_type
        self.seconds_total = seconds_total
        self.duration_s = float(duration_s)
        self.sample_rate = int(sample_rate)
        self.prompt_template = prompt_template
        self.negative_prompt = negative_prompt
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
        tmpl = self.prompt_template or _PROMPT_TEMPLATES.get(key.event_type, _DEFAULT_TEMPLATE)
        return tmpl.format(intensity=key.intensity, material=key.material, event=key.event_type)

    def generate(self, key: CacheKey, *, variant: int = 0) -> AudioClip:
        self._ensure_loaded()
        import torch
        from einops import rearrange
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        conditioning = [{"prompt": self._prompt(key), "seconds_start": 0,
                         "seconds_total": self.seconds_total}]
        extra = {}
        if self.negative_prompt:
            extra["negative_conditioning"] = [{"prompt": self.negative_prompt, "seconds_start": 0,
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
                **extra,
            )
        audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32).cpu().numpy()
        mono = audio.mean(axis=0) if audio.ndim > 1 else audio
        mono = _extract(mono, self.sample_rate, self.duration_s)
        mono = normalize_loudness(mono, target_rms=target_for(key.intensity))
        return AudioClip(mono.astype(np.float32), self.sample_rate, variant_id=variant)
