from __future__ import annotations

import contextlib
import threading

import numpy as np

from .base import Generator
from ..loudness import normalize_loudness, normalize_peak, peak_for, target_for
from ..static_guard import is_static, retry_seed
from ..types import AudioClip, CacheKey
from .stable_audio import _extract, _natural, _seed, build_prompt


@contextlib.contextmanager
def _skip_first_schedule():
    import stable_audio_3.inference.sampling as smp
    orig = smp.build_schedule

    def patched(*a, **k):
        t = orig(*a, **k)
        return t[1:] if t.dim() == 1 else t[..., 1:]

    smp.build_schedule = patched
    try:
        yield
    finally:
        smp.build_schedule = orig

_RANDOM_INITS = ("uniform_", "normal_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_", "xavier_uniform_",
                 "xavier_normal_", "orthogonal_")


@contextlib.contextmanager
def _skip_random_init():
    import torch.nn.init as init
    saved = {n: getattr(init, n) for n in _RANDOM_INITS if hasattr(init, n)}
    for n in saved:
        setattr(init, n, lambda tensor, *a, **k: tensor)
    try:
        yield
    finally:
        for n, f in saved.items():
            setattr(init, n, f)


STATIC_STATS = ("static_checks", "static_flagged", "static_retries", "static_rescued", "static_exhausted")


class StableAudio3Generator(Generator):
    name = "stableaudio3"
    latency_ms = 68.0

    def __init__(self, model_name: str = "small-sfx", device: str | None = None,
                 steps: int = 3, cfg_scale: float = 2.0, gen_duration_s: float = 1.0,
                 duration_s: float = 0.6, sample_rate: int = 44_100,
                 prompt_template: str | None = None, negative_prompt: str | None = None,
                 model_half: bool = True, sampler_type: str = "euler",
                 detail_steps: int = 8, detail_keys=None, fast: bool = True,
                 duration_padding_s: float = 1.0, mid_noise: float | None = None,
                 hoist_prompts: int | None = 8, static_guard: bool = True, max_static_retries: int = 2,
                 compile_mode: str | None = None, compile_decode: bool = False, cfg_interval=None,
                 graph_decode: bool = False, hoist_rope: bool = False, skip_first_step: bool = False,
                 fp16_accumulation: bool = False, graph_text: bool = False, fast_load: bool = False,
                 unfuse_bias: bool = False, triton_gemm: bool = False, decode_prefix: bool = False,
                 decode_margin_frames: int = 2, triton_proj: bool = False, trim: str = "window",
                 level: str = "rms", decoder_ckpt: str | None = None, amp: bool = False, **_):
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
        self.fast = bool(fast)
        self.duration_padding_s = float(duration_padding_s)
        self.mid_noise = None if mid_noise is None else float(mid_noise)
        self._model = None
        self._fast = None
        self.hoist_prompts = hoist_prompts
        self.compile_mode = compile_mode or None
        self.compile_decode = bool(compile_decode)
        self.graph_decode = bool(graph_decode)
        self.hoist_rope = bool(hoist_rope)
        self.skip_first_step = bool(skip_first_step)
        self.fp16_accumulation = bool(fp16_accumulation)
        self.graph_text = bool(graph_text)
        self.fast_load = bool(fast_load)
        self.unfuse_bias = bool(unfuse_bias)
        self.triton_gemm = bool(triton_gemm)
        self.decode_prefix = bool(decode_prefix)
        self.decode_margin_frames = max(0, int(decode_margin_frames))
        self.triton_proj = bool(triton_proj)
        self.trim = str(trim)
        self.level = str(level)
        self.decoder_ckpt = str(decoder_ckpt) if decoder_ckpt else None
        self.amp = bool(amp)
        self._load_error = None
        self.cfg_interval = None if cfg_interval is None else (float(cfg_interval[0]), float(cfg_interval[1]))
        self._hoist = None
        self._lock = threading.Lock()
        self.static_guard = bool(static_guard)
        self.max_static_retries = max(0, int(max_static_retries))
        self.static_stats = {k: 0 for k in STATIC_STATS}
        self._stats_lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        if self._load_error is not None:
            raise self._load_error
        if self.fast_load:
            from ..fastmeta import install
            install()
        import torch
        from stable_audio_3 import StableAudioModel
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        acc16 = self.fp16_accumulation and hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation")
        if acc16:
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
        if self.unfuse_bias and self.compile_mode:
            import torch._inductor.config as icfg
            if hasattr(icfg, "keep_addmm_fused_for_half_dtypes"):
                icfg.keep_addmm_fused_for_half_dtypes = False
        dec_sd = None
        if self.decoder_ckpt:
            from safetensors.torch import load_file
            try:
                dec_sd = load_file(self.decoder_ckpt)
            except Exception as e:
                self._load_error = e
                raise
        with _skip_random_init() if self.fast_load else contextlib.nullcontext():
            model = StableAudioModel.from_pretrained(
                self.model_name, device=self.device, model_half=self.model_half)
        if dec_sd is not None:
            try:
                model.same.model.decoder.load_state_dict(dec_sd, strict=True)
            except Exception as e:
                self._load_error = e
                del model
                raise
        self._model = model
        sr = getattr(self._model, "sample_rate", None)
        if not sr:
            cfg = getattr(self._model, "model_config", None) or {}
            sr = cfg.get("sample_rate")
        if sr:
            self.sample_rate = int(sr)
        keep = None
        if self.decode_prefix:
            ratio = getattr(getattr(self._model.model, "pretransform", None), "downsampling_ratio", None)
            if ratio:
                keep = int(np.ceil(self.gen_duration_s * self.sample_rate / ratio)) + self.decode_margin_frames
        if self.fast:
            try:
                from .sa3_fast import Accelerator
                self._fast = Accelerator(self._model, compile_mode=self.compile_mode,
                                         compile_decode=self.compile_decode, graph_decode=self.graph_decode,
                                         graph_text=self.graph_text, triton_gemm=self.triton_gemm and acc16,
                                         decode_keep_frames=keep, triton_proj=self.triton_proj)
            except Exception:
                self._fast = None
        if self._fast is not None and self.hoist_prompts is not None:
            try:
                from .sa3_hoist import ALL_PARTS, DEFAULT_PARTS, install_hoist
                self._hoist = install_hoist(self._model, self._fast, cache_prompts=int(self.hoist_prompts),
                                            parts=ALL_PARTS if self.hoist_rope else DEFAULT_PARTS)
            except Exception:
                self._hoist = None

    def ensure_ready(self):
        self._ensure_loaded()

    def _prompt(self, key: CacheKey) -> str:
        override = self.prompts.get(str(key))
        if override:
            return override
        if self.prompt_template:
            return self.prompt_template.format(intensity=key.intensity, material=key.material, event=key.event_type)
        return build_prompt(key.event_type, key.material, key.intensity)

    def _schedule(self):
        if not self.skip_first_step:
            return contextlib.nullcontext()
        return _skip_first_schedule()

    def _extra(self) -> dict:
        return {"cfg_interval": self.cfg_interval} if self.cfg_interval is not None else {}

    def _inputs(self, prompts: list) -> dict:
        if self._fast is not None:
            kw = self._fast.kwargs(prompts, self.gen_duration_s, self.negative_prompt)
        else:
            kw = {"prompt": prompts[0] if len(prompts) == 1 else prompts}
            if self.negative_prompt:
                kw["negative_prompt"] = self.negative_prompt
        kw["sampler_type"] = self.sampler_type
        kw["duration_padding_sec"] = self.duration_padding_s
        if self.mid_noise is not None:
            from stable_audio_3.inference.distribution_shift import FluxDistributionShift
            a = self.mid_noise / (1.0 - self.mid_noise)
            kw["dist_shift"] = FluxDistributionShift(alpha_min=a, alpha_max=a)
        return kw

    def fast_stats(self) -> dict:
        out = self._fast.stats() if self._fast is not None else {}
        if self._hoist is not None:
            out.update(self._hoist.stats())
        if self.static_guard:
            with self._stats_lock:
                out.update(self.static_stats)
        return out

    def _count(self, name: str, k: int = 1):
        with self._stats_lock:
            self.static_stats[name] += k

    def _finish(self, a, key: CacheKey) -> np.ndarray:
        while a.ndim > 2:
            a = a[0]
        mono = a.mean(axis=0) if a.ndim == 2 else a
        if self.trim == "natural":
            mono = _natural(mono, self.sample_rate)
        else:
            mono = _extract(mono, self.sample_rate, self.duration_s)
        if self.amp:
            from ..amp import amplify
            mono = amplify(mono, self.sample_rate, key.event_type, key.intensity)
        if self.level == "peak":
            mono = normalize_peak(mono, peak_for(key.intensity))
        else:
            mono = normalize_loudness(mono, target_rms=target_for(key.intensity))
        return mono.astype(np.float32)

    @staticmethod
    def _numpy(audio):
        if hasattr(audio, "detach"):
            import torch
            return audio.detach().to(torch.float32).cpu().numpy()
        return np.asarray(audio, dtype=np.float32)

    def _guard(self, key: CacheKey, variant: int, mono: np.ndarray) -> np.ndarray:
        bad, m = is_static(mono, self.sample_rate)
        self._count("static_checks")
        if not bad:
            return mono
        self._count("static_flagged")
        import torch
        best, best_flat = mono, m["flat"]
        base = _seed(key, variant)
        steps = self._steps_for(key)
        prompt = self._prompt(key)
        for attempt in range(1, self.max_static_retries + 1):
            self._count("static_retries")
            with self._lock, torch.no_grad(), self._schedule():
                audio = self._model.generate(
                    duration=self.gen_duration_s, steps=steps, cfg_scale=self.cfg_scale,
                    seed=retry_seed(base, attempt), batch_size=1, **self._inputs([prompt]), **self._extra())
            cand = self._finish(self._numpy(audio), key)
            bad, m = is_static(cand, self.sample_rate)
            self._count("static_checks")
            if not bad:
                self._count("static_rescued")
                return cand
            if m["flat"] < best_flat:
                best, best_flat = cand, m["flat"]
        self._count("static_exhausted")
        if not np.isfinite(best).all():
            best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
        return best

    def _steps_for(self, key: CacheKey) -> int:
        if key.event_type in self.detail_keys or key.material in self.detail_keys:
            return self.detail_steps
        return self.steps

    def generate(self, key: CacheKey, *, variant: int = 0) -> AudioClip:
        self._ensure_loaded()
        import torch
        with self._lock, torch.no_grad(), self._schedule():
            audio = self._model.generate(
                duration=self.gen_duration_s,
                steps=self._steps_for(key),
                cfg_scale=self.cfg_scale,
                seed=_seed(key, variant),
                **self._inputs([self._prompt(key)]),
                **self._extra(),
            )
        mono = self._finish(self._numpy(audio), key)
        if self.static_guard:
            mono = self._guard(key, variant, mono)
        return AudioClip(mono, self.sample_rate, variant_id=variant)

    def generate_batch(self, items: list) -> list:
        if not items:
            return []
        self._ensure_loaded()
        import torch
        groups = {}
        for i, (key, _v) in enumerate(items):
            groups.setdefault(self._steps_for(key), []).append(i)
        out = [None] * len(items)
        for steps, idx in groups.items():
            sub = [items[i] for i in idx]
            with self._lock, torch.no_grad(), self._schedule():
                audio = self._model.generate(
                    duration=self.gen_duration_s, steps=steps,
                    cfg_scale=self.cfg_scale, seed=_seed(sub[0][0], sub[0][1]),
                    batch_size=len(sub), **self._inputs([self._prompt(key) for key, _v in sub]), **self._extra())
            a = self._numpy(audio)
            for j, (key, variant) in enumerate(sub):
                mono = self._finish(a[j], key)
                if self.static_guard:
                    mono = self._guard(key, variant, mono)
                out[idx[j]] = AudioClip(mono, self.sample_rate, variant_id=variant)
        return out
