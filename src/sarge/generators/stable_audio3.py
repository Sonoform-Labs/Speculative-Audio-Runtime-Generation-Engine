from __future__ import annotations

import contextlib
import os
import threading
import time

import numpy as np

from .base import Generator
from ..loudness import normalize_loudness, normalize_peak, peak_for, target_for
from ..startup import stage
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

SA3_SAMPLE_SIZE = 5292032


class _SpanTorch:
    def __init__(self, torch_mod, frames: int):
        self._torch = torch_mod
        self.frames = int(frames)

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def randn(self, *size, **kw):
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (list, tuple)) else size
        if len(shape) != 3 or not all(isinstance(s, int) for s in shape) or shape[-1] >= self.frames:
            return self._torch.randn(*size, **kw)
        return self._torch.randn([*shape[:-1], self.frames], **kw)[..., :shape[-1]].contiguous()

    def randn_like(self, x, **kw):
        n = int(x.shape[-1])
        if kw or x.dim() != 3 or n >= self.frames:
            return self._torch.randn_like(x, **kw)
        full = x.new_empty((*x.shape[:-1], self.frames))
        return self._torch.randn_like(full)[..., :n].contiguous()


@contextlib.contextmanager
def _noise_span(frames: int):
    import stable_audio_3.inference.sampling as smp
    import stable_audio_3.model as sm
    saved = [(m, m.torch) for m in (sm, smp) if not isinstance(m.torch, _SpanTorch)]
    for m, t in saved:
        m.torch = _SpanTorch(t, frames)
    try:
        yield
    finally:
        for m, t in saved:
            m.torch = t


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


class _StageTorch:
    def __init__(self, torch_mod, marks):
        self._torch = torch_mod
        self._marks = marks

    def __getattr__(self, name):
        if name == "float16" and "copy" in self._marks and "cuda" not in self._marks:
            self._marks["cuda"] = stage("medium to device fp32", self._marks["copy"])
        return getattr(self._torch, name)


def _timed(fn, label, marks, name):
    def run(*a, **k):
        t = time.time()
        out = fn(*a, **k)
        marks[name] = stage(label(a) if callable(label) else label, t)
        return out

    return run


def _mb(a):
    try:
        return "medium weights mmap %.0f MB" % (os.path.getsize(a[0]) / 1e6)
    except (OSError, IndexError, TypeError):
        return "medium weights mmap"


@contextlib.contextmanager
def _load_stages():
    try:
        import stable_audio_3.factory as fac
        import stable_audio_3.loading_utils as lu
    except ImportError:
        yield {}
        return
    marks = {}
    saved = [(lu, n, getattr(lu, n)) for n in ("create_diffusion_cond_from_config", "load_file", "copy_state_dict",
                                                 "torch") if hasattr(lu, n)]
    saved += [(fac, n, getattr(fac, n)) for n in ("T5GemmaConditioner",) if hasattr(fac, n)]
    wrap = {"create_diffusion_cond_from_config": ("medium model build", "build"), "load_file": (_mb, "read"),
            "copy_state_dict": ("medium weights read + state copy", "copy"),
            "T5GemmaConditioner": ("t5gemma load", "t5")}
    for mod, n, f in saved:
        if n == "torch":
            mod.torch = _StageTorch(f, marks)
        else:
            label, key = wrap[n]
            setattr(mod, n, _timed(f, label, marks, key))
    try:
        yield marks
    finally:
        for mod, n, f in saved:
            setattr(mod, n, f)


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
                 level: str = "rms", decoder_ckpt: str | None = None, amp: bool = False,
                 prompt_style: str = "v1", stereo: bool = False, width: float = 1.0, noise_span=None,
                 hoist_diff: bool = False, weight_cache: str | None = None, **_):
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
        self.amp = bool(amp) and amp != "off"
        self.amp_mode = amp if isinstance(amp, str) else ("full" if amp else "off")
        self.prompt_style = str(prompt_style)
        self.stereo = bool(stereo)
        self.width = float(width)
        self.noise_span = None if noise_span is None else (float(noise_span[0]), float(noise_span[1]))
        self._span_frames = None
        self.hoist_diff = bool(hoist_diff)
        self.weight_cache = weight_cache or None
        self._cache_target = None
        self._cache_writer = None
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
        t = time.time()
        if self.fast_load:
            from ..fastmeta import install
            install()
        import torch
        from stable_audio_3 import StableAudioModel
        t = stage("sa3 imports", t)
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
        model = self._cached_model(torch) if self.weight_cache else None
        if model is None:
            with _skip_random_init() if self.fast_load else contextlib.nullcontext(), _load_stages() as marks:
                model = StableAudioModel.from_pretrained(
                    self.model_name, device=self.device, model_half=self.model_half)
            t = stage("medium fp16" if "cuda" in marks else "medium loaded", marks.get("cuda", t))
            if self._cache_target is not None:
                self._store_cache(model)
        t = time.time()
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
            t = stage("accelerator (triton gemm %d linears)" % len(getattr(self._fast, "triton_linears", ())), t)
        if self._fast is not None and self.hoist_prompts is not None:
            try:
                from .sa3_hoist import ALL_PARTS, DEFAULT_PARTS, install_hoist
                self._hoist = install_hoist(self._model, self._fast, cache_prompts=int(self.hoist_prompts),
                                            parts=ALL_PARTS if self.hoist_rope else DEFAULT_PARTS,
                                            diff=self.hoist_diff)
            except Exception:
                self._hoist = None
            stage("hoist install", t)

    def _cached_model(self, torch):
        import gc
        import json
        from .. import weight_cache as wc
        t = time.time()
        self._cache_target = None
        if not (self.model_half and str(self.device).startswith("cuda") and torch.cuda.is_available()):
            stage("weight cache: off, it needs fp16 on cuda")
            return None
        try:
            import stable_audio_3.loading_utils as lu
            from stable_audio_3 import StableAudioModel
            from stable_audio_3.model_configs import all_models
            config_path, ckpt = all_models[self.model_name].resolve()
            key = wc.key_of(wc.fingerprint(self.model_name, ckpt, config_path))
            path = wc.path_for(self.weight_cache, self.model_name, key)
        except Exception as e:
            stage("weight cache: slow path, no key (%s)" % (repr(e)[:160],), t)
            return None
        gone = wc.sweep(self.weight_cache)
        if gone:
            stage("weight cache: removed %d leftovers of interrupted writes (%s)" % (
                len(gone), ", ".join(os.path.basename(p) for p in gone)[:200]))
        self._cache_target = (path, key)
        if not os.path.isfile(path):
            stage("weight cache: slow path, will build %s" % path, t)
            return None
        inner = model = err = None
        try:
            with open(config_path) as fh:
                config = json.load(fh)
            with _skip_random_init() if self.fast_load else contextlib.nullcontext(), _load_stages():
                inner = lu.create_diffusion_cond_from_config(config)
            inner.eval().requires_grad_(False)
            wc.restore(inner, path, key, self.device)
            inner.use_lora = False
            inner.lora_names = []
            model = StableAudioModel(inner, config, self.device, True)
        except Exception as e:
            err = repr(e)[:200]
        if model is None:
            inner = None
            gc.collect()
            torch.cuda.empty_cache()
            try:
                os.remove(path)
            except OSError as e:
                self._cache_target = None
                err = "%s; kept, no rebuild: %s" % (err, repr(e)[:80])
            stage("weight cache: slow path, cache unusable (%s)" % err, t)
            return None
        self._cache_target = None
        wc.touch(path)
        stage("weight cache: fp16 load from %s" % path, t)
        return model

    def _store_cache(self, model):
        from .. import weight_cache as wc
        path, key = self._cache_target
        self._cache_target = None
        try:
            self._cache_writer = wc.save_async(model.model, path, key, {"model": self.model_name}, self.model_name)
        except Exception as e:
            stage("weight cache: not written, %s" % (repr(e)[:200],))

    def ensure_ready(self):
        self._ensure_loaded()

    @property
    def post_tag(self) -> str:
        if self.level != "loudness" and getattr(self, "amp_mode", "") not in ("punch", "auto"):
            return ""
        return "post:%s:%s" % (self.level, self.amp_mode if self.amp else "off")

    def _prompt(self, key: CacheKey) -> str:
        override = self.prompts.get(str(key))
        if override:
            return override
        if self.prompt_template:
            return self.prompt_template.format(intensity=key.intensity, material=key.material, event=key.event_type)
        if self.prompt_style == "generic":
            from ..prompts import SA3_PREFIX, generic
            return generic(key.event_type, key.material, key.intensity, key.context, key.phys, SA3_PREFIX)
        if self.prompt_style == "detailed":
            from ..prompts import build
            return build(key.event_type, key.material, key.intensity, key.context)
        return build_prompt(key.event_type, key.material, key.intensity)

    def _schedule(self):
        if not self.skip_first_step:
            return contextlib.nullcontext()
        return _skip_first_schedule()

    def _noise(self):
        if self.noise_span is None or self._model is None:
            return contextlib.nullcontext()
        if self._span_frames is None:
            gen, pad = self.noise_span
            ratio = int(self._model.model.pretransform.downsampling_ratio)
            self._span_frames = int(self._model._adapt_sample_size([{"seconds_total": gen}], SA3_SAMPLE_SIZE,
                                                                    pad)) // ratio
        return _noise_span(self._span_frames)

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
        if self.stereo and a.ndim == 2 and a.shape[0] == 2:
            mono = np.ascontiguousarray(a.T)
        else:
            mono = a.mean(axis=0) if a.ndim == 2 else a
        if self.level == "loudness" or getattr(self, "amp_mode", "") == "punch":
            from ..post import process
            return process(mono, self.sample_rate, key.event_type, key.intensity, level=self.level,
                           amp_mode=self.amp_mode if self.amp else "off", stereo=self.stereo, width=self.width,
                           trim=self.trim, duration_s=self.duration_s)
        if self.trim == "natural":
            mono = _natural(mono, self.sample_rate)
        else:
            mono = _extract(mono, self.sample_rate, self.duration_s)
        auto = getattr(self, "amp_mode", "") == "auto"
        if self.amp:
            from ..amp import amplify
            mono = amplify(mono, self.sample_rate, key.event_type, key.intensity, auto=auto)
        if self.stereo:
            from ..stereo import widen
            mono = widen(mono, self.sample_rate, key.event_type, self.width, auto=auto)
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
        bad, m = is_static(mono if mono.ndim == 1 else mono.mean(axis=1), self.sample_rate)
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
            with self._lock, torch.no_grad(), self._schedule(), self._noise():
                audio = self._model.generate(
                    duration=self.gen_duration_s, steps=steps, cfg_scale=self.cfg_scale,
                    seed=retry_seed(base, attempt), batch_size=1, **self._inputs([prompt]), **self._extra())
            cand = self._finish(self._numpy(audio), key)
            bad, m = is_static(cand if cand.ndim == 1 else cand.mean(axis=1), self.sample_rate)
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
        with self._lock, torch.no_grad(), self._schedule(), self._noise():
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
            with self._lock, torch.no_grad(), self._schedule(), self._noise():
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
