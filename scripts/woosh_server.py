import argparse
import contextlib
import gc
import os
import socket
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

_T_SCRIPT = time.time()

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from model_server import _recv, _send
from sarge import startup
from sarge.prompts import TEMPLATES, WOOSH_PREFIX, check_style, compose
from sarge.types import CacheKey, seed_for

WOOSH_HOME = os.environ.get("WOOSH_HOME", r"F:\AI\cache\woosh")
MODELS = {"flow": ("checkpoints/Woosh-Flow", 4.5), "dflow": ("checkpoints/Woosh-DFlow", 4.5)}
SR = 48000
END_CUT_S = 0.02
SILENT_PEAK = 0.005
SILENT_RETRIES = 2
GRAPHS = ("none", "sampler", "ae", "full")
PRECISIONS = ("fp32", "tf32", "bf16", "fp16")
AE_PRECISIONS = ("fp32", "bf16", "fp16")
TIME_DTYPES = ("float64", "float32")
OPTIONS = {"cond_cache": 0, "graph": "none", "precision": "fp32", "ae_precision": "fp32", "batch_cfg": False,
           "time_dtype": "float64", "mod_cache": False}
ADOPTED = {"dflow": {"cond_cache": 256, "graph": "full", "mod_cache": True, "precision": "bf16",
                     "ae_precision": "bf16"},
           "flow": {"cond_cache": 256, "graph": "full", "precision": "bf16", "batch_cfg": True}}
FRAMES = {"impact": 101, "step": 101, "other": 101}
WARM = "warmup, a soft thud"
UPGRADES = ("off", "flow")
UPGRADE_SOLVERS = {"dopri5": ("dopri5", 4), "rk4:8": ("rk4", 8), "rk4:16": ("rk4", 16), "midpoint:16": ("midpoint", 16),
                   "midpoint:32": ("midpoint", 32)}
SHARED = ("autoencoder", "conditioners")
ABORT_HOLD_S = 0.5
RETRY_S = 0.2
LOAD_IDLE_S = 0.5
LOAD_COMMIT_GB = 1.5
FAIL_ENV = "SARGE_UPGRADE_FAIL"
VERIFIER_DIR = os.path.join(WOOSH_HOME, "checkpoints", "Woosh-CLAP")
VERIFIER_COMMIT_GB = 0.6
PENDING_KEEP = 32
CHOICE_KEEP = 4096
BEST_LINE_EVERY = 10
SCORE_MODES = ("request", "event-min")
VRAM_SAVES = ("share", "encoder")


class Preempted(Exception):
    pass


def parse_frames(spec=None):
    from sarge.amp import FAMILIES
    out = dict(FRAMES)
    if spec is None or str(spec).strip() == "":
        return out
    spec = str(spec)
    if "=" not in spec:
        return {k: int(spec) for k in out}
    for part in spec.split(","):
        k, v = (s.strip() for s in part.split("="))
        if k not in FAMILIES and k != "other":
            raise ValueError("unknown family %r in --frames, known: %s" % (k, ", ".join(sorted(FAMILIES) + ["other"])))
        out[k] = int(v)
    return out


def frames_for(event, frames):
    from sarge.amp import family_for
    return frames.get(family_for(event) or "other", frames["other"])


def prepare(gen, frames, prompt=WARM):
    torch = gen.torch
    out = {}
    for n in sorted(set(frames.values())):
        torch.cuda.synchronize()
        r0, t0 = torch.cuda.memory_reserved(), time.perf_counter()
        t = time.time()
        gen(prompt, 0, frames=n)
        torch.cuda.synchronize()
        out[n] = {"warm_s": time.perf_counter() - t0, "reserved_mb": (torch.cuda.memory_reserved() - r0) / 2 ** 20}
        startup.stage("woosh warm-up %d frames" % n, t)
    return out


def _timed_reads(base):
    orig = base._load_statedict_from_disk

    def read(self, *a, **kw):
        t = time.time()
        out = orig(self, *a, **kw)
        path = getattr(self, "_weights_path", None) or "?"
        try:
            mb = " %.0f MB" % (os.path.getsize(path) / 1e6)
        except OSError:
            mb = ""
        startup.stage("woosh weights %s%s" % (os.path.basename(os.path.dirname(path)), mb), t)
        return out

    base._load_statedict_from_disk = read
    return orig


HOLD = {"on": False}


def priority(below):
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes as w
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = w.HANDLE
        k.SetPriorityClass.argtypes = [w.HANDLE, w.DWORD]
        k.SetPriorityClass.restype = w.BOOL
        return bool(k.SetPriorityClass(k.GetCurrentProcess(), 0x4000 if below else 0x20))
    except Exception:
        return False


def hold(name, torch=None):
    if not HOLD["on"]:
        return
    t = startup.stage("woosh waiting for the %s gate%s" % (
        name, ", cuda initialised %s" % torch.cuda.is_initialized() if torch is not None else ""))
    for line in sys.stdin:
        if line.strip() == name:
            startup.stage("woosh %s gate open" % name, t)
            return
    os._exit(0)


def stream_safetensors(path, own):
    import json
    import struct
    import torch
    kinds = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        head = json.loads(fh.read(n).decode("utf-8"))
        base = 8 + n
        items = sorted(((k, v) for k, v in head.items() if k != "__metadata__"), key=lambda kv: kv[1]["data_offsets"][0])
        if any(k not in own or v["dtype"] not in kinds or list(own[k].shape) != list(v["shape"]) for k, v in items):
            return False
        with torch.no_grad():
            for k, v in items:
                a, b = v["data_offsets"]
                fh.seek(base + a)
                buf = bytearray(fh.read(b - a))
                t = torch.frombuffer(buf, dtype=kinds[v["dtype"]]).reshape(v["shape"]) if b > a else torch.empty(
                    v["shape"], dtype=kinds[v["dtype"]])
                own[k].copy_(t)
                del t, buf
    return True


def commit_free_gb():
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class PI(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                        ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                        ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                        ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                        ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                        ("HandleCount", wintypes.DWORD), ("ProcessCount", wintypes.DWORD),
                        ("ThreadCount", wintypes.DWORD)]

        p = PI()
        p.cb = ctypes.sizeof(PI)
        if not ctypes.WinDLL("psapi").GetPerformanceInfo(ctypes.byref(p), p.cb):
            return None
        return (p.CommitLimit - p.CommitTotal) * p.PageSize / 2 ** 30
    except Exception:
        return None


def release_cache(torch):
    if torch is None or not torch.cuda.is_available():
        return False
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return True


def drop_encoder(gen):
    ae = getattr(getattr(getattr(gen, "ldm", None), "autoencoder", None), "autoencoder", None)
    enc = getattr(ae, "encoder", None)
    if enc is None:
        return 0
    n = sum(t.numel() * t.element_size() for t in list(enc.parameters()) + list(enc.buffers()) if t.is_cuda)
    enc.to("meta")
    release_cache(getattr(gen, "torch", None))
    return n


def parse_saves(spec):
    names = [x.strip() for x in str(spec or "").split(",") if x.strip()]
    bad = [x for x in names if x not in VRAM_SAVES]
    if bad:
        raise ValueError("unknown vram saving %s (choose from %s)" % (", ".join(bad), ", ".join(VRAM_SAVES)))
    return frozenset(names)


def free_vram_mib():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / 2 ** 20
    except Exception:
        return None


def materialize(dit, base_dit, dtype):
    import torch
    if dtype is not None:
        cast_linear(dit, dtype)
    names = {n for n, b in dit.named_buffers() if b is not None}
    theirs = dict(base_dit.named_buffers()) if base_dit is not None else {}
    dit.to_empty(device="cuda")
    for n in names:
        b = dict(dit.named_buffers())[n]
        hit = [v for k, v in theirs.items() if k.split(".")[-1] == n.split(".")[-1] and v is not None and
               v.shape == b.shape and v.dtype == b.dtype]
        if len(hit) != 1:
            raise RuntimeError("no single served buffer to restore %s from" % n)
        with torch.no_grad():
            b.copy_(hit[0])


@contextlib.contextmanager
def reuse_parts(base, dtype=None):
    import torch
    import woosh.model.ldm as wl
    from woosh.components.base import BaseComponent
    ae = getattr(base, "autoencoder", None)
    conds = getattr(base, "conditioners", None)
    pool = ([ae] if ae is not None else []) + (list(conds.values()) if conds is not None else [])
    used = []
    dits = []
    orig = (wl.AudioAutoEncoder, wl.SFXCLAPTextConditioner, BaseComponent.load_from_config)
    orig_dit = wl.SFXFlow

    def on_meta(args):
        with torch.device("meta"):
            d = orig_dit(args)
        dits.append(d)
        return d

    def where(p):
        return os.path.normcase(os.path.abspath(p)) if p else None

    def take(cfg, fallback):
        want = where(getattr(cfg, "path", None))
        for m in pool:
            have = where(os.path.dirname(getattr(m, "_weights_path", None) or "")) if getattr(
                m, "_weights_path", None) else None
            if type(m) is fallback and want and have == want and not any(m is u for u in used):
                used.append(m)
                return m
        return fallback(cfg)

    def lfc(self):
        if any(self is u for u in used):
            return None
        return orig[2](self)

    read = BaseComponent._load_statedict_from_disk

    def stream(self, only_return_state_dict=False, strict=True):
        path = str(getattr(self, "_weights_path", None) or "")
        if only_return_state_dict or not path.endswith(".safetensors"):
            return read(self, only_return_state_dict=only_return_state_dict, strict=strict)
        mine = [d for d in dits if any(d is m for m in self.modules())]
        for d in mine:
            materialize(d, getattr(base, "dit", None), dtype)
        own = self.state_dict(keep_vars=True)
        if not stream_safetensors(path, own):
            if mine:
                raise RuntimeError("the %s checkpoint does not match the model built for it" % path)
            return read(self, only_return_state_dict=only_return_state_dict, strict=strict)
        kept = {id(t) for m in used for t in m.state_dict(keep_vars=True).values() if t is not None}
        if any(t.is_meta for t in own.values() if t is not None and id(t) not in kept):
            raise RuntimeError("weights missing for part of %s" % type(self).__name__)
        return None

    wl.AudioAutoEncoder = lambda cfg: take(cfg, orig[0])
    wl.SFXCLAPTextConditioner = lambda cfg: take(cfg, orig[1])
    wl.SFXFlow = on_meta
    BaseComponent.load_from_config = lfc
    BaseComponent._load_statedict_from_disk = stream
    try:
        yield used
    finally:
        wl.AudioAutoEncoder, wl.SFXCLAPTextConditioner, BaseComponent.load_from_config = orig
        wl.SFXFlow = orig_dit
        BaseComponent._load_statedict_from_disk = read


def share_parts(ldm, base):
    swap = {}
    for name in SHARED:
        mine, theirs = getattr(ldm, name, None), getattr(base, name, None)
        if mine is None or theirs is None or type(mine) is not type(theirs):
            continue
        a, b = mine.state_dict(), theirs.state_dict()
        if a.keys() != b.keys() or any(a[k].shape != b[k].shape for k in a):
            continue
        swap[id(mine)] = theirs
        if hasattr(mine, "items"):
            for k, m in mine.items():
                swap[id(m)] = theirs[k]
        setattr(ldm, name, theirs)
    subs = getattr(ldm, "_subcomponents", None)
    if isinstance(subs, dict):
        for k, v in list(subs.items()):
            if id(v) in swap:
                subs[k] = swap[id(v)]
    return sorted(n for n in SHARED if getattr(ldm, n, None) is getattr(base, n, None))


def load(kind, share=None, dit_dtype=None):
    t = time.time()
    import torch
    t = startup.stage("woosh torch import", t)
    os.chdir(WOOSH_HOME)
    import woosh.module.audioretrieval_module as wam
    from transformers import AutoConfig
    t = startup.stage("woosh package import", t)

    class ConfigOnlyRoberta(wam.RobertaModel):
        @classmethod
        def from_pretrained(cls, name, *a, **kw):
            return wam.RobertaModel(AutoConfig.from_pretrained(name, **kw))

    wam.RobertaModel = ConfigOnlyRoberta
    from woosh.components.base import BaseComponent, LoadConfig
    ckpt, cfg = MODELS[kind]
    if kind == "flow":
        from woosh.model.ldm import LatentDiffusionModel as build
    else:
        from woosh.model.flowmap_from_pretrained import FlowMapFromPretrained as build
    if share is None:
        hold("load", torch)
    t = time.time()
    orig = _timed_reads(BaseComponent)
    try:
        if share is None:
            ldm = build(LoadConfig(path=ckpt))
        else:
            half = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dit_dtype)
            with reuse_parts(share, half) as reused:
                ldm = build(LoadConfig(path=ckpt))
            startup.stage("woosh %s built around %d served parts" % (kind, len(reused)))
    finally:
        BaseComponent._load_statedict_from_disk = orig
    t = startup.stage("woosh model build + weights", t)
    if share is None:
        hold("gpu", torch)
        if HOLD["on"] and priority(False):
            startup.stage("woosh back to normal priority")
    else:
        shared = share_parts(ldm, share)
        half = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dit_dtype)
        if half is not None:
            cast_linear(ldm.dit, half)
        gc.collect()
        startup.stage("woosh %s shares %s with the served model" % (kind, ", ".join(shared) or "nothing"), t)
    t = time.time()
    if any(p.is_meta for p in ldm.parameters()):
        ldm = ldm.eval()._apply(lambda x: x if x.is_meta else x.to("cuda"))
    else:
        ldm = ldm.eval().to("cuda")
    startup.stage("woosh to cuda", t)
    return torch, ldm, cfg


def renoise_for(steps):
    if steps == 4:
        return [0, 0.5, 0.5, 0.3]
    return [0.0] + [float(v) for v in np.linspace(0.5, 0.3, max(1, steps - 1))]


class CondCache:
    def __init__(self, size):
        self.size = max(0, int(size))
        self.items = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, prompt, make):
        hit = self.items.get(prompt) if self.size else None
        if hit is not None:
            self.items.move_to_end(prompt)
            self.hits += 1
            return dict(hit)
        self.misses += 1
        c = make(prompt)
        if self.size:
            self.items[prompt] = {k: v.clone() if hasattr(v, "clone") else v for k, v in c.items()}
            while len(self.items) > self.size:
                self.items.popitem(last=False)
        return c


class EpsTorch:
    def __init__(self, torch, bufs):
        self._torch = torch
        self._bufs = list(bufs)

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def randn_like(self, x, *a, **kw):
        return self._bufs.pop(0)


class Istft:
    def __init__(self, istft, frames):
        import torch
        self.torch = torch
        self.n, self.hop, self.win = int(istft.n_fft), int(istft.hop_length), istft.window
        self.size = self.n + self.hop * (int(frames) - 1)
        env = torch.ops.aten.unfold_backward(self.win.pow(2).expand(1, int(frames), self.n), [1, self.size], 1, self.n,
                                             self.hop)
        self.env = env[:, self.n // 2:self.size - self.n // 2]

    def __call__(self, spec):
        torch = self.torch
        fr = torch.fft.irfft(spec.transpose(1, 2), n=self.n, dim=-1, norm="backward")
        y = torch.ops.aten.unfold_backward(fr * self.win.view(1, 1, self.n), [spec.shape[0], self.size], 1, self.n,
                                           self.hop)
        return y[:, self.n // 2:self.size - self.n // 2] / self.env


def spectrum(ae, z):
    import torch
    dec = ae.autoencoder.decoder
    head = dec.head
    if type(head).__name__ != "ISTFTCircleHead" or head.softclip != "softplus" or head.istft.padding != "center":
        raise ValueError("spectrum graph needs ISTFTCircleHead with softplus and center padding")
    if ae.normalize:
        z = z * ae.z_std[None, :, None] + ae.z_mean[None, :, None]
    y = dec.backbone(z)
    with torch.autocast(device_type="cuda", enabled=False):
        mag, a, b = head.out(y.float()).chunk(3, dim=1)
        mag = torch.nn.functional.softplus(mag)
        p = torch.sqrt((a ** 2 + b ** 2).clamp(1e-8, 10e2))
        a = a / p
        b = b / p
        return mag * (a + 1j * b)


def cast_linear(module, dtype):
    import torch
    n = 0
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            m.to(dtype)
            n += 1
    return n


def no_mask(dit):
    from woosh.model.dit_blocks import MMMAttention
    n = 0
    for m in dit.modules():
        if not isinstance(m, MMMAttention):
            continue
        for mod in m.modality_list:
            if getattr(mod, "mask_key", None) is not None or "precompute" in vars(mod):
                continue
            orig = mod.precompute

            def pre(d, apply_rope=True, _o=orig):
                r = _o(d, apply_rope=apply_rope)
                return r[:5] + (None,) + r[6:]

            mod.precompute = pre
            n += 1
    return n


class ModConst:
    def __init__(self, dit):
        import torch
        self.mods = [m for n, m in dit.named_modules()
                     if n.split(".")[-1] == "mod_proj" and isinstance(m, torch.nn.Linear)]
        self.mode, self.vals, self.i = None, [], 0
        for m in self.mods:
            m.forward = self._wrap(m.forward)

    def _wrap(self, orig):
        def fwd(x):
            if self.mode == "replay":
                y = self.vals[self.i]
                self.i += 1
                return y
            y = orig(x)
            if self.mode == "record":
                self.vals.append(y)
            return y

        return fwd

    def record(self, fn):
        self.mode, self.vals = "record", []
        try:
            fn()
        finally:
            self.mode = None
        return self.vals


def graphed(torch, fn, warm=3, pool=None):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def _skip(stage):
    return None


class Gen:
    def __init__(self, kind, frames, steps=4, solver="dopri5", cond_cache=0, graph="none", precision="fp32",
                 ae_precision="fp32", batch_cfg=False, time_dtype="float64", mod_cache=False, share=None):
        if graph not in GRAPHS or precision not in PRECISIONS or ae_precision not in AE_PRECISIONS or \
                time_dtype not in TIME_DTYPES:
            raise ValueError("bad option: graph %s precision %s ae_precision %s time_dtype %s" % (
                graph, precision, ae_precision, time_dtype))
        torch, ldm, cfg = load(kind) if share is None else load(kind, share=share.ldm, dit_dtype=precision)
        if share is not None and ldm.autoencoder is share.ldm.autoencoder:
            ae_precision = share.ae_precision
        self.share, self.share_pool = share, False
        self.should_stop = None
        t = time.time()
        self.torch, self.ldm, self.cfg = torch, ldm, cfg
        self.kind, self.frames, self.steps, self.solver = kind, int(frames), int(steps), solver
        self.graph, self.precision, self.ae_precision = graph, precision, ae_precision
        self.batch_cfg, self.time_dtype = bool(batch_cfg), time_dtype
        self.mod_cache = bool(mod_cache) and kind == "dflow" and graph != "none"
        self.cache = CondCache(cond_cache)
        half = {"bf16": torch.bfloat16, "fp16": torch.float16}
        self.dtype, self.ae_dtype = half.get(precision), half.get(ae_precision)
        self.stats = {"nfe": [], "graph_error": None, "graphs": 0, "no_mask": 0, "linear_cast": 0, "ae_linear_cast": 0,
                      "mod_const": 0}
        if precision == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
        if self.dtype is not None:
            self.stats["linear_cast"] = cast_linear(ldm.dit, self.dtype)
        if self.ae_dtype is not None:
            self.stats["ae_linear_cast"] = cast_linear(ldm.autoencoder.autoencoder.decoder, self.ae_dtype)
        if self.dtype is not None or self.ae_dtype is not None:
            torch.cuda.empty_cache()
        if kind == "flow":
            from woosh.inference.flowmatching_sampler import flowmatching_integrate
            self.integrate = flowmatching_integrate
        else:
            import woosh.inference.flowmap_sampler as fms
            self.fms = fms
        self.plain = not (self.cache.size or graph != "none" or precision != "fp32" or ae_precision != "fp32" or
                          self.batch_cfg or time_dtype != "float64")
        self.no_cond, self.buckets, self.mod = {}, {}, None
        startup.stage("woosh precision cast", t)

    def capture(self, frames, prompt=WARM):
        out = {}
        for n in sorted(set(frames.values())):
            t = time.perf_counter()
            if self.kind != "flow" or self.graph == "none" or self.plain:
                self(prompt, 0, frames=n)
            else:
                torch = self.torch
                cond = self._cond(prompt)
                torch.manual_seed(0)
                noise = torch.randn(1, 128, n).to("cuda")
                with torch.inference_mode():
                    if n not in self.no_cond:
                        self.no_cond[n] = self.ldm.get_cond({"audio": noise, **cond}, no_dropout=True, device="cuda",
                                                            no_cond=True)
                    if n not in self.buckets:
                        self._capture_flow(cond, n)
                torch.cuda.synchronize()
            out[n] = time.perf_counter() - t
        return out

    def __call__(self, prompt, seed, mark=None, frames=None):
        mark = mark or _skip
        n = self.frames if frames is None else int(frames)
        if self.plain:
            return self._plain(prompt, seed, mark, n)
        if self.kind == "flow":
            return self._flow(prompt, seed, mark, n)
        return self._dflow(prompt, seed, mark, n)

    def _plain(self, prompt, seed, mark, n):
        torch, ldm, cfg = self.torch, self.ldm, self.cfg
        torch.manual_seed(int(seed))
        noise = torch.randn(1, 128, n).to("cuda")
        mark("noise")
        cond = ldm.get_cond({"audio": None, "description": [prompt]}, no_dropout=True, device="cuda")
        mark("cond")
        with torch.inference_mode():
            if self.kind == "flow" and self.solver == "dopri5":
                x, n = self.integrate(ldm, noise=noise, cond=cond, cfg=cfg, atol=0.001, rtol=0.001, return_steps=True,
                                      device="cuda", dtype=torch.float64)
            elif self.kind == "flow":
                x, n = self.integrate(ldm, noise=noise, cond=cond, cfg=cfg, method=self.solver, device="cuda",
                                      return_steps=True, step_size=1.0 / max(1, self.steps))
            else:
                x = self.fms.sample_euler(model=ldm, noise=noise, cond=cond, num_steps=self.steps,
                                          renoise=renoise_for(self.steps), cfg=cfg)
                n = self.steps + 1
            self.stats["nfe"].append(int(n) - 1)
            mark("sampler")
            audio = ldm.autoencoder.inverse(x)
            mark("ae")
        out = audio[0].detach().to(torch.float32).cpu().numpy().T
        mark("d2h")
        return out

    def _cond(self, prompt):
        return self.cache.get(prompt, lambda p: self.ldm.get_cond({"audio": None, "description": [p]}, no_dropout=True,
                                                                  device="cuda"))

    def _amp(self):
        if self.dtype is None:
            return contextlib.nullcontext()
        return self.torch.autocast("cuda", dtype=self.dtype, cache_enabled=False)

    def _ae_amp(self):
        if self.ae_dtype is None:
            return contextlib.nullcontext()
        return self.torch.autocast("cuda", dtype=self.ae_dtype, cache_enabled=False)

    def _ae_fn(self, st, n):
        ist = Istft(self.ldm.autoencoder.autoencoder.decoder.head.istft, n) if self.graph == "full" else None
        st["ist"] = ist

        def fn():
            with self.torch.inference_mode(), self._ae_amp():
                s = spectrum(self.ldm.autoencoder, st["z"])
                return s if ist is None else ist(s)

        return fn

    def _decode(self, x, mark, stage, b=None):
        torch, ldm = self.torch, self.ldm
        with torch.inference_mode():
            if stage == "latent" and b is not None and b["g_ae"] is not None:
                b["st"]["z"].copy_(x)
                b["g_ae"].replay()
                x, stage = b["st"]["audio"], self.graph
            if stage == "latent":
                with self._ae_amp():
                    audio = ldm.autoencoder.inverse(x)
            elif stage == "ae":
                audio = ldm.autoencoder.autoencoder.decoder.head.istft(x).unsqueeze(1)
            else:
                audio = x.unsqueeze(1)
        mark("ae")
        out = audio[0].detach().to(torch.float32).cpu().numpy().T
        mark("d2h")
        return out

    def _capture_dflow(self, cond, n):
        torch, ldm = self.torch, self.ldm
        ren = renoise_for(self.steps)
        try:
            self.stats["no_mask"] += no_mask(ldm.dit)
            st = {"noise": torch.zeros(1, 128, n, device="cuda"),
                  "eps": [torch.zeros(1, 128, n, device="cuda") for r in ren if r > 0],
                  "cond": {k: v.clone() for k, v in cond.items()},
                  "t": torch.linspace(1, 0, self.steps + 1).to("cuda")}
            ae = None if self.graph == "sampler" else self._ae_fn(st, n)
            t = time.time()
            mc = self._mod_const(st, ren, n) if self.mod_cache else None
            if mc is not None:
                t = startup.stage("woosh modulation record", t)

            def body():
                self.fms.torch = EpsTorch(torch, st["eps"])
                if mc is not None:
                    mc.mode, mc.vals, mc.i = "replay", st["mod"], 0
                try:
                    with torch.inference_mode(), self._amp():
                        x = self.fms.sample_euler(model=ldm, noise=st["noise"], cond=dict(st["cond"]),
                                                  num_steps=self.steps, renoise=ren, cfg=self.cfg, t=st["t"])
                finally:
                    self.fms.torch = torch
                    if mc is not None:
                        mc.mode = None
                if ae is None:
                    return x
                st["z"] = x
                return ae()

            g, st["out"] = graphed(torch, body)
            self.buckets[n] = {"g": g, "g_ae": None, "st": st}
            self.stats["graphs"] += 1
            startup.stage("woosh graph warm + capture %d frames" % n, t)
        except Exception as e:
            self.buckets[n] = None
            self.stats["graph_error"] = repr(e)[:600]
            torch.cuda.synchronize()

    def _mod_const(self, st, ren, n):
        torch, ldm = self.torch, self.ldm
        if self.mod is None:
            self.mod = ModConst(ldm.dit)
        mc = self.mod

        def run(fill, scale):
            g = torch.Generator(device="cuda").manual_seed(7)
            eps = [torch.randn(1, 128, n, device="cuda", generator=g) for r in ren if r > 0]
            noise = torch.full((1, 128, n), fill, device="cuda")
            cond = {k: v * scale if v.is_floating_point() else v for k, v in st["cond"].items()}
            self.fms.torch = EpsTorch(torch, eps)
            try:
                with torch.inference_mode(), self._amp():
                    self.fms.sample_euler(model=ldm, noise=noise, cond=cond, num_steps=self.steps, renoise=ren,
                                          cfg=self.cfg, t=st["t"])
            finally:
                self.fms.torch = torch

        a = list(mc.record(lambda: run(0.0, 1.0)))
        b = list(mc.record(lambda: run(0.7, 0.5)))
        if not a or len(a) != len(b) or not all(torch.equal(x, y) for x, y in zip(a, b)):
            raise ValueError("modulation inputs vary between runs, %d vs %d calls" % (len(a), len(b)))
        st["mod"] = a
        self.stats["mod_const"] = len(a)
        return mc

    def _dflow(self, prompt, seed, mark, n):
        torch, ldm = self.torch, self.ldm
        cond = self._cond(prompt)
        mark("cond")
        if self.graph != "none" and n not in self.buckets:
            self._capture_dflow(cond, n)
        torch.manual_seed(int(seed))
        noise = torch.randn(1, 128, n)
        self.stats["nfe"].append(self.steps)
        b = self.buckets.get(n)
        if b is not None:
            st = b["st"]
            st["noise"].copy_(noise)
            for e in st["eps"]:
                e.copy_(torch.randn_like(st["noise"]))
            for k, v in st["cond"].items():
                v.copy_(cond[k])
            mark("noise")
            b["g"].replay()
            mark("sampler")
            return self._decode(st["out"], mark, "latent" if self.graph == "sampler" else self.graph)
        noise = noise.to("cuda")
        mark("noise")
        with torch.inference_mode(), self._amp():
            x = self.fms.sample_euler(model=ldm, noise=noise, cond=cond, num_steps=self.steps,
                                      renoise=renoise_for(self.steps), cfg=self.cfg)
        mark("sampler")
        return self._decode(x, mark, "latent")

    def _v(self, y, t, c, nc):
        torch, ldm = self.torch, self.ldm
        if self.batch_cfg:
            v = ldm._denoise_dict_no_param(torch.cat([y, y]), torch.cat([t, t]),
                                           {k: torch.cat([c[k], nc[k]]) for k in c})["x_hat"]
            res, res_nc = v.chunk(2)
        else:
            res = ldm._denoise_dict_no_param(y, t, c)["x_hat"]
            res_nc = ldm._denoise_dict_no_param(y, t, nc)["x_hat"]
        return res + self.cfg * (res - res_nc)

    def graph_pool(self):
        if not self.share_pool:
            return None
        for b in (getattr(self.share, "buckets", None) or {}).values():
            if b is not None and b.get("g") is not None:
                return b["g"].pool()
        return None

    def _capture_flow(self, cond, n):
        torch = self.torch
        try:
            self.stats["no_mask"] += no_mask(self.ldm.dit)
            st = {"y": torch.zeros(1, 128, n, device="cuda"),
                  "t": torch.full((1,), 0.5, dtype=torch.float64, device="cuda"),
                  "cond": {k: v.clone() for k, v in cond.items()}}
            nc = self.no_cond[n]

            def body():
                with torch.inference_mode(), self._amp():
                    return self._v(st["y"], st["t"], st["cond"], nc)

            pool = self.graph_pool()
            g, st["out"] = graphed(torch, body, pool=pool)
            g_ae = None
            if self.graph in ("ae", "full"):
                st["z"] = torch.zeros(1, 128, n, device="cuda")
                g_ae, st["audio"] = graphed(torch, self._ae_fn(st, n), pool=g.pool() if self.share_pool else None)
            self.buckets[n] = {"g": g, "g_ae": g_ae, "st": st}
            self.stats["graphs"] += 1 + int(g_ae is not None)
        except Exception as e:
            self.buckets[n] = None
            self.stats["graph_error"] = repr(e)[:600]
            torch.cuda.synchronize()

    def _flow(self, prompt, seed, mark, n):
        torch, ldm = self.torch, self.ldm
        from torchdiffeq import odeint
        cond = self._cond(prompt)
        mark("cond")
        torch.manual_seed(int(seed))
        noise = torch.randn(1, 128, n).to("cuda")
        mark("noise")
        with torch.inference_mode():
            if n not in self.no_cond:
                self.no_cond[n] = ldm.get_cond({"audio": noise, **cond}, no_dropout=True, device="cuda", no_cond=True)
            if self.graph != "none" and n not in self.buckets:
                self._capture_flow(cond, n)
            b, nc = self.buckets.get(n), self.no_cond[n]
            g, st = (b["g"], b["st"]) if b is not None else (None, None)
            if g is not None:
                for k, v in st["cond"].items():
                    v.copy_(cond[k])
            calls = [0]

            def f(t, y):
                calls[0] += 1
                if self.should_stop is not None and self.should_stop():
                    raise Preempted()
                if g is not None:
                    st["t"].copy_((1 - t).reshape(1))
                    st["y"].copy_(y)
                    g.replay()
                    return st["out"].clone()
                with self._amp():
                    if self.batch_cfg:
                        return self._v(y, (1 - t).reshape(1).to(torch.float64), cond, nc)
                    return self._v(y, 1 - t, cond, nc)

            t = torch.linspace(0, 1, steps=2, device=noise.device)
            if self.solver == "dopri5":
                dt = torch.float64 if self.time_dtype == "float64" else torch.float32
                x = odeint(f, noise, t, atol=0.001, rtol=0.001, method="dopri5", options={"dtype": dt})[-1]
            else:
                x = odeint(f, noise, t, atol=0.001, rtol=0.001, method=self.solver,
                           options={"step_size": 1.0 / max(1, self.steps)})[-1]
        self.stats["nfe"].append(calls[0])
        mark("sampler")
        return self._decode(x, mark, "latent", b)


def make_gen(kind, frames, steps=4, solver="dopri5", cond_cache=0, graph="none", precision="fp32", ae_precision="fp32",
             batch_cfg=False, time_dtype="float64", mod_cache=False, share=None):
    return Gen(kind, frames, steps=steps, solver=solver, cond_cache=cond_cache, graph=graph, precision=precision,
               ae_precision=ae_precision, batch_cfg=batch_cfg, time_dtype=time_dtype, mod_cache=mod_cache, share=share)


class Verifier:
    def __init__(self, path=None, device="cpu"):
        self.path = path or VERIFIER_DIR
        self.device = device
        self.tower = None
        self.sr = None
        self.texts = {}
        self.ms = []

    def load(self):
        import torch
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from woosh.module import audioretrieval_module as wam
        from woosh.utils.loading import lazy_loading
        cfg = OmegaConf.load(os.path.join(self.path, "config.yaml"))
        arm = wam.AudioRetrievalModel

        class Tower(torch.nn.Module):
            forward_audio_model = arm.forward_audio_model
            _get_audio_embedding = arm._get_audio_embedding
            _batch_audio = arm._batch_audio
            get_audio_embedding_from_data = arm.get_audio_embedding_from_data

        tower = Tower()
        tower.audio_config, tower.normalize = cfg.audio, bool(cfg.get("normalize", True))
        with lazy_loading():
            tower.audio_frontend, size = wam.get_audio_frontend_model(cfg.audio)
        tower.audio_head = wam.get_audio_head_model(cfg.audio, cfg.shared_representation_size, size)
        for name, val in (("sample_rate", cfg.audio.get("sample_rate", 32000)),
                          ("eval_max_sec", cfg.audio.get("eval_max_sec", 60)),
                          ("segment_length", cfg.audio.get("segment_length", 5))):
            tower.register_buffer(name, torch.tensor(int(val), dtype=torch.int64), persistent=False)
        tower.load_state_dict(load_file(os.path.join(self.path, "weights_audio.safetensors")), strict=True)
        self.tower = tower.eval().to(self.device)
        self.sr = int(cfg.audio.get("sample_rate", 32000))
        return self

    def text(self, ldm, text):
        hit = self.texts.get(text)
        if hit is not None:
            return hit
        import torch
        c = next(m for m in ldm.conditioners.values() if hasattr(m, "sentence_head"))
        tok, _ = c.tokenize_text([text])
        dev = next(c.sentence_frontend.parameters()).device
        with torch.inference_mode():
            out = c.sentence_frontend(input_ids=tok["input_ids"].to(dev), attention_mask=tok["attention_mask"].to(dev),
                                      return_dict=True)
            e = c.sentence_head(out["last_hidden_state"][:, 0, :]).float()
            e = torch.nn.functional.normalize(e, p=2, dim=1)[0].cpu()
        self.texts[text] = e
        return e

    def embed(self, y, sr):
        import torch
        from math import gcd
        from scipy.signal import resample_poly
        m = np.asarray(y, np.float64)
        m = m.mean(axis=1) if m.ndim == 2 else m
        g = gcd(int(sr), self.sr)
        m = resample_poly(m, self.sr // g, int(sr) // g).astype(np.float32)
        t = time.perf_counter()
        with torch.inference_mode():
            e = self.tower.get_audio_embedding_from_data(m, device=self.device, use_tensor=True).float()[0].cpu()
        self.ms.append((time.perf_counter() - t) * 1000.0)
        return e

    @staticmethod
    def match(e, temb):
        return min(float((e * x).sum()) for x in (temb if isinstance(temb, tuple) else (temb,)))

    def score(self, y, sr, temb):
        return self.match(self.embed(y, sr), temb)

    def scores(self, y, sr, tembs):
        e = self.embed(y, sr)
        return [self.match(e, t) for t in tembs]


class Upgrade:
    def __init__(self, base, kind, frames, lock, opts, make=None, clock=time.monotonic, commit=None, solver="dopri5",
                 best=1, verifier=None, score="request", vram_free=0, vram=None, saves=(), guard=False, floor=None):
        self.base, self.kind, self.frames, self.lock, self.opts = base, kind, frames, lock, dict(opts)
        self.guard = bool(guard)
        self.floor = None if floor is None else float(floor)
        self.floor_drops = 0
        self.score_lock = threading.Lock()
        self.verified = 0
        self.saves = frozenset(saves or ())
        self.vram_free = max(0, int(vram_free or 0))
        self.vram = vram or free_vram_mib
        self.vram_wait_s = 0.0
        self.solver = solver
        method, steps = UPGRADE_SOLVERS[solver]
        self.make = make or (lambda: make_gen(kind, frames["other"], steps, method, share=base, **self.opts))
        self.clock = clock
        self.commit = commit or commit_free_gb
        self.gen = None
        self.state = "idle"
        self.error = None
        self.waiting = 0
        self.abort_until = 0.0
        self.renders = 0
        self.preempted = 0
        self.failed = 0
        self.last_request = 0.0
        self.load_s = None
        self.mu = threading.Lock()
        self.thread = None
        self.best = max(1, int(best))
        self.verifier = verifier if self.best > 1 else None
        self.verifier_state = "off" if self.verifier is None else "idle"
        self.verifier_error = None
        self.pending = OrderedDict()
        self.candidates = 0
        self.picked = [0] * self.best
        self.score_mode = score
        self.choice = OrderedDict()
        self.pool_picks = {}

    def stop(self):
        return self.waiting > 0 or self.clock() < self.abort_until

    def abort(self):
        self.abort_until = self.clock() + ABORT_HOLD_S

    def enter(self):
        with self.mu:
            self.waiting += 1
            self.last_request = self.clock()

    def leave(self):
        with self.mu:
            self.waiting -= 1

    def start(self, wait=False):
        with self.mu:
            if self.thread is None:
                self.state = "loading"
                self.thread = threading.Thread(target=self._load, name="woosh-upgrade", daemon=True)
                self.thread.start()
        if wait:
            self.thread.join()
        return self.state

    def _load(self):
        t = time.perf_counter()
        try:
            cf = self.commit()
            if cf is not None and cf < LOAD_COMMIT_GB:
                raise MemoryError("%.1f GB of commit left, the load needs %.1f GB" % (cf, LOAD_COMMIT_GB))
            self._wait_vram()
            gen = self.make()
            if "share" in self.saves:
                gen.share_pool = True
            while self.clock() - self.last_request < LOAD_IDLE_S or self.clock() < self.abort_until:
                time.sleep(RETRY_S)
            with self.lock:
                if callable(getattr(gen, "capture", None)):
                    gen.capture(self.frames)
                else:
                    prepare(gen, self.frames)
            if self.verifier is not None:
                self._load_verifier()
            gen.should_stop = self.stop
            self.gen = gen
            self.state = "ready"
        except Exception as e:
            self.error = "%s: %s" % (type(e).__name__, e)
            self.state = "failed"
        self.load_s = time.perf_counter() - t
        print("woosh upgrade %s %s after %.1f s%s%s" % (
            self.kind, self.state, self.load_s, ": " + self.error if self.error else "",
            "" if self.best == 1 else ", best of %d by the verifier: %s%s" % (
                self.best, self.verifier_state, " (%s)" % self.verifier_error if self.verifier_error else "")),
            flush=True)

    def _wait_vram(self):
        if not self.vram_free:
            return
        t = self.clock()
        while True:
            got = self.vram()
            if got is None or got >= self.vram_free:
                self.vram_wait_s = self.clock() - t
                return
            time.sleep(RETRY_S)

    def _load_verifier(self):
        try:
            cf = self.commit()
            if cf is not None and cf < VERIFIER_COMMIT_GB:
                raise MemoryError("%.1f GB of commit left, the verifier needs %.1f GB" % (cf, VERIFIER_COMMIT_GB))
            self.verifier.load()
            self.verifier_state = "ready"
        except Exception as e:
            self.verifier_error = "%s: %s" % (type(e).__name__, e)
            self.verifier_state = "failed"

    def info(self):
        out = {"type": "upgrade", "engine": self.kind, "state": self.state, "error": self.error,
               "renders": self.renders, "preempted": self.preempted, "failed": self.failed, "load_s": self.load_s,
               "solver": self.solver}
        if self.best > 1:
            out.update({"best": self.best, "verifier": self.verifier_state, "verifier_error": self.verifier_error,
                        "candidates": self.candidates, "picked": list(self.picked),
                        "verifier_ms": float(np.median(self.verifier.ms)) if self.verifier and self.verifier.ms
                        else None})
        if self.score_mode != "request":
            out["score"] = self.score_mode
        if self.vram_free:
            out.update({"vram_free_mib": self.vram_free, "vram_wait_s": self.vram_wait_s})
        if self.saves:
            out["vram_save"] = sorted(self.saves)
        if self.choice:
            out.update({"pool_keys": len(self.choice), "pool_picks": dict(self.pool_picks)})
        if self.guard:
            out["pool_guard"] = True
        if self.floor is not None:
            out.update({"pool_floor": self.floor, "pool_floor_drops": self.floor_drops})
        if self.verified:
            out["verified_takes"] = self.verified
        return out

    def score_takes(self, key, ys, sr, style=None):
        from sarge import verify
        if self.verifier is None or self.verifier_state != "ready" or self.gen is None:
            raise RuntimeError("the verifier is %s" % ("off" if self.verifier is None else self.verifier_state
                                                       if self.verifier_state != "ready" else "waiting for Flow"))
        m = verify.style_texts(style) if style else {}
        want = (verify.key_text(key, **m), verify.event_text(key, **m) if self.score_mode == "event-min" else None)
        with self.lock:
            temb = self._emb(want)
        with self.score_lock:
            out = [self.verifier.score(y, sr, temb) for y in ys]
        self.verified += len(ys)
        return out

    def render(self, key, variant, prompts, prompt, amp, stereo, level, pool=None, texts=None):
        if self.gen is None:
            raise RuntimeError("upgrade %s is %s" % (self.kind, self.state))
        if self.best > 1 and self.verifier_state == "ready" and pool:
            return self._render_pool(key, variant, prompts, prompt, amp, stereo, level, pool, texts)
        if self.best > 1 and self.verifier_state == "ready":
            return self._render_best(key, variant, prompts, prompt, amp, stereo, level, texts)
        fail = os.environ.get(FAIL_ENV, "")
        with self.lock:
            try:
                if fail and fail in str(key):
                    raise RuntimeError("%s holds %r, the bench hook fails this render" % (FAIL_ENV, fail))
                raw = render_key(self.gen, key, variant, self.frames, prompts, prompt)
            except Preempted:
                self.preempted += 1
                raise
            except Exception:
                self.failed += 1
                raise
        self.renders += 1
        return finish_key(raw, key, amp, stereo, level=level)

    def _want(self, key, texts, style=None):
        from sarge import verify
        t = texts if isinstance(texts, dict) and not self.guard else {}
        m = verify.guard_texts(style) if self.guard else verify.style_texts(style) if style else {}
        req = t.get("request") or verify.key_text(key, **m)
        if self.score_mode != "event-min":
            return req, None
        return req, t.get("event") or verify.event_text(key, **m)

    def _emb(self, want):
        req, ev = want
        a = self.verifier.text(self.gen.ldm, req)
        return a if ev is None else (a, self.verifier.text(self.gen.ldm, ev))

    def _floor_want(self, key, style):
        from sarge import verify
        return verify.key_text(key, **verify.guard_texts(style)), None

    def _keep(self, tag, got):
        self.pending[tag] = got
        while len(self.pending) > PENDING_KEEP:
            self.pending.popitem(last=False)

    def _say_best(self):
        if self.renders % BEST_LINE_EVERY == 1:
            print("woosh upgrade %s: best of %d, %d takes from %d candidates, picks by candidate %s, verifier %.0f ms "
                  "a take on the cpu%s" % (self.kind, self.best, self.renders, self.candidates, self.picked,
                                          float(np.median(self.verifier.ms)) if self.verifier.ms else -1.0,
                                          ", prompt pool picks %s over %d keys" % (self.pool_picks, len(self.choice))
                                          if self.choice else ""), flush=True)

    def _render_best(self, key, variant, prompts, prompt, amp, stereo, level, texts=None):
        from sarge import verify
        text = prompt_for(key, prompts, prompt)
        tag = (str(key), int(variant), text, self.best, amp, bool(stereo), level)
        got = self.pending.pop(tag, [])
        fail = os.environ.get(FAIL_ENV, "")
        want = self._want(key, texts, prompts)
        try:
            for j in range(len(got), self.best):
                with self.lock:
                    if fail and fail in str(key):
                        raise RuntimeError("%s holds %r, the bench hook fails this render" % (FAIL_ENV, fail))
                    raw = render_key(self.gen, key, verify.candidate_variant(variant, j), self.frames, prompts, prompt)
                    temb = self._emb(want)
                y = finish_key(raw, key, amp, stereo, level=level)
                got.append((self.verifier.score(y, SR, temb), j, y))
                self.candidates += 1
        except Preempted:
            self.preempted += 1
            self._keep(tag, got)
            raise
        except Exception:
            self.failed += 1
            raise
        _s, j, y = verify.pick(got)
        self.picked[j] += 1
        self.renders += 1
        self._say_best()
        return y

    def _render_pool(self, key, variant, prompts, prompt, amp, stereo, level, pool, texts=None):
        from sarge import verify
        base = prompt_for(key, prompts, prompt)
        cands = [base]
        for p in pool:
            if isinstance(p, str) and p.strip() and p not in cands:
                cands.append(p)
        ck = (str(key), tuple(cands))
        if ck in self.choice:
            self.choice.move_to_end(ck)
            return self._render_best(key, variant, prompts, cands[self.choice[ck]], amp, stereo, level, texts)
        tag = ("pool",) + ck + (int(variant), self.best, amp, bool(stereo), level)
        got = self.pending.pop(tag, [])
        fail = os.environ.get(FAIL_ENV, "")
        want = self._want(key, texts, prompts)
        low = self._floor_want(key, prompts) if self.floor is not None else None
        order = [(i, j) for i in range(len(cands)) for j in range(self.best)]
        try:
            for i, j in order[len(got):]:
                with self.lock:
                    if fail and fail in str(key):
                        raise RuntimeError("%s holds %r, the bench hook fails this render" % (FAIL_ENV, fail))
                    raw = render_key(self.gen, key, verify.candidate_variant(variant, j), self.frames, prompts,
                                     cands[i])
                    temb = self._emb(want)
                    lemb = self._emb(low) if low is not None else None
                y = finish_key(raw, key, amp, stereo, level=level)
                if lemb is None:
                    got.append((self.verifier.score(y, SR, temb), i, j, y, None))
                else:
                    s, f = self.verifier.scores(y, SR, (temb, lemb))
                    got.append((s, i, j, y, f))
                self.candidates += 1
        except Preempted:
            self.preempted += 1
            self._keep(tag, got)
            raise
        except Exception:
            self.failed += 1
            raise
        means = [float(np.mean([s for s, i, _j, _y, _f in got if i == k])) for k in range(len(cands))]
        if low is not None:
            lows = [float(np.mean([f for _s, i, _j, _y, f in got if i == k])) for k in range(len(cands))]
            c = verify.choose_floored(means, lows, self.floor)
            self.floor_drops += len(verify.floor_drops(lows, self.floor))
        else:
            c = verify.choose_guarded(means) if self.guard else verify.choose(means)
        self.choice[ck] = c
        while len(self.choice) > CHOICE_KEEP:
            self.choice.popitem(last=False)
        self.pool_picks[c] = self.pool_picks.get(c, 0) + 1
        _s, j, y = verify.pick([(s, j, y) for s, i, j, y, _f in got if i == c])
        self.picked[j] += 1
        self.renders += 1
        self._say_best()
        return y


def body_peak(x, shape=None):
    m = x.mean(axis=1) if x.ndim == 2 else x
    m = m[:max(1, len(m) - int(END_CUT_S * SR))]
    if shape:
        from sarge.generators.stable_audio import _natural
        from sarge.loudness import dc_highpass
        m = _natural(dc_highpass(np.asarray(m, dtype=np.float32), SR).astype(np.float32), SR,
                     floor_db=float(shape.get("floor_db", -55.0)))
        if shape.get("lowpass_hz"):
            from sarge.post import lowpass
            m = lowpass(m, SR, float(shape["lowpass_hz"]))
    return float(np.abs(m).max())


def render(gen, prompt, seed, shape=None, **kw):
    from sarge.static_guard import retry_seed
    raw = gen(prompt, seed, **kw)
    best = body_peak(raw, shape)
    for attempt in range(1, SILENT_RETRIES + 1):
        if best >= SILENT_PEAK:
            break
        alt = gen(prompt, retry_seed(seed, attempt), **kw)
        p = body_peak(alt, shape)
        if p > best:
            raw, best = alt, p
    return raw


def finish(x, event, intensity, amp, stereo, level="peak", shape=None):
    if level != "peak" or shape or amp == "punch":
        from sarge.post import process
        y = x[:, 0] if x.ndim == 2 and x.shape[1] == 1 else x
        y = np.ascontiguousarray(y[:max(1, len(y) - int(END_CUT_S * SR))], dtype=np.float32)
        return process(y, SR, event, intensity, level=level, amp_mode=amp if amp in ("punch", "auto") else (
            "full" if amp else "off"), stereo=bool(stereo), shape=shape)
    from sarge.generators.stable_audio import _natural
    from sarge.loudness import normalize_peak, peak_for
    y = x[:, 0] if x.ndim == 2 and x.shape[1] == 1 else x
    y = y[:max(1, len(y) - int(END_CUT_S * SR))]
    y = _natural(np.ascontiguousarray(y, dtype=np.float32), SR)
    if amp:
        from sarge.amp import amplify
        y = amplify(y, SR, event, intensity, auto=amp == "auto")
    if stereo:
        from sarge.stereo import widen
        y = widen(y, SR, event, auto=amp == "auto")
    return normalize_peak(y, peak_for(intensity)).astype(np.float32)


def physics_shape(phys):
    import inspect
    if not phys or "shape" not in inspect.signature(finish).parameters:
        return {}
    from sarge.world import shape
    return {"shape": shape(phys)}


def prompt_for(key, style="tuned", prompt=None):
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    return compose(style, key, WOOSH_PREFIX)


def upgrade_style(args):
    return getattr(args, "upgrade_prompts", None) or args.prompts


def render_key(gen, key, variant, frames, style="tuned", prompt=None):
    return render(gen, prompt_for(key, style, prompt), seed_for(key, int(variant)),
                  shape=physics_shape(key.phys).get("shape"), frames=frames_for(key.event_type, frames))


def finish_key(raw, key, amp, stereo, level="peak"):
    return finish(raw, key.event_type, key.intensity, amp, stereo, level=level, **physics_shape(key.phys))


def verify_clips(req, pcm):
    a = np.frombuffer(pcm, dtype="<f4")
    out, at = [], 0
    for n, ch in zip(req["frames"], req["channels"]):
        k = int(n) * int(ch)
        if at + k > a.shape[0]:
            raise ValueError("verify request holds %d samples, its frames need more" % a.shape[0])
        y = np.array(a[at:at + k], dtype=np.float32)
        out.append(y.reshape(int(n), int(ch)) if int(ch) > 1 else y)
        at += k
    if at != a.shape[0]:
        raise ValueError("verify request holds %d samples, its frames need %d" % (a.shape[0], at))
    return out


def verify_reply(up, req, pcm):
    t = time.perf_counter()
    try:
        if up is None:
            raise RuntimeError("this worker has no upgrade and no verifier")
        key = CacheKey(req["event_type"], req["material"], req.get("intensity", "normal"),
                       req.get("context", "default"), req.get("phys", ""), "", req.get("scene", ""))
        ys = verify_clips(req, pcm)
        scores = up.score_takes(key, ys, int(req["sample_rate"]), req.get("style"))
    except (ConnectionError, OSError):
        raise
    except Exception as e:
        return {"type": "error", "message": "%s: %s" % (type(e).__name__, e)}
    return {"type": "scores", "scores": scores, "ms": (time.perf_counter() - t) * 1000.0}


def parent_gone(pid):
    if os.name == "nt":
        import ctypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.restype = ctypes.c_void_p
        k.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k.WaitForSingleObject.restype = ctypes.c_uint32
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        h = k.OpenProcess(0x00100000, 0, int(pid))
        if not h:
            return ctypes.get_last_error() == 87
        try:
            k.WaitForSingleObject(h, 0xFFFFFFFF)
        finally:
            k.CloseHandle(h)
        return True
    while True:
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        time.sleep(1.0)


def watch_parent(pid, on_gone=None):
    def run():
        if parent_gone(pid):
            (on_gone or (lambda: os._exit(0)))()

    t = threading.Thread(target=run, name="parent-watch", daemon=True)
    t.start()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="dflow")
    ap.add_argument("--frames", default=None,
                    help="latent frames per request, 100 per second of audio: one count for every sound, or "
                         "family=count pairs over FRAMES (families from sarge.amp, other for the rest); each count "
                         "gets one CUDA graph, all captured before listening")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--solver", default="dopri5")
    ap.add_argument("--plain", action="store_true", help="start from the original eager fp32 path instead of ADOPTED")
    ap.add_argument("--cond-cache", type=int, default=None)
    ap.add_argument("--graph", choices=GRAPHS, default=None)
    ap.add_argument("--precision", choices=PRECISIONS, default=None)
    ap.add_argument("--ae-precision", choices=AE_PRECISIONS, default=None)
    ap.add_argument("--batch-cfg", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--time-dtype", choices=TIME_DTYPES, default=None)
    ap.add_argument("--mod-cache", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--amp", nargs="?", const="full", choices=("full", "punch", "auto"), default=None,
                    help="amplifier stage: --amp or --amp full is attack emphasis plus saturation by sound family, "
                         "--amp punch is the attack emphasis alone, --amp auto sets attack emphasis, saturation and "
                         "stereo width from the rendered clip (sarge.amp.auto_params)")
    ap.add_argument("--prompts", type=check_style, default="tuned",
                    help="tuned: the sarge.prompts.build phrasebook; generic: a prompt made only from the request "
                         "fields (sarge.prompts.generic); template:NAME: a universal sentence template filled from the "
                         "request fields (sarge.prompts.template; %s); a prompt sent with the request wins over all"
                         % ", ".join(TEMPLATES))
    ap.add_argument("--stereo", action="store_true")
    ap.add_argument("--level", choices=("peak", "loudness"), default="peak",
                    help="peak sets each clip's peak per intensity; loudness sets its 400 ms loudness per intensity "
                         "with at most 6 dB of lookahead limiting (sarge.loudness.normalize_level)")
    ap.add_argument("--parent-pid", type=int, default=None,
                    help="exit when this process ends (the sidecar passes its pid so a killed sidecar never leaves the "
                         "worker holding the port and its VRAM)")
    ap.add_argument("--hold", action="store_true",
                    help="import everything, then wait for a 'load' line on stdin before building the model and "
                         "a 'gpu' line before touching the GPU (the sidecar opens them in turn); exit on EOF; the "
                         "process runs at below-normal CPU priority until the gpu gate")
    ap.add_argument("--upgrade", choices=UPGRADES, default="off",
                    help="flow: on an upgrade_load message, load Woosh Flow beside the served model (sharing its "
                         "autoencoder and text encoder) and render requests that carry engine=flow with it; every "
                         "served request preempts a Flow render within one solver step (docs/hybrid.md). Off by "
                         "default")
    ap.add_argument("--upgrade-prompts", type=check_style, default=None,
                    help="--upgrade: the prompt style of the upgrade renders (same choices as --prompts); left out, "
                         "they use --prompts. A prompt sent with the request wins over both")
    ap.add_argument("--upgrade-solver", choices=sorted(UPGRADE_SOLVERS), default="dopri5",
                    help="--upgrade: the ODE solver of the Flow renders, dopri5 (adaptive, the default) or a fixed "
                         "solver with its step count (docs/hybrid.md)")
    ap.add_argument("--upgrade-best", type=int, default=1,
                    help="--upgrade: render N Flow candidates for each upgraded take and keep the one the verifier "
                         "(Woosh-CLAP's audio tower on the CPU, text from the served text encoder) scores closest to "
                         "the request words; 1 (default) renders one take as before (docs/hybrid.md)")
    ap.add_argument("--upgrade-verifier", default=None,
                    help="--upgrade-best: the Woosh-CLAP checkpoint folder (config.yaml and weights_audio.safetensors), "
                         "default %s" % VERIFIER_DIR)
    ap.add_argument("--upgrade-vram-free", type=int, default=0,
                    help="--upgrade: load Flow only once the card reports at least this many MiB free, checking "
                         "every 0.2 s, so the upgrade waits for room on the GPU (the verifier itself runs on the CPU "
                         "and needs none); 0 (default) loads as before")
    ap.add_argument("--upgrade-vram-save", default="",
                    help="--upgrade: generic VRAM savings, a comma list of %s. share captures the Flow graphs into "
                         "the served model's graph pool, which is safe because one lock serialises every render and "
                         "every graph output is read before the lock is released; encoder puts the autoencoder's "
                         "encoder, which generation never runs, on the meta device before the warm-up. Empty "
                         "(default) changes nothing" % ", ".join(VRAM_SAVES))
    ap.add_argument("--upgrade-score", choices=SCORE_MODES, default="request",
                    help="--upgrade-best: request scores a candidate against the request words; event-min takes the "
                         "lower of that and the score against the event words alone, so a take has to sound like the "
                         "event as well as the material. Default request")
    ap.add_argument("--upgrade-pool-floor", type=float, default=None,
                    help="--upgrade-best: with a prompt pool, also score every take against the cleaned request "
                         "words of the key and drop a pool prompt whose mean there falls more than this margin "
                         "below the key's own prompt, then keep the best of the rest by the usual score. Off by "
                         "default")
    ap.add_argument("--upgrade-pool-guard", action="store_true",
                    help="--upgrade-best: score every candidate prompt and every take against the cleaned request "
                         "words of the key (sarge.verify.key_text with clean on) and ignore the texts a request "
                         "carries, so a pool prompt is kept for the key only when its takes score at least as well "
                         "as the key's own prompt against the typed name. Off by default")
    args = ap.parse_args()
    if args.upgrade != "off" and args.upgrade == args.model:
        ap.error("--upgrade %s needs a different --model" % args.upgrade)
    if args.upgrade_prompts and args.upgrade == "off":
        ap.error("--upgrade-prompts needs --upgrade")
    if args.upgrade_solver != "dopri5" and args.upgrade == "off":
        ap.error("--upgrade-solver needs --upgrade")
    if args.upgrade_best < 1:
        ap.error("--upgrade-best takes 1 or more")
    if (args.upgrade_best > 1 or args.upgrade_verifier) and args.upgrade == "off":
        ap.error("--upgrade-best and --upgrade-verifier need --upgrade")
    if args.upgrade_score != "request" and args.upgrade_best < 2:
        ap.error("--upgrade-score needs --upgrade-best 2 or more")
    if args.upgrade_pool_guard and args.upgrade_best < 2:
        ap.error("--upgrade-pool-guard needs --upgrade-best 2 or more")
    if args.upgrade_pool_floor is not None and (args.upgrade_best < 2 or args.upgrade_pool_guard
                                                or not args.upgrade_pool_floor >= 0.0):
        ap.error("--upgrade-pool-floor takes a margin of 0 or more, needs --upgrade-best 2 or more and leaves out "
                 "--upgrade-pool-guard")
    if args.upgrade_vram_free and args.upgrade == "off":
        ap.error("--upgrade-vram-free needs --upgrade")
    if args.upgrade_vram_free < 0:
        ap.error("--upgrade-vram-free takes 0 or more")
    try:
        saves = parse_saves(args.upgrade_vram_save)
    except ValueError as e:
        ap.error(str(e))
    if saves and args.upgrade == "off":
        ap.error("--upgrade-vram-save needs --upgrade")
    startup.stage("woosh python up", at=_T_SCRIPT)
    startup.stage("woosh base imports", _T_SCRIPT)
    if args.parent_pid:
        watch_parent(args.parent_pid)
    opts = dict(OPTIONS, **({} if args.plain else ADOPTED[args.model]))
    opts.update({k: getattr(args, k) for k in OPTIONS if getattr(args, k) is not None})
    frames = parse_frames(args.frames)
    if args.hold:
        HOLD["on"] = True
        if priority(True):
            startup.stage("woosh below normal priority until the gpu gate")
        t = time.time()
        import scipy.fft
        import scipy.signal
        import sarge.amp
        import sarge.generators.stable_audio
        import sarge.loudness
        import sarge.static_guard
        import sarge.stereo
        startup.stage("woosh finishing imports", t)
    gen = make_gen(args.model, frames["other"], args.steps, args.solver, **opts)
    if "encoder" in saves:
        startup.stage("woosh autoencoder encoder off the card, %.0f MB" % (drop_encoder(gen) / 2 ** 20))
    lock = threading.Lock()
    t0 = time.perf_counter()
    warm = prepare(gen, frames)
    print("woosh %s ready, frames %s, warm-up %.1f s (%s), %s%s" % (
        args.model, " ".join("%s=%d" % kv for kv in sorted(frames.items())), time.perf_counter() - t0,
        ", ".join("%d: %.1f s +%.0f MB" % (n, w["warm_s"], w["reserved_mb"]) for n, w in sorted(warm.items())),
        " ".join("%s=%s" % kv for kv in sorted(opts.items())),
        ", graph failed: %s" % gen.stats["graph_error"] if gen.stats["graph_error"] else ""), flush=True)
    up = None
    if args.upgrade != "off":
        up = Upgrade(gen, args.upgrade, frames, lock, dict(OPTIONS, **ADOPTED[args.upgrade]), solver=args.upgrade_solver,
                     best=args.upgrade_best,
                     verifier=Verifier(args.upgrade_verifier) if args.upgrade_best > 1 else None,
                     score=args.upgrade_score, vram_free=args.upgrade_vram_free, saves=saves,
                     guard=args.upgrade_pool_guard, floor=args.upgrade_pool_floor)
        print("woosh upgrade %s on request: it loads after an upgrade_load message, shares the autoencoder and the "
              "text encoder, and yields to every served render; served prompts %s, upgrade prompts %s, solver %s%s"
              % (args.upgrade, args.prompts, upgrade_style(args), args.upgrade_solver,
                 ("" if args.upgrade_best == 1 else ", best of %d by %s" % (args.upgrade_best, up.verifier.path)) +
                 ("" if args.upgrade_score == "request" else ", score %s" % args.upgrade_score) +
                 ("" if not args.upgrade_pool_guard else ", pool guarded by the cleaned request words") +
                 ("" if args.upgrade_pool_floor is None else ", pool floor %g under the key's own prompt on the "
                  "cleaned request words" % args.upgrade_pool_floor) +
                 ("" if not args.upgrade_vram_free else ", loads once %d MiB are free" % args.upgrade_vram_free) +
                 ("" if not saves else ", vram savings %s" % ",".join(sorted(saves)))),
              flush=True)

    def served(key, req):
        up.enter()
        left = False
        try:
            with lock:
                up.leave()
                left = True
                return render_key(gen, key, req.get("variant", 0), frames, args.prompts, req.get("prompt"))
        finally:
            if not left:
                up.leave()

    def control(conn, kind):
        if kind == "upgrade_abort":
            up.abort()
            return
        _send(conn, dict(up.info(), state=up.start()) if kind == "upgrade_load" else up.info())

    def serve(conn):
        try:
            while True:
                req, pcm = _recv(conn)
                if req.get("type") == "hello":
                    _send(conn, {"type": "hello", "sample_rate": SR, "model": "woosh-" + args.model})
                    continue
                if up is not None and req.get("type") in ("upgrade_load", "upgrade_state", "upgrade_abort"):
                    control(conn, req["type"])
                    continue
                if req.get("type") == "verify":
                    _send(conn, verify_reply(up, req, pcm))
                    continue
                try:
                    key = CacheKey(req["event_type"], req["material"], req.get("intensity", "normal"),
                                   req.get("context", "default"), req.get("phys", ""), "", req.get("scene", ""))
                    t = time.perf_counter()
                    if up is not None and req.get("engine"):
                        if req["engine"] != up.kind:
                            raise ValueError("unknown engine %r, this worker upgrades to %s" % (req["engine"], up.kind))
                        y = up.render(key, req.get("variant", 0), upgrade_style(args), req.get("prompt"), args.amp,
                                      args.stereo, args.level, pool=req.get("pool"), texts=req.get("texts"))
                    elif up is not None:
                        y = finish_key(served(key, req), key, args.amp, args.stereo, level=args.level)
                    else:
                        with lock:
                            raw = render_key(gen, key, req.get("variant", 0), frames, args.prompts, req.get("prompt"))
                        y = finish_key(raw, key, args.amp, args.stereo, level=args.level)
                except (ConnectionError, OSError):
                    raise
                except Exception as e:
                    _send(conn, {"type": "error", "message": "%s: %s" % (type(e).__name__, e)})
                    continue
                ch = 1 if y.ndim == 1 else int(y.shape[1])
                _send(conn, {"type": "audio", "sample_rate": SR, "channels": ch, "frames": int(y.shape[0]),
                             "gen_ms": (time.perf_counter() - t) * 1000.0}, np.ascontiguousarray(y).tobytes())
        except (ConnectionError, OSError):
            pass
        finally:
            conn.close()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(8)
    startup.done("woosh listening")
    print("woosh server listening on %s:%d" % (args.host, srv.getsockname()[1]), flush=True)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=serve, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
