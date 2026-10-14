import argparse
import contextlib
import os
import socket
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from model_server import _recv, _send

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
        gen(prompt, 0, frames=n)
        torch.cuda.synchronize()
        out[n] = {"warm_s": time.perf_counter() - t0, "reserved_mb": (torch.cuda.memory_reserved() - r0) / 2 ** 20}
    return out


def load(kind):
    import torch
    os.chdir(WOOSH_HOME)
    import woosh.module.audioretrieval_module as wam
    from transformers import AutoConfig

    class ConfigOnlyRoberta(wam.RobertaModel):
        @classmethod
        def from_pretrained(cls, name, *a, **kw):
            return wam.RobertaModel(AutoConfig.from_pretrained(name, **kw))

    wam.RobertaModel = ConfigOnlyRoberta
    from woosh.components.base import LoadConfig
    ckpt, cfg = MODELS[kind]
    if kind == "flow":
        from woosh.model.ldm import LatentDiffusionModel
        ldm = LatentDiffusionModel(LoadConfig(path=ckpt))
    else:
        from woosh.model.flowmap_from_pretrained import FlowMapFromPretrained
        ldm = FlowMapFromPretrained(LoadConfig(path=ckpt))
    return torch, ldm.eval().to("cuda"), cfg


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


def graphed(torch, fn, warm=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def _skip(stage):
    return None


class Gen:
    def __init__(self, kind, frames, steps=4, solver="dopri5", cond_cache=0, graph="none", precision="fp32",
                 ae_precision="fp32", batch_cfg=False, time_dtype="float64", mod_cache=False):
        if graph not in GRAPHS or precision not in PRECISIONS or ae_precision not in AE_PRECISIONS or \
                time_dtype not in TIME_DTYPES:
            raise ValueError("bad option: graph %s precision %s ae_precision %s time_dtype %s" % (
                graph, precision, ae_precision, time_dtype))
        torch, ldm, cfg = load(kind)
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
            mc = self._mod_const(st, ren, n) if self.mod_cache else None

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

            g, st["out"] = graphed(torch, body)
            g_ae = None
            if self.graph in ("ae", "full"):
                st["z"] = torch.zeros(1, 128, n, device="cuda")
                g_ae, st["audio"] = graphed(torch, self._ae_fn(st, n))
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
             batch_cfg=False, time_dtype="float64", mod_cache=False):
    return Gen(kind, frames, steps=steps, solver=solver, cond_cache=cond_cache, graph=graph, precision=precision,
               ae_precision=ae_precision, batch_cfg=batch_cfg, time_dtype=time_dtype, mod_cache=mod_cache)


def body_peak(x):
    m = x.mean(axis=1) if x.ndim == 2 else x
    return float(np.abs(m[:max(1, len(m) - int(END_CUT_S * SR))]).max())


def render(gen, prompt, seed, **kw):
    from sarge.static_guard import retry_seed
    raw = gen(prompt, seed, **kw)
    best = body_peak(raw)
    for attempt in range(1, SILENT_RETRIES + 1):
        if best >= SILENT_PEAK:
            break
        alt = gen(prompt, retry_seed(seed, attempt), **kw)
        p = body_peak(alt)
        if p > best:
            raw, best = alt, p
    return raw


def finish(x, event, intensity, amp, stereo):
    from sarge.generators.stable_audio import _natural
    from sarge.loudness import normalize_peak, peak_for
    y = x[:, 0] if x.ndim == 2 and x.shape[1] == 1 else x
    y = y[:max(1, len(y) - int(END_CUT_S * SR))]
    y = _natural(np.ascontiguousarray(y, dtype=np.float32), SR)
    if amp:
        from sarge.amp import amplify
        y = amplify(y, SR, event, intensity)
    if stereo:
        from sarge.stereo import widen
        y = widen(y, SR, event)
    return normalize_peak(y, peak_for(intensity)).astype(np.float32)


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
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--stereo", action="store_true")
    ap.add_argument("--parent-pid", type=int, default=None,
                    help="exit when this process ends (the sidecar passes its pid so a killed sidecar never leaves the "
                         "worker holding the port and its VRAM)")
    args = ap.parse_args()
    if args.parent_pid:
        watch_parent(args.parent_pid)
    from sarge.prompts import build
    from sarge.types import CacheKey, seed_for

    opts = dict(OPTIONS, **({} if args.plain else ADOPTED[args.model]))
    opts.update({k: getattr(args, k) for k in OPTIONS if getattr(args, k) is not None})
    frames = parse_frames(args.frames)
    gen = make_gen(args.model, frames["other"], args.steps, args.solver, **opts)
    lock = threading.Lock()
    t0 = time.perf_counter()
    warm = prepare(gen, frames)
    print("woosh %s ready, frames %s, warm-up %.1f s (%s), %s%s" % (
        args.model, " ".join("%s=%d" % kv for kv in sorted(frames.items())), time.perf_counter() - t0,
        ", ".join("%d: %.1f s +%.0f MB" % (n, w["warm_s"], w["reserved_mb"]) for n, w in sorted(warm.items())),
        " ".join("%s=%s" % kv for kv in sorted(opts.items())),
        ", graph failed: %s" % gen.stats["graph_error"] if gen.stats["graph_error"] else ""), flush=True)

    def serve(conn):
        try:
            while True:
                req, _ = _recv(conn)
                if req.get("type") == "hello":
                    _send(conn, {"type": "hello", "sample_rate": SR, "model": "woosh-" + args.model})
                    continue
                key = CacheKey(req["event_type"], req["material"], req.get("intensity", "normal"),
                               req.get("context", "default"))
                prompt = build(key.event_type, key.material, key.intensity, key.context)
                t = time.perf_counter()
                with lock:
                    raw = render(gen, prompt, seed_for(key, int(req.get("variant", 0))),
                                 frames=frames_for(key.event_type, frames))
                y = finish(raw, key.event_type, key.intensity, args.amp, args.stereo)
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
    print("woosh server listening on %s:%d" % (args.host, srv.getsockname()[1]), flush=True)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=serve, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
