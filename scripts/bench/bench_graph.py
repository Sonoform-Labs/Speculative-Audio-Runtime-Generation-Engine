import os
import sys


def find_root():
    for start in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        p = os.path.abspath(start)
        while True:
            if os.path.isfile(os.path.join(p, "pyproject.toml")) and os.path.isdir(os.path.join(p, "src", "sarge")):
                return p
            q = os.path.dirname(p)
            if q == p:
                break
            p = q
    return None


ROOT = find_root()
if ROOT is None:
    sys.exit("run from the repo root or from scripts\\bench inside it, cwd=%s" % os.getcwd())
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "bench"))

import _bootstrap

import argparse
import json
import math
import time

import numpy as np
import torch
from scipy.io import wavfile

from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.generators.stable_audio import _seed
from sarge.types import CacheKey

KEYS = [
    CacheKey("footstep", "wood", "normal"),
    CacheKey("impact", "glass", "normal"),
    CacheKey("footstep", "metal", "normal"),
]

STAGES = [
    ("baseline", dict(notqdm=False, trim=False, cond=False, graph=False)),
    ("notqdm", dict(notqdm=True, trim=False, cond=False, graph=False)),
    ("trim", dict(notqdm=True, trim=True, cond=False, graph=False)),
    ("cond", dict(notqdm=True, trim=True, cond=True, graph=False)),
    ("graph", dict(notqdm=True, trim=True, cond=True, graph=True)),
    ("drift", dict(notqdm=True, trim=False, cond=False, graph=False)),
]


def sync():
    torch.cuda.synchronize()


def wall(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, (time.perf_counter() - t0) * 1000.0


def gpu_ms(fn, reps):
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    fn()
    sync()
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    sync()
    return a.elapsed_time(b) / reps


def med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


class GraphedForward:
    def __init__(self, orig):
        self.orig = orig
        self.pool = torch.cuda.graph_pool_handle()
        self.entries = {}
        self.failed = {}
        self.captured = 0
        self.replays = 0
        self.eager = 0

    def _key(self, x, t, kw):
        parts = [("x", tuple(x.shape), str(x.dtype)), ("t", tuple(t.shape), str(t.dtype))]
        for k in sorted(kw):
            v = kw[k]
            if torch.is_tensor(v):
                parts.append((k, tuple(v.shape), str(v.dtype)))
            elif v is None or isinstance(v, (bool, int, float, str)):
                parts.append((k, v))
            else:
                return None
        return tuple(parts)

    def _capture(self, x, t, kw):
        sx = x.clone()
        st = t.clone()
        skw = {k: v.clone() for k, v in kw.items() if torch.is_tensor(v)}
        consts = {k: v for k, v in kw.items() if not torch.is_tensor(v)}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.orig(sx, st, **skw, **consts)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
            out = self.orig(sx, st, **skw, **consts)
        return {"graph": graph, "x": sx, "t": st, "kw": skw, "consts": consts, "out": out}

    def __call__(self, x, t, **kw):
        kw["use_checkpointing"] = False
        key = self._key(x, t, kw)
        if key is None or kw.get("return_info") or kw.get("exit_layer_ix") is not None or key in self.failed:
            self.eager += 1
            return self.orig(x, t, **kw)
        ent = self.entries.get(key)
        if ent is None:
            try:
                ent = self._capture(x, t, kw)
            except Exception as e:
                self.failed[key] = repr(e)[:400]
                sync()
                self.eager += 1
                return self.orig(x, t, **kw)
            self.entries[key] = ent
            self.captured += 1
        ent["x"].copy_(x)
        ent["t"].copy_(t)
        for k, v in ent["kw"].items():
            v.copy_(kw[k])
        ent["graph"].replay()
        self.replays += 1
        return ent["out"].clone()


class CondCache:
    def __init__(self, cdm, device):
        self.cdm = cdm
        self.device = device
        self.store = {}
        self.hits = 0
        self.misses = 0

    def _one(self, prompt, seconds):
        k = (prompt, float(seconds))
        ent = self.store.get(k)
        if ent is None:
            self.misses += 1
            with torch.inference_mode():
                out = self.cdm.conditioner([{"prompt": prompt, "seconds_total": seconds}], self.device)
            ent = {n: [x.clone() if torch.is_tensor(x) else x for x in v] for n, v in out.items()}
            self.store[k] = ent
        else:
            self.hits += 1
        return ent

    def get(self, prompts, seconds):
        cond = [{"prompt": p, "seconds_total": seconds} for p in prompts]
        parts = [self._one(p, seconds) for p in prompts]
        if len(parts) == 1:
            return cond, dict(parts[0])
        merged = {}
        for n in parts[0]:
            merged[n] = [torch.cat([p[n][i] for p in parts], 0) for i in range(len(parts[0][n]))]
        return cond, merged


class Patches:
    def __init__(self, sam):
        import stable_audio_3.models.dit as dit_mod
        self.dit_mod = dit_mod
        self.orig_has_lora = dit_mod.has_lora
        self.dit = sam.dit.model
        self.orig_forward = self.dit._forward
        self.lora_present = bool(self.orig_has_lora(self.dit))
        self.graph = None

    def trim_on(self):
        if self.lora_present:
            return False
        self.dit_mod.has_lora = lambda model: False
        return True

    def trim_off(self):
        self.dit_mod.has_lora = self.orig_has_lora

    def graph_on(self):
        self.graph = GraphedForward(self.orig_forward)
        self.dit._forward = self.graph
        return self.graph

    def graph_off(self):
        if "_forward" in self.dit.__dict__:
            del self.dit._forward


def call(sam, g, prompts, seed, steps, mode, cache):
    dur = g.gen_duration_s
    kw = dict(duration=dur, steps=steps, cfg_scale=g.cfg_scale, seed=seed,
              sampler_type=g.sampler_type, batch_size=len(prompts))
    if mode["notqdm"]:
        kw["disable_tqdm"] = True
    if mode["trim"]:
        kw["use_checkpointing"] = False
    if mode["cond"]:
        cond, ct = cache.get(prompts, dur)
        kw["conditioning"] = cond
        kw["conditioning_tensors"] = ct
        if g.negative_prompt:
            ncond, nct = cache.get([g.negative_prompt] * len(prompts), dur)
            kw["negative_conditioning"] = ncond
            kw["negative_conditioning_tensors"] = nct
    else:
        kw["prompt"] = prompts[0] if len(prompts) == 1 else list(prompts)
        if g.negative_prompt:
            kw["negative_prompt"] = g.negative_prompt
    with torch.no_grad():
        return sam.generate(**kw)


def compare(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None, None
    return float(np.max(np.abs(a - b))), bool(np.array_equal(a, b))


def save_wavs(root, stage, tag, arr, sr):
    d = os.path.join(root, "graph_" + stage)
    os.makedirs(d, exist_ok=True)
    for r in range(arr.shape[0]):
        name = tag if arr.shape[0] == 1 else "%s_r%d" % (tag, r)
        wavfile.write(os.path.join(d, name + ".wav"), sr, np.ascontiguousarray(arr[r].T.astype(np.float32)))


def env_facts(sam, g, cdm, dit, patches):
    f = {"torch": torch.__version__, "cuda": torch.version.cuda,
         "device": torch.cuda.get_device_name(0),
         "ENABLE_TORCH_COMPILE": os.environ.get("ENABLE_TORCH_COMPILE", "0")}
    try:
        t5 = cdm.conditioner.conditioners["prompt"].model
        f["t5_dtype"] = str(next(t5.parameters()).dtype)
        f["t5_attn"] = str(getattr(t5.config, "_attn_implementation", None))
    except Exception as e:
        f["t5_dtype"] = "unknown %r" % e
    f["dit_class"] = type(dit).__name__
    f["dit_dtype"] = str(next(dit.parameters()).dtype)
    f["dit_depth"] = len(dit.transformer.layers)
    try:
        n = sam._adapt_sample_size([{"prompt": "x", "seconds_total": g.gen_duration_s}], 5292032, 6.0)
        f["latent_len"] = int(n // cdm.pretransform.downsampling_ratio)
    except Exception as e:
        f["latent_len"] = "unknown %r" % e
    f["decode_chunked_flag"] = bool(getattr(cdm.pretransform, "chunked", False))
    f["lora_present"] = patches.lora_present
    f["steps_default"] = g.steps
    f["cfg_scale"] = g.cfg_scale
    f["sampler"] = g.sampler_type
    f["negative_prompt"] = g.negative_prompt
    f["flash_sdp_enabled"] = bool(torch.backends.cuda.flash_sdp_enabled())
    f["mem_efficient_sdp_enabled"] = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
    fa = getattr(torch.backends.cuda, "is_flash_attention_available", None)
    f["flash_attention_available"] = bool(fa()) if fa else None
    return f


def probe_conditioner(cdm, dev, prompt, dur, reps):
    c = [{"prompt": prompt, "seconds_total": dur}]
    with torch.inference_mode():
        cdm.conditioner(c, dev)
        return med([wall(lambda: cdm.conditioner(c, dev))[1] for _ in range(reps)])


def probe_has_lora(patches, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        patches.orig_has_lora(patches.dit)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return med(ts)


def probe_decode(sam, g, cdm, prompt, seed, steps, reps):
    with torch.no_grad():
        lat = sam.generate(prompt=prompt, duration=g.gen_duration_s, steps=steps, cfg_scale=g.cfg_scale,
                           seed=seed, sampler_type=g.sampler_type, disable_tqdm=True, return_latents=True)
    pt = cdm.pretransform
    lat = lat.to(next(pt.parameters()).dtype)
    need = int(math.ceil(g.gen_duration_s * sam.model.sample_rate / pt.downsampling_ratio))
    k = need + 3
    k = min(int(lat.shape[-1]), k + (k % 2))
    n = int(g.gen_duration_s * sam.model.sample_rate)

    def full():
        torch.manual_seed(seed)
        return pt.decode(lat)

    def pre():
        torch.manual_seed(seed)
        return pt.decode(lat[..., :k])

    with torch.inference_mode():
        full()
        pre()
        t_full = med([wall(full)[1] for _ in range(reps)])
        t_pre = med([wall(pre)[1] for _ in range(reps)])
        a = full()[..., :n].float()
        b = pre()[..., :n].float()
        d = float((a - b).abs().max().item())
    return {"latent_len": int(lat.shape[-1]), "prefix_len": k, "decode_full_ms": t_full,
            "decode_prefix_ms": t_pre, "prefix_max_abs_diff_first_1s": d}


def probe_invariants(dit, ent, reps):
    res = {}
    ctx = ent["kw"].get("cross_attn_cond")
    lc = ent["kw"].get("local_add_cond")
    layers = list(dit.transformer.layers)
    with torch.inference_mode():
        if ctx is not None:
            emb = dit.to_cond_embed(ctx)

            def kv():
                for layer in layers:
                    if getattr(layer, "cross_attend", False):
                        layer.cross_attn.to_kv(emb)

            res["cross_kv_gpu_ms_per_step"] = gpu_ms(kv, reps)
            res["cond_embed_gpu_ms_per_step"] = gpu_ms(lambda: dit.to_cond_embed(ctx), reps)
        if lc is not None:
            lt = lc.transpose(1, 2)

            def le():
                for layer in layers:
                    if layer.to_local_embed is not None:
                        layer.to_local_embed(lt)

            res["local_embed_gpu_ms_per_step"] = gpu_ms(le, reps)
    return res


def probe_forward(gf, dit, reps):
    res = {}
    for ent in gf.entries.values():
        rows = int(ent["x"].shape[0])
        c_on = dict(ent["consts"], use_checkpointing=True)
        c_off = dict(ent["consts"], use_checkpointing=False)
        with torch.inference_mode():
            gf.orig(ent["x"], ent["t"], **ent["kw"], **c_on)
            e_on = med([wall(lambda: gf.orig(ent["x"], ent["t"], **ent["kw"], **c_on))[1] for _ in range(reps)])
            e_off = med([wall(lambda: gf.orig(ent["x"], ent["t"], **ent["kw"], **c_off))[1] for _ in range(reps)])
            g_wall = med([wall(ent["graph"].replay)[1] for _ in range(reps)])
            g_gpu = gpu_ms(ent["graph"].replay, reps)
        r = {"eager_ckpt_ms": e_on, "eager_nockpt_ms": e_off, "graph_wall_ms": g_wall, "graph_gpu_ms": g_gpu}
        r.update(probe_invariants(dit, ent, reps))
        res["rows%d" % rows] = r
    return res


def fit(ms_by_steps):
    xs = sorted(ms_by_steps)
    if len(xs) < 2:
        return None
    slope, icpt = np.polyfit(xs, [ms_by_steps[x] for x in xs], 1)
    return {"ms_per_step": float(slope), "fixed_ms": float(icpt)}


def run_batch(b, sam, g, patches, cache, steps_list, args, sr):
    groups = [[KEYS[(i + j) % len(KEYS)] for j in range(b)] for i in range(len(KEYS))]
    ref, prev, out = {}, {}, {}
    cond_failed = False
    for stage, mode in STAGES:
        if stage == "graph" and cond_failed:
            mode = dict(mode, cond=False)
        info = {"mode": mode, "applied": [], "errors": [], "ms": {}, "cmp": {}, "pass": True}
        gf = None
        outs = {}
        hits0 = cache.hits
        try:
            if mode["notqdm"]:
                info["applied"].append("disable_tqdm")
            if mode["trim"]:
                info["applied"].append("use_checkpointing=False")
                if patches.trim_on():
                    info["applied"].append("has_lora_patch")
                else:
                    info["errors"].append("has_lora patch skipped: lora present")
            if mode["graph"]:
                gf = patches.graph_on()
            for steps in steps_list:
                ts = []
                for grp in groups:
                    prompts = [g._prompt(k) for k in grp]
                    seed = _seed(grp[0], 0)
                    tag = "%s_%s_b%d_s%d" % (grp[0].event_type, grp[0].material, b, steps)
                    call(sam, g, prompts, seed, steps, mode, cache)
                    first = None
                    for _ in range(args.reps):
                        o, ms = wall(lambda: call(sam, g, prompts, seed, steps, mode, cache))
                        ts.append(ms)
                        if first is None:
                            first = o.detach().float().cpu().numpy()
                    outs[tag] = first
                    if not args.no_wav:
                        save_wavs(args.wav_root, stage, tag, first, sr)
                info["ms"][str(steps)] = {"median": med(ts), "per_clip": med(ts) / b, "n": len(ts)}
                print("  b%d %-9s steps %d  median %7.1f ms  per clip %7.1f ms" % (b, stage, steps, med(ts), med(ts) / b))
            if mode["cond"]:
                if cache.hits > hits0:
                    info["applied"].append("cond_cache(hits=%d, prompts=%d)" % (cache.hits - hits0, len(cache.store)))
                else:
                    info["errors"].append("cond cache never hit")
            if gf is not None:
                if gf.replays > 0:
                    info["applied"].append("cuda_graph_dit_forward(captured=%d, replays=%d, eager=%d)"
                                           % (gf.captured, gf.replays, gf.eager))
                else:
                    info["errors"].append("cuda graph not applied")
                for k, v in gf.failed.items():
                    info["errors"].append("capture failed: %s" % v)
                if gf.entries and not args.skip_probes:
                    info["forward_probe"] = probe_forward(gf, patches.dit, args.reps)
                info["max_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        except Exception as e:
            info["errors"].append(repr(e)[:600])
            info["pass"] = False
            if stage == "cond":
                cond_failed = True
            print("  stage %s failed: %r" % (stage, e))
        finally:
            patches.graph_off()
            patches.trim_off()
        strict_prev = not (stage in ("cond", "drift") and b > 1)
        strict_ref = b == 1 or stage in ("notqdm", "trim", "drift")
        for tag, a in outs.items():
            d_prev, eq_prev = compare(a, prev.get(tag))
            d_ref, eq_ref = compare(a, ref.get(tag))
            info["cmp"][tag] = {"d_prev": d_prev, "exact_prev": eq_prev, "d_ref": d_ref, "exact_ref": eq_ref}
            if stage != "baseline":
                if strict_prev and (d_prev is None or d_prev > args.tol):
                    info["pass"] = False
                if strict_ref and (d_ref is None or d_ref > args.tol):
                    info["pass"] = False
        if stage == "baseline":
            ref = outs
        if outs:
            prev = outs
        out[stage] = info
    return out


def print_summary(res, tol):
    print()
    print("=" * 118)
    print(" SA3 OVERHEAD BENCH   tol %.1e   (d_prev = max|diff| vs previous stage, d_base = vs baseline)" % tol)
    print("=" * 118)
    for bkey, run in res["runs"].items():
        base = run.get("baseline", {}).get("ms", {})
        for stage, info in run.items():
            cm = info["cmp"].values()
            dp = [c["d_prev"] for c in cm if c["d_prev"] is not None]
            dr = [c["d_ref"] for c in cm if c["d_ref"] is not None]
            ex = all(c["exact_ref"] for c in cm if c["exact_ref"] is not None) if dr else None
            for st, m in info["ms"].items():
                b0 = base.get(st, {}).get("median", float("nan"))
                print("%-4s %-9s s%-2s %7.1f ms  %+7.1f vs base  per clip %7.1f  d_prev %-9s d_base %-9s exact %-5s %s"
                      % (bkey, stage, st, m["median"], m["median"] - b0, m["per_clip"],
                         ("%.2e" % max(dp)) if dp else "-", ("%.2e" % max(dr)) if dr else "-",
                         ex, "PASS" if info["pass"] else "FAIL"))
            print("     applied: %s" % (", ".join(info["applied"]) if info["applied"] else "none"))
            for e in info["errors"]:
                print("     error: %s" % e)
            fp = info.get("forward_probe")
            if fp:
                for rk, r in fp.items():
                    print("     _forward %s: eager ckpt %.2f ms, eager no-ckpt %.2f ms, graph wall %.2f ms, graph gpu %.2f ms"
                          % (rk, r["eager_ckpt_ms"], r["eager_nockpt_ms"], r["graph_wall_ms"], r["graph_gpu_ms"]))
                    extra = {k: v for k, v in r.items() if k.endswith("per_step")}
                    if extra:
                        print("     step-invariant gpu ms per step: %s" % ", ".join("%s %.2f" % kv for kv in extra.items()))
            fits = {int(s): m["median"] for s, m in info["ms"].items()}
            ft = fit(fits)
            if ft:
                print("     fit: %.1f ms per step + %.1f ms fixed" % (ft["ms_per_step"], ft["fixed_ms"]))
    pr = res.get("probes", {})
    if pr:
        print("-" * 118)
        for k, v in pr.items():
            print(" probe %-28s %s" % (k, v))
    print("=" * 118)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="3")
    ap.add_argument("--batch", default="1")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--wav-root", default=os.path.join("data", "_gate"))
    ap.add_argument("--out", default=os.path.join("scripts", "bench", "out_bench_graph.json"))
    ap.add_argument("--skip-probes", action="store_true")
    ap.add_argument("--no-wav", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("cuda not available")
        sys.exit(2)
    print("root %s" % ROOT)

    steps_list = [int(s) for s in args.steps.split(",")]
    batches = [int(b) for b in args.batch.split(",")]

    g = StableAudio3Generator()
    g.ensure_ready()
    sam = g._model
    cdm = sam.model
    dit = sam.dit.model
    dev = str(sam.device)
    sr = int(g.sample_rate)
    patches = Patches(sam)
    cache = CondCache(cdm, dev)

    facts = env_facts(sam, g, cdm, dit, patches)
    for k, v in facts.items():
        print("%-28s %s" % (k, v))

    warm = dict(notqdm=True, trim=False, cond=False, graph=False)
    for k in KEYS[:2]:
        call(sam, g, [g._prompt(k)], _seed(k, 0), g.steps, warm, cache)

    res = {"facts": facts, "args": vars(args), "runs": {}, "probes": {}}
    for b in batches:
        print("batch %d" % b)
        res["runs"]["b%d" % b] = run_batch(b, sam, g, patches, cache, steps_list, args, sr)

    if not args.skip_probes:
        p = g._prompt(KEYS[0])
        s = _seed(KEYS[0], 0)
        try:
            res["probes"]["conditioner_ms"] = probe_conditioner(cdm, dev, p, g.gen_duration_s, args.reps)
        except Exception as e:
            res["probes"]["conditioner_ms"] = repr(e)[:300]
        try:
            res["probes"]["has_lora_walk_ms"] = probe_has_lora(patches, 20)
        except Exception as e:
            res["probes"]["has_lora_walk_ms"] = repr(e)[:300]
        try:
            res["probes"]["decode"] = probe_decode(sam, g, cdm, p, s, g.steps, args.reps)
        except Exception as e:
            res["probes"]["decode"] = repr(e)[:300]

    print_summary(res, args.tol)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print("saved -> %s" % args.out)
    ok = all(info["pass"] for run in res["runs"].values() for info in run.values())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
