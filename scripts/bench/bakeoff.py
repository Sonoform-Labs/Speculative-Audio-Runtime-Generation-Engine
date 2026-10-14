import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import shutil
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(BENCH))
OUT = os.path.join(ROOT, "data", "_gate", "bakeoff")
RAW = os.path.join(OUT, "raw")
LISTEN = os.path.join(OUT, "listen")
ROUTES = os.path.join(OUT, "routes")
TIERS = os.path.join(OUT, "water_tiers")
SR = 44100
INTENSITY = "normal"
CONTEXT = "default"
KEYS = [("shoot", "pistol"), ("shoot", "rifle"), ("break", "window"), ("bullet impact", "glass"),
        ("footstep", "gravel"), ("footstep", "stone"), ("footstep", "wood"), ("footstep", "metal"),
        ("footstep", "water"), ("splash", "water"), ("punch", "body"), ("reload", "rifle")]
WATER_KEYS = [("footstep", "water"), ("splash", "water")]
EXTRA_TYPES = [("bullet_impact", "glass", "glass_break"), ("footstep", "water", "splash"), ("splash", "water", "splash")]
WARM = [("shoot", "rifle", 99), ("splash", "water", 99)]
LISTEN_CAP_S = 2.0
LISTEN_FLOOR_DB = -55.0
LISTEN_ONSET_DB = -30.0
LISTEN_PREROLL_S = 0.02
LISTEN_PEAK_DBFS = -1.0
JUDGE_S = 0.6
SA3_GEN_S = 3.0
SA3_ARMS = {
    "small_prod": {"model": "small-sfx", "production": True, "gen_s": 1.0,
                   "note": "production: run_sidecar TIERS['default'] + fast_load + detailed prompts (euler 3 steps, "
                           "skip-first, cfg 2, compiled, 1.0 s generation, water keys at 8 steps)"},
    "small_g3": {"model": "small-sfx", "gen_s": SA3_GEN_S,
                 "kw": {"sampler_type": "euler", "steps": 3, "detail_steps": 3, "skip_first_step": True,
                        "cfg_scale": 2.0, "duration_padding_s": 1.0},
                 "note": "production sampler (euler 3, skip-first, cfg 2) at 3.0 s generation, eager"},
    "small_rec": {"model": "small-sfx", "gen_s": SA3_GEN_S,
                  "kw": {"sampler_type": "pingpong", "steps": 8, "detail_steps": 8, "skip_first_step": False,
                         "cfg_scale": 1.0, "duration_padding_s": 6.0},
                  "note": "model-card recipe (pingpong 8 steps, cfg 1, padding 6 s) at 3.0 s generation, eager"},
    "medium_rec": {"model": "medium", "gen_s": SA3_GEN_S,
                   "kw": {"sampler_type": "pingpong", "steps": 8, "detail_steps": 8, "skip_first_step": False,
                          "cfg_scale": 1.0, "duration_padding_s": 6.0},
                   "note": "model-card recipe (pingpong 8 steps, cfg 1, padding 6 s) at 3.0 s generation, eager"},
    "medium_e8": {"model": "medium", "gen_s": SA3_GEN_S,
                  "kw": {"sampler_type": "euler", "steps": 8, "detail_steps": 8, "skip_first_step": False,
                         "cfg_scale": 2.0, "duration_padding_s": 1.0},
                  "note": "SARGE sampler family at 8 steps (euler, cfg 2, padding 1 s) at 3.0 s generation, eager"},
    "medium_e3": {"model": "medium", "gen_s": SA3_GEN_S,
                  "kw": {"sampler_type": "euler", "steps": 3, "detail_steps": 3, "skip_first_step": True,
                         "cfg_scale": 2.0, "duration_padding_s": 1.0},
                  "note": "production sampler (euler 3, skip-first, cfg 2) at 3.0 s generation, eager"},
}
MOSS_REPO = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
MOSS_ARMS = {
    "moss_100": {"steps": 100, "cfg": 4.0, "shift": 5.0, "seconds": SA3_GEN_S, "max_s": 30,
                 "note": "model-card recipe (100 flow-match steps, cfg 4, sigma shift 5), 3.0 s cropped from the "
                         "fixed 30 s latent, bf16, TORCHDYNAMO_DISABLE=1"},
    "moss_100_w10": {"steps": 100, "cfg": 4.0, "shift": 5.0, "seconds": SA3_GEN_S, "max_s": 10,
                     "note": "model-card sampler (100 steps, cfg 4, shift 5), 3.0 s cropped from a 10 s latent "
                             "(max_inference_seconds=10 instead of 30), bf16, TORCHDYNAMO_DISABLE=1"},
    "moss_25_w10": {"steps": 25, "cfg": 4.0, "shift": 5.0, "seconds": SA3_GEN_S, "max_s": 10,
                    "note": "25 steps, cfg 4, shift 5, 3.0 s cropped from a 10 s latent, bf16"},
}
WOOSH_HOME = r"F:\AI\cache\woosh"
WOOSH_ARMS = {
    "woosh_flow": {"ckpt": "checkpoints/Woosh-Flow", "kind": "flow", "cfg": 4.5,
                   "note": "Woosh-Flow, dopri5 atol=rtol=1e-3 in float64, cfg 4.5, 501 latent frames (5.0 s)"},
    "woosh_dflow": {"ckpt": "checkpoints/Woosh-DFlow", "kind": "dflow", "cfg": 4.5,
                    "note": "Woosh-DFlow, 4 euler steps, renoise [0, 0.5, 0.5, 0.3], cfg 4.5, 501 latent frames (5.0 s)"},
}
LICENCES = {
    "small-sfx": {"weights": "Stability AI Community License", "commercial": "yes below USD 1M annual revenue, "
                  "Enterprise licence above; T5Gemma text encoder adds the Gemma Terms of Use",
                  "url": "https://stability.ai/license"},
    "medium": {"weights": "Stability AI Community License", "commercial": "yes below USD 1M annual revenue, "
               "Enterprise licence above; T5Gemma text encoder adds the Gemma Terms of Use",
               "url": "https://stability.ai/license"},
    "moss": {"weights": "Apache-2.0 (model card metadata; Qwen3-1.7B text encoder is Apache-2.0)",
             "commercial": "yes, no revenue cap", "url": "https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0"},
    "woosh": {"weights": "CC-BY-NC 4.0 (code MIT)", "commercial": "no: non-commercial only, tested as a quality "
              "reference", "url": "https://github.com/SonyResearch/Woosh"},
}
_PROMPTS = None


def prompts_module():
    global _PROMPTS
    if _PROMPTS is None:
        spec = importlib.util.spec_from_file_location("sarge_prompts_file", os.path.join(ROOT, "src", "sarge", "prompts.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PROMPTS = mod
    return _PROMPTS


def prompt_for(et, mat):
    return prompts_module().build(et, mat, INTENSITY, CONTEXT)


def key_str(et, mat):
    return "%s|%s|%s|%s" % (et, mat, INTENSITY, CONTEXT)


def seed_of(et, mat, v):
    base = int(hashlib.md5(key_str(et, mat).encode()).hexdigest()[:8], 16)
    return (base ^ (v * 2654435761)) & 0x7FFFFFFF


def ev(et):
    return et.replace(" ", "_")


def fname(et, mat, v):
    return "%s-%s_%02d" % (ev(et), mat, v)


def kname(et, mat):
    return "%s|%s" % (ev(et), mat)


def dump(obj, p):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else list(o) if isinstance(o, tuple) else str(o))


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def busy():
    import subprocess
    cmd = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_sidecar|dec_train|dec_eval|"
           "prompt_search' -and $_.Name -match 'python' } | Select-Object -ExpandProperty ProcessId")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60)
        return [x for x in r.stdout.split() if x.strip()]
    except Exception:
        return []


def wait_free(max_min=60):
    t0 = time.time()
    while True:
        b = busy()
        if not b:
            return
        if time.time() - t0 > max_min * 60:
            raise SystemExit("GPU still busy after %d min: %s" % (max_min, b))
        print("waiting for GPU jobs %s" % b, flush=True)
        time.sleep(60)


def sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_peak(torch):
    if not torch.cuda.is_available():
        return {}
    return {"peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30}


def run_arm(torch, arm, family, model, gen, config, note, load_s, load_peak, seeds, keys, warm=2):
    d = os.path.join(RAW, arm)
    os.makedirs(d, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for et, mat, s in WARM[:warm]:
        gen(et, mat, prompt_for(et, mat), s, 99)
    sync(torch)
    warm_s = time.perf_counter() - t0
    prompts, secs, per_key, sr_out, ch_out = {}, [], {}, None, None
    for et, mat in keys:
        p = prompt_for(et, mat)
        prompts[kname(et, mat)] = p
        ks = []
        for v in range(seeds):
            sync(torch)
            t1 = time.perf_counter()
            x, sr = gen(et, mat, p, seed_of(et, mat, v), v)
            sync(torch)
            dt = time.perf_counter() - t1
            x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
            if x.ndim == 1:
                x = x[:, None]
            np.save(os.path.join(d, fname(et, mat, v) + ".npy"), x)
            ks.append(dt)
            secs.append(dt)
            sr_out, ch_out = int(sr), int(x.shape[1])
        per_key[kname(et, mat)] = float(np.median(ks))
        print("%-12s %-20s median %.3f s  (%d seeds)  %s" % (arm, kname(et, mat), per_key[kname(et, mat)], seeds, p),
              flush=True)
    info = {"arm": arm, "family": family, "model": model, "note": note, "config": config, "sr": sr_out,
            "channels": ch_out, "seeds": list(range(seeds)), "keys": [kname(et, mat) for et, mat in keys],
            "prompts": prompts, "intensity": INTENSITY, "context": CONTEXT, "load_s": load_s, "warm_s": warm_s,
            "median_s": float(np.median(secs)), "mean_s": float(np.mean(secs)), "max_s": float(np.max(secs)),
            "median_s_by_key": per_key, "load_peak": load_peak, **gpu_peak(torch),
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    dump(info, os.path.join(d, "gen.json"))
    print("%-12s done: median %.3f s per clip, warm-up %.1f s, load %.1f s, peak %s" % (
        arm, info["median_s"], warm_s, load_s, json.dumps(gpu_peak(torch))), flush=True)
    return info


def selected_keys(args):
    if not args.keys:
        return list(KEYS)
    want = set(args.keys)
    return [(et, mat) for et, mat in KEYS if kname(et, mat) in want or "%s|%s" % (et, mat) in want]


class Tap:
    def __init__(self, g):
        self.orig = g._finish
        self.raw = {}
        g._finish = self

    def __call__(self, a, key):
        out = self.orig(a, key)
        self.raw[id(out)] = (out, np.array(a, dtype=np.float32, copy=True))
        return out

    def take(self, samples):
        hit = self.raw.get(id(samples))
        self.raw.clear()
        if hit is None or hit[0] is not samples:
            raise SystemExit("raw capture lost for a clip")
        a = hit[1]
        while a.ndim > 2:
            a = a[0]
        return a.T if a.ndim == 2 else a


def cmd_sa3(args):
    import torch
    from sarge.generators.stable_audio3 import StableAudio3Generator
    from sarge.types import CacheKey
    arms = args.arms
    models = {SA3_ARMS[a]["model"] for a in arms}
    if len(models) != 1:
        raise SystemExit("one SA3 model per process, got %s" % sorted(models))
    if any(SA3_ARMS[a].get("production") for a in arms) and len(arms) > 1:
        raise SystemExit("the compiled production arm runs alone")
    wait_free()
    model = models.pop()
    first = SA3_ARMS[arms[0]]
    if first.get("production"):
        sys.path.insert(1, os.path.dirname(BENCH))
        import run_sidecar
        kw = dict(run_sidecar.TIERS["default"], fast_load=True, prompt_style="detailed")
    else:
        kw = dict(first["kw"], fast_load=True, prompt_style="detailed")
    kw["gen_duration_s"] = first["gen_s"]
    kw["model_name"] = model
    t0 = time.perf_counter()
    g = StableAudio3Generator(**kw)
    g.ensure_ready()
    sync(torch)
    load_s = time.perf_counter() - t0
    load_peak = gpu_peak(torch)
    print("loaded %s in %.1f s, peak %s" % (model, load_s, json.dumps(load_peak)), flush=True)
    tap = Tap(g)
    keys = selected_keys(args)
    out = {}
    for arm in arms:
        spec = SA3_ARMS[arm]
        if not spec.get("production"):
            for a, v in spec["kw"].items():
                setattr(g, a, v)
        g.gen_duration_s = spec["gen_s"]
        st0 = dict(g.static_stats)

        def gen(et, mat, prompt, seed, v):
            k = CacheKey(et, mat, INTENSITY, CONTEXT)
            if g._prompt(k) != prompt:
                raise SystemExit("prompt mismatch for %s: %s" % (k, g._prompt(k)))
            clip = g.generate(k, variant=v)
            return tap.take(clip.samples), g.sample_rate

        cfg = {a: getattr(g, a) for a in ("model_name", "sampler_type", "steps", "detail_steps", "cfg_scale",
                                          "gen_duration_s", "duration_padding_s", "skip_first_step", "compile_mode",
                                          "compile_decode", "graph_decode", "fp16_accumulation", "triton_gemm",
                                          "decode_prefix", "prompt_style", "static_guard", "model_half", "fast")}
        cfg["detail_keys"] = sorted(g.detail_keys)
        info = run_arm(torch, arm, "sa3", model, gen, cfg, spec["note"], load_s, load_peak, args.seeds, keys)
        info["static_guard"] = {a: int(g.static_stats.get(a, 0)) - int(st0.get(a, 0)) for a in g.static_stats}
        info["fast_stats"] = g.fast_stats()
        dump(info, os.path.join(RAW, arm, "gen.json"))
        out[arm] = info
    return 0


def cmd_moss(args):
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch
    from moss_soundeffect_v2 import MossSoundEffectPipeline
    wait_free()
    t0 = time.perf_counter()
    pipe = MossSoundEffectPipeline.from_pretrained(MOSS_REPO, torch_dtype=torch.bfloat16, device="cuda",
                                                   local_files_only=True)
    dit_dtype = str(next(pipe.engine.dit.parameters()).dtype)
    if args.dit_bf16:
        for t in list(pipe.engine.dit.parameters()) + list(pipe.engine.dit.buffers()):
            if t.is_floating_point():
                t.data = t.data.to(torch.bfloat16)
        torch.cuda.empty_cache()
    sync(torch)
    load_s = time.perf_counter() - t0
    load_peak = gpu_peak(torch)
    load_peak["after_cast_alloc_gb"] = torch.cuda.memory_allocated() / 2 ** 30
    load_peak["dit_dtype_loaded"] = dit_dtype
    load_peak["dit_cast_bf16"] = bool(args.dit_bf16)
    print("loaded MOSS in %.1f s, peak %s, sr %d, max seconds %d" % (load_s, json.dumps(load_peak), pipe.sample_rate,
                                                                    pipe.max_inference_seconds), flush=True)
    keys = selected_keys(args)
    for arm in args.arms:
        spec = MOSS_ARMS[arm]

        def gen(et, mat, prompt, seed, v):
            a = pipe(prompt=prompt, seconds=spec["seconds"], num_inference_steps=spec["steps"], cfg_scale=spec["cfg"],
                     sigma_shift=spec["shift"], seed=int(seed), max_inference_seconds=spec["max_s"],
                     progress_bar_cmd=lambda it: it)
            return a[0].detach().to(torch.float32).cpu().numpy().T, pipe.sample_rate

        cfg = dict(spec, dtype="bfloat16", repo=MOSS_REPO, duration_suffix=True, dit_weights_bf16=bool(args.dit_bf16))
        run_arm(torch, arm, "moss", "moss-soundeffect-v2.0", gen, cfg, spec["note"], load_s, load_peak, args.seeds,
                keys, warm=args.warm)
    return 0


def cmd_woosh(args):
    import torch
    wait_free()
    if len(args.arms) != 1:
        raise SystemExit("one Woosh model per process")
    arm = args.arms[0]
    spec = WOOSH_ARMS[arm]
    os.chdir(args.woosh_home)
    import woosh.module.audioretrieval_module as wam
    from transformers import AutoConfig

    class ConfigOnlyRoberta(wam.RobertaModel):
        @classmethod
        def from_pretrained(cls, name, *a, **kw):
            return wam.RobertaModel(AutoConfig.from_pretrained(name, **kw))

    wam.RobertaModel = ConfigOnlyRoberta
    from woosh.components.base import LoadConfig
    t0 = time.perf_counter()
    if spec["kind"] == "flow":
        from woosh.inference.flowmatching_sampler import flowmatching_integrate
        from woosh.model.ldm import LatentDiffusionModel
        ldm = LatentDiffusionModel(LoadConfig(path=spec["ckpt"]))
    else:
        from woosh.inference.flowmap_sampler import sample_euler
        from woosh.model.flowmap_from_pretrained import FlowMapFromPretrained
        ldm = FlowMapFromPretrained(LoadConfig(path=spec["ckpt"]))
    ldm = ldm.eval().to("cuda")
    sync(torch)
    load_s = time.perf_counter() - t0
    load_peak = gpu_peak(torch)
    print("loaded %s in %.1f s, peak %s" % (arm, load_s, json.dumps(load_peak)), flush=True)
    steps = []

    def gen(et, mat, prompt, seed, v):
        torch.manual_seed(int(seed))
        noise = torch.randn(1, 128, 501).to("cuda")
        cond = ldm.get_cond({"audio": None, "description": [prompt]}, no_dropout=True, device="cuda")
        with torch.inference_mode():
            if spec["kind"] == "flow":
                x, n = flowmatching_integrate(ldm, noise=noise, cond=cond, cfg=spec["cfg"], atol=0.001, rtol=0.001,
                                              return_steps=True, device="cuda", dtype=torch.float64)
                steps.append(int(n) + 1)
            else:
                x = sample_euler(model=ldm, noise=noise, cond=cond, num_steps=4, renoise=[0, 0.5, 0.5, 0.3],
                                 cfg=spec["cfg"])
            audio = ldm.autoencoder.inverse(x)
        return audio[0].detach().to(torch.float32).cpu().numpy().T, 48000

    cfg = dict(spec, latent_frames=501, home=args.woosh_home)
    info = run_arm(torch, arm, "woosh", arm, gen, cfg, spec["note"], load_s, load_peak, args.seeds, selected_keys(args))
    if steps:
        info["solver_steps"] = {"median": float(np.median(steps)), "min": int(min(steps)), "max": int(max(steps))}
        dump(info, os.path.join(RAW, arm, "gen.json"))
    return 0


def resample(x, sr, to=SR):
    if int(sr) == int(to):
        return x
    from scipy.signal import resample_poly
    g = math.gcd(int(sr), int(to))
    return resample_poly(x, int(to) // g, int(sr) // g, axis=0).astype(np.float32)


def listen_cut(x):
    from sarge.generators.stable_audio import _natural
    y = _natural(x, SR, floor_db=LISTEN_FLOOR_DB, onset_db=LISTEN_ONSET_DB, preroll_s=LISTEN_PREROLL_S, fade_s=0.03)
    cap = int(LISTEN_CAP_S * SR)
    capped = y.shape[0] > cap
    if capped:
        y = y[:cap].copy()
        fo = int(0.05 * SR)
        y[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)[:, None]
    pk = float(np.max(np.abs(y)))
    if pk > 1e-9:
        y = y * np.float32(10.0 ** (LISTEN_PEAK_DBFS / 20.0) / pk)
    return y.astype(np.float32), capped


def judge_cut(mono):
    from sarge.generators.stable_audio import _extract
    from sarge.loudness import normalize_loudness, target_for
    n = int(JUDGE_S * SR)
    if mono.shape[0] < n:
        mono = np.pad(mono, (0, n - mono.shape[0]))
    w = _extract(mono, SR, JUDGE_S)
    return normalize_loudness(w, target_rms=target_for(INTENSITY)).astype(np.float32)


def cmd_post(args):
    from sarge.static_guard import is_static
    from sarge.types import AudioClip
    arms = args.arms or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(RAW, "*", "gen.json")))
    for arm in arms:
        info = load(os.path.join(RAW, arm, "gen.json"))
        ld, rd = os.path.join(LISTEN, arm), os.path.join(ROUTES, arm)
        for d in (ld, rd):
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)
        dur, caps, static, clip_peak, side = [], 0, {}, [], []
        mono_src = int(info["channels"]) == 1
        for k in info["keys"]:
            et, mat = k.split("|")
            fl = []
            for v in info["seeds"]:
                x = np.load(os.path.join(RAW, arm, "%s-%s_%02d.npy" % (et, mat, v)))
                clip_peak.append(float(np.max(np.abs(x))))
                if x.shape[1] == 1:
                    x = np.repeat(x, 2, axis=1)
                x = resample(x[:, :2], info["sr"])
                y, capped = listen_cut(x)
                caps += int(capped)
                dur.append(y.shape[0] / SR)
                side.append(float(np.sqrt(np.mean(((y[:, 0] - y[:, 1]) * 0.5) ** 2)) /
                                  max(1e-12, float(np.sqrt(np.mean(((y[:, 0] + y[:, 1]) * 0.5) ** 2))))))
                AudioClip(y, SR).to_wav(os.path.join(ld, "%s-%s_%02d.wav" % (et, mat, v)))
                mono = x.mean(axis=1)
                w = judge_cut(mono)
                if w.shape[0] != int(JUDGE_S * SR):
                    raise SystemExit("%s %s v%d judge window has %d samples" % (arm, k, v, w.shape[0]))
                AudioClip(w, SR).to_wav(os.path.join(rd, "%s-%s_%02d.wav" % (et, mat, v)))
                fl.append(int(is_static(mono, SR)[0]))
            static[k] = {"rate": float(np.mean(fl)), "flags": fl}
        route = {"steps": info["config"].get("steps"), "prompts": info["prompts"], "clusters": [], "sources": [],
                 "audio": "none", "arm": arm, "family": info["family"], "model": info["model"],
                 "note": "%s; judged clips are the mono 0.6 s _extract window at normalize_loudness "
                         "target_for('normal') cut from the same raw output as listen/%s; no real audio" % (info["note"], arm),
                 "config": info["config"], "native_sr": info["sr"], "native_channels": info["channels"],
                 "mono_duplicated": mono_src, "median_ms": info["median_s"] * 1000.0, "median_s": info["median_s"],
                 "warm_s": info["warm_s"], "load_s": info["load_s"],
                 "peak_alloc_gb": info.get("peak_alloc_gb"), "peak_reserved_gb": info.get("peak_reserved_gb"),
                 "load_peak": info.get("load_peak"), "static": info.get("static_guard", {}),
                 "fast_stats": info.get("fast_stats", {}), "static_rule": static,
                 "listen": {"median_s": float(np.median(dur)), "capped": caps, "n": len(dur),
                            "side_to_mid_rms_median": float(np.median(side)), "raw_peak_max": float(np.max(clip_peak))}}
        dump(route, os.path.join(rd, "route.json"))
        print("%-12s %d clips  listen median %.2f s (%d capped at %.1f s)  side/mid %.3f  static %s  raw peak %.2f" % (
            arm, len(dur), float(np.median(dur)), caps, LISTEN_CAP_S, float(np.median(side)) if side else float("nan"),
            " ".join("%s %.2f" % (k, r["rate"]) for k, r in static.items() if r["rate"]) or "none",
            float(np.max(clip_peak))), flush=True)
    return 0


def cmd_water_prep(args):
    if os.path.isdir(TIERS):
        shutil.rmtree(TIERS)
    n = 0
    arms = args.arms or sorted(os.path.basename(d) for d in glob.glob(os.path.join(ROUTES, "*")) if os.path.isdir(d))
    for arm in arms:
        for et, mat in WATER_KEYS:
            d = os.path.join(TIERS, "%s__%s-%s" % (arm, ev(et), mat))
            ps = sorted(glob.glob(os.path.join(ROUTES, arm, "%s-%s_*.wav" % (ev(et), mat))))
            if ps:
                os.makedirs(d, exist_ok=True)
            for p in ps:
                shutil.copy2(p, os.path.join(d, os.path.basename(p)))
                n += 1
    print("copied %d water clips into %s" % (n, TIERS))
    return 0


def cmd_judge(args):
    import vocab_eval as ve
    have = {(et, mat) for et, mat, _t in ve.KEYS}
    for et, mat, t in EXTRA_TYPES:
        if (et, mat) not in have:
            ve.KEYS.append((et, mat, t))
    print("vocab_eval KEYS extended with %s" % [(et, mat, t) for et, mat, t in EXTRA_TYPES], flush=True)
    return ve.main(args.rest)


def ci_fmt(c):
    if not c or c.get("cons") is None:
        return "-"
    lo, hi = c.get("cons_ci") or [float("nan"), float("nan")]
    return "%.2f [%.2f,%.2f]" % (c["cons"], lo, hi)


def num(x):
    return None if x is None or (isinstance(x, float) and x != x) else float(x)


def cmd_collect(args):
    V = load(args.vocab)
    W = load(args.water) if os.path.isfile(args.water) else None
    wsc = {}
    if W is not None:
        from prompt_search import water_scores
        wsc, wchosen = water_scores(W)
    arms = [s for s in V["meta"]["settings"]]
    routes = {a: load(os.path.join(ROUTES, a, "route.json")) for a in arms}
    res = {"meta": {"vocab": args.vocab, "water": args.water if W else None, "refs": V["meta"]["args"].get("manifest"),
                    "seeds": V["meta"]["args"].get("seeds"), "intensity": INTENSITY, "context": CONTEXT,
                    "keys": [kname(et, mat) for et, mat in KEYS],
                    "judge_types_added": [list(x) for x in EXTRA_TYPES],
                    "water_judges": wchosen if W else None,
                    "listen": {"dir": os.path.relpath(LISTEN, ROOT), "cap_s": LISTEN_CAP_S, "floor_db": LISTEN_FLOOR_DB,
                               "onset_db": LISTEN_ONSET_DB, "preroll_s": LISTEN_PREROLL_S,
                               "peak_dbfs": LISTEN_PEAK_DBFS, "format": "stereo 44.1 kHz 16-bit"},
                    "judge_window": "mono 0.6 s _extract + normalize_loudness(target_for('normal'))"},
           "models": {}, "keys": {}}
    for a in arms:
        r = routes[a]
        fam = r["family"]
        lic = LICENCES["moss" if fam == "moss" else "woosh" if fam == "woosh" else r["model"]]
        res["models"][a] = {"family": fam, "model": r["model"], "note": r["note"], "config": r["config"],
                            "native_sr": r["native_sr"], "native_channels": r["native_channels"],
                            "mono_duplicated": r["mono_duplicated"], "load_s": r["load_s"], "warm_s": r["warm_s"],
                            "median_s_per_clip": r["median_s"], "peak_alloc_gb": r["peak_alloc_gb"],
                            "peak_reserved_gb": r["peak_reserved_gb"], "load_peak": r.get("load_peak"),
                            "static_guard": r.get("static"), "listen": r["listen"], "licence": lic}
    for et, mat in KEYS:
        k = kname(et, mat)
        row = {}
        s = V["summary"].get(k)
        for a in arms:
            c = V["consensus"].get("gen:%s|%s" % (a, k), {})
            cell = {}
            if s is not None and a in s["settings"]:
                ss = s["settings"][a]
                cell.update({"cons": num(ss.get("cons")), "cons_ci": ss.get("cons_ci"),
                             "mean_ratio": num(ss.get("mean_ratio")), "judges": c.get("judges") or {},
                             "reason": ss.get("reason"), "shape_in_range": num(ss.get("shape_in_range")),
                             "pq": num(ss.get("pq")), "pq_real": num(ss.get("pq_real")),
                             "material": num(ss.get("material")), "material_judged": ss.get("material_judged"),
                             "type": s.get("type")})
                if ss.get("cons") is None:
                    cell["clap_identity_top"] = V["generated"].get("_identity", {}).get("gen:%s|%s" % (a, k))
            w = wsc.get((a, "%s|%s" % (ev(et), mat)))
            if w:
                cell["water"] = {"cons": num(w["cons"]), "cons_ci": w["cons_ci"], "mean_ratio": num(w["mean_ratio"]),
                                 "static": w["static"], "judges": w["judges"], "interval": w["interval"]}
            cell["static_rule_rate"] = routes[a]["static_rule"].get(k, {}).get("rate")
            row[a] = cell
        res["keys"][k] = row
    dump(res, args.out)
    W_ = 26
    print("%-20s " % "key" + " ".join("%-*s" % (W_, a) for a in arms))
    for k, row in res["keys"].items():
        line = []
        for a in arms:
            c = row[a]
            if "water" in c:
                txt = "w %s" % ci_fmt(c["water"])
            else:
                txt = ci_fmt(c)
            line.append("%-*s" % (W_, txt))
        print("%-20s " % k + " ".join(line))
        line = []
        for a in arms:
            c = row[a]
            line.append("%-*s" % (W_, "shp %s PQ %s/%s" % (
                "%.2f" % c["shape_in_range"] if c.get("shape_in_range") is not None else "-",
                "%.2f" % c["pq"] if c.get("pq") is not None else "-",
                "%.2f" % c["pq_real"] if c.get("pq_real") is not None else "-")))
        print("%-20s " % "" + " ".join(line))
    print("speed / VRAM")
    for a, m in res["models"].items():
        print("  %-12s %-22s %5d Hz x%d  load %6.1f s  median %7.3f s/clip  peak %.2f GB alloc / %.2f GB reserved" % (
            a, m["model"], m["native_sr"], m["native_channels"], m["load_s"], m["median_s_per_clip"],
            m["peak_alloc_gb"] or float("nan"), m["peak_reserved_gb"] or float("nan")))
    print("saved -> %s" % args.out)
    return 0


def cmd_page(args):
    import html
    arms = args.arms or sorted(d for d in os.listdir(LISTEN) if os.path.isdir(os.path.join(LISTEN, d)))
    notes = {}
    for a in arms:
        rp = os.path.join(ROUTES, a, "route.json")
        if os.path.isfile(rp):
            r = load(rp)
            notes[a] = "%s, %s Hz x%d%s, %.2f s per clip" % (
                r["model"], r["native_sr"], r["native_channels"], " (mono, duplicated to both channels)"
                if r["mono_duplicated"] else "", r["median_s"])
    secs = []
    for et, mat in KEYS:
        rows = []
        for a in arms:
            fs = sorted(glob.glob(os.path.join(LISTEN, a, "%s-%s_*.wav" % (ev(et), mat))))
            if not fs:
                continue
            cells = "".join("<audio controls preload='none' src='%s/%s'></audio>" % (a, os.path.basename(f))
                            for f in fs[:args.per])
            rows.append("<tr><th>%s<div class='m'>%s</div></th><td>%s</td></tr>" % (
                html.escape(a), html.escape(notes.get(a, "")), cells))
        secs.append("<h2>%s</h2><p class='m'>%s</p><table>%s</table>" % (
            html.escape("%s | %s" % (et, mat)), html.escape(prompt_for(et, mat)), "".join(rows)))
    page = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Model Bake-off</title><style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mute:#6b6b66;--line:#ddddd8}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ececea;--mute:#9a9a95;--line:#33332f}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:24px 16px;max-width:1200px}
table{border-collapse:collapse;width:100%%}td,th{border-bottom:1px solid var(--line);padding:6px;text-align:left;vertical-align:top}
th{width:230px;font-weight:600}audio{width:170px;height:30px;margin:2px}.m{color:var(--mute);font-weight:400;font-size:13px}
</style></head><body><h1>Model bake-off</h1>
<p class='m'>Headphones. Every clip is the raw model output: stereo 44.1 kHz 16-bit, start trimmed to the onset with 20 ms
of preroll, tail kept until it falls 55 dB below its loudest point or 2.0 s, peak at -1 dBFS. No amplifier, no widening,
no fine-tuned decoder. Same prompt per row, %d seeds shown per model.</p>%s</body></html>""" % (args.per, "".join(secs))
    p = os.path.join(LISTEN, "index.html")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(page)
    print("wrote", p)
    return 0


def parse(argv):
    ap = argparse.ArgumentParser(description="Bigger-model and stereo bake-off for SARGE")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sa3")
    p.add_argument("--arms", nargs="+", required=True, choices=list(SA3_ARMS))
    p = sub.add_parser("moss")
    p.add_argument("--arms", nargs="+", default=["moss_100"], choices=list(MOSS_ARMS))
    p.add_argument("--warm", type=int, default=1)
    p.add_argument("--dit-bf16", action=argparse.BooleanOptionalAction, default=True)
    p = sub.add_parser("woosh")
    p.add_argument("--arms", nargs="+", required=True, choices=list(WOOSH_ARMS))
    p.add_argument("--woosh-home", default=WOOSH_HOME)
    for name in ("sa3", "moss", "woosh"):
        sub.choices[name].add_argument("--seeds", type=int, default=8)
        sub.choices[name].add_argument("--keys", nargs="+", default=None)
    p = sub.add_parser("post")
    p.add_argument("--arms", nargs="+", default=None)
    p = sub.add_parser("water-prep")
    p.add_argument("--arms", nargs="+", default=None)
    p = sub.add_parser("page")
    p.add_argument("--arms", nargs="+", default=None)
    p.add_argument("--per", type=int, default=8)
    p = sub.add_parser("collect")
    p.add_argument("--vocab", default=os.path.join(OUT, "vocab.json"))
    p.add_argument("--water", default=os.path.join(OUT, "water.json"))
    p.add_argument("--out", default=os.path.join(OUT, "results.json"))
    return ap.parse_args(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "judge":
        return cmd_judge(argparse.Namespace(rest=argv[1:]))
    args = parse(argv)
    fn = {"sa3": cmd_sa3, "moss": cmd_moss, "woosh": cmd_woosh, "post": cmd_post, "water-prep": cmd_water_prep,
          "page": cmd_page, "collect": cmd_collect}[args.cmd]
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
