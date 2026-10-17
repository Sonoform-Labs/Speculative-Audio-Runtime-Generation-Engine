import _bootstrap

import argparse
import atexit
import importlib.metadata
import os
import subprocess
import threading
import time

_T_SCRIPT = time.time()
from sarge import startup

startup.root()
startup.stage("python up", at=_T_SCRIPT)
_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
try:
    _TORCH = importlib.metadata.version("torch").split("+")[0]
except importlib.metadata.PackageNotFoundError:
    _TORCH = "none"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_CACHE, "inductor" if _TORCH.startswith("2.6") else "inductor-" + _TORCH))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_CACHE, "triton-" + _TORCH))

_T = startup.stage("sarge package import", _T_SCRIPT)
from sarge import prefetch, weight_cache
from sarge.sidecar import serve

startup.stage("sarge.sidecar import", _T)

COMPILED = {"compile_mode": "default", "compile_decode": True, "graph_decode": True, "hoist_rope": True,
            "fp16_accumulation": True, "graph_text": True, "unfuse_bias": True, "triton_gemm": True,
            "decode_prefix": True}
PRESENT = {"trim": "natural", "level": "peak"}
TIERS = {
    "quality": {"sampler_type": "euler", "steps": 3, "skip_first_step": True, **COMPILED, **PRESENT},
    "default": {"sampler_type": "euler", "steps": 3, "skip_first_step": True, **COMPILED, **PRESENT},
    "fast":    {"sampler_type": "euler", "steps": 3, "skip_first_step": True, "cfg_interval": (0.9, 1.0), **COMPILED,
                **PRESENT},
}
HYBRID_GUNS = {"model_name": "medium", "sampler_type": "pingpong", "steps": 8, "detail_steps": 8, "skip_first_step": False,
               "cfg_scale": 1.0, "duration_padding_s": 1.0, "noise_span": (3.0, 6.0), "gen_duration_s": 3.0,
               "prompt_style": "detailed", "hoist_diff": True}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIER_COST_MS = {"quality": 28.0, "default": 28.0, "fast": 25.0}
WOOSH_COST_MS = {"dflow": 50.0, "flow": 1010.0}


def cache_dir_of(args):
    return weight_cache.default_dir({"SARGE_WEIGHT_CACHE": args.weight_cache})


def check_weights(model_name):
    t = time.time()
    try:
        import torch
    except Exception:
        pass
    t = startup.stage("torch import", t)
    try:
        from huggingface_hub import constants, try_to_load_from_cache
        from stable_audio_3.model_configs import models
    except Exception:
        return
    t = startup.stage("stable_audio_3 import", t)
    cfg = models.get(model_name)
    if cfg is None or not constants.HF_HUB_OFFLINE:
        return
    found = all(isinstance(try_to_load_from_cache(cfg.repo_id, f), str) for f in (cfg.config_path, cfg.ckpt_path))
    startup.stage("weights check", t)
    if not found:
        raise SystemExit("%s weights (%s) are not in the offline HF cache %s; point HF_HOME or HF_HUB_CACHE at the "
                         "cache that holds them" % (model_name, cfg.repo_id, constants.HF_HUB_CACHE))


def early_woosh(env=None) -> bool:
    return (env if env is not None else os.environ).get("SARGE_WOOSH_EARLY", "1").strip().lower() not in (
        "0", "off", "no")


def start_woosh(args, hold=False):
    if not os.path.isfile(args.woosh_python):
        raise SystemExit("woosh python not found: %s" % args.woosh_python)
    pin = os.path.join(os.path.dirname(REPO), "sarge-research", "modal", "runpinned.py")
    cmd = [args.woosh_python, "-u"] + ([pin] if os.path.isfile(pin) else []) + [
        os.path.join(REPO, "scripts", "woosh_server.py"), "--model", args.woosh_model,
        "--port", str(args.woosh_port), "--parent-pid", str(os.getpid())] + (
        ["--frames", str(args.woosh_frames)] if args.woosh_frames else []) + (["--amp"] if args.amp else []) + (
        ["punch"] if args.amp == "punch" else []) + (
        ["--level", "loudness"] if getattr(args, "level", None) == "loudness" else []) + (
        ["--stereo"] if args.stereo else [])
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    env.pop("TRANSFORMERS_CACHE", None)
    env.update(HF_HUB_CACHE=args.woosh_hf_cache, HF_HUB_OFFLINE="1", SARGE_T0=repr(startup.t0()))
    startup.stage("woosh launch")
    proc = subprocess.Popen(cmd + (["--hold"] if hold else []), cwd=REPO, env=env,
                            stdin=subprocess.PIPE if hold else None, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    atexit.register(lambda: proc.poll() is None and proc.kill())
    if hold:
        return proc
    t0 = time.time()
    for line in proc.stdout:
        print("[woosh] " + line.rstrip(), flush=True)
        if "listening on" in line:
            threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
            print("woosh %s up in %.1f s" % (args.woosh_model, time.time() - t0), flush=True)
            startup.stage("woosh up", t0)
            return proc
    raise SystemExit("woosh server exited before listening (code %s)" % proc.poll())


def launch_woosh(args):
    proc = start_woosh(args, hold=True)
    up, state = threading.Event(), {"listening": False}

    def pump():
        for line in proc.stdout:
            if up.is_set():
                continue
            print("[woosh] " + line.rstrip(), flush=True)
            if "listening on" in line:
                state["listening"] = True
                up.set()
        up.set()

    threading.Thread(target=pump, name="woosh-out", daemon=True).start()
    return proc, up, state


def gate(proc, name):
    try:
        proc.stdin.write(name + "\n")
        proc.stdin.flush()
    except (OSError, ValueError):
        pass


def close_woosh(proc):
    try:
        proc.stdin.close()
    except (AttributeError, OSError, ValueError):
        pass
    startup.stage("woosh worker released, the guns did not get ready")


def open_woosh(args, proc, up, state):
    if proc.stdin is None or proc.stdin.closed:
        return start_woosh(args)
    t0 = time.time()
    gate(proc, "gpu")
    up.wait()
    if not state["listening"]:
        raise SystemExit("woosh server exited before listening (code %s)" % proc.poll())
    print("woosh %s up in %.1f s" % (args.woosh_model, time.time() - t0), flush=True)
    startup.stage("woosh up", t0)
    return proc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the predictive-SFX sidecar audio service.")
    ap.add_argument("--generator", default="sa3")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--pcm", default="f32le", choices=["f32le", "i16le"])
    ap.add_argument("--vram-kb", type=float, default=None,
                    help="clip cache budget in KB; 50000, or 256000 with --world (physics buckets multiply the keys)")
    ap.add_argument("--gpu-ms-budget", type=float, default=10_000_000.0)
    ap.add_argument("--gen-cost-ms", type=float, default=None)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--target-variants", type=int, default=6)
    ap.add_argument("--tier", default=None, choices=["quality", "default", "fast"],
                    help="default=euler@3, quality=euler@3 (4 to 8 steps measured worse on the clap gate with the current prompt), "
                         "fast=default with cfg on the first remaining step only (27 ms; better gunshots, clicks and stone, worse dogs). "
                         "Falls back to SARGE_GEN_TIER env var then 'default'. --steps/--sampler override it.")
    ap.add_argument("--steps", type=int, default=None, help="override the tier's step count")
    ap.add_argument("--sampler", default=None, help="override the tier's sampler (euler; rk4 DISABLED)")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--cache-dir", default=os.path.join(".cache", "sfx_bake"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-deepen", action="store_true")
    ap.add_argument("--deepen-batch", type=int, default=1,
                    help="clips per background generation; with the cuda graph a pair costs the same per clip as one "
                         "and holds the gpu twice as long, so 1 is the default")
    ap.add_argument("--learn", action="store_true",
                    help="use the runtime affordance learner + router (auto-derive affordances, no hand-authored tables)")
    ap.add_argument("--no-compile", action="store_true",
                    help="skip torch.compile and the decoder graph; starts about 30 s faster, each clip about 20 ms slower")
    ap.add_argument("--woosh-model", choices=["dflow", "flow"], default="dflow",
                    help="--generator hybrid: the Woosh model for everything except guns (dflow 4 steps, flow slower and finer)")
    ap.add_argument("--woosh-frames", default=None,
                    help="--generator hybrid: Woosh latent frames per request, 100 per second of audio (101 = 1.0 s): "
                         "one count for every sound, or family=count pairs from sarge.amp (impact, step, other for "
                         "the rest), e.g. impact=501. Each distinct count is one CUDA graph captured before READY, "
                         "about 0.5 s and 60 to 80 MB each. Default: woosh_server.FRAMES, 101 for every family, since "
                         "no count up to 501 brought any family to 5%% or fewer clips still sounding at the cut "
                         "(docs/hybrid.md)")
    ap.add_argument("--woosh-port", type=int, default=8771)
    ap.add_argument("--woosh-hf-cache", default=r"D:\hf_cache",
                    help="--generator hybrid: the HF cache that holds roberta-large for the Woosh text encoder")
    ap.add_argument("--woosh-python", default=os.path.join(REPO, ".venv_woosh", "Scripts", "python.exe"))
    ap.add_argument("--weight-cache", default=weight_cache.default_dir() or "off",
                    help="--generator hybrid: folder for the fp16 copy of the SA3 medium weights, read at startup "
                         "in place of the 9.2 GB fp32 checkpoint (built by the first run, keyed on the checkpoint, "
                         "config, torch, stable_audio_3 and loader code, checked tensor by tensor against checksums "
                         "on every load; off disables). Default SARGE_WEIGHT_CACHE, else " + weight_cache.DEFAULT_DIR
                         + " when its parent folder exists, else off")
    ap.add_argument("--stereo", action="store_true",
                    help="send stereo clips: the model's two channels plus a decorrelated width stage by sound type "
                         "(bass stays centred); Unity builds two-channel clips from the reply metadata")
    ap.add_argument("--detailed-prompts", action="store_true",
                    help="the searched prompt grammar: detailed source, action and acoustic wording plus the space from "
                         "the request context (outdoor_open, indoor_small, indoor_large, tunnel; default is outdoor_open)")
    ap.add_argument("--amp", nargs="?", const="full", choices=["full", "punch"], default=None,
                    help="amplifier stage after generation: attack emphasis and saturation by sound type, strongest on "
                         "gunshots (about +4 to 5 dB of density at the same peak), light on footsteps; --amp punch keeps "
                         "the attack emphasis and drops the saturation")
    ap.add_argument("--level", choices=["peak", "loudness"], default=None,
                    help="loudness: level every clip by its 400 ms loudness (hard -14, normal -17, soft -23 LUFS) with "
                         "a 30 Hz high-pass and at most 6 dB of lookahead limiting, both routes; peak sets the peak per "
                         "intensity on both routes; left out, guns keep the tier's leveller (peak, or rms with "
                         "--short-clips) and woosh sets the peak")
    ap.add_argument("--world", action="store_true",
                    help="physics context: the server buckets the phys block Unity sends into the key and the prompt")
    ap.add_argument("--event-log", default=None, help="write one JSON line per request (timings, served or generated)")
    ap.add_argument("--predict-fixes", action="store_true",
                    help="prediction fixes: learned event aliases, last fired intensity, no lead gate in realtime mode")
    ap.add_argument("--pool-fix", action="store_true", help="plan every variant of a demanded key before moving on")
    ap.add_argument("--concurrent-gen", action=argparse.BooleanOptionalAction, default=True,
                    help="on by default: a realtime miss generates outside the engine lock so cached keys answer "
                         "meanwhile, and no background render starts until the miss lands; served clips are the "
                         "same bits either way (docs/on-time.md); --no-concurrent-gen brings back the single-lock path")
    ap.add_argument("--async-cache-write", action=argparse.BooleanOptionalAction, default=True,
                    help="on by default: write the 16-bit disk-cache copy of each new clip on a background thread so "
                         "the reply never waits for the disk; a read of a clip still being written waits for it, so "
                         "served clips stay the same bits; --no-async-cache-write writes inline")
    ap.add_argument("--keep-warm-ms", type=float, default=0.0,
                    help="keep the GPU warm while a client is connected, one tiny job every this many ms (0 = off)")
    ap.add_argument("--sibling-on-miss", action="store_true",
                    help="on a realtime miss, play a cached sibling key at once and generate the exact key behind it")
    ap.add_argument("--short-clips", action="store_true",
                    help="the old presentation: a 0.6 s window around the loudest point at one rms per intensity; the "
                         "default keeps each sound until it dies away (up to 1.5 s) and sets its peak per intensity")
    ap.add_argument("--no-triton-gemm", action="store_true",
                    help="keep cublas with the bias fused for every DiT linear; the default swaps three shapes to an "
                         "fp16-accumulating triton gemm on pytorch 2.7+, judged equal, about 3.5 ms a clip faster")
    ap.add_argument("--no-skip-first", action="store_true",
                    help="run all 3 euler steps from t=1.0; the default drops the t=1.0 step, judged equal on held-out refs")
    ap.add_argument("--realtime", action="store_true",
                    help="generate each sound on-demand at fire time and cache it; nothing pre-generated at startup")
    ap.add_argument("--decoder-ckpt", default=None,
                    help="safetensors state_dict for the SAME-S decoder (scripts/train/dec_train.py export), loaded "
                         "strict before the decoder graph is captured; relative paths resolve against the repo root. "
                         "Falls back to SARGE_DECODER_CKPT")
    args = ap.parse_args(argv)
    if args.generator == "hybrid":
        prefetch.start_hybrid(HYBRID_GUNS["model_name"], args.woosh_model,
                              weights=lambda: weight_cache.locate(HYBRID_GUNS["model_name"], cache_dir_of(args)))

    tier = (args.tier or os.environ.get("SARGE_GEN_TIER") or "default").lower()
    gen_kwargs = dict(TIERS.get(tier, TIERS["default"]), fast_load=True)
    if args.sampler is not None:
        gen_kwargs["sampler_type"] = args.sampler
    if args.steps is not None:
        gen_kwargs["steps"] = args.steps
        gen_kwargs.pop("mid_noise", None)
    if args.no_compile:
        for k in COMPILED:
            gen_kwargs.pop(k, None)
    if args.stereo:
        gen_kwargs["stereo"] = True
    if args.detailed_prompts:
        gen_kwargs["prompt_style"] = "detailed"
    if args.amp:
        gen_kwargs["amp"] = True if args.amp == "full" else args.amp
    if args.short_clips:
        for k in PRESENT:
            gen_kwargs.pop(k, None)
    if args.level:
        gen_kwargs["level"] = args.level
    if args.no_triton_gemm:
        gen_kwargs.pop("unfuse_bias", None)
        gen_kwargs.pop("triton_gemm", None)
    if args.no_skip_first:
        gen_kwargs.pop("skip_first_step", None)
    if args.generator in ("sa3", "hybrid"):
        check_weights(HYBRID_GUNS["model_name"] if args.generator == "hybrid" else gen_kwargs.get("model_name",
                                                                                                    "small-sfx"))
    if args.generator == "hybrid":
        if not os.path.isfile(args.woosh_python):
            raise SystemExit("woosh python not found: %s" % args.woosh_python)
        gen_kwargs.update(HYBRID_GUNS, woosh_port=args.woosh_port, start_rest=lambda: start_woosh(args),
                          rest_cost_ms=WOOSH_COST_MS[args.woosh_model])
        if cache_dir_of(args):
            gen_kwargs["weight_cache"] = cache_dir_of(args)
        if early_woosh():
            woosh = launch_woosh(args)
            gen_kwargs.update(start_rest=lambda: open_woosh(args, *woosh), load_rest=lambda: gate(woosh[0], "load"),
                              drop_rest=lambda: close_woosh(woosh[0]))
    ckpt = args.decoder_ckpt or os.environ.get("SARGE_DECODER_CKPT")
    if ckpt:
        ckpt = ckpt if os.path.isabs(ckpt) else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ckpt)
        if not os.path.isfile(ckpt):
            raise SystemExit("decoder checkpoint not found: %s" % ckpt)
        gen_kwargs["decoder_ckpt"] = ckpt
    sampler, steps = gen_kwargs["sampler_type"], gen_kwargs["steps"]
    gen_cost = args.gen_cost_ms if args.gen_cost_ms is not None else (
        WOOSH_COST_MS[args.woosh_model] if args.generator == "hybrid" else TIER_COST_MS.get(tier, 200.0))

    vram_kb = args.vram_kb if args.vram_kb is not None else (256_000.0 if args.world else 50_000.0)
    kwargs = dict(vram_budget_kb=vram_kb, gpu_ms_budget=args.gpu_ms_budget,
                  variants_per_key=args.variants, target_variants=args.target_variants,
                  continuous_topk=args.topk, gen_cost_ms=gen_cost)
    extra = {k: v for k, v in (("world", args.world), ("event_log", args.event_log),
                               ("predict_fixes", args.predict_fixes), ("pool_fix", args.pool_fix),
                               ("concurrent_gen", args.concurrent_gen), ("keep_warm_ms", args.keep_warm_ms),
                               ("sibling_on_miss", args.sibling_on_miss), ("cache_async", args.async_cache_write))
             if v}
    cache_dir = None if args.no_cache else args.cache_dir
    cleared = 0
    if cache_dir and os.path.isdir(cache_dir):
        for f in os.listdir(cache_dir):
            if f.endswith(".wav"):
                os.remove(os.path.join(cache_dir, f))
                cleared += 1

    if args.generator != "mock":
        print(f"loading + warming {args.generator} model (first time can take ~30s)...", flush=True)
    server = serve(generator=args.generator, host=args.host, port=args.port, pcm=args.pcm,
                   cache_dir=cache_dir, deepen=not args.no_deepen,
                   gen_kwargs=gen_kwargs,
                   learn=args.learn, realtime=args.realtime, deepen_batch=args.deepen_batch, **extra, **kwargs)
    host, port = server.server_address
    startup.done("READY")
    mode = "realtime-on-demand" if args.realtime else "predict+prewarm"
    print(f"sidecar READY on {host}:{port}  generator={args.generator}  tier={tier} "
          f"sampler={sampler} steps={steps}  gen_cost_ms={gen_cost:.0f}  mode={mode}  "
          f"learn={'on' if args.learn else 'off'}  cache={cache_dir or 'off'} (cleared {cleared})  "
          f"decoder={os.path.basename(gen_kwargs['decoder_ckpt']) if gen_kwargs.get('decoder_ckpt') else 'base'}  "
          f"concurrent-gen={'on' if args.concurrent_gen else 'off'}  "
          f"cache-write={'async' if args.async_cache_write else 'inline'}",
          flush=True)
    print("ctrl-c to stop", flush=True)

    stop = threading.Event()
    try:
        while not stop.wait(1.0):
            pass
    except KeyboardInterrupt:
        print("shutting down")
    finally:
        server.deepening = False
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
