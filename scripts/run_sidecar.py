import _bootstrap

import argparse
import importlib.metadata
import os
import threading

_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
try:
    _TORCH = importlib.metadata.version("torch").split("+")[0]
except importlib.metadata.PackageNotFoundError:
    _TORCH = "none"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_CACHE, "inductor" if _TORCH.startswith("2.6") else "inductor-" + _TORCH))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_CACHE, "triton-" + _TORCH))

from sarge.sidecar import serve

COMPILED = {"compile_mode": "default", "compile_decode": True, "graph_decode": True, "hoist_rope": True,
            "fp16_accumulation": True, "graph_text": True, "unfuse_bias": True, "triton_gemm": True,
            "decode_prefix": True}
TIERS = {
    "quality": {"sampler_type": "euler", "steps": 3, "skip_first_step": True, **COMPILED},
    "default": {"sampler_type": "euler", "steps": 3, "skip_first_step": True, **COMPILED},
    "fast":    {"sampler_type": "euler", "steps": 3, "skip_first_step": True, "cfg_interval": (0.9, 1.0), **COMPILED},
}
TIER_COST_MS = {"quality": 28.0, "default": 28.0, "fast": 25.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the predictive-SFX sidecar audio service.")
    ap.add_argument("--generator", default="sa3")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--pcm", default="f32le", choices=["f32le", "i16le"])
    ap.add_argument("--vram-kb", type=float, default=50_000.0)
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
    ap.add_argument("--no-triton-gemm", action="store_true",
                    help="keep cublas with the bias fused for every DiT linear; the default swaps three shapes to an "
                         "fp16-accumulating triton gemm on pytorch 2.7+, judged equal, about 3.5 ms a clip faster")
    ap.add_argument("--no-skip-first", action="store_true",
                    help="run all 3 euler steps from t=1.0; the default drops the t=1.0 step, judged equal on held-out refs")
    ap.add_argument("--realtime", action="store_true",
                    help="generate each sound on-demand at fire time and cache it; nothing pre-generated at startup")
    args = ap.parse_args(argv)

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
    if args.no_triton_gemm:
        gen_kwargs.pop("unfuse_bias", None)
        gen_kwargs.pop("triton_gemm", None)
    if args.no_skip_first:
        gen_kwargs.pop("skip_first_step", None)
    sampler, steps = gen_kwargs["sampler_type"], gen_kwargs["steps"]
    gen_cost = args.gen_cost_ms if args.gen_cost_ms is not None else TIER_COST_MS.get(tier, 200.0)

    kwargs = dict(vram_budget_kb=args.vram_kb, gpu_ms_budget=args.gpu_ms_budget,
                  variants_per_key=args.variants, target_variants=args.target_variants,
                  continuous_topk=args.topk, gen_cost_ms=gen_cost)
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
                   learn=args.learn, realtime=args.realtime, deepen_batch=args.deepen_batch, **kwargs)
    host, port = server.server_address
    mode = "realtime-on-demand" if args.realtime else "predict+prewarm"
    print(f"sidecar READY on {host}:{port}  generator={args.generator}  tier={tier} "
          f"sampler={sampler} steps={steps}  gen_cost_ms={gen_cost:.0f}  mode={mode}  "
          f"learn={'on' if args.learn else 'off'}  cache={cache_dir or 'off'} (cleared {cleared})", flush=True)
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
