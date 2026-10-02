import _bootstrap

import argparse
import os
import threading

from sarge.sidecar import serve

TIERS = {
    "quality": {"sampler_type": "euler", "steps": 3},
    "default": {"sampler_type": "euler", "steps": 3},
    "fast":    {"sampler_type": "euler", "steps": 2, "mid_noise": 0.68},
}
TIER_COST_MS = {"quality": 130.0, "default": 130.0, "fast": 90.0}


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
                         "fast=euler@2 renoised at 0.68 (experimental). "
                         "Falls back to SARGE_GEN_TIER env var then 'default'. --steps/--sampler override it.")
    ap.add_argument("--steps", type=int, default=None, help="override the tier's step count")
    ap.add_argument("--sampler", default=None, help="override the tier's sampler (euler; rk4 DISABLED)")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--cache-dir", default=os.path.join(".cache", "sfx_bake"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-deepen", action="store_true")
    ap.add_argument("--deepen-batch", type=int, default=2,
                    help="clips per background generation; 2 nearly doubles throughput for +17 ms of GPU hold")
    ap.add_argument("--learn", action="store_true",
                    help="use the runtime affordance learner + router (auto-derive affordances, no hand-authored tables)")
    ap.add_argument("--realtime", action="store_true",
                    help="generate each sound on-demand at fire time and cache it; nothing pre-generated at startup")
    args = ap.parse_args(argv)

    tier = (args.tier or os.environ.get("SARGE_GEN_TIER") or "default").lower()
    gen_kwargs = dict(TIERS.get(tier, TIERS["default"]))
    if args.sampler is not None:
        gen_kwargs["sampler_type"] = args.sampler
    if args.steps is not None:
        gen_kwargs["steps"] = args.steps
        gen_kwargs.pop("mid_noise", None)
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
        print(f"loading + warming {args.generator} model (first time can take ~30s)...")
    server = serve(generator=args.generator, host=args.host, port=args.port, pcm=args.pcm,
                   cache_dir=cache_dir, deepen=not args.no_deepen,
                   gen_kwargs=gen_kwargs,
                   learn=args.learn, realtime=args.realtime, deepen_batch=args.deepen_batch, **kwargs)
    host, port = server.server_address
    mode = "realtime-on-demand" if args.realtime else "predict+prewarm"
    print(f"sidecar READY on {host}:{port}  generator={args.generator}  tier={tier} "
          f"sampler={sampler} steps={steps}  gen_cost_ms={gen_cost:.0f}  mode={mode}  "
          f"learn={'on' if args.learn else 'off'}  cache={cache_dir or 'off'} (cleared {cleared})")
    print("ctrl-c to stop")

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
