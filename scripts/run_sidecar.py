import _bootstrap

import argparse
import os
import threading

from predictive_sfx.sidecar import serve


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
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--cache-dir", default=os.path.join(".cache", "sfx_bake"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-deepen", action="store_true")
    ap.add_argument("--learn", action="store_true",
                    help="use the runtime affordance learner + router (auto-derive affordances, no hand-authored tables)")
    ap.add_argument("--realtime", action="store_true",
                    help="generate each sound on-demand at fire time and cache it; nothing pre-generated at startup")
    args = ap.parse_args(argv)

    kwargs = dict(vram_budget_kb=args.vram_kb, gpu_ms_budget=args.gpu_ms_budget,
                  variants_per_key=args.variants, target_variants=args.target_variants,
                  continuous_topk=args.topk)
    if args.gen_cost_ms is not None:
        kwargs["gen_cost_ms"] = args.gen_cost_ms
    cache_dir = None if args.no_cache else args.cache_dir

    if args.generator != "mock":
        print(f"loading + warming {args.generator} model (first time can take ~30s)...")
    server = serve(generator=args.generator, host=args.host, port=args.port, pcm=args.pcm,
                   cache_dir=cache_dir, deepen=not args.no_deepen,
                   gen_kwargs={"steps": args.steps}, learn=args.learn, realtime=args.realtime, **kwargs)
    host, port = server.server_address
    mode = "realtime-on-demand" if args.realtime else "predict+prewarm"
    print(f"sidecar READY on {host}:{port}  generator={args.generator}  steps={args.steps}  "
          f"mode={mode}  learn={'on' if args.learn else 'off'}  cache={cache_dir or 'off'}")
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
