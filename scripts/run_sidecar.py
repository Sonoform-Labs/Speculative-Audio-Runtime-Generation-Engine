import _bootstrap

import argparse
import threading

from predictive_sfx.sidecar import serve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the predictive-SFX sidecar audio service.")
    ap.add_argument("--generator", default="mock")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--pcm", default="f32le", choices=["f32le", "i16le"])
    ap.add_argument("--vram-kb", type=float, default=50_000.0)
    ap.add_argument("--gpu-ms-budget", type=float, default=10_000_000.0)
    ap.add_argument("--gen-cost-ms", type=float, default=None)
    ap.add_argument("--variants", type=int, default=3)
    args = ap.parse_args(argv)

    kwargs = dict(vram_budget_kb=args.vram_kb, gpu_ms_budget=args.gpu_ms_budget, variants_per_key=args.variants)
    if args.gen_cost_ms is not None:
        kwargs["gen_cost_ms"] = args.gen_cost_ms

    if args.generator != "mock":
        print(f"loading + warming {args.generator} model (first time can take ~30s)...")
    server = serve(generator=args.generator, host=args.host, port=args.port, pcm=args.pcm, **kwargs)
    host, port = server.server_address
    print(f"sidecar READY on {host}:{port}  generator={args.generator}  pcm={args.pcm}")
    print("ctrl-c to stop")

    stop = threading.Event()
    try:
        while not stop.wait(1.0):
            pass
    except KeyboardInterrupt:
        print("shutting down")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
