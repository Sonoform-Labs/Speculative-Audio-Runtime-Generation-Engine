import _bootstrap

import argparse
import os

import numpy as np

from sarge.affordance import make_session
from sarge.sidecar import serve, SfxClient
from sarge.types import AudioClip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="End-to-end sidecar demo over the wire (no Unity, no GPU).")
    ap.add_argument("--generator", default="mock")
    ap.add_argument("--pcm", default="f32le", choices=["f32le", "i16le"])
    ap.add_argument("--out", default=os.path.join("data", "demo", "sidecar_clip.wav"))
    args = ap.parse_args(argv)

    server = serve(generator=args.generator, host="127.0.0.1", port=0, pcm=args.pcm,
                   vram_budget_kb=50_000, gpu_ms_budget=10_000_000, variants_per_key=3)
    host, port = server.server_address
    saved = None
    sr = 48000
    try:
        with SfxClient(host, port, pcm=args.pcm) as client:
            w = client.welcome
            sr = int(w["sample_rate"])
            print(f"connected -> generator={w['generator']}  sample_rate={sr}  pcm={w['pcm']}")
            n_hit = n_fb = 0
            for t, kind, payload in make_session():
                if kind == "context":
                    client.context(payload.time_s, payload.equipped, payload.surfaces, payload.region)
                else:
                    meta, samples = client.event(payload.event_type, payload.material, payload.intensity, payload.context)
                    served = meta["served"]
                    if served == "cached":
                        n_hit += 1
                    else:
                        n_fb += 1
                    print(f"{t:5.1f}s  {payload.event_type:11s}/{payload.material:8s} -> {served.upper():8s}  "
                          f"{meta['n_samples']} samp  v{meta['variant_id']}")
                    if saved is None:
                        saved = samples
            st = client.stats()
            total = n_hit + n_fb
            print(f"\nhit-rate {n_hit / total * 100:.0f}%  ({n_hit}/{total})  "
                  f"cached_keys={st['cached_keys']}  wasted={st['wasted']}  "
                  f"spec_gen={st['stats']['spec_gen']}  demand={st['stats']['demand_gen']}")
    finally:
        server.shutdown()
        server.server_close()

    if saved is not None:
        AudioClip(np.asarray(saved, dtype=np.float32), sr).to_wav(args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
