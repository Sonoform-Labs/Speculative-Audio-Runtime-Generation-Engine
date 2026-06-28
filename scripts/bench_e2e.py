import _bootstrap

import os
import time

import numpy as np

from predictive_sfx.sidecar import serve, SfxClient
from predictive_sfx.types import AudioClip

# (event_type, material) pairs the CLIENT sends; transient-heavy.
EVENTS = [
    ("footstep", "wood"),
    ("impact", "glass"),
    ("hit", "metal"),
    ("scrape", "metal"),
    ("slam", "door"),
    ("splash", "water"),
]


def measure(c, engine, steps, outdir):
    """Set the live tier, force fresh misses, measure full client round-trip per event, save WAV."""
    engine.gen.steps = steps
    engine.gen.sampler_type = "euler"
    engine.cache.clear()
    engine.recent.clear()
    engine.generated.clear()
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for et, mat in EVENTS:
        t0 = time.perf_counter()
        meta, pcm = c.event(et, mat, "normal", "default")     # real EVENT->AUDIO round-trip
        dt = (time.perf_counter() - t0) * 1000.0
        a = np.frombuffer(pcm, dtype=np.float32).copy()
        wav = os.path.join(outdir, f"{et}_{mat}.wav")
        AudioClip(a, int(meta["sample_rate"])).to_wav(wav)
        rows.append(dict(event=f"{et}|{mat}", served=meta["served"], lat_ms=round(dt, 1),
                         sr=meta["sample_rate"], n=meta["n_samples"], saved=os.path.exists(wav)))
    return rows


def main():
    print("loading sa3 (realtime, no disk cache so every event really generates)...")
    srv = serve(generator="sa3", realtime=True, host="127.0.0.1", port=0, cache_dir=None,
                gen_kwargs={"steps": 3, "sampler_type": "euler"}, learn=False, deepen=False,
                vram_budget_kb=400_000, gpu_ms_budget=1e9)
    TIER_STEPS = [8, 4, 3, 2]                 # 8=ref, 4=OLD default, 3=NEW default, 2=fast
    results = {}
    meta_keys = None
    try:
        with SfxClient(*srv.server_address) as c:
            print("welcome generator:", c.welcome.get("generator"))
            c.context(0.0, None, ["wood", "metal", "glass", "door", "water"], "arena")
            c.event("footstep", "grass", "normal", "warm")          # warm kernels
            for st in TIER_STEPS:
                results[st] = measure(c, srv.engine, st, f"docs/e2e-bench/euler_{st}step")
            meta, _ = c.event("footstep", "stone", "normal", "apicheck")
            meta_keys = sorted(meta.keys())
    finally:
        srv.shutdown()
        srv.server_close()

    def med(rs):
        v = sorted(x["lat_ms"] for x in rs)
        return v[len(v) // 2]

    labels = {8: "euler@8 (quality/ref)", 4: "euler@4 (OLD default)",
              3: "euler@3 (NEW default)", 2: "euler@2 (fast)"}
    print("\n=== END-TO-END client round-trip THROUGH THE SIDECAR (gen + framing + PCM) ===")
    for st in TIER_STEPS:
        rs = results[st]
        print(f"\n{labels[st]}: median {med(rs):.0f}ms")
        for x in rs:
            print(f"  {x['event']:16s} served={x['served']:9s} {x['lat_ms']:7.1f}ms  saved={x['saved']}")
    print("\n--- apples-to-apples (all end-to-end) ---")
    print(f"OLD default euler@4: {med(results[4]):.0f}ms")
    print(f"NEW default euler@3: {med(results[3]):.0f}ms   (delta {med(results[3]) - med(results[4]):+.0f}ms)")
    print(f"fast        euler@2: {med(results[2]):.0f}ms   (delta {med(results[2]) - med(results[4]):+.0f}ms)")
    print("\nAUDIO meta keys (API shape):", meta_keys)
    print("all wavs saved:", all(x["saved"] for st in TIER_STEPS for x in results[st]))


if __name__ == "__main__":
    main()
