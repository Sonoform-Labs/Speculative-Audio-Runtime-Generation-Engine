import _bootstrap

import time

from sage.sidecar import serve, SfxClient

EVENTS = [("footstep", "wood"), ("impact", "glass"), ("hit", "metal"),
          ("scrape", "metal"), ("slam", "door"), ("splash", "water")]
N = 30


def pcts(xs):
    xs = sorted(xs)
    def p(q):
        return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]
    return dict(p50=p(.50), p90=p(.90), p95=p(.95), min=xs[0], max=xs[-1], mean=sum(xs) / len(xs))


def main():
    t0 = time.perf_counter()
    srv = serve(generator="sa3", realtime=True, host="127.0.0.1", port=0, cache_dir=None,
                gen_kwargs={"steps": 3, "sampler_type": "euler"}, learn=False, deepen=False,
                vram_budget_kb=400_000, gpu_ms_budget=1e9)
    cold_ms = (time.perf_counter() - t0) * 1000.0

    samples = {4: [], 3: []}
    try:
        with SfxClient(*srv.server_address) as c:
            c.context(0.0, None, ["wood", "metal", "glass", "door", "water"], "arena")
            for st in (4, 3):
                srv.engine.gen.steps = st
                for k in range(3):
                    c.event("footstep", "grass", "normal", f"warm{st}{k}")
            for i in range(N):
                et, mat = EVENTS[i % len(EVENTS)]
                for st in (4, 3):
                    srv.engine.gen.steps = st
                    srv.engine.gen.sampler_type = "euler"
                    ctx = f"m{st}_{i}"
                    t = time.perf_counter()
                    meta, _ = c.event(et, mat, "normal", ctx)
                    dt = (time.perf_counter() - t) * 1000.0
                    assert meta["served"] == "realtime", meta["served"]
                    samples[st].append(dt)
    finally:
        srv.shutdown()
        srv.server_close()

    print(f"\ncold start (model load + warmup gen): {cold_ms:.0f} ms  [one-time, NOT per-event]")
    print(f"warm gens: {len(samples[4])} euler@4 + {len(samples[3])} euler@3, interleaved, "
          f"end-to-end through the sidecar (incl framing+PCM)\n")
    hdr = f"{'tier':16s}{'p50':>7}{'p90':>7}{'p95':>7}{'min':>7}{'max':>7}{'mean':>7}"
    print(hdr); print("-" * len(hdr))
    s4, s3 = pcts(samples[4]), pcts(samples[3])
    for lbl, s in [("euler@4 (OLD)", s4), ("euler@3 (NEW)", s3)]:
        print(f"{lbl:16s}{s['p50']:>7.0f}{s['p90']:>7.0f}{s['p95']:>7.0f}{s['min']:>7.0f}{s['max']:>7.0f}{s['mean']:>7.0f}")

    pair = [samples[3][i] - samples[4][i] for i in range(len(samples[3]))]
    pp = pcts(pair)
    print("\nWITHIN-RUN DELTA (drift-robust; @4 and @3 measured back-to-back each iteration):")
    print(f"  per-tier p50:  euler@3 - euler@4 = {s3['p50'] - s4['p50']:+.0f} ms")
    print(f"  paired @3-@4:  p50 {pp['p50']:+.0f} ms   mean {pp['mean']:+.0f} ms   "
          f"(negative = euler@3 faster; {sum(1 for d in pair if d < 0)}/{len(pair)} iters @3 faster)")
    print("\nNote: absolute latency drifts with GPU clock/thermal state run-to-run; the paired "
          "within-run delta above is the metric to trust, not the absolute p50.")


if __name__ == "__main__":
    main()
