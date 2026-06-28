# End-to-end sidecar latency — Euler@3 default wiring (2026-06-28)

Measured through the REAL path: `serve(generator=sa3, realtime=True, cache_dir=None)` + `SfxClient`
over loopback. Each event is a fresh cache miss → inline realtime generation, so the number is the
full client round-trip: EVENT frame → gen → loudness/extract → PCM encode → AUDIO frame → client
decode. `scripts/bench_e2e.py`.

Apples-to-apples, all end-to-end, measured back-to-back in ONE run (the reliable comparison):

| tier | sampler/steps | e2e median | vs old default |
|------|---------------|-----------:|---------------:|
| quality     | euler @8 | 404 ms | — |
| OLD default | euler @4 | 229 ms | (baseline) |
| **NEW default** | **euler @3** | **179 ms** | **-50 ms** |
| fast        | euler @2 | 134 ms | -95 ms |

NOTE: absolute e2e latency drifts run-to-run ~±30 ms with GPU clock/scheduling state (a separate run
read euler@3=212 ms). The trustworthy signal is the WITHIN-RUN delta: euler@3 is ~50 ms FASTER than
the old euler@4 default, at full quality; euler@2 is ~95 ms faster (softer transients).

- All 6 transient events served `realtime`, sr=44100, every WAV saved OK
  (`docs/e2e-bench/euler_3step/`, `euler_2step/`).
- API shape UNCHANGED — AUDIO meta keys: byte_len, channels, duration_s, format, intent, learned,
  n_samples, sample_rate, served, type, variant_id. No wire/protocol change; only the internal
  default sampler/steps changed.

## Robust stats — 30×30 interleaved (scripts/bench_e2e_stats.py)
Cold start (model load + warmup gen): ~17.5 s, one-time, excluded from per-event numbers.
30 warm gens per tier, interleaved (@4 then @3 each iteration to cancel thermal drift), end-to-end.

```
tier             p50   p90   p95   min   max  mean
euler@4 (OLD)    220   224   224   217   227   221
euler@3 (NEW)    175   178   182   172   184   175
```
Within-run paired delta: -46 ms p50, -45 ms mean, **30/30 iterations euler@3 faster**. Distributions
do NOT overlap (@3 max 184 < @4 min 217). euler@3 is consistently ~46 ms faster e2e at full quality.

## Tiers (reversible)
`run_sidecar.py` selects via `--tier` or `PSFX_GEN_TIER` env var (`--steps`/`--sampler` override):
- `quality` = euler@8 (reference)  · `default` = euler@3 (shipped)  · `fast` = euler@2 (experimental)
- RK4 is DISABLED/unsupported (broken at low steps); it is not a tier.

## Read
Euler@3 e2e ~212 ms = full quality, ~34 ms faster than before, no training. Euler@2 e2e ~149 ms hits
the ~150 ms comfort threshold but softens transients. Sub-130 ms at sharp quality still needs ARC.
A/B pack for ear check: `docs/ab-listen/` (footstep/glass/metalhit/scrape/doorslam/watersplash ×
8/3/2-step, same seed).
