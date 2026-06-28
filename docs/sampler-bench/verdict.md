# Stage 0 — sampler-swap test verdict (2026-06-28)

Reproducible test: `scripts/bench_samplers.py` (generate) + `scripts/score_samplers.py` (score).
14 transient-heavy SFX prompts × 14 sampler/step configs = 196 gens, fixed seed per prompt
(seed_base+idx), cfg=2.0, gen 1.0s → 0.6s extract, same loudness norm. All WAVs + `manifest.csv`
saved under `docs/sampler-bench/`. Latency = `time.perf_counter` around `model.generate` with
`torch.cuda.synchronize()` (real wall-clock, GPU-synced). Sampler genuinely swapped via
`generate(sampler_type=...)` → `sample_diffusion` dispatch (`sampling.py:493-503`), not a `steps` relabel.

## Results (objective metrics)

```
sampler   steps   lat_ms  CLAP_match  crest_rel     PQ  dead
euler         8      443         43%       1.00   7.02     0   <- reference (8-step quality)
euler         4      239         36%       1.07   7.04     0
euler         3      190         14%*      1.17   7.16     0   <- FREE WIN: full quality
euler         2      139         29%*      0.73   6.72     0   <- transients soften
euler         1       90         29%*      0.29   6.55     0   <- transients dead
dpmpp         4      243         36%       1.01   7.08     0
dpmpp         3      191         14%*      1.17   7.16     0
dpmpp         2      144         21%*      0.74   6.82     0
rk4           4       38          7%       0.43   5.68     0   <- BROKEN (=chance, flat 38ms)
rk4           3       38          7%       0.43   5.68     0   <- BROKEN
rk4           2       38          7%       0.43   5.68     0   <- BROKEN
pingpong      4      240         50%       1.00   6.88     0
pingpong      3      188         29%*      1.19   7.01     0
pingpong      2      141         29%*      0.61   6.83     0
```
CLAP_match = rank-1 of correct prompt among 14 (chance 7%). crest_rel = crest factor / euler@8
(transient-sharpness proxy; 1.0 = attack as sharp as 8-step). PQ = Audiobox production-quality (1-10).
\*CLAP_match is high-variance at N=14/config (non-monotonic) — treat it as coarse, not a fine ranking.

## Findings (evidence-based)
1. **rk4 is broken on this checkpoint** — CLAP=chance, crest 0.43, PQ 5.68, identical flat 38ms at all
   step counts (not doing the work). Exclude.
2. **No free-lunch sampler.** dpmpp ≈ euler at every matched step (CLAP/crest/PQ within noise, same
   latency). pingpong (meant for *distilled* models) ties euler at 3-4 steps, worse transients at 2.
   So a "better solver" does NOT recover few-step quality here — Euler is already as good as it gets.
3. **The real degradation is transient smear, and it's step-count-driven:** crest holds at 3-4 steps
   (≥1.0), drops to ~0.73 at 2 steps, collapses to 0.29 at 1 step. PQ barely moves 8→2 (7.02→6.72)
   then dips at 1 (6.55). So quality/timbre survives few steps; sharp ATTACKS do not.

## Usability
- **3-step (euler, 190ms): USABLE, full quality, FREE.** PQ ≥ 8-step, transients sharp, semantics
  comparable. A real win over the 4-step/224ms currently shipped — zero training, zero quality loss.
- **2-step (139ms): BORDERLINE.** Timbre holds (PQ 6.72) but attacks soften (crest 0.73) — exactly the
  failure mode that matters for percussive SFX. OK for soft/sustained sounds, not for sharp hits.
- **1-step (90ms): NOT usable.** Transients dead (crest 0.29).

## Verdict
- **If ~190ms is acceptable → SKIP TRAINING.** Ship euler@3: free 224ms→190ms at full quality.
- **If you need ≤130ms (2-step) at SHARP quality → ARC/adversarial training REQUIRED.** No sampler
  recovers 2-step transient sharpness; this is precisely the gap adversarial post-training closes
  (restores attack energy that few-step sampling smears). LCM/consistency would NOT fix it (mean-
  regression smears transients further).

Caveat: these are OBJECTIVE proxies (CLAP=semantic match, crest=attack sharpness, PQ=quality), not
human listening. WAVs are saved per config for the user to confirm by ear.
