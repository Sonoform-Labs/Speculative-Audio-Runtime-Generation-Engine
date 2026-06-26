# Model research — fast generation + where prediction pays off (2026-06-26)

Deep-research pass (5 angles, 23 sources, 22 verified claims). Source of truth for the
generation-model upgrade + the paper's reframed contribution.

## TL;DR
1. **Generation crossed real-time.** One-step / few-step distilled flow & consistency models do the
   network pass in single-digit-to-tens of ms — vs the current ~393 ms Stable Audio 3 Small. The
   network pass is no longer the bottleneck.
2. **This undercuts latency-hiding as the reason to predict** (confirms the "it looks the same"
   finding): if a clip generates in ~30 ms you barely need to predict ahead in a small palette.
3. **So the paper pivots to a stronger thesis:** the novel delta is **predict-then-GENERATE** (not
   predict-then-fetch, which is prior art), and prediction earns its keep in the **budget / large
   key-space / cold-start** regime — deciding *what* to generate when you can't generate everything.

## Fast generation models (the demo win)
All latencies are author-reported on 3090/4090/A40/H100 — the **RTX 3060 is ~5-6x weaker than a 4090,
so every number must be re-measured end-to-end on the 3060** (incl. text-encode + decode + IPC).

| Model | Speed (reported) | Size | Why | Source |
|---|---|---|---|---|
| ARC-distilled **Stable Audio Open (Small)** | ~75ms / 12s clip (H100) | same family as current | lowest-risk swap, 44.1kHz stereo | arXiv 2505.08175 |
| **MeanAudio** | RTF 0.013, 1-NFE (3090) | 120M (tiny) | smallest/fastest, best for tight VRAM | arXiv 2508.06098 |
| **AudioLCM** | 2 steps, RTF 0.003 | — | well-benchmarked, mel-based | arXiv 2406.00356 |
| **FlashAudio** | 1 step, FAD 1.49 | — | best reported 1-step quality | arXiv 2410.12266 |
| **TangoFlux** | 30s/3.7s (A40) | 515M flow | quality-leaning, fits 12GB | arXiv 2412.21037 |
| **SoundCTM** | 1-step OR multi-step | 1B | quality DIAL: spend extra steps when GPU-idle (great for cache-warming) | arXiv 2405.18503 |
| AudioTurbo | 10 steps | — | middle-ground if 1-step quality insufficient | arXiv 2505.22106 |

KEY local note: the existing `stableaudio` (open-small) and `sa3` backends are ALREADY distilled
few-step models. **Dropping their step count (8 -> 2 -> 1) is the zero-install fast win to test first.**

## The novelty delta (paper)
- **predict-then-FETCH = prior art**: Microsoft US10154072B2 (predicts game context to PREFETCH
  existing assets, retrieval-only); InstCache (pre-computes text); SpeCache (prefetches KV).
- **predict-then-GENERATE media = the open delta** no surveyed work occupies. Frame the contribution
  as coupling event prediction to neural SYNTHESIS + authored fallback.
- **Where prediction beats reactive caching:** InstCache shows speculative cache-population beating
  exact-match caching's theoretical ceiling by 2.3x — when keys rarely repeat (large/combinatorial,
  cold-start). That's the regime; the small-palette demo is the opposite (caching suffices).
- **Formal backbone:** cache theory (Belady / Demand-MIN, ISCA 2018) — speculation changes the
  optimal eviction policy.
- **Closest neural-game-audio prior art:** EA SEED neural SFX synthesis (AIIDE'22). Neural game
  audio exists; the predict->generate->cache->fallback loop does not.

## The reframed paper story
> As neural audio generation crosses real-time, the bottleneck shifts from latency to BUDGET: you
> still can't pre-generate every (event x material x intensity x context) combination under a VRAM
> cap. Prediction decides WHAT to generate — and we show the hit-rate crossover where speculative
> generation beats reactive caching as the sound space grows.

The "it looks the same" small-palette result becomes Figure 1; the hit-rate-vs-key-space crossover
curve (extend the existing benchmark) becomes the contribution.

## Caveats (honest)
- 3060 numbers MUST be measured (published figures are bigger GPUs, batch 1, exclude IPC/decode).
- Self-report bias: most latency/FAD numbers are authors' own vs one chosen baseline.
- SFX-specific quality (Foley realism, material/intensity control) is uncharacterized by the general
  AudioCaps FAD/CLAP benchmarks — may not reflect short game-SFX quality.
- Refuted (do NOT cite): Stable Audio Open ARC mobile/on-device feasibility; SoundCTM exact
  distillation mechanism; AudioLCM "333x" as sub-ms. Microsoft-patent prior-art passed only 2-1 —
  state as "retrieval-scoped prior art," not absolute.
- Field moves fast — re-survey before submission.

## Open questions
- Actual end-to-end per-clip latency of each model on the 3060 (incl. text-encode + decode + IPC) —
  which dominates once the network pass is sub-50ms?
- Audio-domain hit-rate vs key-space-size / budget data (must generate empirically — the paper figure).
- One-step SFX perceptual quality + material/intensity controllability vs authored parity.
- What has actually SHIPPED in runtime neural game audio (to fully bound the novelty claim).
