# Predictive Neural SFX — Roadmap

North star: predict upcoming sound-triggering game events -> pre-generate neural SFX ahead of
time -> cache by (event, material, intensity, context) -> fall back to authored assets on a miss.
The novel claim is the predict -> *generate* -> cache -> fallback loop (speculative prefetching for
sound), and reusing the game's own motion/animation signals as a near-free event predictor.

Target venue: EXAG @ AIIDE 2026 (short paper, archival via CEUR). Deadline ~Aug 28, 2026.
Secondary: ADC Bristol talk, CoG demo. Publication is the goal; attendance secondary.

---

## Why this order

A smart multi-model predictor feeding an engine that blocks the audio path, leaks VRAM, and pumps
loudness is building on sand. So: fix the engine foundation first, then calibrate, then make the
prediction actually smart, then validate + benchmark, then write.

The red-team (75 failure modes -> 43 gaps) was decisive on the core design question:
**the gaps are mostly mechanism fixes and real engine bugs, NOT prediction-coverage holes.**
Adding more predictors does not fix them. Exactly ONE new model is worth building now (calibration).

---

## Phase 1 — Critical engine bug fixes (foundation)

Real defects found in current code. Highest impact, self-contained.

- [x] P1.1 Separate authored-fallback from demand-gen in `affordance.py on_event`: on a miss ALWAYS
      play authored instantly (sub-frame), queue neural gen in the background for next time. Never
      block the audio path on a ~393ms gen. DONE — miss serves in ~0.4ms; cached/fallback/silence
      model ported from pipeline.py; demand queue drained by the background loop.
- [x] P1.2 Stop holding `engine_lock` across generation (`server.py _deepen_loop`): hold the lock
      only to plan the batch and commit results; run generation OUTSIDE the lock; give real events
      lock priority; lock-free read path for cache hits. DONE — plan_batch/commit_clips split +
      inflight guard; cache-hit latency 0.06ms while a 600ms gen is in flight.
- [x] P1.3 Route demand-gen inserts through the same budgeted admission/eviction path as speculative
      gen (fix unbounded VRAM growth -> OOM). DONE — demand-gen goes through commit_clips eviction;
      demand queue capped at 128; verified bounded under a 500-key flood. Also fixed a latent
      eviction crash (min() comparing unorderable CacheKeys).

## Phase 2 — Engine correctness + quality mechanisms

DONE NOW (standalone). The other three are predictor-coupled -> moved next to Phase 3 (they consume
the ETA/confidence only the predictor produces; building them against no predictor = dead hooks).

- [x] P2.3 Loudness normalization. DONE — src/predictive_sfx/loudness.py (RMS target by intensity +
      true-peak ceiling) applied to mock + both neural generators; fixes intra-pool pumping AND the
      neural->authored seam; 7 unit tests.
- [x] P2.4 INVESTIGATED — the "shared seed -> identical variants" claim does NOT hold for SA3:
      torch.manual_seed(seed) then torch.randn([batch_size,...]) gives each batch row distinct noise
      (verified: 0 identical rows, max corr 0.32). No seed fix needed. Bake-time embedding-diversity
      check (CLAP/PANNs) deferred as a future enhancement (needs the judge panel).
- [x] P2.1 Fire-time key-validation. DONE — exact-key match means a mispredicted surface is a
      miss->authored fallback, never a wrong-content hit (test_mispredicted_surface...). Boundary
      biasing lives in the C# predictor's confidence (raycast fan), part of the C# port.
- [x] P2.2 p95 lead-time gate (folded into P3.3 submit_predictions).
- [x] P2.5 ETA-aware eviction + context grace. DONE — engine tracks per-key ETA + _now; eviction
      spares imminent keys; context change keeps imminent predicted clips across the transition but
      still flushes stale/unpredicted ones (dead-set preserved). Server passes now=ctx.time_s. 5 tests.

## Phase 3 — Make the prediction smart (the multi-model layer)

- [x] P3.1 Per-predictor confidence-calibration layer. DONE — calibration.py: binned reliability
      calibration + Laplace shrinkage toward global rate + per-source online hit-rate tracker;
      overconfident sources pulled to true frequency; sparse -> raw. 6 tests.
- [x] P3.2 Motion predictor A. DONE — Python KinematicGaitPredictor (cadence + lead + EMA
      heading-stability confidence + upcoming-surface override) AND C# port (PredictiveSfxMotionPredictor:
      raycast-ahead + cadence + SendPredictions). New PREDICT wire message: client.predict -> server
      fuses -> register_predictions (non-blocking) -> background loop generates off-lock. Proven by
      end-to-end predict->warm->cache-hit test over the wire. In-editor Play = P4.1.
- [x] P3.3 Fusion seam. DONE — fusion.py: PredictionFusion dedups by key (noisy-OR confidence),
      class-owner wins on ETA, calibrates each source's confidence, EV-ranks, min-EV admission gate.
      Bridged into the engine via submit_predictions(). 6 tests.
- [x] P2.2 (folded into P3.3) p95 lead-time gate. submit_predictions skips any proposal whose
      ETA*1000 < gen_cost_ms (can't finish in time -> route to fallback, don't waste the slot).
      Threshold = gen_cost_ms v1; live per-class p95 measurement is a refinement.
- [x] P3.4 Variant pools for high-rate classes. DONE — footstep/shoot/gunshot warm to deeper pools
      (high_rate_variants=8 vs 6); burst serves rotate through 8 distinct variants, never per-event
      live-gen. 4 tests.
- [ ] P3.5 (later) Learned intent predictor B for discrete events (shoot/reload/ability), top-k,
      noisy (F1 ~0.4); used to broaden speculation, gated by calibrated confidence.

## Phase 4 — Validation + benchmarks

- [~] P4.1 In-Editor Unity Play validation. PREPARED (needs human Play — can't drive the Unity GUI).
      C# done: PREDICT added to PredictiveSfxClient; SendPredictions on the service (off-main-thread);
      PredictiveSfxMotionPredictor (raycast-ahead + cadence + EMA confidence) auto-attached by
      PSFX -> Build Walk Environment. Server side proven by 119 tests incl end-to-end
      predict->warm->cache-hit over the wire. Procedure: docs/unity-verification.md.
- [x] P4.3 A/B showcase. Quantitative A/B in docs/benchmark-results.txt; in-game on/off documented
      in docs/unity-verification.md (toggle the MotionPredictor component).
- [x] P4.2 Core benchmarks (sim). DONE — benchmark.py + scripts/benchmark.py + docs/benchmark-results.txt:
      A/B (none 0% vs oracle/noisy ~100%), cache amortizes noise on recurring keys but noisy drops to
      63% on cold/diverse keys, and the lead-vs-latency sweep shows ready_in_time 0%->100% as lead
      crosses the 393ms SA3 latency. 5 tests. Frame-time impact needs the live Unity run -> P4.1.

## Phase 5 — Regroup before writing (NOT first)

- [ ] P5.1 After Phase 4 (system built + benchmarked), STOP and come back to discuss with the user
      before drafting anything: paper scope, framing, which results to feature, venue confirmation.
      The paper is the last step, not a near-term build. Do NOT auto-draft — decide direction
      together first. (When drafting does happen: AI-assistance disclosure; user owns + verifies all
      content; no fabricated results/citations.)

---

## The "more models?" question — resolved

- BUILD NOW (1): per-predictor confidence calibration (P3.1). Cheap, learnable from modest telemetry.
- ROADMAP / scope out (2): per-entity + spatial/occlusion model (multiplayer); learned
  auto-affordance model (kills hand-authored tables). Both real, both big -> future work.

## Inherent limitations (state honestly in the paper)

- Sub-latency reactive events (instant fire/parry/juke faster than one gen window) cannot be
  pre-gen'd reactively. We predict CONTEXT (a firefight is likely), not the millisecond press.
- Shared-GPU contention: gen vs render compete; throttle + bake-once mitigate, can't eliminate.
  Measure + report frame-time impact.
- Generation tail latency (p99) not fully controllable on a shared GPU.
- First-ever-occurrence cold miss; boundary material ambiguity; SA3-small ~76% quality ceiling.
- Coverage = union of predictor label spaces; OOD player/play-style degrades silently.
- Genre boundary: fits locomotion-heavy action/FPS/adventure; not rhythm/strategy/turn-based/VN.
- Affordance v1 is BYO-context (per-title tables); multiplayer needs new architecture.

Value proposition (honest): reliably anticipates context + deterministic kinematic events, degrades
gracefully to authored fallback elsewhere — holds where authored fallback is complete and the GPU
is not starved.
