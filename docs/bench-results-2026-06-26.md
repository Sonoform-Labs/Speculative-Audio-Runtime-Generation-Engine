# Generator benchmark — sa3 step sweep on RTX 3060 (2026-06-26)

Run: `scripts/bench_generators.py` (latency) + `scripts/bench_quality.py` (CLAP quality).
Hardware: RTX 3060. Model: Stable Audio 3 Small SFX (`sa3`). Samples saved in `docs/step-samples/`.

## Latency vs steps (per ~0.6s clip, on the 3060)
| steps | mean ms | p50 ms | p95 ms |
|---|---|---|---|
| 8 | 402.7 | 401.5 | 413.4 |
| 4 | 219.4 | 219.5 | 221.5 |
| 2 | 129.0 | 129.1 | 130.6 |
| 1 | 83.4  | 83.2  | 84.5  |

## Quality vs steps (CLAP judge, footstep clips over 4 materials × 5 variants)
| steps | footstep_rate | material_acc |
|---|---|---|
| 8 | 0.40 | 1.00 |
| 4 | 0.40 | 0.70 |
| 2 | 0.15 | 0.30 |
| 1 | 0.00 | 0.25 (= chance) |

## Conclusion
- The research's "few-step = real-time" holds for LATENCY (1-step = 83ms on the 3060, well under the
  393ms we had), BUT sa3 is NOT a true 1-step model: quality collapses below 4 steps (material
  discrimination falls to chance at 1-2 steps; footstep character dies at 1 step).
- **Sweet spot = 4 steps = ~219ms**: ~2x faster than the 8-step default, footstep character UNCHANGED
  (0.40 either way — that 0.40 is sa3's known footstep-character ceiling, not step-dependent),
  material accuracy still strong (0.70 vs chance 0.25). ADOPTED as the new default.
- To get true sub-100ms WITH quality you need an actually-distilled/consistency model
  (MeanAudio 120M / FlashAudio 1-step / ARC-distilled Stable Audio Open) — see N4. That requires a
  new install (dependency risk), so it's attempted in an isolated venv, not the main one.
- Caveat: CLAP material_acc is one proxy; confirm the 4-step default by EAR using the saved samples
  (`docs/step-samples/footstep_<material>_steps<N>.wav`). Revert to 8 steps if 4 sounds worse to you.

## open-small (`stableaudio`, the ARC-lineage model) — tested, DOMINATED
Latency on the 3060: 8 steps 894.7ms, 4 steps 572.5ms, 2 steps 409.8ms, 1 step 328.6ms.
Even 1-step open-small (328ms) is SLOWER than sa3 at 4 steps (219ms), and sa3 already beat open-small
on quality in prior sessions (material 100% vs 62% — the reason we swapped to sa3). So no installed
model beats sa3@4steps. Quality sweep skipped (latency already rules it out).

## N4 — new distilled models (MeanAudio etc.): feasible but needs an ISOLATED env (NOT done overnight)
MeanAudio (github.com/xiquan-li/MeanAudio, HF AndreasXi/MeanAudio): 120M, 1-step, RTF 0.013 on a 3090,
auto-downloads weights. PROMISING for true sub-100ms WITH quality. BLOCKER: its install targets
PyTorch CUDA 11.8, but this project's venv is torch 2.6 + CUDA 12.4 — installing it in the main venv
would clobber torch (the documented gotcha). So integration requires either (a) a SEPARATE generator
microservice in its own env that the sidecar calls, or (b) a torch downgrade (breaks sa3). Both are
the user's architecture call + need supervised install. Recommended supervised next step:
  1. `conda create -n meanaudio python=3.11` (isolated, won't touch the project venv)
  2. follow the repo README (torch cu118 + deps), let it auto-pull AndreasXi/MeanAudio + FLAN-T5 + CLAP
  3. bench latency+quality with the same harness ported in, compare to sa3@4steps (219ms)
  4. if it wins, wrap it as a separate process the sidecar talks to (keeps the main venv intact).
Other candidates (FlashAudio 1-step, ARC Stable Audio Open) similarly need their own env.

## MeanAudio 1-step measured on the 3060 (2026-06-27)
MeanAudio meanaudio_s, 1 step, 1.0s clip: latency mean=137.5ms p50=131.2ms (load+warmup 8.8s), sampling_rate=16000.
=> ~1.7x faster than sa3 @ 4 steps (219ms) AND holds quality at 1-step (sa3 collapses below 4 steps),
   BUT 16kHz + general-audio vs sa3 44.1kHz SFX-specialized. sa3 = quality tier; MeanAudio = speed tier
   (pluggable via RemoteGenerator). Isolated venv at D:\Python\_meanaudio_probe (cu124, no env damage).
