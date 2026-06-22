# Reward-Loop Tuning Log

Critic-in-the-loop optimization of the Stable Audio Open Small generator against a composite
multi-model reward (`scripts/optimize.py`). Claude is the critic: read each round, design the next.

Composite reward = 0.30·footstep_seq (AST on the rhythmic sequence) + 0.25·material (CLAP) +
0.20·quality (Audiobox PQ) + 0.15·not_music (CLAP) + 0.10·variation.

| round | best config | reward | foot_seq | material | quality | notes |
|---|---|---|---|---|---|---|
| 1 | default_cfg4 (cfg=4, secs=1, simple prompt) | 0.604 | 25% | 62% | 70% | cfg=4 ≫ cfg=1; simple prompt beats foley/percussive; secs=1 fixed "music" |
| 2 | floor_cfg5 (0.538) — did NOT beat round 1 | 0.538 | 25% | 31% | 72% | cfg 5/6/7 all hurt material (62→50/31/38%); cfg≈4 is the sweet spot; plural/floor prompts no help |
| 3 | cfg4 confirmed optimum (PLATEAU) | 0.604 | 25% | 62% | 70% | fine sweep cfg3-4.5 + shorter gen + tighter window — nothing beat cfg4; config search tapped. Switching levers → negative prompting |
| 4 | cfg4_ref still champion | 0.604 | 25% | 62% | 70% | negative conditioning (no music/drone/speech) HURT material+footstep; banked cfg4/1s as generator default |
| 5 | **SA3-small-sfx swap** (cfg2) | 0.581 | 0% | **100%** | 76% | new model `stable-audio-3-small-sfx`: material 62→**100%**, quality 70→76%, ~393ms/clip (faster). Composite ~flat vs 0.604 ONLY due to coarse foot_seq (0%) + not_music dip (69%); per-axis SA3 clearly better. → round 6 recovers not_music |
| 6 | **SA3 cfg2 = optimum (ADOPTED)** | 0.581 | 0% | 100% | 76% | not_music/material tradeoff is FUNDAMENTAL (neg-prompt & cfg≥3 recover not_music but nuke material→25-56%). cfg2 is the SA3 sweet spot. **SA3 adopted** on per-axis truth, banked as default (cfg2). |

## Verdict (round 6): adopt Stable Audio 3 Small SFX

**Decision: SA3-small-sfx, cfg2, is the new generator** (banked as `StableAudio3Generator` default).
The call rests on a model-independent fact, not the composite: **material discrimination 62% → 100%**,
quality 70% → 76%, ~393 ms/clip (faster). Those are the axes that define a good SFX generator, and SA3
wins them decisively — fixing the exact weakness 4 rounds of tuning could not move on the old model.

Why the composite (0.581) is fractionally under open-small's 0.604 anyway — and why that does NOT change
the decision:
- **`foot_seq` is a dead metric here.** It's pinned at the model-class ceiling (both models score 0–25%
  on AudioSet "Walk, footsteps" for short synthetic clips), so its 0.30 weight injects noise, not signal.
- **not_music vs material is an intrinsic tradeoff** for SA3: pushing not_music up (higher cfg / negative
  prompt) collapses material to 25%. cfg2 is the joint optimum.

**Recommended reward recalibration (for review):** drop `foot_seq` weight (ceilinged non-discriminator),
raise `material`. Under e.g. {material .35, quality .25, not_music .15, foot_seq .15, variation .10},
SA3 ≈ 0.72 vs open-small ≈ 0.66 — i.e. the recalibrated composite agrees with the per-axis truth. Left as
a flagged change rather than silently re-scored, to avoid moving goalposts.

## FAD / realism vs real footsteps (kill-gate 2)
Reference: 40 ESC-50 "footsteps" clips (real-world recorded walking; CC BY-NC, eval-only). 24 generated
walk-sequences per model; PANNs embeddings; PCA-20 FAD + cosine-to-real-centroid (`scripts/fad.py`,
`scripts/fetch_real_footsteps.py`).

| model | FAD ↓ | cos-dist-to-real ↓ |
|---|---|---|
| real-vs-real (floor) | 30.7 | 0.000 |
| open-small | 78.1 | 0.239 |
| sa3-small-sfx | 87.3 | 0.244 |
| mock (synthetic) | 126.5 | 0.365 |

Honest read: (1) both neural models are FAR closer to real than synthetic mock — validates the neural
approach. (2) SA3 ≈ open-small on distance-to-real (78 vs 87 = within small-N noise); **FAD does not favor
SA3** — its edge is material distinctness (CLAP 62→100%), not recording-realism. (3) Generated dry SFX
stay distributionally distinguishable from real ESC-50 (~80 vs 31 floor), partly because ESC-50 carries
room/ambience our isolated clips lack. Small-N (24 gen / 40 real) FAD is indicative, not rigorous —
trust the relative ordering (neural ≪ mock).

## Round 5 — model upgrade (Stable Audio 3 Small SFX)
Swapped the generator from `stable-audio-open-small` (341M, music-trained) to the SFX-specialized
`stabilityai/stable-audio-3-small-sfx`. **Material discrimination jumped 62% → 100%** (CLAP) — the exact
weakness 4 rounds of prompt/cfg tuning could not move on the old model. Quality 70→76%, generation
~393ms/clip (faster, native 1s variable-length vs the old 11s+crop). Composite 0.581 is fractionally under
open-small's 0.604 purely because of foot_seq (0%, model-class ceiling for both) and a not_music dip
(69%). The per-axis truth: SA3 is the better SFX model. Round 6 tunes cfg to recover not_music while
holding material at 100% (projected composite ~0.63, beating open-small).

## Plateau (after round 4)
4 rounds, 4 lever types (prompt style, cfg sweep, gen-length/window, negative conditioning) — none beat
**cfg4 + simple prompt + 1s gen = reward 0.604** (material 62%, footstep_seq 25%, quality 70%, not-music
100%, varied 82%). This is the small model's ceiling on prompt/param/negative levers. Banked as the
`StableAudioGenerator` default. To go meaningfully higher the lever is a *more capable model* (Stable
Audio Open 1.0) — not more tuning. Lesser squeezes left: per-material configs, transient-shaping DSP.

## Round 1 detail
- cfg_scale=4 vs 1: footstep_seq 0→25%, material 56→62%, reward 0.51→0.60. Guidance helps.
- Elaborate prompts (foley / percussive / material-forward) all *hurt* material (31–44%) vs the
  plain `"a single footstep on {material}"` (62%). The small model prefers a terse prompt.
- secs=1 (vs the ~11s default) removed the ambient/"music" drift: not_music = 100% across configs.

## Lessons / guardrails
- Reward-hacking watch: not_music could be gamed by silence (cross-check quality/variation stay up);
  variation could be gamed by noise (cross-check footstep_seq/material). Keep raw axes visible.
- "Perfect" is bounded by the model; goal is to find the real ceiling, then bank the best config.
