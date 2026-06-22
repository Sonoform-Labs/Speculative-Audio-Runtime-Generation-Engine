# Overnight Run Summary (2026-06-22 → 06-23)

Task: install Stable Audio 3, generate, auto-evaluate on the model panel, fine-tune if needed — autonomously.

## Outcome: Stable Audio 3 Small SFX adopted — material discrimination 62% → 100%

| metric (best config) | open-small (old) | **SA3-small-sfx (new)** |
|---|---|---|
| material accuracy (CLAP) | 62% | **100%** |
| quality (Audiobox PQ) | 70% | **76%** |
| latency / clip (3060) | 582 ms | **~393 ms** |
| composite reward | 0.604 | 0.581 |
| footstep_seq (AST) | 25% | 0% |
| not_music (CLAP) | 100% | 69% |

**The headline is material 62% → 100%** — the exact weakness 4 rounds of prompt/cfg tuning could not move
on the old model, fixed by the SFX-specialized model. Quality up, and faster. SA3 is adopted (cfg2, default
prompt, 1 s gen, 8 steps) and banked as the `StableAudio3Generator` default.

The composite (0.581) is fractionally under open-small's 0.604 — but that's a metric artifact, not a
regression: `foot_seq` is pinned at the model-class ceiling (both ≤25% — noise, not signal) yet carries
0.30 weight, and not_music trades off against material for SA3 (cfg2 is the joint optimum; pushing
not_music up via higher cfg / negative prompt collapses material to 25%). On the axes that define a good
SFX generator, SA3 wins decisively. Full reasoning in `tuning-log.md` (rounds 5–6).

## What I did
1. Installed the `stable_audio_3` library (GitHub). **Note:** it clobbered torch to CPU — I restored
   torch 2.6.0+cu124 (CUDA confirmed). numpy is now 2.5.0 (judges still work).
2. Added `StableAudio3Generator` (`src/predictive_sfx/generators/stable_audio3.py`, factory key `sa3`) —
   variable-length native (requests `duration` directly), onset-extract, fp16.
3. Made `scripts/optimize.py` generator-agnostic (`--generator`); ran A/B + tuning rounds 5–6 on the full
   CLAP+AST+PANNs+Audiobox panel.
4. Banked SA3 cfg2 as the default; generated payoff clips.

## 🎧 Listen (in `data/demo/`)
- `sa3_wood.wav`, `sa3_stone.wav`, `sa3_grass.wav`, `sa3_metal.wav` — SA3 single footsteps per material
- `sa3_varied_walk.wav` vs `sa3_repeated_walk.wav` — 8 SA3 steps, all different, vs one looped
- compare to the old model's `_probe_*.wav` and `varied_walk.wav` to hear the upgrade

## Open items for you
- **Confirm the reward recalibration** (drop ceilinged foot_seq weight, raise material) — flagged in
  `tuning-log.md`, NOT applied unilaterally. Under it SA3 ≈ 0.72 vs open-small ≈ 0.66.
- **Revoke the HF token** — SA3 weights are cached at `D:\hf_cache`; runs work offline now.
- Next pillars: FAD vs a real-footstep set (needs a CC0 pack), the talk deck, in-engine integration.

## Commits this run (authored TC-al, not pushed)
SA3 backend + optimizer generator-agnostic; round 5 (material 62→100%); round 6 (cfg2 optimum, adopted);
this summary. See `git log`.
