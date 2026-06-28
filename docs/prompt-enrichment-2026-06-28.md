# Acoustic-signature prompt enrichment + per-event steps (2026-06-28)

User ear-test of Stage-0 samples surfaced two quality levers beyond step count:

1. **Prompt specificity is a big lever** — footstep and glass went from unrecognizable to clearly
   right just by enriching the prompt; step count was NOT the issue for those.
2. **Describe the SOUND, not the SCENARIO** — a "specific" watersplash prompt ("a heavy rock dropping
   into a pond") made the model render the *fall/impact*, losing the splash. The descriptor must name
   the acoustic signature (splashing water; the ringing tail of struck metal), not the cause.
3. **A few sounds genuinely need more steps** — watersplash only reads with the detail of ~8 steps.

## What shipped (both generic / generalizable, no per-game authoring)

**A. `build_prompt(event, material, intensity)` in `generators/stable_audio.py`** — event-class-aware
acoustic-signature enrichment, keyed on sound semantics shared across games:
- `footstep` -> "a {material-adj} footstep on {material}, a clear distinct footstep, ..."
- impact-class events (`hit/impact/strike/clang/shatter/break/smash/land`) on `metal` -> adds
  "a bright ringing resonant clang with a sustained metallic tail"; on `glass` -> "sharp shattering
  glass shards"; wood/stone/ceramic have their own tails.
- `splash`/`drip`/`pour` or material in {water,liquid,mud} -> "wet splashing water with droplets..."
- everything else -> material adjective + "{event} on {material}"; UNKNOWN event/material falls back
  to the plain template (= old behavior, no regression).
- Correctness check: a footstep on metal does NOT get the struck-pipe ring (only impact-class events
  do). Verified by ear + by inspecting the produced prompts.
- Override precedence unchanged: per-key `prompts` override > explicit `prompt_template` > build_prompt.

**B. Per-event step count in `StableAudio3Generator`** — `_steps_for(key)`: events/materials in
`detail_keys` (default {splash, water, liquid}) use `detail_steps` (8); everything else uses the
fast default (euler@3). Detail-heavy/tolerant sounds get texture; sharp transients stay fast. The set
is seeded on the ONE confirmed detail case (water) and is extensible as more are found.

## Status
- 151 tests pass (no regression).
- Wired prompts verified: hit|metal->ring, impact|glass->shatter, splash|water->splash & steps=8,
  footstep->material adj & NO ring, unknown->graceful fallback.
- Shipped-prompt ear-check WAVs: `docs/prompt-test/wired_*.wav`.
- Caveat: metal still lacks a perfect "ding" sustain and water is splash-but-generic — model limits of
  sa3-small; "good enough" per user. Disk cache (`.cache/sfx_bake`) keys by CacheKey, so clear it to
  pick up the new prompts on already-cached keys.
