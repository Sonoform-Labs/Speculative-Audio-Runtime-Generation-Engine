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

## Round 1 detail
- cfg_scale=4 vs 1: footstep_seq 0→25%, material 56→62%, reward 0.51→0.60. Guidance helps.
- Elaborate prompts (foley / percussive / material-forward) all *hurt* material (31–44%) vs the
  plain `"a single footstep on {material}"` (62%). The small model prefers a terse prompt.
- secs=1 (vs the ~11s default) removed the ambient/"music" drift: not_music = 100% across configs.

## Lessons / guardrails
- Reward-hacking watch: not_music could be gamed by silence (cross-check quality/variation stay up);
  variation could be gamed by noise (cross-check footstep_seq/material). Keep raw axes visible.
- "Perfect" is bounded by the model; goal is to find the real ceiling, then bank the best config.
