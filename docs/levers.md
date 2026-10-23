# Generic levers on the Flow upgrade (2026-10-23)

The request was "get above 0.9 on everything". This round ran every generic lever the diagnosis left open on the Flow upgrade path, picked one configuration by a rule written before its clips, and confirmed the pick on new seeds and on 32 scenarios nobody had rendered. The pick is best of four Flow candidates per take, chosen by Sony's Woosh-CLAP, and it holds on held-out data: across 119 held-out rows it moves the rows at or above 0.9 from 71 to 79 and the pooled floor by +0.018 [+0.004, +0.033]. That is a real gain and a small one. Forty held-out rows stay under 0.9: thirteen are capped by the judges themselves, fifteen ask the model for the wrong sound because the caption template takes an odd event word literally, and twelve sit near the line and move with the take. The strongest arm measured, a mixed pool across three prompt sources, reached 40 of 52 selection rows and was rejected because it turns grass footsteps into grass rustle.

Everything below ships behind `--woosh-upgrade-best N` (default 1, off). With it off the served audio is bit-identical to HEAD.

A second confirmation the same day (below, "Second held-out confirmation") added today's engine to the comparison and 42 more never-rendered scenarios. It holds the verdict on 165 held-out rows: best of four reaches 0.9 on 101 of them, the plain caption on Flow on 89 and today's engine on 48.

## How the round ran

Three rule files went down before the clips they govern, all in `data/_gate/lev10/eval/` with digests in `rules_sha1.txt`: rule 1 (levers, metric, decision) at 01:53 before any render, rule 2 (combinations) at 04:35 after the identity table and before the water table or any combination number, and rule 3 (confirmation) at 06:50 after the pick and before any held-out clip. Selection used the 75 known scenarios at variants nobody had rendered: 240 to 259 for every arm, 260 to 299 for the pools, 300 to 379 for the regression re-checks. Confirmation used 600 to 639 on the known scenarios and on FRESH3, plus FRESH4 (`data/_gate/lev10/eval/tools/fresh4.py`): 32 new scenarios, each with a validated judge type, 23 on Woosh, 9 on SA3 medium and 3 with a water stem.

Every clip went through the served path in process: Woosh Flow at midpoint 16 steps with the bf16 graph, `--amp full`, stereo and natural trim; the renderer reproduces the flowpath W clips of the same seeds bit for bit. The score is the identity floor of `docs/vocab-eval.md` with the judges flowpath W chose (every judge run of this round re-derived the same judges and real rates) and the water judge of `docs/water-judge.md`. Selection rows are the 45 judged known rows plus 7 water rows; held out there are 107 judged rows and 12 water rows. Intervals are 95% bootstraps over takes. A row reaches 0.9 when its point estimate is at least 0.9.

## The verifier

Woosh-CLAP pairs a PaSST audio tower (32 kHz) with a RoBERTa-large text tower in a 1024-d space. Its text tower is the Woosh text conditioner itself: all 393 tensors of `TextConditionerA` match `Woosh-CLAP/weights_text` bit for bit, so the worker takes text embeddings from the encoder it already holds and only the 348 MB audio tower is new. It scores the finished take (full length, channels averaged) against the request words, which are the caption template with "A single" and "recorded up close" removed, for example "gravel footstep sound". It never sees a judge output, the judge window, a reference clip or any real recording. Inside the worker the CPU path matches the bench path to 3.3e-7.

It is a weak but real signal. Its AUC for the all-judge hit was 0.667 at the median over 28 selection rows and 0.624 over 78 held-out rows, and near chance on keyboard typing (0.47) and grass steps (0.48). It sits outside the panel, yet it is a sibling twice over: contrastive audio-text training like the CLAP judge, and an AudioSet transformer like AST. The held-out gains do not run through either sibling. Per-judge hit-rate gains of best-of-4 were AST +0.050, PANNs +0.040 and CLAP +0.043; the floor recomputed without the CLAP judge rises from 0.766 to 0.795 (rows at 0.9: 43 to 50 of 75), and without AST from 0.764 to 0.781 (45 to 50 of 81).

## Selection

| arm (52 known rows) | rows at 0.9 | judged / water | pooled | pooled without hard-capped (rows at 0.9 of 46) | GPU s per new key | rows more than 0.10 under C (stage 1) |
|---|---|---|---|---|---|---|
| C: caption, cfg 4.5, 101 frames, 240-259 | 29 | 26 / 3 | 0.763 | 0.859 (29) | 2.8 | |
| C on 60 takes (240-299) | 31 | 28 / 3 | 0.762 | 0.858 (31) | | |
| G3, cfg 3 | 28 | 25 / 3 | 0.739 | 0.832 (28) | 2.8 | 6 |
| G6, cfg 6 | 27 | 24 / 3 | 0.770 | 0.866 (27) | 2.8 | 4 |
| G8, cfg 8 | 26 | 21 / 5 | 0.741 | 0.835 (26) | 2.8 | 10 |
| GS, cfg 7.5 falling to 1.5 | 24 | 21 / 3 | 0.736 | 0.829 (24) | 2.8 | 7 |
| F201, 201 frames | 31 | 26 / 5 | 0.768 | 0.863 (31) | 3.9 | 8 |
| F301, 301 frames | 34 | 29 / 5 | 0.812 | 0.916 (34) | 5.4 | 5 |
| R, grounded caption | 27 | 23 / 4 | 0.796 | 0.860 (26) | 2.8 | 10 |
| W, writer named style | 27 | 23 / 4 | 0.734 | 0.804 (26) | 2.8 | 14 |
| E, verifier picks one prompt per key | 35 | 30 / 5 | 0.831 | 0.926 (35) | 6.6 | 2 |
| B4, best of 4 | 32 | 29 / 3 | 0.804 | 0.909 (32) | 11.3 | 1 |
| B8, best of 8 | 33 | 28 / 5 | 0.818 | 0.925 (33) | 22.6 | 2 |
| EB4, E then best of 4 | 38 | 33 / 5 | 0.848 | 0.948 (38) | 15.1 | 2 |
| MX3 / MX6 / MX9, best of 3, 6, 9 across C, R and W | 38 / 38 / 40 | 33-35 / 5 | 0.861 / 0.865 / 0.861 | 0.951-0.956 | 8.5 / 17.0 / 25.4 | 1 |
| FB4 / FB8, best of 4 or 8 at 301 frames | 35 / 34 | 30, 29 / 5 | 0.808 / 0.800 | 0.911 / 0.904 | 21.4 / 42.8 | 5 |

GPU seconds count six takes per key at this round's measured render times (471 ms a Flow render, with the desktop holding about a quarter of the GPU); at the gate's 366 ms, B4 costs 8.8 s. The verifier adds 0.17 to 0.18 s of CPU per candidate, 4.3 s per key for B4.

The cfg the worker uses is already right. No value and no schedule beat 4.5: cfg 6 tied on the pooled floor and lost two rows, and the others lost both. The long window helped glass, slams and water steps and hurt punches: 301 frames gained five rows, then lost b_punch and g_punch_body by 0.145 and g_melee_hit_body by 0.18 on 60 takes, at 1.9 times the render cost, while 201 frames gained nothing. The prompt variants alone lose, and the reason they matter is that they fix different rows from the caption template: the grounded caption and the writer score 0.90 and 1.00 on the bottle shot, where the template scores 0.06, and 0.8 to 1.0 on the water steps, where it scores 0.48. Letting the verifier choose among them (E) or mix their takes (MX) turns that into the largest gains of the round. Best of N on the caption template is the steadiest lever: three rows at N = 4 and four at N = 8, with no large loss except on keyboard typing, where the verifier has no signal.

## The regression check and the pick

The decision rule ranks candidates by rows at or above 0.9, breaks ties on the pooled floor and rejects any candidate that drops a row more than 0.10 under C. Every candidate above B4 dropped a row at stage 1, so each flagged row was re-rendered with more takes (`stage2.txt`):

| candidate, in rank order | flagged row, re-measured | result |
|---|---|---|
| MX9, MX6, MX3, EB4, EB8, E | b_step_grass falls from 1.00 to 0.21-0.48 on 56 to 80 takes; E also loses g_playerfootstep_snow by 0.49 | rejected |
| FB4, FB8 | b_keyboard_typing -0.41, g_melee_hit_body -0.31, b_window_rifle -0.16 to -0.27 on 100 takes | rejected |
| F301 | b_punch and g_punch_body -0.145, g_melee_hit_body -0.18 on 60 takes | rejected |
| B8 | b_keyboard_typing -0.22 on 100 takes | rejected |
| B4 | b_keyboard_typing -0.085 on 100 takes | **passes, picked** |

The grass failure is the verifier's blind spot, and it is structural. On b_step_grass the verifier scores the grounded caption's takes highest (mean 0.175 against 0.143 for the template), while the judges hear a footstep in 53% of the template's takes and in 20% and 7% of the other two prompts' takes: the other prompts render grass rustle, and the verifier rewards the word "grass". So the rule's pick is B4, the plainest lever, and the six prompt-mixing arms with 38 to 40 rows stay on the table as the next thing to fix.

## Held-out confirmation (rule 3)

| held out | rows | C: rows at 0.9 | B4: rows at 0.9 | C pooled | B4 pooled | B4 minus C [95%] | B8: rows at 0.9 (context) |
|---|---|---|---|---|---|---|---|
| known, new seeds 600-639 | 52 | 30 | 35 | 0.770 | 0.797 | +0.028 [+0.008, +0.049] | 36 |
| FRESH3, new seeds | 32 | 26 | 27 | 0.904 | 0.912 | +0.008 [-0.006, +0.021] | 26 |
| FRESH4, never rendered | 35 | 15 | 17 | 0.662 | 0.675 | +0.013 [-0.021, +0.052] | 19 |
| all | 119 | 71 | 79 | 0.774 | 0.792 | +0.018 [+0.004, +0.033] | 81 |

By rule 3 the verdict is CONFIRMED: the lower bound over all held-out rows sits above zero and no set loses. Judged rows go from 63 to 70 of 107 and water rows from 8 to 9 of 12. Without the hard-capped rows B4 reaches 78 of 111, and without every row the diagnosis labelled judge-limited 74 of 101. Four rows lose more than 0.10: q_wade_step (0.44 to 0.19), q_beep_hatchback (0.33 to 0.18), f_jab (0.43 to 0.31) and the cannonball splash on the water judge (0.57 to 0.46); on the first three the verifier's AUC is under 0.5 (0.37, 0.40 and 0.47), so it pushes those rows the wrong way. The literal served takes (seed-order groups of four from 600, the verifier's top of each) give 56 Woosh rows at 0.9 against 57 by the expectation, so the expectation is honest. Variety holds: the six served takes of a key sit at a median pairwise log-mel distance of 0.824 against 0.814 for six unselected takes. Per-row numbers with intervals are in `confirm_B4.txt`.

## What caps the forty rows left

**The judges (13 rows).** The diagnosis judged real recordings of the requested sound that the reference sets do not hold. Real metal footsteps score 0.17 of real against these judges, so b_step_metal, g_footstep_metal, b_iron_boots_step and q_catwalk_step stay at 0.00 to 0.08 held out and under 0.3 in every selection arm; real rain scores 0.66 on the water judge, which caps both rain rows (0.01 and 0.03); real bottle breaks score 0.64 [0.20, 0.97], which caps the bottle shot. Wood steps (real 0.84), snow steps (0.88) and leaves (0.84) carry caps under the line, which covers b_step_wood, g_footstep_wood, f_tread_wood, q_sneak_step_wood, b_snow_step and f_jog_leaves. One label from the diagnosis did not hold: the cabinet slam (real 0.73) reached 0.97 held out, so generated slams can outscore real ones on these judges.

**The wording (15 rows).** The caption template puts the event word in front of "sound", and for odd game names the model renders what the word says. The judges' top labels show it: "water drop" comes back as Plop and dripping, "ruff" as a UI click, "pummel" as Bang and Explosion, "beep" as Beep and a UI click, "footfall" as a punch and Knock, "water cannonball splash" as Cap gun and a punch, "wine glass drop" as Clang, and "magnum fire" on SA3 as a grenade explosion to the CLAP pistol judge in all 20 takes, while AST and PANNs pass it. The known rows of this kind are the water steps, the melee hit, the glass shatter with its empty material slot and the jab. The model is able to make these sounds: Woosh's autoencoder keeps every type at 0.94 to 1.00 of real (diagnosis), and on the selection data the writer's prompts lift the water steps to 0.8 to 1.0, the melee hit to 0.87 and the glass shatter to 0.64. Best of N leaves a wrong prompt wrong, since all four candidates share it.

**The take and the shape (12 rows).** Nine rows sit under the line with intervals that reach it: g_car_door_slam_car (0.89 [0.72, 0.99]), b_window_rifle, g_glass_shatter_glass, b_keyboard_typing, f_splashdown_crate, q_blade_parry_rapier, q_scimitar_clash (0.896) and two SA3 rows, q_pen_click and q_rifle_crack. Three sit clearly lower: the belly flop (0.55 [0.18, 0.88]), whose takes decay far faster than real splashes (diagnosis: tail 244 against 559 ms), q_hook_punch (0.53) and the SA3 row q_pocket_pistol_shot (0.58), where PANNs hears fireworks in 11 of 20 takes, which puts it close to the wording class. The SA3 rows never touch the upgrade path. Best of 8 reaches 81 held-out rows against 79, at twice the cost.

## Runtime (`--woosh-upgrade-best N`, off by default)

With `--woosh-upgrade flow`, `--woosh-upgrade-best 4` makes each Flow upgrade of a held take render four candidates and keep the one the verifier scores highest. Candidate j of take v uses variant v + 4096 j, so j = 0 is exactly today's Flow take and the takes of a key stay distinct seeds. Each candidate renders under the GPU lock and yields to served renders between solver steps as before; scoring runs on four CPU threads outside the lock. A preempted take keeps its finished candidates and resumes from them on the retry (up to 32 takes held). The worker loads the audio tower (`WOOSH_HOME/checkpoints/Woosh-CLAP`, `--woosh-upgrade-verifier` to move it) once Flow is ready, refuses to with less than 0.6 GB of commit free, and falls back to one candidate if the load fails; the `SARGE_UPGRADE_FAIL` hook and the failure backoff work as before. The worker's upgrade lines reach `sidecar.log` only with the flag on, including one every ten takes with the picks per candidate and the verifier's time. `src/sarge/verify.py` holds the request text, the candidate seeds and the pick; `tests/test_upgrade_best.py` (9 tests, CPU) covers the pick, the seeds, the resume, the fallbacks, the commit guard, the fail hook, the log lines and the flags.

Under Unity's environment, on the same desktop load (1,314 MiB before launch), `upgrade_gate.py` ran today's recommended upgrade line and the same line with `--woosh-upgrade-best 4` (`data/_gate/lev10/upgrade/`):

| upgrade_gate | midpoint 16 | midpoint 16, best of 4 | bound |
|---|---|---|---|
| startup peak, MiB | 10,597 | 10,597 | 10,752 |
| READY, MiB | 10,099 | 10,099 | 9,830 (fails in both, see below) |
| peak with both Woosh models and gun renders, MiB | 11,433 | 11,456 | 11,571 |
| stress misses while upgrades run, guns / Woosh median (ms) | 191.5 / 59.6 | 197.4 / 64.1 | |
| Flow upgrade per take, median (ms) | 446 | 1,832 | |
| swaps in the run / preempted takes | 44 / 15 | 21 / 33 | |
| first hit to swap, median / p95 (s) | 12.5 / 28.6 | 78.9 / 97.2 | |
| commit free at the lowest, GB | 3.77 | 3.32 | |
| later plays served from Flow at revision 1 | 30 of 30 | 30 of 30 | |

Flow and the verifier were ready 6.2 s after the request, and the verifier scored at 172 ms a take; its picks spread over all four candidates (4, 4, 6 and 6 over twenty takes). One take rendered with a single candidate while the verifier was still loading, so the build now loads the verifier before Flow reports ready; a rerun on the final code (`gate_lev_b4b.json`) drew every take from four candidates (44 candidates for 11 takes) and repeated the other numbers within noise (peak 11,428 MiB, first hit to swap 82.3 s at the median, Woosh misses 66.9 ms). READY fails its bound in both runs for the reason recorded in `docs/hybrid.md`: the bound was set when the desktop held 1,163 MiB, and READY is the same with every flag off. Best of four costs no VRAM worth measuring (+23 MiB at the peak) and about 0.45 GB of commit. Its price is time: a key keeps its DFlow first-hit take for about 79 s at the median in this stress pattern, six times longer than today, because each take now needs four renders that served traffic keeps preempting. With every flag off, `e2e_hybrid.py --extra --learn` served the 75 clips of `baseline_warm_1` bit-identical twice, max abs 0.0: once before the load-order fix (`compare_lev10_off.json`, READY at 53.5 s) and once at 08:13 on the final runtime files, which last changed at 08:11 (`compare_lev10_off2.json`). The CPU suite passes 607 of 607.

The configuration these runs support, for anyone who wants the measured gain and accepts the slower swap:

```
--prompts template:caption+tt --woosh-upgrade flow --woosh-upgrade-solver midpoint:16 --woosh-upgrade-best 4
```

## Second held-out confirmation, against today's engine (push09)

The rule (`data/_gate/push09/eval/RULE_P9_confirm.txt`, with the FRESH5 list in `tools/fresh5.py`; both digests in `rules_sha1.txt`) went down at 08:30, before any clip of the round. The only render before it was a smoke test on three known rows at variants 220 and 221, where the round's renderer reproduced flowpath W's DFlow phrasebook clips and its Flow caption clips bit for bit (`smoke_cmp.txt`). Three arms were rendered on the same seeds: today's engine (the phrasebook on DFlow, and on SA3 medium for guns), the plain caption template on Flow at midpoint 16, and best of four over that caption pool. The data held out every row: the known, FRESH3 and FRESH4 scenarios at variants 700 to 739, which no run had touched, and FRESH5, 42 new scenarios that share no event name or render key with any earlier set and each have a validated judge type (six guns, three explosions, ten footsteps on ten surfaces, three glass breaks, three splashes, four doors, three dogs, one of them a hellhound, three punches, three clicks, two horns, two swords). The judge run chose the same judges and real rates as flowpath W, and the water judge matched lev10.

| held out | rows | today: rows at 0.9 | caption on Flow | best of 4 | pooled today / caption / best of 4 | best of 4 minus caption [95%] | best of 4 minus today [95%] |
|---|---|---|---|---|---|---|---|
| known, seeds 700-739 | 52 | 21 | 26 | 33 | 0.677 / 0.766 / 0.817 | +0.050 [+0.028, +0.078] | +0.140 [+0.055, +0.224] |
| FRESH3, new seeds | 32 | 9 | 25 | 25 | 0.589 / 0.926 / 0.931 | +0.004 [-0.018, +0.028] | +0.341 [+0.231, +0.453] |
| FRESH4, new seeds | 35 | 8 | 15 | 17 | 0.543 / 0.680 / 0.733 | +0.053 [+0.012, +0.098] | +0.190 [+0.068, +0.307] |
| FRESH5, never rendered | 46 | 10 | 23 | 26 | 0.574 / 0.689 / 0.719 | +0.030 [+0.002, +0.065] | +0.145 [+0.035, +0.257] |
| all | 165 | 48 | 89 | 101 | 0.603 / 0.757 / 0.794 | +0.036 [+0.022, +0.053] | +0.191 [+0.137, +0.246] |

Both verdicts are CONFIRMED by the rule. The verifier family does not carry the gain: per-judge hit gains over the caption pool were AST +0.060, PANNs +0.051 and CLAP +0.045, and with the CLAP judge left out best of four still beats the caption by +0.031 [+0.016, +0.047] (93 of 141 judged rows at 0.9 against 87), with AST left out by +0.032 [+0.015, +0.049]. The verifier's held-out AUC was 0.615 at the median over 109 rows. The literal served takes (seed-order groups of four, the verifier's top of each) put 76 Woosh rows at 0.9 against 78 by the expectation. Two costs showed up that lev10 did not see. Six rows fell more than 0.10 under the plain caption (f_tread_wood, f_jab, q_sneak_step_wood, q_wardrobe_open, v_stomp_stone, v_body_blow), and on all of them the verifier's AUC sits at 0.53 or under, so it pushes wood steps, jabs and body blows the wrong way. And the six served takes of a key came out slightly closer together than six unselected ones (median log-mel distance 0.828 against 0.873).

Sixty-four rows stay under 0.9, so "above 0.9 on everything" fails on this data. The rule's mechanical labels split them 18 judge-class, 19 wrong sound, 19 near the line and 8 model or shape, and the evidence reads as follows. Seven rows are capped by the judges outright: the five metal footsteps (b_step_metal, g_footstep_metal, b_iron_boots_step, q_catwalk_step, v_ladder_step_metal), where real metal steps score 0.17 of real and no arm scores above 0.10, and the two rain rows on the water judge (real rain 0.66). The other eleven judge-class rows (bottle and drinking-glass breaks, wood, snow, grass and leaves steps) carry soft caps from real recordings, but sibling rows of the same classes reach 0.9 here (b_step_grass 1.00, q_trudge_snow 1.00, v_plank_walk_wood 1.00, f_goblet_break and v_tumbler_smash 1.00), so the judges do credit those classes and the failing rows sit with the wording group. Today's engine is the evidence that the model can make many of the failing sounds: it beats best of four by more than 0.10 on 24 rows, among them the bottle shot (0.81 against 0.01), the wine glass drop (0.71 against 0.07), the stone stomp (0.58 against 0.04), the harbor fall (0.73 against 0.20), the snow crunch (0.75 against 0.28) and the lever-action rifle on SA3 (0.89 against 0.42). Those are prompt-wording failures of the caption template, and four candidates of one prompt all inherit them. Six wrong-sound labels rest on a PANNs top-1 of "Music" or an AST top-1 of "Silence", which also appear on correct takes, so treat b_keyboard_typing, f_motorcycle_horn, q_beep_hatchback, v_elbow_strike, v_jeep_horn and v_menu_click as weak labels. The model-or-shape group is mostly short hits: punches whose takes end within about 0.1 s (f_jab, q_hook_punch, v_body_blow), a pool jump, and three SA3 guns that lose to the phrasebook on these seeds. Seven rows that today's engine puts at 0.9 fall under it with the configuration: three SA3 rows (b_pistol_shot and g_shoot_pistol at 0.84, b_explosion at 0.87, all with intervals to 1.00), b_snow_step, q_sneak_step_wood, v_iron_gate_open and the cannonball splash on the water judge. Per row, with intervals and evidence: `data/_gate/push09/eval/confirm_p9.txt` and `table_compact.txt`.

The gates ran on the final runtime files (unchanged since 08:11) under Unity's environment, desktop at 1,320 MiB (`data/_gate/push09/eval/gate_result.txt`):

| | today's flags | plain upgrade | best of 4 | bound for best of 4 |
|---|---|---|---|---|
| startup peak, MiB | 10,602 | 10,586 | 10,482 | 10,752 |
| READY, MiB | 10,104 | 10,104 | 10,104 | today + 150 |
| peak after READY, MiB | 10,624 | 11,436 | 11,418 | 11,571 |
| first hits before Flow loads, guns / Woosh median (ms) | 179.3 / 57.8 | 183.4 / 57.7 | 183.8 / 57.4 | today + 5 |
| served misses while upgrades run, guns / Woosh median (ms) | 395.0 / 89.4 | 189.5 / 66.3 | 196.9 / 72.0 | plain upgrade + 10 |
| first hit to swap, median / p95 (s) | | 14.7 / 30.6 (35 keys) | 82.2 / 101.1 (20 keys) | |
| commit free at the lowest, GB | 4.64 | 4.05 | 2.86 | |

Every bound passed. The gun fill is +4.5 ms over today against a 5 ms bound, and the plain upgrade sits at +4.1 on the same code path, so that number is spread. In the scripted session with the game60 load (`data/_gate/ontime/push09/`) best of four moved the time from first hit to swap from 5.2 to 28.4 s at the median (9.2 to 39.4 s at p95), and by the second pass every Woosh play came from Flow in both runs. Two session numbers moved the wrong way with best of four, one run each: the Woosh first-play median went from 0.95 to 17.1 ms and the p95 over all plays from 67 to 121 ms, with the GPU busy 77% of the pass against 68%. Both runs also reached 11,571 MiB with the game60 load on the card, exactly the 11.3 GiB bound, so at today's desktop load the plain upgrade has no VRAM margin left in that session either. With every flag off, `e2e_hybrid.py` served the 75 clips of `baseline_warm_1` bit-identical (max abs 0.0, `compare_p9_off.json`), and the CPU suite passes 607 of 607.

The blind page `data/_gate/push09/page/index.html` (key in `data/_gate/push09/page_key.json`) plays 12 rows picked by the rule: today's engine against the configuration after its upgrades land, three takes each, the configuration's takes being the verifier's top of each group of four. The rule asked for the bottle shot twice, so its place passed to a second FRESH5 gun row. `tools/p9_page_check.py` matched all 72 takes to their source clips, checked that each served take is its group's verifier top, and passed the page script through `node --check`.

## What would move the rest

The judges come first, because nothing generic gets past a cap set by real recordings: per-surface footstep references from the FoleySet Walk recordings, a rain class for the water judge, and bottle and cabinet references, each validated on held-out real clips only (diagnosis lever 5). The wording comes second and is worth more rows: the mixed pool already lifts 38 to 40 of 52 selection rows, and it needs a verifier that rewards the event as well as the material, for example the minimum of the request-text score and an event-only score, measured on the existing judged takes before any new render and then confirmed on FRESH4. A small grounding step for odd event names ("ruff", "beep", "pummel", "drop", "magnum") belongs to the same work. Best of eight, the long window and the cfg sweep are spent levers under these judges.
