# Push 09: the held-out table, rechecked

The goal was "above 0.9 on everything", and on held-out data it fails. Best of four Flow candidates picked by Woosh-CLAP (B4) puts 101 of 165 held-out rows at or above 0.9, against 89 for the plain caption template on Flow (C) and 48 for today's engine (H). Sixty-four rows stay under the line. Seven of them are capped by the judges themselves; the other 57 fail on wording, on take luck or on the shape of short hits, and no lever tried so far moves them generically. This page rebuilds the confirmation from the saved judge outputs with separate code, checks the integrity rules one by one, and lists what each remaining row would need.

A later round, p10 at the end of this page, built five generic levers on top of B4 (canonical names, loop wording, a prompt pool, an event-aware verifier score, a VRAM wait), kept none of them, and confirmed B4 again on new variants and a never-rendered FRESH6 set: 67 of 99 decision rows at or above 0.9, against 63 for the caption and 27 for today's engine.

Round p11, after it, cleaned file prefixes and take numbers out of event names, pooled the cleaned caption with a canonical-name caption and cut the Flow worker's VRAM. The pool reached 68 of 109 held-out decision rows against 54 for B4 and 27 for today's engine, yet six rows fell more than 0.10 below B4 with intervals wholly under zero, so the rule kept B4 in Unity. Name cleaning alone lowered the FRESH7 rows it touched by 0.063.

The metric is the identity floor of `docs/vocab-eval.md`: per row, the minimum over the type's validated judges of (hit rate on generated clips / hit rate on held-out real recordings), capped at 1, with the water judge on water rows. B4 is the exact expected served hit of the verifier's top pick from 4 random candidates of a 40-take pool (SA3 rows play the caption template on SA3 medium, so B4 equals C there). A row reaches 0.9 only when its held-out B4 point estimate is at least 0.9.

## What was rechecked

**The numbers.** `scratchpad/verify09/recompute.py` reads `vocab.json`, `water.json`, `verifier.json` and both `scenarios.json` files from `G:\sarge\push09` (byte-identical to the copies in `data/_gate/push09/eval/judge_outputs.zip`), picks the judges from the run's chosen set at 0.6 s, and recomputes every row from the per-take hits. It derives the best-of-N weights by enumerating all 91,390 four-take subsets of 40 and matches the closed form. All 165 rows and all four arms (H, C, B4, B8) agree with `confirm_p9.json` to 1e-9. The bootstrap intervals, drawn with a different seed, land within 0.032 of the round's at the 95th percentile (0.070 at worst), and the row-level differences carry the same signs and nearly the same bounds (B4 minus C over all rows +0.036 [+0.021, +0.053] here, +0.036 [+0.022, +0.053] in the round). One interval label moves at the margin: f_tread_wood's B4 upper bound is 0.8997 here and just over 0.9 in the round.

**The judges.** The run chose the same judges and real rates as flowpath W at 0.6 s for every type, and the same water judge (AST 0.575, CLAP 0.95, PANNs 0.725 real) as lev10. The per-row consensus judges equal the type's chosen set on all 149 judged rows.

**Rules before renders.** `RULE_P9_confirm.txt` still hashes to 6c64ab85 and `tools/fresh5.py` to e4d57ef8, both as recorded in `rules_sha1.txt`. The rule was written at 08:30:55. The earliest of the 17,280 Woosh clip and cut files was written at 08:31:37 and the earliest SA3 file at 09:08:24. The only earlier render of the round was a smoke test at 08:29 on three known rows at variants 220 and 221. The configuration itself comes from lev10's rule 3 (digest 20be042c, 06:50:18), written before lev10 rendered any confirmation clip, and push09 tunes nothing: N, the verifier, the request text, prompt, solver, cfg and window are those of lev10.

**Confirmation data never used in selection.** A scan of every render and scenario file and every clip folder under `G:\sarge`, `D:\Python\SAGE\data` and the session scratchpads (`scratchpad/verify09/scan.py`) found no clip at variants 700 to 739 and no FRESH5 render key or scenario name outside push09. The same scan pointed at FRESH4 and variants 600 to 639 finds 64 render-key hits and 6,739 clip files in lev10, so it does see earlier rounds. The four sets are held out to different degrees, and the cleanest is FRESH5. The known scenarios chose the configuration in lev10, so they are held out by seed only; FRESH3's old takes were read by the diagnosis; FRESH4 was rendered once by lev10's confirmation. FRESH5 alone gives B4 26 of 46 rows, C 23 and H 10, with B4 minus C +0.030 [+0.002, +0.065].

**The served verifier equals the scored one.** On the CPU, the worker's `Verifier` class (the audio tower loaded from `F:\AI\cache\woosh\checkpoints\Woosh-CLAP`, the text from the served Woosh text conditioner) reproduced 64 offline scores to 6.7e-7 and picked the same take in 16 of 16 groups of four; its weights file is byte-identical to the bench copy. The literal served takes (the verifier's top of each seed-order group of four) put 76 of 128 Woosh rows at 0.9 against 78 by the expectation, recomputed here.

**The page.** `data/_gate/push09/page/index.html` holds the 12 rows the rule asks for, in its order, when rebuilt from the recomputed table: the three judged Woosh rows with the lowest B4 minus H (b_bottle_shot_pistol, q_wine_glass_drop, v_stomp_stone), the three with the highest (g_car_horn_car, q_lorry_horn, q_wade_step), the two water rows with the largest gap, b_glass_shatter and b_step_wood, and two FRESH5 SA3 guns. All 72 takes match the rule's variants sample for sample: today at 700 to 702, the verifier's top of 700-703, 704-707 and 708-711, and SA3 caption takes at 700 to 702. One reading of the rule is a judgment call. The bottle shot was named twice, and the round passed its place to a second gun row (v_lever_action_fire); read as "the next in line of the same list", the place would go to v_fall_in_harbor, the fourth-lowest B4 minus H row.

**Tests and flags off.** The CPU suite passes 607 of 607. `startup_compare.py` against `baseline_warm_1`, rerun on the round's saved e2e output (written 10:57 to 10:58, after the last runtime edit at 08:11), reports 75 of 75 clips identical, max abs 0.0, PASS.

## How much of the gain could come from the verifier's relatives

Woosh-CLAP sits outside the judge panel, but it is a relative of two judges: its audio tower is PaSST, an AudioSet spectrogram transformer like the AST judge, and the model is a contrastive audio-text pair like the CLAP judge. Measured on the held-out caption pools, its per-take score tracks every judge weakly and about equally. Mean Spearman correlation between the verifier score and a judge's hit, over Woosh rows: AST 0.234 (69 rows), PANNs 0.185 (104), CLAP 0.172 (112); on the rows judged by all three families they sit at 0.234, 0.217 and 0.242 in the same order. The judges agree with each other more than that (AST with PANNs 0.378, AST with CLAP 0.335, PANNs with CLAP 0.225). Its AUC for a judge's hit is highest against AST (0.709 mean, against 0.656 for PANNs and 0.653 for CLAP), and the per-judge hit gain over C follows the same order (AST +0.060, PANNs +0.051, CLAP +0.045). So the lean toward its AudioSet sibling is real and small.

| floor computed over | judged rows | H ≥ 0.9 | C ≥ 0.9 | B4 ≥ 0.9 | B4 minus C [95%] |
|---|---|---|---|---|---|
| every validated judge | 149 | 34 | 79 | 88 | +0.035 [+0.019, +0.052] |
| without CLAP | 141 | 45 | 87 | 93 | +0.031 [+0.016, +0.047] |
| without AST | 149 | 34 | 81 | 93 | +0.032 [+0.016, +0.049] |
| without CLAP and AST (PANNs alone) | 141 | 51 | 93 | 101 | +0.026 [+0.011, +0.042] |

Dropping a judge loosens the floor for every arm, which is why all counts rise; the comparison that matters is the gain. With both relatives gone the gain keeps three quarters of its size, so at most about 0.01 of the pooled +0.035 can ride on sibling agreement. Eight door-open rows leave the table without CLAP because CLAP is their only validated judge.

## Per set

| held out | rows | H ≥ 0.9 | C ≥ 0.9 | B4 ≥ 0.9 | pooled H / C / B4 | B4 minus C [95%] |
|---|---|---|---|---|---|---|
| known, seeds 700-739 | 52 | 21 | 26 | 33 | 0.677 / 0.766 / 0.817 | +0.050 [+0.028, +0.078] |
| FRESH3, new seeds | 32 | 9 | 25 | 25 | 0.589 / 0.926 / 0.931 | +0.004 [-0.018, +0.028] |
| FRESH4, new seeds | 35 | 8 | 15 | 17 | 0.543 / 0.680 / 0.733 | +0.053 [+0.012, +0.098] |
| FRESH5, never rendered | 46 | 10 | 23 | 26 | 0.574 / 0.689 / 0.719 | +0.030 [+0.002, +0.065] |
| all | 165 | 48 | 89 | 101 | 0.603 / 0.757 / 0.794 | +0.036 [+0.022, +0.053] |

The 165 rows are 154 distinct clip and judge pairs, because eleven known pairs share a clip (b_step_stone and g_footstep_stone, for example); the 101 rows at 0.9 are 94 distinct ones. Two rows sit on the line: q_windshield_shatter at 0.9015 counts, q_wardrobe_open at 0.8993 does not. By type, B4 reaches 0.9 on every door slam (8 of 8), 8 of 9 explosions, 7 of 8 swords, 6 of 8 door opens, 6 of 8 rifles, 13 of 16 water-judge rows, 11 of 15 glass breaks and 7 of 9 dogs; the weak types are footsteps (13 of 35), punches (3 of 10), splashes (3 of 8) and pistols on SA3 (3 of 9).

## Every row

Intervals are the round's 95% bootstraps on B4 (1,000 resampled pools). "Today H" is the phrasebook on DFlow (SA3 medium for SA3 rows), "caption C" the caption template on Flow at midpoint 16. The cause column is the rule's label read against the evidence; "today's engine" marks rows where H beats B4 by more than 0.10.

| row | set | type | today H | caption C | best of 4 [95%] | ≥ 0.9 | cause if under | what it needs |
|---|---|---|---|---|---|---|---|---|
| b_iron_boots_step | known | footstep | 0.04 | 0.00 | 0.00 [0.00, 0.00] | no | judge cap: real metal 0.17 | a judge that credits real metal steps (validated on held-out real clips) |
| b_step_metal | known | footstep | 0.07 | 0.00 | 0.00 [0.00, 0.00] | no | judge cap: real metal 0.17 | a judge that credits real metal steps (validated on held-out real clips) |
| g_footstep_metal | known | footstep | 0.07 | 0.00 | 0.00 [0.00, 0.00] | no | judge cap: real metal 0.17 | a judge that credits real metal steps (validated on held-out real clips) |
| w:b_rain_tin_roof | known | water | 0.22 | 0.00 | 0.00 [0.00, 0.00] | no | judge cap: real rain 0.66 | a judge that credits real rain (validated on held-out real clips) |
| w:g_rain_on_roof_metal | known | water | 0.00 | 0.04 | 0.00 [0.00, 0.00] | no | judge cap: real rain 0.66 | a judge that credits real rain (validated on held-out real clips) |
| b_bottle_shot_pistol | known | glass_break | 0.81 | 0.06 | 0.01 [0.00, 0.04] | no | wrong sound (Ding, all-judge hit 0.05); soft cap bottle 0.64; today's engine 0.81 | event-aware generic wording or a verifier that scores the event |
| b_snow_step | known | footstep | 0.92 | 0.48 | 0.52 [0.24, 0.84] | no | soft cap snow 0.88, siblings reach 0.9; today's engine 0.92 | event-aware generic wording or a verifier that scores the event |
| g_glass_shatter_glass | known | glass_break | 0.74 | 0.35 | 0.60 [0.28, 0.91] | no | wrong sound (Coin (dropping), all-judge hit 0.28); today's engine 0.74; take, interval to 0.91 | event-aware generic wording or a verifier that scores the event |
| b_glass_shatter | known | glass_break | 0.71 | 0.32 | 0.66 [0.29, 0.94] | no | soft cap drinking glass 0.47, siblings reach 0.9; take, interval to 0.94 | a sharper verifier (V AUC 0.77) |
| g_melee_hit_body | known | punch | 0.31 | 0.66 | 0.70 [0.33, 0.93] | no | take, interval to 0.93 | a verifier that ranks this row the right way (V AUC 0.48) |
| b_step_water | known | footstep | 0.51 | 0.61 | 0.76 [0.41, 1.00] | no | take, interval to 1.00 | a sharper verifier (V AUC 0.73) |
| g_footstep_water | known | footstep | 0.51 | 0.61 | 0.76 [0.42, 1.00] | no | take, interval to 1.00 | a sharper verifier (V AUC 0.73) |
| g_keyboard_click_plastic | known | ui_click (SA3) | 0.35 | 0.77 | 0.77 [0.49, 1.00] | no | take, interval to 1.00 | more takes or a verifier with signal on SA3 output |
| b_keyboard_typing | known | ui_click | 0.45 | 0.87 | 0.80 [0.47, 1.00] | no | weak label (Music); take, interval to 1.00 | a verifier that ranks this row the right way (V AUC 0.41) |
| b_step_wood | known | footstep | 0.71 | 0.67 | 0.80 [0.38, 1.00] | no | soft cap wood 0.84, siblings reach 0.9; take, interval to 1.00 | a sharper verifier (V AUC 0.60) |
| g_footstep_wood | known | footstep | 0.71 | 0.67 | 0.80 [0.41, 1.00] | no | soft cap wood 0.84, siblings reach 0.9; take, interval to 1.00 | a sharper verifier (V AUC 0.60) |
| b_pistol_shot | known | gunshot_pistol (SA3) | 1.00 | 0.84 | 0.84 [0.63, 1.00] | no | today's engine 1.00; take, interval to 1.00 | a better generic SA3 wording plus a verifier with signal on SA3 output |
| g_shoot_pistol | known | gunshot_pistol (SA3) | 1.00 | 0.84 | 0.84 [0.63, 1.00] | no | today's engine 1.00; take, interval to 1.00 | a better generic SA3 wording plus a verifier with signal on SA3 output |
| b_explosion | known | explosion (SA3) | 1.00 | 0.87 | 0.87 [0.60, 1.00] | no | today's engine 1.00; take, interval to 1.00 | a better generic SA3 wording plus a verifier with signal on SA3 output |
| b_car_horn | known | horn | 0.17 | 0.80 | 0.93 [0.59, 1.00] | yes |  |  |
| g_car_door_slam_car | known | door_slam | 0.33 | 0.71 | 0.94 [0.72, 1.00] | yes |  |  |
| b_punch | known | punch | 0.70 | 0.95 | 0.95 [0.76, 1.00] | yes |  |  |
| g_punch_body | known | punch | 0.70 | 0.95 | 0.95 [0.76, 1.00] | yes |  |  |
| b_car_door_slam | known | door_slam | 0.85 | 0.95 | 0.99 [0.94, 1.00] | yes |  |  |
| b_step_stone | known | footstep | 0.36 | 0.71 | 0.99 [0.55, 1.00] | yes |  |  |
| g_footstep_stone | known | footstep | 0.36 | 0.71 | 0.99 [0.54, 1.00] | yes |  |  |
| b_dog_bark | known | dog_bark | 0.26 | 0.95 | 1.00 [1.00, 1.00] | yes |  |  |
| b_door_creak | known | door_open | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| b_rifle_shot | known | gunshot_rifle (SA3) | 0.94 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| b_step_grass | known | footstep | 0.44 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| b_step_gravel | known | footstep | 1.00 | 1.00 | 1.00 [0.97, 1.00] | yes |  |  |
| b_sword_clash | known | sword | 0.98 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| b_window_rifle | known | glass_break | 0.74 | 0.97 | 1.00 [1.00, 1.00] | yes |  |  |
| g_break_window | known | glass_break | 0.89 | 0.97 | 1.00 [0.97, 1.00] | yes |  |  |
| g_bullet_impact_glass | known | glass_break | 1.00 | 1.00 | 1.00 [0.82, 1.00] | yes |  |  |
| g_car_horn_car | known | horn | 0.00 | 0.86 | 1.00 [0.97, 1.00] | yes |  |  |
| g_dog_bark_dog | known | dog_bark | 0.88 | 0.97 | 1.00 [1.00, 1.00] | yes |  |  |
| g_door_creak_wood | known | door_open | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| g_explosion_large_debris | known | explosion (SA3) | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| g_footstep_gravel | known | footstep | 1.00 | 1.00 | 1.00 [0.98, 1.00] | yes |  |  |
| g_footsteps_concrete_concrete | known | footstep | 0.54 | 1.00 | 1.00 [0.85, 1.00] | yes |  |  |
| g_pistol_shot_9mm | known | gunshot_pistol (SA3) | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| g_playerfootstep_snow | known | footstep | 0.71 | 1.00 | 1.00 [0.78, 1.00] | yes |  |  |
| g_shoot_rifle | known | gunshot_rifle (SA3) | 0.94 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| g_shotgun_fire_shotgun | known | gunshot_shotgun (SA3) | 0.90 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| g_splash_water | known | splash | 0.47 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| g_sword_clash_steel | known | sword | 0.98 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| w:b_drip_puddle | known | water | 1.00 | 0.83 | 1.00 [0.76, 1.00] | yes |  |  |
| w:b_step_water | known | water | 0.97 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| w:g_footstep_water | known | water | 0.97 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| w:g_splash_water | known | water | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| w:g_water_drip_water | known | water | 1.00 | 0.83 | 1.00 [0.76, 1.00] | yes |  |  |
| f_jab | fresh3 | punch | 0.27 | 0.58 | 0.40 [0.13, 0.75] | no | short or shaped takes (t40 0.11 s) | fuller hit tails from the model; no tested generic lever gave them |
| f_jog_leaves | fresh3 | footstep | 0.54 | 0.40 | 0.49 [0.11, 0.78] | no | wrong sound (Crushing, all-judge hit 0.10); soft cap leaves 0.84 | event-aware generic wording or a verifier that scores the event |
| f_tread_wood | fresh3 | footstep | 0.13 | 0.75 | 0.62 [0.30, 0.92] | no | soft cap wood 0.84, siblings reach 0.9; take, interval to 0.92 | a verifier that ranks this row the right way (V AUC 0.39) |
| f_sidearm_fire | fresh3 | gunshot_pistol (SA3) | 0.22 | 0.77 | 0.77 [0.55, 0.98] | no | take, interval to 0.98 | more takes or a verifier with signal on SA3 output |
| f_belly_flop | fresh3 | splash | 0.65 | 0.77 | 0.85 [0.48, 1.00] | no | take, interval to 1.00 | a sharper verifier (V AUC 0.61) |
| f_npc_walk_stone | fresh3 | footstep | 0.67 | 0.94 | 0.87 [0.49, 1.00] | no | take, interval to 1.00 | a sharper verifier (V AUC 0.52) |
| f_motorcycle_horn | fresh3 | horn | 0.09 | 0.91 | 0.89 [0.65, 1.00] | no | weak label (Music); take, interval to 1.00 | a sharper verifier (V AUC 0.57) |
| f_fist_strike | fresh3 | punch | 0.16 | 0.92 | 0.96 [0.58, 1.00] | yes |  |  |
| f_yap_terrier | fresh3 | dog_bark | 0.12 | 0.97 | 0.97 [0.88, 1.00] | yes |  |  |
| f_cabinet_slam | fresh3 | door_slam | 0.35 | 0.88 | 0.97 [0.90, 1.00] | yes |  |  |
| f_slam_shut | fresh3 | door_slam | 0.88 | 0.97 | 1.00 [0.98, 1.00] | yes |  |  |
| f_artillery_explosion | fresh3 | explosion (SA3) | 1.00 | 1.00 | 1.00 [0.80, 1.00] | yes |  |  |
| f_assault_rifle_shot | fresh3 | gunshot_rifle (SA3) | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| f_buckshot_fire | fresh3 | gunshot_shotgun (SA3) | 0.86 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| f_demolition_blast | fresh3 | explosion (SA3) | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| f_gate_open | fresh3 | door_open | 0.91 | 1.00 | 1.00 [0.96, 1.00] | yes |  |  |
| f_goblet_break | fresh3 | glass_break | 1.00 | 1.00 | 1.00 [0.96, 1.00] | yes |  |  |
| f_hatch_open | fresh3 | door_open | 0.74 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| f_horn_beep_cab | fresh3 | horn | 0.23 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| f_hound_bark | fresh3 | dog_bark | 0.15 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| f_marksman_fire | fresh3 | gunshot_rifle (SA3) | 0.89 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| f_pane_smash | fresh3 | glass_break | 0.98 | 1.00 | 1.00 [0.82, 1.00] | yes |  |  |
| f_pistol_gunshot | fresh3 | gunshot_pistol (SA3) | 0.47 | 1.00 | 1.00 [0.89, 1.00] | yes |  |  |
| f_remote_press | fresh3 | ui_click | 0.10 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| f_saber_clash | fresh3 | sword | 0.76 | 0.99 | 1.00 [0.81, 1.00] | yes |  |  |
| f_splashdown_crate | fresh3 | splash | 0.52 | 0.77 | 1.00 [0.70, 1.00] | yes |  |  |
| f_sprint_asphalt | fresh3 | footstep | 0.58 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| f_swordfight_hit | fresh3 | sword | 0.93 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| f_toggle_click | fresh3 | ui_click (SA3) | 0.07 | 1.00 | 1.00 [0.91, 1.00] | yes |  |  |
| f_vase_break | fresh3 | glass_break | 0.57 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| w:f_belly_flop | fresh3 | water | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| w:f_splashdown_crate | fresh3 | water | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| q_magnum_fire | fresh4 | gunshot_pistol (SA3) | 0.05 | 0.00 | 0.00 [0.00, 0.00] | no | wrong sound (Music, all-judge hit 0.00) | a better generic SA3 wording plus a verifier with signal on SA3 output |
| q_catwalk_step | fresh4 | footstep | 0.10 | 0.07 | 0.01 [0.00, 0.06] | no | judge cap: real metal 0.17 | a judge that credits real metal steps (validated on held-out real clips) |
| q_ruff_retriever | fresh4 | dog_bark | 0.03 | 0.03 | 0.02 [0.00, 0.07] | no | wrong sound (Music, all-judge hit 0.00) | event-aware generic wording or a verifier that scores the event |
| q_wine_glass_drop | fresh4 | glass_break | 0.71 | 0.03 | 0.07 [0.00, 0.24] | no | wrong sound (Clang, all-judge hit 0.03); soft cap drinking glass 0.47, siblings reach 0.9; today's engine 0.71 | event-aware generic wording or a verifier that scores the event |
| q_beep_hatchback | fresh4 | horn | 0.00 | 0.29 | 0.21 [0.03, 0.46] | no | weak label (Music) | a verifier that ranks this row the right way (V AUC 0.49) |
| q_drop_in_lake | fresh4 | splash | 0.52 | 0.09 | 0.26 [0.00, 0.48] | no | wrong sound (Sound effect, all-judge hit 0.00); today's engine 0.52 | event-aware generic wording or a verifier that scores the event |
| q_pummel_thug | fresh4 | punch | 0.00 | 0.11 | 0.28 [0.00, 0.38] | no | wrong sound (Bang, all-judge hit 0.00) | event-aware generic wording or a verifier that scores the event |
| q_cannonball_splash | fresh4 | splash | 0.65 | 0.13 | 0.37 [0.00, 0.59] | no | wrong sound (Bang, all-judge hit 0.03); today's engine 0.65 | event-aware generic wording or a verifier that scores the event |
| q_hook_punch | fresh4 | punch | 0.39 | 0.62 | 0.60 [0.28, 0.88] | no | short or shaped takes (t40 0.12 s) | fuller hit tails from the model; no tested generic lever gave them |
| q_footfall_concrete | fresh4 | footstep | 0.40 | 0.31 | 0.61 [0.18, 1.00] | no | take, interval to 1.00 | a sharper verifier (V AUC 0.81) |
| q_sneak_step_wood | fresh4 | footstep | 0.95 | 0.85 | 0.64 [0.29, 0.87] | no | soft cap wood 0.84, siblings reach 0.9; today's engine 0.95 | event-aware generic wording or a verifier that scores the event |
| q_checkbox_click | fresh4 | ui_click (SA3) | 0.28 | 0.70 | 0.70 [0.42, 0.97] | no | take, interval to 0.97 | more takes or a verifier with signal on SA3 output |
| w:q_cannonball_splash | fresh4 | water | 1.00 | 0.48 | 0.71 [0.27, 1.00] | no | today's engine 1.00; take, interval to 1.00 | event-aware generic wording or a verifier that scores the event |
| q_pocket_pistol_shot | fresh4 | gunshot_pistol (SA3) | 0.68 | 0.84 | 0.84 [0.63, 1.00] | no | wrong sound (Firecracker, all-judge hit 0.75); take, interval to 1.00 | a better generic SA3 wording plus a verifier with signal on SA3 output |
| q_blade_parry_rapier | fresh4 | sword | 0.00 | 0.55 | 0.86 [0.60, 1.00] | no | take, interval to 1.00 | a sharper verifier (V AUC 0.80) |
| q_bulldog_bark | fresh4 | dog_bark | 0.17 | 0.85 | 0.86 [0.67, 1.00] | no | take, interval to 1.00 | a sharper verifier (V AUC 0.69) |
| q_rifle_crack | fresh4 | gunshot_rifle (SA3) | 0.21 | 0.89 | 0.89 [0.65, 0.99] | no | take, interval to 0.99 | more takes or a verifier with signal on SA3 output |
| q_wardrobe_open | fresh4 | door_open | 0.85 | 1.00 | 0.90 [0.63, 1.00] | no | take, interval to 1.00 | a verifier that ranks this row the right way (V AUC 0.32) |
| q_windshield_shatter | fresh4 | glass_break | 0.86 | 0.90 | 0.90 [0.61, 1.00] | yes |  |  |
| q_pen_click | fresh4 | ui_click (SA3) | 0.35 | 0.97 | 0.97 [0.74, 1.00] | yes |  |  |
| q_screen_door_slam | fresh4 | door_slam | 0.60 | 0.97 | 0.98 [0.92, 1.00] | yes |  |  |
| q_bang_shut_office | fresh4 | door_slam | 0.85 | 0.82 | 0.99 [0.95, 1.00] | yes |  |  |
| q_wade_step | fresh4 | footstep | 0.03 | 0.61 | 0.99 [0.69, 1.00] | yes |  |  |
| q_battle_rifle_fire | fresh4 | gunshot_rifle (SA3) | 0.91 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| q_creak_open_shed | fresh4 | door_open | 0.88 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| q_grenade_blast | fresh4 | explosion (SA3) | 0.93 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| q_lorry_horn | fresh4 | horn | 0.03 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| q_mine_explosion | fresh4 | explosion (SA3) | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| q_pump_shotgun_fire | fresh4 | gunshot_shotgun (SA3) | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| q_scimitar_clash | fresh4 | sword | 0.52 | 1.00 | 1.00 [0.81, 1.00] | yes |  |  |
| q_smash_storefront | fresh4 | glass_break | 0.55 | 1.00 | 1.00 [0.95, 1.00] | yes |  |  |
| q_trudge_snow | fresh4 | footstep | 0.71 | 0.98 | 1.00 [0.82, 1.00] | yes |  |  |
| q_walk_cobblestone | fresh4 | footstep | 0.78 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| w:q_drop_in_lake | fresh4 | water | 1.00 | 0.70 | 1.00 [0.60, 1.00] | yes |  |  |
| w:q_wade_step | fresh4 | water | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| v_haymaker_brawler | fresh5 | punch | 0.19 | 0.00 | 0.00 [0.00, 0.00] | no | wrong sound (Music, all-judge hit 0.00); today's engine 0.19 | event-aware generic wording or a verifier that scores the event |
| v_ladder_step_metal | fresh5 | footstep | 0.00 | 0.00 | 0.00 [0.00, 0.00] | no | judge cap: real metal 0.17 | a judge that credits real metal steps (validated on held-out real clips) |
| v_patrol_step_concrete | fresh5 | footstep | 0.00 | 0.00 | 0.00 [0.00, 0.00] | no | wrong sound (Thunk, all-judge hit 0.00) | event-aware generic wording or a verifier that scores the event |
| v_stomp_stone | fresh5 | footstep | 0.58 | 0.18 | 0.04 [0.00, 0.16] | no | wrong sound (Music, all-judge hit 0.05); today's engine 0.58 | event-aware generic wording or a verifier that scores the event |
| v_lawn_step_grass | fresh5 | footstep | 0.31 | 0.09 | 0.07 [0.00, 0.24] | no | wrong sound (Crack, all-judge hit 0.05); soft cap grass 0.60, siblings reach 0.9; today's engine 0.31 | event-aware generic wording or a verifier that scores the event |
| v_body_blow | fresh5 | punch | 0.27 | 0.31 | 0.14 [0.02, 0.30] | no | today's engine 0.27; short or shaped takes (t40 0.21 s) | event-aware generic wording or a verifier that scores the event |
| v_fall_in_harbor | fresh5 | splash | 0.73 | 0.17 | 0.20 [0.00, 0.31] | no | wrong sound (Water, all-judge hit 0.03); today's engine 0.73 | event-aware generic wording or a verifier that scores the event |
| v_light_switch_click | fresh5 | ui_click (SA3) | 0.35 | 0.21 | 0.21 [0.00, 0.42] | no | wrong sound (Finger snapping, all-judge hit 0.15); today's engine 0.35 | a better generic SA3 wording plus a verifier with signal on SA3 output |
| v_pool_jump | fresh5 | splash | 0.43 | 0.26 | 0.21 [0.02, 0.52] | no | today's engine 0.43; short or shaped takes (t40 0.45 s) | event-aware generic wording or a verifier that scores the event |
| v_tiptoe_tile | fresh5 | footstep | 0.58 | 0.31 | 0.27 [0.02, 0.40] | no | wrong sound (Music, all-judge hit 0.05); today's engine 0.58 | event-aware generic wording or a verifier that scores the event |
| v_crunch_step_snow | fresh5 | footstep | 0.75 | 0.20 | 0.28 [0.04, 0.57] | no | wrong sound (Music, all-judge hit 0.10); soft cap snow 0.88, siblings reach 0.9; today's engine 0.75 | event-aware generic wording or a verifier that scores the event |
| v_elbow_strike | fresh5 | punch | 0.00 | 0.23 | 0.37 [0.09, 0.68] | no | weak label (Music) | a sharper verifier (V AUC 0.74) |
| v_lever_action_fire | fresh5 | gunshot_rifle (SA3) | 0.89 | 0.42 | 0.42 [0.18, 0.66] | no | today's engine 0.89; SA3 caption wording loses to the phrasebook | a better generic SA3 wording plus a verifier with signal on SA3 output |
| v_double_barrel_fire | fresh5 | gunshot_shotgun (SA3) | 0.86 | 0.61 | 0.61 [0.41, 0.81] | no | today's engine 0.86; SA3 caption wording loses to the phrasebook | a better generic SA3 wording plus a verifier with signal on SA3 output |
| v_iron_gate_open | fresh5 | door_open | 0.91 | 0.59 | 0.63 [0.35, 0.91] | no | today's engine 0.91; take, interval to 0.91 | event-aware generic wording or a verifier that scores the event |
| v_beach_walk_sand | fresh5 | footstep | 0.40 | 0.31 | 0.65 [0.25, 0.85] | no | short or shaped takes (t40 0.48 s) | fuller hit tails from the model; no tested generic lever gave them |
| v_derringer_shot | fresh5 | gunshot_pistol (SA3) | 0.71 | 0.66 | 0.66 [0.44, 0.88] | no | SA3 takes short of the line | a better generic SA3 wording plus a verifier with signal on SA3 output |
| v_jeep_horn | fresh5 | horn | 0.17 | 0.54 | 0.79 [0.51, 1.00] | no | weak label (Music); take, interval to 1.00 | a sharper verifier (V AUC 0.82) |
| v_menu_click | fresh5 | ui_click (SA3) | 0.28 | 0.80 | 0.80 [0.57, 1.00] | no | weak label (Silence); take, interval to 1.00 | more takes or a verifier with signal on SA3 output |
| v_hike_step_dirt | fresh5 | footstep | 0.72 | 0.85 | 0.89 [0.56, 1.00] | no | take, interval to 1.00 | a sharper verifier (V AUC 0.56) |
| v_puddle_stomp | fresh5 | footstep | 0.10 | 0.85 | 0.93 [0.53, 1.00] | yes |  |  |
| v_door_bang | fresh5 | door_slam | 0.07 | 0.95 | 0.95 [0.85, 1.00] | yes |  |  |
| v_hellhound_bark | fresh5 | dog_bark | 0.62 | 0.78 | 0.99 [0.95, 1.00] | yes |  |  |
| v_watchdog_bark | fresh5 | dog_bark | 0.82 | 0.97 | 1.00 [1.00, 1.00] | yes |  |  |
| v_apartment_door_slam | fresh5 | door_slam | 0.64 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| v_beagle_bark | fresh5 | dog_bark | 0.42 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| v_blade_strike | fresh5 | sword | 0.20 | 0.49 | 1.00 [0.83, 1.00] | yes |  |  |
| v_bolt_action_shot | fresh5 | gunshot_rifle (SA3) | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| v_button_tap_elevator | fresh5 | ui_click | 0.35 | 1.00 | 1.00 [0.87, 1.00] | yes |  |  |
| v_c4_detonation | fresh5 | explosion (SA3) | 0.05 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| v_car_explosion | fresh5 | explosion (SA3) | 0.81 | 1.00 | 1.00 [0.80, 1.00] | yes |  |  |
| v_closet_open | fresh5 | door_open | 0.97 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| v_dynamite_blast | fresh5 | explosion (SA3) | 0.93 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| v_horn_toot | fresh5 | horn | 0.33 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| v_lamp_shatter | fresh5 | glass_break | 0.57 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| v_longsword_clash | fresh5 | sword | 0.84 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| v_oar_splash | fresh5 | splash | 0.77 | 1.00 | 1.00 [0.91, 1.00] | yes |  |  |
| v_plank_walk_wood | fresh5 | footstep | 0.51 | 0.98 | 1.00 [0.65, 1.00] | yes |  |  |
| v_sawed_off_blast | fresh5 | gunshot_shotgun (SA3) | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| v_service_pistol_fire | fresh5 | gunshot_pistol (SA3) | 0.53 | 1.00 | 1.00 [0.89, 1.00] | yes |  |  |
| v_skylight_break | fresh5 | glass_break | 0.79 | 0.93 | 1.00 [0.84, 1.00] | yes |  |  |
| v_tumbler_smash | fresh5 | glass_break | 0.90 | 1.00 | 1.00 [0.75, 1.00] | yes |  |  |
| w:v_fall_in_harbor | fresh5 | water | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| w:v_oar_splash | fresh5 | water | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| w:v_pool_jump | fresh5 | water | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |  |
| w:v_puddle_stomp | fresh5 | water | 1.00 | 1.00 | 1.00 [0.71, 1.00] | yes |  |  |

## Rows the current judges hold under 0.9

Seven rows are out of reach for any faithful generator under these judges, and the evidence for each is a measurement on real recordings. The five metal footsteps (b_step_metal, g_footstep_metal, b_iron_boots_step, q_catwalk_step, v_ladder_step_metal) face a cap of 0.17: real metal steps from FoleySet, picked by their tags and kept out of every reference set, score 0.17 of real [0.00, 0.20] on these judges, and in the diagnosis CLAP called 15 of 20 of them "a knock" and PANNs called them "Music", the same labels the generated metal steps get. No arm here puts a metal row above 0.10 (today's engine peaks at 0.10 on q_catwalk_step), and the diagnosis found the same for every prompt tried before. That cap rests on 16 clips from 5 uploaders, which is thin, but the direction is not in doubt. The two rain rows on the water judge (w:b_rain_tin_roof, w:g_rain_on_roof_metal) face a cap of 0.66 from 40 ESC-50 rain clips, and the generated rain sits far below even that (0.00 to 0.22), partly because the caption template asks for "A single ... rain loop", which contradicts itself.

Every other judge-class cap turned out soft on this data, so those rows belong with the wording group. Real bottle breaks score 0.64, yet today's engine puts the bottle shot at 0.81; real drinking glasses score 0.47, yet f_goblet_break and v_tumbler_smash reach 1.00; real cabinet and car-door slams score 0.73, yet f_cabinet_slam reaches 0.97 and g_car_door_slam_car 0.94. Wood, snow and grass steps each have a sibling row at 1.00; leaves has one row, f_jog_leaves, at 0.49 under a 0.84 cap. The Woosh autoencoder round trip of the diagnosis keeps every type at 0.94 to 1.00 of real, so the decoder limits none of them; that number comes from the diagnosis and was not rerun here.

## What each remaining row needs

The 57 rows under 0.9 that the judges do not cap split into three needs.

Wording holds back 29 rows, 21 on Woosh and 8 on SA3. The caption template takes odd event words at face value ("ruff", "pummel", "drop", "magnum", "patrol step", "stomp"), the judges hear what the words say, and four candidates of one prompt all carry the same mistake. Today's engine beats B4 by more than 0.10 on 24 rows (the bottle shot 0.81 against 0.01, the wine glass drop 0.71 against 0.07, the stone stomp 0.58 against 0.04), which shows the models can make these sounds. It does not hand over a fix, because the phrasebook behind it is the per-sound and per-category table the generic principle rules out. Taking the better of H and B4 per row would put 108 rows at 0.9; that is an oracle chosen on held-out data with a forbidden table in it, so it bounds what a perfect generic wording rule could do and is no result. The candidates already built point the way: the prompt-mixing pool lifted 38 to 40 of 52 selection rows and broke grass steps because the verifier rewards the material word, so the next verifier has to score the event too, for example the minimum of the request-text score and an event-only score. That can be measured on takes already judged before any render. On SA3 the verifier has no signal (AUC 0.50 on the sidearm in the diagnosis) and B4 plays the caption template there, so the gun and click rows need a better generic SA3 wording that carries the cues the phrasebook gets right (report, perspective, distance) without any per-sound entry.

Take luck holds back 24 rows, 19 on Woosh and 5 on SA3, whose B4 intervals reach 0.9. More candidates from the same verifier add nothing (best of eight reaches 100 rows against 101), so the lever is a sharper verifier. On five of these rows the current one ranks takes the wrong way (q_wardrobe_open AUC 0.32, f_tread_wood 0.39, b_keyboard_typing 0.41, g_melee_hit_body 0.48, q_beep_hatchback 0.49), and q_wardrobe_open fell from 1.00 under C to 0.899 under B4 for that reason. The five SA3 take rows get no selection at all today.

Shape holds back 4 rows: f_jab, q_hook_punch, v_beach_walk_sand and v_derringer_shot. The punches end within about 0.1 s (f_jab's median take falls 40 dB within 0.11 s). The longer 301-frame window cost punches 0.145 in lev10, and the decoder carries real punches at 0.95 of real, so the shortfall sits in generation, and no generic lever tried so far gives fuller hit tails.

For the seven capped rows the need is a judge change, with the references validated on held-out real clips only: a footstep judge that credits real metal steps (FoleySet offers 43 eligible metal-step clips from few uploaders, which may be too few) and a rain class in the water judge, plus a generic fix for the contradictory rain caption. A judge change is a new metric version, so every arm would need rescoring under it before any comparison.

## Caveats

The B4 numbers are an expectation over a 40-take pool, and the served path draws its candidates at variants v + 4096 j, so a session plays a different sample of the same distribution. The literal check above (76 against 78 Woosh rows) says the expectation is fair. The known rows are held out by seed only, so FRESH5 is the set to quote for generalization (26 of 46). `docs/levers.md` proposes confirming the next verifier on FRESH4, but FRESH4 has now been rendered by two rounds and FRESH5 by one, so the next confirmation needs a new never-rendered set. In the scripted session both arms reached 11,571 MiB, exactly the 11.3 GiB bound, at a desktop load of 1,320 MiB, and in the gate the gun fill sits 4.5 ms over today against a 5 ms bound; both pass with almost no margin. D: holds 21.43 GB decimal (19.96 GiB); this check wrote this page to D: and kept everything else on F:.

## Unity launcher for the configuration

Leave `pythonExe` (`D:\Python\SAGE\.venv214\Scripts\python.exe`), `workingDir` (`D:\Python\SAGE`), `hfHome` (`F:\AI\cache\hf`) and `port` (8765) as the scene carries them, and set `arguments` to:

```
D:\Python\sarge-research\modal\runpinned.py scripts\run_sidecar.py --generator hybrid --realtime --stereo --amp --detailed-prompts --learn --cache-dir D:\Python\SAGE\.cache\sfx_bake --prompts template:caption+tt --woosh-upgrade flow --woosh-upgrade-solver midpoint:16 --woosh-upgrade-best 4
```

The gate ran exactly this line with a port, an event log and a fresh cache directory added (`data/_gate/push09/upgrade/gate_p9_b4.json`). The verifier loads from `F:\AI\cache\woosh\checkpoints\Woosh-CLAP`, which holds the same weights as the bench copy. Dropping `--woosh-upgrade-best 4` gives the plain Flow upgrade, and dropping the last four flags gives today's engine, which served 75 of 75 clips bit-identical to `baseline_warm_1`.

## Round p10: five generic levers, chosen on 740-779 and confirmed on new data

The p10 round built five generic levers behind flags that default off (canonical names from the CPU writer, loop wording, a prompt pool, an event-aware verifier score and a VRAM wait), chose among them on the known scenarios at variants 740-779, and confirmed the choice on variants 780-819 plus FRESH6, 48 scenarios rendered for the first time and named the way game projects name events. The choice was A0, push09's best of four with nothing added, because every lever arm pushed at least one row more than 0.10 below it. On the 99 held-out decision rows the pick reaches 0.9 on 67, the plain caption template on Flow on 63 and today's engine on 27. Counting the five judge-capped rows of these sets, the pick has 67 of 104. "Above 0.9 on everything" fails by 32 rows.

| held out | rows | today ≥ 0.9 | caption ≥ 0.9 | pick ≥ 0.9 | pooled today / caption / pick | pick minus today [95%] | pick minus caption [95%] |
|---|---|---|---|---|---|---|---|
| known, variants 780-819 | 47 | 20 | 33 | 36 | 0.744 / 0.854 / 0.873 | +0.130 [+0.042, +0.222] | +0.019 [-0.014, +0.052] |
| FRESH6, first render | 52 | 7 | 30 | 31 | 0.504 / 0.741 / 0.784 | +0.280 [+0.185, +0.379] | +0.043 [+0.017, +0.071] |
| all decision rows | 99 | 27 | 63 | 67 | 0.618 / 0.795 / 0.827 | +0.209 [+0.142, +0.276] | +0.032 [+0.011, +0.054] |
| with the 5 capped rows | 104 | 27 | 63 | 67 | 0.592 / 0.758 / 0.788 | | |

The differences are this recheck's bootstrap over rows (4,000 draws, seed 99); the round's land within 0.005 of them. By the rule both are CONFIRMED: the lower bound sits above zero and no set's point estimate falls below -0.03. Best of eight, as context, reaches 68 rows.

### What was rechecked

**The confirmation numbers.** `scratchpad/verify10/recompute10.py` reads the round's `vocab.json`, `water.json`, `verifier.json` and both scenario files from `G:\sarge\p10\conf` (byte-identical to `data/_gate/p10/confirm/judge_outputs.zip`), takes each type's judges from the chosen set at 0.6 s, and rebuilds every row from the per-take hits, with the best-of-four weights enumerated over all 91,390 four-take subsets of 40. All 104 rows, all four arms (today H, caption C, B4, B8) and the literal served score agree with `confirm_p10.json` to 1e-9. Per-row intervals drawn with a different seed land within 0.035 of the round's at the 95th percentile (0.072 at worst). The cause labels, rebuilt from the rule's text by separate code, match the round's on all 32 rows under 0.9 (J 7, W 8, T 7, M 10). The judges and real rates equal flowpath W at 0.6 s, the water judge equals lev10's, and the known rows carry push09's keys, prompts and verifier texts unchanged.

**The selection.** `scratchpad/verify10/sel_recompute.py` rebuilds all ten arms on the 52 selection rows from `G:\sarge\p10\sel` (byte-identical to its zip), with the pool's prompt choice enumerated over every four-take subset from the verifier's scores alone. Every arm's point score matches `select_p10.json` to 1e-9, and rule 5 applied to the recomputed numbers returns A0. A4 (39 of 47) and A3 (38) lose b_step_grass by 0.56, A5 loses b_step_gravel by 0.56, and A1 and A2 each drop six rows by more than 0.10.

**Rules before renders, and no confirmation data in the selection.** The selection rule hashes to 8ba3ae12 and was written at 12:46:02; its prompts followed at 12:52:33 and its first clip at 12:55:25. All 5,240 Woosh clips and 520 SA3 cuts of the selection sit at variants 740-779, the last written at 13:37, and the selection result was written at 14:37:48. FRESH6 (`fresh6.py`, 1cbef694) dates from 14:47:58, after the selection had finished, and the confirmation rule (3fc70b25) from 15:01:16. The first confirmation clip followed at 15:37:14, and every confirmation clip sits at 780-819. No FRESH6 event name or render key appears in any selection-round file. Every lever source edit is dated 11:47 to 12:33, before both rules, and none has changed since.

How new FRESH6 is deserves one qualification. None of its rows repeats a render key or an event name, yet with numbers and file prefixes stripped x_sfx_magnum_fire_01 reduces to FRESH4's q_magnum_fire and x_footstep_snow_deep_l to b_snow_step, and 13 FRESH6 rows contain every word of an earlier held-out name (the brief asked for names like these). The pick has no part tuned on any of them, so the confirmation loses nothing. FRESH6 is new by render key and only partly new by vocabulary, and both twins land under 0.9, as their earlier rows did.

**No evaluation judge chose anything.** The served choice code (the worker's `Verifier`, `verify.pick`, `verify.choose` and `sarge.pool`) loads Woosh-CLAP alone, on the CPU by default in HEAD as well as now, from weights byte-identical to the bench copy, and no served file names AST, PANNs, laion CLAP or the water judge. Judge outputs chose the arm in the selection round, on 740-779 as its rule says, and chose the page rows for display; those choices never reach what a session serves.

**Verifier leakage.** Woosh-CLAP is still a relative of two judges, and this round it leans harder toward its AudioSet sibling than push09 did. Its per-take score tracks AST hits at a mean Spearman of 0.238, CLAP at 0.168 and PANNs at 0.161, while the judges agree with each other at 0.22 to 0.31, and its served-hit gain over the caption is +0.083 on AST, +0.053 on PANNs and +0.051 on CLAP. On the 70 judged Woosh decision rows the pick gains +0.049 [+0.023, +0.076] over the caption. Scored by PANNs alone, on the 66 rows PANNs judges, it gains +0.036 [+0.013, +0.060]. Dropping CLAP changes nothing and dropping AST takes 0.011, so up to about a quarter of the verifier's gain over the caption may be sibling agreement. The remainder holds with an interval clear of zero, and the comparison that decides, the pick against today's engine, stays at +0.229 [+0.168, +0.292] with both relatives dropped.

**The build.** The canonicaliser's system prompt and its nine few-shot examples (shark bite, cat meow, wing flap, chair scrape, wind, ceiling fan hum, kiss, goat chewing, page turn) name no judged category and no evaluation sound. The only words they share with the 195 evaluation event names are file-format words (sfx, 01) and two words inside example sentences ("shut", "wooden"); the system prompt's "fire" and "water" appear only in its list of steady sounds. The loop rule's word list holds format words alone (loop, ambience, ambient, amb), and its rewrites touch template phrases. The rest of the diff wires flags, requests and the pool and holds no word list. One process note stands: the build read the first canonicaliser prompt's outputs on the 29 push09 wrong-wording rows once before tuning on a separate development set, so its table on those rows is partly seen data. The pick does not use the canonicaliser, so the confirmation is untouched by it.

**Tests, flags off and the page.** The CPU suite passes 635 of 635. `startup_compare.py`, rerun on the round's saved e2e output (written 17:21 to 17:23, after the last source edit at 12:33), reports 75 of 75 clips identical to `baseline_warm_1`, max abs 0.0, PASS. The page `data/_gate/push09b/page/index.html` holds the 12 rows that rule 9 gives when rebuilt from the recomputed table, in its order. All 72 takes match the rule's sources sample for sample (today at 780-782, the verifier's top of 780-783, 784-787 and 788-791, SA3 caption takes at 780-782), the script passes `node --check`, and outside the intro the options are named only in the headings and the "Download picks" and "Clear picks" buttons. The second SA3 place was a near tie: g_keyboard_click_plastic beat x_gun_revolver_fire_03 on the size of B4 minus H by 0.0004.

### What the levers did on 740-779

Each lever fixed something and broke something else, and the rule's 0.10 guard caught every break. Canonical names took the bottle shot from 0.00 to 1.00 and b_explosion on SA3 from 0.89 to 1.00, then rewrote words that were already plain ("water footstep" became "water splash", "melee hit" became "body hit") and cost six rows more than 0.10, four of them with paired intervals below zero. The prompt pool put 38 and 39 of 47 rows at 0.9 against A0's 36 but took b_step_grass from 1.00 to 0.44: the verifier keeps choosing the grounded and named prompts, whose word "grass" it rewards, while the judges hear rustle. Scoring the event words too (A5) cut the grass loss to 0.67 and broke gravel the same way, 1.00 to 0.44. Loop wording moved the judge-capped tin-roof rain from 0.18 to 0.49, near the 0.66 that real rain gets, and changed little else. Shape got no lever: the build found that the post chain removes nothing above -55 dB on the four short push09 rows, and f_jab already falls 40 dB within 0.127 s before any trim. The VRAM lever rested on a wrong premise, since the verifier already ran on the CPU in HEAD.

### Every confirmation row

Intervals are the round's 95% bootstraps on the pick (1,000 resampled pools). Causes follow rule 7 of `RULE_P10_confirm.txt`, applied mechanically with the first label primary. "Today's engine X" marks rows where today's engine beats the pick by more than 0.10, and "V wrong way" marks rows where the verifier's AUC for the all-judge hit falls under 0.5.

| row | set | type | today H | caption C | pick, best of 4 [95%] | ≥ 0.9 | cause if under |
|---|---|---|---|---|---|---|---|
| b_bottle_shot_pistol | known | glass_break | 0.66 | 0.03 | 0.00 [0.00, 0.01] | no | soft class (real bottle 0.64, pick more than 0.10 under it); wrong sound (AST hears Ding in 23 of 40, all-judge hit 0.00); today's engine 0.66 |
| b_step_water | known | footstep | 0.44 | 0.31 | 0.24 [0.01, 0.53] | no | model or shape (takes 0.84 s long, 40 dB decay 0.56 s); today's engine 0.44 |
| g_footstep_water | known | footstep | 0.44 | 0.31 | 0.24 [0.01, 0.54] | no | model or shape (takes 0.84 s long, 40 dB decay 0.56 s); today's engine 0.44 |
| b_snow_step | known | footstep | 0.90 | 0.48 | 0.51 [0.19, 0.82] | no | soft class (real snow 0.88, pick more than 0.10 under it); today's engine 0.90 |
| g_melee_hit_body | known | punch | 0.51 | 0.54 | 0.58 [0.24, 0.93] | no | wrong sound (PANNs hears Whip in 25 of 40, all-judge hit 0.20); take, interval to 0.93 |
| b_step_wood | known | footstep | 0.76 | 0.58 | 0.58 [0.15, 1.00] | no | soft class (real wood 0.84, pick more than 0.10 under it); take, interval to 1.00; today's engine 0.76 |
| g_footstep_wood | known | footstep | 0.76 | 0.58 | 0.58 [0.20, 1.00] | no | soft class (real wood 0.84, pick more than 0.10 under it); take, interval to 1.00; today's engine 0.76 |
| b_glass_shatter | known | glass_break | 0.64 | 0.29 | 0.59 [0.25, 0.87] | no | soft class (real drinking glass 0.47) |
| b_keyboard_typing | known | ui_click | 0.24 | 0.91 | 0.60 [0.29, 0.97] | no | wrong sound (PANNs hears Music in 31 of 40, all-judge hit 0.55); take, interval to 0.97; V wrong way (AUC 0.37) |
| w:b_drip_puddle | known | water | 1.00 | 1.00 | 0.74 [0.37, 1.00] | no | take, interval to 1.00; today's engine 1.00; V wrong way (AUC 0.37) |
| w:g_water_drip_water | known | water | 1.00 | 1.00 | 0.74 [0.35, 1.00] | no | take, interval to 1.00; today's engine 1.00; V wrong way (AUC 0.37) |
| g_glass_shatter_glass | known | glass_break | 0.71 | 0.58 | 0.92 [0.63, 1.00] | yes |  |
| b_car_door_slam | known | door_slam | 1.00 | 0.95 | 0.94 [0.82, 1.00] | yes |  |
| b_pistol_shot | known | gunshot_pistol (SA3) | 1.00 | 0.95 | 0.95 [0.84, 1.00] | yes |  |
| g_shoot_pistol | known | gunshot_pistol (SA3) | 1.00 | 0.95 | 0.95 [0.84, 1.00] | yes |  |
| g_shotgun_fire_shotgun | known | gunshot_shotgun (SA3) | 0.85 | 0.96 | 0.96 [0.89, 1.00] | yes |  |
| g_break_window | known | glass_break | 0.68 | 0.99 | 0.97 [0.80, 1.00] | yes |  |
| b_explosion | known | explosion (SA3) | 1.00 | 0.97 | 0.97 [0.83, 1.00] | yes |  |
| b_step_stone | known | footstep | 0.44 | 0.75 | 0.99 [0.61, 1.00] | yes |  |
| g_footstep_stone | known | footstep | 0.44 | 0.75 | 0.99 [0.63, 1.00] | yes |  |
| g_car_door_slam_car | known | door_slam | 0.60 | 0.88 | 1.00 [0.81, 1.00] | yes |  |
| b_car_horn | known | horn | 0.17 | 0.83 | 1.00 [0.79, 1.00] | yes |  |
| b_dog_bark | known | dog_bark | 0.31 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| b_door_creak | known | door_open | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| b_punch | known | punch | 0.74 | 0.91 | 1.00 [0.83, 1.00] | yes |  |
| b_rifle_shot | known | gunshot_rifle (SA3) | 0.99 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| b_step_grass | known | footstep | 0.34 | 1.00 | 1.00 [0.86, 1.00] | yes |  |
| b_step_gravel | known | footstep | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| b_sword_clash | known | sword | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| b_window_rifle | known | glass_break | 0.61 | 0.87 | 1.00 [0.79, 1.00] | yes |  |
| g_bullet_impact_glass | known | glass_break | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| g_car_horn_car | known | horn | 0.00 | 0.91 | 1.00 [0.89, 1.00] | yes |  |
| g_dog_bark_dog | known | dog_bark | 0.85 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| g_door_creak_wood | known | door_open | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| g_explosion_large_debris | known | explosion (SA3) | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| g_footstep_gravel | known | footstep | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| g_footsteps_concrete_concrete | known | footstep | 0.63 | 1.00 | 1.00 [0.84, 1.00] | yes |  |
| g_keyboard_click_plastic | known | ui_click (SA3) | 0.45 | 1.00 | 1.00 [0.84, 1.00] | yes |  |
| g_pistol_shot_9mm | known | gunshot_pistol (SA3) | 1.00 | 1.00 | 1.00 [0.96, 1.00] | yes |  |
| g_playerfootstep_snow | known | footstep | 0.65 | 0.95 | 1.00 [0.77, 1.00] | yes |  |
| g_punch_body | known | punch | 0.74 | 0.91 | 1.00 [0.83, 1.00] | yes |  |
| g_shoot_rifle | known | gunshot_rifle (SA3) | 0.99 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| g_splash_water | known | splash | 0.56 | 1.00 | 1.00 [0.77, 1.00] | yes |  |
| g_sword_clash_steel | known | sword | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| w:b_step_water | known | water | 0.92 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| w:g_footstep_water | known | water | 0.92 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| w:g_splash_water | known | water | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| x_sfx_magnum_fire_01 | fresh6 | gunshot_pistol (SA3) | 0.00 | 0.00 | 0.00 [0.00, 0.00] | no | wrong sound (AST hears Music in 18 of 40, all-judge hit 0.00) |
| x_sfx_body_fall_water | fresh6 | splash | 0.04 | 0.03 | 0.00 [0.00, 0.01] | no | wrong sound (AST hears Sound effect in 7 of 40, all-judge hit 0.00) |
| x_footstep_stomp_concrete | fresh6 | footstep | 0.24 | 0.13 | 0.02 [0.00, 0.10] | no | wrong sound (PANNs hears Knock in 14 of 40, all-judge hit 0.07); today's engine 0.24; V wrong way (AUC 0.25) |
| x_hit_fist_face_01 | fresh6 | punch | 0.19 | 0.27 | 0.29 [0.05, 0.62] | no | wrong sound (PANNs hears Music in 21 of 40, all-judge hit 0.12) |
| x_ar15_fire_single | fresh6 | gunshot_rifle (SA3) | 0.77 | 0.39 | 0.39 [0.21, 0.55] | no | model (SA3 caption wording); today's engine 0.77 |
| x_gate_open_squeak_rusty | fresh6 | door_open | 0.76 | 0.47 | 0.42 [0.18, 0.71] | no | model or shape (takes 0.98 s long, 40 dB decay 0.39 s); today's engine 0.76; V wrong way (AUC 0.44) |
| x_sfx_dog_ruff_01 | fresh6 | dog_bark | 0.05 | 0.23 | 0.43 [0.16, 0.64] | no | model or shape (takes 0.54 s long, 40 dB decay 0.19 s) |
| x_blade_hit_parry_01 | fresh6 | sword | 0.73 | 0.35 | 0.46 [0.21, 0.76] | no | model or shape (takes 0.98 s long, 40 dB decay 0.52 s); today's engine 0.73 |
| x_sfx_truck_horn_01 | fresh6 | horn | 0.20 | 0.34 | 0.47 [0.18, 0.68] | no | wrong sound (PANNs hears Music in 37 of 40, all-judge hit 0.15) |
| x_fist_impact_gut | fresh6 | punch | 0.85 | 0.43 | 0.47 [0.20, 0.81] | no | model or shape (takes 0.51 s long, 40 dB decay 0.34 s); today's engine 0.85 |
| x_sfx_step_sand_sneak | fresh6 | footstep | 0.81 | 0.17 | 0.47 [0.13, 0.76] | no | wrong sound (PANNs hears Gasp in 3 of 40, all-judge hit 0.07); today's engine 0.81 |
| w:x_sfx_body_fall_water | fresh6 | water | 0.48 | 0.26 | 0.51 [0.14, 0.82] | no | model or shape (takes 0.59 s long, 40 dB decay 0.19 s) |
| x_sfx_mirror_crack_shatter | fresh6 | glass_break | 0.05 | 0.35 | 0.53 [0.22, 0.86] | no | model or shape (takes 0.40 s long, 40 dB decay 0.31 s) |
| x_step_plank_bridge | fresh6 | footstep | 0.13 | 0.54 | 0.55 [0.16, 0.90] | no | soft class (real wood 0.84, pick more than 0.10 under it) |
| x_water_wade_step_shallow | fresh6 | footstep | 0.00 | 0.54 | 0.59 [0.30, 0.94] | no | take, interval to 0.94 |
| x_water_splash_big | fresh6 | splash | 0.09 | 0.56 | 0.66 [0.30, 1.00] | no | take, interval to 1.00 |
| x_puppy_yip_02 | fresh6 | dog_bark | 0.40 | 0.28 | 0.68 [0.43, 0.83] | no | model or shape (takes 0.21 s long, 40 dB decay 0.03 s) |
| x_ui_button_click_01 | fresh6 | ui_click (SA3) | 0.28 | 0.70 | 0.70 [0.49, 0.91] | no | take, interval to 0.91 |
| x_char_footstep_tile_heel | fresh6 | footstep | 0.67 | 0.85 | 0.73 [0.34, 1.00] | no | take, interval to 1.00 |
| x_splash_small_object_01 | fresh6 | splash | 0.65 | 0.65 | 0.81 [0.35, 1.00] | no | take, interval to 1.00 |
| x_footstep_snow_deep_l | fresh6 | footstep | 1.00 | 0.75 | 0.87 [0.46, 1.00] | no | soft class (real snow 0.88); take, interval to 1.00; today's engine 1.00; V wrong way (AUC 0.42) |
| x_sfx_door_slam_heavy_01 | fresh6 | door_slam | 0.82 | 0.95 | 0.95 [0.82, 1.00] | yes |  |
| x_wpn_pistol_shot_close | fresh6 | gunshot_pistol (SA3) | 0.68 | 0.95 | 0.95 [0.84, 1.00] | yes |  |
| x_dog_woof_big | fresh6 | dog_bark | 0.00 | 0.90 | 0.95 [0.84, 1.00] | yes |  |
| x_punch_heavy | fresh6 | punch | 0.47 | 0.97 | 0.95 [0.74, 1.00] | yes |  |
| x_car_horn_honk_short | fresh6 | horn | 0.37 | 0.91 | 0.97 [0.77, 1.00] | yes |  |
| x_gun_revolver_fire_03 | fresh6 | gunshot_pistol (SA3) | 0.44 | 0.98 | 0.98 [0.88, 1.00] | yes |  |
| w:x_splash_small_object_01 | fresh6 | water | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| w:x_water_splash_big | fresh6 | water | 0.26 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| w:x_water_wade_step_shallow | fresh6 | water | 0.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| x_boomstick_fire | fresh6 | gunshot_shotgun (SA3) | 0.56 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| x_car_window_smash_02 | fresh6 | glass_break | 0.81 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| x_cupboard_open_01 | fresh6 | door_open | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| x_door_close_hard | fresh6 | door_slam | 0.85 | 0.97 | 1.00 [1.00, 1.00] | yes |  |
| x_exp_car_bomb_01 | fresh6 | explosion | 0.03 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| x_fire_sniper_01 | fresh6 | gunshot_rifle (SA3) | 0.84 | 1.00 | 1.00 [0.87, 1.00] | yes |  |
| x_foley_boots_mud_squelch | fresh6 | footstep | 0.72 | 0.99 | 1.00 [0.84, 1.00] | yes |  |
| x_fs_dirt_walk_03 | fresh6 | footstep | 0.68 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| x_glass_break_window_large_01 | fresh6 | glass_break | 0.13 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| x_impact_glass_bottle_smash | fresh6 | glass_break | 0.32 | 0.90 | 1.00 [0.90, 1.00] | yes |  |
| x_melee_punch_light | fresh6 | punch | 0.69 | 1.00 | 1.00 [0.96, 1.00] | yes |  |
| x_mortar_impact_explosion | fresh6 | explosion (SA3) | 0.30 | 1.00 | 1.00 [0.87, 1.00] | yes |  |
| x_mouse_click_left | fresh6 | ui_click (SA3) | 0.91 | 1.00 | 1.00 [0.87, 1.00] | yes |  |
| x_play_footstep_gravel_run | fresh6 | footstep | 0.92 | 1.00 | 1.00 [0.80, 1.00] | yes |  |
| x_run_asphalt_fast | fresh6 | footstep | 0.88 | 1.00 | 1.00 [0.85, 1.00] | yes |  |
| x_sfx_grenade_explode_near | fresh6 | explosion (SA3) | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| x_sfx_keyboard_tap_03 | fresh6 | ui_click | 0.17 | 0.73 | 1.00 [0.72, 1.00] | yes |  |
| x_sword_clang_metal | fresh6 | sword | 0.70 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| x_vehicle_horn_beep_double | fresh6 | horn | 0.40 | 1.00 | 1.00 [0.85, 1.00] | yes |  |
| x_walk_grass_soft_02 | fresh6 | footstep | 0.58 | 1.00 | 1.00 [0.81, 1.00] | yes |  |
| x_weapon_sword_impact | fresh6 | sword | 0.32 | 1.00 | 1.00 [1.00, 1.00] | yes |  |
| x_wpn_shotgun_blast_close | fresh6 | gunshot_shotgun (SA3) | 0.99 | 1.00 | 1.00 [1.00, 1.00] | yes |  |

Judge-capped rows of these sets, kept out of every count. FRESH6 holds no metal step and no rain, so only five of the seven capped rows appear here.

| row | type | today H | caption C | pick [95%] | cap |
|---|---|---|---|---|---|
| b_iron_boots_step | footstep | 0.22 | 0.00 | 0.00 [0.00, 0.00] | real metal steps 0.17 |
| b_step_metal | footstep | 0.04 | 0.00 | 0.00 [0.00, 0.00] | real metal steps 0.17 |
| g_footstep_metal | footstep | 0.04 | 0.00 | 0.00 [0.00, 0.00] | real metal steps 0.17 |
| w:b_rain_tin_roof | water | 0.09 | 0.13 | 0.13 [0.00, 0.37] | real rain 0.66 on the water judge |
| w:g_rain_on_roof_metal | water | 0.00 | 0.00 | 0.00 [0.00, 0.00] | real rain 0.66 on the water judge |

### What remains, and why

Thirty-two decision rows stay under 0.9, and the largest share is still wording. The confirmation shows its plainest form: the caption template copies file prefixes and take numbers into the prompt ("A single sfx magnum fire 01 sound", "A single hit fist face 01 sound"), and six of the nine judged FRESH6 rows whose names start with SFX_ land under 0.9, against 14 of the other 39. The eight rows the rule calls the wrong sound sit in this group, with the magnum and the body fall into water both at 0.00. Stripping numbers and file prefixes before the template is a generic rule about file names, but it was learned on confirmation data, so it needs a new rule and a new never-rendered set before it counts.

Seven rows sit on judge classes where real recordings score under 0.9 (wood steps 0.84, snow steps 0.88, bottles 0.64, drinking glasses 0.47), and five of them fall more than 0.10 under their class. Today's engine beats the pick by more than 0.10 on five of the seven, so the models can make sounds these judges accept there. The bottle shot is the plainest case at 0.66 today against 0.00 for the pick, and it is also the row that canonical names fixed at 740-779.

Seven rows depend on the take, with intervals that reach 0.9. The two drip rows scored 1.00 at 740-779 and 0.74 here while the caption alone scores 1.00, because the verifier ranks their takes the wrong way (AUC 0.37); it does so on nine decision rows in all, and best of eight adds a single row overall. Labels near the line also move between seed blocks: both wood-step rows went from 0.98 at 740-779 to 0.58 here and both stone rows from 0.67 to 0.99, so one row's yes or no at 40 takes is weaker evidence than the counts.

Ten rows carry no other label. Four are short: x_puppy_yip_02's takes run 0.21 s with a 40 dB decay of 0.03 s, x_sfx_dog_ruff_01 decays in 0.19 s, x_sfx_mirror_crack_shatter runs 0.40 s, and the water stem of x_sfx_body_fall_water decays in 0.19 s. The build found that the post chain cuts no real tail on the push09 punch rows, so this shortfall comes out of the generator. On six others today's phrasebook wins by more than 0.10 (x_ar15_fire_single 0.77 against 0.39 on SA3, x_fist_impact_gut 0.85 against 0.47, the rusty gate, the parry and both water-step rows), which again says the model can reach the sound under different words.

The arm the selection numbers point to was never measured: a pool of the caption and the canonical prompt only, which keeps the canonicaliser's fixes as one candidate and lets the caption keep the plain words that canonical names damaged. It belongs in a new rule written before any render, on new variants and a FRESH7 set that can test the file-prefix rule at the same time. The five capped rows still need a judge change validated on held-out real clips.

### Runtime

The pick passes every gate bound under Unity's environment. Its peak after READY is 11,396 MiB against an 11,421 MiB bound, 25 MiB to spare, and the gun first hit sits 4.7 ms over today's against a 5 ms bound, with Woosh 0.3 ms faster. The scripted session with the game60 load peaked at 11,549 MiB, which fails the 150 MiB margin by 128 MiB and sits 22 MiB under 11.3 GiB. The verifier already ran on the CPU (about 165 ms a take on four threads), so that growth comes from Flow and its renders. `--woosh-upgrade-vram-free 2200` only delays the Flow load; it opened at once because READY leaves about 2,216 MiB free, and the session peaked at 11,548 MiB. A real margin needs a cap on what Flow allocates while it renders. D: holds 26.4 GB decimal (24.6 GiB) after this check.

### Unity launcher for the p10 pick

The pick is the configuration above, so the launcher stays as it was. Leave `pythonExe` (`D:\Python\SAGE\.venv214\Scripts\python.exe`), `workingDir` (`D:\Python\SAGE`), `hfHome` (`F:\AI\cache\hf`) and `port` (8765) as the scene carries them, and set `arguments` to:

```
D:\Python\sarge-research\modal\runpinned.py scripts\run_sidecar.py --generator hybrid --realtime --stereo --amp --detailed-prompts --learn --cache-dir D:\Python\SAGE\.cache\sfx_bake --prompts template:caption+tt --woosh-upgrade flow --woosh-upgrade-solver midpoint:16 --woosh-upgrade-best 4
```

The p10 gate ran exactly this line with two ports, an event log and a fresh cache directory added (`data/_gate/p10/confirm/upgrade/gate_p10c_b4.json`). Every p10 flag (`--writer-style canon`, `+loop`, `--woosh-upgrade-pool`, `--woosh-upgrade-score`, `--woosh-upgrade-vram-free`) stays off in it.

## Round p11: cleaned names, a two-caption pool and VRAM savings, chosen on 820-859 and confirmed on new data

The p11 round built name cleaning (the `+clean` template modifier strips file prefixes, take numbers, side markers and engine words from event and material names before any template) and two VRAM savings for the Flow worker, all behind flags that default off. It chose among five arms on the known scenarios at variants 820-859 under a noise-aware guard written before any render, and confirmed the choice on variants 860-899 plus FRESH7, 57 scenarios rendered for the first time. The selection picked A3: a pool of the cleaned caption and the canonical-name caption, where Woosh-CLAP keeps one caption per key and then the best of four takes, with the savings on. On the 109 held-out decision rows the pick reaches 0.9 on 68, B4 (Unity's line) on 54 and today's engine on 27, and its pooled score sits 0.106 above B4's. The confirmation rule still turns it down. Six rows fall more than 0.10 below B4 with paired intervals wholly under zero, so guard R1 fails and B4 stays in Unity. Counting the five judge-capped rows, the pick has 69 of 114 and B4 54. "Above 0.9 on everything" fails by 41 rows under the pick and by 55 under B4.

| held out | rows | today ≥ 0.9 | caption ≥ 0.9 | B4 ≥ 0.9 | clean K ≥ 0.9 | pick ≥ 0.9 | pooled today / B4 / K / pick | pick minus B4 [95%] | pick minus today [95%] |
|---|---|---|---|---|---|---|---|---|---|
| known, variants 860-899 | 47 | 23 | 30 | 32 | 32 | 35 | 0.753 / 0.889 / 0.895 / 0.921 | +0.032 [-0.010, +0.085] | +0.168 [+0.084, +0.255] |
| FRESH7, first render | 62 | 4 | 19 | 22 | 22 | 33 | 0.322 / 0.554 / 0.520 / 0.716 | +0.162 [+0.069, +0.256] | +0.394 [+0.295, +0.498] |
| all decision rows | 109 | 27 | 49 | 54 | 54 | 68 | 0.508 / 0.699 / 0.681 / 0.805 | +0.106 [+0.049, +0.167] | +0.297 [+0.226, +0.369] |
| with the 5 capped rows | 114 | 27 | 49 | 54 | 54 | 69 | 0.490 / 0.669 / 0.652 / 0.779 | | |

The differences are this recheck's bootstrap over rows (4,000 draws, seed 271828); the round's land within 0.006 of them. R2 (more rows at 0.9), R3 (pooled lower bound +0.049, worst set +0.032) and R4 (runtime) pass, so the guard alone decides. Against today's engine both configurations are CONFIRMED: the pick at +0.297 and B4 at +0.191 [+0.108, +0.272]. The six counted rows, with the round's paired intervals, are b_step_water and g_footstep_water (-0.265 each, upper ends -0.054 and -0.041), z_expl_tank_shell_hit (-0.907 [-0.990, -0.735]), z_fs_sneaker_linoleum_l (-0.395 [-0.626, -0.172]), z_jog_dune (-0.574 [-0.850, -0.061]) and z_fishtank_bust_01 (-0.318 [-0.543, -0.074]). Three more drops over 0.10 reach zero and count as noise: z_slosh_through_flood_02, z_doorkick_burst_01 and z_boulder_plunge_pond.

### What was rechecked

**The confirmation numbers.** `scratchpad/verify11/recompute11.py` reads units.json, vocab.json, water.json and verifier.json from `G:\sarge\p11\conf` (byte-identical to `data/_gate/p11/confirm/judge_outputs.zip`; both scenario files match their digests in `rules_sha1.txt`), takes each type's judges from the chosen set at 0.6 s and rebuilds every row from the per-take hits. Best of four is enumerated over all 91,390 four-take subsets (equal to the closed form to 2e-16), and the pool's share P_k is enumerated over the same subsets from the verifier scores alone. All 114 rows, all five arms (today H, caption C, B4, clean K, pick P) and every P_k agree with `confirm_p11.json` to 1e-9. Per-row intervals drawn with a different seed land within 0.027 of the round's at the 95th percentile (0.045 at worst), and every row gets the same guard verdict. The literal served takes (62 pick rows and 43 B4 rows at 0.9 over 91 Woosh decision rows, against 55 and 42 by the expectation) and the wrong-way counts (15 rows for the pick's dominant caption, 17 for B4's takes) match row for row. The cause labels, rebuilt from rule 7 by separate code, match the round's on all 41 rows under 0.9. The judges and real rates equal flowpath W at 0.6 s, the water judge equals lev10's, the consensus judges equal the chosen set on every judged row, and the 49 known scenarios carry push09's keys, render keys, caption and phrasebook prompts and B4 verifier texts unchanged.

**The selection.** `scratchpad/verify11/sel_recompute11.py` rebuilds all five arms on the 52 selection rows from `G:\sarge\p11\sel`. Every point score matches `select_p11.json` to 1e-9, a separate bootstrap gives the same guard verdicts (A2 counted on b_window_rifle and g_bullet_impact_glass, A3 noise only), and rule 6 applied to the recomputed numbers returns A3. The edit to `s11_select.py` after the renders, which zero-pads takes shorter than one STFT frame in the G1 check, therefore changed no score, guard or ranking.

**Rules before renders, and no confirmation data in the selection.** The selection rule hashes to 3a5ac035 and was written at 18:32:38, with the guard in its section 5. Its first clip followed at 18:39:38, all 6,320 Woosh and 520 SA3 clip and cut files sit at 820-859, and the selection result was written at 19:48:28. FRESH7 (`fresh7.py`, 6ccd0413) dates from 20:11:57, its check from 20:14:20 and the confirmation rule (bf3bf42e) from 20:14:21. The confirmation prompts followed at 20:30:19 and the first clip at 20:31:37, all 24,160 Woosh and 1,672 SA3 files sit at 860-899, and the analysis code (`c11_conf.py`) was hashed at 20:33:24, before the judge outputs of 23:06 and 23:17. All 22 digests in the confirmation's `rules_sha1.txt` still match, as do the selection's (`s11_select.py` at its post-render digest). No FRESH7 event name, scenario id or render key appears in any of 113 selection-round and build files, and no selection file cites a variant at 860-899. A file scan of G:\sarge, D:\Python\SAGE\data, F:\sarge_p11 and the scratchpads finds no clip numbered 860-899 outside the round apart from the six real FoleySet recordings the round named. Every source edit of the round is dated 18:00 to 18:03:56, before both rules.

**FRESH7 is new.** `scratchpad/verify11/fresh7_indep.py` gathered 366 earlier eval names from every scenario file under G:\sarge, data\_gate and the scratchpads, plus generic_eval's key sets, the e2e and vocab_eval keys, scenes40, the p10 canonicaliser sets and the build's 40 printed names. With a separate tokenizer and stemmer, no FRESH7 name equals an earlier one and none shares all its content words with an earlier name in either direction, with common prefixes (wpn, fs, ui, npc, gun, hit and the like) kept or dropped.

**No evaluation judge chose anything.** The served choice code (the worker's `Verifier`, `verify.pick`, `verify.choose`, `sarge.pool` and the canonical writer) names no AST, PANNs, laion CLAP or water judge, and the round's prep and render tools read no judge output. Woosh-CLAP's bench and served weights are byte-identical, and the 11,360 saved verifier scores equal the dot products of the saved audio and text embeddings exactly. Rebuilding the caption, clean and canonical prompts, the three verifier texts and the pool's candidate lists from `sarge.prompts`, `sarge.canon` and `sarge.verify` with the saved canonical records reproduces all 106 scenarios, so every prompt came from shipped code. The canonical writer gave the same raw output as the selection on 49 of 49 known keys and failed on none of the 57 FRESH7 keys. Judge outputs chose the arm in the selection, on 820-859 as its rule says, and the page rows for display.

**Verifier leakage.** On the pick's dominant takes the verifier's per-take score tracks AST hits at a mean Spearman of 0.18, CLAP at 0.17 and PANNs at 0.16, below the judges' agreement with each other (0.22 to 0.30). The pick's served-hit gain over B4 is largest on PANNs, the judge with no family tie to Woosh-CLAP: +0.128, against +0.085 on CLAP and +0.070 on AST. The pool's caption choice matches the caption with the higher plain hit rate on 81% of rows under AST, 83% under PANNs and 72% under CLAP, so it leans toward no sibling. Scored by PANNs alone, the pick gains +0.103 [+0.047, +0.161] over B4 on 94 judged rows, against +0.105 with every judge; dropping CLAP leaves +0.101 and dropping AST +0.107. Sibling agreement carries none of the measurable gain this round.

**The cleaning code.** `words.clean_words` holds five word lists: format words (sfx, fx, snd, sound, audio, file extensions), engine words (prefab, clone, instance, lod, mesh, gameobject, geo, sm, prop, variant), side letters (l, r), two lead words (a, play) and take markers (v, ver, var, take, tk, alt). None names a sound, a material or a game, and the 15 new tests use invented names (Zorbo, Krell). One generic rule clips more than the build's summary says: a word of three or more letters glued to digits loses the digits, so Kar98 becomes "kar" while ar15, c4 and 9mm keep theirs. That row lost 0.03 under K.

**Tests, flags off and the page.** The CPU suite passes 650 of 650. `startup_compare.py`, rerun on both saved flags-off outputs (the build's from 18:25 and the confirmation's from 23:35, both after the last source edit at 18:03), reports 75 of 75 clips identical to `baseline_warm_1`, max abs 0.0, PASS. The page `data/_gate/push09c/page/index.html` holds the 12 rows that rule 9 gives when repeats are skipped by scenario, in its order, all 72 takes match the rule's sources sample for sample, and its script passes `node --check`. That reading is a judgment call. Skipping repeats by row name would give the second water place to w:z_slosh_through_flood_02 ahead of w:z_boulder_plunge_pond, and those takes would repeat the judged slosh row's audio. Outside the intro "pick" appears in the two headings and in the "Download picks" and "Clear picks" buttons; the round's report mentions only the buttons. One summary line of the round overstates its table: of the 18 FRESH7 rows labelled wrong sound, 17 have no arm at 0.9 (B4 reaches 1.00 on the tank shell) and 5 stay at or under 0.03 in every arm.

### What the round found

Name cleaning alone did nothing where names carry no markers and hurt where they do. On the known rows K and B4 differ by +0.005, the size of the served path's decoder difference. On FRESH7, K minus B4 is -0.035 [-0.075, -0.002], and on the 34 rows where cleaning changes the prompt or the verifier text it is -0.063 [-0.130, -0.002], with 13 of those rows at 0.9 under both. The largest losses are z_doberman_alarm_loud_03 (1.00 to 0.16) and z_cutlass_ring_02 (0.57 to 0.12); the gains are fewer and smaller (z_ftstp_rug_soft_03 0.79 to 1.00, the SA3 snubnose 0.88 to 1.00). Dropping a take number or a prefix moves the caption's takes a long way on some names, which points at how fragile a single caption is on odd names.

The pick's whole gain comes from the canonical caption. On the 49 Woosh decision rows where the pool mostly keeps it, the pick averages +0.241 over B4, 33 rows up and 7 down; on the 42 rows where the clean caption wins or stands alone it averages -0.011. Over FRESH7 the pick beats clean-only by +0.196. The large wins are names the caption template misreads: z_pc_boot_pavement_r and z_fish_flop_bucket from 0.00 to 1.00, z_fx_detpack_burst_03 from 0.00 to 0.91, z_k9_alert_bay_l from 0.13 to 1.00, and the bottle shot from 0.03 to 1.00.

The six counted rows come from one design choice: every candidate is scored against the canonical record's request text. When the canonical name is wrong, its takes match its wrong words best and win the pool. "Shell impact" wins 93% of subsets on the tank shell although its takes score 0.00 by themselves (B4 1.00, pick 0.09); "foot stomp" on sand wins 84% on z_jog_dune; on z_fs_sneaker_linoleum_l "sneaker slide" scores 0.03 by itself and still wins 45%. When the clean caption wins, its takes are then ranked against the canonical words: on both water steps the clean caption is kept in 54% of subsets, but its best of four falls from 0.60 under the clean text to 0.38 under "water splash". A pool can only be as safe as its worst canonical name, and this verifier does not notice a wrong one. The next rule worth writing keeps the canonical caption only when its takes also score at least as well as the clean caption's against the clean text, a test the selection takes at 820-859 can answer with one more CPU verifier pass before any render.

The judge-capped b_iron_boots_step went from 0.00 under B4 to 1.00 under the pick, because the canonical caption drops "iron boot" and asks for a plain stone footstep. The judges accept that sound; whether a player hears iron boots in it is a listening question, and the row stays out of every count by rule.

### Every confirmation row

Point values are this recheck's (equal to the round's); intervals are the round's 95% bootstraps under the rule's seed, with the guard verdict on the pick minus B4. Causes follow rule 7 of `RULE_P11_confirm.txt` for the pick, applied mechanically with the first label primary; "today's engine X" and "B4 X" mark rows where that configuration beats the pick by more than 0.10, and "V wrong way" marks rows where the verifier's AUC for the all-judge hit on the pick's dominant caption falls under 0.5.

| row | set | type | today H | B4 [95%] | clean K | pick [95%] | pick minus B4 [paired 95%] | cause if under |
|---|---|---|---|---|---|---|---|---|
| b_step_water | known | footstep | 0.61 | 0.61 [0.29, 0.90] | 0.60 | 0.34 [0.14, 0.58] | -0.26 [-0.48, -0.05] counted | wrong sound (PANNs hears Water in 21 of 40, all-judge hit 0.17); today's engine 0.61; B4 0.61; V wrong way (AUC 0.49) |
| g_footstep_water | known | footstep | 0.61 | 0.61 [0.27, 0.92] | 0.60 | 0.34 [0.14, 0.58] | -0.26 [-0.50, -0.04] counted | wrong sound (PANNs hears Water in 21 of 40, all-judge hit 0.17); today's engine 0.61; B4 0.61; V wrong way (AUC 0.49) |
| b_step_wood | known | footstep | 0.98 | 0.65 [0.25, 1.00] | 0.66 | 0.66 [0.26, 1.00] | +0.01 [+0.00, +0.06] | soft class (real wood 0.84, pick more than 0.10 under it); take, interval to 1.00; today's engine 0.98 |
| g_footstep_wood | known | footstep | 0.98 | 0.65 [0.25, 1.00] | 0.66 | 0.66 [0.26, 1.00] | +0.01 [+0.00, +0.06] | soft class (real wood 0.84, pick more than 0.10 under it); take, interval to 1.00; today's engine 0.98 |
| g_melee_hit_body | known | punch | 0.27 | 0.55 [0.14, 0.93] | 0.55 | 0.67 [0.30, 0.91] | +0.12 [-0.42, +0.54] | take, interval to 0.91 |
| b_snow_step | known | footstep | 0.63 | 0.68 [0.30, 0.96] | 0.69 | 0.77 [0.49, 1.00] | +0.09 [-0.12, +0.42] | soft class (real snow 0.88, pick more than 0.10 under it); take, interval to 1.00 |
| b_glass_shatter | known | glass_break | 0.82 | 0.76 [0.41, 0.97] | 0.76 | 0.82 [0.51, 0.99] | +0.06 [-0.37, +0.48] | soft class (real drinking glass 0.47); take, interval to 0.99 |
| b_explosion | known | explosion (SA3) | 1.00 | 0.84 [0.68, 0.95] | 0.84 | 0.84 [0.68, 0.95] | +0.00 [+0.00, +0.00] | take, interval to 0.95; today's engine 1.00 |
| b_step_stone | known | footstep | 0.22 | 0.73 [0.39, 0.98] | 0.85 | 0.85 [0.46, 1.00] | +0.12 [+0.00, +0.32] | take, interval to 1.00 |
| g_footstep_stone | known | footstep | 0.22 | 0.73 [0.40, 0.98] | 0.85 | 0.85 [0.48, 1.00] | +0.12 [+0.00, +0.33] | take, interval to 1.00 |
| b_car_door_slam | known | door_slam | 0.93 | 0.85 [0.59, 1.00] | 0.85 | 0.85 [0.58, 1.00] | -0.01 [-0.03, +0.00] | take, interval to 1.00 |
| g_break_window | known | glass_break | 0.74 | 0.84 [0.54, 1.00] | 0.85 | 0.85 [0.54, 1.00] | +0.00 [+0.00, +0.01] | take, interval to 1.00; V wrong way (AUC 0.46) |
| g_playerfootstep_snow | known | footstep | 0.90 | 0.85 [0.51, 1.00] | 0.85 | 0.91 [0.65, 1.00] | +0.06 [-0.23, +0.38] |  |
| g_splash_water | known | splash | 0.60 | 0.97 [0.54, 1.00] | 0.97 | 0.97 [0.55, 1.00] | -0.00 [-0.03, +0.03] |  |
| g_glass_shatter_glass | known | glass_break | 0.64 | 0.97 [0.67, 1.00] | 0.97 | 0.97 [0.66, 1.00] | -0.01 [-0.02, +0.00] |  |
| g_car_door_slam_car | known | door_slam | 0.25 | 0.98 [0.92, 1.00] | 0.98 | 0.98 [0.92, 1.00] | +0.00 [+0.00, +0.00] |  |
| g_shotgun_fire_shotgun | known | shotgun (SA3) | 0.91 | 0.99 [0.94, 1.00] | 0.99 | 0.99 [0.94, 1.00] | +0.00 [+0.00, +0.00] |  |
| b_bottle_shot_pistol | known | glass_break | 0.61 | 0.03 [0.00, 0.13] | 0.03 | 1.00 [0.90, 1.00] | +0.97 [+0.82, +1.00] |  |
| b_car_horn | known | horn | 0.20 | 1.00 [0.73, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.27] |  |
| b_dog_bark | known | dog_bark | 0.31 | 1.00 [0.99, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.01] |  |
| b_door_creak | known | door_open | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| b_keyboard_typing | known | ui_click | 0.42 | 0.50 [0.23, 0.82] | 0.50 | 1.00 [1.00, 1.00] | +0.50 [+0.18, +0.77] |  |
| b_pistol_shot | known | pistol (SA3) | 1.00 | 1.00 [0.92, 1.00] | 1.00 | 1.00 [0.92, 1.00] | +0.00 [+0.00, +0.00] |  |
| b_punch | known | punch | 0.82 | 1.00 [0.77, 1.00] | 1.00 | 1.00 [0.78, 1.00] | +0.00 [+0.00, +0.03] |  |
| b_rifle_shot | known | rifle (SA3) | 0.97 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| b_step_grass | known | footstep | 0.40 | 1.00 [0.89, 1.00] | 1.00 | 1.00 [0.90, 1.00] | +0.00 [+0.00, +0.00] |  |
| b_step_gravel | known | footstep | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| b_sword_clash | known | sword | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| b_window_rifle | known | glass_break | 0.68 | 1.00 [0.84, 1.00] | 1.00 | 1.00 [0.75, 1.00] | +0.00 [-0.20, +0.08] |  |
| g_bullet_impact_glass | known | glass_break | 1.00 | 1.00 [0.92, 1.00] | 1.00 | 1.00 [0.86, 1.00] | +0.00 [-0.14, +0.07] |  |
| g_car_horn_car | known | horn | 0.03 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| g_dog_bark_dog | known | dog_bark | 0.82 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| g_door_creak_wood | known | door_open | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| g_explosion_large_debris | known | explosion (SA3) | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| g_footstep_gravel | known | footstep | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| g_footsteps_concrete_concrete | known | footstep | 0.65 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| g_keyboard_click_plastic | known | ui_click (SA3) | 0.38 | 1.00 [0.94, 1.00] | 1.00 | 1.00 [0.94, 1.00] | +0.00 [+0.00, +0.00] |  |
| g_pistol_shot_9mm | known | pistol (SA3) | 1.00 | 1.00 [0.97, 1.00] | 1.00 | 1.00 [0.97, 1.00] | +0.00 [+0.00, +0.00] |  |
| g_punch_body | known | punch | 0.82 | 1.00 [0.78, 1.00] | 1.00 | 1.00 [0.78, 1.00] | +0.00 [+0.00, +0.03] |  |
| g_shoot_pistol | known | pistol (SA3) | 1.00 | 1.00 [0.92, 1.00] | 1.00 | 1.00 [0.92, 1.00] | +0.00 [+0.00, +0.00] |  |
| g_shoot_rifle | known | rifle (SA3) | 0.97 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| g_sword_clash_steel | known | sword | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| w:b_drip_puddle | known | water | 1.00 | 1.00 [0.70, 1.00] | 1.00 | 1.00 [0.70, 1.00] | +0.00 [+0.00, +0.00] |  |
| w:b_step_water | known | water | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| w:g_footstep_water | known | water | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| w:g_splash_water | known | water | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| w:g_water_drip_water | known | water | 1.00 | 1.00 [0.68, 1.00] | 1.00 | 1.00 [0.68, 1.00] | +0.00 [+0.00, +0.00] |  |
| z_barefoot_ceramic_pad | FRESH7 | footstep | 0.82 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | wrong sound (PANNs hears Music in 34 of 40, all-judge hit 0.00); today's engine 0.82 |
| z_bomb_dive_pool_01 | FRESH7 | splash | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | wrong sound (AST hears Explosion in 12 of 40, all-judge hit 0.00) |
| z_fuel_drum_erupt | FRESH7 | explosion | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | wrong sound (AST hears Music in 17 of 40, all-judge hit 0.00) |
| z_heel_click_parquet | FRESH7 | footstep (SA3) | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | wrong sound (PANNs hears Finger snapping in 36 of 40, all-judge hit 0.00) |
| z_windscreen_spiderweb_crack | FRESH7 | glass_break (SA3) | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | wrong sound (AST hears Crack in 13 of 40, all-judge hit 0.00) |
| z_dash_flagstone_v2 | FRESH7 | footstep | 0.03 | 0.00 [0.00, 0.00] | 0.03 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | wrong sound (PANNs hears Music in 19 of 40, all-judge hit 0.00) |
| z_wpn_carbine_tap_03 | FRESH7 | rifle | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 0.00 [0.00, 0.01] | +0.00 [+0.00, +0.01] | wrong sound (AST hears Tick-tock in 21 of 40, all-judge hit 0.00) |
| z_slosh_through_flood_02 | FRESH7 | footstep | 0.37 | 0.20 [0.00, 0.47] | 0.30 | 0.04 [0.00, 0.15] | -0.16 [-0.42, +0.04] noise | wrong sound (PANNs hears Drip in 21 of 40, all-judge hit 0.05); today's engine 0.37; B4 0.20; V wrong way (AUC 0.25) |
| w:z_bomb_dive_pool_01 | FRESH7 | water | 0.00 | 0.00 [0.00, 0.00] | 0.10 | 0.05 [0.00, 0.14] | +0.05 [+0.00, +0.14] | wrong sound (water judges' joint hit 0.07) |
| z_expl_tank_shell_hit | FRESH7 | explosion | 0.03 | 1.00 [0.97, 1.00] | 1.00 | 0.09 [0.01, 0.26] | -0.91 [-0.99, -0.74] counted | wrong sound (AST hears Music in 34 of 40, all-judge hit 0.00); B4 1.00 |
| z_hit_jaw_cross_r | FRESH7 | punch | 0.23 | 0.00 [0.00, 0.00] | 0.00 | 0.11 [0.00, 0.18] | +0.11 [+0.00, +0.18] | wrong sound (AST hears Sound effect in 14 of 40, all-judge hit 0.00); today's engine 0.23 |
| z_jog_dune | FRESH7 | footstep | 0.13 | 0.82 [0.37, 1.00] | 0.82 | 0.24 [0.03, 0.53] | -0.57 [-0.85, -0.06] counted | wrong sound (PANNs hears Music in 10 of 40, all-judge hit 0.05); B4 0.82 |
| z_ford_river_knee_deep | FRESH7 | footstep | 0.40 | 0.00 [0.00, 0.00] | 0.00 | 0.27 [0.05, 0.52] | +0.27 [+0.05, +0.52] | wrong sound (PANNs hears Drip in 22 of 40, all-judge hit 0.05); today's engine 0.40 |
| z_sfx_sawnoff_fire_01 | FRESH7 | shotgun (SA3) | 0.35 | 0.05 [0.00, 0.13] | 0.28 | 0.28 [0.15, 0.41] | +0.23 [+0.10, +0.35] | wrong sound (AST hears Crack in 14 of 40, all-judge hit 0.10) |
| z_npc_shuffle_loam_02 | FRESH7 | footstep | 0.85 | 0.03 [0.00, 0.09] | 0.03 | 0.43 [0.08, 0.82] | +0.39 [+0.05, +0.79] | model or shape (takes 0.82 s long, 40 dB decay 0.49 s); today's engine 0.85 |
| z_sfx_beretta_fire_l | FRESH7 | pistol (SA3) | 0.03 | 0.50 [0.34, 0.66] | 0.46 | 0.46 [0.30, 0.63] | -0.04 [-0.25, +0.18] | model (SA3 clean caption) |
| z_sfx_pane_burst_big_02 | FRESH7 | glass_break | 0.00 | 0.39 [0.09, 0.66] | 0.10 | 0.50 [0.21, 0.80] | +0.11 [-0.27, +0.50] | model or shape (takes 0.98 s long, 40 dB decay 0.42 s) |
| z_fs_sneaker_linoleum_l | FRESH7 | footstep | 0.31 | 1.00 [1.00, 1.00] | 0.99 | 0.60 [0.37, 0.83] | -0.40 [-0.63, -0.17] counted | model or shape (takes 0.81 s long, 40 dB decay 0.49 s); B4 1.00; V wrong way (AUC 0.44) |
| z_ui_tick_select | FRESH7 | ui_click | 0.42 | 0.65 [0.29, 0.95] | 0.65 | 0.64 [0.32, 0.92] | -0.01 [-0.22, +0.19] | wrong sound (PANNs hears Music in 38 of 40, all-judge hit 0.35); take, interval to 0.92; V wrong way (AUC 0.37) |
| z_vial_crash_lab | FRESH7 | glass_break | 0.00 | 0.28 [0.07, 0.46] | 0.27 | 0.66 [0.35, 0.96] | +0.38 [+0.02, +0.76] | wrong sound (AST hears Coin (dropping) in 23 of 40, all-judge hit 0.40); take, interval to 0.96 |
| z_fishtank_bust_01 | FRESH7 | glass_break | 0.68 | 1.00 [0.90, 1.00] | 0.66 | 0.68 [0.44, 0.91] | -0.32 [-0.54, -0.07] counted | take, interval to 0.91; B4 1.00 |
| z_boulder_plunge_pond | FRESH7 | splash | 0.00 | 1.00 [0.69, 1.00] | 1.00 | 0.70 [0.34, 1.00] | -0.30 [-0.65, +0.13] noise | take, interval to 1.00; B4 1.00; V wrong way (AUC 0.47) |
| z_handcannon_discharge_v3 | FRESH7 | pistol | 0.08 | 0.09 [0.00, 0.25] | 0.00 | 0.78 [0.56, 0.96] | +0.69 [+0.43, +0.90] | take, interval to 0.96; V wrong way (AUC 0.48) |
| z_landmine_trigger_big | FRESH7 | explosion | 0.07 | 0.60 [0.31, 0.83] | 0.60 | 0.82 [0.61, 0.95] | +0.22 [-0.00, +0.48] | wrong sound (PANNs hears Vehicle in 28 of 40, all-judge hit 0.50); take, interval to 0.95 |
| z_brawl_clobber_02 | FRESH7 | punch | 0.64 | 0.50 [0.22, 0.72] | 0.20 | 0.83 [0.49, 0.97] | +0.33 [-0.07, +0.62] | take, interval to 0.97; V wrong way (AUC 0.42) |
| z_cutlass_ring_02 | FRESH7 | sword | 0.41 | 0.57 [0.29, 0.84] | 0.12 | 0.85 [0.59, 1.00] | +0.28 [-0.08, +0.62] | take, interval to 1.00 |
| z_doorkick_burst_01 | FRESH7 | door_slam | 0.28 | 0.99 [0.92, 1.00] | 0.83 | 0.86 [0.69, 0.99] | -0.13 [-0.30, +0.01] noise | take, interval to 0.99; B4 0.99; V wrong way (AUC 0.47) |
| z_gut_slug_heavy | FRESH7 | punch | 0.23 | 0.00 [0.00, 0.00] | 0.00 | 0.88 [0.49, 1.00] | +0.88 [+0.49, +1.00] | wrong sound (PANNs hears Whip in 22 of 40, all-judge hit 0.65); take, interval to 1.00; V wrong way (AUC 0.44) |
| z_typewriter_keystrike_03 | FRESH7 | ui_click | 0.34 | 0.80 [0.52, 0.99] | 0.69 | 0.89 [0.68, 1.00] | +0.09 [-0.14, +0.38] | take, interval to 1.00 |
| z_fx_detpack_burst_03 | FRESH7 | explosion | 0.03 | 0.00 [0.00, 0.00] | 0.00 | 0.91 [0.59, 1.00] | +0.91 [+0.59, +1.00] |  |
| z_scattergun_shot_take2 | FRESH7 | shotgun (SA3) | 0.56 | 0.99 [0.91, 1.00] | 0.94 | 0.94 [0.84, 0.99] | -0.05 [-0.15, +0.04] |  |
| z_ui_click_confirm_02 | FRESH7 | ui_click (SA3) | 0.24 | 1.00 [1.00, 1.00] | 0.94 | 0.94 [0.73, 1.00] | -0.06 [-0.27, +0.00] |  |
| z_kar98_bolt_fire_r | FRESH7 | rifle (SA3) | 0.08 | 1.00 [1.00, 1.00] | 0.97 | 0.97 [0.86, 1.00] | -0.03 [-0.14, +0.00] |  |
| z_hike_pebbles_03 | FRESH7 | footstep | 0.54 | 0.87 [0.50, 1.00] | 0.69 | 0.97 [0.65, 1.00] | +0.10 [-0.26, +0.44] |  |
| z_entry_slammed_hard | FRESH7 | door_slam | 0.60 | 0.49 [0.25, 0.74] | 0.49 | 1.00 [0.97, 1.00] | +0.51 [+0.26, +0.74] |  |
| z_barn_gate_clatter_shut | FRESH7 | door_slam | 0.51 | 1.00 [0.85, 1.00] | 1.00 | 1.00 [0.89, 1.00] | -0.00 [-0.08, +0.13] |  |
| z_btn_press_menu_v2 | FRESH7 | ui_click | 0.03 | 1.00 [0.73, 1.00] | 1.00 | 1.00 [0.69, 1.00] | -0.00 [-0.29, +0.24] |  |
| z_chihuahua_yips_02 | FRESH7 | dog_bark | 0.65 | 1.00 [0.97, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.03] |  |
| w:z_boulder_plunge_pond | FRESH7 | water | 0.10 | 1.00 [0.94, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.06] |  |
| w:z_fish_flop_bucket | FRESH7 | water | 1.00 | 0.08 [0.00, 0.26] | 0.08 | 1.00 [1.00, 1.00] | +0.92 [+0.74, +1.00] |  |
| w:z_ford_river_knee_deep | FRESH7 | water | 0.03 | 0.90 [0.61, 1.00] | 0.90 | 1.00 [1.00, 1.00] | +0.10 [+0.00, +0.39] |  |
| w:z_slosh_through_flood_02 | FRESH7 | water | 0.03 | 0.90 [0.69, 1.00] | 0.97 | 1.00 [1.00, 1.00] | +0.10 [+0.00, +0.31] |  |
| z_display_case_smashed | FRESH7 | glass_break | 0.68 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| z_doberman_alarm_loud_03 | FRESH7 | dog_bark | 0.17 | 1.00 [1.00, 1.00] | 0.16 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| z_fish_flop_bucket | FRESH7 | splash | 0.47 | 0.00 [0.00, 0.00] | 0.00 | 1.00 [0.80, 1.00] | +1.00 [+0.80, +1.00] |  |
| z_ftstp_rug_soft_03 | FRESH7 | footstep | 0.09 | 0.79 [0.35, 1.00] | 1.00 | 1.00 [0.90, 1.00] | +0.21 [+0.00, +0.65] |  |
| z_hinge_squeal_shed | FRESH7 | door_open | 1.00 | 1.00 [0.92, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.08] |  |
| z_hooter_sedan_old | FRESH7 | horn | 0.00 | 1.00 [0.86, 1.00] | 1.00 | 1.00 [0.93, 1.00] | +0.00 [-0.07, +0.14] |  |
| z_k9_alert_bay_l | FRESH7 | dog_bark | 0.05 | 0.13 [0.01, 0.27] | 0.07 | 1.00 [1.00, 1.00] | +0.87 [+0.73, +0.99] |  |
| z_katana_clang_hit_r | FRESH7 | sword | 0.26 | 1.00 [0.91, 1.00] | 1.00 | 1.00 [0.96, 1.00] | +0.00 [-0.04, +0.09] |  |
| z_lamp_switch_snap | FRESH7 | ui_click | 0.59 | 0.16 [0.00, 0.41] | 0.16 | 1.00 [1.00, 1.00] | +0.84 [+0.59, +1.00] |  |
| z_m4a1_semi_shot | FRESH7 | rifle (SA3) | 0.86 | 1.00 [0.97, 1.00] | 1.00 | 1.00 [0.97, 1.00] | +0.00 [+0.00, +0.00] |  |
| z_melee_knuckle_thump | FRESH7 | punch | 0.49 | 0.18 [0.02, 0.37] | 0.19 | 1.00 [0.77, 1.00] | +0.82 [+0.54, +0.97] |  |
| z_minivan_toot_twice | FRESH7 | horn | 0.03 | 0.58 [0.30, 0.83] | 0.58 | 1.00 [1.00, 1.00] | +0.42 [+0.17, +0.70] |  |
| z_pc_boot_pavement_r | FRESH7 | footstep | 0.58 | 0.00 [0.00, 0.00] | 0.00 | 1.00 [1.00, 1.00] | +1.00 [+1.00, +1.00] |  |
| z_pooch_arf_excited | FRESH7 | dog_bark | 0.12 | 0.81 [0.55, 0.91] | 0.81 | 1.00 [1.00, 1.00] | +0.19 [+0.09, +0.45] |  |
| z_portcullis_raise_slow | FRESH7 | door_open | 0.76 | 0.15 [0.01, 0.35] | 0.14 | 1.00 [0.86, 1.00] | +0.85 [+0.58, +0.99] |  |
| z_sfx_trapdoor_lift_01 | FRESH7 | door_open | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| z_squelch_bog_walk_l | FRESH7 | footstep | 0.54 | 1.00 [0.72, 1.00] | 1.00 | 1.00 [0.95, 1.00] | +0.00 [-0.05, +0.28] |  |
| z_swordplay_deflect | FRESH7 | sword | 0.12 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| z_veh_horn_blare_01 | FRESH7 | horn | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] |  |
| z_wpn_snubnose_shot_02 | FRESH7 | pistol (SA3) | 0.00 | 0.88 [0.74, 0.97] | 1.00 | 1.00 [0.95, 1.00] | +0.12 [+0.02, +0.26] |  |

Judge-capped rows of these sets, kept out of every count. FRESH7 holds no metal step and no rain, so the five capped rows are all known ones.

| row | type | today H | B4 | clean K | pick [95%] | cap |
|---|---|---|---|---|---|---|
| b_iron_boots_step | footstep | 0.09 | 0.00 | 0.00 | 1.00 [0.68, 1.00] | real metal steps 0.17 |
| b_step_metal | footstep | 0.13 | 0.01 | 0.01 | 0.01 [0.00, 0.04] | real metal steps 0.17 |
| g_footstep_metal | footstep | 0.13 | 0.01 | 0.01 | 0.01 [0.00, 0.04] | real metal steps 0.17 |
| w:b_rain_tin_roof | water | 0.09 | 0.07 | 0.07 | 0.04 [0.00, 0.11] | real rain 0.66 on the water judge |
| w:g_rain_on_roof_metal | water | 0.00 | 0.00 | 0.00 | 0.09 [0.02, 0.22] | real rain 0.66 on the water judge |

### What remains, and why

Under the pick 41 decision rows stay under 0.9, and under B4, which stays in Unity, 55. Wrong sound is the largest group either way (20 under the pick, 28 under B4), followed by take luck (13 and 13), soft judge classes (4 and 6) and model or shape (4 and 8). The pick turns 19 of B4's misses into rows at 0.9 and loses 5 (the tank shell, the sneaker, the fishtank, the door kick and the boulder plunge), which is the trade the guard refused.

Eighteen of the pick's 20 wrong-sound rows are FRESH7 names, and on 17 of them no configuration of this round reaches 0.9. Five stay at or under 0.03 everywhere: z_wpn_carbine_tap_03 (AST hears a tick-tock), z_fuel_drum_erupt and z_bomb_dive_pool_01 (music and an explosion), z_windscreen_spiderweb_crack on SA3, and z_heel_click_parquet, which the router sends to SA3 for the word "click" and where PANNs hears finger snapping in 36 of 40 takes. These are names whose words point somewhere else (a "tap" on a carbine, a "bomb" dive), and the caption and the canonical writer both miss the intended sound. The other two wrong-sound rows are the known water steps, which the pick took from 0.61 to 0.34 through the pool's text.

Thirteen rows depend on the take, with intervals that reach 0.9, and the verifier ranks the dominant caption's takes the wrong way on 15 Woosh rows in all (z_slosh_through_flood_02 at AUC 0.25, g_footsteps_concrete_concrete at 0.34). Labels near the line keep moving between seed blocks: both wood steps sat at 0.83 at 820-859 and 0.66 here, both stone steps at 0.98 and 0.85, while both drips went from 0.74 to 1.00. One row's yes or no at 40 takes is weaker evidence than the counts.

Four rows sit on soft judge classes (wood steps at 0.84 of real, snow 0.88, drinking glass 0.47), and today's engine still beats the pick on both wood steps (0.98 against 0.66). Four more carry no other label: z_npc_shuffle_loam_02 (0.43, today 0.85), z_sfx_pane_burst_big_02 (0.50), z_fs_sneaker_linoleum_l (0.60, B4 1.00) and the SA3 Beretta (0.46). Today's engine beats the pick by more than 0.10 on ten rows under 0.9, so the models can make several of these sounds under other words; the phrasebook that finds them is the per-sound table the principle rules out.

### Runtime

The pick passes every gate bound under Unity's environment at a desktop load of 1,249 MiB: startup peak 10,567 MiB (185 to spare), peak after READY 10,859 (562 to spare), READY 9,593 (590 under G0's plus 150), gun first hits 3.96 ms over G0 against the 5 ms bound and Woosh 0.22 ms faster. Its scripted session peaked at 11,013 MiB, 408 MiB inside the margin. It upgrades more slowly: first hit to swap has a median of 84.6 s against 72.8 s for B4 and a session lag of 31.2 s against 20.2 s, because the CPU writer decides each key before Flow renders it, and the lowest free commit falls to 3.95 GB from 4.78.

B4 as Unity runs it passes the gate (peak after READY 11,359 MiB, 62 to spare) and fails the scripted session: 11,511 MiB, 90 MiB past the 150 MiB margin and 60 MiB under 11.3 GiB. The savings are the measured fix. In the served path they changed no sample on 256 of 256 takes at 820-827 (rechecked here from the saved takes) or on the build's 48 Flow and 48 DFlow takes, B4 with them passed the build's gate (V2 margin 574 MiB, gun fill +4.1 ms) and peaked at 10,996 MiB in the same scripted session, and the encoder they move off the card has no caller in the served path. The confirmation rule chooses no configuration besides the pick, so adding the flag to Unity's line is a runtime decision outside this round's verdict. D: holds 24.5 GiB after this check.

### Unity launcher for the p11 round

The pick was not confirmed, so the launcher stays on B4. Leave `pythonExe` (`D:\Python\SAGE\.venv214\Scripts\python.exe`), `workingDir` (`D:\Python\SAGE`), `hfHome` (`F:\AI\cache\hf`) and `port` (8765) as the scene carries them, and set `arguments` to:

```
D:\Python\sarge-research\modal\runpinned.py scripts\run_sidecar.py --generator hybrid --realtime --stereo --amp --detailed-prompts --learn --cache-dir D:\Python\SAGE\.cache\sfx_bake --prompts template:caption+tt --woosh-upgrade flow --woosh-upgrade-solver midpoint:16 --woosh-upgrade-best 4
```

Appending `--woosh-upgrade-vram-save share,encoder` to that line serves the same audio sample for sample and brought the build's scripted session from 11,496 to 10,996 MiB. The pick's line, kept for a later round, swaps `--prompts template:caption+tt` for `--prompts template:caption+tt+clean` and adds `--scene --woosh-upgrade-pool canon --woosh-upgrade-vram-save share,encoder`. In the B4 line every p11 flag stays off.
