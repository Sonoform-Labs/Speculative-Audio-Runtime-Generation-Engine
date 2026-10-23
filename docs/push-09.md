# Push 09: the held-out table, rechecked

The goal was "above 0.9 on everything", and on held-out data it fails. Best of four Flow candidates picked by Woosh-CLAP (B4) puts 101 of 165 held-out rows at or above 0.9, against 89 for the plain caption template on Flow (C) and 48 for today's engine (H). Sixty-four rows stay under the line. Seven of them are capped by the judges themselves; the other 57 fail on wording, on take luck or on the shape of short hits, and no lever tried so far moves them generically. This page rebuilds the confirmation from the saved judge outputs with separate code, checks the integrity rules one by one, and lists what each remaining row would need.

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
