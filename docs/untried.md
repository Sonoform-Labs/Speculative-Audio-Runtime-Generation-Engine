# Untried levers

Written 2026-10-22 at HEAD 2baa2ae, with the round-four changes uncommitted. The question was "did u try everything". Until this round the answer was no. Five prompt rounds had all changed the words sent to the same two models, and none had changed the Woosh model, written the best prompt a strong model could manage, or chosen among seeds. This round ran those three levers. One of them works, and it is the model: the caption template rendered on Woosh Flow is the first generic setup to clear all four goal bars on clips rendered after its rule was written. This file records what came before, what the round found and how it was re-checked, the model scan, and what is still untried, in the order I would run it.

## Before this round

### Models and realism routes

The bake-off (`scripts/bench/bakeoff.py`, `data/_gate/bakeoff/`) put SA3 small-sfx, SA3 medium, MOSS-SoundEffect v2, Woosh Flow and Woosh DFlow on the same keys, scored by the judges and played blind to the ear. The hybrid that ships came out of it: guns go to SA3 medium at 146 ms a clip and everything else to Woosh DFlow at 28 ms (`docs/hybrid.md`). The ear preferred Flow on glass, water and quality, and Flow stayed out of the served path because a clip took about a second.

Three routes went after realism head on, and all three closed or stalled. The LoRA on real recordings (`docs/lora-route.md`) ran five times over two caption styles and two step counts, and every judged key fell: rifle 0.94 to 0.18, wood steps 1.00 to 0.09, with a drift ratio of 3 to 8 seed distances from base. The decoder fine-tune (`docs/decoder-ft.md`) cut the 172 Hz comb on generated decodes by 6 to 14 dB and the grain error on real audio from 0.383 to 0.245, yet glass, explosion, sword and grass stayed at 0 under every checkpoint, and two checkpoints still wait for the ear. The `init_audio` anchor route (`docs/anchor-route.md`) lifted punches from 0.00 to 0.86 at noise 0.5, but a third of those clips sit as close to their anchor as a decoded copy does, glass stayed at 0, and an EA patent (US12420192B2) keeps it research only. The one cheap realism fix that held is a prompt word: dropping "dry" brings back gunshot tails and room (`docs/vocab-eval.md`, "Prompt wording").

### Judges

Every verdict here rests on the identity floor of `docs/vocab-eval.md`. Held-out real references (`data/refs_heldout`) set a real rate per judge; a judge counts for a sound type only when it separates those references from negatives and from silence; a clip set scores the minimum over valid judges of min(rate / real, 1) in a 0.6 s window. The judges are AST (`MIT/ast-finetuned-audioset-10-10-0.4593`), PANNs and CLAP (`laion/clap-htsat-unfused`), with Audiobox Aesthetics for production quality and a separate water judge (`docs/water-judge.md`). They measure category identity and nothing about how good a correct sound is; the realism judges built on 2026-10-14 failed their validation (project record), so that gap stays open. Two known biases run in the phrasebook's favour: its demo lines were picked on these judges, and the water judge scores its one splash sentence as perfect water when it answers a drip.

### Five prompt rounds

| round | what changed | unseen minus phrasebook | demo | water |
|---|---|---|---|---|
| 1, 2026-10-17 (`docs/generic.md`) | bare word list, `--prompts generic` | -0.05 (unseen names, 24 judged requests in all) | -0.11 | water steps and splashes lost on identity (whole path -0.13 with `--amp auto`) |
| 2, 2026-10-21 (`docs/generic.md`) | eight one-sentence templates; pick `template:caption+tt` | +0.15 [+0.04, +0.28] on 36 names | -0.11 [-0.30, +0.08] | -0.21, failed its bar |
| 3, 2026-10-21 (`docs/llm-layer.md`, "On sound") | Gemma 4 E2B writer v1 on the CPU | +0.04 [-0.16, +0.27] on 22 rows | -0.13 | -0.29 |
| 4, 2026-10-21 ("Writer styles") | four writer styles; `named` confirmed on FRESH | +0.06 [-0.06, +0.16] on 52 rows | -0.15 | -0.14 |
| 5, 2026-10-21 ("Caption grounding") | 273k real captions; request words before the nearest caption | +0.13 [-0.01, +0.28] on 22 rows | -0.09 | -0.22, failed its guard |

The shape repeated from round 2 on. Generic sources won the sounds nobody tuned for and lost the demo keys and water, and each round traced its losses to vocabulary that Woosh DFlow hears as the wrong sound: "slap" and "thud" come out as punches, short captions come out as ticks.

## What this round found

Each rule went into `data/_gate/untried/eval/` before its first clip (`rules_sha1.txt` holds the digests and file times; the digests still match). The known scenarios ran at seeds no earlier run used (160 to 219) and the 30 FRESH3 scenarios were rendered for the first time. Numbers below are from the harness reports and match the independent recompute described further down.

### The prompt ceiling: verdict B

A strong model wrote one prompt per scenario for all 105 rows from the scene facts alone, one arm with no selection.

| ceiling minus phrasebook | rows | delta | bar |
|---|---|---|---|
| unseen | 52 | +0.12 [+0.01, +0.23] | pass, by a lower bound of +0.004 in the recompute |
| demo | 23 | -0.06 [-0.15, +0.06] | fail (-0.05) |
| guns | 11 | +0.14 [+0.03, +0.30] | pass |
| water | 9 | -0.28 [-0.57, -0.05] | fail |
| ceiling minus `rg_caption`, unseen | 52 | +0.01 [-0.08, +0.10] | |

The line that matters is the last one. On unseen sounds the strong prompts land where the nearest real caption with the request words in front already sits, so a better or distilled small writer has almost nothing left to win under these judges. The strong model fell into the same traps as the small one: its belly flop said "slap" and dropped from 0.73 to 0.09, and the judges' top labels for those clips were "Whip" and "a punch hitting a body". Its drip prompts described real drips and scored 0.61 and 0.90 on the water judge, where the phrasebook's splash scores 1.00. A follow-up written after the ceiling numbers (`rule_1b_hybrid.txt`) kept the phrasebook on its tuned keys and used the caption template on the 48 keys where `prompts.build` falls to its generic frame; it passed all four bars (unseen +0.13 [+0.04, +0.23]), and it is weaker evidence because its rule came after the table was read.

### Woosh Flow: BETTER, and it carries the template over the bars

| same prompts and seeds, Flow minus DFlow | rows | delta |
|---|---|---|
| phrasebook, Woosh rows | 36 | +0.11 [+0.01, +0.23], 25 up and 6 down |
| caption template, Woosh rows | 36 | +0.19 [+0.10, +0.30], 22 up and 3 down |
| water judge, phrasebook | 7 | +0.00 |
| water judge, caption template | 7 | 0.30 to 0.73 |
| Audiobox PQ, phrasebook | 36 | +0.26 [+0.17, +0.36] |

Most of the water gap that sank every generic round was DFlow. Rule 4 then tested the caption template on Flow against what ships today, on new seeds (K2, variants 200 to 219) and the 30 FRESH3 scenarios:

| rule 4, caption on Flow minus phrasebook on DFlow | rows | delta | bar |
|---|---|---|---|
| unseen | 52 | +0.24 [+0.14, +0.35], 38 up and 8 down | pass |
| demo | 23 | -0.01 [-0.15, +0.14] | pass (-0.05) |
| guns | 11 | +0.12 [+0.01, +0.27] | pass |
| water | 9 | -0.03 [-0.14, +0.02] | pass |

Read the passes with their intervals. The demo bar is a point bar and the interval still reaches -0.15. The guns row is the caption template on SA3 medium, since guns never touch Woosh. The template's weak rows on Flow are glass on its short sentence (the bottle shot 0.71 to 0.06, where the phrasebook on Flow scores 1.00) and the wood steps. The price is time: a Flow clip takes 1.0 s at the median (86 function evaluations) against 28.8 ms for DFlow.

The upgrade path is built behind `--woosh-upgrade flow`, off by default (`docs/hybrid.md`, "Flow upgrade"). First hits play DFlow, and idle-time Flow renders of the held takes swap in per key. Gate V formally failed: READY read 9,948 MiB against a 9,830 bound, the same as the 9,950 MiB measured with every flag off that day, because the desktop already held 1,163 MiB. The startup peak (10,446) and the peak with both Woosh models and a gun rendering (11,244, bound 11,571) passed.

### Best of four: small gain, nothing built

`laion/larger_clap_general` picked one take from each group of four. It passed its positive control (median AUC 0.95), but on the served prompt the gain was +0.06 [-0.04, +0.16], which the rule calls a small gain. It helped the ceiling prompts more (+0.13 [+0.02, +0.24]), left Audiobox PQ flat and narrowed the picked takes (median log-mel distance 2.29 against 2.54), a cost for a system that serves several takes per key. One key at six takes would pay about 1.1 s of extra GPU on DFlow and 3.3 s on SA3.

## Independent check (no GPU)

Every table above was recomputed by `data/_gate/untried/eval/verify/recompute.py`, which reads the raw per-seed hits in the saved `vocab*.json`, `water*.json` and `picks.json`, takes each row's judges from the `chosen` table and its real rates from `validity`, and bootstraps rows and seeds with a separate random generator (seed 777). It found no row where those judges disagree with the harness consensus, reproduced every delta to the reported rounding and every interval within bootstrap noise, and returned the same verdicts: B, BETTER, SMALL GAIN, and all four rule-4 bars passing (`recompute_out.txt`). The picks recomputed from the saved verifier scores matched in all 225 row-arm records, and the control's median AUC recomputed to 0.954. The rendered ceiling prompts equal `ceiling_prompts.json` on all 180 scenario rows, and that file's digest matches the one recorded before the first clip.

The ceiling writer stayed blind, on the strongest evidence available: its session transcript. Its 41 tool calls (`verify/ceiling_writer_calls.txt`) read the scenario fact fields, the Woosh README and test scripts, the SA3 package's reprompt guide, and grep hits in `src/sarge` that excluded `prompts.py`, which is where it saw the three `writer.py` lines it disclosed. It never opened `prompts.py`, a doc, a judge output, a report or a writer store, and nothing it saw contains a phrasebook sentence, a judge type name or a reference note ("foley" appears once, inside the SA3 guide's music example). File access times add little: NTFS updates them here with about an hour of slack, the writer's one repo-wide grep (file names only, for "FRESH3") stamped most of `docs` and `scripts` at 00:50:58, and later runs, this check's n-gram pass among them, rewrote the stamps on `prompts.py`, `writer.py`, `world.py` and the writer stores. The text agrees with the transcript (`verify/ngrams_out.txt`). 94 of 105 ceiling prompts share no three-word run holding two content words with their row's phrasebook prompt. The longest shared run, eight words on two ball rows ("rubber ball bouncing hard on a metal floor"), is what the row's phys string spells out, and `world.py` builds the phrasebook line from the same string. The phrasebook's markers ("foley" in 95 of its 105 prompts, "shooter perspective", "close-distance", "rolling away") appear in no ceiling prompt, and the SA3 frame "recorded up close" that the writer saw appears in none.

The verifier is outside the judge panel: `laion/larger_clap_general` appears nowhere under `scripts/` or `src/`, and the judges load `laion/clap-htsat-unfused`, the AST checkpoint above, PANNs and Audiobox. It is a larger sibling of the CLAP judge in the LAION-CLAP family, so independence is partial. The data show no CLAP-specific leak: within a row the verifier tracks AST, PANNs and CLAP hits about equally (r 0.27 to 0.29), and on the phrasebook arm the seed gains split AST +13/-7, CLAP +21/-18, PANNs +24/-16.

The page (`data/_gate/untried/page/index.html`) holds 27 rows and 276 takes. `verify/page_check.py` matched every take to exactly one rendered clip at max abs under 1e-4, checked that the clip's prompt and engine are the ones the key names for that letter, found no missing or unreferenced file, and passed the page script through `node --check`. No arm name appears in the markup; the second section's heading names the two Woosh models, by design.

The runtime files last changed at 05:19:56 and the flags-off end-to-end run started at 05:24, so it measured the final code. `startup_compare.py` re-run on its saved clips gives 75 of 75 identical to `baseline_warm_1`, max abs 0.0, and the CPU suite passes 580 of 580. Reading the diff turned up one robustness gap in the upgrade path, left unfixed because a runtime change needs a GPU gate: a Flow render that fails (as opposed to one preempted) returns the same `None`, and the deepen loop picks the first pending take again after a second, so a key whose Flow render fails every time would stall every later upgrade.

The flow-path round that followed (items 2 and 3 below, the robustness fix, `docs/hybrid.md` under "Cheaper upgrades") got the same treatment, with scripts and outputs in `data/_gate/flowpath/eval/verify/`. Its three rule files match their recorded digests and were written at 08:48, two minutes before the first render. A separate recompute from the raw per-seed hits reproduced every point delta of rules A and B to the printed digit and every interval within 0.013, and the gate rows, VRAM timelines and event logs reproduce rule C, so midpoint 16 stays the only passing solver and `--prompts template:caption+tt --woosh-upgrade flow --woosh-upgrade-solver midpoint:16` the configuration the rules recommend. Its gun fill gate passed at +4.9 ms on three keys, a tie at the 5 ms bound that the render bench resolves for the template (a new gun prompt at 160.0 ms on the phrasebook, 159.5 on the template). `startup_compare.py` on the saved flags-off clips gives 75 of 75 identical, the HEAD identity script finds no difference on the final files, the suite passes 598 of 598, the page check matches all 66 takes, and the diff review found no defect in the retry fix, the first-hit prompt switch or the revision bumps. The configuration still loses the bottle shot (0.00 against 0.60 today) and the glass shatter (0.32 against 0.88) after the swap, plays template DFlow takes for about half of a session's first minute, and has 120 MiB of VRAM margin under the scripted game load, so the ear and one real Play session decide it. The full Unity line is in `docs/hybrid.md` under "Unity launcher".

## Model scan, ranked (from the scan run of 2026-10-22)

The scan found no doc in this repo or `sarge-research` that mentions best-of-N, reranking or a verifier. Speeds marked est. are guesses from published hardware numbers. I re-checked the two licences the ranking leans on: Resonate's repo and model card both say MIT, and PE-AV's card says Apache 2.0 at about 2B parameters. Everything else in this table is the scan's reading of its sources.

| rank | lever or model | licence | fit for SARGE | scan's note |
|---|---|---|---|---|
| 1 | best-of-N on seeds with a verifier outside the judges | n/a | both current models | run this round: small gain on the served prompt |
| 2 | per-key prompt search at prewarm | n/a | one-time cost per key | the ceiling result says little headroom under these judges |
| 3 | prompt distillation into the Gemma writer | n/a | runtime cost of writer v1 | needs item 1's verifier; same headroom problem |
| 4 | Resonate (Interspeech 2026) | MIT | 470M, 44 kHz VAE, 25 steps, 200 to 400 ms per 1 s clip est. | the one open model that could replace Woosh's CC-BY-NC; trained on 10 s clips, no SFX eval |
| 5 | MMAudio, text-only | code MIT, weights CC-BY-NC | S-44k 157M; 8 s clips | AudioCaps CLAP 0.331 to 0.351; same licence class as Woosh |
| 6 | ControlFoley, AC-Foley (reference-audio timbre) | weights CC-BY-NC | video-conditioned as documented | aims at the footstep timbre gap; patent caution as for anchors |
| 7 | AudioX-Turbo | CC-BY-NC | stable-audio-tools stack, 4 steps | cheap second probe of audio-prompt conditioning |
| 8 | RL on the generator (GRPO and kin) | n/a | about 90 GB of GPU memory | AWS or Kaggle only |
| 9 | MeanAudio-L, FdAudio (one step) | MIT; CC-BY-NC-SA | 16 kHz output | a latency tier at best; cuts the top octave glass and guns need |
| 10 | HunyuanVideo-Foley XL | Tencent community licence with EU, UK and Korea exclusions | 48 kHz, 16 GB or slow offload, video required | bake-only research at most |

The scan ruled out TangoFlux (Woosh-Flow beats it on both AudioCaps and the SFX set), ThinkSound and PrismAudio (commercial use needs authorisation, video only), Foley-Omni, Ovi and the LTX-2.3 Foley LoRA (16 kHz or far past 12 GB), SoundReactor (video only, code pending), Foley Control and Kling-Foley (no weights), SA3 Large (API only), and the timing and physics controls (none takes physics state).

## Still untried, in order

Values are my estimates from this round's numbers; costs assume the 3060 and the existing harness.

**1. The ear on the round-four page.** The judges have spoken on 79 of 105 scenarios and on none of the other 26, and they measure category identity in 0.6 s. Thirty minutes of listening to `data/_gate/untried/page/index.html` decides whether the caption template on Flow becomes a default. This is the highest-value step and only you can take it.

**2. Switch the prompt at upgrade time.** The candidate default `--prompts template:caption+tt --woosh-upgrade flow` plays the caption template on DFlow for every first hit, the arm rule 1 measured at -0.14 on the demo and -0.42 on water against today. Letting the first hit keep whatever ships today and handing the template to the Flow re-render removes the only measured regression of that default, and the upgraded takes are then exactly the rule-4 arm. The cost is small: the upgrade request carries the template prompt, `prompt_rev` already moves the client, plus tests, a gate V run and a flags-off end-to-end run, about half a day. A first and a later play of a key would then differ in prompt as well as model, which the ear should hear before it ships. Built after this round as `--first-prompt` (`docs/hybrid.md`, "First-hit prompt, failed renders and the upgrade line"), default `same`, and measured on fresh seeds (variants 220 to 239, same file, "Cheaper upgrades"): the template first hit sits 0.16 under the phrasebook on the demo keys and 0.26 under it on water, +0.05 on unseen names, so `tuned` stays available as a temporary trade-off, off by default. With the faster upgrade below, a key plays its template DFlow take for about 5 s at the median in a scripted session before its Flow take swaps in.

**3. Judge the cheaper Woosh settings that already exist.** DFlow at 6 and 8 steps (41 and 54 ms) and Flow on fixed solvers (rk4 8 steps at 361 ms, midpoint 16 at 364 ms) were rendered for the speed page and never judged on these scenarios. Rule 2 shows the gain comes with Flow, and nobody knows yet whether it is the undistilled weights or the 86 evaluations. If fixed-step Flow holds the gain, upgrades land three times sooner; if DFlow at 8 steps keeps part of it, the first hit gains at +26 ms. My guess is that fixed-step Flow holds most of it and that more DFlow steps add little, since a distilled model usually peaks at its trained step count. About two GPU hours with `llm_eval.py` engine arms and no runtime code until a result. Done 2026-10-22 under a rule written first: Flow at midpoint 16 steps (32 evaluations, 366 ms) scored +0.028 [-0.041, +0.089] on identity and +0.005 on water against dopri5 and was adopted for upgrades as `--woosh-upgrade-solver midpoint:16`, with Audiobox PQ 0.058 lower for the ear to check; rk4 at 8 steps missed the lower bound by 0.004; DFlow at 6 and 8 steps scored 0.08 and 0.07 under 4 steps, so the gain is the undistilled weights and the first hit gets nothing from more DFlow steps. Under Unity's environment the faster solver cut the time from first hit to swap from 61.6 to 9.8 s at the median in the upgrade gate and from 10.1 to 4.5 s in the scripted session.

**4. VRAM and memory under a real Play session.** The upgrade peak is 11,244 of 12,288 MiB with 327 MiB to its bound, the gate harness never renders the Unity scene, and the worker's commit dipped to 2.2 GB against its 1.5 GB load guard. One Play session with the flag on, read from the logs, guards against an out-of-memory in play that no bench sees. It costs one session of yours and a log read. Gate V's READY bound also wants a fresh baseline, since it fails today with every flag off. The 2026-10-22 runs measured READY against a flags-off run of the same day (9,977 against 9,981 MiB with the desktop at 1,194 to 1,197) and kept the upgrade peak at 11,311 MiB in the gate, but the scripted session with its game60 load process reached 11,451 MiB, 120 MiB under the bound, which makes this Play session more pressing once the upgrade is on.

**5. Make the water judge tell a drip from a splash.** Five water rows get the same "Big water splash" sentence from the phrasebook, two of them drips (`b_drip_puddle`, `g_water_drip_water`), and all five score 1.00 on the water judge, the bar that sank four of five prompt rounds. A drip-versus-splash split validated on held-out drips would change which arm wins water and no audio. Half a day of judge validation; medium value.

**6. Best-of-N with a verifier from another family, on Flow.** PE-AV is the scan's pick and would remove the family overlap of this round's verifier. This round's numbers put the expected gain on the served prompt near +0.03 to +0.06, and on Flow each extra candidate costs about a second of idle GPU. Low to medium value, about three GPU hours plus an 8 GB download. Done 2026-10-23 with Woosh-CLAP as the verifier (its text tower is the served Woosh text encoder, so only a 348 MB audio tower is new), together with cfg, the render window and a three-prompt ensemble (`docs/levers.md`): best of four on Flow won the preregistered rule, held up on new seeds and 32 never-rendered scenarios (+0.018 [+0.004, +0.033] pooled, rows at 0.9 from 71 to 79 of 119) and ships behind `--woosh-upgrade-best`, off by default. A second confirmation on 165 held-out rows (42 of the scenarios never rendered before) held it against the plain caption, +0.036 [+0.022, +0.053], and against today's engine, +0.191 [+0.137, +0.246], with 101 rows at 0.9. PE-AV stayed untried.

**7. Resonate as a licence route.** It is the only open candidate that could take the non-gun sounds off CC-BY-NC weights. Its quality on 1 s game sounds is unknown and it trained on 10 s clips. About a day for an install on Windows, a latency bench and a rule-1-style judged run; worth it only if commercial use is on the table.

**8. MMAudio S-44k, text only.** Same licence class as Woosh with strong AudioCaps CLAP. Low to medium value, about a day.

**9. Reference-audio timbre (ControlFoley, AudioX-Turbo).** These go after the footstep timbre gap that the LoRA and the anchors left, at about 0.2 of real on CLAP-hard. Video conditioning, non-commercial weights and the anchor route's patent caution all apply. Uncertain value, one to two days each.

**10. Writer work: prompt search at prewarm, distillation, Phi-4-mini.** The ceiling tied the cheap caption source on unseen rows, so a better writer has little to gain under these judges, and a search scored by a verifier would mostly fit the verifier. Low value until the judges change.

**11. RL on the generator.** Prior art exists (Resonate's GRPO recipe, AJPO, T2A-Feedback), the default config wants about 90 GB, and it would take weeks on rented GPUs. Low priority while item 3 is open.

**12. One-step and video-first models (MeanAudio, FdAudio, HunyuanVideo-Foley).** 16 kHz output or licence exclusions with video required. I would skip them.

The robustness gap in the upgrade path (a key that always fails its Flow render stalls later upgrades) is a small fix to make together with item 2, since both touch the same code and need the same gate runs. It is now in the same build: a failing key retries twice with backoff and is then skipped for the session while later keys keep upgrading, and a forced failure through the `SARGE_UPGRADE_FAIL` bench hook showed exactly that on the GPU (two errors, 6.6 and 10.8 s apart, one skip, later keys kept swapping).
