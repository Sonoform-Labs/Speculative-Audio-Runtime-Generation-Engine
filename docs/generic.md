# Generic runtime path

The engine should work for any game with no hand-authored per-game tables, no per-sound prompt tuning and no per-category specialists. The audit of 2026-10-17 found the shipped runtime path broke that in four places: a prompt phrasebook keyed to the demo's exact names, a substring family list that also picked the model, demo words baked into the Unity package as fallbacks, and prediction rules written around the demo's naming. This change builds a path that reads only what the request carries, and leaves the user's one standing decision intact: guns render on SA3 medium and everything else on Woosh. What changes is how "gun" gets decided.

Every switch that changes served audio defaults off. With them off, served audio is bit-identical to HEAD 1391339: the end-to-end bench with the shipped scene's flags (`e2e_hybrid.py --extra --learn`) served 75 (key, variant) clips that all match `data/_gate/startup/baseline_warm_1` with a max abs difference of 0.0 and no clip missing, in two runs (`generic_off` mid-change and `generic_off_final` on the final tree, `data/_gate/startup/compare_generic_off.json`). The prediction, Unity package and help-text fixes are on, because none of them changes a served clip.

## Switches

| flag | default | what it does |
|---|---|---|
| `--prompts generic` | `tuned` | every prompt on both routes comes from the request fields alone |
| `--route words` | `families` | guns are chosen by whole words of the event name (and firearm words in the material) |
| `--route class` | `families` | guns are chosen only by a `class` field the game sends; `--gun-class` lists the values (default `weapon`) |
| `--route single` | `families` | every sound goes to `--single-model` (default `woosh`, or `sa3`) |
| `--amp auto` | `--amp` (family table) | attack emphasis, saturation and stereo width come from the rendered clip |
| `--rapid-hz` | 1.5 | prediction only: an event that fires again faster than this stays out of the habit model |

A prompt the game sends always wins, on both routes and under every flag. It rides on a prewarm key or on any event or prediction as `"prompt"`, the server keeps it for that key, and a new prompt for the same key drops the clips cached from the old one. No shipped Unity component sends one, which is why the flags-off proof holds with this always on. The seed never sees the prompt or the class, so a prompt or routing comparison is always seed-matched.

## Prompts

`--prompts generic` uses one template for every sound: the event words, the material words, the intensity word, the context words and, with `--world`, the contact words, joined by commas. Words are split on spaces, underscores, capitals and digits and lower-cased, so `Footstep_Concrete`, `PlayerFootstep` and `Fire1` become `footstep concrete`, `player footstep` and `fire 1`. The protocol defaults `default` (context) and `normal` (intensity) add nothing, so an untouched context means no space phrase at all, where the tuned phrasebook reads it as "exterior, open air". A material whose words all appear in the event name is dropped, which keeps `dog_bark|dog` from saying dog twice.

SA3 prompts start with `TrackType: SFX, `. The SA3 report (section 5.1) says every AudioSparx-trained Stable Audio model, SA3 included, was trained with that prefix on sound effects and recommends it at inference, and the official prompting guide lists it as the tag that "tends to produce more semantically reasonable sound effects". The one local measurement is mixed and was taken on small-sfx with the old prompts: the prefix moved CLAP-hard footsteps -8.8 points with the "dry" ending and +11.9 without it (`sarge-research/realism/report_history.md` section 2.2). Woosh gets no prefix. Its README, test scripts and Gradio demo show plain descriptions, and its conditioner lower-cases the text and strips punctuation before encoding.

| request | tuned (default) | generic |
|---|---|---|
| footstep, gravel | Boots Gravel Footstep, single boot step on loose gravel, heel and toe, coarse crunch with grinding pebbles and grit, close perspective foley, exterior, open air | footstep, gravel |
| shoot, pistol | Gunshot, 9mm pistol, single shot, sharp cracking report with a metallic slide clack, shooter perspective, close, exterior, open field, echo rolling away into the distance | TrackType: SFX, shoot, pistol |
| grappling_hook, rope | Rope grappling_hook, single grappling_hook on a rope surface, a short knock on rope, foley, close-distance, exterior, open air | grappling hook, rope |
| dog_bark, dog | Dog dog_bark, single dog_bark on a dog surface, a short knock on dog, foley, close-distance, exterior, open air | dog bark |
| footstep, ice, cave | Footstep on ice, single boot step, heel and toe, short ice footstep texture, close perspective foley, exterior, open air | footstep, ice, cave |
| explosion, grenade, hard | Grenade explosion, single heavy explosion on a grenade surface, a short knock on grenade, foley, close-distance, exterior, open air | explosion, grenade, hard |
| door_creak, wood, indoor_small | Wood door_creak, single door_creak on wooden planks, a hollow knock on wood, foley, close-distance, inside a small room, short tight reflections off nearby walls | door creak, wood, indoor small |
| impact, psfx_bouncy, soft | Psfx_bouncy impact, single light impact on a psfx_bouncy surface, a short knock on psfx_bouncy, foley, close-distance, exterior, open air | impact, psfx bouncy, soft |

The cost is known in direction and unknown in size. On small-sfx, wording picked per key scored 0.75 of real against 0.50 for a generic prompt, with the window, punch, stone, metal and water steps as the real wins, so expect some demo keys to lose ground. Those numbers were never re-measured on the two models that ship. Every unseen sound stops being described as "a short knock on a surface", and that is the bigger effect for any game other than the demo.

## Routing

`--route families` is HEAD: a substring of `sarge.amp.FAMILIES` in the blast or handling group sends the sound to SA3 medium. `--route words` keeps a word list but matches whole words: a word of the event name in `DISCHARGE_WORDS` (shoot, shot, gunshot, gunfire, fire and their inflections) or `FIREARM_WORDS` (gun, firearm, pistol, revolver, handgun, rifle, carbine, shotgun, smg, musket, cannon and plurals), or a firearm word in the material. The material check keeps the demo's convention of naming the weapon in the material slot on the gun model, so `reload|rifle`, `empty_click|pistol` and `equip|rifle` stay where the bake-off put them. It is still a table. `--route class` needs none: the game declares `class: "weapon"` on the sounds it wants on the gun model (`SargeEmitter.eventClass`, `SargeAutoCapture.eventClass`, `SfxPrediction.eventClass`, or the last argument of `SargeService.Request`), and an event without a class goes to Woosh. Predictions without a class take the last class fired for that event, so a predicted key matches the fired one. `--route single` sends everything to one model. It still loads SA3 medium at startup, so it saves no VRAM yet.

Where each mode sends the audit's names (CPU probe of `sarge.generators.routed.side_for`; `tests/test_generic.py::test_route_modes_on_the_audit_names`):

| event | families (default) | words | class, no hint | class, hint `weapon` | single (woosh) | single sa3 |
|---|---|---|---|---|---|---|
| campfire | SA3 | Woosh | Woosh | SA3 | Woosh | SA3 |
| camera_snapshot | SA3 | Woosh | Woosh | SA3 | Woosh | SA3 |
| magic_missile | SA3 | Woosh | Woosh | SA3 | Woosh | SA3 |
| ice_crack | SA3 | Woosh | Woosh | SA3 | Woosh | SA3 |
| peacock_call | SA3 | Woosh | Woosh | SA3 | Woosh | SA3 |
| keyboard_click | SA3 | Woosh | Woosh | SA3 | Woosh | SA3 |
| pistol | Woosh | SA3 | Woosh | SA3 | Woosh | SA3 |
| cannon | Woosh | SA3 | Woosh | SA3 | Woosh | SA3 |
| laser_zap | Woosh | Woosh | Woosh | SA3 | Woosh | SA3 |
| rocket_launch | Woosh | Woosh | Woosh | SA3 | Woosh | SA3 |
| gunshot | SA3 | SA3 | Woosh | SA3 | Woosh | SA3 |
| fire_alarm | SA3 | SA3 | Woosh | SA3 | Woosh | SA3 |
| WeaponFire | SA3 | SA3 | Woosh | SA3 | Woosh | SA3 |
| shotgun_shell_drop | SA3 | SA3 | Woosh | SA3 | Woosh | SA3 |

Words mode fixes every misroute the audit listed and still sends two probe names to SA3 by mistake, `fire_alarm` and `shotgun_shell_drop`, because "fire" and "shotgun" are whole words there. Laser and rocket names stay on Woosh on purpose: the bake-off judged firearms, and nothing measured says SA3 medium is better at a sci-fi zap or a launch roar. Class mode is the generic answer, since a one-word hint from the game costs nothing and no list on the server can be right for every naming scheme. On the demo keys words mode changes one route: `equip|iron boots` moves from SA3 (it matched "equip") to Woosh. Explosions also move to Woosh, since "explosion" sits in the blast family and in no firearm list.

## Amplifier

`--amp auto` measures four things on each rendered clip after the natural trim and before levelling: the attack time (onset at -20 dB of the loudest 2 ms window to the first point within 3 dB of the peak of the next 100 ms), the crest factor of the body from that onset to the -40 dB point, the spectral flatness of the body (octave bands from 125 Hz to 16 kHz, weighted by band energy), and the decay time to -40 dB (`sarge.post.decay_time`).

| quantity | mapping (`sarge.amp.AUTO`) |
|---|---|
| fast | 1 at an attack of 5 ms or less, 0 at 60 ms or more, linear in log time |
| peaky | crest 10 to 18 dB ramps from 0 to 1, flat to 24 dB, back to 0 at 32 dB |
| impulsive | fast times peaky |
| noisy | flatness 0.2 to 0.7 ramps from 0 to 1 |
| punch | 2 dB plus 4 dB times impulsive (the family table spans 2 to 6) |
| drive | 1 plus 1.4 times impulsive times noisy (the family table spans 1 to 2.4) |
| width | 0.2 plus 0.25 times the decay ramp from 0.15 to 1.0 s (the family table spans 0.2 to 0.45) |

The crest window is a trapezoid because both ends mean the stage has little to work with. A steady hum or engine sits near 10 dB and has no transient to emphasise. A click or a hard step on stone sits near 30 dB, all transient and almost no body, so punch only thins it. Gun bodies measured 20 to 24 dB. Intensity still scales punch and drive by the protocol's soft, normal and hard factors.

Saturation runs only on the body. The tanh stage is blended in where the 5 ms envelope sits within 20 dB of its peak, fading out over the next 10 dB, so the tail keeps its level relative to the peak. If the decay still comes out more than 2% (plus 2 ms) longer, the clip goes out without saturation. On 40 raw clips from both models (20 demo keys and 20 unseen names, rendered without the amplifier), the family table's full-clip saturation lengthened the decay of 5 clips, the pistol from 0.74 to 1.90 s, campfire from 0.19 to 0.72 s and a wooden jump from 0.12 to 0.35 s. Auto lengthened none beyond a steady lightsaber hum moving from 0.03 to 0.04 s, which the attack stage did on both paths. Its saturation was kept on 33 clips, dropped by the decay check on 2 (campfire, engine hum) and not needed on 5.

| clip | auto punch / drive / width | family punch / drive / width |
|---|---|---|
| shoot, pistol | 5.8 / 2.32 / 0.37 | 6.0 / 2.40 / 0.45 |
| shoot, rifle | 6.0 / 2.40 / 0.42 | 6.0 / 2.40 / 0.45 |
| empty_click, pistol | 2.6 / 1.20 / 0.39 | 3.0 / 1.00 / 0.25 |
| break, window | 5.8 / 2.03 / 0.33 | 4.0 / 1.70 / 0.35 |
| bullet impact, glass | 5.7 / 1.00 / 0.27 | 4.0 / 1.70 / 0.35 |
| footstep, stone | 2.9 / 1.33 / 0.41 | 2.5 / 1.25 / 0.20 |
| footstep, gravel | 4.4 / 1.83 / 0.32 | 2.5 / 1.25 / 0.20 |
| footstep, grass | 6.0 / 2.35 / 0.36 | 2.5 / 1.25 / 0.20 |
| splash, water | 6.0 / 2.40 / 0.21 | 2.0 / 1.00 / 0.35 |
| punch, body | 2.0 / 1.00 / 0.20 | 4.0 / 1.70 / 0.35 |
| glass bottle hit, stone | 6.0 / 1.00 / 0.22 | 4.0 / 1.70 / 0.35 |
| engine_hum, car | 2.2 / 1.07 / 0.26 | 2.0 / 1.00 / 0.35 |
| cannon | 4.9 / 1.84 / 0.24 | 2.0 / 1.00 / 0.35 |

Guns land near the family values with no word lookup, and tonal sounds (glass bottle, glass bullet) lose saturation altogether. The grass footstep is the clearest disagreement: Woosh renders it as a sharp, noisy rustle, and on the audio alone it looks like an impact. Expect footsteps on soft ground and splashes to come out denser than today, and judge that on the blind page. Flatness is measured on the first second of the body. The whole stage with width costs about 10 ms of CPU on a 1 s Woosh clip against 4 ms for the family stage, and 41 ms against 16 ms on a 2.8 s gun clip.

## GPU check and blind page

Six arms rendered the same 46 requests through the real sidecar (`--generator hybrid --realtime --stereo --learn`, ports 8767 and 8772, `scripts/bench/generic_probe.py TAG OUT --extra FLAGS`): the shipped flags, `--prompts generic`, `--amp auto`, all three switches together, `--route class` with `class: "weapon"` on the 8 gun requests, and `--route single`. No arm produced an error or a silent clip. Words routing moved explosion, campfire, camera_snapshot, magic_missile, ice_crack, peacock_call and keyboard_click to Woosh and pistol and cannon to SA3; class routing sent exactly the 8 hinted requests to SA3; single sent all 46 to Woosh. The two requests that carried a game prompt (a splash on Woosh, a pistol on SA3) came out bit-identical under the tuned and the generic builder, so the game's text reached both models and beat both builders, and the splash rendered fresh when its prompt arrived after a prompt-free splash. Generic prompts left first hits where they were (guns 157 to 159 ms, Woosh 50 to 50 ms at the median). `--amp auto` added 10 ms to guns (157 to 167) and 4 ms to Woosh (50 to 54), the CPU cost above.

The blind page is `data/_gate/listen/generic_ab/index.html`, built by `scripts/bench/generic_page.py`: 132 pairs in three sections (generic prompts, the auto amplifier, all three switches), each against the shipped flags on the same request, shuffled per row, with picks kept in the browser. Reveal shows which side was which, the model that rendered it and the prompt it got. The two game-prompt requests are left off the page, since they sound the same in both arms by design.

## Judged on the models that ship (2026-10-17)

`scripts/bench/generic_eval.py` renders 35 requests through the served path in process, 40 seeds each: the 12 bake-off keys and 23 names another game would send (the audit's list plus `car_horn|car`, `Pistol_Shot|9mm` and `melee_hit|body`). Woosh goes through `woosh_server.render_key` and `finish_key` with the worker's adopted options, SA3 medium through `StableAudio3Generator` built from `run_sidecar.TIERS` and `HYBRID_GUNS`. Variant 0 of 15 requests per arm, both models, was compared with the clips the real sidecar served in the runs above, and all 45 matched bit for bit. Arms: the shipped flags (`phrasebook`), `--prompts generic`, `--prompts generic --route words`, `--route words` alone, `--route class` with firearms tagged `weapon`, and all three switches (`generic_all`). The held-out vocabulary panel scored the 24 requests that map to a validated type, with the same validity rule and the 0.6 s window as `docs/vocab-eval.md`, and the water judge of `docs/water-judge.md` scored the four water requests. Each number below is the consensus (min over chosen judges of min(rate / real, 1)); a delta is paired on seed, with a 95% bootstrap interval over seeds and McNemar counts per judge. Everything is in `report.json` under the work folder given to `--work`.

| request | phrasebook | generic | generic_all | generic minus phrasebook |
|---|---|---|---|---|
| shoot, pistol | 0.97 | 0.16 | 0.27 | -0.81 [-0.92, -0.67] |
| shoot, rifle | 0.94 | 0.66 | 0.75 | -0.28 [-0.48, -0.07] |
| break, window | 0.77 | 0.61 | 0.58 | -0.15 [-0.37, +0.04] |
| bullet impact, glass | 0.96 | 0.71 | 0.58 | -0.25 [-0.44, -0.09] |
| footstep, gravel | 1.00 | 0.95 | 0.71 | -0.05 [-0.22, +0.07] |
| footstep, stone | 0.34 | 0.85 | 0.78 | +0.51 [+0.27, +0.75] |
| footstep, wood | 0.82 | 0.44 | 0.31 | -0.37 [-0.65, -0.14] |
| footstep, metal | 0.03 | 0.18 | 0.09 | +0.15 [-0.02, +0.33] |
| footstep, water | 0.51 | 0.44 | 0.37 | -0.07 [-0.34, +0.24] |
| splash, water | 0.52 | 0.56 | 0.47 | +0.04 [-0.30, +0.30] |
| punch, body | 0.54 | 0.58 | 0.54 | +0.04 [-0.23, +0.31] |
| dog_bark, dog | 0.88 | 0.78 | 0.75 | -0.10 [-0.23, +0.03] |
| door_creak, wood | 1.00 | 0.76 | 0.68 | -0.24 [-0.41, -0.06] |
| keyboard_click, plastic | 0.45 | 1.00 | 0.56 | +0.55 [+0.32, +0.72] |
| footsteps_Concrete, concrete | 0.71 | 0.98 | 0.88 | +0.27 [+0.10, +0.55] |
| PlayerFootstep, snow | 0.78 | 0.36 | 0.40 | -0.42 [-0.67, -0.09] |
| shotgun_fire, shotgun | 0.91 | 0.90 | 0.93 | -0.01 [-0.14, +0.13] |
| sword_clash, steel | 0.98 | 0.96 | 0.99 | -0.02 [-0.16, +0.10] |
| car_door_slam, car | 0.42 | 0.42 | 0.42 | +0.00 [-0.17, +0.20] |
| explosion_large, debris | 1.00 | 0.92 | 0.84 | -0.08 [-0.18, +0.00] |
| glass_shatter, glass | 0.64 | 0.73 | 0.74 | +0.08 [-0.15, +0.32] |
| car_horn, car | 0.00 | 0.60 | 0.57 | +0.60 [+0.40, +0.73] |
| Pistol_Shot, 9mm | 1.00 | 0.16 | 0.16 | -0.84 [-0.95, -0.67] |
| melee_hit, body | 0.43 | 0.00 | 0.00 | -0.43 [-0.62, -0.23] |
| water judge: footstep, water | 1.00 | 0.76 | 0.59 | -0.24 [-0.45, -0.05] |
| water judge: splash, water | 1.00 | 0.86 | 0.66 | -0.14 [-0.34, +0.00] |
| water judge: water_drip, water | 1.00 | 0.79 | 0.79 | -0.21 [-0.41, -0.03] |
| water judge: rain_on_roof, metal | 0.00 | 0.17 | 0.26 | +0.17 [+0.04, +0.35] |

Over the 24 vocabulary requests generic prompts move the consensus by -0.08 on average [-0.22, +0.06], 8 up and 15 down; the demo keys average -0.11 and the unseen names -0.05. AST and CLAP side with the phrasebook on discordant seeds (+64/-146 and +141/-237), PANNs splits evenly (+151/-129). The whole generic path is worse: -0.13 [-0.26, -0.01], since `--amp auto` costs identity on top of the prompts on gravel and wood steps, the glass bullet and both water steps and splashes. `--route words` alone changes two judged requests and nets zero: the keyboard click gains on Woosh (0.45 to 0.77) and the explosion loses (1.00 to 0.70).

The guns carry most of the loss. SA3 medium turns "shoot, pistol" into a sustained boom: CLAP calls 23 of 40 seeds a grenade explosion, the level 100 ms after the peak is 5.5 dB down where real pistols are 16 dB down, 55% of the energy sits under 500 Hz against 9%, and the noise floor sits at -19 dB against -42. The prefix is no excuse. Without "TrackType: SFX," the rifle falls further (0.66 to 0.33), the shotgun drops to 0.70 and the pistol barely moves (0.16 to 0.14), so the bare word list is what SA3 medium mishears. The phrasebook's fallback sentence makes the case: "9mm pistol_shot, single pistol_shot on a 9mm surface, a short knock on 9mm, foley, close-distance, exterior, open air" scores 1.00 where "TrackType: SFX, pistol shot, 9 mm" scores 0.16. On Woosh the losses are wood steps, the glass bullet, the snow step and the melee hit, which collapses to clicks (CLAP top-1 "a user interface click" on 20 of 40).

The phrasebook breaks wherever an unseen name is something other than a knock. "Car car_horn, single car_horn on a car surface, a short knock on car" never produced a horn in 40 seeds. "Metal rain_on_roof, single rain_on_roof on a steel deck plate, a ringing clang of steel" never produced water. `water_drip|water` became "Big water splash, a heavy plop into deep water", which the water judge scores as perfect water, though it answers a drip with a splash. The door creak win is the door judge's blind spot: the phrasebook said "a hollow knock on wood", CLAP's top-1 for those clips is "a knock" on 27 of 40, and the only valid door judge passes 74% of real door knocks.

Two caveats run in the phrasebook's favour. Its demo prompts were picked on these held-out judges (on small-sfx), so the demo rows carry a selection edge that the unseen rows lack. And every judge here measures category identity in a 0.6 s window, which says nothing about whether a footstep sounds good.

The blind page for these renders is `data/_gate/generic/page/index.html` (35 rows, 496 clips, 4 takes per option, the conventions of `data/_gate/feel/world_page`). Each row shows the distinct audio among the six arms plus the no-prefix diagnostic on SA3 rows, so a row has three to six options. Run order: `render-woosh` in `.venv_woosh`, `render-sa3` in `.venv214`, then `verify`, `routes`, `judge-vocab` and `judge-water` in `.venv`, `report` and `page`; the water folders go under `data/_gate/generic/_water` because `water_judge.py` stores paths relative to the repo.

## Prediction

These changes touch only what gets rendered ahead of time. Served clips keep their bits, which the e2e proof above covers since `--predict-fixes` is on in it.

- `INTENT_IGNORE` is gone. An event that fires again within 1/`--rapid-hz` seconds (0.67 s by default) stays out of the habit model, so a walking bout feeds it one step, whatever the step is called, and full-auto fire feeds it the first shot of each burst.
- The alias that maps a predicted name to a fired one now compares split, lower-cased words in a row, so `footstep` reaches `Footstep_Concrete` and `PlayerFootstep`, and `land` still never reaches `landmine`.
- A prediction may leave `event_type` empty, and both Unity predictors now do by default. The server keeps, per prediction source, the materials and contexts of its latest prediction (for 10 s) and credits any event that fires on one of them. Rapid events and the motion source's current kind only credit the motion source, and an event named in the prediction's `after` never credits it. The kind is the credited event seen on the most distinct predicted surfaces, then the highest score with each credit decaying the others by 0.8, then the most recent. Until a source has a kind its empty predictions are dropped, so the very first step of a session goes unpredicted unless the game names it.
- The `after` expansion no longer keeps successors whose material differs from the trigger's, a rule that assumed the weapon is named in the material slot. It keeps successors that have landed on at least two of the source's predicted surfaces: bullets reach that after hitting two aimed materials, while a reload, which always carries the weapon, never does. Until then the event the prediction names goes out unexpanded.
- The fusion owner `{"footstep": "motion"}` became "the motion source's ETA wins for any key it proposes".

## Unity package

Every component that sent the literal context `"default"` now has a `context` field that falls back to `SargeService.context` (default `"default"`), so one assignment moves the whole game into a cave. Untagged colliders send an empty material: `SargeSurface.material`, `SargeFootsteps.defaultMaterial`, `SargeMotionPredictor.defaultMaterial`, `SargeAutoCapture.defaultMaterial` and `SargeAimPredictor.defaultMaterial` all default to empty, where they guessed grass, stone and metal. A PhysicMaterial name goes out as written, with only Unity's ` (Instance)` suffix cut and no lower-casing; its bounciness and friction already travel in the `phys` block when `sendPhysics` is on. `SargeEmitter` lost its `shoot` and `metal` defaults and warns when asked to emit nothing. `SargeMotionPredictor.eventType` is a new field and `SargeAimPredictor.eventType` and `after` default to empty, which hands the choice to the learned kinds above. `eventClass` and `prompt` reach the wire from `SargeEmitter`, `SargeAutoCapture`, `SfxPrediction` and `SargeService.Request`.

Scenes keep the values they serialized, so WalkArena still sends grass, stone and the demo's names. The demo scripts that add components at runtime now set the old values explicitly (`SargeTestRange` props keep `stone`, the aim predictor keeps `bullet impact`, `shoot` and `metal`, the motion predictor keeps `footstep`, `PSfxDemoSetup` writes `stone` and `grass`), and `SargeFirstPerson` tags its gun sounds and weapon predictions with `class: "weapon"`, which the server ignores outside `--route class`. The package, the demo, the editor scripts, the probe editor scripts and the package sample all compile with Unity 2022.3.8f1's Roslyn, and a .NET run of `SargeService` and `SargeClient` against a recording server showed the class and prompt arriving, a null material sent as empty, a null context taking the service context, and an empty motion prediction dropped until the first step credited it. The demo files were backed up to `D:\PSfxUnityTest_backup_20261017` first.

## Tables that remain in the runtime path

| table | where | runs when | why it stays |
|---|---|---|---|
| tuned phrasebook: `SPACES`, `SPACE_ALIASES` (with "walkarena"), `GUNS`, `SURFACE_WORDS`, `STEPS`, `BALLISTIC`, `PROPS`, `BREAKS`, `INT`, `EVENT_VERBS` and the exact-name branches of `build` | `src/sarge/prompts.py:10-119, 220, 279` | `--prompts tuned` (default) | flags-off audio stays bit-identical until the blind page decides; `--prompts generic` retires all of it |
| `UNSET_WORDS` ("", "default", "normal") | `prompts.py:8` | generic | the protocol's defaults meaning "unset" |
| `SA3_PREFIX` "TrackType: SFX, " | `prompts.py:6` | generic, SA3 route | the vendor's recommended tag; Woosh has none |
| `CONTACT_WORDS` | `src/sarge/world.py:49` | generic with `--world` | the one validated word set (79 of 90 seed pairs) |
| world model: `MATERIALS`, `NOUN_MATERIAL`, the density ladder, `GROUNDS`, `FRAGMENTS`, `BREAK_WORDS` | `world.py:6-56, 88-107, 223-230` | `--world` only | they turn the physics block into the bucket and the decay shape; generic prompts take only its contact class, and `--world` is off by default |
| `FAMILIES` (substrings) | `src/sarge/amp.py:7` | `--route families` (default), `--amp` / `--amp full` / `--amp punch` | flags-off identity; `--route words/class/single` and `--amp auto` retire it |
| `WIDTH` by family | `src/sarge/stereo.py:7` | `--stereo` without `--amp auto` | same |
| `DISCHARGE_WORDS`, `FIREARM_WORDS` | `src/sarge/generators/routed.py:17-21` | `--route words` | a whole-word list by design; class and single carry none |
| `GUN_CLASSES` ("weapon") | `routed.py:16` | `--route class` | the contract value, set with `--gun-class` |
| `WARM_KEY` shoot, pistol | `routed.py:13` | startup | warms the gun model; the clip is thrown away and affects speed only |
| `HYBRID_GUNS`, `WOOSH_COST_MS`, gun cost 155 ms | `scripts/run_sidecar.py:40, 45` | hybrid | settings and cost per model; the route decides which applies |
| `FRAMES` by family, `--woosh-frames` family pairs | `scripts/woosh_server.py:39` | hybrid | inert, every family gets 101 frames |
| `AUTO` constants, `INTENSITY_SCALE` | `amp.py:17-21` | `--amp auto` / any `--amp` | thresholds on measured audio and the protocol's intensity scale; no words |
| `RAPID_HZ`, `KIND_WINDOW_S`, `KIND_DECAY`, `MOTION_SOURCE` "motion" | `src/sarge/sidecar/server.py:26-29` | prediction | rate and window constants and the motion predictor's protocol source name |
| suffixes s, es, ed, ing | `src/sarge/words.py:17` | prediction alias | a four-ending English inflection rule |
| warm-up key `CacheKey("warmup", "grass")` | `server.py:638, 687` | startup and keep-warm | never served |
| v1 prompts: `_PROMPT_TEMPLATES`, `_MATERIAL_ADJ`, `_FOOTSTEP_PLAIN`, `_IMPACT_EVENTS`, `_IMPACT_TAIL`, `_SPLASHY` | `src/sarge/generators/stable_audio.py:9-50` | `--generator sa3` without `--detailed-prompts` (the launcher's default arguments), `--generator stableaudio` | reachable outside the hybrid and pinned by `tests/test_footstep_prompt.py`; `--prompts generic` replaces them for sa3 too |
| `detail_keys` splash, water, liquid | `src/sarge/generators/stable_audio3.py:173` | `--generator sa3` tiers (8 steps against 3) | inert under the hybrid, where both counts are 8 |
| `ITEM_SOUNDS`, `ITEM_MATERIAL`, `SURFACE_SOUNDS` | `src/sarge/affordance.py:9-16` | no `--realtime` | the prefetch planner of the non-realtime mode; dormant in the shipped config |
| `high_rate_events` {footstep, shoot, gunshot} | `src/sarge/affordance.py:68` | no `--realtime`, and under `--realtime` for keys a game prebakes (8 background variants against 6) | no shipped component sends a prebake, so it is dormant in the shipped config; the server's measured fire rate (`--rapid-hz`) could replace it, which would change the variant count of prebaked keys and so belongs behind a flag |
| `_CONTINUOUS` | `src/sarge/affordance_learn.py:8` | `--learn` without `--realtime` | dormant in the shipped config; the `--learn` help now says so |
| `REALTIME_TOLERANT` | `src/sarge/router.py:3` | `--learn` without `--realtime` | same |
| `_MATERIAL_PROFILE` | `src/sarge/generators/mock.py:9` | `--generator mock`, authored fallback without `--realtime` | test tones; tests depend on it |
| Unity `SargeFootsteps.eventType` "footstep", `SargeAutoCapture.eventType` "impact" | `unity/Sarge/Runtime` | those components | each component's default event name, an Inspector field |
| Unity `persistentPredictionSources` | `SargeService.cs:24` | `resendPredictions` (off) | empty in the package since the verification pass; the demo's `SargeOnTime` preset sets it to its `equip` source |

No table was dead. A search for module-level names in `src/sarge` that nothing else references found only the constant `post.LEVELS`, so nothing was deleted: every table above runs under some flag or mode, and the old `stable_audio.py` tables still serve `--generator sa3` and its tests. The demo scripts outside the package (`SargeFirstPerson`, `SargeMix`, `SargeTestRange`, `SargeBreakable`) keep their demo vocabulary, since they are the demo.

## Tests

`tests/test_generic.py` (31 tests) covers the word splitter, the generic template and its no-new-words property, the contact words, the tuned style staying identical to `build`, the SA3 prefix and the game-prompt override on both routes (SA3 in-process and Woosh through `RemoteGenerator`), a per-event prompt that re-renders when it changes, routing in all four modes on the audit's names and the demo keys, the class field reaching the key only under `--route class` and never the seed, predictions taking the last fired class, the split-word alias, learned kinds for empty motion and aim predictions with steps staying with motion, rapid names crediting only motion, the two-surface `after` rule, the motion owner, `--amp auto` ignoring the event name, its saturation never lengthening the decay while plain tanh does, auto width following the decay, both finishing chains, and the new `run_sidecar.py` flags. `tests/test_intent_server.py` drives a fake clock and checks the rate filter on any name.
