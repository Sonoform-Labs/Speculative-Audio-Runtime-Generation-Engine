# Prompt writer (the LLM layer)

The phrasebook breaks on any sound it was never tuned for (car horn 0.00, rain on a roof rendered as a steel clang, a drip answered with a splash), and the bare word list collapses guns on SA3 medium (pistol 0.97 to 0.16, `docs/generic.md`). The prompt writer replaces both with one generic step: a small CPU language model reads what the engine knows about a moment (the event, the object, what touched what, who made it happen, the room) and writes a prompt for that exact scenario. SARGE then puts the three short fields it returns into the sentence form each audio model is known to follow. Nothing in it is written per game or per sound, so a sound the demo never had gets the same treatment as a footstep.

Everything here is off by default. With every new flag off, the end-to-end bench under Unity's environment (`e2e_hybrid.py --extra --learn`) served 75 of 75 clips bit-identical to `data/_gate/startup/baseline_warm_1` with a max abs difference of 0.0 and equal shapes, in two runs (`llm_off` mid-change and `llm_off_final` on the final tree, `data/_gate/startup/compare_llm_off.json`), and the server replies are byte-identical with and without the new wire fields (`tests/test_writer.py`).

On sound the writer does not earn its flag yet. Judged on 45 scenarios over variants 40 to 79 (section "On sound"), it ties the phrasebook on sounds nobody tuned for (+0.04 [-0.16, +0.24]) and loses to it on the demo (-0.11 [-0.24, +0.03]), about where the caption template already sits. It wins big on the car horn, the keyboard, the wine glass and the car door, and it loses where its sound words name an impact (wood and concrete steps, the splash) or where it renames the event (the exploding barrel becomes "Barrel break").

## What "jev" was

Almost certainly Jev from TypeSafe AI, launched on 15 September 2026 as a "System One" decision model. It answers typed questions (yes or no, a choice, a score) with calibrated probabilities at 70 to 500 ms per call. It writes no text, its weights are closed and it runs only as a hosted service, so it fits none of SARGE's rules. The second reading is JEPA (Meta's joint-embedding world models, LeWorldModel at about 15M parameters being the "barely anything to run" one). JEPA models embed video and flag surprising frames; they write no text, and they would only estimate from pixels the masses, speeds and materials Unity already holds exactly. The "understanding" the user asked for therefore comes from two cheap sources: the engine's exact state, extended with a scene record, and a small LLM's knowledge of how real things sound. The Jev idea that survives is a local yes or no gate with a confidence value; it is designed (section 4 of the design) and left unbuilt.

## How a sound gets its prompt

1. Unity sends an EVENT, PREDICT or MANIFEST with the usual fields, plus an optional `scene` object when `SargeService.sendScene` is on.
2. Under `--scene` the server quantises `scene` into a new `CacheKey.scene` string, the way `--world` turns `phys` into `CacheKey.phys`. The seed ignores it, so a scene key and its plain key draw the same noise.
3. The first hit of a key with no written prompt plays the configured builder (`--prompts`, the tuned phrasebook by default, or the `template:` pair once the template search lands) and the key enters the writer queue. Fired keys go first, predicted keys next by ETA, manifest keys last.
4. A writer thread in the sidecar sends one request at a time over loopback HTTP to a CPU-only `llama-server` it spawned after READY. It never holds the engine lock while it waits.
5. The validator checks the answer and repairs it deterministically; an accepted answer is stored on disk and becomes the key's prompt on its route (SA3 form for guns, Woosh form for everything else).
6. A key that has no clips yet takes the prompt at once, so its first render already uses it. A key that already played its template take gets a background render of take 0 with the new prompt; the swap happens under the lock once that render has landed, and the key's `prompt_rev` goes up by one. A game-sent prompt always wins, before or after the writer's.

## Switches

| server flag | default | what it does |
|---|---|---|
| `--writer off\|llama\|ollama\|store` | `off` | `llama` spawns `llama-server` on the CPU for `--writer-model` after READY; `ollama` uses an Ollama server already running on `--writer-port`; `store` serves a baked prompt file with no model at all |
| `--writer-model` | `gemma4:e2b-it-qat` | a GGUF file or an Ollama tag, resolved through the manifests under `OLLAMA_MODELS`, `F:\OllamaModels` or `G:\sarge\llm\ollama_models` |
| `--writer-bin` | `SARGE_LLAMA_SERVER`, else `F:\Ollama\lib\ollama\llama-server.exe` | the llama.cpp server binary |
| `--writer-port` | 11540 | loopback port of the model server |
| `--writer-threads` | 4 | `-t` and `-tb` of the model server |
| `--writer-affinity` | `0xEFF` | CPU mask enforced on the model server (core 8 off) |
| `--writer-store` | `.cache/writer` | folder of `prompts_<writer id>.json`, or the file itself with `--writer store` |
| `--writer-when` | `seen,predict,manifest` | which triggers queue a key |
| `--writer-swap revalidate\|next-miss` | `revalidate` | render take 0 with the new prompt in the background and swap it in, or drop the template takes so the next hit renders it |
| `--writer-readonly` | off | serve the store, never start the model or write |
| `--writer-yield` | on with the writer | suspend `llama-server` while the sidecar renders on the GPU |
| `--scene` | off | quantise the `scene` object into the key; the writer also works without it, on the plain keys |

| Unity field | default | what it does |
|---|---|---|
| `SargeService.sendScene` | off | package components attach the scene record to events and predictions |
| `SargeService.roomProbe` | off | seven raycasts at the AudioListener every 0.5 s classify the room (1 s hysteresis) |
| `SargeService.sendManifest` | off | on each scene load, send every `SargeEmitter` key and every `SargeFootsteps` key on every `SargeSurface` material, 100 keys per message |
| `SargeEmitter.madeBy` | empty | overrides the detected source (player, npc, physics, projectile, world, ui) |
| `SargeAimPredictor.SceneResolver` | null | a game hook that states the scene its hit code will send |

## What the writer reads

The writer sees the quantised key and nothing else, rendered as plain lines, so one key always gives one input:

```
event: melee hit
material: body
made by: npc (orc brute)
object: club wooden
action: hit
force: hard
room: open
```

A physics key adds the object and the surface it touched, each as the developer's name plus the bucket's material and noun, and the contact class:

```
event: impact
made by: physics
object: auto capture ball, rubber ball
touches: surface metal, metal floor
contact: soft
action: bounce
force: hard
room: small
```

Two of the benchmark's findings changed this format. Size and build words (large, inflated, hollow, static) are gone from the physics lines, because 38 of 40 benchmark physics outputs repeated them and `docs/physics-context.md` measured that "inflated" makes Woosh brighter. Name cleaning now drops colour words, which every model had carried from `Lightsaber_Blue` into the prompt.

### The scene record

| field | values | how Unity fills it | in the key |
|---|---|---|---|
| `src` | player, npc, physics, projectile, world, ui | a Canvas parent gives ui; a root that holds the AudioListener or carries the Player tag gives player; a root with a NavMeshAgent or Animator gives npc; an AutoCapture body with continuous collision detection, or one under 3 s old moving faster than 20 m/s, gives projectile (decided once per body); other captured bodies give physics; everything else is world | yes |
| `who` | the root's name | `transform.root.name` | npc only, up to 3 cleaned words |
| `obj` | the emitter's name | the GameObject name; for a physics event the server takes `phys.a.name` and adds `on` from `phys.b.name` | up to 3 cleaned words |
| `act` | hit, bounce, break, slide, roll, step, use, loop | footsteps send step, a looping AudioSource sends loop, emitters send use; for physics the server takes `world.kind_of` | yes |
| `room` | open, small, medium, large, corridor | an upward 50 m ray that misses gives open; otherwise six 30 m axis rays give a box: a horizontal aspect over 4 with a short side under 4 m is a corridor, under 60 m³ small, under 1000 m³ medium, larger large; inside an AudioReverbZone its preset decides | yes |

Name cleaning is the same for every game: split on camel case, underscores and digits, lower-case, drop digits, engine words (cube, sphere, capsule, cylinder, plane, quad, clone, instance, prefab, lod, mesh, variant, gameobject, object, geo, sm, fx, sfx, prop, go, new, a, b, l, r) and colour words, keep 3 words. `Barrel_Rusty_02` becomes "barrel rusty", `Cube (12)` becomes nothing. The key is a fixed-order string such as `src=npc;who=orc brute;obj=club wooden;act=hit;room=open`, appended to the key string as `|scene=...` only when it is non-empty.

Physics keys use the contact kind and leave slides and rolls out of the key. A predicted contact and the real one often disagree on the tangential speed, and a key that flickers between hit and slide would make every impact prediction miss.

## The model

Gemma 4 E2B-it QAT (Apache-2.0) is the default, served by `llama-server` with a compact grammar. It won the 40-scene blind grading at 9.22 of 10 against 8.47 for Phi-4-mini (MIT), the only model that never named the player or an npc, and on the final code it ran 40 of 40 scenes with no rejection. Phi-4-mini is the second model worth an audio test, from a different family; Qwen3 1.7B (Apache-2.0, 1.36 GB) is the small fallback. Granite 4.0 350M and Qwen3 0.6B only make sense as distillation students.

| on the final code, 40 scenes, 4 threads | Gemma 4 E2B-it QAT | Phi-4-mini |
|---|---|---|
| file | 3.35 GB | 2.49 GB |
| load to ready, first request | 8.3 s, 8.6 s | 9.1 s, 9.8 s |
| per prompt p50 / p95 | 1.68 / 1.94 s | 2.70 / 3.03 s |
| rejected / repaired | 0 / 3 | 0 / 15 |
| peak working set (private) | 2.52 GB (1.33 GB) | 3.91 GB (1.64 GB) |
| same text in reversed order | 40 of 40 | not run |
| longest Woosh / SA3 prompt (real tokens) | 33 / 43 of 70 / 128 | 33 / 44 |

Gemma's three repairs were all size or loudness words ("loud, brief honk" became "brief honk"). Phi's fifteen were mostly "a player steps" and "metallic echo". Through Ollama the same Gemma peaks at 4.6 GB, because Ollama loads its vision projector and turns off mmap, so the llama-server path is the one to use.

### Prompt and composer

The system prompt and the four examples are the design's, with the size and build words taken out of the physics example to match the new input. The model answers three fields under a GBNF grammar: `source` (2 to 5 words, 40 characters), `action` (4 to 14 words, 90) and `sound` (2 to 6 words, 60). Decoding is temperature 0, top_k 1, seed 7, 96 tokens, a 2048-token context, batch 512.

| route | composition |
|---|---|
| Woosh | `{source}, {action}, {sound}[, contact words][, room words]` |
| SA3 | `TrackType: SFX, {Source}, {action}, {sound}[, contact words], a single sound recorded up close[, room words].` |

Contact words come from `world.CONTACT_WORDS`, the one validated set. Room words are outdoors, in a small room, indoors, in a large hall and in a corridor; a key without a room falls back to the words of its context. Both forms are stored, so changing `--route` needs no rewrite.

Examples from the final run (Woosh form):

| scenario | prompt |
|---|---|
| rain on a tin shack roof | Tin roof rain, steady rain hitting a tin roof, patter, metallic drumming, in a small room |
| car horn | Car horn, a car horn emits a sharp blast, brief honk, outdoors |
| water drip in a puddle | Water dripping, water falling from a puddle onto a surface, plink, steady drip, indoors |
| ball on the metal tile | Ball bounce, a rubber ball bounces off a metal surface and onto a metal floor, dull thud, soft bounce, a soft muffled bump, outdoors |
| crate onto gravel | Crate impact, a wooden crate striking a gravel floor, dull thud, wood resonance, a dull thud, outdoors |
| troll boss growl | Troll growl, a deep guttural sound from a troll, low, rumbling growl, in a large hall |
| pistol (SA3 form) | TrackType: SFX, Pistol discharge, a metallic pistol firing into the air, sharp crack, sharp report, a single sound recorded up close, outdoors. |

### Validator and repair

The checks are generic and none of them names a sound:

- the answer parses and every field is non-empty;
- no 3-word sequence repeats inside one field (the benchmark's loop flags were all a title repeated in the action, so the check no longer crosses fields);
- no 4-word sequence inside a field is copied from an example unless all its words are in the input;
- no example noun (tin, can, counter, frog, croak, wind, desk, drawer, derived from the example inputs) appears unless the input has it; the drawer example had put four models' punches on a desk;
- at least one content word of the source or action matches the input;
- no colour, light, smell, loudness, distance, echo, reverb, size or build word that the input lacks, and never player, enemy or npc;
- no digits the input lacks, an action of at least two words, and both compositions inside their token budgets (an approximate count that never undercounted the real roberta or T5Gemma tokenizer on 600 benchmark outputs).

A failed check is repaired by rule, up to three passes: a banned word is deleted (with its article for "a player"), a comma item of the sound field that holds a flagged word is dropped, and a copied or leaked phrase in the action is cut from the preposition or article before it. The benchmark showed that a seed-8 retry returns the same text at temperature 0 in 36 of 37 cases, so there is no retry. An answer still flagged after repair is stored as rejected for this writer id and the key keeps its template.

## Store and determinism

Records live in `.cache/writer/prompts_<id12>.json`, keyed by the key string, holding the slots, both compositions, the flags, the raw answer, the input text and the time it took. The sidecar's startup cleanup removes only `*.wav` files in its audio cache, so the store survives restarts. Writes are atomic (a temporary file, then `os.replace`) and debounced to one second. The writer id is a sha1 over the model file digest, the runtime version line and binary size, the thread count, the system prompt, the examples, the grammar, the decoding options and the composer constants; the Gemma setup above is `07d28eff9ea0`.

Temperature 0 gave identical text after a fresh load in every benchmark model, while a different request order inside one session changed up to 5 of 40 texts. The store is therefore the guarantee: a key's first accepted text is kept until the writer id changes, and a stored key is never sent to the model again. At startup the writer loads the store and hands every stored prompt to the generators before the first hit.

A game can bake its store at build time from its manifest and ship the JSON; `--writer store --writer-store prompts_x.json` then serves it with no model on the player's machine, which is the real "barely anything to run" path.

## Where it runs

The sidecar spawns `llama-server` suspended, puts it in a Windows job object with affinity `0xEFF`, BelowNormal priority and kill-on-close, then resumes it, so the model server can never run on core 8 and dies with the sidecar even after a hard kill. Its command line is `-ngl 0 --device none -t 4 -tb 4 -np 1 -c 2048 -b 512 -ub 512 --jinja --no-webui --threads-http 2 --cache-ram 0` with `CUDA_VISIBLE_DEVICES=-1`, and it uses no GPU memory. It starts after READY, so startup is untouched: READY came at 49.9 s with the writer on against 50.7 s with every flag off, and VRAM at READY was 9,925 MiB in both runs.

With `--writer-yield` (on with the writer) the sidecar suspends the model server for the length of every GPU render and resumes it after; the counter nests for overlapping renders. Two paired sessions on the same 26 unseen keys (`data/_gate/llm/lat_s1_*`, `lat_s2_*`) measured the cost of a busy writer and what the yield buys back:

| first hits on unseen keys, writer busy | without yield | with yield, paired by key |
|---|---|---|
| Woosh, median reply | 58.8 ms | 4.0 and 4.3 ms faster |
| guns, median reply | 177.8 ms | 5.2 and 5.1 ms faster |
| control: the same keys with the writer idle | | 0.7 and 0.2 ms apart |

Two identical sessions without the yield differed by 0.6 ms on the busy Woosh keys, so the 4 ms is real and small. These first hits run slower than the e2e bench (about 55 ms against 34 ms for Woosh) because every key is a brand-new prompt that misses the text-encoder cache; that cost belongs to any unseen sound.

## Live check on the real sidecar

`--generator hybrid --realtime --stereo --amp --detailed-prompts --learn --scene --writer llama`, Unity's environment, ports 8767 and 8772 (`data/_gate/llm/live/`, clips under `wav/`): 14 keys fired once (demo keys, the generic-eval failures and two scene-bearing ones), then fired again after the writer finished, then a 20-key manifest kept the writer busy while 8 more unseen keys fired.

- The writer was ready about 10 s after READY (model load plus the first request at 8 s); after that each prompt took 1.4 s at the median.
- All 14 second plays came back `cached` with `prompt_rev` 1 and different audio; 22 swap renders took 123 ms at the median and happened only while no realtime render held the GPU.
- 42 keys were written with no rejection and 3 repairs; the manifest keys took their prompts before any hit, so they never needed a swap.
- First plays right after READY came back at 160 ms for guns and 51 ms for Woosh at the median, in line with the flags-off bench for new prompts. The 8 unseen keys fired while the writer worked through the manifest came back at 72 ms at the median; one of them waited 81 ms behind a swap render on the Woosh worker, which a realtime miss does behind any background render. This run predates `--writer-yield`; the paired sessions above are the measure of the CPU cost.

That run surfaced one repair bug: deleting "loud" from "a loud, sustained blast" left "a, sustained blast". The repair now drops a comma that follows an article, and a test pins it.

## On sound (2026-10-21)

The writer was judged on audio against the two builders it would replace. `scripts/bench/llm_eval.py` reuses the render path, the judges, the statistics and the page of `generic_eval.py` and renders 75 scenarios on the shipped models: the 40 bench scenes with their scene record (`data/_gate/llm/scenes40.json`, served as `--scene --world` keys, so the six physics scenes carry their bucket and every arm goes through the same `--world` chain) and the 35 plain requests of the generic run. Each scenario gets three prompt sources on the same key and the same seeds: the phrasebook (the shipped default), the caption template (`--prompts template:caption+tt`, the template search's pick, committed in 95be80e), and the writer at its defaults (Gemma 4 E2B-it QAT, writer id `07d28eff9ea0`, a CPU `llama-server` confined to mask `0xEFF` with 4 threads). The route is the shipped `families`, so the writer's SA3 form goes to whatever that route calls a gun and its Woosh form everywhere else. Variants 40 to 59 were new to every arm, 20 per arm, with `--amp full` and stereo as served.

The writer wrote all 75 prompts in 1.56 s at the median (p95 1.92 s) with no rejection and five repairs, every one of them a deleted "loud" or "small". On 38 of the 40 bench scenes it returned the same raw text as the stage A run; the two shots differ because the families route sends no class line. The harness hears what the game hears: phrasebook and caption clips at variants 20 and 21 of five requests matched the template confirm run by sha1, 20 of 20. Forty-five scenarios map to a validated held-out judge type and seven to the water judge, with the validity rule, the 0.6 s window and the consensus of `docs/generic.md`. The question and its bars went into `preregistered_rule.txt` in the work folder before any judge ran.

The writer does not beat the phrasebook on unseen scenarios. Over the 22 judged unseen rows it moves the consensus by +0.04 [-0.16, +0.27], 11 rows up and 10 down, a tie by the rule; with the water rows it is +0.00 [-0.19, +0.20]. It trails the caption template there by 0.04 [-0.21, +0.14]. On the demo it loses 0.13 [-0.27, +0.00] over 23 rows, 5 up and 15 down, and the caption template loses about as much (-0.16 [-0.29, -0.01]); against each other the two generic sources tie on the demo (+0.03). AST and CLAP side with the phrasebook on the demo seeds where they disagree (+13/-39 and +26/-81), PANNs splits (+70/-79). On the water judge the writer falls 0.29 [-0.61, -0.05] below the phrasebook. Guns hold at 0.94 against 0.97. By the rule the writer fails the bar for the demo (-0.05), fails the comparison with caption on all rows (-0.01, the bar was above zero) and fails the water bar (-0.15), so it stays off.

| rows (variants 40-59) | n | phrasebook | caption | writer | writer minus phrasebook, up/down | writer minus caption | caption minus phrasebook |
|---|---|---|---|---|---|---|---|
| every judged scenario | 45 | 0.66 | 0.62 | 0.62 | -0.05 [-0.17, +0.09] 16/25 | -0.01 [-0.11, +0.10] | -0.04 [-0.15, +0.08] |
| unseen, both sets | 22 | 0.65 | 0.74 | 0.70 | +0.04 [-0.16, +0.27] 11/10 | -0.04 [-0.21, +0.14] | +0.08 [-0.08, +0.25] |
| unseen, bench scenes | 9 | 0.64 | 0.67 | 0.69 | +0.05 [-0.32, +0.39] 5/4 | +0.02 [-0.35, +0.36] | +0.03 [-0.16, +0.24] |
| unseen, generic run | 13 | 0.66 | 0.79 | 0.70 | +0.04 [-0.22, +0.31] 6/6 | -0.08 [-0.24, +0.08] | +0.12 [-0.11, +0.36] |
| demo, both sets | 23 | 0.67 | 0.51 | 0.54 | -0.13 [-0.27, +0.00] 5/15 | +0.03 [-0.10, +0.16] | -0.16 [-0.29, -0.01] |
| demo, bench scenes | 12 | 0.62 | 0.50 | 0.54 | -0.09 [-0.25, +0.09] 3/8 | +0.03 [-0.16, +0.21] | -0.12 [-0.29, +0.08] |
| demo, generic run | 11 | 0.73 | 0.52 | 0.55 | -0.18 [-0.41, +0.03] 2/7 | +0.03 [-0.16, +0.22] | -0.21 [-0.41, -0.01] |
| guns | 6 | 0.97 | 0.98 | 0.94 | -0.04 [-0.21, +0.09] 3/1 | -0.04 [-0.21, +0.08] | +0.00 [-0.09, +0.10] |
| footsteps | 15 | 0.56 | 0.51 | 0.43 | -0.13 [-0.31, +0.07] 3/11 | -0.07 [-0.26, +0.10] | -0.05 [-0.22, +0.13] |
| rows SA3 renders | 9 | 0.89 | 0.98 | 0.86 | -0.03 [-0.33, +0.24] 4/2 | -0.13 [-0.35, +0.04] | +0.09 [-0.04, +0.29] |
| rows Woosh renders | 36 | 0.61 | 0.53 | 0.56 | -0.05 [-0.18, +0.10] 12/23 | +0.02 [-0.10, +0.14] | -0.07 [-0.20, +0.06] |
| water judge | 7 | 0.73 | 0.48 | 0.44 | -0.29 [-0.61, -0.05] 1/6 | -0.04 [-0.29, +0.18] | -0.25 [-0.51, -0.04] |
| unseen plus their water rows | 26 | 0.63 | 0.67 | 0.64 | +0.00 [-0.19, +0.20] 12/13 | -0.03 [-0.18, +0.12] | +0.03 [-0.12, +0.18] |
| demo plus their water rows | 26 | 0.71 | 0.54 | 0.55 | -0.16 [-0.30, -0.02] 5/18 | +0.01 [-0.12, +0.13] | -0.17 [-0.29, -0.04] |

The writer wins where the phrasebook has nothing to say, which is also where the caption template wins. The plain car horn goes from 0.00 to 1.00 on "Car horn, a vehicle emits a sustained blast, continuous honk", the bench horn from 0.23 to 0.74, the keyboard click on SA3 from 0.14 to 1.00 and the typing loop from 0.28 to 1.00. Two wins are the writer's alone: the wine glass that falls and shatters (0.32 for the phrasebook, 0.00 for caption, 0.88 for "Glass shatter, a glass object breaking into sharp pieces, sharp crack, tinkling debris, indoors") and the car door slam (0.35, 0.25 and 0.65). The water step gains on both sets (0.54 to 1.00 and 0.90), and the bench bark, whose material is empty, rises from 0.35 to 0.70 where the phrasebook says "single bark on a  surface".

The losses follow two patterns. The first is an impact word in the sound field that Woosh turns into a punch: "Wood footstep, a foot stepping onto a wooden surface, solid thud, dull thud" scores 0.09 where the phrasebook's "resonant hollow wood tone, small creak" scores 0.88, and CLAP calls all 20 seeds "a punch hitting a body". The concrete step (0.54 to 0.09, "solid thud, dull impact"), the splash (0.77 to 0.00 on the splash judges and 1.00 to 0.00 on the water judge, "Water splash, water hitting a surface, sharp wet slap", CLAP punch on 20 of 20) and the plain punch (0.54 to 0.23) go the same way. The second is an action that describes a different event from the one the developer named. The exploding barrel becomes "Barrel break, a wooden barrel ruptures violently" and scores 0.07 where both builders score 1.00, with AST hearing an explosion on 1 of 20 seeds; `glass_shatter` becomes "a piece of glass striking a surface" and drops from 0.82 to 0.06, with CLAP hearing swords and an empty gun. The 9 mm pistol keeps its gun for CLAP (19 of 20), but "sharp crack, metallic bang" puts 55% of its energy under 500 Hz against 9% in real pistols, and PANNs hears a gunshot on 12 of 20 seeds against 19 for the phrasebook (0.63 against 1.00). The bench sword clash (two longswords, "sharp metallic ring") falls from 1.00 to 0.41 and CLAP hears music on 3 seeds.

Three caveats cut in the phrasebook's favour, and one against it. Its demo prompts were picked on these held-out judges on small-sfx, so the demo rows carry a selection edge the other two lack. The door creak win (1.00 against 0.65 and 0.82) is the door judge's blind spot from `docs/generic.md`: the phrasebook asks for "a hollow knock on wood" and CLAP's top label is "a knock" on 13 of 20 seeds. The water judge rewards the phrasebook for answering both drips with its "Big water splash" sentence, and those two rows hold -0.48 and -0.45 of the writer's water deltas, against -1.00 for the splash it really misses. Against it: eight bench scenes share their plain key with a generic-run request (the five plain steps, the punch, the sword clash and the drip), so their phrasebook and caption clips are the same audio in both sets and count twice in the pooled rows, where the writer had two different inputs.

| scenario | set | group | model | phrasebook | caption | writer | writer minus phrasebook | writer minus caption |
|---|---|---|---|---|---|---|---|---|
| pistol shot | bench | demo | SA3 | 1.00 | 0.93 | 1.00 | +0.00 [-0.11, +0.00] | +0.07 [-0.09, +0.23] |
| rifle shot | bench | demo | SA3 | 0.99 | 1.00 | 1.00 | +0.01 [+0.00, +0.11] | +0.00 [+0.00, +0.00] |
| step stone | bench | demo | Woosh | 0.41 | 0.48 | 0.20 | -0.20 [-0.48, +0.16] | -0.27 [-0.61, +0.07] |
| step gravel | bench | demo | Woosh | 1.00 | 1.00 | 0.95 | -0.05 [-0.32, +0.20] | -0.05 [-0.32, +0.20] |
| step wood | bench | demo | Woosh | 0.88 | 0.27 | 0.36 | -0.52 [-0.90, -0.07] | +0.09 [-0.24, +0.36] |
| step metal | bench | demo | Woosh | 0.00 | 0.00 | 0.09 | +0.09 [+0.00, +0.27] | +0.09 [+0.00, +0.27] |
| step grass | bench | demo | Woosh | 0.63 | 0.88 | 0.27 | -0.36 [-0.63, +0.14] | -0.61 [-0.86, -0.20] |
| step water | bench | demo | Woosh | 0.54 | 0.54 | 1.00 | +0.46 [+0.04, +0.73] | +0.46 [+0.11, +0.72] |
| iron boots step | bench | demo | Woosh | 0.09 | 0.09 | 0.00 | -0.09 [-0.27, +0.00] | -0.09 [-0.27, +0.00] |
| bottle shot pistol* | bench | demo | Woosh | 0.45 | 0.06 | 0.32 | -0.13 [-0.45, +0.13] | +0.26 [+0.00, +0.52] |
| window rifle | bench | demo | Woosh | 0.91 | 0.32 | 0.77 | -0.15 [-0.34, +0.16] | +0.44 [+0.20, +0.69] |
| punch | bench | demo | Woosh | 0.54 | 0.47 | 0.47 | -0.08 [-0.54, +0.41] | +0.00 [-0.39, +0.39] |
| car horn | bench | unseen | Woosh | 0.23 | 0.47 | 0.74 | +0.51 [+0.23, +0.75] | +0.28 [-0.16, +0.64] |
| door creak | bench | unseen | Woosh | 1.00 | 0.88 | 0.82 | -0.18 [-0.41, +0.06] | -0.06 [-0.35, +0.24] |
| dog bark | bench | unseen | Woosh | 0.35 | 0.85 | 0.70 | +0.35 [+0.14, +0.62] | -0.15 [-0.35, +0.05] |
| sword clash | bench | unseen | Woosh | 1.00 | 0.99 | 0.41 | -0.59 [-0.81, -0.35] | -0.58 [-0.81, -0.29] |
| keyboard typing* | bench | unseen | Woosh | 0.28 | 0.49 | 1.00 | +0.72 [+0.49, +0.93] | +0.51 [+0.28, +0.79] |
| explosion* | bench | unseen | SA3 | 1.00 | 1.00 | 0.07 | -0.93 [-1.00, -0.80] | -0.93 [-1.00, -0.80] |
| snow step | bench | unseen | Woosh | 0.72 | 0.68 | 0.68 | -0.04 [-0.55, +0.34] | +0.00 [-0.41, +0.29] |
| glass shatter* | bench | unseen | Woosh | 0.32 | 0.00 | 0.88 | +0.55 [+0.26, +0.80] | +0.88 [+0.66, +1.00] |
| car door slam* | bench | unseen | Woosh | 0.85 | 0.65 | 0.90 | +0.05 [-0.16, +0.21] | +0.25 [+0.05, +0.40] |
| shoot, pistol | generic | demo | SA3 | 1.00 | 0.93 | 1.00 | +0.00 [-0.11, +0.00] | +0.07 [-0.09, +0.23] |
| shoot, rifle | generic | demo | SA3 | 0.99 | 1.00 | 1.00 | +0.01 [+0.00, +0.11] | +0.00 [+0.00, +0.00] |
| break, window | generic | demo | Woosh | 0.88 | 0.48 | 0.64 | -0.23 [-0.45, +0.06] | +0.16 [-0.06, +0.45] |
| bullet impact, glass* | generic | demo | Woosh | 1.00 | 0.39 | 0.98 | -0.02 [-0.23, +0.00] | +0.60 [+0.32, +0.79] |
| footstep, gravel | generic | demo | Woosh | 1.00 | 1.00 | 0.82 | -0.18 [-0.46, +0.12] | -0.18 [-0.46, +0.14] |
| footstep, stone | generic | demo | Woosh | 0.41 | 0.48 | 0.34 | -0.07 [-0.41, +0.34] | -0.14 [-0.54, +0.27] |
| footstep, wood | generic | demo | Woosh | 0.88 | 0.27 | 0.09 | -0.79 [-1.00, -0.34] | -0.18 [-0.54, +0.18] |
| footstep, metal | generic | demo | Woosh | 0.00 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| footstep, water | generic | demo | Woosh | 0.54 | 0.54 | 0.90 | +0.36 [-0.16, +0.70] | +0.36 [-0.10, +0.66] |
| splash, water | generic | demo | Woosh | 0.77 | 0.17 | 0.00 | -0.77 [-1.00, -0.43] | -0.17 [-0.43, +0.00] |
| punch, body | generic | demo | Woosh | 0.54 | 0.47 | 0.23 | -0.31 [-0.70, +0.08] | -0.23 [-0.62, +0.16] |
| dog_bark, dog | generic | unseen | Woosh | 0.85 | 0.95 | 0.95 | +0.10 [-0.05, +0.30] | +0.00 [-0.15, +0.15] |
| door_creak, wood | generic | unseen | Woosh | 1.00 | 0.88 | 0.65 | -0.35 [-0.59, -0.12] | -0.24 [-0.53, +0.12] |
| keyboard_click, plastic* | generic | unseen | SA3 | 0.14 | 0.97 | 1.00 | +0.86 [+0.58, +1.00] | +0.03 [-0.16, +0.28] |
| footsteps_Concrete, concrete | generic | unseen | Woosh | 0.54 | 0.72 | 0.09 | -0.45 [-0.75, +0.00] | -0.63 [-0.91, -0.27] |
| PlayerFootstep, snow | generic | unseen | Woosh | 0.75 | 0.63 | 0.68 | -0.07 [-0.46, +0.34] | +0.05 [-0.38, +0.46] |
| shotgun_fire, shotgun | generic | unseen | SA3 | 0.86 | 1.00 | 1.00 | +0.14 [+0.00, +0.30] | +0.00 [+0.00, +0.00] |
| sword_clash, steel | generic | unseen | Woosh | 1.00 | 0.99 | 0.87 | -0.13 [-0.34, +0.07] | -0.12 [-0.29, +0.07] |
| car_door_slam, car* | generic | unseen | Woosh | 0.35 | 0.25 | 0.65 | +0.30 [-0.05, +0.54] | +0.40 [+0.10, +0.65] |
| explosion_large, debris* | generic | unseen | SA3 | 1.00 | 1.00 | 1.00 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| glass_shatter, glass | generic | unseen | Woosh | 0.82 | 0.06 | 0.06 | -0.76 [-0.90, -0.45] | +0.00 [-0.19, +0.19] |
| car_horn, car | generic | unseen | Woosh | 0.00 | 0.91 | 1.00 | +1.00 [+0.86, +1.00] | +0.09 [+0.00, +0.26] |
| Pistol_Shot, 9mm | generic | unseen | SA3 | 1.00 | 1.00 | 0.63 | -0.37 [-0.58, -0.11] | -0.37 [-0.58, -0.11] |
| melee_hit, body* | generic | unseen | Woosh | 0.31 | 0.86 | 0.54 | +0.23 [-0.16, +0.61] | -0.31 [-0.61, +0.08] |
| water judge: step water | bench | demo | Woosh | 1.00 | 0.83 | 0.95 | -0.05 [-0.31, +0.00] | +0.12 [-0.13, +0.38] |
| water judge: rain tin roof* | bench | unseen | Woosh | 0.09 | 0.00 | 0.00 | -0.09 [-0.26, +0.00] | +0.00 [+0.00, +0.00] |
| water judge: drip puddle | bench | unseen | Woosh | 1.00 | 0.43 | 0.52 | -0.48 [-0.83, -0.21] | +0.09 [-0.25, +0.48] |
| water judge: footstep, water | generic | demo | Woosh | 1.00 | 0.83 | 0.89 | -0.11 [-0.38, +0.05] | +0.07 [-0.16, +0.32] |
| water judge: splash, water | generic | demo | Woosh | 1.00 | 0.61 | 0.00 | -1.00 [-1.00, -1.00] | -0.61 [-0.96, -0.26] |
| water judge: water_drip, water | generic | unseen | Woosh | 1.00 | 0.43 | 0.55 | -0.45 [-0.72, -0.17] | +0.12 [-0.23, +0.48] |
| water judge: rain_on_roof, metal* | generic | unseen | Woosh | 0.00 | 0.21 | 0.17 | +0.17 [+0.00, +0.32] | -0.04 [-0.32, +0.17] |

A star marks a stand-in reference class (house door slams for a car door, fireworks for an explosion, glass breaks for a wine glass or a shot bottle, clicks for typing, drops for rain), as in `generic_eval.py`. Twenty-six scenarios have no validated class (the reloads, the four balls, the crate and the barrel drops, the campfires, the lightsabers, the grapples, the coins, the growls, the bows, the lasers, the magic missile and the ice cracks) and rest on the page. The static rule flagged 7 of the 4,500 clips, six of them the writer's: three gravel step takes on "crunchy scrape", one snow step, one coin and one glass shatter.

Fresh variants 60 to 79, rendered for every arm in the follow-up below, repeat the verdict. There the writer moves the unseen rows by +0.04 [-0.15, +0.24] (9 up, 10 down) and the demo by -0.10 [-0.25, +0.06] (6 up, 15 down), the water rows by -0.19 [-0.50, +0.04], and the guns sit at 0.95 against 0.96. Pooled over both blocks, 40 seeds per row (`llm_eval.py combine`):

| rows, variants 40-79 | n | phrasebook | caption | writer | writer minus phrasebook, up/down | writer minus caption |
|---|---|---|---|---|---|---|
| every judged scenario | 45 | 0.67 | 0.63 | 0.63 | -0.04 [-0.16, +0.09] 15/26 | +0.00 [-0.09, +0.11] |
| unseen | 22 | 0.68 | 0.76 | 0.72 | +0.04 [-0.16, +0.24] 10/10 | -0.04 [-0.19, +0.13] |
| demo | 23 | 0.67 | 0.51 | 0.55 | -0.11 [-0.24, +0.03] 5/16 | +0.04 [-0.07, +0.16] |
| guns | 6 | 0.97 | 0.98 | 0.95 | -0.02 [-0.14, +0.08] 3/2 | -0.03 [-0.13, +0.06] |
| footsteps | 15 | 0.55 | 0.49 | 0.45 | -0.10 [-0.27, +0.10] 3/11 | -0.04 [-0.18, +0.11] |
| rows SA3 renders | 9 | 0.89 | 0.98 | 0.87 | -0.02 [-0.29, +0.25] 4/3 | -0.12 [-0.33, +0.03] |
| rows Woosh renders | 36 | 0.62 | 0.54 | 0.58 | -0.04 [-0.18, +0.10] 11/23 | +0.03 [-0.08, +0.15] |
| water judge | 7 | 0.73 | 0.48 | 0.51 | -0.23 [-0.52, +0.01] 1/6 | +0.02 [-0.23, +0.22] |
| all plus water | 52 | 0.68 | 0.61 | 0.62 | -0.06 [-0.17, +0.05] 16/32 | +0.01 [-0.08, +0.10] |

Forty seeds do not move the answer. The writer and the caption template land in the same place overall (+0.00), each ahead of the phrasebook on unseen sounds by an amount the intervals cover and behind it on the demo, and they win on different rows: the writer on the wine glass (0.93 against 0.00), the car door (0.75 against 0.38) and the water step (1.00 against 0.68), caption on the melee hit (0.89 against 0.62), the dog bark from the bench (0.93 against 0.70) and the SA3 rows (0.98 against 0.87).

### Two compositions of the same fields

Both failure patterns suggested a change that needs no new model call, so `llm_eval.py variants` recomposes the stored fields two ways: `llm_nosound` drops the sound field, and `llm_named` puts the request's material and event words in front whenever the writer's source line lacks one of them ("Explode, Barrel break, ..." on SA3, "glass shatter, Glass breaking, ..." on Woosh). The rule, written before any variant clip existed (`preregistered_followup.txt`), picks on variants 40 to 59 the best of the three by the mean over all 52 rows, counts a variant only with a lead of at least 0.02 over the writer, and confirms the pick on variants 60 to 79, rendered fresh for all five arms with no choice made from them.

Dropping the sound field won the selection by 0.025 and lost the confirmation: -0.01 [-0.10, +0.08] against the writer over 52 rows, -0.03 [-0.15, +0.09] on the unseen rows with water, and -0.15 [-0.42, +0.02] on guns, where four of the six gun rows fell and none rose (the 9 mm pistol to 0.05). The pooled number hides a split that repeats in both blocks. Without its sound field the splash recovers (0.00 and 0.09 to 0.77 and 0.86 on the splash judges, 0.00 and 0.26 to 1.00 on the water judge), and so do the concrete step (0.09 and 0.18 to 0.54 and 0.81) and the bench stone step (0.20 and 0.27 to 0.54 and 0.61). The gravel step (0.95 and 0.75 to 0.34 and 0.18), the plain horn (1.00 to 0.57 and 0.51) and every gun need it. So "sharp wet slap" and "solid thud, dull impact" do turn into punches, while "crunchy scrape" and "continuous honk" carry the identity, and no fixed composition keeps the second kind and drops the first. The wood steps stay far under the phrasebook's 0.85 either way (0.09 to 0.31 and 0.40 to 0.27 over 40 seeds), so most of their loss sits in "a foot stepping onto a wooden surface" itself.

Putting the request words in front repeated a small gain in both blocks: +0.02 [-0.02, +0.05] over all 52 rows across the 40 seeds, +0.03 [-0.02, +0.09] on the unseen rows (8 up, 2 down), +0.04 on guns. It lifts the 9 mm pistol (0.74 to 0.92 over 40 seeds) and `glass_shatter` (0.13 to 0.41), and the exploding barrel barely moves (0.13 to 0.17), since "Explode" in front of "Barrel break, a wooden barrel ruptures violently" still describes a breaking barrel. It stays under the bar the rule set, and it is the one direction worth a fresh preregistered test, as a validator check that the source line keeps a word of the event name.

### Blind page

`data/_gate/llm/page/index.html` holds all 75 scenarios: the bench scenes in two sections with the facts the writer read under each title, then the generic run's demo and unseen requests. Every row has three options (no two sources wrote the same text on any row), four takes each (variants 40 to 43), 900 clips at the game's playback gain, letters shuffled per row (the phrasebook sits behind A on 27 rows, the writer and caption on 24 each), best, worst and "none sound right" kept in the browser, Reveal with the source, the model, the judge score and the prompt, and Download. The judges measure category identity in a 0.6 s window and nothing about whether a footstep sounds good, so the page is where the writer can still win: the balls, the crate, the barrel, the growls and the laser are page-only.

Run order (paths are flags, the defaults write under `data/_gate/llm`): `write` in `.venv214` with `CUDA_VISIBLE_DEVICES=-1` (CPU writer, 75 prompts, `scenarios.json`), `render-woosh` in `.venv_woosh`, `render-sa3` in `.venv214`, `routes`, then `judge-vocab` and `judge-water` in `.venv`, then `report` and `page`. `variants` adds the two diagnostic compositions below from the stored fields, and every command takes `--arms`, `--first-seed` and `--tag`. Here the work folders were `scratchpad\llm_audio\work` (variants 40 to 59) and `work60` (60 to 79), each command launched through `runpinned.py` with 4 threads; the reports, the scenario file with every prompt, the writer store, both preregistered rules, the block A selection and the 40-seed pool are copied to `data/_gate/llm/eval/`.

## Unity package

`SargeScene` builds the record and caches the static part per component. `SargeEmitter`, `SargeFootsteps` and `SargeAutoCapture` attach it when `sendScene` is on, and the predictors attach the scene their paired emitter will send, so a predicted key matches the fired one: `SargeImpactPredictor` uses its `SargeAutoCapture`, `SargeMotionPredictor` sends a step scene only when its root carries a `SargeFootsteps` (the demo's `SargeFirstPerson` fires steps itself, so its predictions stay plain), and `SargeAimPredictor` sends one only through `SceneResolver`. The server also gives a prediction or habit guess without a scene the last scene fired for the same event, material and context, as `--world` does for physics.

Replies carry `scene_key` and `prompt_rev` when they apply. With `localPlayback` the client maps each raw scene string to the server's `scene_key`, keys its held takes on it, sends `have_scene` and `have_rev`, and on a `prompt_rev` that differs from the one it holds destroys the held takes of that key before storing the new one, so a template take never outlives its upgrade. A lower revision also counts: it comes from a restarted sidecar or from an older reply on another lane overtaken by a newer one, and either way the next play brings the key back in line. The server reads the revision under the same lock that picks the clip, so a reply never pairs an old take with a new revision. The package, the demo, the editor scripts, the probe editor and the sample compile with Unity 2022.3.8f1's Roslyn, and a Mono run of the new `SargeClient` against the Python server with a fake writer showed the scene key coming back, `meta_only` honouring `have_scene` and `have_rev`, the revision reaching 1 after the swap, a manifest ack and a prediction carrying a scene. No demo file changed.

Old clients still work: every new field is optional both ways, old servers ignore the new request fields and answer a MANIFEST with a non-fatal `unknown_type` error that the client swallows, and with the new flags off a new server answers an old request byte for byte as before. An old client that already holds take 0 of a key keeps playing that take after an upgrade, since its clip cache knows nothing of revisions; a new client is needed to hear upgrades of keys it has already played.

## Tests

`tests/test_writer.py` (30 tests, CPU): name cleaning and the fixed-order scene key; physics objects and kinds from `phys`; the key string and seed with and without a scene; the cache path moving only with a prompt; the writer input reading only the key and dropping size and build; the writer id; acceptance, composition, repair of banned, controller, leaked and copied words, the per-field loop check and rejections; the store keeping the first text across reloads; queue order; through a live server the template first and the swapped prompt after with `prompt_rev` and `have_rev`, a predicted key taking its prompt before the first hit with its seed unchanged, the SA3 form on guns and the Woosh form elsewhere, game prompts winning both ways, a store pre-filled at startup with a read-only writer never calling the model, next-miss mode and a swap that waits for renders in flight, the scene on the wire and predictions taking the last scene; byte-identical replies with and without the new fields; the manifest with and without a writer; the `run_sidecar.py` flags; the CPU-only llama-server command line and the deterministic request bodies of both backends; model resolution from Ollama manifests; the yield gate wrapping renders and nesting; and the job object pinning a child. The whole suite passes.

## Still open

- **The ear.** The judges have spoken (section "On sound"); the blind page `data/_gate/llm/page/index.html` is the user's call, and it is the only verdict on the 26 scenarios with no validated judge.
- **Plausible inventions cost sound.** The exploding barrel still comes out as "Barrel break, a wooden barrel ruptures violently" and scored 0.07 against 1.00 for both builders; `glass_shatter` became "a piece of glass striking a surface" (0.82 to 0.06). The key has the event name, and the writer is free to drop it. Keeping the event words in front gained +0.02 over 40 seeds; a validator rule that rejects a source line without a word of the event name is the next thing to test, preregistered on fresh seeds and the 23 fresh requests of the template run.
- **The sound field is right on some rows and wrong on others.** Without it the splash and the concrete step recover and the gravel step, the horn and every gun fall, in both seed blocks. The rule that would tell them apart is still unknown; a per-row choice made from these judges would only fit them.
- **The second model.** Phi-4-mini (MIT) never went through the audio test.
- **More first hits.** Scene fields multiply keys and each new key pays one template first hit.
- **Swap renders share the GPU.** A swap starts only while no realtime render is in flight, but a miss that arrives during a swap waits for it (about 50 to 120 ms on Woosh, 150 ms on a gun), as it would behind a prediction render.
- **Unbuilt pieces.** The Jev-style yes or no gate (`--writer-gate`), the cloud plugin (a backend is any object with `ident`, `start`, `complete` and `close`, so it would slot into `sarge.writer.build`) and distillation into Granite 350M are designed only.
