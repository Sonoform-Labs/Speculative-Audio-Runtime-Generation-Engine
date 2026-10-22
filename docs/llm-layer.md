# Prompt writer (the LLM layer)

The phrasebook breaks on any sound it was never tuned for (car horn 0.00, rain on a roof rendered as a steel clang, a drip answered with a splash), and the bare word list collapses guns on SA3 medium (pistol 0.97 to 0.16, `docs/generic.md`). The prompt writer replaces both with one generic step: a small CPU language model reads what the engine knows about a moment (the event, the object, what touched what, who made it happen, the room) and writes a prompt for that exact scenario. SARGE then puts the three short fields it returns into the sentence form each audio model is known to follow. Nothing in it is written per game or per sound, so a sound the demo never had gets the same treatment as a footstep.

Everything here is off by default. With every new flag off, the end-to-end bench under Unity's environment (`e2e_hybrid.py --extra --learn`) served 75 of 75 clips bit-identical to `data/_gate/startup/baseline_warm_1` with a max abs difference of 0.0 and equal shapes, in two runs (`llm_off` mid-change and `llm_off_final` on the final tree, `data/_gate/startup/compare_llm_off.json`), and the server replies are byte-identical with and without the new wire fields (`tests/test_writer.py`).

On sound the writer does not earn its flag yet. Judged on 45 scenarios over variants 40 to 79 (section "On sound"), it ties the phrasebook on sounds nobody tuned for (+0.04 [-0.16, +0.24]) and loses to it on the demo (-0.11 [-0.24, +0.03]), about where the caption template already sits. It wins big on the car horn, the keyboard, the wine glass and the car door, and it loses where its sound words name an impact (wood and concrete steps, the splash) or where it renames the event (the exploding barrel becomes "Barrel break").

A second round with four general writer styles (section "Writer styles") moved the needle a little and missed the goal. The preregistered rule picked `named`, which keeps the request's event and material words in the title. On 30 fresh requests and the known scenarios at new seeds it led the phrasebook on unseen rows by +0.06 [-0.06, +0.16] and lost 0.15 on the demo and 0.14 on water, so it fails three of the four goal bars and every default stays where it was.

A third round grounded the prompts in 273,644 captions of real recordings (section "Caption grounding") and stopped at selection. All three grounded arms failed the water guard of the preregistered rule, so no confirmation clip was rendered and the 30 new fresh scenarios stay unused. The nearest caption with the request words in front came closest: +0.13 [-0.01, +0.28] over the phrasebook on unseen rows and level with it on guns, but -0.22 on water, where the bar was -0.15.

A fourth round asked where the cap sits (section "Untried levers"). Prompts written by a strong model from the scene facts tie the cheap caption sources on unseen sounds and still miss the demo bar, so a better writer has nothing left to win under these judges. The cap was the audio model: the caption template rendered on Woosh Flow in place of DFlow met all four goal bars on new clips, +0.24 [+0.14, +0.35] on 52 unseen rows (38 up, 8 down), -0.01 on the demo, +0.12 on guns and -0.03 on water, the first generic setup to do so. Flow is too slow for a first hit, so it ships as an upgrade of the held takes behind `--woosh-upgrade flow` (docs/hybrid.md), off by default like every other flag here.

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
| `--writer-style v1\|named\|texture\|title\|caption` | `v1` | how the writer is instructed, validated and composed (section "Writer styles"); every style writes to a separate writer id and store, and `v1` keeps `07d28eff9ea0` |
| `--writer ground` | | writes each prompt from real recording captions with no language model (section "Caption grounding"); needs `--writer-ground words` or `caption` |
| `--writer-ground off\|anchors\|words\|caption` | `off` | `anchors` adds the nearest real captions to the writer's input (`--writer llama` or `ollama`); `words` and `caption` compose the prompt from them (`--writer ground`); every mode writes to a separate writer id and store |
| `--writer-ground-index` | `G:\sarge\captions\index` | the caption index folder: `index.json`, `captions.json`, `emb.npy` and `embedder/` |
| `--writer-ground-k` | 4, 8, 1 | captions per key for anchors, words and caption |
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

## Writer styles (2026-10-21)

The judged losses had two causes that no single scene explains: the writer renames the event, and its sound words reach for collisions that Woosh renders as a punch. Four styles behind `--writer-style` go after both with rules that run the same way on every scene and name no event, material or demo object. `named` keeps the v1 examples and adds a naming rule: the title must hold every content word of the request's event and material lines, leaving out stop words, the scene's action words, size and loudness words and controller words, and a title that misses one gets the request words put in front by the validator. That is the post-hoc `llm_named` composition (+0.02 over 40 seeds) moved into the writer, and the model now does most of it alone: the exploding barrel comes back as "Barrel explode, a barrel ruptures violently" and `glass_shatter` as "Glass shatter, a piece of glass breaking against a surface". `named` also stops deleting hollow and static, since the input no longer carries build words.

`texture` adds a sound line on the character of the sound itself (how it starts, its pitch and texture, how it ends), a rule against generic collision words, and four new examples: a nylon zipper as a plain key, a cat purring, a kettle whistle as a world loop and a quill scratch as a player action. None is an impact, and none shares an event, material, object or creature word with any eval scenario. Instructions alone barely moved the 2B model, which still put thud or impact in 35 of 105 raw sound fields against 37 for v1. So the three texture styles also drop a sound item that holds thud, thump, bang, slap, smack, whack, thwack, bump, impact or collision when the request lacks the word, and they allow the sound field to end empty. The wood step becomes "Wood footstep, a foot stepping onto a wooden surface", the splash "Water splash, water hitting a surface, spreading ripple", and gravel keeps "crunchy granular scrape" because only the "sharp impact" item goes. The 9 mm pistol lost "metallic bang" to the instruction itself and now ends "sharp metallic crack, sharp report".

`title` asks the model exactly what `texture` asks (105 of 105 raw answers matched across the two sessions) and gives Woosh only the title, the sound words and the tails, so it tests whether the action line is what sinks the wood step. `caption` asks for the action as one plain library caption and gives Woosh "{action}, {sound}" and SA3 "TrackType: SFX, {Source}. {Action}, {sound}, ..., a single sound recorded up close". The model left an event word out of 69 of 105 caption actions, so in practice the validator puts the request words at the head of most of them.

| style, 105 scenarios, 4 CPU threads | writer id | p50 / p95 per prompt | rejected | repaired | what the repairs were |
|---|---|---|---|---|---|
| v1 | `07d28eff9ea0` | 1.56 / 1.92 s | 0 | 8 | loud, small |
| named | `39800c4737a4` | 1.56 / 1.92 s | 0 | 47 | 38 titles given the request words, 11 banned words |
| texture | `5baf1e7d1565` | 1.63 / 1.94 s | 0 | 64 | 49 collision items, 29 titles, 7 player, 6 loud |
| title | `89c76b1fe56a` | 1.61 / 1.93 s | 0 | 64 | the same answers as texture |
| caption | `018e49d20d4c` | 1.63 / 1.92 s | 0 | 86 | 69 actions given the request words, 41 collision items |

The 105 scenarios are the 75 known ones and 30 fresh requests in `llm_eval.FRESH`, each mapped to a validated held-out judge type (5 guns, 2 explosions, 4 footsteps, 3 glass breaks, 4 doors, 2 barks, 2 punches, 2 clicks, 2 horns, 2 blade clashes, 2 splashes that also get the water judge), half of them carrying a scene record, and none rendered by any earlier run. `llm_eval.py write --set all --styles v1 named texture title caption` writes them, one server session and one store per style. The decision rule went into `preregistered_rule.txt` (copied with the fresh list, every prompt and the five stores to `data/_gate/llm/eval/wstyles/`) before any style prompt existed, with a dated amendment for the collision rule before any clip: selection on variants 80 to 99 of the 75 known scenarios by the mean of the unseen and demo deltas against the phrasebook, with guards on guns, water and demo and a 0.02 margin over v1, then confirmation of the single pick on the 30 fresh requests at variants 0 to 39 and the known ones at 100 to 119. With every flag off the end-to-end bench served 75 of 75 clips bit-identical to `baseline_warm_1` (max abs 0.0, `wstyles/compare_wstyle_off.json`); the only edits after that run were instruction and help strings that the server never reads with the writer off.

### Selection, variants 80 to 99

All seven prompt sources were rendered new on the 75 known scenarios at variants 80 to 99 and judged as in the first round. `llm_eval.py select` applies the rule to that report, and it wrote `selection.txt` before any confirmation clip existed. Two styles failed the water bar: `title` at -0.29 (and -0.15 on the demo) and `caption` at -0.31. Of the two left, `named` had Q -0.001 against -0.007 for `texture`, a gap of 0.006 that falls just outside the 0.005 tie band, and it cleared v1 by 0.033.

| arm, variants 80-99, minus the phrasebook | writer id | unseen U (22) | demo D (23) | guns G (6) | water W (7) | all A (52) | Q = (U + D) / 2 | eligible |
|---|---|---|---|---|---|---|---|---|
| v1 | `07d28eff9ea0` | -0.01 | -0.06 | -0.04 | -0.30 | -0.07 | -0.034 | reference |
| named | `39800c4737a4` | +0.07 | -0.08 | -0.03 | -0.13 | -0.02 | -0.001 | yes, the pick |
| texture | `5baf1e7d1565` | +0.05 | -0.07 | -0.01 | -0.14 | -0.03 | -0.007 | yes |
| title | `89c76b1fe56a` | +0.03 | -0.15 | -0.01 | -0.29 | -0.09 | -0.058 | no (water, demo) |
| caption | `018e49d20d4c` | -0.00 | -0.08 | -0.09 | -0.31 | -0.08 | -0.045 | no (water) |

The collision rule did what it was built for and paid for it elsewhere. Against `named`, `texture` lifted the rows where an impact word had turned into a punch: the plain punch from 0.16 to 0.39, the bench wood step from 0.18 to 0.45, the splash from 0.27 to 0.43, the concrete steps from 0.18 to 0.48, and the 9 mm pistol from 0.58 to 0.95 once "metallic bang" was gone. It gave that back where its sound line traded a signature word for acoustic adjectives. The plain car horn fell from 0.97 to 0.07 when "continuous honk" became "deep resonant blast, metallic vibration", the bench horn from 0.91 to 0.46 and the plain door creak from 0.82 to 0.41. Dropping the action line (`title`) sank the gravel steps (0.95 to 0.07 on the plain key) and the water rows.

### Confirmation, 30 fresh requests and variants 100 to 119

`named` went on alone. The 30 fresh requests were rendered at variants 0 to 39 and the 75 known scenarios at 100 to 119, with the phrasebook, the caption template and v1 on the same seeds, and `llm_eval.py confirm` pooled the two parts row by row (`wstyles/confirm.txt`). The pick fails three of the four bars.

| rows | n | phrasebook | caption | named | v1 | named minus phrasebook, up/down | named minus caption | named minus v1 | caption minus phrasebook |
|---|---|---|---|---|---|---|---|---|---|
| C1 unseen identity rows, bar: lower bound above 0 (FAIL) | 52 | 0.62 | 0.70 | 0.68 | 0.64 | +0.06 [-0.06, +0.16] 24/24 | -0.02 [-0.11, +0.07] | +0.04 [-0.04, +0.11] | +0.08 [-0.04, +0.19] |
| fresh requests | 30 | 0.60 | 0.65 | 0.64 | 0.60 | +0.05 [-0.09, +0.17] 16/12 | -0.00 [-0.13, +0.12] | +0.05 [-0.05, +0.13] | +0.05 [-0.11, +0.21] |
| known unseen | 22 | 0.65 | 0.77 | 0.73 | 0.70 | +0.08 [-0.11, +0.28] 8/12 | -0.03 [-0.18, +0.11] | +0.03 [-0.10, +0.16] | +0.11 [-0.06, +0.28] |
| C2 demo, bar -0.05 (FAIL) | 23 | 0.70 | 0.54 | 0.55 | 0.56 | -0.15 [-0.28, -0.01] 7/13 | +0.01 [-0.13, +0.17] | -0.02 [-0.10, +0.07] | -0.17 [-0.31, -0.03] |
| C3 guns, bar -0.10 (PASS) | 11 | 0.83 | 0.90 | 0.94 | 0.93 | +0.11 [-0.04, +0.23] 7/1 | +0.03 [-0.14, +0.23] | +0.01 [-0.08, +0.06] | +0.07 [-0.08, +0.23] |
| C4 water judge, bar -0.10 (FAIL) | 9 | 0.82 | 0.53 | 0.68 | 0.50 | -0.14 [-0.36, -0.02] 1/7 | +0.15 [-0.05, +0.35] | +0.18 [-0.00, +0.38] | -0.29 [-0.55, -0.11] |
| every judged scenario | 75 | 0.64 | 0.65 | 0.64 | 0.62 | -0.01 [-0.10, +0.09] 31/37 | -0.01 [-0.08, +0.08] | +0.02 [-0.04, +0.08] | +0.00 [-0.09, +0.10] |
| every judged scenario plus water | 84 | 0.66 | 0.63 | 0.64 | 0.60 | -0.02 [-0.11, +0.06] 32/44 | +0.01 [-0.07, +0.09] | +0.04 [-0.02, +0.09] | -0.03 [-0.12, +0.06] |
| footsteps | 19 | 0.58 | 0.53 | 0.45 | 0.42 | -0.13 [-0.28, +0.02] 7/11 | -0.08 [-0.21, +0.06] | +0.03 [-0.05, +0.12] | -0.05 [-0.19, +0.07] |
| rows SA3 renders | 16 | 0.84 | 0.93 | 0.94 | 0.83 | +0.09 [-0.04, +0.20] 8/3 | +0.01 [-0.12, +0.14] | +0.11 [-0.03, +0.25] | +0.09 [-0.05, +0.22] |
| rows Woosh renders | 59 | 0.59 | 0.57 | 0.56 | 0.56 | -0.03 [-0.14, +0.08] 23/34 | -0.01 [-0.10, +0.09] | +0.00 [-0.06, +0.06] | -0.02 [-0.14, +0.09] |

On the unseen rows the pick leads the phrasebook by 0.06 with 24 rows up and 24 down, and the interval reaches -0.06. The caption template does a little better there (+0.08), so on sounds nobody tuned for the writer still adds nothing a one-line template lacks. The demo loss (-0.15, 7 up and 13 down) is twice what the selection block showed (-0.08), a swing between seed blocks of the size the first round already saw. AST and CLAP side with the phrasebook on the demo seeds where the two differ (+8/-53 and +33/-70, both p < 0.001), and PANNs splits (+63/-79). Guns pass with room, mostly because the phrasebook breaks on fresh gun names: a Glock "handgun_fire" scores 0.37 on "Glock handgun_fire, single handgun_fire on a glock surface", the bandit's revolver 0.58. Water fails at -0.14 [-0.36, -0.02], seven rows down and one up, even though the pick holds +0.18 over v1 there. Across every pooled group but the demo the naming rule is worth about +0.04 over v1, the size the post-hoc `llm_named` composition showed over 40 seeds, and no interval clears zero.

The fresh requests show what each builder does with a request nobody wrote for. The phrasebook falls back to its generic frame, and that frame breaks when the request carries no material: "woof" from a shepherd dog becomes " woof, single woof on a  surface, a short knock on , foley, ..." and scores 0.03, where the pick scores 0.93 on "Dog woof, a dog shepherd barks, sharp bark, vocal growl, outdoors". The same frame on a delivery van's "honk" scores 0.00 and the pick 0.40, under the caption template and v1 at 1.00 ("brief air blast" carries less horn than v1's "brief honk"). The pick wins the locker slam (0.43 to 0.90), the katana parry (0.15 to 0.61), the bedroom door (0.25 to 0.65) and the revolver (0.58 to 1.00), and the caption template takes three of those four as well.

| fresh request | judge | model | phrasebook | caption | named | v1 | named minus phrasebook | named minus caption | seeds gained/lost, AST PANNs CLAP |
|---|---|---|---|---|---|---|---|---|---|
| barn door open | door_open | Woosh | 1.00 | 1.00 | 0.85 | 0.91 | -0.15 [-0.29, +0.00] | -0.15 [-0.29, +0.03] | n/a n/a +2/-10 |
| beer bottle break | glass_break | Woosh | 0.88 | 0.74 | 0.13 | 0.23 | -0.75 [-0.90, -0.58] | -0.61 [-0.77, -0.42] | +0/-29 +1/-17 +3/-17 |
| brawl punch | punch | Woosh | 0.72 | 0.66 | 0.47 | 0.39 | -0.26 [-0.53, +0.08] | -0.19 [-0.47, +0.12] | +12/-7 +6/-15 +5/-6 |
| bus horn | horn* | Woosh | 0.07 | 0.20 | 0.30 | 0.20 | +0.23 [+0.07, +0.40] | +0.10 [-0.10, +0.30] | +8/-1 +8/-12 +7/-5 |
| carbine shot | gunshot_rifle | SA3 | 1.00 | 1.00 | 1.00 | 0.86 | +0.00 [-0.03, +0.03] | +0.00 [-0.03, +0.00] | +1/-1 +1/-1 +0/-3 |
| dagger clash | sword* | Woosh | 0.12 | 0.75 | 0.32 | 0.29 | +0.20 [+0.03, +0.38] | -0.44 [-0.61, -0.26] | +13/-4 +7/-6 +13/-2 |
| detonate tank | explosion* | SA3 | 1.00 | 1.00 | 0.83 | 0.13 | -0.17 [-0.37, -0.10] | -0.17 [-0.37, -0.10] | +3/-13 +0/-9 +0/-7 |
| dive splash | splash | Woosh | 0.69 | 0.30 | 0.30 | 0.04 | -0.39 [-0.73, -0.04] | +0.00 [-0.17, +0.17] | +0/-21 +0/-7 +5/-14 |
| door shut bedroom | door_slam | Woosh | 0.25 | 0.93 | 0.65 | 0.53 | +0.40 [+0.19, +0.61] | -0.28 [-0.45, -0.10] | +18/-1 +19/-1 +12/-5 |
| footstep mud | footstep | Woosh | 0.85 | 1.00 | 0.85 | 0.85 | +0.01 [-0.31, +0.31] | -0.15 [-0.37, +0.14] | n/a +13/-3 +6/-9 |
| guard dog bark | dog_bark | Woosh | 0.77 | 0.93 | 0.90 | 0.90 | +0.13 [+0.00, +0.30] | -0.03 [-0.18, +0.10] | +6/-0 +7/-1 +8/-3 |
| gunfire revolver | gunshot_pistol | SA3 | 0.58 | 0.11 | 1.00 | 0.90 | +0.42 [+0.22, +0.58] | +0.89 [+0.74, +0.97] | +4/-0 +17/-1 +6/-2 |
| handgun fire | gunshot_pistol | SA3 | 0.37 | 0.95 | 0.77 | 0.79 | +0.40 [+0.18, +0.55] | -0.18 [-0.37, -0.07] | +1/-0 +19/-3 +0/-12 |
| honk van | horn* | Woosh | 0.00 | 1.00 | 0.40 | 1.00 | +0.40 [+0.20, +0.54] | -0.60 [-0.77, -0.43] | +12/-0 +12/-4 +6/-11 |
| hunting rifle shot | gunshot_rifle | SA3 | 0.78 | 1.00 | 0.97 | 1.00 | +0.18 [-0.03, +0.39] | -0.03 [-0.11, +0.00] | +14/-3 +0/-3 +0/-3 |
| jar smash | glass_break | Woosh | 0.64 | 0.32 | 0.55 | 0.26 | -0.10 [-0.32, +0.13] | +0.23 [+0.00, +0.45] | +5/-8 +0/-8 +2/-7 |
| key press keypad | ui_click | Woosh | 1.00 | 0.94 | 0.94 | 0.94 | -0.06 [-0.24, +0.09] | +0.00 [-0.21, +0.20] | +2/-6 +7/-10 +3/-1 |
| locker slam | door_slam* | Woosh | 0.43 | 0.89 | 0.90 | 1.00 | +0.47 [+0.30, +0.64] | +0.01 [-0.06, +0.15] | +21/-2 +22/-2 +13/-0 |
| mirror break | glass_break | Woosh | 0.74 | 0.13 | 0.55 | 0.16 | -0.19 [-0.43, -0.03] | +0.42 [+0.19, +0.55] | +2/-13 +1/-11 +5/-12 |
| npc footstep sand | footstep | Woosh | 1.00 | 0.76 | 0.45 | 0.37 | -0.55 [-0.78, -0.35] | -0.31 [-0.62, -0.05] | n/a +1/-19 +2/-16 |
| parry katana | sword* | Woosh | 0.15 | 0.70 | 0.61 | 1.00 | +0.46 [+0.23, +0.67] | -0.09 [-0.29, +0.20] | +26/-4 +14/-11 +26/-2 |
| plop pond | splash | Woosh | 0.47 | 0.04 | 0.04 | 0.04 | -0.43 [-0.69, -0.17] | +0.00 [-0.13, +0.13] | +1/-27 +0/-23 +1/-11 |
| rocket explosion | explosion* | SA3 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 [-0.20, +0.00] | +0.00 [-0.17, +0.10] | +3/-7 +0/-0 +0/-0 |
| shotgun shot | gunshot_shotgun | SA3 | 0.90 | 1.00 | 0.93 | 0.99 | +0.03 [-0.10, +0.15] | -0.07 [-0.18, +0.00] | +1/-1 +4/-3 +1/-2 |
| step marble | footstep | Woosh | 0.10 | 0.03 | 0.17 | 0.07 | +0.07 [-0.10, +0.27] | +0.14 [+0.02, +0.31] | n/a +5/-3 +11/-3 |
| swing open cellar | door_open | Woosh | 0.94 | 0.03 | 0.85 | 0.76 | -0.09 [-0.26, +0.09] | +0.82 [+0.65, +0.97] | n/a n/a +4/-7 |
| switch flip | ui_click* | Woosh | 0.66 | 0.66 | 0.84 | 0.56 | +0.17 [-0.10, +0.44] | +0.17 [-0.07, +0.42] | +9/-3 +11/-6 +3/-1 |
| uppercut | punch* | Woosh | 0.29 | 0.04 | 0.27 | 0.16 | -0.02 [-0.20, +0.25] | +0.24 [+0.08, +0.47] | +16/-2 +4/-5 +11/-9 |
| walk carpet | footstep | Woosh | 0.48 | 0.41 | 0.54 | 0.63 | +0.07 [-0.22, +0.24] | +0.14 [-0.20, +0.40] | n/a +6/-4 +6/-13 |
| woof shepherd | dog_bark | Woosh | 0.03 | 0.85 | 0.93 | 0.90 | +0.90 [+0.80, +0.97] | +0.08 [-0.03, +0.18] | +34/-0 +29/-0 +39/-0 |
| water judge: dive splash | water | Woosh | 1.00 | 0.87 | 0.97 | 0.17 | -0.03 [-0.17, +0.00] | +0.11 [-0.05, +0.22] | +0/-14 +0/-9 +0/-3 |
| water judge: plop pond | water | Woosh | 1.00 | 0.74 | 0.71 | 0.30 | -0.29 [-0.48, -0.16] | -0.03 [-0.24, +0.18] | +0/-22 +2/-9 +0/-13 |

A star marks a stand-in reference class, n/a a judge that is not valid for that type, and the fresh rows carry 40 seeds each. The known rows at variants 100 to 119, with every comparison and McNemar count, are in `wstyles/confirm.txt`.

### Where the pick loses

The losses repeat the first round's two patterns. `named` has no collision rule, so impact words still turn into punches: "Body punch, a fist striking a surface, solid thud, dull impact" scores 0.16 where the phrasebook scores 0.97, "Water splash, water hitting a surface, wet slap, liquid impact" 0.00 against 0.60 (CLAP hears a punch on 18 of 20 seeds), the plain wood step on "solid thud, gentle creak" 0.20 against 1.00, and the 9 mm pistol on "sharp crack, metallic bang" 0.63 against 0.95. The fresh beer bottle falls from 0.88 to 0.13 on "Glass beer bottle break, Glass bottle break, a glass beer bottle shattering upon impact, sharp shatter, liquid splash"; AST and PANNs lose 29 and 17 of its 40 seeds against the phrasebook's "a crunch of broken glass". Water loses where the phrasebook answers every water request with its one tuned splash sentence. The pond plop drops from 1.00 to 0.71 on the water judge and from 0.47 to 0.04 on the splash judges with "Water plop, a lily pad hitting the water surface, soft splash, gentle ripple", and the two drips sit at 0.78 against 1.00 for "Big water splash, a heavy plop into deep water". The other large losses are the blade clashes on Woosh (the bench sword clash from 0.98 to 0.41 on "sharp metallic clang, ringing") and the door creaks, where the phrasebook's "hollow knock on wood" wins through the door judge's blind spot (`docs/generic.md`). Every losing row with all four prompts, CLAP's top labels and the shape medians is in `wstyles/confirm.txt`.

By the rule `named` does not meet the goal, and nothing changes: the writer stays behind `--writer` (off by default) and the style behind `--writer-style` (`v1` by default). The two rounds agree on what naming buys (+0.02 to +0.04 over v1) and on where the phrasebook keeps its edge, the 23 demo rows its sentences were tuned on and the water rows it answers with one splash. What is left to move is the sound field, and the selection block shows that no fixed rule tried so far moves it the right way on every row: dropping impact words saves the punch, the wood step and the pistol, and it costs the horn and the creak.

### Blind page, round two

`data/_gate/llm2/page/index.html` holds the 105 confirmation rows: the bench scenes and the plain requests at variants 100 to 103, then the 30 fresh requests at variants 0 to 3 in the last section. Every row has three options (no two sources wrote the same text on any row), four takes each, 1,260 clips (256 MB) at the game's playback gain. Letters are shuffled per row (the pick sits behind A on 41 rows, the phrasebook and caption on 32 each), with best, worst, "none sound right", Reveal and Download as on the first page, and the key in `data/_gate/llm2/page_key.json`.

Run order for this round: `subset` cuts the 105-row scenario file into a known and a fresh work folder, then `render-woosh`, `render-sa3`, `routes`, `judge-vocab`, `judge-water` and `report` run as before. `select` writes `selection.txt` and `selection.json` from the selection report, `confirm --parts` pools the fresh and known folders (rows with 40 and 20 seeds side by side) and applies the goal bars, and `page --parts --report --intro llm2` builds the page from both. `water_judge.py` reads its tier folders relative to the repo, so the water tiers sat under `data/_gate/llm2` while it ran and were deleted after; the judge clips stayed on F: and were deleted once the page was built. The launch scripts, the logs, every report and both decision files are in `data/_gate/llm/eval/wstyles/`. With every flag off the end-to-end bench under Unity's environment (`e2e_hybrid.py --tag writer2_off --extra --learn`) again served 75 of 75 clips bit-identical to `baseline_warm_1`, max abs 0.0 (`wstyles/compare_writer2_off.json`).

### Independent check

A second pass on the CPU rebuilt every number from the raw per-seed hits in the judge files, with a script that shares no code with `llm_eval.py`. It reproduced every point estimate of the selection and confirmation tables, every count of rows up and down and every McNemar sum (the demo's AST +8/-53, the water judge's AST +6/-94), and a separate bootstrap on another seed put the four bars at +0.06 [-0.05, +0.17], -0.15 [-0.28, -0.01], +0.11 [-0.03, +0.23] and -0.14 [-0.36, -0.01]. All four verdicts stand. `report`, `select` and `confirm` rerun with the final `llm_eval.py` on copies of those files gave byte-identical JSON and text, so the edits to the script after the confirmation run touched nothing that was judged. The pick does not hang on the tie band either: inside it, `named` would still have won on A (-0.020 against -0.027 for `texture`).

The order of events holds up. The rule file was created at 10:41 and amended at 11:40, a minute after the last style prompt and twelve minutes before the first clip. The selection block holds only the 75 known scenarios at variants 80 to 99. `selection.txt` was written at 12:58:00, and the first confirmation render began at 12:58:24 under a script written at 12:50 that reads the pick from `selection.json`. None of the 30 fresh event names appears under `data/_gate` or in the session's earlier work folders outside this round. Re-judging the 525 stored raw answers with the final `writer.py` returns the stored prompt 525 times out of 525, each writer id recomputes from its store's ident, and every rendered clip carries the prompt of the arm that used it. The v1 path is untouched: HEAD's `writer.py` and the final one give byte-identical judgments, messages and writer ids over all 1,260 raw answers of this round, superseded stores included. No style text or example names an eval event, material or demo object; "zipper" turns up elsewhere only as a mislabelled reference clip that `vocab_refs.py` drops from the punch judge.

On the page, every option matches the length, peak and decay of its claimed arm's four takes and fits no other arm, all 1,260 clips are present with none missing or unreferenced, the row markup names no arm, and the script equals round one's apart from its storage key and row count. With every flag off, `startup_compare` on `writer2_off` gives 75 of 75 identical at max abs 0.0, and the suite passes 550 tests. The raw judge files (renders, `vocab.json` and `water.json` for the S, Cf and Ck parts, 2.8 MB zipped) are in `wstyles/judge_outputs.zip`, since the reports alone hold no per-seed hits.

## Caption grounding (2026-10-21)

Both judged rounds trace the writer's losses to vocabulary: its sound words name collisions that Woosh renders as punches, and its splash and break lines are thinner than the captions Woosh learned from. Woosh was trained on Freesound, AudioCaps and WavCaps style captions, so the third round hands the writer those captions directly. A caption index holds 273,644 captions of real recordings, a small CPU embedder finds the ones nearest to a key, and three arms use them in three ways. The corpus is general knowledge of how recordings are described, and the grounding code holds no branch on an event, a material or an object.

| corpus | licence as published | captions kept of raw |
|---|---|---|
| Clotho v2.1 captions (Zenodo 4783391) | Tampere University licence: non-commercial research with attribution | 23,378 of 29,645 |
| AudioCaps (github cdjkim/audiocaps) | MIT licence file; the README limits use to academic purposes | 51,660 of 57,188 |
| WavCaps FreeSound (hf cvssp/WavCaps) | card says cc-by-4.0 with "only academic uses" | 171,332 of 262,300 |
| WavCaps BBC Sound Effects | the same card; BBC RemArc terms (personal, educational, research) | 25,197 of 31,201 |
| WavCaps SoundBible | the same card; SoundBible terms per clip | 814 of 1,232 |
| FoleySet uploader descriptions (on disk) | FoleySet CC BY 4.0, clips CC0 | 1,263 of 10,000 |

Only FoleySet is clear for shipping; the other five allow research use, so a shipped index would be rebuilt from open corpora. WavCaps AudioSet_SL was left out (machine captions of 10 s clips holding several sources at once), and ESC-50, HEAR and C3GD hold no captions. A caption enters only if it is plain ASCII, 3 to 25 words, free of digits, links and gear words, and not a repeat of one already kept.

No caption tied to a judge reference can be retrieved. `caption_index.py exclusions` collects the Freesound ids of every reference clip in `data/refs_heldout` and `data/refs`, every ESC-50 file under `data/esc50` and `data/real_footsteps`, the FoleySet files whose PCM equals a reference clip (128), the manifests' excluded lists and the WavCaps blacklist of ESC-50, UrbanSound8K and VGGSound clips: 3,098 ids. Every caption of those ids leaves, so does every caption by one of the 499 uploaders who own a reference clip (names compared without case across corpora, since Clotho and WavCaps spell some differently), every Clotho caption with no Freesound id, and every caption whose text equals one of the 5,811 texts tied to an excluded id. `caption_index.py overlap` then finds 0 index rows by id, by text and by uploader. Paraphrases remain: 303 of the 1,521 texts of the reference clips themselves have an index caption at cosine 0.95 or more, all generic lines from other recordings ("a door is being opened and closed" against "A door is being open and closed").

The embedder is bge-small-en-v1.5 (MIT, 33M parameters) on onnxruntime's CPU provider with 4 threads, CLS pooling and 64 tokens, 2.9 ms per text at the median. The query sums the embeddings of the request's material and event words and of the same words with the object, the surface, the creature and the action. One query takes the 256 nearest captions in 11.6 ms; the ranking (filters for speech words, banned writer words, quotes, brackets and capitals; a bonus for the request's event words and a smaller one for its material words; small penalties for people, a second event, production words and counts) brings the whole retrieval to 18.0 ms at the median. The index sits on `G:\sarge\captions\index`: 19.7 MB of captions, 210 MB of float16 embeddings held as float32 in memory, 134 MB of embedder; a process holding it measured 687 MB resident.

| arm | flags | what the prompt is | writer id |
|---|---|---|---|
| `rg_anchor` | `--writer llama --writer-style named --writer-ground anchors` | the named writer with the four nearest captions under its input, judged on the scene facts alone | `576771608f4c` |
| `rg_words` | `--writer ground --writer-ground words` | the request's event and material words, then up to four words shared by 2 or more of the 8 nearest captions | `60f081e13c37` |
| `rg_caption` | `--writer ground --writer-ground caption` | the request's event and material words in front of the nearest caption that passes the validator, in the v1 form | `de4d2bbb0255` |

The two arms with no model write a prompt in 22 ms at the median; `rg_anchor` takes 2.15 s, half a second over named. The decision rule (`data/_gate/llm/eval/ground/preregistered_rule.txt`) was written before the index existed: selection on the 75 known scenarios at variants 120 to 139 against named with a 0.02 margin, then confirmation of the pick against the phrasebook, the caption template and named on 30 new fresh scenarios (`llm_eval.FRESH3`, none reusing an earlier event name) at variants 0 to 39 plus the known ones at 140 to 159, with the round-2 goal bars. Development used only the round-2 fresh set and the template round's fresh requests, read as text, and a dated freeze recorded every digest, parameter and writer id before any of the 105 eval prompts existed.

The text already shows where each arm stands. The 2B writer barely reads the anchors: `rg_anchor` keeps named's collision words on most rows ("Wood footstep, a foot stepping onto a wooden floor, solid thud, wood creak") and equals named on 8 of 105. The caption arm swaps in real phrasing ("Wood footstep, foot is walking on wood floor", "Water splash, splashing of water", "Hound bark, hounds are barking") and sometimes the wrong material or event ("Stone footstep, footstep on gravel", "Sprint asphalt, skateboards on asphalt"), and every physics key named "impact" gets "an impact is expressed". The words arm often lands on setting words (range, floor, road). With every flag off the end-to-end bench under Unity's environment served 75 of 75 clips bit-identical to `baseline_warm_1`, max abs 0.0 (`ground/compare_ground_off.json`).

### Selection, variants 120 to 139

All six prompt sources were rendered new on the 75 known scenarios at variants 120 to 139 (445 scenario prompts, 423 distinct clips per seed, 8,460 clips in all) and judged as in the earlier rounds. `llm_eval.py select --ref wr_named --candidates rg_anchor rg_words rg_caption` wrote `selection.txt` at 17:15:45, and the confirmation script, written before the first clip, read `selection.json`, found no pick and stopped. No grounded arm is eligible: all three fail the water guard, and `rg_words` fails the gun guard as well. Nothing was rendered for confirmation, the 30 new fresh scenarios stay unused, and every default stays.

| arm, variants 120-139, minus the phrasebook | writer id | unseen U (22) | demo D (23) | guns G (6) | water W (7) | all A (52) | Q = (U + D) / 2 | eligible |
|---|---|---|---|---|---|---|---|---|
| named | `39800c4737a4` | +0.08 | -0.11 | -0.04 | -0.24 | -0.05 | -0.014 | reference |
| rg_anchor | `576771608f4c` | +0.10 | -0.09 | -0.03 | -0.34 | -0.04 | +0.004 | no (water) |
| rg_words | `60f081e13c37` | +0.07 | -0.10 | -0.15 | -0.30 | -0.05 | -0.013 | no (water, guns) |
| rg_caption | `de4d2bbb0255` | +0.13 | -0.09 | +0.01 | -0.22 | -0.02 | +0.018 | no (water) |

| rows (variants 120-139) | n | phrasebook | caption | named | rg_anchor | rg_words | rg_caption | rg_caption minus phrasebook, up/down | rg_caption minus named |
|---|---|---|---|---|---|---|---|---|---|
| unseen identity rows | 22 | 0.68 | 0.73 | 0.76 | 0.78 | 0.75 | 0.81 | +0.13 [-0.01, +0.28] 9/9 | +0.05 [-0.07, +0.18] 10/7 |
| demo identity rows | 23 | 0.65 | 0.55 | 0.54 | 0.56 | 0.55 | 0.55 | -0.09 [-0.24, +0.06] 9/12 | +0.01 [-0.15, +0.16] 13/8 |
| guns | 6 | 0.97 | 0.98 | 0.93 | 0.95 | 0.82 | 0.99 | +0.01 [-0.05, +0.10] 2/1 | +0.06 [-0.04, +0.18] 3/1 |
| water judge | 7 | 0.76 | 0.51 | 0.52 | 0.43 | 0.46 | 0.54 | -0.22 [-0.52, +0.05] 1/6 | +0.02 [-0.27, +0.28] 4/3 |
| every judged scenario | 45 | 0.66 | 0.63 | 0.65 | 0.66 | 0.65 | 0.68 | +0.02 [-0.09, +0.13] 18/21 | +0.03 [-0.07, +0.13] 23/15 |
| all plus water | 52 | 0.68 | 0.62 | 0.63 | 0.63 | 0.62 | 0.66 | -0.02 [-0.12, +0.09] 19/27 | +0.03 [-0.06, +0.12] 27/18 |
| footsteps | 15 | 0.52 | 0.48 | 0.45 | 0.45 | 0.60 | 0.51 | -0.01 [-0.20, +0.18] 8/6 | +0.06 [-0.13, +0.26] 9/6 |
| rows SA3 renders | 9 | 0.94 | 0.98 | 0.95 | 0.96 | 0.88 | 0.99 | +0.05 [-0.02, +0.17] 3/1 | +0.04 [-0.03, +0.13] 4/1 |
| rows Woosh renders | 36 | 0.59 | 0.55 | 0.57 | 0.59 | 0.59 | 0.60 | +0.01 [-0.13, +0.14] 15/20 | +0.03 [-0.09, +0.15] 19/14 |

The nearest real caption with the request words in front is the strongest source of this block on sounds nobody tuned for, and it still misses. It leads every other arm on the unseen rows (0.81), holds the guns at 0.99, and had it cleared the water guard it would have beaten named by 0.032, past the 0.02 margin. Its wins are the rows where the phrasebook falls back to its generic frame: the bench car horn from 0.20 to 0.93, the plain horn from 0.00 to 0.91, the bench bark from 0.15 to 0.85, the concrete steps from 0.63 to 0.99 on "Footsteps concrete, footsteps on concrete", the SA3 keyboard click from 0.62 to 1.00. One win is the judge's blind spot: the bench stone step rises from 0.27 to 0.82 on "Stone footstep, footstep on gravel, outdoors", the wrong material, which a footstep judge never hears.

The losses show what a three-to-six-word caption lacks. On the demo keys the phrasebook's long tuned sentences carry texture that a library caption leaves out. "Gravel footstep, footstep on gravel" drops the gravel steps from 1.00 to 0.41 and 0.61 where "coarse crunch with grinding pebbles and grit" scores 1.00; "Body punch, someone is punching or smacking a body, outdoors" takes the bench punch from 0.92 to 0.00, and CLAP hears a magic spell on 8 of 20 seeds and a footstep on 7. The plain wood step falls from 0.82 to 0.41 and the window shot by a rifle from 0.58 to 0.26, where named scored 0.88. On water the arm loses the bench water step (1.00 to 0.48 on "Water footstep, footstep in shallow water, outdoors") and the bench drip (1.00 to 0.21 on "Water drip, water splashes and drips, indoors"), the drip being the row where the phrasebook answers with its one splash sentence. The sparse-splash failure moved the right way: the plain splash rises from 0.00 for the phrasebook and named to 0.34 on the splash judges and from 0.34 for named to 0.83 on the water judge.

The other two arms confirm what the text showed. `rg_anchor` stays within 0.023 of named on the unseen, demo, gun and all-row pools and is the worst arm on water (0.43). `rg_words` is the best footstep source (0.60) and breaks the pistols on SA3, where its prompts shrink to the bare request: "TrackType: SFX, Pistol shoot, a single sound recorded up close, outdoors." scores 0.44 on the bench pistol and "Pistol shoot, fired" 0.60 on the plain one, and CLAP loses 22 gun seeds against the phrasebook. The CLAP caveat of the rule applies to the caption arm. Against named on the demo rows CLAP gains 83 seeds and loses 48 while AST (+26/-36) and PANNs (+58/-94) side with named, the pattern the caveat warned of: CLAP learned from AudioCaps and Clotho captions, the index holds them, and part of the arm's CLAP gain may be that shared phrasing. The consensus is a minimum over judges, which keeps that from carrying a row alone. The water reference itself moved between seed blocks: named sat at -0.13 on water at variants 80 to 99 and at -0.24 here.

### Blind page, round three

`data/_gate/rag/page/index.html` holds the 75 selection rows, since the rule rendered no confirmation rows: six sources per row (five on the five rows where `rg_anchor` wrote named's exact text), four takes each at variants 120 to 123, 1,780 clips (352 MB) at the game's playback gain, letters shuffled per row, with best, worst, "none sound right", Reveal and Download as on the earlier pages. The key is in `data/_gate/rag/page_key.json`.

### Independent check

`ground/tools/indep.py` shares no code with `llm_eval.py` and rebuilds every number from the raw per-seed hits in `vocab.json` and `water.json`. It reproduced every selection number to the fourth decimal and the same empty eligible list, and on round 2's selection block it reproduced that table exactly; a separate bootstrap (seed 777) puts the caption arm's water delta at -0.22 [-0.52, +0.04]. All 8,460 render rows carry the prompt of the arm that used them, with 20 seeds for every scenario and arm. `ground.py` and `writer.py` still match the freeze. `llm_eval.py` gained only the round-3 page intro after `selection.txt` existed (sha1 now `31423940`), and `report` and `select` rerun with it on copies of the judge files gave byte-identical `report.json` and `report.txt` and the same selection. On the page every option's four takes equal the clip of its claimed arm (max abs 1.5e-5, under one int16 step), the markup names no arm, and all 1,780 files are present and referenced. With every flag off, `e2e_hybrid.py --tag rag_off --extra --learn` under Unity's environment served 75 of 75 clips bit-identical to `baseline_warm_1`, max abs 0.0 (`ground/compare_rag_off.json`), and the suite passes 563 tests. The launch scripts, progress logs, the report, the rule, the selection files and the raw judge files (`judge_outputs_S3.zip`) are in `data/_gate/llm/eval/ground/`.

### Second independent check

A third pass on the CPU, with scripts that share no code with `llm_eval.py`, `indep.py` or `caption_index.py` (`ground/verify2/`), repeated every check from the saved files. It rebuilt the selection table from `vocab.json` and `water.json` to the fourth decimal, the same empty eligible list, every group mean and every McNemar count of the pooled tables, and a third bootstrap seed puts the caption arm at U +0.13 [-0.01, +0.28], D -0.09 [-0.24, +0.06], G +0.01 [-0.05, +0.10] and W -0.22 [-0.52, +0.04]. The verdict does not hang on a seed. The water guard fails by 0.07, and the arm would have cleared the margin over named had it passed.

The exclusion check started from the reference audio itself. The 6,086 reference files in `data/refs_heldout` and `data/refs`, mapped through the manifests and the FoleySet table with no mismatch, plus every ESC-50 file on disk, give 979 Freesound ids; hashing the PCM of all 10,000 FoleySet files adds 116 ids of re-uploads. No index row carries one of those ids, comes from one of their 496 uploaders (names folded to lower-case letters and digits, which also joins Clotho's garbled spellings) or repeats one of their 1,516 caption texts. 132 index captions share a stemmed word set with a caption of a reference clip, all generic lines of other recordings ("Door open and closing" from AudioCaps), and none of the 317, 607 and 82 caption rows the three arms retrieved for the 105 scenarios is tied to a reference. A scan of `ground.py` finds no string literal or comparison that holds an eval event, material or object word apart from the protocol words loop and object, and the lines added to `writer.py` hold none.

The order of events holds. Every grounded prompt of the 105 eval scenarios was written between 16:02:40 and 16:06:31, after the freeze; the 11 development stores hold no eval scenario text; the renders hold variants 120 to 139 of the 75 known scenarios and nothing else; `C3f` and `C3k` hold only their scenario files; and no file under `data/_gate`, `G:\sarge` or the work folder is named after a FRESH3 scenario. The index digest and four of the five frozen code hashes match the freeze, `llm_eval.py` being the one edited after selection, and the archived judge files equal the originals byte for byte.

On the round-3 page each of the 1,780 takes fits the clip of its claimed arm at gain 1.0 (worst residual 1.6e-5) and fits no other arm's clip, arm words appear only in the intro, and the script differs from round two's only in its storage key, row count and download name. On the decision page (`data/_gate/decide`) all 208 files are present and referenced and equal their source files by md5, the prompt rows map letters to arms as the round-2 key does, both merged options are one audio file on the loudness page, each prompt source sits behind A, B and C five times, the identity and water scores shown at Reveal equal `wstyles/confirm.json`, and the script passes `node --check`. The suite passes 563 tests, `startup_compare` on `rag_off` again gives 75 of 75 identical at max abs 0.0, and the CPU bench repeats within a tenth (19.9 ms per retrieval at the median against 18.0). One number moved: the generic-frame split under "Still open" had come from a pattern that missed the bench explosion's "single heavy explode on a surface" and caught the metal rain row's "on a steel deck plate", and it now follows its stated definition.

## Untried levers (2026-10-22)

Five prompt rounds had the same shape: every generic source won on sounds nobody tuned for and lost on the demo keys and on water. This round tested the three levers no round had touched: a prompt ceiling, the slower Woosh model, and choosing among seeds. Each rule went into `data/_gate/untried/eval/` before its first clip (`rules_sha1.txt` holds the digests and file times), the known scenarios ran at seeds no earlier run had used (160 to 219), and the 30 scenarios of `llm_eval.FRESH3` were rendered for the first time. Judges, windows and statistics are those of the earlier rounds. `docs/untried.md` holds the independent recheck, the model scan and the ordered list of what is still untried.

### The prompt ceiling

A strong model wrote one prompt per scenario from the scene facts alone, blind to the phrasebook, the judges and every earlier prompt (`ceiling_prompts.json`, method in `ceiling_method.txt`): the plain label a sound library would use, two or three acoustic details, space words from the room field only, and each model's preferred form ("TrackType: SFX, ... Length: 1 seconds" for SA3, one lowercase caption for Woosh). It ran as a single arm against the phrasebook, the caption template and round 3's grounded caption on the 75 known scenarios (variants 160 to 179) and FRESH3 (variants 0 to 39).

| ceiling minus the phrasebook | rows | delta | bar | verdict |
|---|---|---|---|---|
| unseen (22 known, 30 fresh) | 52 | +0.12 [+0.01, +0.23] | lower bound above 0 | pass |
| demo | 23 | -0.06 [-0.15, +0.06] | -0.05 | fail |
| guns | 11 | +0.14 [+0.03, +0.30] | -0.10 | pass |
| water | 9 | -0.28 [-0.57, -0.05] | -0.10 | fail |
| ceiling minus `rg_caption`, unseen | 52 | +0.01 [-0.08, +0.10] | | |

That is verdict B of the rule: a judge-blind writer wins the unseen sounds and still loses the demo, by 0.01 past the bar. The more telling line is the last one. On unseen sounds the strong prompts land where the nearest real caption with the request words in front already sits, so a better or distilled small writer has no headroom left under these judges. The strong model fell into the old traps too. Its splash prompts said "slap" ("a person belly flopping into a pool, loud flat slap on the water", 0.73 to 0.09; "heavy splash with a hollow wooden slap", 0.43 to 0.00), which Woosh hears as a punch whoever writes it, and its footstep details ("hollow knock of a shoe heel on boards", "heel click with a light gritty scuff") lost the wood, snow and concrete steps to the phrasebook's tuned lines. The guns are the one place it led everything (0.99 against 0.85).

A follow-up tested the split the last round left open (`rule_1b_hybrid.txt`, written after the ceiling numbers and before any hybrid number): keep the phrasebook where `prompts.build` has a tuned line and use a generic source only on keys that reach its last branch. Forty-eight of the 75 judged rows are such frame rows and no demo row is, so the demo stays where it was. With the caption template on the frame rows the hybrid gained +0.13 [+0.04, +0.23] on unseen rows (+0.17 [+0.04, +0.30] on the fresh ones), +0.10 on guns and +0.01 on water, inside all four bars. The rule Flow passed below makes it a side note, since the template alone does as well once it plays on Flow.

### Woosh Flow

Rule 2 rendered the phrasebook and the caption template on Woosh Flow (`fl_phrasebook`, `fl_caption`) with the same seeds as the DFlow clips. On the 36 judged Woosh rows Flow raised the phrasebook by +0.11 [+0.01, +0.23] and the caption template by +0.19 [+0.10, +0.30], held the water judge on the phrasebook and lifted it from 0.30 to 0.73 on the template; the class is BETTER, and the numbers, the cost and the upgrade path are in `docs/hybrid.md`.

The same report held a comparison no rule had asked for: the caption template on Flow against the served path cleared all four goal bars on that block. Rule 4 (`rule_4_flow_caption.txt`) tested it on clips rendered afterwards, the known scenarios at variants 200 to 219 and FRESH3 at 0 to 39, all four arms judged together.

| rule 4, against the phrasebook on DFlow | rows | phrasebook | caption on Flow | delta, up/down | phrasebook on Flow | bar |
|---|---|---|---|---|---|---|
| unseen | 52 | 0.61 | 0.86 | +0.24 [+0.14, +0.35] 38/8 | +0.08 [+0.02, +0.15] | pass |
| fresh | 30 | 0.58 | 0.87 | +0.29 [+0.16, +0.42] 24/3 | +0.05 | |
| demo | 23 | 0.67 | 0.65 | -0.01 [-0.15, +0.14] 12/10 | +0.03 | pass |
| guns | 11 | 0.81 | 0.93 | +0.12 [+0.01, +0.27] 8/2 | +0.00 | pass |
| water | 9 | 0.81 | 0.78 | -0.03 [-0.14, +0.02] 0/3 | -0.03 | pass |
| every judged row | 75 | 0.63 | 0.79 | +0.17 [+0.08, +0.26] 50/18 | +0.06 | |

The caption template on Flow meets the goal the prompt rounds set, the first generic setup to do so (`flow_caption_confirm.txt`). The guns in that row are the caption template on SA3 medium, which the template round already showed to be strong. Its wins are the frame rows (barks 0.10 to 1.00, horns 0.00 and 0.10 to 0.97 and 1.00, the fresh fist strike 0.12 to 1.00), its losses are glass on the template's short sentence (a pistol through a bottle 0.71 to 0.06, the wine glass 0.64 to 0.26, where the phrasebook on Flow scores 1.00) and the wood steps (0.88 to 0.36). Keeping the phrasebook's tuned lines on Flow and the template on the frame rows (post hoc on the same data, `hybrid_flow_posthoc.txt`) adds +0.01 overall and +0.04 on the demo, inside the noise. Served, the setup is `--prompts template:caption+tt --woosh-upgrade flow`: the first play of a key is the template on DFlow and later plays are Flow, so its first hits score like the caption arm above until the upgrade lands. The ear decides before any default moves.

### Best of four

Rule 3 rendered 20 seeds per arm (variants 180 to 199) and picked one take from each group of four with `laion/larger_clap_general`, a CLAP checkpoint none of the judges use (a larger sibling of the CLAP judge in the LAION-CLAP family, so its independence is partial), by the cosine between the served clip and its prompt with the SA3 tags removed. The verifier passed its positive control before any pick (median AUC 0.95 on the held-out references of each prompt's type against every other family), the picks were saved before any judge ran, and the first take of each group stands for a single random take.

| pick minus first take, 45 judged rows (5 picks each) | phrasebook | caption | ceiling |
|---|---|---|---|
| identity | +0.06 [-0.04, +0.16] 17/6 | +0.03 [-0.07, +0.14] | +0.13 [+0.02, +0.24] 20/5 |
| water judge, 7 rows | +0.00 | +0.21 [-0.17, +0.57] | +0.24 [-0.09, +0.53] |
| Audiobox PQ | -0.04 [-0.14, +0.06] | +0.04 | -0.05 |

On the served prompt the gain is small and its interval holds zero, so by the rule nothing is built. The verifier tracks the judges inside a row (correlation 0.27 to 0.29 with the AST, PANNs and CLAP hits, positive on 75 to 83% of rows), leaves production quality where it was and narrows the picked takes by about a tenth (median log-mel distance 2.29 against 2.54). It helps most on the descriptive ceiling prompts, and post hoc the ceiling with best of four reaches 0.78 against 0.62 for the served path. One pick costs three extra renders and four verifier passes per take, 192 ms on DFlow and 553 ms on SA3, or 1.1 s and 3.3 s for a key at six takes, all off the first hit.

### Blind page, round four

`data/_gate/untried/page/index.html` holds the rows that decide the most: eleven where the four prompt sources part ways (the ceiling's three largest wins and losses, the drips and splashes the water judge reads, two rows no judge covers) and sixteen where the Woosh model is the question (every glass and water row, the punches and the bark), with the phrasebook on DFlow, the phrasebook on Flow and the caption template on Flow. Three takes per option, 276 clips (45 MB) at the game's playback gain, letters shuffled per row, best, worst, "none sound right", Reveal and Download as on the earlier pages, and the key in `data/_gate/untried/page_key.json`. Every take was checked against the clip of its claimed arm and against every other arm's clip.

## Unity package

`SargeScene` builds the record and caches the static part per component. `SargeEmitter`, `SargeFootsteps` and `SargeAutoCapture` attach it when `sendScene` is on, and the predictors attach the scene their paired emitter will send, so a predicted key matches the fired one: `SargeImpactPredictor` uses its `SargeAutoCapture`, `SargeMotionPredictor` sends a step scene only when its root carries a `SargeFootsteps` (the demo's `SargeFirstPerson` fires steps itself, so its predictions stay plain), and `SargeAimPredictor` sends one only through `SceneResolver`. The server also gives a prediction or habit guess without a scene the last scene fired for the same event, material and context, as `--world` does for physics.

Replies carry `scene_key` and `prompt_rev` when they apply. With `localPlayback` the client maps each raw scene string to the server's `scene_key`, keys its held takes on it, sends `have_scene` and `have_rev`, and on a `prompt_rev` that differs from the one it holds destroys the held takes of that key before storing the new one, so a template take never outlives its upgrade. A lower revision also counts: it comes from a restarted sidecar or from an older reply on another lane overtaken by a newer one, and either way the next play brings the key back in line. The server reads the revision under the same lock that picks the clip, so a reply never pairs an old take with a new revision. The package, the demo, the editor scripts, the probe editor and the sample compile with Unity 2022.3.8f1's Roslyn, and a Mono run of the new `SargeClient` against the Python server with a fake writer showed the scene key coming back, `meta_only` honouring `have_scene` and `have_rev`, the revision reaching 1 after the swap, a manifest ack and a prediction carrying a scene. No demo file changed.

Old clients still work: every new field is optional both ways, old servers ignore the new request fields and answer a MANIFEST with a non-fatal `unknown_type` error that the client swallows, and with the new flags off a new server answers an old request byte for byte as before. An old client that already holds take 0 of a key keeps playing that take after an upgrade, since its clip cache knows nothing of revisions; a new client is needed to hear upgrades of keys it has already played.

## Tests

`tests/test_writer.py` (30 tests, CPU): name cleaning and the fixed-order scene key; physics objects and kinds from `phys`; the key string and seed with and without a scene; the cache path moving only with a prompt; the writer input reading only the key and dropping size and build; the writer id; acceptance, composition, repair of banned, controller, leaked and copied words, the per-field loop check and rejections; the store keeping the first text across reloads; queue order; through a live server the template first and the swapped prompt after with `prompt_rev` and `have_rev`, a predicted key taking its prompt before the first hit with its seed unchanged, the SA3 form on guns and the Woosh form elsewhere, game prompts winning both ways, a store pre-filled at startup with a read-only writer never calling the model, next-miss mode and a swap that waits for renders in flight, the scene on the wire and predictions taking the last scene; byte-identical replies with and without the new fields; the manifest with and without a writer; the `run_sidecar.py` flags; the CPU-only llama-server command line and the deterministic request bodies of both backends; model resolution from Ollama manifests; the yield gate wrapping renders and nesting; and the job object pinning a child. The whole suite passes. `tests/test_writer_styles.py` (11 tests, CPU) pins the v1 writer id and its judging, gives every style a separate id, checks the naming rule, the collision rule and both forms on both routes, traces the executed lines of `writer.py` to show every style takes one code path for 191 eval event, material and object words and 60 random ones with outputs equal up to the substitution (the only words left out are protocol words such as break, loop, open and step, and the validator classes: blue, large, impact, into, player), and checks that no style text or example names an eval scenario and that the 30 fresh requests reuse no earlier event name. `tests/test_llm_eval_rule.py` (5 tests, CPU) checks the selection rule as `llm_eval.py select` runs it (inclusive bars, the tie band and its order, the margin over v1, the stop when no style is eligible), the goal verdict of `confirm` (the unseen lower bound must clear zero, the demo bar sits at -0.05) and the pairs the report compares. `tests/test_ground.py` (13 tests, CPU) re-judges every record of the five round-2 stores with the final `writer.py` and gets the stored prompt back, recomputes each writer id from its store, checks that `llm_eval` builds the round-2 scenarios and phrasebook and caption prompts unchanged and that every ground flag defaults off, traces the executed lines of `ground.py` to show all three modes take one code path for the eval event and material words and 40 random ones with outputs equal up to the substitution, checks the filters, the ranking, both compositions and the anchors reaching the model while the validator reads the facts alone, checks that retrieval repeats across query order and thread counts, that no index row carries an excluded id or text and that every reference clip's id is excluded, that the 30 new fresh scenarios reuse no earlier event name, and the round-3 selection against named. The whole suite, 563 tests, passes.

## Still open

- **The ear.** The judges have spoken (sections "On sound", "Writer styles", "Caption grounding" and "Untried levers"); the blind pages `data/_gate/llm/page/index.html` (v1, variants 40 to 43), `data/_gate/llm2/page/index.html` (`named`, with the 30 fresh requests), `data/_gate/rag/page/index.html` (all six sources of the round-3 selection block) and `data/_gate/untried/page/index.html` (the ceiling, and the phrasebook and the caption template on Flow) are the user's call, and they are the only verdict on the 26 scenarios with no validated judge.
- **The caption template on Flow as a default.** It met all four goal bars on new clips (section "Untried levers"); the step left is the ear, then a default change of `--prompts template:caption+tt --woosh-upgrade flow`, whose first plays stay on DFlow until the upgrade lands. Glass on the template's short sentence and the wood steps are its weak rows.
- **Captions win where the phrasebook has no tuned line.** Confirmed on FRESH3 and new seeds (`rule_1b_hybrid.txt`): the phrasebook on its tuned keys and the caption template on frame keys passes all four bars on DFlow, and on Flow it adds nothing measurable over the template alone.
- **Plausible inventions cost sound.** The naming rule fixed the renaming: the exploding barrel comes back as "Barrel explode, a barrel ruptures violently" and scored 0.95 against 0.00 for v1 at variants 100 to 119. Invention inside the right name is still there. The fresh beer bottle gains a "liquid splash" (0.88 to 0.13), and the pond plop turns into a lily pad hitting the water (0.47 to 0.04).
- **The sound field is right on some rows and wrong on others.** Without it the splash and the concrete step recover and the gravel step, the horn and every gun fall, in both seed blocks. The collision rule of `texture` saves the punch, the wood step and the pistol and loses the horn and the creak in the same block, because the sound line that steers the model away from "thud" also steers it away from "honk". A sound field that keeps the event's signature word and drops only collision words is untested; a per-row choice made from these judges would only fit them.
- **The second model.** Phi-4-mini (MIT) never went through the audio test.
- **More first hits.** Scene fields multiply keys and each new key pays one template first hit.
- **Swap renders share the GPU.** A swap starts only while no realtime render is in flight, but a miss that arrives during a swap waits for it (about 50 to 120 ms on Woosh, 150 ms on a gun), as it would behind a prediction render.
- **Unbuilt pieces.** The Jev-style yes or no gate (`--writer-gate`), the cloud plugin (a backend is any object with `ident`, `start`, `complete` and `close`, so it would slot into `sarge.writer.build`) and distillation into Granite 350M are designed only.
