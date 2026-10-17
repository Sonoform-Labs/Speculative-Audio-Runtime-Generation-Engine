# Physics context (the "world model")

SARGE used to hear a collision as three words from the game: an event name, a surface material and an intensity. A 1 kg inflated ball bouncing on a steel tile therefore arrived as `impact|metal|hard`, the prompt grammar turned that into a struck steel plate with a ringing clang, and the model rendered a 609 Hz tone that rang for the whole 1 s window. Physics context fixes the cause. Unity sends a small description of the two bodies in contact, the server folds it into a coarse physical bucket, and the bucket drives two things: the words in the prompt and the numbers in the post chain. Everything stays generated and nothing is authored per game or per sound.

All of it is off by default. With the flags off, every prompt, cache key, seed, cache path and server reply is bit-identical to HEAD (proof under "Checks run on CPU").

## What Unity captures

`SargePhysics.Json(...)` (package, `unity/Sarge/Runtime/SargePhysics.cs`) writes one JSON object per contact. `SargeAutoCapture` calls it from `OnCollisionEnter` when `sendPhysics` is on, `SargeImpactPredictor` calls it for a predicted contact, and the demo `SargeBreakable` calls it for a pane that breaks.

| field | meaning |
|---|---|
| `a`, `b` | the body that reports the contact and the body it touched (`b` is absent for a break) |
| `shape` | collider type: sphere, box, capsule, mesh, terrain, wheel |
| `ext` | rotation-invariant collider size in metres (local size times lossy scale), so a tumbling box keeps one size |
| `dyn`, `m` | 1 with the mass for a non-kinematic rigidbody, 0 for static or kinematic colliders |
| `pm`, `bnc`, `fr` | PhysicMaterial name, bounciness and dynamic friction |
| `rm`, `tag`, `name` | renderer material name, `SargeSurface.material`, and the GameObject name without `(Clone)` |
| `n`, `vn`, `vt` | contact normal, normal and tangential relative speed (m/s) |
| `j`, `dist` | impulse magnitude (N s) and distance to the AudioListener (logged; the bucket ignores both) |
| `kind` | optional override: `hit`, `bounce` or `break` |

The static part of each body is cached per collider instance, so a contact costs one string build. The JSON rides on the existing EVENT and PREDICT messages as an optional `phys` field. Old servers ignore it, and old clients never send it.

## From physics to a bucket

`src/sarge/world.py` turns the JSON into a short string such as

```
bounce;a=rubber,large,inflated,ball;b=metal,huge,static,floor;c=soft;r=short,metal
```

that becomes the fifth field of the cache key (`CacheKey.phys`). The steps, all generic:

1. **Material ladder** per body: the `SargeSurface` tag, then material words in the PhysicMaterial name, the renderer material and the object name (wood, wooden, steel, metallic, glass and so on), then a small English noun lexicon (window and bottle mean glass, barrel means metal, crate means wood), then bounciness of at least 0.6 means rubber, then a density band from mass and collider volume, then `object`.
2. **Build**: static for colliders without a dynamic rigidbody; inflated for a light sphere or capsule of rubber or plastic (density under half the material's); hollow for anything under a quarter of the material's density; solid otherwise. Unity's default mass of 1 kg counts as real, because the physics engine simulates that mass.
3. **Size class** from the largest extent: tiny under 5 cm, small under 20 cm, medium under 60 cm, large under 2 m, huge above. The shape gives a noun (ball, box, pane, plate, board, rod, cylinder); static bodies become a floor or a wall from the contact normal.
4. **Contact time** at a reference speed of 3 m/s, so the bucket stays put across a tenfold speed range: a pneumatic spring for inflated bodies (`pi * sqrt(m / (2 pi R P0))`, P0 = 50 kPa) and Hertz otherwise (`2.87 (m*^2 / (R* E*^2 v))^(1/5)`), with E, density and Poisson ratio from textbook values per material word. Classes: sharp under 0.25 ms, crisp under 1 ms, dull under 4 ms, soft above.
5. **Ring class** (dead, short, ringing, long) from the material ring rank (metal 3, glass and ceramic 2, stone, wood and ice 1, the rest 0): static bodies ring one rank less, hollow resonant bodies one more, a body landing on a dead surface one less, tiny bodies one less, and soft contacts (over 4 ms) one less. The ringer is the body with the higher rank.
6. **Kind**: `break` from the override or an event name with break, smash or shatter; `bounce` when the reporting body has bounciness of at least 0.5; `hit` otherwise. Slides and rolls are detected and logged (`world.analyse(...)[1]["motion"]`) and play as the same bucket in v1.
7. **Intensity** keeps the client's value. Only when a client sends phys without an intensity does the server derive it from the normal speed (soft under 1.5 m/s, hard from 6 m/s).

Contact time sanity against measured values (unit tests): a glass marble on stone at 1 m/s gives 61 us (measured about 50 us), a tennis ball 7.3 ms (measured 4.6 to 5.4 ms), a basketball 12.7 ms (measured 10 to 15 ms). The demo ball comes out at 10.2 ms.

## From the bucket to prompt words

`prompts.build(event, material, intensity, context, phys)` returns the HEAD string whenever `phys` is empty. With a bucket it writes one template from `world.prompt_words`: a head, the object and what it does, the contact sound, the ring, then the usual foley tag and space words from the context. The object noun comes from the game's event name when the name carries one (`glass bottle hit` gives "glass bottle", `impact` gives nothing and the collider shape supplies "ball"). Examples from the demo:

| contact | prompt |
|---|---|
| bouncer ball on the metal tile, hard | Rubber ball bounce, a large inflated rubber ball bouncing hard on a metal floor, a soft muffled bump, a brief metal ring that dies away quickly, foley, close-distance, exterior, open air |
| same ball on grass | ... bouncing hard on the grass, a soft muffled bump, short and dry, stopping at once, ... |
| glass bottle onto gravel | Glass bottle impact, a hollow glass bottle hitting the gravel, a dull thud, a ringing glass tail, ... |
| crate onto the metal barrel | Wooden crate impact, a large hollow wooden crate hitting a large hollow metal cylinder, a crisp hard knock, a long resonant metal ring, ... |
| window pane break | Glass pane break, a huge glass pane smashed, a hard crack and a dense shatter, bright tinkling shards scattering and settling with a long tail on paving stones outdoors, close perspective foley |

Every physics prompt over the demo pairs, three intensities and four spaces fits in 37 to 54 roberta tokens, under the Woosh text encoder's 77.

## From the bucket to post-chain numbers

`world.shape(bucket)` hands numbers to the generic DSP in `src/sarge/post.py` (the loudness work package owns that file; Woosh `finish(..., shape=...)` routes a non-empty shape through `post.process`):

| ring | `t40_s` target | `floor_db` (natural trim) | `drive` cap |
|---|---|---|---|
| dead | 0.08 | -45 | 1.0 (no saturation) |
| short | 0.2 | -45 | 1.0 (no saturation) |
| ringing | 0.6 | -55 | none |
| long | 1.5 | -55 | none |

| contact | `punch_scale` | `lowpass_hz` |
|---|---|---|
| sharp | 1.0 | none |
| crisp | 0.8 | none |
| dull | 0.5 | none |
| soft | 0.25 | 400 |

`decay_cap` acts only when a clip's measured t40 exceeds 1.25 times the target, applies an exponential gain from 15 ms after the peak, trims at -50 dB with a 20 ms fade and never lengthens a clip. The prompt words carry identity; the DSP enforces decay, attack and saturation, because a diffusion model follows "dies away quickly" only loosely.

## Flags

| flag | where | effect |
|---|---|---|
| `--world` | `scripts/run_sidecar.py`, `serve(world=True)` | the server buckets a `phys` object into the key, the reply gains `phys_key`, the PREDICT ack gains `phys_keys` (aligned with the predictions, empty strings for plain ones); a prediction or habit guess that carries no `phys` takes the bucket of the last fired event with the same event type, material and context (a habit guess also takes that event's intensity), so a game that predicts without physics still warms the key the real hit will ask for; `--vram-kb` defaults to 256000 with `--world` |
| `sendPhysics` | `SargeAutoCapture` (Inspector, default off) | attach the contact JSON to every captured collision |
| `speedGain` | `SargeAutoCapture` (default off) | PlayOneShot volume `(vn / sqrt(soft * hard))^1.2`, clamped to plus or minus 6 dB |
| `writePhys` | `SargeImpulseProbe` (default off) | adds a quoted phys column to the hit CSV for offline replay |
| P key | demo `SargeFirstPerson` | toggles `sendPhysics` on every AutoCapture and `SargeBreakable.SendPhysics` |

The Woosh worker needs no flag: it reads `phys` from the request, builds the key, the prompt and the seed with it, and hands `finish` the shape. Guns stay on SA3 with HEAD prompts, since impacts route to Woosh.

## Key cardinality

A bucket is a function of the two bodies and the normal class, so each colliding pair yields one bucket per intensity, at most two when the same pair meets on both a floor and a wall. Replaying the 17:40 session (272 logged ball hits plus 130 synthetic prop hits, as in audit 3) gives 25 keys with `--world` against 25 at HEAD and zero bucket changes across 376 consecutive hits of the same pair. A game grows by one key per new object class and intensity on each surface it touches; the 256 MB clip budget under `--world` holds about 650 one-second stereo clips.

## Checks run on CPU

- `tests/test_world.py` (25 tests): determinism, contact times within 2x of the measured values, bucket stable from 1 to 10 m/s, material ladder order, ring classes, break kind, slides logged and served as the same bucket, shape numbers, prompt words, the HEAD seed pinned, empty phys leaves prompt, key string, seed and cache path untouched, replay at most 33 keys with flicker at most 2%.
- `tests/test_world_wire.py` (8 tests): without `--world` the replies are byte-identical with and without phys; `--world` without phys equals world off; with phys the reply carries `phys_key` and the clip is a separate cache entry; intensity from speed only when absent; malformed phys falls back to the plain key; PREDICT acks align `phys_keys`; the remote generator omits empty phys and keeps `gen_ms`; the Woosh shape hook follows the `finish` signature.
- HEAD identity (scratch `wpb/identity.py`): 18,220 keys (56 session keys, 20 vocab-eval keys, an 18,144-key grid of events, materials, intensities and contexts) give identical prompts, key strings, seeds and cache paths against `git show HEAD`; a 26-frame scripted session, phys fields included, returns byte-identical replies from HEAD and the new server in realtime and predict modes.
- Decay cap on the Woosh sweep raws (impact, break, metal and splash renders at 101 and 501 frames): never longer than the input (0 of 480); within 0.67 to 1.5 times the target on 97% (dead), 91% (short), 85% (ringing) and 58% (long) of the clips it acted on; onset flux lower on 2 of 480.
- The served ball-on-metal clip (t40 0.96 s, 0.98 s long) passes through the `short` shape at t40 0.20 s and 0.28 s long.
- Roslyn: the package, the demo and the editor assemblies compile with Unity 2022.3.8f1's compiler; a Mono run of `SargePhysics` and `SargeClient` produced valid JSON that the Python server buckets.

## GPU checks still to run

1. **Word following.** For each word group (contact, ring, size, build), render 6 seeds of a minimal pair that differs only in that group, measure t40, the spectral centroid of the first 50 ms and the energy below 300 Hz, and keep a group only if at least 5 of 6 seed pairs move the physical way (softer contact: lower centroid; shorter ring: shorter t40; larger: more low energy).
2. **Decay targets after the full chain.** Render the demo pairs with `--world` (ball on each tile, bottle on glass and gravel, barrel and crate on gravel, crate on barrel, window break) and confirm the post-chain t40 lands within 0.67 to 1.5 times the target on at least 80% of clips for dead, short and ringing.
3. **Blind page.** Record a Play session with `writePhys` on the probe, replay the same events through HEAD and `--world`, and rate them blind.
4. **Play session with P.** The ball on metal should sound like a big rubber ball with a short tick of metal, at most about 0.3 s long.

## Risks

The model may ignore physical words, which is why the numbers live in DSP. A bouncy steel ball whose names carry no material word reads as rubber. Unity's default mass makes most props hollow, which matches the simulation and may surprise a designer. Slides and rolls play as hits in v1. The demo's bounciness 1 with Maximum combine adds energy on each bounce, so every ball hit stays hard. Each new prompt misses the Woosh prompt cache once (about 18 ms).
