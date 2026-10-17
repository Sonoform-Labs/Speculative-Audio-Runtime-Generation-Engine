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

`prompts.build(event, material, intensity, context, phys)` returns the HEAD string whenever `phys` is empty. With a bucket it writes one template from `world.prompt_words`: a head, the contact sound, the object and what it does, then the usual foley tag and space words from the context. The object noun comes from the game's event name when the name carries one (`glass bottle hit` gives "glass bottle", `impact` gives nothing and the collider shape supplies "ball"). Examples from the demo:

| contact | prompt |
|---|---|
| bouncer ball on the metal tile, hard | Rubber ball bounce, a soft muffled bump, a rubber ball bouncing hard on a metal floor, foley, close-distance, exterior, open air |
| same ball on grass | Rubber ball bounce, a soft muffled bump, a rubber ball bouncing hard on the grass, ... |
| glass bottle onto gravel | Glass bottle impact, a dull thud, a glass bottle hitting the gravel, ... |
| crate onto the metal barrel | Wooden crate impact, a crisp hard knock, a wooden crate hitting a metal cylinder, ... |
| window pane break | Glass pane break, a glass pane smashed, a hard crack and a dense shatter, bright tinkling shards scattering and settling with a long tail on paving stones outdoors, close perspective foley |

The bucket still carries size, build and ring, and the DSP still uses them, but the prompt no longer says them. The first version wrote "a large inflated rubber ball ... a brief metal ring that dies away quickly", and the GPU check below found that Woosh DFlow follows the contact words and ignores or reverses the size, build and ring words, so those three word tables (`SIZE_WORD`, `BUILD_WORDS`, `RING_WORDS`) now hold empty strings. They stay as tables so `scripts/bench/world_render.py explore` can try new wording against them. Every physics prompt over the demo pairs, three intensities and four spaces fits in 30 to 46 roberta tokens, under the Woosh text encoder's 77.

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

`decay_cap` acts only when a clip's measured t40 exceeds 1.25 times the target, applies an exponential gain from 15 ms after the peak, trims at -50 dB with a 20 ms fade and never lengthens a clip. It runs last in `post.process`, after the low-pass, the amplifier, the stereo widening and the level stage. In the first version it ran right after the natural trim, and the 400 Hz low-pass that follows a soft contact then took the bright attack off the peak and left the low tail standing: two dead ball clips the cap had brought to 0.07 and 0.06 s came out of the chain at 0.32 and 0.21 s, and the loudness limiter stretched two more. Capping the finished envelope fixes that by construction. The prompt words carry identity; the DSP enforces decay, attack and saturation, because the model does not follow decay words (measured below).

A physics request also judges its silent takes by what the chain keeps. `woosh_server.render` retries a take whose peak sits under 0.005 with up to two `retry_seed` seeds; with a shape it measures that peak after the chain's 30 Hz high-pass, the natural trim and the low-pass. One ball take had a raw peak of 0.017 that came entirely from the end-of-window burst, which the trim's fade removes, and the 400 Hz low-pass left a clip at -81 LUFS that loudness levelling (correctly) refused to boost, so one bounce in six would have been silent. The worker's request path is now `render_key` and `finish_key`, which `world_render.py` calls too; without a physics key both reduce to the old calls.

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
- `tests/test_woosh_server.py` and `tests/test_level.py` (3 tests added with the GPU checks): a physics take counts as silent by what the chain keeps (a bright tone under the soft low-pass, a burst inside the trim's fade) and gets retried while a HEAD take is judged as before; `render_key` and `finish_key` reproduce the old request path without phys and hand the shape over with it; a soft-contact clip whose attack the low-pass removes still lands within 0.67 to 1.5 times its t40 target at peak and at loudness level.
- Roslyn: the package, the demo and the editor assemblies compile with Unity 2022.3.8f1's compiler; a Mono run of `SargePhysics` and `SargeClient` produced valid JSON that the Python server buckets.

## GPU checks run

`scripts/bench/world_render.py` imports the Woosh worker and renders requests through the worker's code: `words` and `explore` render minimal pairs with `woosh_server.render` at one seed per pair, `demo` renders the demo pairs through `render_key` (the request path of the worker, retries included), and `analyse`, `routes` and `check-served` work on the saved raws on the CPU. HEAD variant 0 of the ball on metal, finished this way, matches the clip the 10-16 session served bit for bit (`check-served`, max abs 0.0 at 16 bits), so the harness hears what the game heard. Everything ran on Woosh DFlow at the adopted settings, one GPU job at a time; outputs sit in `data/_gate/feel/world/`.

### Word following

Each cell renders two prompts that differ only in one word group, at the same six seeds, and measures the raw render (the final 20 ms dropped and the chain's 30 Hz high-pass applied, nothing else) and the clip after the full `--world` chain. The physical direction: a harder contact gives a brighter first 50 ms (spectral centroid), a longer ring gives a longer peak-to-minus-40-dB time, a larger body gives a larger share of energy between 20 and 300 Hz, a hollow body rings longer, and an inflated ball sounds duller than a solid one. A cell passes when at least 5 of 6 seed pairs move that way on the raw render.

Three runs used seeds 0 to 5. v1 is the first wording. Before v2, `world_render.py explore` tried two alternative tables per group and the order of the template on seeds 6 to 11, which the retest never sees; v2 puts the contact and ring words right after the head and uses the ring wording that did best there. The shipped run checks the template that ships, on fresh seeds 12 to 17, after the groups that failed twice lost their words (their pairs are then the same prompt, so they show 0 of 6 by construction).

| cell | v1 raw | v2 raw | shipped raw | shipped after chain |
|---|---|---|---|---|
| contact soft to sharp, ball on metal | 4/6 (479 to 1225 Hz) | 6/6 (608 to 1323) | 6/6 (170 to 680) | 6/6 |
| contact soft to sharp, bottle on gravel | 6/6 (712 to 1582) | 6/6 (570 to 1868) | 4/6 (1676 to 2210) | 6/6 |
| contact soft to sharp, crate on barrel | 6/6 (369 to 661) | 4/6 (383 to 397) | 6/6 (303 to 1538) | 6/6 |
| contact dull to crisp, bottle on gravel | 6/6 (724 to 1639) | 5/6 (1224 to 1423) | 4/6 (1237 to 1706) | 4/6 |
| contact dull to crisp, crate on gravel | 4/6 (831 to 1124) | 5/6 (392 to 436) | 2/6 (474 to 262) | 2/6 |
| ring dead to short, ball | 3/6 (0.54 to 0.58 s) | 4/6 (0.54 to 0.55) | dropped | 6/6 (0.08 to 0.19) |
| ring short to ringing, ball | 1/6 (0.58 to 0.23) | 3/6 (0.55 to 0.45) | dropped | 6/6 (0.19 to 0.45) |
| ring ringing to long, ball | 5/6 (0.23 to 0.64) | 5/6 (0.45 to 0.81) | dropped | 2/6 |
| ring dead to short, bottle | 1/6 (0.25 to 0.14) | 3/6 (0.21 to 0.20) | dropped | 2/6 |
| ring short to ringing, bottle | 3/6 (0.14 to 0.14) | 5/6 (0.20 to 0.27) | dropped | 4/6 |
| ring ringing to long, bottle | 4/6 (0.14 to 0.19) | 2/6 (0.27 to 0.26) | dropped | 1/6 |
| size small to large, ball | 3/6 (0.31 to 0.25) | 3/6 (0.74 to 0.37) | dropped | 0/6 |
| size small to large, barrel | 3/6 (0.76 to 0.89) | 3/6 (0.84 to 0.91) | dropped | 0/6 |
| size small to large, bottle | 5/6 (0.11 to 0.19) | 4/6 (0.06 to 0.09) | dropped | 0/6 |
| build solid to hollow, barrel | 1/6 (0.75 to 0.58 s) | 0/6 (0.51 to 0.48) | dropped | 0/6 |
| build solid to hollow, bottle | 5/6 (0.12 to 0.14) | 4/6 (0.24 to 0.27) | dropped | 0/6 |
| build solid to hollow, crate | 2/6 (0.26 to 0.27) | 1/6 (0.84 to 0.14) | dropped | 0/6 |
| build inflated to solid, ball | 1/6 (479 to 375 Hz) | dropped | dropped | 0/6 |
| groups passing (cells) | contact 3/5, ring 1/6, size 1/3, build 1/4 | contact 4/5, ring 2/6, size 0/3, build 0/4 | contact 2/5 | |

Numbers in brackets are the medians of the two members. The model follows the contact words and little else. Soft against sharp moved the attack the right way in 79 of 90 seed pairs over the five runs that held the current contact words (v1, the two exploration baselines, v2 and the shipped template); dull against crisp did so in 26 of 30 on the bottle and 18 of 30 on the crate, so the middle two classes blur on wood. Ring, size and build failed in v1 and again after rewording, and "inflated" made the ball brighter in 29 of 30 pairs, the opposite of an inflated ball. No alternative wording passed its group on the exploration seeds either (`explore_x1.log`, seeds 6 to 11, both word orders). The best size wording, "big heavy", moved 2 of the 3 size cells each time: the bottle missed with 4 of 6 under the late order and the ball with 3 of 6 under the early one. "Hollow echoing" moved 1 of its 3 hollow cells under each order (barrel 3 of 6 both times, bottle 5 then 4, crate 3 then 5). So the data holds no wording that makes the model follow size or hollowness, and these two would also have claimed a mass and a room echo that the bucket does not carry. With those words gone the prompts are shorter and the chain alone orders the ball's decay: dead under short under ringing in 6 of 6 seeds each.

### Decay targets after the full chain

`world_render.py demo` rendered every demo pair under `--world` semantics, `analyse` finished variants 0 to 3 through the `--world --amp --stereo` chain (peak level), and the table gives t40 per take against the target of the pair's ring class.

| pair | ring | target (s) | t40 of takes 0 to 3 (s) | in 0.67 to 1.5x | cap acted |
|---|---|---|---|---|---|
| ball on metal | short | 0.20 | 0.25 0.20 0.20 0.20 | 4/4 | 3 |
| ball on stone | dead | 0.08 | 0.08 0.08 0.07 0.07 | 4/4 | 4 |
| ball on wood | dead | 0.08 | 0.06 0.07 0.06 0.07 | 4/4 | 4 |
| ball on grass | dead | 0.08 | 0.05 0.06 0.03 0.03 | 2/4 | 2 |
| ball on glass | dead | 0.08 | 0.08 0.07 0.10 0.07 | 4/4 | 3 |
| bottle on glass | long | 1.50 | 0.54 0.12 0.07 0.33 | 0/4 | 0 |
| bottle on gravel | ringing | 0.60 | 0.26 0.05 0.13 0.59 | 1/4 | 0 |
| barrel on gravel | ringing | 0.60 | 0.16 0.30 0.03 0.58 | 1/4 | 1 |
| crate on gravel | dead | 0.08 | 0.07 0.08 0.08 0.08 | 4/4 | 3 |
| crate on barrel | long | 1.50 | 0.00 0.16 0.84 0.78 | 0/4 | 0 |
| window break | long | 1.50 | 0.59 0.68 0.14 0.64 | 0/4 | 0 |

Dead lands in range on 18 of 20 clips and short on 4 of 4, ringing on 2 of 8: 24 of 32 over the three classes, 75%, under the 80% bar. Where the cap acted it landed in range on 19 of 20 clips. Every miss is a take the model made shorter than its target, which the cap by design leaves alone: two dead grass bounces at 0.03 s, six ringing props between 0.03 and 0.30 s, and all twelve long clips, whose 1.5 s target sits outside a 1.0 s render anyway. The ring words were the only lever that could have lengthened those takes, and they failed above. Before the cap moved to the end of the chain the same renders gave 15 of 20 dead clips in range; that is the change described under "From the bucket to post-chain numbers".

The renders also show the cost of the stricter silent-take check. Leaving out each pair's first request (a prompt-cache miss), HEAD retried 4 of 55 takes and the physics keys 17 of 55, each retry adding about 28 ms to a first hit. Three of the 66 physics takes stayed under the bar after both retries (kept peaks of 0.0010 for a ball on grass, 0.0026 for the barrel on gravel and 0.0045 for the crate on the barrel, all variants 4 and 5, so none of them is on the page), and peak levelling lifts such a take by 46 to 60 dB.

### Identity

Of the demo classes only the window break has validated identity judges in `scripts/bench/vocab_eval.py` (`glass_break`); nothing there covers impacts on metal, balls, bottles, barrels or crates, so those rest on the page. `world_render.py routes` cut variants 0 to 19 of the window break, finished as served, to the judges' 0.6 s window for HEAD and `--world`, and `vocab_eval.py --refs data/refs_heldout --settings none` scored them as two routes (`scripts/bench/out_vocab_eval_refs_heldout_nobase_world_head_world_phys.json`, log in `data/_gate/feel/world/vocab.log`):

| judge (held-out real rate) | HEAD | `--world` | paired on variant index |
|---|---|---|---|
| AST, shatter labels (84%) | 12/20, 0.77 of real | 11/20, 0.71 | 6 HEAD-only, 5 world-only, p 1.0 |
| CLAP, any glass phrase (97%) | 12/20, 0.66 | 19/20, 1.04 | 1 against 8, p 0.04 |
| PANNs, glass labels (98%) | 15/20, 0.81 | 20/20, 1.07 | 0 against 5, p 0.06 |
| consensus (min over judges) | 0.66 [0.41, 0.87] | 0.71 [0.43, 1.00] | |
| Audiobox PQ (real 6.27) | 6.88 | 6.92 | |

No judge drops: AST moves by one clip, and CLAP and PANNs both hear more breaking glass in the physics prompt. CLAP's free top-1 phrase says the same thing in words: "a window breaking" leads for `--world` (7 of 20), and "swords clashing" leads for HEAD (9 of 20). The two arms draw different seeds, since the seed hashes the physics key, so the pairing is by variant index.

### The ball on the metal tile

All 68 logged hits on the metal tile in the 10-16 session (12.2 to 25.3 m/s) bucket to one key, `bounce;a=rubber,large,inflated,ball;b=metal,huge,static,floor;c=soft;r=short,metal`, at hard, so speed changes nothing in the request and the game rotates the same takes. Over variants 0 to 5, finished the way each flag set serves them:

| | M400 (median, takes 0 to 5) | t40 | length | PLR | crest |
|---|---|---|---|---|---|
| HEAD (`--amp --stereo`) | -12.0 LUFS (-4.7, -13.7, -9.9, -12.8, -14.7, -11.1) | 0.96 s | 0.98 s | 11.5 dB | 18.3 dB |
| `--world` | -19.0 LUFS (-19.0 to -17.1, two at -21.3) | 0.20 s | 0.37 s | 18.5 dB | 20.2 dB |
| `--world --level loudness` | -18.3 LUFS (-20.6 to -16.2) | 0.20 s | 0.32 s | 17.3 dB | 18.3 dB |

The served clip was the loudest of the six HEAD takes. Under `--world` the ball lands 7 dB under the HEAD median and 14 dB under the served clip, and its tail ends 0.2 s after the peak where HEAD rang to the end of the 1 s window in five takes of six. By these numbers "a loud sound that lasts way too long" is fixed. The loudness flag adds about a dB, because the soft-contact clip is too sparse to reach the -14 LUFS hard target inside the 6 dB limiter budget. In that session the bounce itself kept coming: the demo balls gained energy on every bounce and hit hard every 3.9 s (median of the logged intervals).

### The demo balls now lose energy

The energy gain came from the scene. `D:\PSfxUnityTest\Assets\WalkArena.unity` gives the four `AutoCaptureBall` spheres and the grass, metal, stone and wood tiles one shared PhysicMaterial, `psfx_bouncy`, with bounciness 1, Maximum bounce combine and zero friction. A restitution of exactly 1 leaves no loss to offset the small separating push PhysX gives two touching bodies, which is the likely source of the gain: the logged hits climbed from 12 to 25 m/s over the session. The material's bounciness is now 0.8, about the restitution of a real rubber ball; the combine mode and the zero friction stay. A ball dropped from the 8 m start height lands at about 12.5 m/s and then at 0.8 of the previous speed each time: about ten hits over 9 s, four hard (6 m/s and up, the AutoCapture threshold) and six normal, until the speed falls under Unity's 2 m/s bounce threshold and it comes to rest. K (`SargeImpulseProbe`, already on each ball) launches them again at 3 to 14 m/s, so the arena still has ball hits on demand, and each run of bounces now steps from hard down to normal the way a game's would. Props that land on those four tiles bounce at 0.8 for the same reason. Nothing in the physics bucket moves: the ball still reads as rubber (0.6 and up) and as a bounce (0.5 and up), and all 24 ball and bottle contacts checked on the four tiles at 2 to 18.8 m/s give the same bucket at 0.8 as at 1.0. The scene was copied to `D:\PSfxUnityTest_backup_20261016\WalkArena.unity` first; setting `bounciness` back to 1 restores the old behaviour. The editor script `Assets/Editor/PSfxDemoSetup.cs` still writes bounciness 1 if the scene is ever rebuilt from it.

### Flags off

The worker's request path changed shape (`render_key`, `finish_key`, the silent-take check), so the startup harness ran once more under Unity's environment with every flag off (`e2e_hybrid.py --tag world_q0_warm --extra --learn`, ports 8767 and 8772): all 75 (key, variant) clips match `data/_gate/startup/baseline_warm_1` at max abs 0.0 with equal shapes (`data/_gate/startup/compare_world_q0.json`). VRAM peaked at 10,236 MiB during startup, held 9,702 MiB at READY (50.3 s, warm) and 10,224 MiB while both models rendered, inside the 10.5, 9.6 and 11.3 GiB bounds. First hits matched the earlier runs: guns 150.7 ms median, Woosh 34.2 ms. After `--concurrent-gen` became the default (`docs/on-time.md`), the check ran again with every flag off (`--no-concurrent-gen`) and with the new defaults cold and warm: 75 of 75 clips identical each time (`data/_gate/startup/compare_ontime_q0.json`, `compare_ontime.json`). `--world` stays off by default.

## GPU checks still to run

1. **Blind page.** `data/_gate/feel/world_page/index.html` plays the eleven demo pairs through HEAD, `--world` and `--world --level loudness`, three takes each, letters shuffled per row, at the game's playback gain, with best, worst and "none sound right", Reveal (option names with loudness, decay and length per take) and Download. A replay of a recorded `writePhys` session through the same arms is still open; the page uses the demo colliders at their logged speeds.
2. **Play session with P.** The ball on metal should sound like a big rubber ball with a short tick of metal, at most about 0.3 s long. The renders above come out 0.32 to 0.37 s long with the tail 0.2 s after the peak. With the scene change above, each drop or K kick now gives a run of hits that steps from hard to normal, so the session also hears normal-intensity ball takes, which the page does not hold (its ball rows are all hard).
3. **Short ringing props.** Bottles, barrels and crate-on-barrel hits render shorter than their ringing and long targets. Whether that matters is an ear question for the page; a longer latent for impacts (`--woosh-frames impact=501`) is the measured route to longer natural tails if it does.

## Risks

The model ignores the ring, size and build words (measured above), so decay lives entirely in the DSP, and a take the model renders short stays short. Physics keys retry silent takes about four times as often as HEAD keys, which adds about 28 ms to the first hit of the affected takes. A bouncy steel ball whose names carry no material word reads as rubber. Unity's default mass makes most props hollow, which matches the simulation and may surprise a designer. Slides and rolls play as hits in v1. With the demo balls losing energy, a ball now asks for two keys per surface (hard and normal) where it used to ask for one, so each surface pays one more first hit before its takes are cached. Each new prompt misses the Woosh prompt cache once (about 18 ms).
