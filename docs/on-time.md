# On-time playback

The "slightly late, then it catches up" effect has two sources. The first play of a sound the engine has never made waits for generation: Woosh takes 33 to 52 ms, guns 148 to 220 ms, and 210 to 490 ms after the GPU has idled. In the 17:40 session 42 of 56 keys played late that way. Every play, cached or not, also pays about one frame of polling plus the audio ring buffer, about 60 to 100 ms at Unity's default 1024-sample buffer. The changes below attack both. After the GPU checks further down, `--concurrent-gen` is on by default (`--no-concurrent-gen` turns it off); every other flag and every Unity switch is still off by default.

## Server flags (`scripts/run_sidecar.py`)

| flag | effect |
|---|---|
| `--event-log PATH` | one JSON line per EVENT (`t_recv`, key, served, variant, `await_ms`, `lock_wait_ms`, `engine_ms`, `gen_ms` as the Woosh worker measured it for the clip that was served, null for a gun or a disk-cache hit, `encode_ms`, `send_ms`, bytes, `meta_only`, connection), per PREDICT (keys, eta, queued, lead gated) and per background generation (`deepen` with a `gen_ms` per item, `keep_warm`); a failed write turns the log off and leaves every connection open |
| `--predict-fixes` | (a) a predicted event the game never fired maps to the most recently fired event whose name ends in the same stem as a separate word with an optional s, es, ed or ing, so the motion predictor's `footstep` becomes `iron boot footstep` and `land` becomes `iron boot landing` while the boots are on (`land` never becomes `landmine`, and an empty name maps to nothing); (b) a prediction without physics takes the last fired intensity of that event; (c) realtime mode drops the lead gate, so a late prediction still starts generation early; (d) the background loop skips its 50 ms rest while predicted keys wait; (e) a prediction with `after` expands to the events the habit model has seen follow that event, keeping only events whose material differs from the trigger's (`after: "shoot"` with a wood target predicts `bullet impact|wood` and skips `reload`) |
| `--pool-fix` | the background loop keeps a predicted key queued until every variant is planned (HEAD planned one), choosing only variant ids that are free (absent from the cache and from the in-flight set), so a realtime render of variant 0 is never repeated |
| `--concurrent-gen` (on by default, `--no-concurrent-gen` turns it off) | a realtime miss claims its key, generates outside the engine lock and commits under it, so cached keys answer while a gun renders; a duplicate request waits for the first through the existing in-flight wait; the habit model and the alias table record the event when it is claimed, so a slow gun miss still counts before the bullet impact that follows it. While any claim is in flight, no prediction queues a claimed key and the background loop (and `--keep-warm-ms`) starts no new render; the loop wakes when the claim lands. That keeps the background work what the single-lock path does, where a realtime render holds the engine lock and the loop has to wait for it to plan |
| `--async-cache-write` (on by default, `--no-async-cache-write` turns it off) | the 16-bit copy of each new clip goes to the disk cache on a background thread, so a reply never waits for the disk; a read of a clip whose write is still pending waits for it and then reads the file, so every served clip keeps its bits |
| `--keep-warm-ms N` | while a client is connected and nothing has generated for N ms, run one warm-up render (never cached); build-only, use it only if the event log shows the idle penalty during play |
| `--sibling-on-miss` | on a realtime miss, play the nearest cached key of the same event (same material first) at once and generate the exact key in the background; trades one slightly-off first play per key for no wait. A sibling must carry the same physics key, so with `--world` a ball on glass never borrows the ball on metal (with `--world` off every key has an empty physics key and any material can stand in) |

The EVENT message also accepts `have` (variant ids the client already holds) and `have_phys` (the physics key of the held clips, empty when left out). When the reply is `cached` or `realtime`, the served variant is one of them and the physics key matches, the reply carries `meta_only: true` and no PCM. A `sibling`, `fallback` or `silence` reply always carries PCM. Old clients never send `have`. Unity plays those three kinds once and never stores them under the requested key, so the exact render replaces them on the next hit.

## Unity switches

`SargeService` (Inspector fields, all default off; `SargeService.Setup` is a static hook that runs at the start of `Awake`, for settings that must exist before the worker threads start):

| field | effect |
|---|---|
| `dspBufferSize` | 0 keeps the project setting; 256 or 512 applies through `AudioSettings.Reset` in `Awake` (this stops sounds already playing) and warns when the active buffer exceeds 512 |
| `logTimings` | CSV in `persistentDataPath`: request, dequeue, reply and play times (ms), `dspTime`, frame, served, lane, variant, samples, key, with the DSP buffer on the first line |
| `resendPredictions` | remembers predictions with absolute ETAs, one slot per source, event, material, context and collider pair (a persistent source keeps only its last set), and resends the unexpired ones, plus every `equip` one (`persistentPredictionSources`), on each reconnect; fixes the rifle prediction lost before the first connect |
| `failFastOffline` | a request while disconnected fails in the next frame, so the startup backlog of ball hits never forms; a clip the client already holds still plays (with `localPlayback`), and a disconnected connection retries every `reconnectDelayMs` |
| `localPlayback` | a key whose variants are already held plays inside `Request`, rotating them, and the EVENT still goes out with `have`; results are also drained in `LateUpdate`; physics keys map from collider pair and intensity, learned from replies and PREDICT acks |
| `eventConnections` | events spread over this many connections, least busy first, where busy counts queued items plus the one in flight |
| `controlConnection` | context, predictions and pings travel on a separate connection |

New package components, each opt-in:

- `SargeImpactPredictor`: every 0.1 s, and at once on any velocity jump of more than 1 m/s (a kick, a shot, a bounce), it steps the ballistic arc in 0.02 s segments over a 2.5 s horizon with a sphere cast of the collider, and at the first hit sends a PREDICT with the ETA and the AutoCapture intensity rules, plus the contact JSON when the AutoCapture has `sendPhysics` on (without it the prediction carries the plain key the hit will ask for). A surface the body slides or rolls on (in contact, moving along it slower than `minVelocity`) is left out of the cast and its gravity component is dropped, so a rolling ball never predicts a hit on the floor under it. It resends only when the target collider or the intensity changes, at most once per `minSendInterval` (0.05 s), and all predictors share `castsPerFrame` (1000) sphere casts per frame. With `--world` the server buckets it to the key the real hit will carry.
- `SargeAimPredictor`: while `armed`, it raycasts from the camera every 0.1 s; once the aimed material holds for 0.15 s it sends a PREDICT. Static `MaterialResolver`, `EventResolver` and `PhysResolver` let a game state what its hit code will fire. When `EventResolver` names the event, the prediction goes out as that literal event; otherwise it goes out with `after: "shoot"` for the server to expand. A prediction with physics carries the aimed collider as its signature, so local playback learns the key from the ack.

Demo keys: P toggles physics context, O adds and toggles both predictors (the aim resolvers mirror `SargeFirstPerson.Bullet`: windows map to `break` and, with P on, to the same physics block `SargeBreakable` sends), T toggles the on-time set (`localPlayback`, `resendPredictions`, `failFastOffline`, `logTimings` at once; DSP buffer 512, two event connections and a control connection from the next Play, stored in PlayerPrefs). The preset applies only in WalkArena. When it applies the buffer it records the buffer it replaced, and the first Play with T off (or in another scene) puts that buffer back, so a HEAD timing session after an on-time session starts from the project's 1024. The Unity scripts were backed up to `D:\PSfxUnityTest_backup_20261016` before the first edit.

## Checks run on CPU

- `tests/test_ontime.py` (32 tests; the last four came with the defaults change): a claimed key is never queued by a prediction and a released or landed claim leaves no trace; under `--concurrent-gen` a background render predicted during a 200 ms miss starts only when the miss lands, and variant 1 of the missed key itself is never rendered beside it (with the claim set disabled, which is the earlier code, the miss's variant 1 renders beside it from 3 ms in, and the next test fails too); a nine-event session with the background loop on renders the same (key, variant) set with and without `--concurrent-gen`; `--async-cache-write` returns a miss in under 60 ms while a 120 ms disk write runs (inline: over 100 ms) and serves the same bits on the first play, on a read that has to wait for the pending write, and in a batch. `tests/test_level.py` checks that `run_sidecar.py` passes `concurrent_gen=True` by default and nothing else from the on-time set, that `--no-concurrent-gen` passes none of it, and that `--async-cache-write` reaches `serve`. The 28 earlier tests: with `--concurrent-gen` a cached key answers in under 60 ms while a 150 ms miss generates (HEAD waits over 80 ms), clips, stats and the in-flight set match a serial run, three duplicate requests generate once, a failed render releases its claim; the event log writes one line per request with every field; nothing changes by default; `have` gives a meta-only reply and a different `have_phys` still gets PCM; the alias maps `footstep` to `iron boot footstep` and `land` to `iron boot landing` with the last fired intensity, and takes predictions literally without the flag; the lead gate turns off in realtime mode; `after: "shoot"` expands to `bullet impact` on the aimed material and skips weapon-bound events; `--pool-fix` plans all three variants where HEAD planned one; the background loop finishes six predicted keys in under 0.6 of HEAD's time; sibling on miss and keep-warm behave as described and stay off by default, and a sibling wins over a concurrent claim when both flags are on. Added after review: with `--world` a prediction and a habit guess without physics take the last fired bucket and no plain key gets queued, while `--world` off stays plain; a sibling needs the same physics key and always carries PCM; the alias needs the stem at the end of the name; a failed log write keeps the connection; each logged `gen_ms` matches the clip it is logged with while two renders overlap; a slow concurrent miss is recorded before the event that arrives after it; `--pool-fix` never plans a variant that is cached or in flight.
- World prediction replay (scratch `review/worldpred_fix.py`, `--world --predict-fixes --concurrent-gen`, mock generator): with the demo's `PhysResolver` the first window break and the four after it are all served `cached` and no plain window key is ever generated; without a resolver the first break still misses (the server has seen no window yet) and every later prediction and habit guess lands on the physics key.
- Mono harness (scratch `wpb/mono`, the real `SargeService.cs` and `SargeClient.cs` on Unity stubs against the Python server): HEAD settings keep the HEAD clip name and play on the next update; with `localPlayback` a held key plays inside `Request` (0 frames), the server reply never plays twice, a learned physics key plays the next hit at once and an unseen collider pair waits; one lane queues a cached key behind a 400 ms miss (411 ms) while two lanes plus a control lane and `--concurrent-gen` play it at 32 ms; `failFastOffline` fails at 0 ms against 2128 ms; `resendPredictions` delivers an equip prediction sent while the server was down and drops an expired motion one.
- Ballistic ETA against 268 logged bounce pairs from the probe CSVs: median error 20 ms (p90 20 ms) with the analytic arc.
- Coverage replay of the 17:40 session (scratch `wpb/coverage_replay.py`, conservative rules): of 43 keys late at HEAD, 29 are covered by the predictors and fixes (all 8 ball impacts, the rifle shot, the window, 6 bullet impacts, 10 of 11 iron boot steps, 2 prop hits with a known lead); 40 play at once when `--sibling-on-miss` is added. The rest are two equips (key presses), the first iron boot step and prop hits after short flights.

## GPU checks run

`scripts/bench/ontime_bench.py run` launches the real sidecar under Unity's environment (`HF_HOME=F:\AI\cache\hf`, `HF_HUB_OFFLINE=1`, no `HF_HUB_CACHE`, `TRANSFORMERS_CACHE` or `CUDA_VISIBLE_DEVICES`) on ports 8767 and 8772 with `--generator hybrid --realtime --stereo --amp --detailed-prompts --learn`, a separate cache folder and `--event-log` in every arm. Its client copies the T preset: two event lanes picked least busy first, plus a control lane. Phase a fires head-of-line trials with no game load: a gun first hit on lane 1 and, 30 ms later, a Woosh request on lane 2 that is either cached or a fresh key, plus a fresh Woosh key alone, eight of each. Phase b plays a scripted 58 s session (110 events, predictions included) beside a 60 Hz game-like GPU load (three 2048² matmuls a frame, about 40% busy). Phase c measures the first render after idle with no load, a light 30 Hz load and the 60 Hz load. A reference arm renders variants 0 and 1 of all 99 keys under HEAD code. Every arm ran twice. Outputs are in `data/_gate/ontime/`: `s1` (the first round, each flag alone), `s2diag` (six instrumented runs), `s2` (the defaults as shipped against HEAD) and `s2c` (phase c again after the bench fix below); `report --set NAME` writes each set's `report.json`.

### Gates V and L

VRAM (whole card, limits 10,752, 9,830 and 11,571 MiB) passed in every run of every arm: startup peaks 9,862 to 10,248 MiB, 9,702 to 9,714 at READY, 10,222 to 10,224 while a gun and a Woosh sound rendered together, and at most 10,417 with the synthetic game load on the card. Gate L compares each served clip by the sha1 of its float32 bytes and its shape, per (key, variant), against the HEAD and reference runs. In `s1` all of them matched: 58 of 58 clips for `--concurrent-gen` (159 replies per run), 68 of 68 with `--predict-fixes --pool-fix` added, including the variant-1 clips pool-fix produces, 34 of 34 for `--predict-fixes --pool-fix` alone and 36 of 36 for each keep-warm arm; the two HEAD runs never disagreed with the reference, and no reply came from the 16-bit disk cache.

### a. Head-of-line, no game load (`s1`, 16 trials per arm, reply latency in ms)

| | HEAD | `--concurrent-gen` | with `--predict-fixes --pool-fix` |
|---|---|---|---|
| cached Woosh on lane 2 while a gun first hit renders on lane 1, p50 / p95 / max | 258 / 320 / 321 | 0.53 / 0.65 / 0.68 | 0.57 / 0.66 / 0.67 |
| fresh Woosh behind a gun miss, p50 / p95 | 169 / 231 | 93 / 221 | 171 / 275 |
| that render's `gen_ms` in the worker, p50 / p95 | 33 / 91 (it runs after the gun) | 75 / 199 | 75 / 190 |
| gun reply in the same trial, p50 | 155 | 188 | 207 |
| fresh Woosh alone, p50 / p95 / max | 52 / 208 / 496 | 51 / 89 / 90 | 51 / 156 / 167 |

A gun and a Woosh sound that render at the same time share the GPU: Woosh goes from 33 to 75 ms and the gun from 155 to 188 ms. The 171 ms fresh median of the third arm is the subject of "What held some first hits" below; it has nothing to do with the prediction flags, which phase a never exercises.

### b. Scripted session (`s1`, 110 events, 44 first plays per arm over two runs)

| | HEAD | `--predict-fixes --pool-fix` | `--concurrent-gen` | all three |
|---|---|---|---|---|
| served on demand | 6.4% | 7.3% | 6.4% | 7.3% |
| reply p50 / p95 (ms) | 0.71 / 82.5 | 0.68 / 106.9 | 0.71 / 79.1 | 0.67 / 88.8 |
| first plays over 20 ms | 31.8% | 45.5% | 31.8% | 36.4% |
| shots over 20 ms | 5% | 5% | 0 | 0 |
| prediction renders (n / GPU s) | 21 / 2.72 | 39 / 5.15 | 24 / 3.22 | 40 / 5.38 |
| wasted (n / GPU s / share of session) | 6 / 0.65 / 1.1% | 13 / 1.83 / 3.2% | 6 / 0.62 / 1.1% | 13 / 1.83 / 3.2% |
| total GPU time against HEAD | | +69% | +14% | +75% |

Both runs of every arm gave the same on-demand counts. Equips and the first window break play on demand in every arm (about 250 and 80 to 100 ms). Under `--concurrent-gen` the one shot that waited 82 ms behind a window break replies in about 1 ms.

`--predict-fixes` and `--pool-fix` made the session later for two reasons. `_fixed` in `src/sarge/sidecar/server.py` (`inten = ... server.last_intensity.get(et, k.intensity)`) replaces the intensity the impact predictor sent, so the 580 ms-lead `glass bottle hit|glass|normal` became `hard`, already cached, and the normal hit rendered on demand; it should only fill an intensity the prediction leaves out. And pool-fix puts a key back at the front of the queue (`self.demand.insert(0, key)` in `plan_batch`), so every variant of the equip predictions rendered before the first footstep prediction (250 ms lead), and that step went on demand at 161 to 174 ms. The alias part works: the iron boot step on metal was covered. Both flags stay off.

### What held some first hits

In one of the four `--concurrent-gen` runs of round one (`fix_1`), 7 of 9 Woosh first hits that overlapped a gun ended within 1 ms of the gun, 106 to 120 ms after the worker had replied. That run looked like a defect of the concurrent path, so six instrumented runs (`s2diag`, the round-one code, three of each arm) added a lock wrapper that logs every wait over 5 ms with the previous holder, a probe thread that wakes every 3 ms and arms `faulthandler` to dump every stack if the GIL stays taken for 45 ms, `gc.callbacks`, and timings for the worker round trip and for the disk-cache write. None of the six ever held the GIL for more than 12 ms after READY, and no garbage collection took over 2 ms. Every Woosh first hit that spent more than 10 ms outside the time the worker measured did so for one of two reasons:

- **The 16-bit disk-cache write.** `CachedGenerator.generate` writes the wav before it returns the clip, and in these runs that write took 13 to 45 ms where it normally takes 0.8 ms. The extra time outside the worker matched the write to within 1.5 ms every time (45.9 against a 45.1 ms write, 17.3 against 16.6, 14.2 against 13.2).
- **A wait for the Woosh lock** behind a background render of a different key that started first: once per run, the bottle hit 30.8 s into the session waited 31 to 41 ms for a predicted ball render.

The slow writes come in spells that last a whole run and differ between runs, which is the shape of `fix_1`: one run with a near-constant 106 to 120 ms, its twin `fix_2` at 2 ms. HEAD has the same spells. In round one the HEAD-path reference run spent more than 20 ms outside the worker on 27 of 48 Woosh first hits (median 33 ms, max 524), `head_1` had one 445 ms first hit with nothing else running, and one keep-warm run had a 31 ms median; the round-one diagnostics timed 13 of 81 writes at 46 to 102 ms. During a spell the gun's reply waits on its write too (the held trials of `fix_1` ran 35 to 121 ms longer on the gun than the same trials of `fix_2`), and the two replies ending within a millisecond fits both writes waiting out the same stall; `fix_1` had no write timing, so that last step is inferred. So the hold is the reply path's disk write, which every flag setting shares, and the concurrent path holds nothing back.

The instrumented runs also showed what the concurrent path did add: every gun or Woosh claim queued the claimed key itself through the habit model, so variant 1 rendered beside variant 0 (50 of 80 misses in round one, +14% GPU), and the background loop could start a render while a claim was still on the GPU, which is how a first hit lost the Woosh lock three times by 38 to 118 ms. Two changes, active only while a claim is in flight (so only under `--concurrent-gen`), give the concurrent path the background work of the single-lock path: a prediction never queues a claimed key, and the background loop and keep-warm start nothing until every claim has landed. `--async-cache-write` takes the disk write off the reply path.

### The defaults as shipped (`s2`)

`s2` ran the shipped code three ways, twice each and interleaved: HEAD (`--no-concurrent-gen`), the new default, and the new default with `--async-cache-write`. Gates held in all six runs: startup peaks 10,031 to 10,259 MiB, 9,721 to 9,725 at READY, 10,243 to 10,247 while both models rendered, and 55 of 55 clips in every run identical to the reference by sha1 and shape (159 replies per run).

| head-of-line, 16 trials per arm (ms) | HEAD | default | default + `--async-cache-write` |
|---|---|---|---|
| cached Woosh behind a gun first hit, p50 / p95 / max | 245 / 327 / 414 | 0.57 / 0.67 / 0.70 | 0.57 / 0.73 / 0.80 |
| fresh Woosh behind a gun first hit, p50 / p95 | 164 / 230 | 85 / 194 | 100 / 195 |
| the gun in those trials, p50 | 153 | 188 | 191 |
| fresh Woosh alone, p50 / p95 / max | 51 / 172 / 179 | 41 / 52 / 52 | 34 / 51 / 52 |

| scripted session, 110 events, two runs per arm | HEAD | default | default + `--async-cache-write` |
|---|---|---|---|
| served on demand | 6.4% | 6.4% | 6.4% |
| reply p95 (ms) | 81.6 | 79.7 | 79.4 |
| first plays over 20 ms (of 44) | 34.1% | 34.1% | 31.8% |
| prediction renders, n / GPU s | 21 / 2.68 and 3.05 | 21 / 2.62 and 2.76 | 21 / 2.54 and 2.56 |
| wasted share of the session | 1.1% | 1.1% | 1.1% |
| background renders of the missed key while the miss rendered | 0 | 0 (round one: 25 per run) | 0 |
| Woosh first hits with over 20 ms outside the worker, per run | 1 and 5 (max 132 ms) | 1 and 2 (max 59) | 1 and 1 (max 31) |
| median time outside the worker | 1.4 to 1.5 ms | 1.4 to 1.5 ms | 0.6 to 0.65 ms |

The default now does what round one promised and nothing more. A cached sound no longer waits behind a gun miss (245 ms to 0.6 ms at the median), a fresh Woosh sound behind a gun miss arrives about 80 ms sooner, and the background work is HEAD's: the same 21 prediction renders, the same wasted share, no variant 1 rendered beside the miss of the same key, and total GPU time back to HEAD's where round one spent 14% more. The price is GPU sharing. A gun first hit that overlaps a Woosh render finishes about 35 ms later (153 to 188 ms), because the two renders now run side by side where HEAD ran them one after the other. In the session every arm played the equips, the first window break and the first iron boot steps on demand, and the other late replies were the same kinds in every arm: the one bottle first hit that renders on demand (102 to 113 ms) and the odd request that arrives while the prediction of the same key is still rendering and waits for it (HEAD: a shot at 83 ms and a bottle at 108 ms; the default: one ball at 22 ms).

`--async-cache-write` removed the disk write from every reply: the median time a Woosh first hit spent outside the worker fell from about 1.5 to 0.6 ms, and the one remaining wait over 20 ms per run is the bottle hit queued behind a predicted ball render, which HEAD has too. Its 100 ms fresh median against 85 ms is the worker's render time (99 against 75 ms of `gen_ms`), which depends on how much of the gun render a trial overlapped; the time outside the worker went down.

### c. First render after idle

The round-one table for this phase was mislabelled, and `run_idle` in the bench is fixed. Its first trial of each condition followed the 12 s idle-power window, so each "1 s" Woosh cell with keep-warm off mixed two trials after 15 to 19 s of idle with two true short-idle ones (no load: 299 and 300 ms at P8 against 35 and 34 ms at P2; light load: 208 and 240 ms at P8 and P5 against 38 and 38 ms), and its median said 167 and 123 ms where the short-idle trials said about 35 and 38; and since the gap only started after `quiesce` had waited 0.5 s, every keep-warm-off label was 0.5 s short. The bench now renders one throwaway Woosh key and one gun key after the window, measures each gap from the previous reply (`since_reply_s`, exactly 1.00, 3.00 and 8.00 s in every HEAD trial), and reports that and the idle since the last GPU work from the event log beside each cell. `s2c` ran the three arms again, twice each, with `--async-cache-write` in all of them so disk-write spells stay out of the clock measurement; four trials per cell, medians in ms, Woosh / gun:

| condition | idle power, keep-warm share of the GPU | after 1 s | after 3 s | after 8 s |
|---|---|---|---|---|
| no load, keep-warm off | 24.5 to 26.3 W | 42 / 261 | 237 / 360 | 275 / 418 |
| no load, keep-warm 200 | 48.1 to 48.2 W, 28% | 78 / 166 | 56 / 194 | 59 / 182 |
| no load, keep-warm 50 | 91.4 to 92.4 W, 50% | 51 / 162 | 55 / 159 | 56 / 164 |
| light 30 Hz load, off | 34.5 to 34.9 W | 36 / 272 | 121 / 362 | 258 / 341 |
| light 30 Hz load, keep-warm 200 | 63.1 to 63.8 W, 27% | 68 / 174 | 41 / 189 | 42 / 175 |
| light 30 Hz load, keep-warm 50 | 101.8 to 102.0 W, 46% | 36 / 161 | 34 / 192 | 36 / 164 |
| 60 Hz game-like load, off | 105.3 to 105.4 W | 42 / 223 | 48 / 226 | 49 / 229 |
| 60 Hz load, keep-warm 200 | 132.0 to 132.1 W, 30% | 45 / 228 | 43 / 222 | 44 / 226 |
| 60 Hz load, keep-warm 50 | 148.4 to 148.6 W, 59% | 89 / 228 | 44 / 225 | 49 / 272 |

With the labels right, the picture changes at the short end. After 1 s without load, Woosh is still fast with keep-warm off (42 ms at P2), and keep-warm 200 makes it slower (78 ms): the worker renders the request in 40 ms, and the rest is the wait for the Woosh lock behind a warm-up render already running (keep-warm 50 under the game load shows the same, 89 against 48 ms of render). The Woosh clocks drop between 1 and 3 s, and from 3 s on keep-warm helps: 237 and 275 ms become 56 and 59 at 200 ms, 55 and 56 at 50 ms. Guns are slow sooner, 261 ms already 1 s after a short Woosh render with no load (the memory clock falls during that render), and keep-warm brings them to 159 to 194 ms throughout. The light 30 Hz load does not hold the clocks either (P3 to P8 after 3 s), while the 60 Hz load holds P2 at about 1,780 MHz, and under it keep-warm buys nothing at 27 to 43 W more.

## Defaults and what is still yours to decide

- **On by default: `--concurrent-gen`** (`scripts/run_sidecar.py`, `--no-concurrent-gen` turns it off). It passed gates V and L in every run, and as shipped it changes only when replies go out: a cached sound behind a gun miss went from 245 ms to 0.6 ms at the median, a fresh Woosh sound behind a gun miss from 164 to 85 ms, and the background work is HEAD's. A gun first hit that overlaps a Woosh render costs about 35 ms. End to end under Unity's environment (`docs/hybrid.md`, "Concurrent misses on by default"), the cold and warm runs with the new default and the warm run with `--no-concurrent-gen` each served all 75 checked clips identical to `baseline_warm_1`. The scene needs no new argument. The gain needs separate lanes: with the T preset off, Unity sends everything down one connection and queues a cached sound behind the miss itself.
- **On by default since 2026-10-17: `--async-cache-write`** (`--no-async-cache-write` turns it off). It only moves the disk write: 75 of 75 clips identical end to end, 55 of 55 in both bench runs, and the time a Woosh first hit spends outside the worker fell from about 1.5 to 0.6 ms at the median, with the 13 to 120 ms write spells gone. A read of a clip whose write is still pending waits for it, and the sidecar clears the disk cache on every start, so a write cut short by a kill never gets served.
- **Off: `--keep-warm-ms`.** Without a game load it turns 237 to 275 ms Woosh and 360 to 418 ms gun renders after 3 to 8 s of idle into 55 to 59 and 159 to 194 ms, at +23 W (200 ms, 28% of the GPU) or +66 W (50 ms, 50%), and it makes a Woosh request 1 s after the last one slower (42 to 78 ms at 200). Under the 60 Hz load it buys nothing. If you want it, 200 is the better value; the real game's clocks during Play decide (log `nvidia-smi --query-gpu=clocks.sm,pstate,power.draw -lms 100` in one session).
- **Off: `--predict-fixes` and `--pool-fix`.** The intensity override and the front-of-queue requeue above made the session later; both are small fixes, and the pair is worth measuring again after them.
- **Off and never measured: `--sibling-on-miss`.** It trades one slightly-off first play per key for no wait; that is a taste call.
- **Yours: the T preset's DSP buffer**, 256 (about 35 ms cached event-to-ear, may crackle on a loaded machine) or 512.

## Still open

1. A Play session at HEAD settings and one with T, P and O, both with `logTimings`: at most 10% of events served `realtime`; cached Request-to-Play p50 at most 1 ms with `localPlayback`; cached event-to-ear about 35 ms with a 256 buffer. The bench copies the lanes of the T preset but not Unity's frame loop or audio buffer.
2. The real game's clocks during Play (`nvidia-smi --query-gpu=clocks.sm,pstate,power.draw -lms 100`), which decide whether `--keep-warm-ms` earns its power.
3. `--sibling-on-miss` was never measured on the GPU.

## Risks

`AudioSettings.Reset` stops sounds already playing, and a 256-sample buffer may crackle on a loaded machine. Overlapping GPU jobs under `--concurrent-gen` slow each other. Without `--async-cache-write` every first play still waits for its disk write, which comes in spells of 13 to 120 ms (once 524 ms) that no other setting removes. Prediction waste costs GPU time only, since nothing plays from a prediction. Aim prediction needs one observed shot per follow-up event under `after`. Local rotation depth depends on `--pool-fix`. With several event connections, two events sent in the same frame race to the server by microseconds, so the habit model can still see them swapped now and then; a slow miss no longer reorders them.
