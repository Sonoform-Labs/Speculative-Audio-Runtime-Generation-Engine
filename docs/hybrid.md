# Hybrid engine (run_sidecar.py --generator hybrid)

The hybrid sends every gun sound to SA3 medium inside the sidecar and everything else to Sony's Woosh in a second process. The split follows the ear tests: SA3 medium at its recommended settings was among the best for guns, and Woosh Flow was far ahead on glass, water and quality sounds; its distilled sibling DFlow runs by default for speed. Routing goes by sound family from `sarge.amp.family_for`. The `blast` and `handling` families (shots, reloads, empty clicks, equips) go to SA3 medium and the rest goes to Woosh (`src/sarge/generators/routed.py`).

## How a request travels

SA3 medium runs in the sidecar's `.venv214` with `HYBRID_GUNS` from `scripts/run_sidecar.py` on top of the compiled stack in `COMPILED`: pingpong sampler, 8 steps, cfg 1, 3.0 s generation, 1.0 s padding, detailed prompts from `src/sarge/prompts.py`, and the cross-attention keys and values of the differential layers hoisted once per prompt. The padding used to be 6.0 s. The noise is still drawn at the old 3 + 6 s shape and cropped, so every kept latent frame sees the same noise it saw before.

Woosh runs in `.venv_woosh` as `scripts/woosh_server.py` on port 8771. It speaks the model-server wire from `scripts/model_server.py` (an 8-byte length header, a JSON body, raw float32 PCM) and finishes each clip itself with the same natural trim, amplifier, stereo widening and peak level the SA3 path uses, so the sidecar passes its PCM straight through. DFlow at 4 steps is the default and `--woosh-model flow` swaps in the slower Flow model. Each request runs at the latent length of its sound family (`FRAMES` in `woosh_server.py`, 100 frames per second of audio), and every family stays at 101 frames (1.0 s) for the reasons measured below.

Each reply names its format. Guns arrive at 44.1 kHz and Woosh sounds at 48 kHz, both with two channels under `--stereo`. The welcome message still announces 44100 Hz and one channel for the whole server, which is harmless because the Unity client (`SargeClient.cs`, `SargeService.cs`) builds every AudioClip from the `sample_rate` and `channels` of each reply.

## Woosh clip length

The worker can run each sound family at a separate latent length. `woosh_server.py --frames` takes one count for every sound or `family=count` pairs over `FRAMES`: `impact`, `step`, and `other` for any event that `sarge.amp.family_for` leaves unnamed, which today means splash. `run_sidecar.py --woosh-frames` hands the same string to the worker. Each distinct count gets a separate CUDA graph, condition buffers and modulation constants, and `prepare` captures all of them before the worker prints its listening line, so a request only ever replays a graph. The table ships with 101 frames for all three families, which is the 1.0 s cut the Unity run showed on window breaks, glass, water steps and splashes.

That default is a measured result. `scripts/bench/woosh_buckets.py` ran the adopted DFlow path at 101, 151, 201, 301 and 501 frames, one fresh process per count, on 9 keys at normal and loud with 6 seeds each (108 clips per count, the prompts from `sarge.prompts.build`). A clip counts as truncated when its raw output is still above the -55 dB floor of `_natural`, measured with the same 10 ms envelope, inside the last 50 ms before the cut. The plan was to give each family the smallest count with at most 5% of its clips truncated, and no family reached that at any count:

| frames | ms per clip, cached / new prompt | finishing | impact truncated (36) | step truncated (60) | other truncated (12) | natural length, median: impact / step / other |
|---|---|---|---|---|---|---|
| 101 | 28.4 / 45.5 | 3.5 ms | 34 | 60 | 12 | 0.79 / 0.98 / 1.00 s |
| 151 | 33.0 / 50.1 | 5.1 ms | 33 | 59 | 12 | 1.33 / 1.44 / 1.50 s |
| 201 | 41.6 / 58.8 | 6.4 ms | 33 | 58 | 12 | 1.56 / 1.92 / 2.00 s |
| 301 | 49.6 / 66.7 | 9.8 ms | 34 | 60 | 12 | 2.75 / 2.97 / 3.00 s |
| 501 | 76.6 / 93.6 | 15.6 ms | 11 | 56 | 11 | 1.60 / 4.96 / 5.00 s |

Two separate things hold the trim open to the last sample. Below 501 frames, Woosh tends to close the window with a burst in its final 10 ms. At 101 frames the last 10 ms window sits at a median -14 dB under the clip peak, against -35 dB a fifth of a second earlier, 69% of clips rise by 10 dB or more there, and in 22 of 108 clips (16 of them footsteps) the burst is the loudest moment of the clip. The eager fp32 path from before the speed work has the same burst (-9.7 against -25.1 dB over its 48 clips), and so does Flow at 1.0 s, while the 5 s bake-off renders and the 501-frame sweep show none, so it belongs to running the model on windows shorter than 5 s. The second cause is sound that keeps going. With the final 20 ms set aside, 56 of 60 footsteps at 101 frames still sound above the floor near the cut, and 56 of 60 still do at 501 frames, where one footstep prompt comes back as 3 separate hits at the median and 10 at most. Splash keeps sloshing to the end of every window it gets.

Impact at 501 frames is the one cell where length clearly helps. Glass goes from 12 of 12 truncated to 0 of 12 (natural median 1.60 s, longest 4.60 s) and punch from 10 to 0 (0.67 s), while window break stays at 11 of 12 with a 4.4 s natural length, which leaves the family at 31%. That would cost about 52 ms more per impact clip as served (84.0 against 31.9 ms, finishing included) for a family that still misses the bar, so it stays opt-in as `--woosh-frames impact=501`. Capping long clips would change what the trim keeps on every sound, and that belongs to an ear test.

The burst itself is gone: `finish` drops the final 20 ms of every raw render (`END_CUT_S`) before the trim. Over the 108 clips at 101 frames, the finished clip's last 5 ms moved from a median -33 to -47 dB under its peak, clips ending within 6 dB of their peak went from 8 to 2, and 6 clips that had been levelled against the click came back 1 to 15 dB louder, while the first 200 ms of the median clip did not move. One of the two left, a loud metal footstep, was a silent take (peak 0.002 apart from the click), and about 1% of the sweep's renders look like it, so `render` now retries any take whose peak outside the cut is under 0.005 with up to two `retry_seed` seeds and keeps the loudest. Under Unity's environment (`e2e_endcut.json`) first hits, VRAM and READY time held, gun clips still match the bench bit for bit, and each Woosh clip is 20 ms shorter, or ends earlier where the burst had held the trim open.

Extra buckets are cheap. With `impact=501,step=201,other=301` the worker captured 201 frames in 0.94 s (+64 MB reserved, the first capture), 301 in 0.52 s (+56 MB) and 501 in 0.58 s (+78 MB), against 0.96 s and +58 MB for the single 101-frame graph, and the 3.47 GB load peak stayed where it was. Every bucket reproduced the single-count server at the same length bit for bit (108 of 108 clips at max abs 0.0, across processes and across the old and new server code), repeats inside a process matched to 0.0, a second sweep in fresh processes matched the first on all 540 clips, and the shipped table matched the old 101-frame server on all 108. A longer bucket draws its noise at that length, so its onset differs from the 101-frame clip of the same seed.

Through the sidecar under Unity's environment (`HF_HOME=F:\AI\cache\hf`, `HF_HUB_OFFLINE=1`, `e2e_hybrid.py --tag buckets --extra --learn`, saved as `data/_gate/opt/_e2e/e2e_buckets.json`) the shipped table behaves like the run before it. The Woosh first hit has a 34.5 ms median and a 52.2 ms p90 (34.6 and 52.7 before), the card holds 9.19 GB at READY with 3.06 GB of it in the Woosh worker, 22 of 22 clips match the bench bit for bit, and the same 12 of 31 generated Woosh clips still come back exactly 1.0 s long. READY took 94.4 s against 76.1 s, and all of the difference came before the Woosh launch: the worker was up 21.7 s after its start in both runs, with a 1.0 s warm-up. The same run with `--woosh-frames impact=501` (`e2e_buckets_i501.json`) reached READY in 82.1 s with a 1.5 s warm-up for its two graphs, held 9.29 GB at READY (the worker at 3.16 GB, 102 MB more), and moved the Woosh first hit to a 51.7 ms median and a 100.9 ms p90, with impact hits at 100 to 117 ms. Eight of its 31 clips ended at the cut; window break and glass dropped off that list, and a soft punch joined it at a full 5.0 s.

## Startup and shutdown

The sidecar now loads and warms SA3 medium alone, hands the cached load memory back to the driver (`torch.cuda.empty_cache`), starts Woosh and waits for its "listening" line, connects, warms the guns a second time, warms Woosh and prints READY (`RoutedGenerator(start_rest=...)`, passed by `run_sidecar.py`). The old order started Woosh first. The SA3 load alone peaks near 9.3 GB, and with Woosh's 3.1 GB already resident the card reached 11.7 GB of its 12 GB during startup, and READY took 98.9 s. Guns first peaks at 10.0 GB and reaches READY in 82.7 s. Through the sidecar, the 22 clips checked against the bench arrays came out bit-identical in both orders (max abs 0.0), so the reorder is a G0 change.

The second gun warm-up also removed a slow first request. With one warm-up the very first gun request took 426 ms (412 ms in the earlier medium check), and with two it took 154 ms in both runs. My reading is that the first real request was the first call down the cached-prompt path, which the second warm-up now runs before READY. I have not traced it further.

Two failures turned up in the end-to-end run, and both are fixed. Unity stops the sidecar with `Process.Kill`, which is TerminateProcess. The venv launcher's job takes the sidecar python down with it, but the Woosh worker survived: 15 s after the kill it still held port 8771 and 3.1 GB of VRAM, and the next launch would have bound a second worker beside it. `woosh_server.py --parent-pid` now waits on the sidecar process and exits when it ends, `run_sidecar.py` passes its pid, and the worker was gone 0.7 s after the kill.

The second failure was silent. The test scene sets `HF_HOME=D:\hf_cache` with `HF_HUB_OFFLINE=1`, and stable-audio-3-medium only exists in `F:\AI\cache\hf\hub`. The sidecar printed READY after 31 s with no gun model (`data/_gate/opt/_e2e/e2e_unitydefault.json`), and the first gun request dropped the client connection (each later one retries the load and would do the same), because the `FileNotFoundError` from the model load looked like a socket error to the request handler. `run_sidecar.py` now checks the SA3 checkpoint in the offline cache before it starts anything and stops with a message naming the cache (`check_weights`), and the server answers any OSError raised during generation with an `internal` error frame while the connection stays open. A missing Woosh python also stops the sidecar now, before the SA3 load starts.

## Speed before and after

All times are per clip on the RTX 3060, hit medians from the bench runs. The ratio is the log-mel distance to the reference as a fraction of the reference's seed-to-seed distance; G1 needs 0.5 or less plus the J4 artifact meter and the identity floor (details in `data/_gate/opt/woosh/results.json` and `data/_gate/opt/medium/results.json`).

| Woosh DFlow, 4 steps, 1.0 s | ms per clip | gate |
|---|---|---|
| before: eager fp32 (the brief's `dflow.json` said 120 ms; the same path measured 104 here) | 104.1 | baseline |
| + text-condition cache, 256 prompts | 86.7 | G0, 48/48 bit-identical |
| + CUDA graph over sampler, autoencoder and ISTFT | 69.2 | G0 |
| + modulation constants computed once per schedule | 65.5 | G0 |
| + bf16 DiT | 33.5 | G1, ratio 0.021 |
| + bf16 autoencoder | 31.3 | G1, ratio 0.021 |
| adopted: all of the above in one chain | 28.4 | G0 against that chain |

tf32 passed G1 at 52.0 ms (ratio 0.003) and lost to bf16, and fp16 produced NaN on 48 of 48 clips. Through the Woosh server a repeat prompt costs 32.0 ms and a new prompt 49.6 ms, finishing included.

| Woosh Flow, adaptive dopri5, 1.0 s | ms per clip | gate |
|---|---|---|
| before: fp32 eager, median 86 solver evaluations | 3584 | baseline |
| + condition cache and cached unconditional branch | 3611 | G0 |
| + CUDA graph over velocity and autoencoder | 2853 | G0 |
| + batched CFG | 2265 | G1, ratio 0.001 |
| + bf16 | 1266 | G1, ratio 0.226 |
| adopted: cache, graph, bf16, batched CFG | 1005 | G1, ratio 0.206 |

Flow bf16 is the closest G1 pass of this speed work. Its panns sign test shows 4 clips lost and none gained (p = 0.0625, just above the line), and `--precision fp32` on the Woosh server turns it off. A float32 solver clock passed G1 with no speed gain and stays off.

| SA3 medium for guns | ms per clip | gate |
|---|---|---|
| the ear-approved bake-off recipe, eager, 6 s padding | 450.2 | reference |
| before: compiled stack + `HYBRID_GUNS` with 6 s padding (the brief's 0.26 s) | 237.5 | G1 against the recipe, ratio 0.045 |
| + differential cross-attention K/V hoisted per prompt | 206.8 | G1 against the recipe, ratio 0.036 |
| + Triton GEMM for ff_in and q, tuned at 162 tokens | 195.6 | G1, ratio 0.034 |
| + noise drawn through the 9 s span proxy | 194.8 | G0, 40/40 |
| + 2 s padding, noise-matched | 158.2 | G1, ratio 0.076 |
| + 1 s padding, noise-matched | 149.9 | G1, ratio 0.093 |
| + four GEMM shapes retuned at 108 tokens, adopted | 146.4 | G1 against the recipe (0.087), the old production (0.094) and the 1 s arm (0.042) |
| production defaults with no bench flags | 146.3 | G0 against the adopted arm, 40/40 |

The hoist by itself missed G1 against the old production on one J4 reload interval, [0.003, 0.049]. That meter spans 1.10 to 1.19 on reload across near-identical arms, and every stack that contains the hoist passes against both references, so it stays in; `hoist_diff=False` costs about 31 ms a clip. A clip at the new default splits into 112 ms of sampling, 20 ms of decode, 12 ms of CPU finishing and 3 ms of copying back. Shorter generation failed G1 (2.0 s at ratio 0.92, 1.5 s at 1.42) because gun tails at the -55 dB trim floor end between 2.0 and 2.9 s.

In short: DFlow went from 104 to 28 ms (3.7x), Flow from 3.6 s to 1.0 s (3.6x), and guns from 238 to 146 ms (1.6x, or 3.1x against the eager recipe the ear picked).

## End to end through the sidecar

`scripts/bench/e2e_hybrid.py` launches the real sidecar the way Unity does, fires events with `sarge.sidecar.client`, reads VRAM from `nvidia-smi` and the per-process GPU counters, then kills the launched python with TerminateProcess and checks that nothing is left. The run below is `data/_gate/opt/_e2e/e2e_final.json`; `e2e_base.json` is the same script on the old start order.

| measurement | old start order | now |
|---|---|---|
| startup to READY | 98.9 s | 82.7 s (imports, SA3 load and warm about 62 s, then Woosh 20 to 27 s) |
| VRAM peak during startup (whole card) | 11.7 GB | 10.0 GB |
| VRAM at READY (whole card, desktop 0.65 GB) | 8.9 GB | 9.2 GB: sidecar 5.5 GB, Woosh 3.1 GB |
| VRAM after the session (43 and 54 generated sounds) | 9.6 GB | 9.7 GB: sidecar 6.0 GB, Woosh 3.1 GB |
| very first request (a gun) | 426 ms | 154 ms |
| gun first hit, 14 more keys, median / p90 / max | 151 / 162 / 166 ms | 153 / 201 / 219 ms |
| Woosh first hit, 24 keys, median | 34 ms | 34 ms |
| Woosh first hit, new prompt / cached prompt | 50 to 53 / 33 to 34 ms | 50 to 53 / 33 to 34 ms |
| repeat of a cached sound, guns / Woosh | 1.2 / 0.4 ms | 1.2 / 0.4 ms |
| replies with 2 channels and the right rate | 82 of 82 | 93 of 93 (44100 guns, 48000 Woosh) |
| clips bit-identical to the bench | 22 of 22 | 22 of 22 |
| Woosh worker after the sidecar is killed | still running after 15 s | gone after 0.7 s |

Per key, first hit at normal, soft and loud intensity (ms, final run):

| key | route | normal | soft | loud |
|---|---|---|---|---|
| shoot pistol | SA3 medium | 153.7 | 164.9 | 150.3 |
| shoot rifle | SA3 medium | 155.7 | 149.3 | 148.7 |
| reload rifle | SA3 medium | 157.6 | 219.2 | 149.2 |
| empty_click pistol | SA3 medium | 157.9 | 216.8 | 147.7 |
| equip rifle | SA3 medium | 161.7 | 150.6 | 149.7 |
| break window | Woosh DFlow | 52.4 | 34.0 | 32.8 |
| bullet impact glass | Woosh DFlow | 52.2 | 33.9 | 34.0 |
| footstep gravel | Woosh DFlow | 52.0 | 50.8 | 34.1 |
| footstep stone | Woosh DFlow | 52.1 | 51.3 | 34.1 |
| footstep water | Woosh DFlow | 52.5 | 34.1 | 33.8 |
| splash water | Woosh DFlow | 50.7 | 33.5 | 33.2 |
| footstep wood | Woosh DFlow | 51.1 | 48.7 | 33.7 |
| punch body | Woosh DFlow | 49.9 | 32.6 | 33.8 |

The two slow gun hits (217 and 219 ms) probably collided with the two intent-predicted clips the deepen loop generated in the background during that run (`demand_gen` 2 in the stats); the old-order run had none. A second connection showed no cost beyond the usual spread: 161 to 226 ms for a gun with a new space word and 50 to 52 ms for Woosh.

An idle GPU is the one large cost left, and it comes from the card's power states. After 3 to 8 s with nothing to do, the RTX 3060 drops to P3 to P8 at 210 to 450 MHz, and the next gun takes 413 to 485 ms and the next Woosh sound 210 to 361 ms. With 0 to 1 s of idle the card was still at P2 (1830 to 1970 MHz), and guns took 160 to 222 ms and Woosh 50 ms. A game that renders every frame keeps the clocks up, so in play the table above applies; an empty editor scene may sit closer to the idle numbers.

With both models loaded the card holds 9.2 to 9.7 GB, desktop included, which leaves about 2.3 GB on a 12 GB card for Unity. The startup peak of 10.0 GB comes from the SA3 load before Woosh exists.

## Unity launcher

These are the values `D:\PSfxUnityTest\Assets\WalkArena.unity` now carries. The end-to-end run `data/_gate/opt/_e2e/e2e_unity_learn.json` used them with `--port 8766` added, under Unity's environment (`HF_HOME=F:\AI\cache\hf`, `HF_HUB_OFFLINE=1`, no `TRANSFORMERS_CACHE`): READY in 76 s, gun first hit median 150 ms, Woosh 35 ms, no errors, 22 of 22 clips bit-identical to the bench.

```
pythonExe   D:\Python\SAGE\.venv214\Scripts\python.exe
workingDir  D:\Python\SAGE
arguments   D:\Python\sarge-research\modal\runpinned.py scripts\run_sidecar.py --generator hybrid --realtime --stereo --amp --detailed-prompts --learn --cache-dir D:\Python\SAGE\.cache\sfx_bake
hfHome      F:\AI\cache\hf
port        8765
```

`hfHome` moved from the scene's old `D:\hf_cache`, because stable-audio-3-medium only lives under `F:\AI\cache\hf\hub`; with the old value the sidecar now stops at startup and says so. Woosh reads its roberta-large text config from `D:\hf_cache`; `run_sidecar.py` hands the worker `HF_HUB_CACHE` from `--woosh-hf-cache` (default `D:\hf_cache`), so the worker finds it whatever `HF_HOME` or `TRANSFORMERS_CACHE` the game passes. `runpinned.py` keeps the sidecar off core 8 (Woosh is pinned by `run_sidecar.py` either way), and because it moves the working directory to `scripts\`, the cache directory is given as an absolute path. Without pinning, `scripts\run_sidecar.py --generator hybrid --realtime --stereo --amp --detailed-prompts` gives the same engine. `--learn` (the runtime affordance learner) was part of the Unity-environment run.

## Candidates waiting for your ear

`data/_gate/opt/page/index.html` (built by `scripts/bench/opt_page.py`, key in `data/_gate/opt/page_key.json`) holds every G2 candidate for blind listening: 23 sounds, three takes per option, letters shuffled per sound, best, worst and "none sound right", picks kept in the browser, Reveal and Download. Each sound also offers the production setting and the setting before the speed work, so the same page doubles as an ear check on the G1 adoptions.

| group | candidates (ms per clip as they would ship) | production |
|---|---|---|
| guns, SA3 medium | 6 steps 119; 2.0 s generation 134; 1.5 s generation 128 | 146 (recipe before the speed work 450) |
| Woosh DFlow | 4 steps at 1.5 s 33; 6 steps at 1.0 s 41 and 1.5 s 48; 8 steps at 1.0 s 54 and 1.5 s 63 | 28 (fp32 before 104) |
| Woosh Flow | rk4 8 steps 361; rk4 16 steps 720; midpoint 16 steps 364; midpoint 32 steps 723 | 1005 (fp32 before 3584) |

The Woosh candidates were rendered in fp32 and would ship in bf16; the bf16 versions sit within 0.03 of the same ratios. Six SA3 steps is the candidate I would listen to first: it saves 27 ms on every gun and moved the sound by about one seed (ratio 1.03), with pistol identity at 0.79. The shorter generations cut tails on most clips and will probably lose. On the Woosh side, 6 steps at 1.0 s (ratio 0.42) and 4 steps at 1.5 s keep a new-prompt first hit near 60 ms, while 8 steps pushes it to about 80 ms. The fixed Flow solvers would bring Flow down to about 0.36 s, still too slow for an unpredicted first hit, though fine for sounds the predictor asks for early.

## Still open

The hybrid keeps the sidecar's flat 28 ms cost estimate (`gen_cost_ms`), which feeds the prediction lead gate and the fusion ranking. Real costs are about 150 ms for guns and 34 to 52 ms for Woosh, so the gate lets gun predictions through with less lead than a gun needs. A per-route cost would fix it, and nothing measured here depends on it.

## Licences

Woosh code is MIT, and the Woosh and Woosh-DFlow weights are released by Sony AI under CC-BY-NC 4.0 (github.com/SonyResearch/woosh-sfx; tech report Hadjeres et al. 2026, arXiv 2604.01929). Anything that runs the Woosh route stays non-commercial and has to credit Sony AI. SA3 medium is released by Stability AI under the Stability AI Community License, which covers research, non-commercial use and commercial use below USD 1M of annual revenue; above that it needs an enterprise licence from Stability AI. Read both licences in full before anything ships. The repository has no NOTICE file crediting either model yet.
