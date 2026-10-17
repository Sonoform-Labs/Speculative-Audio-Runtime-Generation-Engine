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

The sidecar now loads and warms SA3 medium alone, hands the cached load memory back to the driver (`torch.cuda.empty_cache`), starts Woosh and waits for its "listening" line, connects, warms the guns a second time, warms Woosh and prints READY (`RoutedGenerator(start_rest=...)`, passed by `run_sidecar.py`). The old order started Woosh first. The SA3 load alone peaks near 9.3 GB, and with Woosh's 3.1 GB already resident the card reached 11.7 GB of its 12 GB during startup, and READY took 98.9 s. Guns first peaks at 10.0 GB and reaches READY in 82.7 s. Through the sidecar, the 22 clips checked against the bench arrays came out bit-identical in both orders (max abs 0.0), so the reorder is a G0 change. The GPU side of this order still holds, while the worker process itself now starts much earlier and waits behind two gates (see Startup below).

The second gun warm-up also removed a slow first request. With one warm-up the very first gun request took 426 ms (412 ms in the earlier medium check), and with two it took 154 ms in both runs. My reading is that the first real request was the first call down the cached-prompt path, which the second warm-up now runs before READY. I have not traced it further.

Two failures turned up in the end-to-end run, and both are fixed. Unity stops the sidecar with `Process.Kill`, which is TerminateProcess. The venv launcher's job takes the sidecar python down with it, but the Woosh worker survived: 15 s after the kill it still held port 8771 and 3.1 GB of VRAM, and the next launch would have bound a second worker beside it. `woosh_server.py --parent-pid` now waits on the sidecar process and exits when it ends, `run_sidecar.py` passes its pid, and the worker was gone 0.7 s after the kill.

The second failure was silent. The test scene sets `HF_HOME=D:\hf_cache` with `HF_HUB_OFFLINE=1`, and stable-audio-3-medium only exists in `F:\AI\cache\hf\hub`. The sidecar printed READY after 31 s with no gun model (`data/_gate/opt/_e2e/e2e_unitydefault.json`), and the first gun request dropped the client connection (each later one retries the load and would do the same), because the `FileNotFoundError` from the model load looked like a socket error to the request handler. `run_sidecar.py` now checks the SA3 checkpoint in the offline cache before it starts anything and stops with a message naming the cache (`check_weights`), and the server answers any OSError raised during generation with an `internal` error frame while the connection stays open. A missing Woosh python also stops the sidecar now, before the SA3 load starts.

## Startup

The 76 to 94 s READY times above came from back-to-back runs with every file already in the Windows file cache. A player mostly meets the cold case: the first Play after a reboot, or after other work has pushed SARGE out of RAM. `scripts/bench/evict_cache.py` reproduces that without admin rights by opening every SARGE file unbuffered, and `e2e_hybrid.py --evict` then waits until Microsoft Defender has stayed under 0.05 of a core for 15 s (`--settle`) before it launches. The first version evicted 179,925 files and 33.9 GB (15.5 GB on F:, 10.6 on D:, 4.6 on G:, 3.2 on C:), which included the whole shared user site-packages of the Store python, 106,894 files that mostly belong to other projects. It now takes from that folder only the files of the 45 distributions the sidecar imports from it, found by a probe process that imports the sidecar's modules (18,413 files, 0.5 GB), so a run evicts 91,445 files: 31.4 GB with one weight cache on G:, 36.0 GB with the two it holds now. The harness also imports numpy and the sidecar client only after READY, so numpy no longer stays mapped through the eviction. Measured the first way, HEAD reached READY in 171.0, 171.9 and 168.9 s (the last one run at the end of the evening as a control). The sidecar spent most of it waiting: 37 s on memory-mapped reads of the 9.2 GB fp32 checkpoint from the QLC drive at about 250 MB/s, then 46 to 54 s on a Woosh worker that only started once the guns had compiled.

Four changes ship, each added on top of the last and each bit-identical to HEAD: every run served the same 75 (key, variant) clips with the same shapes at max abs 0.0, the run that built the weight cache included (`scripts/bench/startup_compare.py`, results in `data/_gate/startup/compare_*.json`). Times are READY in seconds as the harness saw it; cold runs are listed one by one because Defender's share swings between runs (below).

| change | cold READY | warm READY | adopted |
|---|---|---|---|
| HEAD | 171.0, 171.9, 168.9 | 83.2, 83.0, 83.8 | |
| + prefetch: from launch, background threads read the T5 encoder range, the SA3 weights and the Woosh checkpoints, one queue per drive (`src/sarge/prefetch.py`) | 143.3, 140.9 | 90.3 | yes |
| + early Woosh: the worker starts right after the sidecar's imports with `--hold`, builds its model once the guns have loaded and touches the GPU only after gun warm-up 1 | 85.9, 85.3 | 67.3, 67.0 | yes |
| + fp16 weight cache on G: (`src/sarge/weight_cache.py`) | 68.6, 95.8 (build run 92.9) | 55.7 | yes |
| + decoder compiled once (`sa3_fast.GraphedDecode._prefill`) | 51.8, 85.8, 69.8, 86.5 | 46.5, 46.1 | yes |
| the same four changes in the working tree beside the physics-world edits | 67.3, 68.8 | 46.7, 46.3 | yes |
| HEAD again, measured with the narrower eviction (below) | 129.1, 163.0 | | control |
| + review fixes: leftover sweep, checksums and loader-code key, below-normal Woosh build, worker release on a gun failure, prefill once per shape; narrower eviction | 51.7, 71.6 (build run 84.1) | 46.3 | shipped |
| tried on top: run `ptxas --version` early | 86.3, 78.7 | 46.5 | no, the cold DiT compile kept its stall |
| tried on top: load T5Gemma with torchvision and sklearn hidden from transformers | 73.7, 83.8 | 46.0 | no, the import cost moved to an earlier stage |
| tried on top: launch Woosh when the guns have loaded | 72.9, 81.8, 80.0 | 46.3 | no gain, the T5Gemma build stayed slow |

Where the time goes, median per stage over the 3 HEAD cold, 6 shipped cold, 3 HEAD warm and 4 shipped warm runs, with the stage boundaries taken from the `[startup]` lines (`data/_gate/opt/_e2e/e2e_<tag>.json`; READY here is the line the sidecar prints, a few tenths before the harness sees it):

| stage (s) | HEAD cold | now cold | HEAD warm | now warm |
|---|---|---|---|---|
| imports (python, torch, stable_audio_3) | 6.8 | 10.9 | 4.3 | 4.5 |
| T5Gemma build | 7.2 | 17.0 | 5.6 | 5.8 |
| SA3 medium weights to the GPU | 39.5 | 2.0 | 10.3 | 2.0 |
| T5 graph and first conditioning | 1.8 | 0.9 | 1.4 | 0.6 |
| DiT compile and capture | 37.0 | 29.6 | 22.0 | 21.1 |
| decoder compile and capture | 20.2 | 10.0 | 19.9 | 9.9 |
| gun warm-up 1 | 0.2 | 0.2 | 0.2 | 0.2 |
| Woosh on the critical path | 52.5 | 2.4 | 18.1 | 1.8 |
| gun warm-up 2 and warmup request | 4.7 | 0.2 | 0.6 | 0.2 |
| READY | 170.7 | 69.1 | 83.0 | 46.2 |

The weight cache does the most per byte. HEAD reads 9.2 GB of fp32, copies it, moves it to the card in fp32 and casts it there, so the card briefly holds the fp32 model (the 10.4 to 10.5 GB startup peak) and the process holds about 18 GB of RAM. The first run after this change takes the HEAD path, then sums the 16-bit words of every tensor on the card (two int64 sums per tensor, stored in the file), copies the fp16 state dict off the card and writes it on a background thread to `G:\sarge\cache\sa3-medium-fp16-<key>.safetensors` (4.6 GB, 3.3 s on the NVMe including a flush to disk). Before it copies anything it checks that the folder can be created and has room for the cache plus 2 GB, so a missing or full drive costs a log line and nothing else. The writer works inside a private folder, `.part-<pid>` next to the cache, because safetensors puts a temporary file beside whatever target it gets; the file is flushed and then renamed into place, so the final name only ever holds a complete file. A kill mid-write does leave that part folder behind with a 4.6 GB temporary file in it. Every start that uses the cache now removes part folders whose process has ended, anything of that kind older than an hour, and the top-level temporary files the first version of this code could leave (`.tmp*` older than a minute, `*.<pid>.tmp` of a dead process). The kill test below shows both cases. Every later run builds the model with random init skipped, allocates fp16 storage straight on the card, copies the cached tensors in and recomputes the sums on the card (0.4 s for 997 tensors); one differing tensor, a missing sum, or a module with a load_state_dict hook or an override of `_load_from_state_dict` sends the start down the HEAD path, since the cache path skips `load_state_dict`. The key hashes the checkpoint's size, mtime and safetensors header, the config, the torch, safetensors and stable_audio_3 versions, the stable_audio_3 sources, `weight_cache.py`, `fastmeta.py`, and the loader parts of `stable_audio3.py` (`_RANDOM_INITS`, `_skip_random_init` and `StableAudio3Generator._cached_model`, cut from the parsed source, so edits elsewhere in that file leave the cache valid). Any of those changing sends the next start down the HEAD path, which writes a new cache. After a write the folder keeps the new cache and the newest other one, so two venvs or torch versions sharing it do not keep rebuilding each other's, and every load refreshes the mtime of the file it used. An unusable cache is deleted and rebuilt; when it is locked the run keeps it, skips the rebuild and says so. With no `SARGE_WEIGHT_CACHE` set, the default `G:\sarge\cache` applies only when `G:\sarge` exists, so another machine's G: never receives 4.6 GB by surprise. The log says which path ran (`weight cache: fp16 load from ...`, `slow path, will build ...`, `slow path, cache unusable (...)`, `removed N leftovers of interrupted writes`). A GPU check loaded the model both ways in one process and compared all 997 parameters and buffers, non-persistent ones included, plus the 136 T5Gemma tensors: same dtype, shape, stride and bytes. The startup VRAM peak fell to the READY level, 9.6 to 9.7 GB. One cold run sampled 10.1 GB while the Woosh worker moved its fp32 weights to the card; HEAD has that same transient, it only stayed hidden under the bigger SA3 peak.

The early Woosh start is the biggest single cut. `run_sidecar.py` launches the worker as soon as the sidecar's imports are done (`launch_woosh`), the worker imports everything including the finishing modules its first request used to pay for, then blocks on stdin. `RoutedGenerator(load_rest=...)` writes `load` when the guns have loaded, so the worker builds its model and reads its weights during the gun compile, and `start_rest` writes `gpu` after gun warm-up 1 and the cache release, the same moment HEAD launched the worker. The worker logs that CUDA is still uninitialised at both gates, so its GPU work never overlaps the gun compile and the VRAM profile matches HEAD. EOF on stdin ends the worker, as does the sidecar's death (`--parent-pid`). The Woosh window on the critical path went from 46 to 54 s cold and 18 s warm down to about 2 s.

Two follow-ups came out of the review. The model build now overlaps the gun compile and graph capture, a window HEAD kept free of extra CPU work, and on a real Play start Unity loads in that same window. The worker therefore runs at below-normal CPU priority from launch until the gpu gate opens and returns to normal priority before it touches the GPU, so its build yields to the sidecar and to Unity whenever they compete; it still finished its build and then waited 24 to 28 s for the gpu gate in all four runs below. And if the guns fail to load or to warm, the sidecar closes the worker's stdin, the worker exits at its next gate and its 5 GB of RAM come back. HEAD never launched the worker in that case either. A later startup attempt launches a fresh worker the HEAD way (`close_woosh`, `RoutedGenerator(drop_rest=...)`).

The prefetch threads read with 8 MB blocks on two streams per file. A file is skipped when available RAM is under its size plus 6 GB; with the fp32 load that happened to the Woosh checkpoints in every cold run, and with nothing else changed, the prefetch made one warm run 7 s slower, because the early read of the Woosh files was pushed out of RAM again before Woosh read them. In the shipped stack that pressure is gone: without the prefetch (`abl_noP_*`), the cache load took 4.9 s cold against 2.0 s with it, READY was 88.9 and 88.2 s against 85.8 and 86.5 s for the two shipped runs where Defender also stretched the T5Gemma build to about 20 s, and warm READY was 46.1 s both ways. It also keeps the HEAD path fast when the cache is missing.

The decoder used to compile twice. Its first compiled call fills `_SLIDING_WINDOW_BLOCK_MASK_CACHE` in `stable_audio_3/models/transformer.py`, the guard on that dict fails on the second call, and the CUDA graph is captured from the recompiled code. `GraphedDecode` now runs the uncompiled decode once with `flex_attention_compiled` swapped for a stub that raises, which sends that pass down the chunked SDPA fallback and fills the same dict entry, so the compiled decode traces the cached branch at once and the 9.5 s second compile disappears. The single compile takes the same 10 s the first of the two used to take, which points to the same cached inductor code, and all 75 clips stayed bit-identical, the first run included. The stub is a process-wide swap of a module global, so it now refuses only calls from the thread running the prefill and passes any other thread's call to the real compiled flex attention, which keeps a concurrent forward on its usual kernel. The prefill also runs once per latent shape and device: the dict key depends on the sequence length and window, and the batch size leaves both alone, so a capture for a new batch size at run time (prewarm, the deepen planner) skips the eager pass and leaves no extra eager reserve on the dead capture stream.

Defender is what is left. Even after the settle wait, MsMpEng used anywhere from 3 to 59 s of CPU between launch and READY, and that alone spreads the shipped cold runs from 51.8 to 86.5 s. Two stages carry it. The DiT compile runs 22 s in the quiet runs and 36 s in the heavy ones, where the sidecar sits at 0.15 of a core while MsMpEng runs at 0.8; HEAD has the same stall, it coincides with triton launching `ptxas.exe --version`, launching ptxas early changed nothing, and the window opens no files and loads no DLLs, so its cause is still open. The T5Gemma build is the one stage that got worse: 5.4 and 5.7 s in the two quiet shipped runs and 5.3 to 5.7 s in all four runs of the prefetch and early-Woosh arms, but 10 to 21 s in four of six shipped cold runs while Defender scans the transformers files on C:, against 6.0 to 8.5 s in all three HEAD runs, the evening's control included. It shows up with and without the prefetch and with Woosh launched later, so it comes with the cached load path, and I have not found which access starts the scanning. Even in those runs the shipped stack reaches READY 82 s ahead of HEAD. The DiT compile itself (21 s warm) is the next big target, and it belongs to the compile work.

The review fixes were measured with the narrower eviction, so HEAD ran again under it (`fixed_head_cold`, `fixed_head_cold2`). HEAD reached READY in 129.1 s with Defender at 3.8 s of CPU and in 163.0 s with Defender at 37.9 s. The fixed stack took 51.7 s (Defender 1.8 s) and 71.6 s (Defender 36.3 s, with the same DiT stall at 36.3 s), plus 84.1 s on the run that built the new cache, and 46.3 s warm. At matched Defender load that is 2.3 to 2.5 times faster. The old eviction itself fed Defender: HEAD saw 56 to 61 s of MsMpEng CPU under it, which is most of the gap between 129 and 169 to 183 s. All six runs, both HEAD controls included, served the same 75 clips as `baseline_warm_1` at max abs 0.0 with equal shapes (`compare_fixed.json`; `compare_final_shared.json` now covers `final_shared_cold_2` too), and the rebuilt cache matches the one before it byte for byte in all 997 tensors. VRAM: the startup peak was 10,194 MiB cold and 9,662 MiB warm on the cache path and 10,463 MiB on the build run, against HEAD's 10,423 and 10,461, with 9,670 to 9,696 MiB at READY against HEAD's 9,648. First hits: gun median 149.3 to 150.1 ms (HEAD 149.3 and 149.5), Woosh median 34.3 to 35.1 ms (HEAD 34.3), the very first request 153.0 ms warm and 154.6 to 361.4 ms cold (HEAD cold 154.5 and 386.2 ms). The Woosh first-hit median sits on an edge in every run, HEAD included: 13 of its 24 requests land near 33 ms and 11 near 50 ms, so two slow requests can move it by 15 ms. The median over all generated Woosh requests is the steadier number, 49.7 to 50.6 ms here against HEAD's 49.9 and 50.4.

The kill test (`data/_gate/startup/kill_fixed/`) ran against `G:\sarge\verify_kill`, which held the half-written `.tmpTqYbDB` from the review. The first start removed it at once (`removed 1 leftovers of interrupted writes (.tmpTqYbDB)`), took the slow path, and was killed 0.3 s after `state copied to RAM`. That left `.part-8996` holding a 4.6 GB temporary file and no final cache. The next start removed `.part-8996` before anything else (`removed 1 leftovers of interrupted writes (.part-8996)`). Both kills cleared every SARGE process within 1.3 s and freed ports 8766 and 8771.

Harness cold and Unity cold still differ in both directions. Opening each .dll and .pyd with write access is what flushes its image pages, and Defender then rescans those files on their next load, which a start after a reboot may skip. On the other side, 51 files of the base python under WindowsApps refuse the write open (access denied, so their image pages stay), the harness keeps the venv launcher and psutil mapped, and the CPU, GPU and disk load of Unity itself is absent. Unity's real HEAD cold start took 209 s, above every harness HEAD run, so expect the real cold READY of this stack to land above the harness numbers as well. The `[startup]` lines in `sidecar.log` after the next real Play give its real stage times.

Nothing moved under gate R. The only new data is the cache folder `G:\sarge\cache`, which now holds two 4.6 GB files: the current cache (`sa3-medium-fp16-a050e74fc4551e9e591e`, with checksums) and the one before it (`e64ed3c524c8fc8241e5`), kept as the newest other key. The SA3 folder on F:, the Woosh checkpoints, roberta-large and both venvs stay where they were. To undo: delete `G:\sarge\cache`, or keep it and start with `--weight-cache off` (or `SARGE_WEIGHT_CACHE=off`); `SARGE_PREFETCH=0` turns the prefetch off and `SARGE_WOOSH_EARLY=0` brings back the HEAD Woosh start. `--weight-cache <folder>` or `SARGE_WEIGHT_CACHE` moves the cache. Tests: `tests/test_startup_speed.py` (cache keying including the loader segments, round trip, checksums and hook refusal, the part folder and the leftover sweep, pruning, the space check, fallback, prefetch plans, gate order, worker release, the thread-aware prefill); `tests/conftest.py` turns the three switches off for the rest of the suite.

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

The arguments did not change for the on-time defaults: `--concurrent-gen` is now on inside `run_sidecar.py` (see "Concurrent misses on by default" below), so the scene gets it without a new flag, and adding `--no-concurrent-gen` to the arguments brings back the single-lock path. The one scene edit of 2026-10-17 is physical: the shared ball and tile PhysicMaterial `psfx_bouncy` now has bounciness 0.8 where it had 1, so the demo balls lose energy on each bounce (`docs/physics-context.md`, "The demo balls now lose energy"; the old scene is in `D:\PSfxUnityTest_backup_20261016\WalkArena.unity`).

`hfHome` moved from the scene's old `D:\hf_cache`, because stable-audio-3-medium only lives under `F:\AI\cache\hf\hub`; with the old value the sidecar now stops at startup and says so. Woosh reads its roberta-large text config from `D:\hf_cache`; `run_sidecar.py` hands the worker `HF_HUB_CACHE` from `--woosh-hf-cache` (default `D:\hf_cache`), so the worker finds it whatever `HF_HOME` or `TRANSFORMERS_CACHE` the game passes. `runpinned.py` keeps the sidecar off core 8 (Woosh is pinned by `run_sidecar.py` either way), and because it moves the working directory to `scripts\`, the cache directory is given as an absolute path. Without pinning, `scripts\run_sidecar.py --generator hybrid --realtime --stereo --amp --detailed-prompts` gives the same engine. `--learn` (the runtime affordance learner) was part of the Unity-environment run.

## Loudness levelling and world playback (flags off)

The Unity session on 2026-10-16 found the glass inaudible next to the guns and the ball on metal far too loud. Most of the glass gap is playback: windows and bullet hits play through `AudioSource.PlayClipAtPoint`, a 3D source with a 1 m minimum distance, which costs 15 to 26 dB at the range's 5 to 18 m, while guns, steps and balls play 2D. Peak levelling adds a few dB on sparse sounds, and the amplifier's saturation made a ringing metal clip the loudest thing served (-4.7 LUFS). Everything below is opt-in and the defaults are unchanged.

`--level loudness` (run_sidecar, passed on to the Woosh worker) levels each clip by its maximum 400 ms K-weighted loudness: hard -14, normal -17, soft -23 LUFS (`LUFS_BY_INTENSITY` in `src/sarge/loudness.py`), after a 30 Hz high-pass, with a 2 ms lookahead limiter at -1 dBFS that may take at most 6 dB (`MAX_GR_DB`). A clip too sparse to reach its target within those 6 dB stays quieter. A clip whose loudest 400 ms stays under -70 LUFS (the BS.1770 absolute gate, `SILENCE_LUFS`) is left as it is, so a silent take is never boosted into noise. The target sits near the median loudness of the current chain, so the change moves sounds relative to each other and leaves the overall level about where it was. In this mode the amplifier runs at its normal strength for every intensity and the intensity shows up as level. `--amp punch` keeps the attack emphasis and drops the saturation; `--amp` alone still means the full amplifier. The chain lives in `src/sarge/post.py` (`process`, plus `decay_cap` for the physics context), `woosh_server.finish` and `StableAudio3Generator._finish` hand off to it only when a flag is set, and the clip cache adds a tag to its hash only then.

With every flag off, the current `finish` and `_finish` reproduce HEAD bit for bit: 1296 of 1296 Woosh outputs (108 raw renders, three intensities, amplifier and stereo on and off) and 1280 of 1280 SA3 outputs (40 raw renders, 32 settings), max abs 0.0. On the same raw renders (`woosh_buckets/sweep/101` and `medium/_work/raw/m1_prod`):

| at normal | current chain (`--amp --stereo`) | `--level loudness --amp punch` | `--level loudness --amp` |
|---|---|---|---|
| median LUFS blast / handling / impact / other / step | -13.6 / -19.6 / -16.6 / -16.6 / -17.7 | -17.1 / -18.4 / -18.5 / -17.2 / -18.4 | -17.0 / -18.4 / -17.0 / -17.2 / -17.3 |
| spread p10 to p90, same order (dB) | 1.6 / 4.8 / 8.8 / 5.9 / 10.0 | 0.9 / 2.8 / 6.4 / 1.1 / 6.9 | 0.0 / 2.8 / 3.1 / 1.1 / 4.6 |
| glass minus pistol (dB) | -2.2 | -0.9 | 0.0 |
| normal to hard, median step; renders with 3 dB or more | 0.5 dB; 0 of 148 | 1.0 dB; 8 of 148 | 1.4 dB; 33 of 148 |
| limiter, median / max (dB) | none | 1.8 / 6.0 | 0.0 / 6.0 |
| clips with over 30% of energy below 40 Hz | 12 | 2 | 2 |

Footsteps and impacts keep the widest spread because the shortest of them would need more than 6 dB of limiting to reach a 400 ms target, and the hard step shrinks for the same reason: most renders hit the limiter cap at hard. A 100 ms meter brought every family under 2.5 dB of spread in a side sweep, at the cost of making long ringing sounds louder relative to short ones, so it stays a measurement for now.

Playback is a Unity switch. `unity/Sarge/Runtime/SargePlayback.cs` is a pool of 16 hidden 3D sources (logarithmic rolloff, no doppler, spread 60) that does nothing unless called. The demo's `Assets/Demo/SargeMix.cs` holds the switch: L in `SargeFirstPerson` toggles it (K already kicks the balls), and the HUD shows "mix: HEAD" or "mix: v2". With v2 on, windows and bullet hits play through the pool with an 8 m minimum distance, every `SargeAutoCapture` source turns 3D with the same distance and no doppler (the originals come back on the next toggle), and the volume table matches by sound family, so "iron boot footstep", "iron boot landing" and "take off" stop falling through to the impact volume (the iron boot sounds drop 5.0 dB to the footstep volume and take off drops 2.5 dB to the handling volume). A family word counts only as a whole word with an English ending (s, es, ed, ing, a doubled consonant or a dropped e) and the last matching word of the name wins, so "glass crack" stays an impact, "shotgun reload" is handling and "fireplace" is no blast. `SargeMix.DuckDb` (0, off) dips the player's source for 150 ms when a world sound starts within 50 ms of a shot. Relative to the pistol at the ear, with the full amplifier and v2 at 8 m, a window 9.6 m away goes from -22.7 to -1.6 dB, at 17.8 m from -28.0 to -6.9 dB, a bullet on the glass floor 16 m away from -25.8 to -8.0 dB, and a hard ball on metal 7 m away from +8.9 to +3.0 dB (-1.9 at a 4 m minimum distance). In the window's first 150 ms its 1 to 2, 2 to 4 and 4 to 8 kHz bands sit -32.9, -24.1 and -17.2 dB under a simultaneous pistol today and -15.2, -4.8 and +2.8 dB with both changes; a 6 dB duck gives -9.2, +1.2 and +8.8.

Finishing costs more on the CPU (pinned, 4 threads): a Woosh clip 4.5 ms today, 5.9 ms with `--amp punch` and 6.5 ms with the full amplifier; a gun clip 19.7, 19.8 and 23.4 ms. `scripts/bench/level_page.py` builds the blind page `data/_gate/feel/loudness_page/index.html`: 46 single clips through the three chains and 7 game scenes (six with the current chain against v2 at 4, 8 and 12 m, plus six footsteps through the three chains), at the levels the demo plays them, with best and worst picks, Reveal and Download. Tests: `tests/test_level.py`.

## Concurrent misses on by default

Since 2026-10-17 `run_sidecar.py` turns `--concurrent-gen` and `--async-cache-write` on unless `--no-concurrent-gen` or `--no-async-cache-write` is given. Under `--concurrent-gen` a realtime miss renders outside the engine lock, so a cached sound on another connection answers while a gun renders, and no prediction or background render starts beside a miss until it lands. It is the one on-time flag whose checks all passed; `docs/on-time.md` has the measurements, the check that found what held some first hits (the 16-bit disk-cache write on the reply path, which HEAD has too), and the flags that stay off. Every sound-changing flag (`--world`, `--level loudness`, `--amp punch`, mix v2) stays off, and so do `--keep-warm-ms`, `--pool-fix` and `--sibling-on-miss`; `--async-cache-write` came on later the same day, and `--predict-fixes` after that (next section).

`e2e_hybrid.py` ran four times under Unity's environment on ports 8767 and 8772 with `--learn`: once cold (`--evict --settle 15`) and once warm with the new defaults, once warm with `--no-concurrent-gen` (every flag off) and once warm with `--async-cache-write` added. Results are in `data/_gate/opt/_e2e/e2e_ontime_*.json`; the clip comparisons are `data/_gate/startup/compare_ontime.json` and `compare_ontime_q0.json` (the saved arrays were deleted after the comparison; each run folder keeps its `index.json` with a sha1 per clip).

| | cold, defaults | warm, defaults | warm, every flag off | warm, + `--async-cache-write` |
|---|---|---|---|---|
| READY | 66.1 s | 47.1 s | 46.6 s | 46.4 s |
| VRAM peak during startup / at READY (MiB) | 9,687 / 9,721 | 9,721 / 9,721 | 9,721 / 9,721 | 9,687 / 9,721 |
| VRAM peak during requests (MiB) | 10,245 | 10,245 | 10,245 | 10,245 |
| very first request (a gun) | 154.0 ms | 153.3 ms | 153.8 ms | 153.3 ms |
| gun first hit, 14 keys, median / p90 / max | 150.0 / 158 / 162 ms | 150.2 / 161 / 187 ms | 149.8 / 160 / 162 ms | 149.1 / 157 / 161 ms |
| Woosh first hit, 24 keys, median / p90 | 34.5 / 52.0 ms | 34.2 / 52.1 ms | 34.5 / 52.3 ms | 33.7 / 51.6 ms |
| repeat of a cached sound, guns / Woosh | 1.19 / 0.57 ms | 1.21 / 0.61 ms | 1.29 / 0.58 ms | 1.31 / 0.43 ms |
| clips against `baseline_warm_1` | 75 of 75 identical | 75 of 75 identical | 75 of 75 identical (gate Q0) | 75 of 75 identical |
| Woosh worker after the kill | gone in 0.8 s | gone in 0.8 s | gone in 0.8 s | gone in 0.8 s |

All four runs passed gate V (startup peak at most 10,752 MiB, READY at most 9,830 MiB). This harness samples the card every 0.5 s; the on-time bench samples every 0.1 s and saw startup peaks of 10,031 to 10,259 MiB with the same code, still inside the bound; the peak while both models rendered at once, which this harness does not set up on purpose, was 10,243 to 10,247 MiB in the on-time bench's `s2` runs, under 11,571. The cold run evicted 91,449 files (36.0 GB) in 139 s and waited 110 s for Defender to settle; Defender still used 17.6 s of CPU before READY, and the DiT compile took 37.4 s (22 s in the warm runs), the same stall described under Startup, so 66.1 s sits inside the 51.7 to 86.5 s spread of the earlier cold runs. This harness sends one request at a time on one connection, so the first-hit columns match whatever the flags; the head-of-line gain of `--concurrent-gen` shows only with the T preset's two event lanes, as measured in `docs/on-time.md` (a cached sound behind a gun miss: 245 ms at the median under HEAD, 0.6 ms now).

## Prediction fixes on by default

`run_sidecar.py` also turns `--predict-fixes` on unless `--no-predict-fixes` is given. It maps a predicted stem to the event the game fires (`footstep` to `iron boot footstep` while the boots are on), fills the intensity of a prediction that sends none from the last fired one and keeps an intensity a prediction sends, drops the lead gate in realtime mode and expands `after` to learned follow-up events. In the on-time bench's scripted session (`docs/on-time.md`, `s3`) it served 5 and 6 of 110 events on demand against 7 and 7, played 6 of 22 first hits over 20 ms against 7 in both runs, and used 1.6% less GPU time in all; every served clip matched the reference. `--pool-fix` stays off: with its deadline order fixed it still cost 72% more GPU time and gained nothing in the session.

`e2e_hybrid.py` ran twice warm under Unity's environment on ports 8767 and 8772 with `--learn`: once with the new defaults and once with every flag off (`--no-predict-fixes --no-concurrent-gen --no-async-cache-write`). Results are in `data/_gate/opt/_e2e/e2e_predfix.json` and `e2e_predfix_q0.json`, the clip comparisons in `data/_gate/startup/compare_predfix.json` and `compare_predfix_q0.json` (the saved arrays were deleted after the comparison; each run folder keeps its `index.json`).

| | warm, defaults | warm, every flag off |
|---|---|---|
| READY | 46.8 s | 46.6 s |
| VRAM peak during startup / at READY (MiB) | 9,682 / 9,716 | 9,716 / 9,716 |
| VRAM peak during requests (MiB) | 10,240 | 10,240 |
| very first request (a gun) | 153.6 ms | 153.3 ms |
| gun first hit, 14 keys, median / p90 / max | 149.5 / 159 / 161 ms | 149.8 / 159 / 168 ms |
| Woosh first hit, 24 keys, median / p90 | 33.7 / 51.4 ms | 34.3 / 52.2 ms |
| repeat of a cached sound, guns / Woosh | 1.25 / 0.39 ms | 1.27 / 0.48 ms |
| clips against `baseline_warm_1` | 75 of 75 identical (gate L) | 75 of 75 identical (gate Q0) |
| Woosh worker after the kill | gone in 0.8 s | gone in 0.8 s |

Both runs passed gate V. This harness sends no PREDICT messages, so the prediction fixes reach only the guesses of the habit model here, and the first-hit columns match within run-to-run spread. The scene needs no new argument, and `--no-predict-fixes` in its arguments brings back literal predictions.

## Candidates waiting for your ear

`data/_gate/opt/page/index.html` (built by `scripts/bench/opt_page.py`, key in `data/_gate/opt/page_key.json`) holds every G2 candidate for blind listening: 23 sounds, three takes per option, letters shuffled per sound, best, worst and "none sound right", picks kept in the browser, Reveal and Download. Each sound also offers the production setting and the setting before the speed work, so the same page doubles as an ear check on the G1 adoptions.

| group | candidates (ms per clip as they would ship) | production |
|---|---|---|
| guns, SA3 medium | 6 steps 119; 2.0 s generation 134; 1.5 s generation 128 | 146 (recipe before the speed work 450) |
| Woosh DFlow | 4 steps at 1.5 s 33; 6 steps at 1.0 s 41 and 1.5 s 48; 8 steps at 1.0 s 54 and 1.5 s 63 | 28 (fp32 before 104) |
| Woosh Flow | rk4 8 steps 361; rk4 16 steps 720; midpoint 16 steps 364; midpoint 32 steps 723 | 1005 (fp32 before 3584) |

The Woosh candidates were rendered in fp32 and would ship in bf16; the bf16 versions sit within 0.03 of the same ratios. Six SA3 steps is the candidate I would listen to first: it saves 27 ms on every gun and moved the sound by about one seed (ratio 1.03), with pistol identity at 0.79. The shorter generations cut tails on most clips and will probably lose. On the Woosh side, 6 steps at 1.0 s (ratio 0.42) and 4 steps at 1.5 s keep a new-prompt first hit near 60 ms, while 8 steps pushes it to about 80 ms. The fixed Flow solvers would bring Flow down to about 0.36 s, still too slow for an unpredicted first hit, though fine for sounds the predictor asks for early.

The physics context has a separate page, `data/_gate/feel/world_page/index.html` (built by `scripts/bench/world_page.py` from the `world_render.py demo` renders, key in `data/_gate/feel/world_page_key.json`): the eleven demo collisions, each through HEAD, `--world` and `--world --level loudness`, three takes each at the level the game plays them. The numbers behind it are in `docs/physics-context.md` under "GPU checks run".

## Still open

The hybrid keeps the sidecar's flat 28 ms cost estimate (`gen_cost_ms`), which feeds the prediction lead gate and the fusion ranking. Real costs are about 150 ms for guns and 34 to 52 ms for Woosh, so the gate lets gun predictions through with less lead than a gun needs. A per-route cost would fix it, and nothing measured here depends on it.

## Licences

Woosh code is MIT, and the Woosh and Woosh-DFlow weights are released by Sony AI under CC-BY-NC 4.0 (github.com/SonyResearch/woosh-sfx; tech report Hadjeres et al. 2026, arXiv 2604.01929). Anything that runs the Woosh route stays non-commercial and has to credit Sony AI. SA3 medium is released by Stability AI under the Stability AI Community License, which covers research, non-commercial use and commercial use below USD 1M of annual revenue; above that it needs an enterprise licence from Stability AI. Read both licences in full before anything ships. The repository has no NOTICE file crediting either model yet.
