# SARGE generation plan

Revised 2026-10-02 at HEAD 1513139. The August version of this plan lives at `D:\Python\sarge-research\model-plan.md`. It blamed latency on fixed overhead around the model, and the measurements below refute that.

Every number names its source. Bench outputs sit in `scripts/bench/` and come from the RTX 3060 running SA3 small-sfx in fp16 with the euler sampler at cfg 2.0, unless a row says otherwise. **Open** marks an item with no measurement yet. **Carried over** marks a claim from the August literature survey that this pass did not re-check.

---

## 1. Stay on Stable Audio 3 small-sfx

The verdict stands. The reasoning behind it changed.

Latency was sampling bound from the first profile. `model.generate` took 174.3 ms of a 176.5 ms clip, while prompt building, the numpy copy, onset extraction, loudness and PCM encoding added 1.9 ms between them (`out_bench_stages.txt`). A two-point fit over the eager 3 and 8 step medians (167.5 and 391.0 ms in `out_bench_graph.json`) gives 44.7 ms per step and 33.4 ms of fixed cost. August put about 135 ms of a 180 ms clip outside the step loop. The real fixed cost was a quarter of that.

The August arithmetic also miscounted the sequence. It priced a 1 s clip at about 11 latent frames, but the wrapper padded each request with 6 s of headroom and the DiT sampled 76 frames (`out_bench_graph.json`, `facts.latent_len`). One frame spans 4096 samples at 44.1 kHz, or 92.9 ms (`downsampling_ratio` and `sample_rate` in the model's `model_config.json` in the Hugging Face cache). The current build pads 1 s (`duration_padding_s` in `src/sarge/generators/stable_audio3.py`), so the sampler sees about 2 s of latent. `_adapt_sample_size` in the installed `stable_audio_3/model.py` rounds the request up to a whole frame and then to an even frame count, which gives 76 frames for 7 s and 22 frames for 2 s. No bench file logs the 22 directly.

### Where the time goes

| Change | 3 steps (ms) | 8 steps (ms) | Output vs eager | Source |
|---|---|---|---|---|
| Eager baseline, 6 s headroom | 167.5 | 391.0 | reference | `out_bench_graph.json` `runs.b1.baseline` |
| tqdm off | 166.5 | 391.6 | bit identical | `runs.b1.notqdm` |
| Checkpointing off, `has_lora` walk patched out | 153.8 | 359.6 | bit identical | `runs.b1.trim` |
| T5 conditioning cached per prompt | 139.4 | 343.2 | bit identical | `runs.b1.cond` |
| CUDA graph of `DiffusionTransformer._forward` | 98.3 | 232.4 | bit identical | `runs.b1.graph` |
| Headroom cut from 6 s to 1 s | 67.6 | 165.8 | bit identical vs eager at 1 s | `out_gate_tiers.json`, `out_fast_equiv.json` |

The graph was the largest single cut at 41 ms. The T5 cache saved 14 ms, in line with the 14.4 ms conditioner probe (`out_bench_graph.json`, `probes.conditioner_ms`), and the `has_lora` walk alone cost 1.2 ms per call (`probes.has_lora_walk_ms`). Cutting the headroom took another 31 ms. At 1 s of headroom the 3 and 8 step rows fit to about 19.6 ms per step and 9 ms fixed, and the 2 step fast tier lands at 47.8 ms (`out_gate_fast_tier.json`). Sampling is still seven eighths of the clip.

The headroom cut changed the audio, so it went through paired tests. On 40 footstep seeds at 3 steps, 1 s against 6 s scored +7/-6 on CLAP material and +5/-4 on CLAP footstep, both p=1.000 (`out_fewstep_paired.json`). Across eight keys (four impacts, a splash, a landing, a jump and a gravel footstep, 10 clips each), CLAP key identification went +4/-6 with p=0.754 and Audiobox PQ rose 0.11 (`out_check_pad.json`). Those are ties at n=40 and n=80, and McNemar tests at those sizes only catch large losses.

Through the sidecar, the first request for each of 8 keys took a median 84.0 ms (max 88.3). A repeat came back from cache in 0.1 ms, and a footstep predicted 400 ms ahead was served from cache in 0.26 ms (`out_e2e_sidecar.json`). The 16 ms between the 67.6 ms generate and the 84.0 ms first hit sits close to the 14.4 ms conditioner probe, and every first hit in that run used a prompt the cache had never seen. That link is an inference with no direct measurement.

Batching pays less than August expected. In the eager build, 8 clips took 514.2 ms against 169.5 ms for one (`out_bench_launch.json`). That is 64.3 ms per clip and three times the single-clip wall time, where August had predicted barely more than one. After the graph and the headroom cut, a pair takes 126 ms, or 63 ms per clip against 67.7 ms alone (`out_fast_equiv.json`). Batches above one skip the T5 cache to stay bit identical (`Accelerator.kwargs` in `src/sarge/generators/sa3_fast.py`), which probably eats most of the gain; that split is unmeasured. The background loop's batch of 2 now buys about 7% throughput.

### Compiling and the schedule (2026-10-11)

The profile said the old figure undersold the slack. An eager `torch.profiler` pass put most GPU time in small elementwise kernels and dtype casts (about 2,100 multiplies, 1,600 copies and 3,300 casts per clip, plus a composite RMSNorm), with the matrix multiplies in the minority (`scripts/bench/out_speed_levers.json`). That is the case kernel fusion exists for, and `triton-windows` 3.2.0.post21 made `torch.compile` work on this machine.

| Change, inside one run each | ms per clip | Output vs eager | Source |
|---|---|---|---|
| Eager graph and hoist (same-run baseline) | 61.8 to 62.3 | reference | `out_compile_probe.txt`, `out_compile_probe2.txt` |
| `torch.compile` of `_forward` inside the CUDA graph | 45.2 | 0.044 of a seed change (log-mel) | `out_compile_probe.txt` |
| plus the compiled decoder | 42.4 | 0.107 | same |
| max-autotune in place of default | 44.8 | 0.045 | same |
| Decoder CUDA graph alone | 61.2 | bit identical | `out_compile_probe2.txt` |
| Rope hoist alone | 61.5 | bit identical | same |
| Compiled DiT and decoder, decoder graph, rope hoist | 41.5 | 0.107 | same |
| plus skipping the t=1.0 Euler step | 30.2 | 0.248 | same |

The held-out judges scored the compiled engine the same as eager on every judged key (`scripts/bench/out_vocab_speed.txt`), and the full set with the first step dropped stayed inside every interval with no McNemar drop: rifle 0.84 against 0.94, pistol 0.55 against 0.60, dog 0.85 against 0.90, wood and gravel steps 1.00 (`out_vocab_full.txt`). The first step moves the noise level from 1.0 to 0.970, so the model call there buys almost nothing. Both now run in the default and quality tiers of `scripts/run_sidecar.py`, with `--no-compile` and `--no-skip-first` to opt out. Through the sidecar in realtime mode a first hit dropped from 74.3 to 48.3 ms (max 51.6) and repeats stayed at 0.2 ms (`out_e2e_sidecar_rt_old.json`, `out_e2e_sidecar_rt_new.json`). The cost is startup: 36.6 s to READY against 18.5 s, almost all of it tracing, which the inductor cache under `.cache/inductor` does not save.

Three levers lost on the judges. CFG 1 and CFG on the first step only lifted pistol, punch and stone steps but dropped dog barks to 0.20 to 0.25 and gravel steps to 0.72. Two Euler steps lost both gunshots (rifle 0.48) and let 3 static clips through the guard in 400. A fourth arm became the fast tier. With the first step dropped, CFG on the remaining first step only runs 27.1 ms a clip and moves pistol from 0.55 to 0.93, clicks from 0.70 to 1.00 and stone steps from 0.41 to 0.88, while dog barks fall from 0.85 to 0.55 and metal steps from 0.20 to 0.00 (`out_vocab_skip1cfg.txt`). Eight keys clear 0.8 against six for the default, and the dog drop is outside its interval, so it is `--tier fast`, a choice, and it replaces the old two-step fast tier that tonight's two-step runs showed leaking static. After all of this the clip is GPU bound. On an idle GPU the default takes 26.2 ms: 21.5 ms of sampling, 3.4 ms of decode, 0.4 ms of `_finish` and 0.7 ms of static guard (`out_breakdown.json`), and a cProfile pass shows the CPU waiting on the GPU in `.cpu()` and the hoist's own syncs, with Python overhead under a millisecond. The matrix multiplies now run near 73 percent of the 3060's fp16 rate with fp32 accumulation (ff_in at 18 TFLOPS), and `torch._int_mm` is no faster at these shapes (1.0 to 1.9 times slower at M=172), so int8 is out. The next lever is fp16 accumulation, which GeForce cards run at twice the rate; `torch.backends.cuda.matmul.allow_fp16_accumulation` arrives in PyTorch 2.7, so it waits on an upgrade and the judges. The CPU window search in `_extract` moved from `np.convolve` to a cumulative sum with the same argmax on 600 random clips, about 2.7 ms less CPU per clip.

### Constraints against switching (carried over)

| Constraint | Why alternatives fail |
|---|---|
| **License (shipping middleware)** | Woosh and Woosh-DFlow weights are CC-BY-NC, MMAudio weights are CC-BY-NC, and TangoFlux, Tango 2 and AudioLDM 2 are non-commercial. SA3, SAO Small and MeanAudio (MIT) are the shippable ones. |
| **SFX-specific quality** | SA3 small-sfx is the one candidate with a published short-SFX eval (5 s BBC-SFX: FAD 0.395, CLAP 0.351), about 10% behind SA3 large on FAD and 5% on CLAP. |
| **Format** | SA3 outputs 44.1 kHz stereo. MeanAudio is mel plus BigVGAN, mono, with the sample rate unstated in its paper. |

The parameter counts now come from the safetensors headers in the Hugging Face cache: 459.1 M in the DiT, 108.2 M in the autoencoder and 281.6 M in the T5Gemma encoder.

### Alternatives

August picked Stable Audio Open Small with ARC ([arXiv:2505.08175](https://arxiv.org/abs/2505.08175)) as the fallback because its t5-base encoder would attack "the actual bottleneck". The text encoder was 14.4 ms of a 167.5 ms clip (`out_bench_graph.json`), and the per-prompt cache now skips it on repeats. The proposed two-model router, with SAO Small as a 135 ms cold-miss tier, has no job left either. SA3 small-sfx serves the default tier in 67.6 ms and a 2 step fast tier in 47.8 ms on one model (`out_gate_tiers.json`, `out_gate_fast_tier.json`, tier table in `scripts/run_sidecar.py`).

The checkpoint is already adversarially post-trained, so a second distillation pass would repeat Stability's work. Its `model_config.json` carries a `training.arc` block (relativistic discriminator loss, contrastive term, the model reused as its discriminator) and demo settings of 8 steps at cfg 1. The installed package defaults `rf_denoiser` models to the pingpong sampler (`stable_audio_3/inference/sampling.py`, line 434), and its registry lists a separate `small-sfx-base` checkpoint (`stable_audio_3/model_configs.py`).

**Open: SA3 medium.** The same registry lists a `medium` checkpoint. It is the one untested lever aimed at the realism gap in section 3. Its VRAM and latency on the 3060 are unmeasured; small-sfx peaks at 2.28 GB in the graph build (`out_bench_graph.json`, `max_mem_gb`).

**Research arm only (carried over):** the Woosh autoencoder could serve in an ablation table for decoder questions. Its CC-BY-NC license keeps it out of the runtime.

### Config items

**cfg stays at 2.0.** August called it "~2x DiT compute for nothing" and said to set 1.0. In the eager build cfg did nothing to latency: 171.2, 172.0 and 171.4 ms at cfg 2.0, 1.5 and 1.0 (`out_bench_stages.txt`). Each step was bound by CPU dispatch. The profiler pass in `out_bench_launch.json` adds little here: under the profiler one generate took 937 ms of wall time, its summed kernel time of 554 ms exceeds the 169.5 ms unprofiled generate, and it counted zero kernel launches, so its 59% busy figure is a profiler artifact. cfg 2.0 does double the DiT rows (the graph probe times a two-row forward, `out_bench_graph.json`, `forward_probe.rows2`), and once the graph removed the dispatch cost the doubling began to show. In the few-step probe, which called `generate` directly with the graph on, the T5 cache off, per-layer checkpointing left at the package default and 6 s of headroom (`probe_fewstep.py`), cfg 1.0 took 94.9 ms against 134.3 ms (`out_fewstep.json`). Quality settles it. At 3 steps, cfg 1.0 dropped CLAP material from 77.5% to 52.5%, +1/-11, p=0.006 at n=40 (`out_fewstep_paired.json`).

**Open: dead backend import.** `scripts/model_server.py` lines 60 to 62 still import `meanaudio_backend`, and no file of that name exists in the repo. Delete the branch or land the file.

---

## 2. Streaming

SA3 latents arrive in 92.9 ms frames (4096 samples at 44.1 kHz, from `model_config.json`), so no sampler setting can emit audio from them in smaller pieces. The SA3 paper's description of the autoencoder as non-streaming and the 49 ms latency JND from Schmid et al. (Audio Mostly 2024, n=37) are carried over.

The case for a separate causal transient head got weaker. In August the head had to cover a 180 ms generate. Today a cold first hit lands in 84.0 ms end to end, and a predicted or repeated key comes back in under 0.3 ms (`out_e2e_sidecar.json`). The head would hide about 84 ms, and only on cold misses. Its second motivation was smeared attacks, and at 3 steps those are gone: generated footsteps rise 10 to 90% in 4.41 ms against 10.40 ms for the 40 real ones (`out_gate_tiers.json`). Plain 2 steps did soften the attack to a 17.7 dB crest, and the 0.68 schedule shift of the fast tier brought it back to 19.8 dB with the decoder untouched (`out_fewstep_paired.json`). That points at the 2 step noise schedule as the cause.

The design stays on file as a research arm (carried over): a BRAVE-scale causal decoder of about 5 M parameters at compression ratio about 128, keyed by an embedding of the discrete `CacheKey`, distilled from the first 50 to 80 ms of SA3 output, and crossfaded into the SA3 tail.

**Open: Unity DSP buffer floor.** Unity's "Best latency" setting is 256 samples, about 5.3 ms at 48 kHz, and the default is 512 (carried over). No file in the repo records what the test project reports. It is a one-hour check in `D:\PSfxUnityTest`.

**Open: real-recording attack splice.** A small bank of real attacks from `data/real_footsteps`, crossfaded into generated tails, would test the splice question with zero training. It stays the cheapest experiment before any head is trained.

---

## 3. Quality measurements

The base gate looks healthy at 68 ms. With the 7-label CLAP set, 80% of 40 generated footsteps read as footsteps and 80% carry the right material, with a 20.0 dB onset crest against 18.7 dB for real clips and Audiobox PQ 7.43 (`out_gate_tiers.json`). The material score owes a lot to the prompt. Saying the intensity in every footstep prompt lifted CLAP material from 45% to 76% over 80 seeds, +32/-7, p=0.0001 (`out_prompt_confirm.json`, commit f36daa1).

### The judge correction

Two of the three taggers fail their positive control on real 0.6 s crops. AST tags 22.5% of them as footsteps and PANNs 45%, both under the 50% floor (`out_gate_v2.json`, `real_crop`). CLAP with the 13-label hard set (the 7 base labels plus 6 percussive distractors) hits 87.5% of real crops and 18% of ESC-50 negatives, so it is the one valid judge in the gate. Base CLAP also clears the real crops, but it calls 41% of negatives footsteps, including 72.5% of wooden door knocks and 67.5% of water drops (`out_recalibrate.json`, `judges["clap_base|native|hit"]`). With that much over-crediting, base CLAP rated generated steps at 0.92 of real (`consensus.base` in the same file).

### The realism gap

Under CLAP-hard, generated 3 step footsteps sit far below real ones. The gate scores 15% of the 40 clips, a ratio of 0.17, and the gate fails (`out_gate_v2.json`). The recalibration scorer gives the same 40 seeds 25%, a ratio of 0.29 (`out_recalibrate.json`, `consensus.hard`). A fresh 80-seed run of the shipped prompt, at 3 steps with 6 s of headroom, scored 31%, a ratio of 0.36 (`out_score_hard_pfc.json`, cell `01110`). The two scoring paths disagree by 10 points on identical seeds, which is itself open.

None of the tested factors closes the gap. **Steps:** the CLAP-hard ratio is 0.17 at 1 step and 0.23 at plain 2 (n=20 each), 0.29 at 3, 0.31 for the fast tier and 0.20 at 8 (`out_recalibrate.json`, `consensus.hard`). **Attack:** fade-in ramps from 4 to 24 ms slowed the attack to 14.1 ms and moved the hard rate from 25% to at most 32.5%, best p=0.25 (`out_diag_attack.json`). **Prompt:** the 32-cell factorial spans 10% to 62.5% at n=40 (`out_score_hard_pf.json`). On confirmation at n=80, the best cell `11100` reached 49%, 0.56 of real, while the shipped prompt scored 31% (`out_score_hard_pfc.json`). The "clear distinct footstep" phrase looked strongest in the first pass (35% against 24%, p<0.0001) and reversed on confirmation (34% against 49%, p=0.088). Against the older enriched prompt `00110`, cell `11100` also lost base-CLAP footstep, 65% against 84%, p=0.0041 (`out_prompt_confirm.json`).

**Dryness is only partly ruled out.** Two synthetic rooms lowered the hard rate to 20% and 17.5%, a -40 dB noise floor raised it to 32.5%, and a 0.6 s RT60 room at 35% wet raised it to 52.5% (`out_diag_room.json`). The milder room also dropped real crops from 87.5% to 67.5%. The 52.5% cell is one unpaired run at n=40. **Open:** repeat it paired on fresh seeds before calling dryness irrelevant.

### Step tiers

Plain 2 steps collapses: base-CLAP footstep fell to 45% against 85%, +0/-16, p<0.001 (`out_fewstep_paired.json`). A constant timestep shift that maps t=0.5 to 0.68 (`mid_noise` in `stable_audio3.py`, called renoising in commit 4d90ac8) recovers it. At 1 s of headroom the fast tier scored 77.5% material and 85% footstep against 80% and 87.5% for 3 steps, +2/-3 and +1/-2, both p=1.000 at n=40 (`out_tiers_paired.json`). One step stays unusable, with a 6.7 dB crest and 0% CLAP-hard (`out_gate_v2.json`).

Eight steps scores worse than 3 with the current prompt: 60% material and 70% footstep, +5/-13 at p=0.096 and +0/-7 at p=0.016 (`out_tiers_paired.json`). At 6 s of headroom the same comparison gave 57.5% and 72.5%, p=0.078 and p=0.031. The tier probe repeated the sweep on 40 paired seeds (`out_tier_probe.json`). Euler at 4, 6 and 8 steps and pingpong at 4 and 8 all scored below euler at 3 on both base-CLAP footstep and material. The vendor recipe (pingpong, cfg 1, 8 steps) came out worst, with footstep 50% (+3/-15, p=0.008) and material 47.5% (+1/-14, p=0.001) against 80% and 80%. Commit 1513139 then moved the `quality` tier from euler at 8 steps to euler at 3, identical to `default` (`TIERS` in `scripts/run_sidecar.py`).

**Open: pingpong at 4 steps.** It doubled CLAP-hard from 15% to 30% (+11/-5, p=0.21) while material fell to 57.5% (p=0.078) in the same probe. One cell at n=40 is a lead, and the judge disagreement in this section applies to it.

**Open: water and splash.** `detail_keys` at HEAD still sends splash, water and liquid keys to 8 steps (`src/sarge/generators/stable_audio3.py`), at 166.9 ms for a water footstep (`out_fast_equiv.json`). Nothing measured favors those 8 steps. The tier probe ran four water keys at 3, 4 and 8 steps, 40 clips each, and CLAP ranked "water splashing" first among 10 labels for 1 of 40 clips at 3 steps and 0 of 40 at 4 and 8, with Audiobox PQ at 6.31, 6.39 and 6.39 (`out_tier_probe.json`). The 8-key identification test ran splash at 8 steps through `detail_keys` and matched 0 of 10 splash clips in all three of its configs (`out_check_pad.json`). No bench script runs the water label on real water recordings, so CLAP has no positive control for these keys and a 0% rate says little about the audio. Ears are the only judge here so far. Timings in that probe ran high (80.5 ms for the euler 3 step config that `out_gate_tiers.json` times at 67.6 ms), so latency figures in this plan come from the gate files.

---

## 4. Ranked paths

The August list, with what happened to each item:

| # | August item | Status | Measured result | Source |
|---|---|---|---|---|
| 1 | Cache T5 conditioning ("largest single latency cut") | done | 14 ms, bit identical; the graph cut 41 ms | `out_bench_graph.json` |
| 2 | cfg 2.0 to 1.0 | rejected | material 77.5% to 52.5%, p=0.006 | `out_fewstep_paired.json` |
| 3 | Per-stage timers | done | `model.generate` is 98.8% of the clip | `out_bench_stages.txt` |
| 4 | Batch the prefetch path | done at batch 2 (commit 48fd907) | now about 7% throughput per clip | `out_fast_equiv.json` |
| 5 | Autoencoder transient A/B | dropped | generated attacks are faster than real | `out_gate_tiers.json` |
| 6 | CUDA graph of the DiT | done | 139.4 to 98.3 ms, bit identical | `out_bench_graph.json` |
| 7 | Domain fine-tune or LoRA | open | its "76% ceiling" premise predates the judge correction; CLAP-hard puts generated steps at 0.17 to 0.36 of real | `out_gate_v2.json`, `out_score_hard_pfc.json` |
| 8 | Causal transient head | open, lower priority | first hit is 84.0 ms end to end | `out_e2e_sidecar.json` |

What is left, in the order I would run it:

**1. Hoist step-invariant work out of the graph (open, speed).** Each graphed step recomputes the cross-attention K and V (2.95 ms GPU), the local embedding (2.09 ms) and the conditioning embedding (0.14 ms) from inputs that never change between steps (`out_bench_graph.json`, `forward_probe.rows2`). That is 5.2 ms per step. It was measured at 76 frames, and the local embedding scales with latent length, so the saving at 22 frames is probably smaller. The output should stay bit identical, so `check_fast_equiv.py` can gate it.

**2. Decode only the kept second (open, speed).** At 76 frames, decoding the first 14 frames took 8.1 ms against 16.2 ms for all of them, with a 0.0017 maximum difference over the first second (`out_bench_graph.json`, `probes.decode`). At 22 frames the saving is unmeasured. The audio changes slightly, so it needs a paired gate.

**3. Pre-encode the enumerable prompts at startup (open, speed).** `CondCache` fills on first use (`src/sarge/generators/sa3_fast.py`). Encoding the known key set at startup should move the 14.4 ms conditioner cost off the first hit, if the 16 ms first-hit gap in section 1 is that cost.

**4. SA3 medium (open, realism).** This is the largest realism lever left, and nothing in the repo measures it yet.

**5. Paired retest of the reverb cell (open, realism).** See section 3.

**6. Domain fine-tune (open, realism).** T-Foley's 5.4 hours and about 5,000 clips is the August budget precedent (carried over). `data/real_footsteps` holds 40 ESC-50 footstep clips, far short of that.

`torch.compile` is off (`ENABLE_TORCH_COMPILE` "0" in `out_bench_graph.json`, `facts`) and stays off.

---

## 5. Explicitly do not do these

| Rule | Evidence |
|---|---|
| Keep cfg at 2.0 | cfg 1.0 lost CLAP material at 3 steps, 77.5% to 52.5%, p=0.006 (`out_fewstep_paired.json`) |
| Skip plain 2 steps and 1 step | plain 2 collapsed to 45% footstep, +0/-16; 1 step has a 6.7 dB crest (`out_fewstep_paired.json`, `out_gate_v2.json`) |
| Skip a second ARC or step distillation | small-sfx already carries a `training.arc` config and a separate base checkpoint exists (`model_config.json`, `stable_audio_3/model_configs.py`) |
| Leave SA3 streaming alone | 92.9 ms latent frames (`model_config.json`) |
| Keep non-commercial models out of the runtime | licenses as listed in section 1 (carried over) |
| Keep MeanAudio as a license escape hatch only | mono mel plus BigVGAN output (carried over) |
| Judge realism with CLAP-hard only | AST and PANNs fail real crops, base CLAP credits 41% of negatives (`out_gate_v2.json`, `out_recalibrate.json`) |
| Treat an n=40 tie as weak evidence of equal quality | McNemar at n=40 only catches large losses |

---

## 6. Files

Today's speed work lives in `src/sarge/generators/sa3_fast.py` (graph, T5 cache, `has_lora` patch, checkpointing off) and `src/sarge/generators/stable_audio3.py` (fast path, 1 s headroom, `mid_noise`, `detail_keys`). The footstep prompt is `build_prompt` in `src/sarge/generators/stable_audio.py`. The tier table is `scripts/run_sidecar.py`, and its `quality` tier became euler at 3 steps in commit 1513139.

The hoist would touch `sa3_fast.py`. At HEAD, prefix decoding has no call site in the repo: `stable_audio3.py` and `sa3_fast.py` never call decode, and the installed package decodes the full latent inside `generate` (`decode=not return_latents` in `stable_audio_3/model.py`), so that item needs a latent return path or a patch of the pretransform decode. The dead backend branch is in `scripts/model_server.py`. Two numbers in `scripts/run_sidecar.py` at HEAD predate today's measurements: `TIER_COST_MS` prices the default and quality tiers at 130 ms and the fast tier at 90 ms against a measured 67.6 and 47.8 ms, and the `--deepen-batch` help claims +17 ms of GPU hold for a pair, which matched the eager build (`out_bench_launch.json`) and is about 58 ms now (`out_fast_equiv.json`). `docs/unity-verification.md` still cites a 180 ms default tier.
