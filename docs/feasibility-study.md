# Predictive Neural SFX for Real-Time Audio
### Feasibility Study & Implementation Blueprint

*A technical assessment of a hybrid system that predicts game events, pre-generates sound effects with a neural audio model, caches them by context, and falls back to authored assets on a miss — “speculative prefetching, but for sound.”*

---

**Date:** 2026-06-20  
**Method:** Multi-agent research workflow — 8 technical dimensions researched in parallel against current (2023–2026) sources, 30 load-bearing claims independently adversarially fact-checked (26 corroborated / 0 refuted / 4 uncertain). All figures are cited inline with source URLs; where a fact-check corrected a researcher's number, the corrected value is used and noted in the text.  
**TL;DR:** Feasible **today** as an offline-baked neural variant library + cache + instant authored fallback on PC; feasible **near-future** as live speculative prefetch as few-step models and NPUs mature; **not** feasible as synchronous on-trigger neural synthesis or as a replacement for authored precise-synced transients.

---

## Contents

1. [Executive Summary & Feasibility Verdict](#1-executive-summary--feasibility-verdict)
2. [Technical Feasibility Deep-Dive](#2-technical-feasibility-deep-dive)
3. [System Architecture](#3-system-architecture)
4. [Engineering Deep-Dives](#4-engineering-deep-dives)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Risks, Evaluation & Open Questions](#6-risks-evaluation--open-questions)

---

## 1. Executive Summary & Feasibility Verdict

**Overall verdict: PARTIALLY FEASIBLE — and feasible TODAY, but only as a hybrid cache-and-fallback layer over a conventional authored-audio pipeline, not as live per-event neural synthesis.** The concept is sound precisely because it does not require the impossible thing (synthesizing a novel sound on the frame an event fires); it requires the achievable thing (predicting events early, generating ahead of time, caching by context, and falling back to authored assets on a miss).

### The core argument

The whole system rests on a single inequality. A neural text-to-SFX generator turns an event into audio in roughly **75 ms on a datacenter H100** (Stable Audio Open Small, 341M params, 8-step ARC sampling, ~12 s of 44.1 kHz stereo) ([arXiv 2505.08175](https://arxiv.org/abs/2505.08175)), **~130 ms for a 10 s clip on a consumer RTX 3090** (MeanAudio, 1-NFE, RTF 0.013) ([arXiv 2508.06098](https://arxiv.org/abs/2508.06098)), and **~7–8 s on an Arm mobile CPU** ([Arm/Stability](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/audio-generation-arm-cpus-stable-audio-open-small-kleidiai)). A game audio frame, by contrast, must be filled in **~2.7–10 ms** (128–512-sample buffers) and the perceptual action-to-sound window is only **~10–40 ms**, with AV-sync tolerating at most **~45 ms of audio lead / ~185 ms of lag** ([ITU-R BT.1359](https://www.tvtechnology.com/opinions/updating-lip-sync-issues); [Jack et al. 2018](https://online.ucpress.edu/mp/article/36/1/109/92063/Action-sound-Latency-and-the-Perceived-Quality-of)). Generation is therefore **1–3 orders of magnitude too slow to invoke synchronously** — confirmed across every model surveyed, all of which are one-shot, not streaming.

Prediction plus caching closes this gap *only if* two conditions hold simultaneously:

1. **Lead time ≥ generation + decode latency.** Available lead from game signals is modest — animation anticipation is ~4–8 frames (**~100–300 ms**), physics time-to-impact and input windows give tens to a few hundred ms ([gameanim](https://www.gameanim.com/2019/05/15/the-12-principles-of-animation-in-video-games/)). This comfortably hides ~75–130 ms of GPU generation, but **cannot hide cold generation for genuinely novel/reactive events** — so prediction's real job is *cache pre-warming over seconds*, not just-in-time synthesis.
2. **Cache hit rate is high enough.** A hit replays a pre-decoded PCM buffer in near-zero time; a miss **must** fall back to an authored asset instantly, because on-the-fly regeneration is impossible. Analogous semantic/predictive caches hit only **~20–68%** ([GPTCache](https://aclanthology.org/2023.nlposs-1.24/)), so the authored pipeline is permanent, not optional.

When both hold, a seconds-latency generator becomes a real-time system. When either fails, it degrades gracefully to today's authored audio — never worse.

### Bottom line up front

- **Feasible today (PC, dedicated/second GPU):** Offline-baked neural variant libraries keyed by (event, material, context, intensity), played through existing Wwise/FMOD random containers with avoid-repeat and pitch jitter. This is systems integration of proven parts — runtime PCM injection is a first-class API in Wwise, FMOD, Unreal, and Unity ([Unity PCMReaderCallback](https://docs.unity3d.com/ScriptReference/AudioClip.PCMReaderCallback.html)) — **not new research**.
- **Feasible near-future (live speculative prefetch):** Few-step models (ARC 8-step, MeanAudio 1-NFE) generate fast enough that *wasted* speculation is cheap, enabling a high-recall/modest-precision strategy mirroring speculative decoding's ~60%-acceptance regime. The binding constraint is **GPU co-residency**, not model speed: sharing one consumer GPU with rendering causes unpredictable interference (~1–35%, no robust graphics+compute preemption) ([MJP](https://therealmjp.github.io/posts/breaking-down-barriers-part-6-experimenting-with-overlap-and-preemption/)), so generation must be confined to idle frames, a second device, or an NPU.
- **Quality-limited scope:** Neural SFX is best for **ambiences, textures, tails, and variation layers** — loosely-timed material outside the ~90/185 ms sync budget. It is **not yet shippable for precise synced transients** (footsteps, gunshots, impacts), where models under-render attacks and a single "odd" variation breaks immersion; a measurable perceptual gap to a sound-designer reference persists ([DCASE 2024](https://arxiv.org/abs/2501.08587)).
- **Mobile/console: weakest target.** On-device generation is multi-second per clip ([Arm/Stability](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/audio-generation-arm-cpus-stable-audio-open-small-kleidiai)); console fixed memory and certification make shipping a multi-GB model on-device near-term impractical. Viable only via load/idle pre-baking or cloud-assisted prefetch.
- **Scope is cosmetic-only.** Neural output is non-deterministic and high-latency, so it cannot be authoritative in multiplayer/replay/competitive netcode — it must be a client-side cosmetic layer with authored audio as the canonical, synced source ([FP non-associativity](https://arxiv.org/pdf/2408.05148)).
- **Novelty is the predictor, not the mechanism.** Predict-pregenerate-cache-fallback is mature in LLM serving and already patented for game audio prefetch (US 11,632,645). The project's real contribution — and its make-or-break risk — is whether game state predicts (event, material, context) accurately and early enough to deliver a useful hit rate.

**In one sentence:** this is a *predictive content pre-baking and variation system* that wraps a slow neural generator behind a game-state predictor with an instant authored fallback — feasible and useful for non-transient SFX on PC today, increasingly viable for live prefetch as few-step models and NPUs mature, and impractical only if framed as real-time neural *replacement* of the authored pipeline.

---

## 2. Technical Feasibility Deep-Dive

The entire concept hinges on one number: how long neural audio generation takes versus how much warning the game can give before a sound must play. This section works that math out concretely, identifies which models clear the bar today, quantifies what optimization realistically adds, and confronts the constraint that ends up dominating every other one — sharing a single consumer GPU with the renderer.

### The latency math: why synchronous generation is dead on arrival

A game audio frame is brutally short. At 60 fps the frame budget is 16.6 ms (8.3 ms at 120 fps), and the audio render callback itself must return a filled buffer within roughly 2.7 ms (128 samples @ 48 kHz) to ~10.7 ms (512 samples), with hard no-allocation, no-lock constraints ([Bencina](http://www.rossbencina.com/code/real-time-audio-programming-101-time-waits-for-nothing)). Worse, the default pipeline already spends most of the tolerance budget before any neural step: FMOD's default mixer is 1024 samples × 4 buffers ≈ 85 ms worst case / ~53 ms average ([FMOD docs](https://documentation.help/fmod-studio-api/FMOD_System_SetDSPBufferSize.html)). Because the just-noticeable difference for *added* latency from a non-zero base is only ~27 ms ([Schmid et al.](https://dl.acm.org/doi/10.1145/3678299.3678331)), the cached-playback path must add near-zero latency.

Against this, even the fastest neural generator is hopelessly slow on-trigger. The state of the art is Stability AI's Stable Audio Open Small (341M-param DiT, ARC adversarial post-training, 8 steps), which produces ~12 s of 44.1 kHz stereo in **~75 ms on an H100** ([ARC paper](https://arxiv.org/abs/2505.08175)). That is the *best case*, and it is still ~5–50× a single frame. The fastest research one-shot model, MeanAudio (1-NFE mean-flow, RTF 0.013 on an RTX 3090), generates a 10 s clip in ~130 ms — roughly 13–65 ms for a short 1–5 s SFX ([MeanAudio](https://arxiv.org/html/2508.06098)). On-device the picture collapses: the same Stable Audio model takes ~6.6 s (int8) to ~8 s for an 11 s clip on an Armv9 smartphone CPU ([Arm/KleidiAI](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/audio-generation-arm-cpus-stable-audio-open-small-kleidiai)). The action-to-sound perceptual window is ~10–40 ms ([Jack et al.](https://online.ucpress.edu/mp/article/36/1/109/92063/Action-sound-Latency-and-the-Perceived-Quality-of)) and AV-sync tolerates audio leading video by at most ~45 ms ([ITU-R BT.1359](https://www.tvtechnology.com/opinions/updating-lip-sync-issues)). The gap between generation and budget is therefore ~1 order of magnitude even at the bleeding edge and 2–3 orders for production models. **Synchronous on-trigger neural generation is physically impossible; the cache is mandatory, not an optimization.**

This forces a feasibility inequality. For a sound to be ready when its event fires:

> **lead_time ≥ generation_latency + decode_latency + normalization_latency**, OR the entry is already cache-resident.

The problem is that available lead time is small. Animation anticipation for quick reactions is only ~6–8 frames (~100–130 ms at 60 fps) ([gameanim](https://www.gameanim.com/2019/05/15/the-12-principles-of-animation-in-video-games/)); physics time-to-impact and input-to-event windows give tens to a few hundred ms. Since cold generation is ~1–4 s for mainstream models plus seconds of cold-start, **lead time alone cannot hide generation of a genuinely novel event.** The predictor's job is therefore not just-in-time generation but *cache pre-warming over seconds* — and any cache miss must fall back to an authored asset instantly, because it cannot be regenerated in time.

One correction to a common framing: the worst-case scheduling claim is narrower than it looks. Continuous-collision detection gives time-of-impact only ~1 frame ahead (~16.7 ms; physics sub-steps ~2–3 ms), *not* seconds — TOI is solved within the current timestep. The genuinely deterministic, ML-free, ~100%-precision subset is essentially footsteps on the already-playing locomotion clip (lead ~100–400 ms within a stride), not "a large fraction" of SFX as sometimes claimed.

### Which models are viable today

| Model | Params | Speed | Role |
|---|---|---|---|
| Stable Audio Open Small (ARC) | 341M | ~75 ms / 12 s clip (H100); ~6.6–8 s on-device | Primary generator (offline/prefetch) |
| MeanAudio (1-NFE) | 120M | RTF 0.013 (RTX 3090), ~130 ms / 10 s | Fast generator, research-grade |
| TangoFlux (flow matching) | 515M | 30 s in ~3.7 s (A40) | Quality/throughput alternative |
| AudioLDM2 / Tango2 / AudioGen | — | RTF ~1.25–2.19 (AR is the bottleneck) | Too slow — avoid |
| EnCodec / Vocos / HiFi-GAN | small | 13 ms latency / 5.8–169× RT on CPU | Real-time decode/variation layer |
| BRAVE (RAVE redesign) | 4.9M | ~10 ms latency, ~3 ms jitter, RTF 0.29 (M1 Pro) | Real-time conditioning synth |

Two architectural truths emerge. First, every competitive end-to-end SFX generator is **one-shot, not streaming** — latency is paid in full per clip, so there is no "first audio" to stream out early. Second, the only models that genuinely meet the <10–20 ms budget on CPU (RAVE/BRAVE, DDSP, the neural codecs) are **transforms of a live input or control stream**, not generators of a novel sound from an abstract event label ([BRAVE](https://fcaspe.github.io/brave/)). RAVE itself runs ~20× real-time on CPU but carries 244–440 ms end-to-end latency; only the causal BRAVE redesign reaches sub-10 ms. This mandates a strict two-tier split: the heavy generator runs ahead of time into the cache, and a lightweight codec/vocoder handles real-time playback and cheap variation of cached material.

### What optimization realistically buys

The acceleration toolkit is mature but yields multipliers, not orders of magnitude beyond what step-reduction already gave. Step reduction (diffusion → 4–8-step adversarial/consistency/mean-flow) is the big win — ARC's ~100× over the base model, MeanAudio's ~100× over diffusion. On top of that, int8/fp8 quantization via TensorRT adds ~35–45% on SDXL-class workloads with near-FP16 quality ([NVIDIA](https://developer.nvidia.com/blog/nvidia-tensorrt-model-optimizer-v0-15-boosts-inference-performance-and-expands-model-support/)); KleidiAI/ExecuTorch delivered ~30× on Arm CPU versus a >4-minute FP32 baseline. The honest caveat: quantization degrades high-frequency and transient detail first ([Arm](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/audio-generation-arm-cpus-stable-audio-open-small-kleidiai)) — exactly where impact/contact SFX live — so transient-heavy sounds may need fp16 while ambiences tolerate int8. None of this brings on-device generation under the per-event budget; it brings a multi-second clip down to *seconds*, which is only useful for ahead-of-time fill, not on-trigger synthesis.

### The GPU-contention problem

This is the binding constraint for any on-the-player's-machine deployment, and it is *not* model speed. Sharing one consumer GPU between the renderer and a generation burst is structurally hostile on NVIDIA hardware: graphics and compute do not overlap (there is an implicit sync point per dispatch/draw), and preemption is coarse-grained at dispatch/batch boundaries with multi-millisecond switch latency in the worst case ([MJP](https://therealmjp.github.io/posts/breaking-down-barriers-part-6-experimenting-with-overlap-and-preemption/)). MPS-style spatial sharing produces unpredictable interference — measured roughly 1–35% latency degradation depending on contention, with cache and memory-controller resources not isolated even under SM partitioning. There is no production-grade fine-grained graphics+compute preemption. A multi-hundred-ms generation burst will therefore cause visible frame stutter unless it is (a) confined to idle frames under a strict per-frame compute budget, (b) offloaded to a second device or NPU, or (c) precomputed at load/idle time. Dual-GPU frame-generation tests confirm the mitigation works: offloading AI work to a second GPU keeps the render GPU at ~97% utilization. VRAM is a second pressure point — a 341M fp16 model carries a ~4 GB working set that competes with textures and the render pipeline on 8 GB consumer cards.

Crucially, fast generation flips the problem from *latency-bound* to *throughput/scheduling-bound*. At RTF 0.013, a short SFX costs tens of ms of GPU time, so a wasted speculation is cheap — the system can run high-recall/modest-precision prefetch (mirroring speculative decoding's tolerance of ~60% draft acceptance) provided it adopts hard concurrency caps, à la web speculation rules, to stop over-speculation from starving the frame ([ConsumerBench](https://arxiv.org/abs/2506.17538) shows uncapped concurrent inference can collapse a co-running game's FPS).

**Verdict:** feasible *today* as an offline/idle precompute + cache + fast-playback system on a capable or dedicated GPU; infeasible as live per-frame neural synthesis; weakest on mobile/console, where generation is multi-second per clip and fixed memory rules out a resident large model — pushing those targets toward pre-session baking or cloud-assisted prefetch.

---

## 3. System Architecture

```
GAME / SIM THREAD  (sub-millisecond)
  player velocity | surface material | animation state | physics events
        |
        v   feature frame + cache key = (event, material, context, intensity)
+============== SLOW LANE :: ahead of time, GPU / NPU, 100s of ms - seconds ==============+
|  (1) GAME-STATE OBSERVER  -> normalized features + discrete key                          |
|         |                                                                               |
|         v                                                                               |
|  (2) PREDICTION ENGINE  -> speculative request (key, predicted_fire_time, confidence)    |
|         Tier-1 deterministic (anim-notify, time-of-impact)                               |
|         Tier-2 ML intent  (~70-83% acc, ~200 ms lead)                                     |
|         |                                                                               |
|         v                                                                               |
|  (3) GENERATION SERVICE  (out-of-process; few-step latent model; per-frame GPU budget)    |
|         | normalized PCM (-23 LUFS), routed through the shared DSP / reverb / HRTF chain  |
|         v                                                                               |
|  (4) MULTI-TIER CACHE   [ offline-baked variant library ]  +  [ runtime warm cache ]      |
+=========================================+===============================================+
                                          |  key HIT -> pre-decoded PCM buffer
==========================================+================================================
                                          v
+============== FAST LANE :: real time, audio thread, ~2.7-10 ms deadline ===============+
|  (5) MIXER / PLAYBACK   (lock-free SPSC ring buffer; no alloc, no locks)                 |
|        on event:  play AUTHORED asset immediately                                        |
|                   +- if cached neural ready in time -> constant-power crossfade           |
|                   +- else -> authored only   (worst case == today's quality)             |
|        |                                                                                |
|        v                                                                                |
|  (6) MIDDLEWARE BOUNDARY   Wwise Audio Input | FMOD OPENUSER | Unreal QueueAudio | Unity |
|        scope: client-side COSMETIC ONLY -- authored audio stays canonical & synced       |
+==========================================================================================+
```

The system is a strict two-lane pipeline. A **slow, ahead-of-time generation lane** (GPU/NPU, hundreds of ms to seconds) populates a cache; a **real-time playback lane** (audio thread, single-digit ms) only ever reads from that cache or from authored fallback assets. This split is forced by the latency physics: even the fastest 2025-2026 text-to-SFX model — Stable Audio Open Small with ARC adversarial post-training — generates ~12 s of 44.1 kHz stereo in ~75 ms on an H100 ([arXiv 2505.08175](https://arxiv.org/abs/2505.08175)), but ~6.6-7 s on an Arm mobile CPU ([Arm/KleidiAI](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/audio-generation-arm-cpus-stable-audio-open-small-kleidiai)), while the audio render callback must return a buffer within ~2.7 ms (128 samples @ 48 kHz) to ~10-20 ms and may not block or allocate ([Bencina](http://www.rossbencina.com/code/real-time-audio-programming-101-time-waits-for-nothing)). Synchronous on-trigger neural generation is therefore impossible by 1-3 orders of magnitude; the cache is mandatory, not an optimization. The diagram above shows the full data flow.

### 1. Game-State Observer / Feature Extractor

**Runs on:** game/simulation thread, sub-millisecond. **In:** raw engine signals — player velocity, surface/material under contact, animation state, physics broad/narrow-phase events. **Out:** a normalized feature frame plus a discrete cache key `(event_type, material, context-bucket, intensity-bucket)`.

This keying scheme is not novel research: shipped games already structure audio along exactly these axes — Wwise Switch Containers keyed on surface material and FMOD labeled parameters per surface type, with continuous intensity driven by RTPC and quantizable into discrete buckets ([Audiokinetic](https://www.audiokinetic.com/en/blog/footsteps-material-management-using-wwise-unreal-engine-4-unity-3d/)). Because the dimensions are already discrete in practice, the key space is naturally finite and bucketable, which bounds cache size. The observer's only hard constraint is that feature extraction stays off the audio thread and adds no measurable cost to the frame.

### 2. Prediction Engine (Speculation)

**Runs on:** a worker thread, fired from observer frames. **In:** the feature stream. **Out:** speculative generation requests `(cache_key, predicted_fire_time, confidence)`.

This is the genuine novelty and the load-bearing risk. The mechanism — predict, pre-generate, cache, fall back — is mature elsewhere (LLM speculative decoding, web Speculation Rules, and an existing game-audio prefetch patent, [US 11,632,645](https://patents.google.com/patent/US11632645B2/en)), so feasibility rests on hit-rate, not on inventing a paradigm. A two-tier predictor is appropriate:

- **Tier 1 (deterministic):** anim-notify positions on the *currently playing* locomotion clip give footstep timing sub-second ahead, since notify positions are static baked asset data ([techarthub](https://techarthub.com/working-with-anim-notifies-in-unreal-engine-5/)). Continuous-collision time-of-impact, contrary to a common overstatement, yields only ~1 frame (~16.7 ms) of lead because TOI is solved within the current physics step ([fact-check correction]) — useful for confirming an impact, not for hiding seconds of generation.
- **Tier 2 (ML):** movement/gaze intent models reach ~70-83% accuracy with ~200-240 ms lead ([PMC10248176](https://pmc.ncbi.nlm.nih.gov/articles/PMC10248176/)).

Crucially, available lead time is only ~100-300 ms (animation anticipation 4-8 frames; reaction windows 200-250 ms), which is **shorter than cold generation time** (~1-4 s for mainstream models; ~75-130 ms only for bleeding-edge few-step models on a strong GPU). The predictor's real job is therefore **pre-warming the cache seconds ahead** for predictable contexts, not just-in-time generation. The reactive tail (sudden reversals, surprise contacts) is high-entropy and unpredictable in every analogue — branch mispredict, rollback rubber-banding, prefetch waste — which is precisely why an authored fallback is non-optional ([MICRO 2024](https://dl.acm.org/doi/10.1109/MICRO61859.2024.00042)).

### 3. Generation Service (Model Serving)

**Runs on:** GPU/NPU, out-of-process worker. **In:** batched speculative requests. **Out:** rendered, normalized PCM written into the runtime cache.

Anchor on a sub-0.5B few-step latent model — Stable Audio Open Small (341M, 8-step) or MeanAudio (120M, 1-NFE, RTF 0.013 on an RTX 3090 ≈ 130 ms for a 10 s clip, [arXiv 2508.06098](https://arxiv.org/abs/2508.06098)). Because a wasted speculation now costs only tens-to-low-hundreds of ms of GPU throughput rather than diffusion-grade seconds, the system can run **high-recall / modest-precision**, mirroring speculative decoding's tolerance of ~60% acceptance. The binding constraint is **co-residency**, not raw speed: on NVIDIA, graphics and compute do not overlap (implicit sync per dispatch), preemption is coarse, and MPS sharing inflates latency unpredictably (~1-35%, ~20% mid-range), so a generation burst risks visible frame stutter ([gopebble](https://www.gopebble.com/case-studies/nvidia-mps-vs-dedicated-gpu-allocation-for-llm-inference/), [MJP](https://therealmjp.github.io/posts/breaking-down-barriers-part-6-experimenting-with-overlap-and-preemption/)). Mitigation: confine inference to idle-frame slack with a strict per-frame compute budget, adopt hard concurrency caps modeled on web Speculation Rules eagerness tiers ([Chrome](https://developer.chrome.com/blog/speculation-rules-improvements)), or offload to a second device/NPU. Mobile/console generation remains multi-second per clip even at int8 (~1.4x real-time for an 11 s clip on Armv9), so those platforms lean almost entirely on offline/at-load pre-baking.

### 4. Multi-Tier Cache

**Runs on:** system RAM (hot PCM) backed by disk. **In:** generated and offline-baked buffers. **Out:** decoded PCM on a key hit.

Two tiers: an **offline-baked neural variant library** (the safe, shippable 90% — N variants per key, perceptually de-duplicated at bake time with CLAP cosine similarity, human-curated) and an **optional runtime warm cache** filled by the prediction engine. Memory is in-budget: shipped footstep libraries already ship 0.7-1.3 GB of WAVs ([Krotos](https://www.krotosaudio.com/products/ultimate-footsteps-sound-effects-library/)), so a few-hundred-MB cache is unremarkable. Repetition fatigue is mitigated with proven, cheap tooling — Wwise/FMOD shuffle-bags and avoid-repeat, plus ±semitone pitch and granular jitter on cached seeds ([FMOD](https://www.fmod.com/docs/2.03/studio/working-with-instruments.html)). Stored entries are **rendered waveforms keyed deterministically**, so a hit replays a byte-identical buffer with zero regeneration drift. Realistic semantic-cache hit rates are modest (~20-68%, not the often-quoted ~95% which is match-accuracy-on-hit), reinforcing that the offline tier carries most of the value.

### 5. Mixer / Playback with Fallback

**Runs on:** the real-time audio thread (~2.7-20 ms deadline, lock-free, no allocation). **In:** PCM from cache or authored assets. **Out:** mixed buffer to middleware. Generated buffers reach this thread via a single-producer/single-consumer ring buffer; the model is never called here. On every event the **authored asset is triggered immediately**, and the engine constant-power-crossfades to the generated buffer only if it is cache-ready in time — guaranteeing the worst case equals today's quality with zero audible miss. Generated voices are loudness-normalized to ~-23 LUFS (ITU-R BS.1770 / Sony ASWG-R001) and routed through the identical bus/reverb/HRTF/HDR chain as authored audio so they share spectral and spatial character ([Audiokinetic HDR](https://www.audiokinetic.com/en/blog/wwise-hdr-overview-and-best-practices-for-game-mixing/)).

### 6. Audio Middleware Boundary

Runtime PCM injection is a first-class, documented API in every major engine — Wwise Audio Input plugin, FMOD `FMOD_OPENUSER` + `FMOD_SOUND_PCMREAD_CALLBACK`, Unreal `USoundWaveProcedural::QueueAudio()`, Unity `AudioClip.Create` with `PCMReaderCallback` ([Unity](https://docs.unity3d.com/ScriptReference/AudioClip.PCMReaderCallback.html), [FMOD](https://documentation.help/FMOD-API/FMOD_SOUND_PCMREAD_CALLBACK.html)). Integration is the lowest-risk dimension. The boundary also enforces scope: because neural generation is non-deterministic across hardware and high-latency, generated SFX must be **client-side cosmetic-only** and never gate netcode, replays, or competitive cues, with authored audio as the canonical synced layer.

---

## 4. Engineering Deep-Dives

The feasibility verdict for predictive neural SFX hinges on four engineering problems. Across all four, one architectural fact is load-bearing and confirmed: synchronous on-trigger neural generation is impossible within a game's audio budget. Even the fastest 2025–2026 generator, Stable Audio Open Small (341M params, ARC adversarial post-training, ~8 steps), produces ~12s of 44.1 kHz stereo in **~75 ms on an H100** but **~6.6–8 s on an Arm mobile CPU** ([arXiv 2505.08175](https://arxiv.org/abs/2505.08175)), against an audio-callback deadline of ~2.7 ms (128 samples @ 48 kHz) to ~10–20 ms and a 60 fps frame of 16.6 ms ([Bencina](http://www.rossbencina.com/code/real-time-audio-programming-101-time-waits-for-nothing)). The whole system is therefore a *speculative pre-bake + cache + instant authored fallback* pipeline, not real-time synthesis.

### A. Prediction strategy

Build a **two-tier predictor**. Tier-1 is deterministic and ML-free: read anim-notify/foot-sync-marker positions, which are baked static data on the animation asset and queryable ahead of the playhead, plus physics time-of-impact. This is the cheapest, highest-precision signal — but with an important corrected caveat: continuous collision detection (CCD/TOI) yields only **~1 frame of lead (~16.7 ms at 60 fps)**, not seconds, because TOI is solved within the current physics timestep; and "upcoming clip" notifies are reactive (motion-matching re-searches several times/sec), so their lead is bounded by blend/transition time. The reliably schedulable set is narrower than "a large fraction" — chiefly footsteps on the *already-playing* locomotion clip, with sub-second lead (~100–400 ms within a stride). Tier-2 is cheap ML on velocity, locomotion state, AI/enemy state, gaze, and the input buffer for telegraphed-but-unscheduled events; movement/gaze intent models land **~70–83% accuracy with ~200–240 ms median lead** ([PMC10248176](https://pmc.ncbi.nlm.nih.gov/articles/PMC10248176/)).

The economics now favor **high recall, modest precision**. A wasted speculation costs only tens-to-low-hundreds of ms of GPU throughput rather than diffusion-grade seconds: MeanAudio reaches RTF 0.013 on an RTX 3090 (~130 ms for a 10s clip; ~13 ms for a 1s clip, before fixed text/vocoder overhead) ([arXiv 2508.06098](https://arxiv.org/abs/2508.06098)). This mirrors speculative decoding, which tolerates ~60% draft acceptance and still wins because rejected drafts are cheaply discarded ([NVIDIA](https://developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x/)).

**Speculation must be budgeted with hard caps**, borrowing the web Speculation Rules model (immediate eagerness = 50 prefetch / 10 prerender; conservative = 2 / 2, FIFO eviction) ([Chrome](https://developer.chrome.com/blog/speculation-rules-improvements)). Uncapped speculation on a shared GPU is dangerous: concurrent inference can saturate the submission queue and degrade a co-running app's latency up to ~12.4× ([ConsumerBench, arXiv 2506.17538](https://arxiv.org/abs/2506.17538)). Default conservative; escalate eagerness per event-class by measured hit-rate. Two further controls: **gate playback on a confirmation signal** (the event actually fired) so a generated-but-unconfirmed SFX is silently discarded rather than producing a phantom sound; and accept that the **reactive tail** (sudden reversals, surprise contacts) is high-entropy and irreducible — every analogue (branch mispredict 3.6–20%, rollback rubber-banding) confirms it ([MICRO 2024](https://dl.acm.org/doi/10.1109/MICRO61859.2024.00042)). Authored fallback is mandatory, not optional.

### B. Cache design

The cache key `(event_type, material, context-bucket, intensity-bucket)` maps almost 1:1 onto how shipped games already structure audio — Wwise Switch Containers keyed on material/state, RTPC for intensity ([Audiokinetic](https://www.audiokinetic.com/en/blog/footsteps-material-management-using-wwise-unreal-engine-4-unity-3d/)). Bucketing continuous params into discrete switches is a documented, low-risk feature, so the key space is naturally finite. (Caveat: intensity is often a *continuous* blend/RTPC crossfade in practice, so bucketing it is a design choice, not the universal pattern — interpolate at bucket boundaries to avoid a soft step using a hard-impact variant.)

**Memory is in-budget.** Shipped footstep libraries already ship 1,655–8,000+ WAVs at 0.77–1.3 GB across ~20 surfaces ([Krotos](https://www.krotosaudio.com/products/ultimate-footsteps-sound-effects-library/)), so a cache of a few hundred MB of pre-decoded PCM/Vorbis variants per event class is comfortably within norms. Store a **rendered waveform per key** (not a re-run prompt): a cache hit replays a byte-identical buffer with zero regeneration drift, which neutralizes generator stochasticity as a consistency concern.

**Anti-repetition** is solvable with proven, cheap techniques layered together: avoid-repeat / shuffle-bag playback (Wwise/FMOD remove a clip from the pool until exhausted) ([FMOD](https://www.fmod.com/docs/2.03/studio/working-with-instruments.html)); ±2–3 semitone pitch and ±5–10% time jitter via granular re-grain on cached PCM; and single-seed variation models (SpecSinGAN, [arXiv 2110.07311](https://arxiv.org/abs/2110.07311)) to expand one seed into many "different takes." De-duplicate at **bake time** with CLAP-embedding cosine similarity to maximize perceptual diversity per MB. Eviction: keep the offline tier a fixed pre-sized budget; apply GPTCache-style cost-aware eviction only on the runtime tier, keeping hot buckets resident. Set realistic expectations on the runtime tier — analogous semantic caches hit only ~20–68% ([GPTCache](https://aclanthology.org/2023.nlposs-1.24/)), and the corrected LLM-serving figure is that "~95%" claims are *match-accuracy-on-hit*, not hit-rate. Most value comes from the offline-baked library.

### C. Model optimization & serving

Standardize on a **sub-0.5B latent model with few-step generation**: Stable Audio Open Small (ARC 8-step) or MeanAudio (1-NFE), with TangoFlux (515M flow matching, 30s in ~3.7s on an A40) as a quality alternative ([TangoFlux](https://tangoflux.github.io/)). Treat AudioLDM2/Tango2/AudioGen (RTF >1) as too slow for any near-real-time path. Apply quantization per backend: TensorRT int8/fp8 gives ~35–45% speedup near FP16 quality on desktop NVIDIA ([NVIDIA](https://developer.nvidia.com/blog/nvidia-tensorrt-model-optimizer-v0-15-boosts-inference-performance-and-expands-model-support/)); KleidiAI/ExecuTorch delivered a **30× speedup on Arm CPUs** (>4 min → ~6.6 s, 5.2 GB → 2.9 GB, RAM 6.5 GB → 3.6 GB) ([Arm](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/audio-generation-arm-cpus-stable-audio-open-small-kleidiai)). Validate quantization perceptually per event class: high-frequency/transient content degrades first under reduced bitwidth, so keep fp16 for impact/transient sounds and reserve int8 for ambiences and loops.

The binding constraint is **GPU co-residency with rendering**, not raw speed. On NVIDIA, graphics and compute do not overlap (implicit sync per dispatch/draw) and preemption is coarse-grained; MPS sharing produces workload-dependent interference roughly **1–35%** (≈20% mid-range) with no isolated memory subsystem ([MJP](https://therealmjp.github.io/posts/breaking-down-barriers-part-6-experimenting-with-overlap-and-preemption/)). Therefore: never run inference in the audio callback; confine generation to a strict **per-frame GPU time budget** (run only when the frame has headroom), to loading screens and lulls, or offload to a second GPU/NPU. Measure 99th-percentile frame-time under concurrent render+generate before committing. The fast layer is a separate concern: a real-time neural codec/vocoder handles playback and cheap variation of cached latents — HiFi-GAN ~13× real-time on CPU, Vocos ~169× on 8-core CPU, EnCodec 13.3 ms streaming latency ([arXiv 2306.00814](https://arxiv.org/abs/2306.00814)). Note BRAVE-class streaming synthesis (<10 ms latency, ~3 ms jitter, RTF 0.29 on M1 Pro) is a *transform of a live input*, not label-driven generation — it fits the variation layer, not the generator ([arXiv 2503.11562](https://arxiv.org/abs/2503.11562)).

### D. Integration & mixing

Buffer injection is a **solved, first-class API** in every major engine: Wwise Audio Input plugin (the raw-PCM path; *not* External Sources, which swaps pre-encoded WEM files), FMOD `FMOD_OPENUSER` + `FMOD_SOUND_PCMREAD_CALLBACK`, Unreal `USoundWaveProcedural::QueueAudio()` (the native path; Runtime Audio Importer is third-party), and Unity `AudioClip.Create` with `PCMReaderCallback` ([Unity](https://docs.unity3d.com/ScriptReference/AudioClip.PCMReaderCallback.html)). Hand finished, pre-normalized buffers from the inference worker to the audio thread via a **lock-free single-producer/single-consumer ring buffer**; never allocate or lock on the render thread ([USPTO 8166194](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8166194)).

**Glitch-free fallback:** trigger the authored asset immediately on every event, and only **constant-power crossfade** to the generated buffer if it is cache-ready in time. Constant-power (sin/sqrt curves crossing at 0.707) holds perceived loudness constant through the transition, so the worst case degrades gracefully to today's authored quality. For consistency, normalize every generated clip to the project loudness target (**−23 LUFS ±2 per ITU-R BS.1770 / Sony ASWG-R001** — note −24 LKFS is US broadcast, not the game standard) and route generated voices through the *identical* bus, DSP chain, reverb sends, HRTF/spatializer, and HDR/ducking as authored audio, governed by the same priority/virtual-voice/playback-limit system ([Audiokinetic HDR](https://www.audiokinetic.com/en/blog/wwise-hdr-overview-and-best-practices-for-game-mixing/)).

Two hard scope boundaries close the section honestly. First, neural generation is non-deterministic across heterogeneous clients (GPU floating-point non-associativity defeats even fixed-seed reproducibility, [arXiv 2408.05148](https://arxiv.org/pdf/2408.05148)) and high-latency, so generated SFX must be **client-side cosmetic-only** — never authoritative in multiplayer, replay, or competitive cues. Second, on quality, scope neural generation to **ambiences, textures, tails, and variation layers** mixed *under* authored transients, and keep authored assets authoritative for precise synced one-shots (footstep contacts, gunshots, impacts, UI). Models under-render transients ([arXiv 2309.06649](https://arxiv.org/abs/2309.06649)), and sync tolerance is tight and asymmetric (~90 ms early / ~185 ms late acceptability per ITU-R BT.1359, [TV Tech](https://www.tvtechnology.com/opinions/updating-lip-sync-issues)). Finally, automated quality gates are unreliable — FAD is embedding-dependent (VGGish Spearman ρ<0.1 vs PANNs >0.5, [arXiv 2403.17508](https://arxiv.org/abs/2403.17508)) and CLAP measures semantics, not fidelity — so the play-vs-fallback decision should rest on **bake-time human curation** plus domain-matched embeddings, not a runtime FAD/CLAP threshold.

---

## 5. Implementation Roadmap

This roadmap takes the system from an offline proof to a shipping feature in five phases. Each phase is structured to **retire the single biggest risk** before spending effort on the next. The ordering reflects the central feasibility finding: synchronous on-trigger neural generation is impossible inside a game-audio frame (the fastest model, Stable Audio Open Small, needs ~75 ms on an H100 and ~7-8 s on an Arm mobile CPU for a multi-second clip — orders of magnitude above the ~2.7-16 ms callback budget ([arXiv 2505.08175](https://arxiv.org/abs/2505.08175))). The whole system is therefore a *predictive content-pre-baking pipeline with an authored fallback*, not real-time synthesis. We de-risk the cheap, certain parts first (cache, plumbing, fallback) and treat live speculative generation — the riskiest, most overhyped part — last.

### Recommended starter stack

| Layer | Choice | Why |
|---|---|---|
| Engine | **Unreal 5.x** (MetaSounds) or Unity | First-party runtime PCM injection: `USoundWaveProcedural::QueueAudio()` / `OnGeneratePCMAudio()` ([Epic docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/metasounds-the-next-generation-sound-sources-in-unreal-engine)); Unity via `AudioClip.Create` + `PCMReaderCallback` ([Unity](https://docs.unity3d.com/ScriptReference/AudioClip.PCMReaderCallback.html)) |
| Middleware | **Wwise** (Audio Input plugin) or **FMOD** (`FMOD_OPENUSER` + `FMOD_SOUND_PCMREAD_CALLBACK`) | Switch containers key on material/state, RTPC on intensity — maps 1:1 to the cache key ([Audiokinetic](https://www.audiokinetic.com/en/blog/footsteps-material-management-using-wwise-unreal-engine-4-unity-3d/)); shuffle/avoid-repeat fights fatigue ([FMOD](https://www.fmod.com/docs/2.03/studio/working-with-instruments.html)) |
| Generator | **Stable Audio Open Small** (341M, ARC, 8-step) | ~75 ms/H100, commercial license, one-shot SFX-length output ([Stability/Arm](http://stability.ai/news-updates/stability-ai-and-arm-release-stable-audio-open-small-enabling-real-world-deployment-for-on-device-audio-control)); TangoFlux (515M) as quality alt ([TangoFlux](https://tangoflux.github.io/)) |
| Runtime | **ONNX Runtime / TensorRT** (desktop), **KleidiAI/ExecuTorch** (Arm) | int8/fp8 gives 35-45% on NVIDIA ([NVIDIA](https://developer.nvidia.com/blog/nvidia-tensorrt-model-optimizer-v0-15-boosts-inference-performance-and-expands-model-support/)); 30x on Arm CPU ([Arm](https://newsroom.arm.com/blog/stability-ai-arm-kleidi-text-to-audio-generation)) |
| Variation | **Granular re-grain + RAVE/DDSP** | Sub-10 ms timbre/variation layer on cached PCM ([BRAVE](https://arxiv.org/abs/2503.11562)) |
| De-dup / QA gate | **PANNs-based FAD** (not VGGish) + human curation | VGGish-FAD correlates <0.1 with perception; PANNs >0.5 ([arXiv 2403.17508](https://arxiv.org/abs/2403.17508)) |

---

### Phase 0 — Offline-baked proof (no real-time generation)

**Goal:** Prove neural SFX can sit convincingly inside a real game mix at all.
**Scope:** One event class (e.g. footsteps) across 4-6 materials. Generate N variants per `(event, material, intensity-bucket)` *entirely offline*, human-curate them, normalize to **-23 LUFS** (ITU-R BS.1770 / Sony ASWG-R001 ([ASWG-R001](http://gameaudiopodcast.com/ASWG-R001.pdf))), and load them into ordinary Wwise random containers with avoid-repeat/shuffle and ±2-3 semitone pitch jitter.
**Deliverables:** A baked variant library; a blind A/B test vs. an authored footstep set.
**Success metric:** Blind raters cannot reliably distinguish baked-neural from authored on *loosely-timed* steps; per-MB perceptual diversity validated by CLAP-cosine de-dup at bake time.
**Effort:** ~2-3 weeks.
**Risk retired:** *Quality/consistency.* This confronts the ~36% perceptual gap finding head-on — but note that gap is for text-to-audio *scenes* dominated by semantic fit, not curated one-shots ([arXiv 2501.08587](https://arxiv.org/abs/2501.08587)). Curating offline sidesteps stochastic drift entirely (a cache hit replays a byte-identical waveform), so this is the safest possible first step.

### Phase 1 — Prediction + cache plumbing with placeholder generation

**Goal:** Build the predictor and cache as systems infrastructure, decoupled from any model.
**Scope:** A two-tier predictor — **Tier-1 deterministic** (anim-notify lookahead on the *active* locomotion clip; physics time-of-impact) and a **Tier-2 ML stub**. A content-addressed cache keyed by `(event, material, context-bucket, intensity-bucket)`. "Generation" returns the Phase-0 baked clips (placeholder). A lock-free SPSC ring buffer hands buffers to the engine; every miss triggers the authored asset immediately and constant-power-crossfades.
**Deliverables:** Predictor service; cache with cost-aware eviction; instrumentation panel (hit-rate, lead-time distribution, wasted-prediction count) modeled on web Speculation Rules tooling.
**Success metric:** Measured **cache hit-rate** and **lead-time distribution** per event class. Honest reality check: deterministic Tier-1 footsteps approach ~100% on the active clip but with *sub-second* lead (notify positions are baked, but "upcoming clip" choice is reactive under motion matching); CCD gives only ~1 frame of impact lead, not seconds ([corrected](https://www.aclockworkberry.com/automated-foot-sync-markers-using-animation-modifiers-unreal-engine/)). Tier-2 ML intent tops out ~70-83% at ~200 ms ([PMC10248176](https://pmc.ncbi.nlm.nih.gov/articles/PMC10248176/)).
**Effort:** ~4-6 weeks.
**Risk retired:** *Prediction hit-rate* — the true make-or-break variable, since real semantic/predictive caches hit only ~20-68%, not the often-quoted ~95% ([Tianpan](https://tianpan.co/blog/2026-04-09-semantic-caching-llm-production)). Measuring this with a placeholder generator means we learn whether the concept is viable *before* paying for inference engineering.

### Phase 2 — Runtime speculative generation for a narrow event class

**Goal:** Replace the placeholder with live, off-thread neural generation for footsteps only.
**Scope:** Integrate Stable Audio Open Small (or MeanAudio, RTF 0.013, ~130 ms for a 10 s clip on an RTX 3090 ([arXiv 2508.06098](https://arxiv.org/abs/2508.06098))) as an **out-of-process worker** that pre-warms cache entries the predictor flags. Strict per-frame GPU budget: inference only runs when the frame has headroom; never on the audio callback. Authored fallback remains the guaranteed floor.
**Deliverables:** Worker inference service; speculative scheduler with **hard concurrency caps** (web Speculation Rules pattern: e.g. conservative = 2 in-flight ([Chrome](https://developer.chrome.com/blog/speculation-rules-improvements))).
**Success metric:** 99th-percentile frame-time impact under concurrent render+generate stays within budget; warm-cache hit serves within the hot-path ceiling (<5-10 ms, pre-decoded PCM).
**Effort:** ~6-8 weeks.
**Risk retired:** *GPU co-residency.* This is the binding constraint — MPS sharing causes unpredictable interference (~1-35%, mid-range ~20%) and NVIDIA lacks fine-grained graphics+compute preemption ([ConsumerBench/MPS](https://therealmjp.github.io/posts/breaking-down-barriers-part-6-experimenting-with-overlap-and-preemption/)). If frame spikes exceed budget here, the verdict is "offline-bake only" or "second GPU/NPU required."

### Phase 3 — Broaden + fallback hardening + middleware

**Goal:** Extend to more event classes and make degradation inaudible at scale.
**Scope:** Add impacts, ambiences, and textures — prioritizing *loosely-timed* material (ambience/tails), where neural quality is strongest and the ~90 ms-early/~185 ms-late AV-sync window is forgiving ([ITU-R BT.1359](https://www.tvtechnology.com/opinions/updating-lip-sync-issues)). Keep authored assets authoritative for precise synced transients (gunshots, melee) where diffusion under-models attack/transients ([arXiv 2309.06649](https://arxiv.org/abs/2309.06649)). Route every generated voice through the *identical* bus/reverb/HRTF/HDR chain and confirm-before-play gating (discard unconfirmed speculations silently).
**Deliverables:** Per-class eagerness tuning; sonically-matched authored fallbacks; multiplayer policy (generated SFX is **client-side cosmetic only**, never authoritative, due to non-determinism + latency ([arXiv 2408.05148](https://arxiv.org/pdf/2408.05148))).
**Success metric:** No audible hit/miss inconsistency in blind play sessions; the "odd sample" immersion break ([A Sound Effect](https://www.asoundeffect.com/game-audio-immersion/)) does not appear.
**Effort:** ~8-10 weeks.
**Risk retired:** *Graceful degradation and scope discipline* — proving the system never sounds worse than today's authored audio.

### Phase 4 — Optimization & shipping

**Goal:** Hit target-hardware budgets and ship on PC first.
**Scope:** Quantize (int8/fp8 via TensorRT desktop; KleidiAI/CoreML on edge), keeping **fp16 for transient-heavy SFX** since high-frequency detail degrades first under quantization ([Arm](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/audio-generation-arm-cpus-stable-audio-open-small-kleidiai)). Add a load-time/idle precompute pass to fill likely contexts, sidestepping live latency. Target PC (no console-cert/NDA blocker for an on-device model); plan console/mobile as offline-bake or cloud-prefetch.
**Deliverables:** Quantized model with per-class perceptual validation; idle/load-time cache pre-warm; target-hardware end-to-end benchmark (RTX 4060/4070, including cold-start, text-encode, VAE decode — *not* H100 headline numbers).
**Success metric:** Per-event latency and VRAM co-residency verified on shipping hardware; quantization passes the PANNs-FAD + human MOS acceptance gate.
**Effort:** ~6-8 weeks.
**Risk retired:** *Platform reality* — VRAM contention (~4 GB model working set vs. render budget on 8 GB cards) and the on-device latency cliff.

---

### Minimal MVP you can start this week

A standalone harness, no engine required:

1. **Generate:** Run Stable Audio Open Small locally; produce 20 footstep variants each for `wood / stone / grass / metal` at 3 intensity levels (~240 clips). Prompts like *"single footstep on wet stone, close mic, dry."*
2. **Curate + normalize:** Listen, discard failures, CLAP-cosine de-dup, normalize all to -23 LUFS.
3. **Cache:** A dict keyed by `(material, intensity)` → list of PCM buffers, plus one authored fallback per key.
4. **Predict:** Replay a recorded movement log; fire footstep events from animation phase, serve from cache (round-robin with avoid-repeat), measure hit-rate.
5. **Measure:** Log hit-rate, lead-time, and run a 10-person blind A/B vs. an authored set.

This MVP exercises the two highest-risk dimensions — perceptual quality and hit-rate — in days, with zero real-time-generation or GPU-contention work, and directly informs whether to proceed to the engine-integrated phases.

---

## 6. Risks, Evaluation & Open Questions

Predictive neural SFX is feasible only in a narrow, honestly-scoped form: a hybrid that pre-generates loosely-timed material ahead of predicted events, caches it by `(event, material, context)`, and falls back to authored assets on every miss. The risks below cluster around three load-bearing assumptions — that prediction lead time exceeds generation latency, that the cache hit rate is high enough to matter, and that generated audio is perceptually shippable. Each is contestable.

### Risk register

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Cold-miss latency cliff: reactive events (sudden reversals, surprise contacts) are unpredictable; generation (~1–4 s mainstream, ~75 ms only on H100) cannot recover a miss in time ([2505.08175](https://arxiv.org/abs/2505.08175), [snapnet](https://www.snapnet.dev/blog/netcode-architectures-part-2-rollback/)) | High | High | Authored-asset fallback on every event, triggered immediately; constant-power crossfade to a late-arriving generated buffer ([HDR best practices](https://www.audiokinetic.com/en/blog/wwise-hdr-overview-and-best-practices-for-game-mixing/)) |
| Low cache hit rate: analogous semantic caches hit ~20–68%, not the ~95% sometimes claimed (that figure is match-accuracy-on-hit, not hit rate) ([semantic caching](https://tianpan.co/blog/2026-04-09-semantic-caching-llm-production)) | High | Medium | Treat hit rate as the primary KPI; pre-warm common contexts at load; restrict speculation to in-frame/in-range entities |
| GPU co-residency stutter: sharing one consumer GPU with rendering causes ~1–35% interference and no fine-grained graphics+compute preemption on NVIDIA ([MJP](https://therealmjp.github.io/posts/breaking-down-barriers-part-6-experimenting-with-overlap-and-preemption/), [ConsumerBench](https://arxiv.org/abs/2506.17538)) | High | Medium | Hard per-frame compute budget and concurrency caps (web Speculation Rules model: 2–50 in-flight); idle-time generation; second GPU / NPU |
| Perceptual quality gap on transients: neural models under-model attacks; a ~36% gap to a designer reference exists on text-to-audio *scenes* (note: not one-shot impacts, and gap is dominated by semantic fit, not artifacts) ([2501.08587](https://arxiv.org/abs/2501.08587), [percussive](https://arxiv.org/abs/2309.06649)) | High | Medium | Scope neural to ambience/textures/tails; keep authored assets for synced one-shots (footsteps, gunshots, impacts, UI) |
| Repetition / "odd sample" immersion break ([asoundeffect](https://www.asoundeffect.com/game-audio-immersion/)) | Medium | High | Avoid-repeat/shuffle bags + pitch/time jitter + granular re-grain ([FMOD](https://www.fmod.com/docs/2.03/studio/working-with-instruments.html)); CLAP de-dup at bake time |
| Timing jitter and early-firing: variable hit/miss latency is worse than constant; audio leading video is noticeable >~45 ms ([NIME 2016](http://eecs.qmul.ac.uk/~andrewm/mcpherson_nime2016.pdf), [BT.1359](https://www.tvtechnology.com/opinions/updating-lip-sync-issues)) | Medium | Medium | Bias predicted sounds toward slight lag, never lead; pre-decode cached buffers to resident PCM |
| Non-determinism breaks multiplayer/replay integrity (cross-hardware FP non-associativity) ([2408.05148](https://arxiv.org/pdf/2408.05148)) | Medium | High | Scope generated SFX to client-side cosmetic-only; authored audio is the canonical synced layer |
| Console cert + fixed memory block on-device large models; mobile generation ~6.6–8 s/clip ([Arm KleidiAI](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/audio-generation-arm-cpus-stable-audio-open-small-kleidiai)) | Medium | High | Target PC first; offline/at-load pre-bake or cloud-assisted prefetch on console/mobile |
| IP overlap: predictive game-audio prefetch is already patented (US 11,632,645, HRTF prefetch) ([patent](https://patents.google.com/patent/US11632645B2/en)) | Low | Medium | Position as novel *application*; freedom-to-operate audit before building |

### Evaluation & metrics plan

The system must be measured on prediction, timing, throughput, and perceptual quality independently — a good FAD with a bad hit rate is worthless, and vice versa.

- **Prediction precision/recall**, per event class. Given cheap generation (MeanAudio RTF 0.013 ≈ 130 ms/10 s clip on an RTX 3090, ~tens of ms for short SFX), wasted speculations are cheap, so favor high recall at modest precision — mirroring speculative decoding's ~60%-acceptance-still-wins regime ([2508.06098](https://arxiv.org/html/2508.06098)).
- **Cache hit rate** as the primary feasibility KPI, bucketed by `(event, material, context, intensity)`. Benchmark against the ~20–68% range seen in production semantic caches; below that, most audio heard is authored and the neural layer adds little.
- **Timing accuracy**: per-event onset error against the actual trigger. Target slight lag (tolerated to ~185 ms late per BT.1359 acceptability) and avoid early firing (noticeable >~45–90 ms). Track jitter, not just mean — variable latency is rated worse than constant at equal mean.
- **Miss-glitch rate**: frequency of audible artifacts at hit→miss transitions. Worst-case must equal today's authored quality (zero audible miss) via constant-power crossfade.
- **Performance budget**: 99th-percentile frame-time impact under concurrent render+generate load (not mean); cache lookup/decode latency on the hot path held to **near-zero** — the audio callback deadline is ~2.7 ms (128 samples @48 kHz), and the pipeline already spends ~53–85 ms in a default FMOD mixer, leaving JND headroom of only ~27 ms from a non-zero base ([3678299](https://dl.acm.org/doi/10.1145/3678299.3678331)). Never run inference on the audio thread.
- **Quality (FAD/CLAP/MOS)** — use with caution. **Do not use VGGish-FAD** (Spearman ρ < 0.1 with human perception of environmental audio; PANNs-based embeddings reach ρ > 0.5) ([2403.17508](https://arxiv.org/abs/2403.17508)). CLAP measures text-semantic alignment, not fidelity, so it cannot certify ship quality. The only trustworthy gate today is a **human MOS panel** scoring clarity/artifacts and library-consistency, supplemented by a domain-matched (PANNs) embedding or a small learned MOS predictor.
- **Repetition perception**: A/B listening tests for the "odd sample" effect, since no standard objective metric exists.

A critical unknown: the runtime **accept-vs-fallback gate** has no reliable automated metric, so cached results should be **human-curated offline** before going live — the cache is a reviewed asset library, not raw model output.

### Open questions / unknowns

- What fraction of a real game's SFX is deterministically schedulable far enough ahead? Anim-notify lookahead works for footsteps on an *active* clip (~100–400 ms lead), but continuous-collision time-of-impact gives only ~1 frame (~16.7 ms), and motion-matched "upcoming" clips are reactive — so the truly free, high-precision subset is narrower than hoped.
- Can per-instance timbral consistency across cached variants be maintained well enough to read as "different takes of one sound" rather than an uncanny outlier? (Stochasticity itself is not the problem — a cache hit replays an identical waveform — but cross-asset coherence with the authored library is unproven.)
- What is the real cache-key cardinality once context is bucketed, and does it fit a fixed memory budget without boundary artifacts at bucket edges?
- Does int8/fp16 quantization erode the high-frequency/transient detail exactly where game SFX needs it most? (High-freq content degrades first under quantization.)
- Is there a viable on-device path (NPU/CoreML/ANE) that closes the ~7–8 s mobile generation gap, or is mobile permanently offline-prebake-only?

### Where this is — and isn't — worth doing

**Worth doing now**, on PC with a dedicated/second GPU: an **offline-baked neural variant library** keyed by `(event, material, context, intensity)`, played through existing Wwise/FMOD avoid-repeat + jitter machinery, with authored fallback. This is largely systems integration of proven components — runtime PCM injection, crossfade fallback, loudness normalization (-23 LUFS, BS.1770), and voice management are all first-class engine APIs today. It excels for ambiences, textures, weather/crowd beds, decay tails, and variation layers, and degrades gracefully to today's quality.

**Not worth doing** as advertised: live per-frame neural synthesis, neural-generated *precise synced one-shots* at AAA quality, neural-only coverage with no authored floor, or on-device real-time generation on mobile/console. These collide with the latency gap (1–3 orders of magnitude), the transient-quality wall, and certification/memory limits.

**Adjacent applications are stronger bets.** Real-time *instruments* are the natural fit for the genuinely low-latency neural class — RAVE/BRAVE-style conditioning synthesis hits <10 ms with ~3 ms jitter on CPU ([2503.11562](https://arxiv.org/abs/2503.11562)) — but these *transform* a live input/control stream and cannot conjure a labeled SFX in the hot path, so they sidestep prediction entirely rather than implementing it. **Adaptive media** (non-interactive or slow-cadence: dynamic ambiences, generative soundscapes, music beds, VR environmental audio) tolerates seconds of lead and is where prefetch+cache pays off with the least risk. The honest framing throughout: this is **neural variation and speculative prefetch atop authored assets**, not a replacement for the SFX pipeline.

---

## Appendix — About this study

This document was produced by a deterministic multi-agent research workflow. Eight specialist agents researched, in parallel: (1) the neural-audio model landscape & generation speed, (2) real-time audio latency budgets, (3) game-state prediction & speculative-prefetch strategy, (4) audio cache architecture & variation, (5) model optimization & serving, (6) middleware integration / fallback / mixing, (7) prior art, and (8) perceptual quality & evaluation. Each agent's most load-bearing claims were then handed to independent fact-checker agents instructed to *refute* them with their own web searches; surviving claims and corrected values fed the synthesis. Of 30 critical claims checked, 26 were corroborated, 0 were refuted, and 4 remained uncertain (treated conservatively in the text).

Key corrections applied during fact-checking (worth knowing if you cite these numbers):

- **Cache hit-rate reality:** the often-quoted ~95% for semantic caches is *match-accuracy on a hit*, not the hit rate; real predictive/semantic cache hit rates are ~20–68%. Hit rate is the true make-or-break KPI.
- **Physics lead time:** continuous-collision time-of-impact gives only ~1 frame (~16.7 ms) of warning, not seconds — it confirms an impact, it doesn't hide generation latency.
- **Loudness target:** game standard is ~‒23 LUFS (ITU-R BS.1770 / Sony ASWG-R001); ‒24 LKFS is US *broadcast*, not games.
- **Quality metrics:** VGGish-based FAD correlates poorly with human perception of environmental audio (ρ < 0.1); use PANNs-based embeddings (ρ > 0.5) or a human MOS panel. CLAP measures semantics, not fidelity.
- **“Real-time” models:** RAVE/BRAVE-class sub-10 ms synthesis *transforms a live input stream*; it cannot conjure a labeled SFX from an abstract event in the hot path — it belongs in the variation layer, not the generator.
