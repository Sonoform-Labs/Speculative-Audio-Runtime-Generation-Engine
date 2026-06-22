# Affordance-Driven Speculative Cache for Generative Game SFX — Design & Prior-Art Memo

## 1. Closest prior art

| System | What it is | How close |
|---|---|---|
| EA SEED — Neural Synthesis of SFX (US12420192B2) ([ea.com/seed](https://www.ea.com/seed/news/aiide22-neural-synthesis-sfx)) | Flow-based model that synthesizes *variations* of a given SFX; library can be baked in advance during production | Adjacent — generative game audio, but reactive/offline, no context-driven speculative warming or eviction |
| Outatime — speculative cloud gaming (MSR, MobiSys'15) ([cs.princeton.edu](https://www.cs.princeton.edu/courses/archive/fall15/cos518/papers/outatime.pdf)) | Pre-renders the *plausible set* of future frames one RTT ahead: Markov chain for continuous inputs, render-all-branches for impulsive discrete inputs, discards misses | Very close (structurally) — predict→pre-generate plausible set→discard misses, but for video frames not SFX |
| Chrome Speculation Rules API ([developer.chrome.com](https://developer.chrome.com/blog/speculation-rules-improvements)) | Confidence-tiered (immediate/eager/moderate/conservative) speculative prefetch/prerender with per-tier concurrency budgets + FIFO eviction + prefetch→prerender upgrade | Very close (control structure) — confidence-tiered budgeted speculative warming, but of fetched pages |
| Binaural-sound prefetch patent family (US11632645) ([patents.google.com](https://patents.google.com/patent/US11632645)) | Claims *what/when/how-much* audio data to prefetch+preprocess+cache, reasoning about too-early (waste) vs too-late (miss) | Very close (problem framing) — but caches static HRTF/audio data, never *generates* |
| Microsoft Xbox ML asset preload (US11033824) ([patents.google.com](https://patents.google.com/patent/US11033824)) | NN over ~90M players/wk predicts which assets a user picks, bounded preload, **cancels** non-selected candidates on resolution | Very close (loop shape) — predict→bounded preload→cancel-on-resolution, but static assets |
| Pythia — agent-native LLM serving (2026) ([arxiv](https://arxiv.org/html/2604.25899v1)) | Predicts next request from workflow graph, warms cache on idle GPU into cheaper memory tiers, frees "dead" unreachable blocks immediately | Very close (mechanics) — context-warming + dead-set eviction + GPU-contention staging, on KV blocks |
| InstCache ([arxiv](https://arxiv.org/html/2411.13820v2)) / CacheSack ([usenix](https://www.usenix.org/system/files/atc22-yang-tzu-wei.pdf)) / Baleen ([usenix](https://www.usenix.org/system/files/fast24-wong.pdf)) | Likelihood-threshold enumeration of plausible set; categorized fractional-knapsack admission; benefit/size scoring + "ML-When" prefetch gate | Adjacent — the ranking/admission math, on text/storage |
| GDSF cost-aware eviction (Cao & Irani / Cherkasova) ([hpl.hp.com](https://www.hpl.hp.com/personal/Lucy_Cherkasova/projects/gdfs.html)) | `H = L + freq·cost/size`, evict min, aging clock `L` — proven-competitive cost/size/frequency policy | Very close (the scoring core) — domain-agnostic, maps directly |
| Unreal Level Streaming Volumes ([epicgames](https://dev.epicgames.com/documentation/en-us/unreal-engine/level-streaming-volumes-reference-in-unreal-engine)) / Unity Addressables refcount ([unity](https://docs.unity3d.com/Packages/com.unity.addressables@1.21/manual/MemoryManagement.html)) / FMOD idle pool ([fmod](https://www.fmod.com/docs/2.03/api/studio-guide.html)) / Wwise proximity banks ([audiokinetic](https://www.audiokinetic.com/en/blog/quick-and-easy-how-to-manage-your-soundbanks-in-wwise/)) | Region/proximity-triggered load with priority+hysteresis; refcount lifetime; resident-pin + idle LRU pool; proximity-scoped bank loading | Adjacent — context-driven prefetch + eviction machinery for *authored* assets |

**Verdict on novelty.** Every individual mechanism we need is borrowed and battle-tested: cost/size/frequency-aware ranking and eviction (GDSF, proven-competitive), confidence-tiered budgeted speculation (Chrome Speculation Rules), plausible-set pre-render with miss-fallback (Outatime), context-driven asset prefetch with cancel-on-resolution (Unreal/Wwise/FMOD, Xbox US11033824), and GPU-contention-aware idle-cycle warming (Pythia, NVIDIA CiG). What no shipped or published system does is the *combination*: enumerating a plausible sound set from **affordances/player intent** and **speculatively SYNTHESIZING** (not retrieving) it ahead of the trigger, ranked under a GPU+VRAM budget that competes with rendering, with authored fallback on miss. EA owns the variation *generator*; the binaural family and Xbox patent own *prefetch of static audio/assets*. Our defensible contribution is the **closed loop — affordance prediction → ranked speculative generation → cache → authored fallback** — treating the generator (Stable Audio 3 Small) as swappable. The honest caveat: the loop is novel, but it is an assembly of known parts, and the two nearest patents (US11632645, US11033824) plus EA's US12420192B2 are explicit freedom-to-operate items for any prediction-keyed-prefetch or generative claim.

## 2. Ranking — which candidates to pre-generate first

Frame "what to pre-generate under a bounded budget" as a **0/1 knapsack**, solved online by **value-density greedy fill** (within 1/2 of optimal for 0/1, exact for fractional ([US8112755](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8112755))). The scoring key is **GreedyDual-Size-Frequency (GDSF)** reinterpreted for generation ([Cao & Irani](https://www.usenix.org/conference/usits-97/cost-aware-www-proxy-caching-algorithms); [Cherkasova](https://www.hpl.hp.com/personal/Lucy_Cherkasova/projects/gdfs.html)), with the **frequency** term replaced by predicted use-probability:

```
priority(s) = L + P(use | context, s) · gen_cost_ms(s) / bytes(s)
```

- `P(use|context,s)` — from the affordance/intent model (§4), 0–1.
- `gen_cost_ms(s)` — measured GPU-ms to synthesize (≈393 ms/clip for SA3-small, but scales with duration/cfg). In the numerator because expensive-to-generate clips hurt most if missed on the critical path.
- `bytes(s)` — clip VRAM/RAM, in the denominator so the score reads as **"expected playback latency saved per byte."**
- `L` — global aging clock (§3).

**Generate in DESCENDING priority** until either the per-context-change GPU-ms budget or the VRAM budget binds (whichever first). This is the same scalar used for eviction (§3), collapsing ranking and eviction into one priority queue (`O(log k)` insert/evict).

**Admission gate before spending any GPU** (Mogul/Dahlin cost-benefit, [Padmanabhan-Mogul](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/01/ccr-july96.pdf); Baleen "ML-When", [usenix](https://www.usenix.org/system/files/fast24-wong.pdf)). Only generate when expected benefit exceeds expected waste:

```
generate s  iff  P(use) · latency_saved  >  (1 − P(use)) · gen_cost_ms
            i.e.  P(use)  >  gen_cost_ms / (gen_cost_ms + latency_saved)
```

`latency_saved` = the stall/quality cost avoided by having the clip cached vs falling back. This protects the renderer: **raise the effective threshold when GPU frame-time is tight, lower it when idle.** Cheap or high-stall-cost sounds get warmed at low confidence; expensive ones demand high confidence.

**Confidence-tiered budgeting** (Chrome Speculation Rules, [developer.chrome.com](https://developer.chrome.com/blog/speculation-rules-improvements)): bucket candidates by `P(use)` into *immediate / eager / moderate / conservative* tiers, each with a hard concurrency cap against the VRAM/GPU ceiling. Use the **prefetch→prerender "upgrade"** pattern: generate a cheap low-cfg/short **draft** first, regenerate at full SA3 quality only if confidence climbs — directly exploiting SA3's cfg/length-vs-GPU-ms trade.

## 3. Budget + eviction

**Caps.** Two hard ceilings: a **VRAM/RAM byte budget** for the cache and a **per-context-change GPU-ms budget** for speculative generation. Generation runs as **low-priority, idle-cycle work** (NVIDIA Compute-in-Graphics; Pythia idle-GPU warming, [arxiv](https://arxiv.org/html/2604.25899v1)) that always **yields to rendering** and is **preempted by any real cache-miss demand generation** (ProMoE/SpeCache two-priority queue, [arxiv](https://arxiv.org/html/2410.22134v1)).

**Eviction = the same GDSF scalar, ascending.** Evict minimum `priority(s)`; on each eviction set `L = priority(evicted)`. Because the aging clock only re-adds `L` to items that get touched, sounds from abandoned contexts (whose `P(use)→0`) sink automatically — **no per-frame sweep needed**. Under heterogeneous size+cost, optimal eviction is **NP-complete** ([Berger et al.](https://www.pdl.cmu.edu/PDL-FTP/associated/pomacs32-berger.pdf)), so greedy GDSF is correct engineering, not a shortcut.

**Three tiers** (FMOD/Addressables pattern):
- **PINNED** — refcount-held by every live context (gun equipped pins shoot/reload/empty-click). Never speculatively evicted. ([Unity Addressables](https://docs.unity3d.com/Packages/com.unity.addressables@1.21/manual/MemoryManagement.html))
- **SPECULATIVE** — GDSF-managed; admitted at a *lower* initial `H` than confirmed-fired clips so unused speculation evicts first.
- **IDLE POOL** — small fixed-byte LRU tail of recently-fired clips kept for cheap reuse (player re-enters a room; a sound re-fires). ([FMOD idle pool](https://www.fmod.com/docs/2.03/api/studio-guide.html))

**On context change, two complementary triggers:**
1. **Immediate dead-set eviction** (Pythia; Xbox cancel-on-resolution, [US11033824](https://patents.google.com/patent/US11033824)): when a context becomes *unreachable* (gun fired → reload/empty-click siblings now impossible; player left room), free those clips at once. Also **cancel in-flight speculative generations** of now-impossible siblings.
2. **Asymmetric hysteresis** (Unreal Level Streaming Volumes, [epicgames](https://dev.epicgames.com/documentation/en-us/unreal-engine/level-streaming-volumes-reference-in-unreal-engine)): warm *instantly* when an affordance appears (no load hysteresis), but apply a **cooldown** before de-warming when a context merely *leaves* (holster, pace out of a room), so quick re-equip / boundary-pacing doesn't thrash the generator.

**Anti-pollution admission** (W-TinyLFU, [caffeine](https://github.com/ben-manes/caffeine/wiki/Efficiency)): keep a count-min sketch (~8 B/entry) of historical `(context, sound)` firings; a fresh speculative clip displaces an existing entry only if its sketch frequency (or `P(use)`) beats the victim's. A small recency window still admits a brand-new context's sounds with zero history — so a one-shot footstep can't evict a proven-hot reload.

## 4. Context / affordance model

**Stage 1 — enumerate (deterministic, no ML for v1).** Tag each object/surface archetype with an **affordance → sound-event table** at authoring time (gun → {shoot, reload, empty-click}; plate → {drop, shatter}; metal floor → {footstep, impact}) — the vocabulary level designers already think in ([Gibson/Norman applied to games](https://machinations.io/articles/affordances-in-game-systems-design)). On each context change, **union the affordance tables of all in-scope objects** to get the raw candidate set. This mirrors Wwise's Event-Preparation / proximity-scoped bank loading ([audiokinetic](https://www.audiokinetic.com/library/edge/?id=method_5_preparing_events_and_game_syncs_switches_and_states)). If the game already runs GOAP/BT/utility AI, the planner's set of **precondition-satisfied actions** *is* a free candidate set.

**Stage 2 — score `P(use)` with a Utility-AI consideration model** ([Game AI Pro 3, ch.13](https://www.gameaipro.com/GameAIPro3/GameAIPro3_Chapter13_Choosing_Effective_Utility-Based_Considerations.pdf)). Each consideration is a raw game value mapped through a designer-tuned **response curve** to [0,1], then combined:

```
P(use) = base_event_rate(affordance)
       · proximity_curve(distance)        // inverted-U / decay
       · intent_curve(aim, last_action)    // equipped, aiming, moving-toward
       · recency_decay(time_since_context)
```

Use the **compensated product / geometric mean** (Rez Graham, GDC'18) so candidates with many considerations aren't unfairly driven toward 0. A decay/inverted-U curve makes a holstered gun or left room **smoothly drop its scores**, which also drives eviction (§3). Runs in microseconds; tunable by audio designers; no ML.

**Refinements:**
- **Affordance-landscape size as temperature** ([affordance-intention prediction](https://www.tandfonline.com/doi/full/10.1080/10407413.2024.2427403)): a *tight* landscape (just-equipped weapon) concentrates probability → warm aggressively + hedge variants; a *broad* landscape (open room, many surfaces) spreads it thin → warm only the top few.
- **Outatime split** ([MSR](https://www.cs.princeton.edu/courses/archive/fall15/cos518/papers/outatime.pdf)): classify each affordance's sounds as **predictable-continuous** (footsteps tracking movement → Markov/velocity model, warm top 1–2 surfaces) vs **impulsive-discrete** (shoot/reload/empty-click → warm the *whole* small action-set, since you can't predict which).
- **Learned upgrade**: layer an **n-gram/Markov next-action predictor with backoff** ([Gatech](https://faculty.cc.gatech.edu/~surban6/2019fa-gameAI/lectures/2019_11_18-Prediction_Ngram.pdf)) trained on confirmed-event telemetry (equip→shoot 80% / reload 15% / empty-click 5%) to re-weight `P(use)`; calibrated priors per [US7934058](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7934058). For PvP, an LSTM opponent-model head predicts opponent-caused sounds.
- **Likelihood threshold τ** ([InstCache](https://arxiv.org/html/2411.13820v2)): admit to the precompute set only candidates with `P(use) ≥ τ`; sweeping τ gives a principled budget-vs-hit-rate curve (expected hit-rate ≈ CDF of `P(use)` at τ) instead of an arbitrary top-N.

## 5. Concrete algorithm sketch

```python
# Runs on each context-change event. Generation is best-effort, idle-GPU, preemptible.
on_context_change(world_state):
    # 1. ENUMERATE candidate sounds (union affordance tables of in-scope objects)
    candidates = set()
    for obj in in_scope_objects(world_state):          # equipped, proximate, current room
        for action, sound_proto in affordance_table[obj.archetype]:
            candidates.add((obj, action, sound_proto))

    # 2. SCORE P(use) via utility considerations (compensated geometric product)
    for c in candidates:
        c.p_use      = utility_score(c, world_state)    # base·proximity·intent·recency, [0,1]
        c.gen_ms     = est_gen_cost(c.sound_proto)      # duration/cfg -> GPU ms
        c.bytes      = est_bytes(c.sound_proto)
        c.latency_saved = quality_delta_vs_fallback(c)

    # 3. ADMISSION GATE: drop low-EV candidates (protect renderer; threshold rises under GPU load)
    gpu_pressure = current_frame_gpu_headroom()
    admitted = [c for c in candidates
                if c.p_use >= max(TAU,
                                  c.gen_ms / (c.gen_ms + c.latency_saved)) * gpu_pressure
                and not cache.has_fresh(c)]

    # 4. RANK by GDSF value-density (descending) = expected latency saved per byte
    admitted.sort(key=lambda c: L + c.p_use * c.gen_ms / c.bytes, reverse=True)

    # 5. SPLIT predictable-continuous vs impulsive-discrete; bound branch count
    queue = outatime_split(admitted)                    # warm whole impulse set; top-k continuous

    # 6. GENERATE greedily under DUAL budgets, in idle GPU cycles, preemptible by demand
    spent_ms, used_bytes = 0, cache.used_bytes()
    for c in queue:
        if spent_ms + c.gen_ms > GPU_MS_BUDGET: break
        while used_bytes + c.bytes > VRAM_BUDGET:
            victim = cache.min_priority_speculative()   # ascending GDSF, never PINNED
            if victim is None: break
            L = victim.priority                          # advance aging clock
            used_bytes -= evict(victim)
        if used_bytes + c.bytes > VRAM_BUDGET: continue
        clip = generate_draft(c)                         # cheap low-cfg first (upgrade later)
        cache.insert(c, clip, init_priority=L + DISCOUNT*c.p_use*c.gen_ms/c.bytes)  # speculative tier
        spent_ms += c.gen_ms; used_bytes += clip.bytes

    # 7. DEAD-SET: free clips whose context is now unreachable; cancel their in-flight gens
    for c in cache.speculative():
        if not reachable(c, world_state):
            cancel_inflight(c); evict(c)                 # but honor de-warm COOLDOWN (hysteresis)

on_real_event(event):                                   # the actual trigger
    clip = cache.get(event) or semantic_cache.nearest(event, thresh=family_thresh)
    if clip: play(clip); cache.touch(clip)              # refcount/recency bump
    else:    play(authored_fallback(event));            # never a wrong sound
             demand_generate(event, priority=HIGH)       # preempts speculative gen
    cancel_inflight(impossible_siblings(event))          # cancel-on-resolution
```

## 6. Honest risks

- **Over-speculation / GPU thrash starving the renderer.** The generator competes with rendering. *Mitigations:* the Mogul/Dahlin **admission gate** with a threshold that rises under GPU pressure; run generation as **low-priority idle-cycle CiG work** preempted by demand gens (Pythia/ProMoE); **asymmetric hysteresis** + cooldown so boundary-pacing/re-equip doesn't re-trigger generation; instrument **accuracy = fired/generated** (wasted-GPU rate) and raise τ when it drops ([prefetch metrics](https://arxiv.org/pdf/1911.09829)).
- **Cold / novel events with no history.** A brand-new context has no telemetry priors and could be starved by W-TinyLFU. *Mitigations:* n-gram **backoff** to the largest observed context; W-TinyLFU **recency window** that admits zero-history candidates; affordance tables are **designer-authored**, so v1 needs no telemetry at all; and the **authored fallback always covers a miss** — a cold event degrades gracefully to authored audio, never to silence or a wrong sound.
- **Combinatorial blowup** (many objects × many affordances × variants). *Mitigations:* **likelihood threshold τ** (InstCache) prunes the long tail to a CDF-bounded set; **bound the branch count** per context (Outatime caps to ~4 sequences); **affordance-landscape temperature** warms only the top few in broad scenes; **CacheSack categorized fractional budgets** ([usenix](https://www.usenix.org/system/files/atc22-yang-tzu-wei.pdf)) — pre-generate *k of n* footstep variants now, defer the rest, and shrink low-value categories' fractions under pressure rather than dropping them.
- **Over-fit / wasted variety from hedging.** Pre-generating many variants of one sound can burn budget for marginal perceptual gain. *Mitigations:* **semantic-cache reuse** (cosine ≥ ~0.85 footsteps, ~0.95 distinctive weapons, [spheron](https://www.spheron.network/blog/semantic-cache-llm-inference-gpu-cloud/)) so one clip serves near-duplicates; **DeepCache-style intra-generation feature reuse** ([arxiv](https://arxiv.org/html/2312.00858v2)) to cut below 393 ms/clip if SA3 uses an iterative sampler, raising how many candidates fit the budget.
- **Local-tuning blind spots.** Greedy GDSF is not optimal under size+cost heterogeneity (it can't be — NP-complete). *Mitigation:* build a **size-aware Belady offline oracle** (evict max `size × next-use-distance`) over recorded sessions ([Berger et al.](https://www.pdl.cmu.edu/PDL-FTP/associated/pomacs32-berger.pdf)) to bound regret and tune the VRAM budget, τ, and the frequency exponent — and report it as the upper bound rather than chasing intractable optimality.
- **Freedom-to-operate.** US11632645 (audio prefetch), US11033824 (predict→preload→cancel), and EA US12420192B2 (SFX variation generation) are adjacent. *Mitigation:* lead the claim with the **generative speculative loop**, treat the generator as swappable, and run a formal FTO review on the prediction-keyed-prefetch language.