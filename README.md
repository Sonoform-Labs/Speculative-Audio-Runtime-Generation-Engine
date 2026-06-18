# Predictive Neural SFX

**Speculative prefetching, but for sound.** A hybrid game-audio system that watches game
state to *predict* which sound effects are about to be needed, *pre-generates* them with a
neural audio model **before** they fire, *caches* them by context, and *falls back* to
authored assets the instant a prediction misses.

This repo is the standalone simulation / MVP harness for validating the two risks that
decide whether the idea works at all: **prediction hit-rate** and **audio quality**.

---

## The core insight (read this first)

You **cannot** make neural generation fast enough to call the moment a sound fires. The
fastest model on Earth is ~75 ms on a datacenter H100, and hundreds of ms to *seconds* on
normal hardware — against a ~10 ms audio-callback budget. Synchronous on-trigger generation
is impossible by 1–3 orders of magnitude, no matter how fast the model gets.

That impossibility is *why* prediction + caching exist. The whole system rides on one
inequality:

> **lead_time ≥ generation_latency**  (or the key is already cache-resident)

Predict the event ~100–300 ms early, generate during that head start, cache the result so
you pay the cost **once, ahead of time** — never on the hot path. On a miss, fall back to an
authored asset so the floor is never worse than today's pipeline.

The simulator in this repo makes that inequality visible: sweep generation latency and watch
the hit-rate collapse the moment it exceeds the prediction lead.

## Quickstart

```bash
# 1. (optional) create a venv, then install in editable mode
pip install -e .            # just numpy; or `pip install -e .[dev]` for tests

# 2. generate the sample movement log (a player walking across surfaces)
python scripts/make_sample_log.py

# 3. run the simulation (uses the dependency-free mock generator)
python scripts/run_sim.py

# 4. see the feasibility inequality in action
python scripts/run_sim.py --sweep
```

No API key or model download needed — the **mock generator** synthesizes placeholder SFX so
the full loop runs out of the box. Real backends drop in behind the same interface (see
below).

### Useful runs

```bash
# realistic ML predictor: 80% recall, 85% key accuracy, with an offline-baked floor
python scripts/run_sim.py --predictor noisy --recall 0.8 --accuracy 0.85 --prebake

# bake a real variant library to WAV (mock here; swap --generator for real audio)
python scripts/bake_library.py --variants 8
```

## Swapping in a real generator

Everything is built against one `Generator` interface, so the pipeline never cares where
audio comes from:

| Backend | Install | Notes |
|---|---|---|
| `mock` | nothing | Synthetic SFX. Start here; validates the loop with zero setup. |
| `stableaudio` | `pip install -e .[stableaudio]` + `huggingface-cli login` | Local Stable Audio Open Small. Low-latency, on-device, no per-call cost — the runtime-story backend. |

```bash
python scripts/run_sim.py --generator stableaudio --prebake
```

## Architecture

```
GameEvent timeline ──▶ Predictor ──▶ Prediction(key, emit_time, target_time)
                                          │  (speculative request)
                                          ▼
                                     Generator ──▶ AudioClip
                                          │
                                          ▼
                                    VariantCache  (offline-baked pinned tier
                                          │        + runtime LRU tier, avoid-repeat)
                event fires ──▶ cache HIT? ──▶ play cached variant   (≈0 ms)
                                     └─ miss ─▶ authored fallback     (never worse than today)
```

| Module | Role |
|---|---|
| `types.py` | `GameEvent`, `CacheKey`, `AudioClip` |
| `generators/` | `Generator` interface + `mock` / `elevenlabs` / `stableaudio` backends |
| `cache.py` | context-keyed cache; pinned offline tier + LRU runtime tier; avoid-repeat |
| `predictor.py` | `Tier1DeterministicPredictor` (anim-notify-style) + `NoisyPredictor` (ML tier) |
| `pipeline.py` | virtual-time discrete-event sim of predict→generate→cache→fallback |
| `metrics.py` | hit-rate, fallback-rate, lead-time, speculation precision, repetition |

## Roadmap

The goal is a standalone proof that backs a conference (ADC) talk. Full plan and status:
**[`docs/PLAN.md`](docs/PLAN.md)**.

- [x] **Phase 0/1** — predict→cache→fallback loop + metrics on a mock generator *(this repo)*
- [ ] **Phase A** — local Stable Audio Open Small; bake a real library; measure real GPU latency
- [ ] **Phase B** — blind A/B vs authored; FAD/PANNs + CLAP; the repetition centerpiece
- [ ] **Phase C** — systems proof on a realistic trace with real measured latency
- [ ] **Phase D** — standalone audible demo + the ADC talk deck

See also [`docs/feasibility-study.md`](docs/feasibility-study.md) (full feasibility analysis)
and [`docs/prior-art.md`](docs/prior-art.md) (novelty / prior-art assessment).

## Prior art & novelty (short version)

The full **predict → pre-generate → cache → authored-fallback** loop for game audio does not
appear to exist as a shipped product, but the pieces do. Notably, **EA SEED's granted patent
US12420192B2** already covers neural SFX *variation* generation (and contemplates runtime use)
— so the "infinite variation engine" framing is **not** the novel part. The defensible
contribution is the speculative-prefetch control loop driven by game-state prediction. See
[`docs/prior-art.md`](docs/prior-art.md).

## License

MIT
