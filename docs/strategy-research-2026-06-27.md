# Strategy research — direction + generalization (2026-06-27)

Deep-research (5 angles, 22 sources, 22 verified claims) to decide: realtime vs predictive, and how
to make the system GENERALIZE to any game without hand-authored affordance tables (the core problem).

## Recommendation: ADAPTIVE HYBRID + auto-derived affordances
- **Route by event class:** near-realtime **on-demand generation** for the majority of events
  (footsteps, ambient, one-shots, distant impacts — tolerate a short async buffer), and
  **predictive prefetch only for the latency-critical minority** (sub-frame button presses, dense
  bursts, moments of GPU contention).
- **Generalize via auto-derivation, NOT hand-authored tables:** derive the sound vocabulary + event
  triggers + context from (a) **engine reflection** over standard APIs (Unity collision /
  physics-material / Component / event streams), (b) **runtime telemetry learning** from the live
  event stream, and (c) **LLM scoping** of game code/description (already prototyped: scope_game.py).

## Why hybrid is the only defensible design (latency evidence)
- Fast gen is real: MeanAudio 1-NFE (RTF 0.013 / ~130ms for 10s on a 3090), AudioLCM 2-step (333x
  realtime), ConsistencyTTA (single query), WaveFlow (EA/SEED, AIIDE'22, 10.4x realtime on a GTX 1080).
- BUT per-event wall-clock on a **3060 lands ~100-200ms+** (incl. text-encode + VAE decode + contention;
  consistent with our measured sa3 219ms), which **exceeds tight interactive tolerances**: musician
  percussive 10ms target / 20ms degradation; action-to-sound JND ~27-77ms (baseline-dependent).
- So on-demand suffices for passive/ambient SFX; tight triggers + bursts still need prefetch.
  -> route by event class.

## The genuine novel delta (the paper's contribution)
Auto-deriving a game's sound vocabulary/triggers is **unsolved by existing tools**:
- Unity physics-material->sound tooling is **entirely hand-authored** (the exact problem to kill).
- Sound Event Detection (SED) needs a predefined class set + strong labels; open-vocabulary SED is
  early-stage ("far from satisfactory").
- ProVE learns procedural-audio *synthesis* (footsteps only), NOT a game's event taxonomy.
=> an **auto-derivation layer** (engine reflection + runtime learning + LLM scoping) is unclaimed.

## Integration contract (drop-in, cross-game) — borrow DLSS/NVIGI
- DLSS generalizes by consuming **standard engine intermediate representations** (motion vectors,
  depth) + non-game-specific training, not per-game authored data. NVIGI exposes function-specific,
  **swappable plugins** behind a unified inference API.
- For us: subscribe to engine collision/physics-material/event streams (standard IRs), infer
  event+material+intensity+context from them, expose a **swappable generator backend**
  (we already built RemoteGenerator for exactly this). Near-zero per-game authoring.

## Deployment path
- Near-term: Python sidecar (current) + RemoteGenerator for swappable/fast models.
- Long-term no-sidecar: **Unity Sentis** (built-in on-device ONNX inference, opset 7-25) — viable for
  a distilled 1-step model exported to ONNX; verify feasibility on the 3060 (open question).

## Paper framing + evaluation (EXAG/AIIDE)
- Framing: **"a generalizable adaptive-hybrid generative-audio middleware that drops into any game
  with near-zero per-game authoring."**
- Evaluation: **ERA** (Expressive Range Analysis) — there's a 2025 AIIDE precedent applying ERA to
  text-to-audio (arXiv 2510.27102) — PLUS cross-game operational metrics over a **multi-game test set**
  (FPS / platformer / racing / sim): coverage & hit-rate (fraction of fired events that got an
  appropriate auto-derived sound), false-trigger rate, **dev-effort (lines/minutes -> near-zero)**,
  and on-time-delivery rate per event class on the 3060.

## Honest risks / open questions
- **Biggest unproven assumption (= the contribution):** can engine reflection + telemetry actually
  auto-derive a usable, low-false-trigger affordance table across structurally different games with
  zero authoring? This is the project's opportunity AND its biggest risk.
- **Must measure:** real end-to-end per-event latency + jitter on the 3060 under game GPU contention
  -> sets the realtime-vs-prefetch routing threshold per event class.
- **Sentis feasibility** for a distilled model on the 3060 (to drop the sidecar) — unverified.
- 1-step models trade some quality (CLAP 0.321->0.292); SFX-specialized 44.1k (sa3) still the quality
  floor — the fast model is the realtime tier, sa3 the quality tier (both swappable via RemoteGenerator).
