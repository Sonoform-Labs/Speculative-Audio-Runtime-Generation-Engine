# Unity Integration & Open-Source Roadmap

## Honest traction read (stars)
Stars ≈ **demo × distribution**, not code quality.
- **High ceiling:** "ship zero SFX files — your game's sound effects generate themselves, by context, in
  real time" is novel + screenshottable, and gen-AI-for-games is hot. A striking 60-sec demo video +
  Show HN + r/gamedev/r/Unity3D could plausibly hit **hundreds–low-thousands** of stars early.
- **Floor without a great demo:** tens–low-hundreds. The demo video is ~90% of the outcome.
- **Sustained adoption is harder than the spike.** Real barriers: needs a runtime GPU; model ~1–3 GB;
  quality is "stylized," not AAA-photoreal; SA3 license caps free commercial at $1M rev; every dev asks
  "why not just ship WAVs?" So curiosity-stars ≫ production installs at first.
- **What maximizes it:** (1) jaw-dropping demo, (2) works as a real Unity package (not a script),
  (3) the "zero SFX files" hook.

## Unity implementation plan
### U0 — offline-bake (fastest "it's in a real game", ~1–2 days)
Bake the affordance library to WAVs with the Python engine → import to Unity → trigger via normal
AudioSource on game events. Proves generated SFX in a real game with zero runtime Python/IPC.

### U1 — runtime bridge (the real thing, ~1–2 weeks, 1 dev)
- **Python sidecar service** (local socket/WebSocket): `on_context_change` / `on_event` → returns PCM.
  *(Buildable + testable in this repo now, with a mock client.)*
- **Unity C# client**: connect, send context/events, receive PCM, inject via `AudioClip.Create` or
  `OnAudioFilterRead`.
- **Sample scene wiring:** footsteps (animation event), a gun (input), surface (downward raycast).

### U2 — polish (demo-ready)
Variant rotation, authored/auto-baked fallback assets, off-thread generation + latency budget, a clean
demo scene, and **the demo video**.

### Later (shipping / expansion)
- Embed the model via **ONNX/TensorRT** so there's no Python sidecar (much more adoptable).
- GPU scheduling vs the renderer (the known contention wall).
- **Unreal C++ port** (`USoundWaveProcedural::QueueAudio`).

## OSS form & license
- **Unity package (UPM)** + Python sidecar (v1) → embedded ONNX (v2).
- **License:** MIT the package, but **BYO-model** — SA3 weights are under the Stability AI Community
  License; do not redistribute weights in the repo, users accept Stability's terms separately.
- Make-or-break artifact = **the demo video**. Positioning: *"Ship zero SFX files."*

## Why it's defensible / novel
Per `docs/affordance-cache-design.md` + `docs/prior-art.md`: the affordance → ranked speculative
*synthesis* → cache → fallback loop is unbuilt for audio; all prior art is graphics/assets/web/LLM. FTO
flags (US11632645, US11033824, EA US12420192B2) matter only if commercialized.
