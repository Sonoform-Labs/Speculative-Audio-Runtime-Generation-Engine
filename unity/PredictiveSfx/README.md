# Predictive SFX — Unity package

Runtime neural sound effects driven by game context, over a local sidecar process. The game sends
**context** (equipped item, surface underfoot, region) and **events** (footstep, shoot, ...); the sidecar
returns ready-to-play PCM from a speculative cache, with an authored clip as fallback. The whole
predict → pre-generate → cache → authored-fallback loop runs outside your render thread.

## 1. Start the sidecar

From the repo root (Python 3.12 + the project venv):

```
python scripts/run_sidecar.py                 # mock generator, 127.0.0.1:8765 (no GPU)
python scripts/run_sidecar.py --generator sa3 # neural SFX (needs the GPU model)
```

## 2. Add the package to your project

Either:
- **Copy** the `PredictiveSfx` folder into your project's `Assets/`, or
- Add it as a local package: in `Packages/manifest.json`, add
  `"org.sonoform.predictive-sfx": "file:<absolute-path-to>/unity/PredictiveSfx"`.

## 3. Hear it in 30 seconds

1. Make an empty GameObject, add an **Audio Source**, add the **Predictive Sfx Example** component.
2. Press Play. It sets a context (`gun` on `grass`) and fires footsteps + shots over the wire — you'll
   hear generated audio. (No Audio Source assigned? It adds one.)

The first GameObject that needs the service auto-creates a persistent `PredictiveSfxService`
(`DontDestroyOnLoad`). To set the host/port explicitly, add the **Predictive Sfx Service** component to a
GameObject yourself and fill in the fields, or call `PredictiveSfxService.Instance.Configure(host, port)`
before the first event.

## 4. Use it in your game

Minimal, from any script (main thread):

```csharp
using PredictiveSfx;

PredictiveSfxService.Instance.SetContext("gun", new[] { "grass" }, "field");   // on state change
PredictiveSfxService.Instance.Play("footstep", "grass", myAudioSource);        // on an event
PredictiveSfxService.Instance.Play("shoot", "metal", myAudioSource);
```

Or with components (no code):
- **Predictive Sfx Emitter** — set `eventType`/`material`, call `Emit()` (e.g. from an animation event,
  UnityEvent, or input handler). Optional `authoredFallback` plays if the sidecar is unreachable.
- **Predictive Sfx Footsteps** — raycasts down for the surface, fires footsteps on an interval, and pushes
  context automatically. Tag ground objects with **Predictive Sfx Surface** (`material = "wood"`, etc.).

## How it behaves

- **Off the main thread.** All socket I/O runs on a background worker; `AudioClip`s are built and **cached**
  on the main thread (in `Update`). A cached hit returns in sub-millisecond + transfer; a cold miss
  (`fallback`) is slower (~400 ms with `sa3`) — prebake/context-warming keeps misses rare.
- **Resilient.** Auto-connects, replays the last context on reconnect (the warmed cache lives on the
  server), keepalive-pings when idle, and plays your `authoredFallback` if the sidecar is down. Non-fatal
  server errors don't drop the connection.
- **Requires the sidecar running.** This package is the client half; the engine lives in the Python service.

See `docs/sidecar-protocol.md` in the repo for the wire protocol (and the Unreal C++ port notes).
