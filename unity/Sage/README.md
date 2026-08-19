# SAGE Unity package

Real-time generated sound effects over a local sidecar. The game sends context and events, the sidecar answers with ready-to-play PCM. Nothing is pre-made, every sound is generated.

## 1. Start the sidecar

From the repo root:

```
python scripts/run_sidecar.py --realtime
```

Use `--generator mock` for a GPU-free placeholder.

## 2. Add the package

Copy the `Sage` folder into your project's `Assets`, or add it as a local package in `Packages/manifest.json`:

```
"org.sonoform.sage": "file:<path-to>/unity/Sage"
```

## 3. Hear it

1. Make an empty GameObject, add an Audio Source and the **Sage Example** component.
2. Press Play. It sets a context and fires footsteps and shots over the wire.

The first object that needs the service auto-creates a persistent `SageService`. To set host or port, add the component yourself or call `SageService.Instance.Configure(host, port)` before the first event.

## 4. Use it

```csharp
using Sage;

SageService.Instance.SetContext("gun", new[] { "grass" }, "field");
SageService.Instance.Play("footstep", "grass", myAudioSource);
```

Or with components and no code:

- **Sage Emitter** fires a fixed event from `Emit()`, with an optional fallback clip.
- **Sage Footsteps** raycasts for the surface and fires steps on an interval.
- **Sage Surface** tags ground objects with a material name.

## Behavior

Socket I/O runs on a background worker and clips are built on the main thread. The service auto-reconnects, replays the last context, pings while idle, and plays your fallback clip when the sidecar is down. See `docs/sidecar-protocol.md` for the wire format.
