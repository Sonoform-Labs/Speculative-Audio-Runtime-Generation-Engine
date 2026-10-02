# SARGE Unity package

Real-time generated sound effects over a local sidecar. The game sends context and events, the sidecar answers with ready-to-play PCM. Nothing is pre-made, every sound is generated.

## 1. Start the sidecar

From the repo root:

```
python scripts/run_sidecar.py --realtime
```

Use `--generator mock` for a GPU-free placeholder.

## 2. Add the package

Copy the `Sarge` folder into your project's `Assets`, or add it as a local package in `Packages/manifest.json`:

```
"org.sonoform.sarge": "file:<path-to>/unity/Sarge"
```

The package ships its `.meta` files, so script GUIDs stay the same either way. A scene saved against one copy keeps its component references in the other.

## 3. Hear it

1. Make an empty GameObject, add an Audio Source and the **Sarge Example** component.
2. Press Play. It sets a context and fires footsteps and shots over the wire.

The first object that needs the service auto-creates a persistent `SargeService`. To set host or port, add the component yourself or call `SargeService.Instance.Configure(host, port)` before the first event.

## 4. Use it

```csharp
using Sarge;

SargeService.Instance.SetContext("gun", new[] { "grass" }, "field");
SargeService.Instance.Play("footstep", "grass", myAudioSource);
```

Or with components and no code:

- **Sarge Emitter** fires a fixed event from `Emit()`, with an optional fallback clip.
- **Sarge Footsteps** raycasts for the surface and fires steps on an interval.
- **Sarge Surface** tags ground objects with a material name.

## Impulse probe

`SargeImpulseProbe` measures how hard things hit, for tuning the intensity axis.

- On a character (a `CharacterController`, a kinematic Rigidbody, or no physics at all) it logs landings and gait to `sarge_impulse_probe.csv`.
- On a dynamic Rigidbody it logs one row per `OnCollisionEnter` to `sarge_impulse_probe_<object>.csv`: object, time, `Collision.impulse` magnitude, relative speed, and the other collider.
- Set **Mode** to force either behavior. With the legacy Input Manager, **K** kicks each collision probe's body upward at a random speed between **Kick Min** and **Kick Max**.

Both files land in `Application.persistentDataPath`. `python scripts/analyze_impulse.py` reads the per-object files and reports whether impulse varies enough to drive soft, normal and hard.

## Behavior

Socket I/O runs on a background worker and clips are built on the main thread. The service auto-reconnects, replays the last context, pings while idle, and plays your fallback clip when the sidecar is down. See `docs/sidecar-protocol.md` for the wire format.
