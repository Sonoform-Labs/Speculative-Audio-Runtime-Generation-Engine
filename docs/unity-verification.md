# Unity verification

The one check that needs a human in the Editor. It covers the three integration risks: main-thread AudioClip build, native memory over a long session, and a clean worker-thread join on stop.

## Setup

1. Open the test project `D:\PSfxUnityTest` in Unity 2022.3.
2. The package lives at `D:\Python\SARGE\unity\Sarge`, copied into the project's `Assets`.
3. The launcher auto-starts the sidecar from `D:\Python\SARGE`. For a GPU-free check, set the `SargeLauncher` arguments to `scripts\run_sidecar.py --generator mock`.

## Run

1. Menu: **Sarge, Build Walk Environment**. This builds a scene with four surface tiles, a first-person player, the launcher, and the motion predictor.
2. Press Play. Watch `sidecar.log` for `sidecar READY`. The starting overlay clears once connected.

## Verify

- No console errors during play.
- Walk around. Footsteps play, the HUD's cached and realtime counts climb, fallback stays 0.
- Cross a surface boundary. The next surface should already be warm when you arrive.
- Play for a few minutes. Memory stays flat, the clip cache caps at 256 and destroys evicted clips.
- Stop Play. Clean exit, no hung-thread warning.

## A/B

Disable the `SargeMotionPredictor` component on the player to see misses climb, re-enable it to see cache hits dominate.

## Notes

`leadSeconds` on the motion predictor must exceed the sidecar's generation latency or the prediction is dropped. The default 0.6 s clears the roughly 180 ms default tier. Confidence drops on sharp turns, so expect a few more fallbacks when juking.

## Phase 3, the engine question

Roadmap Phase 3 asks whether a kinematic character controller emits a usable contact impulse for a footfall, or whether it has to be derived. It gates the conditioning story, because the parametric path currently conditions on a rank computed from the audio itself, which makes any monotonicity result partly self-referential. A measured impulse in newton-seconds is independent of the audio and settles it.

### Run

1. Build the walk environment as above.
2. Add `SargeImpulseProbe` to the player object, the one carrying the `CharacterController`.
3. Set `mass` to the character mass you want the impulse expressed against. The default is 70.
4. Press Play. Walk, run, jump, and drop off ledges for about thirty seconds. Vary how hard you land.
5. Read the on-screen panel.

### What the panel answers

The top half is the Phase 3 answer. `ControllerColliderHit` carries collider, controller, gameObject, moveDirection, moveLength, normal, point, rigidbody and transform, and no impulse or force field of any kind. The panel reports whether `Collision.impulse` ever arrived, which it will not for a purely kinematic controller. If both stay negative, the conditioning input is derived rather than solver-provided, and that is the finding.

The bottom half is the derived quantity: mass times the change in vertical velocity across the airborne to grounded edge, in newton-seconds, logged per landing.

### Then

1. Copy `sarge_impulse_probe.csv` from `Application.persistentDataPath` into `D:\Python\sarge-research\modal\`.
2. Run `python runpinned.py p1h_impulse.py sarge_impulse_probe.csv`.

That reports the impulse distribution, sweeps the parametric path over the full conditioning axis and over the band a typical landing actually occupies, and scores five descriptors for monotonicity. The number to watch is the ratio at the end. A low ratio means real play clusters into a narrow slice of the axis, so a player hears little variation even where the control is monotonic.

`p1h_impulse.py` has been validated end to end against `_synthetic_probe_check.csv`, so it runs as soon as real data exists.
