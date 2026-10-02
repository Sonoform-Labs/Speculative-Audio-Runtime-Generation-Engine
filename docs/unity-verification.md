# Unity verification

The one check that needs a human in the Editor. It covers the three integration risks: main-thread AudioClip build, native memory over a long session, and a clean worker-thread join on stop.

## Setup

1. Open the test project `D:\PSfxUnityTest` in Unity 2022.3.
2. The project reads the package straight from the repo through `Packages\manifest.json` (`"org.sonoform.sarge": "file:D:/Python/SARGE/unity/Sarge"`). The package ships its `.meta` files, so scene references survive edits on either side.
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

A prediction whose lead is shorter than the tier's generation cost gets dropped. The default tier now costs about 53 ms per clip on the RTX 3060 (74 ms round trip for a first hit over the wire), so the default `leadSeconds` of 0.6 s clears it with a wide margin. Confidence drops on sharp turns, so expect a few more fallbacks when juking.

## Phase 3, the engine question

Roadmap Phase 3 asks whether a kinematic character controller emits a usable contact impulse for a footfall, or whether it has to be derived. It gates the conditioning story, because the parametric path currently conditions on a rank computed from the audio itself, which makes any monotonicity result partly self-referential. A measured impulse in newton-seconds is independent of the audio and settles it.

### Run

1. Build the walk environment as above.
2. Add `SargeImpulseProbe` to the player object, the one carrying the `CharacterController`.
3. Set `mass` to the character mass you want the impulse expressed against. The default is 70.
4. Press Play. Walk, run, jump, and drop off ledges for about thirty seconds. Vary how hard you land.
5. Read the on-screen panel.

### What the panel answers

The top half is the Phase 3 answer. `ControllerColliderHit` carries collider, controller, gameObject, moveDirection, moveLength, normal, point, rigidbody and transform, and no impulse or force field of any kind. The panel reports whether `Collision.impulse` ever arrived, which it will not for a purely kinematic controller. If both stay negative, the conditioning input has to be derived, and that is the finding. The first run found exactly that: every landing came out at 337 Ns and every gait sample at 4.00 m/s, so this controller carries no continuous intensity.

The bottom half is the derived quantity: mass times the change in vertical velocity across the airborne to grounded edge, in newton-seconds, logged per landing.

### Then

1. Copy `sarge_impulse_probe.csv` from `Application.persistentDataPath` into `D:\Python\sarge-research\modal\`.
2. Run `python runpinned.py p1h_impulse.py sarge_impulse_probe.csv`.

That reports the impulse distribution, sweeps the parametric path over the full conditioning axis and over the band a typical landing actually occupies, and scores five descriptors for monotonicity. The number to watch is the ratio at the end. A low ratio means real play clusters into a narrow slice of the axis, so a player hears little variation even where the control is monotonic.

`p1h_impulse.py` has been validated end to end against `_synthetic_probe_check.csv`, so it runs as soon as real data exists.

### Bouncers, a real continuous impulse

The four `AutoCaptureBall` Rigidbodies in WalkArena already carry the probe in Collisions mode (menu **Sarge, Add Impulse Probe To Bouncers** adds it to any scene). Each probe writes one row per `OnCollisionEnter` to `sarge_impulse_probe_<object>.csv` with the object, time, `Collision.impulse` magnitude, relative speed, and the other collider.

1. Open `Assets/WalkArena.unity` and press Play. A panel at the bottom right lists each ball's hits and impulse range.
2. Click into the Game view and press **K** about ten times over a minute. Each press kicks every ball upward at a random speed, so the hits spread out.
3. Stop Play and run `python D:\Python\SARGE\scripts\analyze_impulse.py`. It reads `Application.persistentDataPath` by default and ends with a VERDICT block on whether impulse varies enough to drive soft, normal and hard.

All four balls weigh 1 kg, so impulse tracks speed almost exactly. Giving them different masses is the test of whether impulse adds anything beyond the speed bins `SargeAutoCapture` already uses.
