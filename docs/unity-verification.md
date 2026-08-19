# Unity verification

The one check that needs a human in the Editor. It covers the three integration risks: main-thread AudioClip build, native memory over a long session, and a clean worker-thread join on stop.

## Setup

1. Open the test project `D:\PSfxUnityTest` in Unity 2022.3.
2. The package lives at `D:\Python\SAGE\unity\Sage`, copied into the project's `Assets`.
3. The launcher auto-starts the sidecar from `D:\Python\SAGE`. For a GPU-free check, set the `SageLauncher` arguments to `scripts\run_sidecar.py --generator mock`.

## Run

1. Menu: **Sage, Build Walk Environment**. This builds a scene with four surface tiles, a first-person player, the launcher, and the motion predictor.
2. Press Play. Watch `sidecar.log` for `sidecar READY`. The starting overlay clears once connected.

## Verify

- No console errors during play.
- Walk around. Footsteps play, the HUD's cached and realtime counts climb, fallback stays 0.
- Cross a surface boundary. The next surface should already be warm when you arrive.
- Play for a few minutes. Memory stays flat, the clip cache caps at 256 and destroys evicted clips.
- Stop Play. Clean exit, no hung-thread warning.

## A/B

Disable the `SageMotionPredictor` component on the player to see misses climb, re-enable it to see cache hits dominate.

## Notes

`leadSeconds` on the motion predictor must exceed the sidecar's generation latency or the prediction is dropped. The default 0.6 s clears the roughly 180 ms default tier. Confidence drops on sharp turns, so expect a few more fallbacks when juking.
