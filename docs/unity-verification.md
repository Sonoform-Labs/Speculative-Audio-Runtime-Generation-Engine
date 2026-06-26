# Unity in-editor verification (P4.1)

This is the one step that needs a human at the Editor — everything else is verified by the
119 passing Python tests + the benchmark. It validates the three integration risks the red-team
flagged: main-thread AudioClip build, native-memory eviction over a long session, and a clean
worker-thread join on stop.

## Setup (one-time)
1. Open the test project `D:\PSfxUnityTest` in Unity 2022.3.8f1.
2. The package is referenced from `D:\Python\Predictive-Neural-SFX\unity\PredictiveSfx`.
3. Make sure the sidecar venv exists at `D:\Python\Predictive-Neural-SFX\.venv` (the Launcher
   auto-starts `scripts\run_sidecar.py --generator sa3`). For a fast GPU-free check, edit the
   `PredictiveSfxLauncher` component's `arguments` to `scripts\run_sidecar.py --generator mock`.

## Build + run
1. Menu: **PSFX -> Build Walk Environment**. This builds `WalkArena.unity` with 4 colored surface
   tiles, a first-person player, the `PredictiveSfxLauncher`, and (now) the
   `PredictiveSfxMotionPredictor` attached to the Player automatically.
2. Press **Play**. The Launcher starts the sidecar (watch `D:\Python\Predictive-Neural-SFX\sidecar.log`
   for `sidecar READY`). The "Starting sound engine..." overlay clears once connected.

## What to verify
- **No console errors** on AudioClip.Create/SetData (main-thread clip build).
- **Walk around.** Footsteps should be served from cache — `PredictiveSfxService.Instance.CachedCount`
  climbs, `FallbackCount` stays low. The motion predictor pre-warms the surface you are *about to*
  step on (check `sidecar.log` for PREDICT activity ahead of each footfall).
- **Cross a surface boundary** (grass -> metal tile): the new surface should already be warm by the
  time you arrive (no fallback burst), because the predictor raycasts ahead.
- **Long session (a few minutes):** memory should be stable — the LRU caps at `maxCachedClips` (256)
  and Destroys evicted clips (no native-memory leak).
- **Stop Play:** clean exit, no hung-thread warning (worker joins on Shutdown).

## A/B demo (P4.3)
To show the value on/off:
- **OFF:** disable the `PredictiveSfxMotionPredictor` component on the Player (or set its
  `leadSeconds` below the sidecar's gen latency, ~0.4s for sa3). Footsteps will miss more (cold) and
  fall back. Watch `FallbackCount` climb.
- **ON:** re-enable it. Footsteps become cache hits; `CachedCount` dominates.

The quantitative A/B is already in `docs/benchmark-results.txt` (none = 0% hit, predicted ~100%; and
the lead-vs-latency sweep showing ready_in_time climbs once lead exceeds the 393ms SA3 latency).

## Notes
- `leadSeconds` on the motion predictor MUST exceed the sidecar's per-clip gen latency, or the
  server lead-gate drops the prediction (it can't finish in time) and the step falls back. Default
  0.6s > sa3's 0.393s. If you raise generation steps/latency, raise leadSeconds too.
- The predictor's confidence drops on sharp turns (EMA heading stability), so the ranker hedges
  during erratic movement — expect slightly more fallback when juking.
