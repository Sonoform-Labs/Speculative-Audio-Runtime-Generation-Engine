# Speculative Audio Runtime Generation Engine

Generates game sound effects in real time, predicting events before they happen and caching every result in the instance of repeat events.

Games are shipped with dozens of variants per sound to avoid repetition fatigue. SAGE generates them on the fly instead, so the game isn't limited to the sounds it came with.

## How it works

A drop-in Unity package streams game events to a local Python sidecar which predicts upcoming sounds from motion, physics, and player actions, generates them with a diffusion model, and caches each result by event, material, intensity, and context. The first instance of generating a sound effect takes about 180 ms on an RTX 3060 with every other one afterwards being instant. Furthermore, runtime learner picks up a game's sound vocabulary from play and covers surfaces.

## Quickstart

Sidecar:

```bash
pip install -e .
python scripts/run_sidecar.py --realtime
```

Unity:

1. Copy `unity/Sage` into your project's `Assets` folder.
2. Add `SageService` to a scene object.
3. Tag walkable meshes with `SageSurface` and fire events from your controller.
4. Press Play. See `unity/Sage/Samples~` for a working first-person scene.

## Generation tiers

Choose with `--tier` or the `SAGE_GEN_TIER` environment variable:

| tier | sampler / steps | first-hit latency | notes |
|---|---|---|---|
| quality | euler @8 | ~400 ms | highest quality |
| default | euler @3 | ~180 ms | medium quality but faster |
| fast | euler @2 | ~135 ms | fastest but softer transients |

## Model

Stable Audio 3 Small SFX, 44.1 kHz, fp16 on a consumer GPU.

## License

MIT
