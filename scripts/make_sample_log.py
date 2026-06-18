"""Generate a sample movement log: a player walking/running across surfaces.

Footsteps follow a gait cadence; the surface material changes in segments; a few
impacts are sprinkled in. Deterministic given --seed.
"""
import _bootstrap  # noqa: F401  (adds src/ to path)

import argparse
import os

import numpy as np

from predictive_sfx.logs import save_events
from predictive_sfx.types import GameEvent

# Each segment: (material, duration_s, gait) where gait is seconds between steps.
_SEGMENTS = [
    ("grass", 8.0, 0.52),
    ("dirt", 6.0, 0.50),
    ("stone", 10.0, 0.46),   # speeds up
    ("wood", 8.0, 0.44),
    ("metal", 6.0, 0.42),    # running
    ("stone", 8.0, 0.48),
    ("grass", 10.0, 0.54),   # slows down
]


def build(seed: int = 0) -> list[GameEvent]:
    rng = np.random.default_rng(seed)
    events: list[GameEvent] = []
    t = 0.5
    for material, dur, gait in _SEGMENTS:
        seg_end = t + dur
        while t < seg_end:
            # intensity correlates with cadence: faster gait -> harder steps
            if gait <= 0.43:
                inten = rng.choice(["normal", "hard"], p=[0.4, 0.6])
            elif gait <= 0.50:
                inten = rng.choice(["soft", "normal", "hard"], p=[0.2, 0.6, 0.2])
            else:
                inten = rng.choice(["soft", "normal"], p=[0.6, 0.4])
            jitter = rng.normal(0.0, 0.015)  # human cadence isn't a metronome
            events.append(GameEvent("footstep", material, str(inten), round(t, 4)))
            t += gait + jitter
        # an impact at most segment boundaries (jump-land / bump)
        if rng.random() < 0.7:
            events.append(GameEvent("impact", material, "hard", round(seg_end - 0.05, 4)))

    events.sort(key=lambda e: e.time_s)
    return events


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=os.path.join("data", "movement_logs", "sample_run.json"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    events = build(args.seed)
    meta = {"seed": args.seed, "n_events": len(events),
            "duration_s": round(events[-1].time_s, 2) if events else 0.0,
            "description": "synthetic walk/run across surfaces"}
    save_events(args.out, events, meta)
    print(f"Wrote {len(events)} events ({meta['duration_s']}s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
