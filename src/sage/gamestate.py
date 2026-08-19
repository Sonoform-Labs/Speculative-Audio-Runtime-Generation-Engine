from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .predictor import Prediction
from .types import CacheKey, GameEvent

_REGIONS = [("grass", 6.0), ("stone", 8.0), ("wood", 5.0), ("metal", 4.0), ("stone", 6.0), ("grass", 8.0)]


@dataclass(slots=True)
class StateFrame:
    time_s: float
    speed: float
    gait_phase: float
    material: str
    moving: bool


@dataclass(slots=True)
class GameStateTrace:
    frames: list
    events: list


def _material_at(pos, regions):
    x = 0.0
    for mat, length in regions:
        if pos < x + length:
            return mat
        x += length
    return regions[-1][0]


def step_rate(speed):
    return 0.0 if speed < 0.05 else min(3.2, 1.4 + 0.55 * speed)


def intensity_for(speed):
    if speed < 1.2:
        return "soft"
    if speed < 2.4:
        return "normal"
    return "hard"


def simulate(seed: int = 0, fps: int = 60, duration_s: float = 60.0, regions=None) -> GameStateTrace:
    rng = np.random.default_rng(seed)
    regions = regions or _REGIONS
    total = sum(length for _, length in regions)
    dt = 1.0 / fps
    frames, events = [], []
    pos, speed, phase, t = 0.0, 0.0, 0.0, 0.0
    target = 1.6
    for _ in range(int(duration_s * fps)):
        if rng.random() < 0.01:
            target = float(rng.choice([0.0, 1.4, 1.8, 2.8, 3.2]))
        speed = max(0.0, speed + float(np.clip(target - speed, -3.0 * dt, 3.0 * dt)))
        pos = (pos + speed * dt) % total
        mat = _material_at(pos, regions)
        moving = speed > 0.05
        phase += step_rate(speed) * dt
        if phase >= 1.0 and moving:
            phase -= 1.0
            events.append(GameEvent("footstep", mat, intensity_for(speed), round(t, 4)))
        frames.append(StateFrame(round(t, 4), float(speed), float(phase), mat, moving))
        t += dt
    return GameStateTrace(frames, events)


class AnticipatoryPredictor:
    def __init__(self, horizon_ms: float = 350.0, refractory_ms: float = 150.0):
        self.horizon_s = horizon_ms / 1000.0
        self.refractory_s = refractory_ms / 1000.0

    def predict(self, trace: GameStateTrace) -> list[Prediction]:
        preds: list[Prediction] = []
        last_target = -1.0
        for fr in trace.frames:
            if not fr.moving:
                continue
            sr = step_rate(fr.speed)
            if sr <= 0.0:
                continue
            time_to_next = (1.0 - fr.gait_phase) / sr
            if time_to_next > self.horizon_s:
                continue
            target = fr.time_s + time_to_next
            if last_target >= 0.0 and target - last_target < self.refractory_s:
                continue
            key = CacheKey("footstep", fr.material, intensity_for(fr.speed))
            preds.append(Prediction(key, fr.time_s, round(target, 4), 0.9))
            last_target = target
        return preds
