from __future__ import annotations

import numpy as np

from .generators import MockGenerator
from .metrics import summarize
from .pipeline import PredictivePipeline
from .predictor import KinematicGaitPredictor, NoisyPredictor, Tier1DeterministicPredictor
from .types import GameEvent


def footstep_stream(n: int = 80, dt: float = 0.45, surfaces=("grass",),
                    intensity: str = "normal", unique: bool = False):
    out = []
    for i in range(n):
        surf = surfaces[(i // 8) % len(surfaces)]
        ctx = str(i) if unique else "default"
        out.append(GameEvent("footstep", surf, intensity, round(1.0 + i * dt, 3), context=ctx))
    return out


def _evaluate(events, predictions, gen_latency_ms):
    pipe = PredictivePipeline(MockGenerator(), gen_latency_ms=gen_latency_ms)
    return summarize(pipe.run(events, predictions))


def compare_predictors(events, gen_latency_ms: float = 120.0, lead_ms: float = 300.0, seed: int = 0):
    return {
        "none": _evaluate(events, [], gen_latency_ms),
        "oracle": _evaluate(events, Tier1DeterministicPredictor(lead_ms=lead_ms).predict(events), gen_latency_ms),
        "noisy": _evaluate(events, NoisyPredictor(lead_ms=lead_ms, seed=seed).predict(events), gen_latency_ms),
    }


def make_trajectory(speed: float, dur: float, dt: float = 0.05, heading_rate: float = 0.0):
    samples = []
    x = y = t = ang = 0.0
    for _ in range(int(dur / dt)):
        samples.append((t, (x, y, 0.0)))
        ang += heading_rate * dt
        x += speed * dt * float(np.cos(ang))
        y += speed * dt * float(np.sin(ang))
        t += dt
    return samples


def surface_seam(pos, seam: float = 6.0, a: str = "grass", b: str = "metal"):
    return a if pos[0] < seam else b


def _pos_at(samples, t):
    if t <= samples[0][0]:
        return np.asarray(samples[0][1])
    for i in range(1, len(samples)):
        t0, p0 = samples[i - 1]
        t1, p1 = samples[i]
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return np.asarray(p0) + f * (np.asarray(p1) - np.asarray(p0))
    return np.asarray(samples[-1][1])


def gait_scenario(speed: float = 1.4, dur: float = 14.0, heading_rate: float = 0.0,
                  surface_fn=surface_seam, stride_m: float = 0.75, lead_ms: float = 300.0,
                  lookahead_m: float = 0.6, gen_latency_ms: float = 120.0):
    samples = make_trajectory(speed, dur, heading_rate=heading_rate)
    pred = KinematicGaitPredictor(stride_m=stride_m, lead_ms=lead_ms)
    predictions, events = [], []
    prevp = None
    for t, pos in samples:
        p = np.asarray(pos)
        upcoming = surface_fn(pos)
        if prevp is not None:
            d = p - prevp
            nrm = float(np.linalg.norm(d))
            if nrm > 1e-9:
                upcoming = surface_fn(tuple(p + d / nrm * lookahead_m))
        r = pred.observe(t, pos, surface_fn(pos), upcoming)
        if r is not None:
            actual_surface = surface_fn(tuple(_pos_at(samples, r.target_time_s)))
            predictions.append(r)
            events.append(GameEvent("footstep", actual_surface, "normal", r.target_time_s))
        prevp = p
    return {
        "predicted": _evaluate(events, predictions, gen_latency_ms),
        "none": _evaluate(events, [], gen_latency_ms),
        "n_steps": len(events),
    }
