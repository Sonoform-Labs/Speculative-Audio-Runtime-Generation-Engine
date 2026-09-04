from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .types import CacheKey, GameEvent, Intensity, Material


@dataclass(slots=True)
class Prediction:
    key: CacheKey
    emit_time_s: float
    target_time_s: float
    confidence: float = 1.0

    @property
    def lead_s(self) -> float:
        return self.target_time_s - self.emit_time_s


class Predictor(ABC):
    @abstractmethod
    def predict(self, events: list[GameEvent]) -> list[Prediction]:
        raise NotImplementedError


class Tier1DeterministicPredictor(Predictor):

    def __init__(self, lead_ms: float = 250.0):
        self.lead_s = lead_ms / 1000.0

    def predict(self, events: list[GameEvent]) -> list[Prediction]:
        return [
            Prediction(e.key, max(0.0, e.time_s - self.lead_s), e.time_s, 1.0)
            for e in events
        ]


class KinematicGaitPredictor:
    def __init__(self, stride_m: float = 0.75, lead_ms: float = 300.0,
                 min_speed: float = 0.4, stab_alpha: float = 0.4,
                 event_type: str = "footstep", intensity: str = "normal",
                 context: str = "default"):
        self.stride_m = float(stride_m)
        self.lead_s = lead_ms / 1000.0
        self.min_speed = float(min_speed)
        self.stab_alpha = float(stab_alpha)
        self.event_type = event_type
        self.intensity = intensity
        self.context = context
        self.reset()

    def reset(self) -> None:
        self._t = None
        self._pos = None
        self._vel = None
        self._stab = 1.0
        self._dist = 0.0
        self._next_step_dist = self.stride_m
        self._step_count = 0
        self._emitted_step = -1

    def observe(self, t: float, pos, surface: str,
                upcoming_surface: str | None = None) -> Prediction | None:
        p = np.asarray(pos, dtype=np.float64)
        out: Prediction | None = None
        if self._t is not None and t > self._t:
            dt = t - self._t
            step = p - self._pos
            dist = float(np.linalg.norm(step))
            speed = dist / dt
            vel = step / dt
            stable = 1.0
            if self._vel is not None:
                nv = float(np.linalg.norm(vel))
                npv = float(np.linalg.norm(self._vel))
                if nv > 1e-6 and npv > 1e-6:
                    stable = max(0.0, float(np.dot(vel, self._vel) / (nv * npv)))
            self._stab = (1.0 - self.stab_alpha) * self._stab + self.stab_alpha * stable
            self._dist += dist
            self._vel = vel
            if speed >= self.min_speed:
                while self._next_step_dist - self._dist <= 0:
                    self._next_step_dist += self.stride_m
                    self._step_count += 1
                remaining = self._next_step_dist - self._dist
                ttn = remaining / speed
                if ttn <= self.lead_s and self._emitted_step < self._step_count:
                    surf = upcoming_surface or surface
                    key = CacheKey(self.event_type, surf, self.intensity, self.context)
                    out = Prediction(key, t, t + ttn, self._stab)
                    self._emitted_step = self._step_count
        self._t = t
        self._pos = p
        return out

    def run(self, samples) -> list[Prediction]:
        self.reset()
        preds: list[Prediction] = []
        for s in samples:
            t, pos, surface = s[0], s[1], s[2]
            upcoming = s[3] if len(s) > 3 else None
            p = self.observe(t, pos, surface, upcoming)
            if p is not None:
                preds.append(p)
        return preds


class EventSequencePredictor:
    def __init__(self, order: int = 2, top_k: int = 3, smoothing: float = 0.5,
                 min_total: int = 3, history_cap: int = 64):
        self.order = int(order)
        self.top_k = int(top_k)
        self.smoothing = float(smoothing)
        self.min_total = int(min_total)
        self.history_cap = int(history_cap)
        self.ngram: dict = {}
        self.unigram: dict = {}
        self.vocab: set = set()
        self.history: list = []

    def reset(self) -> None:
        self.history = []

    def observe(self, event_type: str, context: str = "default") -> None:
        self.vocab.add(event_type)
        key = (context, tuple(self.history[-self.order:]))
        d = self.ngram.setdefault(key, {})
        d[event_type] = d.get(event_type, 0) + 1
        u = self.unigram.setdefault(context, {})
        u[event_type] = u.get(event_type, 0) + 1
        self.history.append(event_type)
        if len(self.history) > self.history_cap:
            self.history = self.history[-self.history_cap:]

    def predict(self, context: str = "default") -> list:
        d = self.ngram.get((context, tuple(self.history[-self.order:])))
        total = sum(d.values()) if d else 0
        if not d or total < self.min_total:
            d = self.unigram.get(context, {})
            total = sum(d.values())
        if not d or total == 0:
            return []
        denom = total + self.smoothing * max(1, len(self.vocab))
        scored = sorted(((c, (n + self.smoothing) / denom) for c, n in d.items()),
                        key=lambda x: x[1], reverse=True)
        return scored[:self.top_k]


class NoisyPredictor(Predictor):

    def __init__(self, lead_ms: float = 250.0, lead_jitter_ms: float = 80.0,
                 recall: float = 0.9, key_accuracy: float = 0.85,
                 false_pos_rate: float = 0.1,
                 materials: list[str] | None = None,
                 intensities: list[str] | None = None,
                 seed: int = 0):
        self.lead_ms = lead_ms
        self.lead_jitter_ms = lead_jitter_ms
        self.recall = recall
        self.key_accuracy = key_accuracy
        self.false_pos_rate = false_pos_rate
        self.materials = materials or [m.value for m in Material]
        self.intensities = intensities or [i.value for i in Intensity]
        self.seed = seed

    def _mutate(self, key: CacheKey, rng: np.random.Generator) -> CacheKey:
        if rng.random() < 0.5:
            choices = [m for m in self.materials if m != key.material] or self.materials
            return CacheKey(key.event_type, str(rng.choice(choices)), key.intensity, key.context)
        choices = [i for i in self.intensities if i != key.intensity] or self.intensities
        return CacheKey(key.event_type, key.material, str(rng.choice(choices)), key.context)

    def predict(self, events: list[GameEvent]) -> list[Prediction]:
        rng = np.random.default_rng(self.seed)
        preds: list[Prediction] = []
        for e in events:
            if rng.random() > self.recall:
                continue
            lead = max(0.0, (self.lead_ms + rng.normal(0.0, self.lead_jitter_ms)) / 1000.0)
            emit = max(0.0, e.time_s - lead)
            key = e.key if rng.random() <= self.key_accuracy else self._mutate(e.key, rng)
            preds.append(Prediction(key, emit, e.time_s, self.key_accuracy))
            if rng.random() < self.false_pos_rate:
                preds.append(Prediction(self._mutate(e.key, rng), emit, e.time_s, 0.3))
        return preds
