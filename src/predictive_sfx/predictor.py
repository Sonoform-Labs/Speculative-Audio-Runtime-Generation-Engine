"""Predictors: turn a stream of game events into speculative pre-generation requests.

A prediction says "I think event with this key will fire at ~target_time, and
I'm telling you now (emit_time)". The gap (target - emit) is the lead time the
generator gets to have audio ready. Two predictors model the two tiers from the
feasibility study:

  * Tier1DeterministicPredictor - perfect key, fixed lead. Models footsteps
    scheduled off the *active* locomotion clip via anim-notify markers.
  * NoisyPredictor - tunable recall / key-accuracy / false-positive rate and
    lead jitter. Models the ML intent tier (and lets you stress hit-rate).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .types import CacheKey, GameEvent, Intensity, Material


@dataclass(slots=True)
class Prediction:
    key: CacheKey
    emit_time_s: float       # when the prediction is made
    target_time_s: float     # when the event is expected to fire
    confidence: float = 1.0

    @property
    def lead_s(self) -> float:
        return self.target_time_s - self.emit_time_s


class Predictor(ABC):
    @abstractmethod
    def predict(self, events: list[GameEvent]) -> list[Prediction]:
        """Return speculative pre-generation requests for the given timeline."""
        raise NotImplementedError


class Tier1DeterministicPredictor(Predictor):
    """Perfect key, fixed lead time (anim-notify-style scheduling)."""

    def __init__(self, lead_ms: float = 250.0):
        self.lead_s = lead_ms / 1000.0

    def predict(self, events: list[GameEvent]) -> list[Prediction]:
        return [
            Prediction(e.key, max(0.0, e.time_s - self.lead_s), e.time_s, 1.0)
            for e in events
        ]


class NoisyPredictor(Predictor):
    """ML-tier model: imperfect recall, key accuracy, and spurious guesses."""

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
        """Return a *wrong* key by swapping material or intensity."""
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
                continue  # predictor simply missed this event -> it will be a fallback
            lead = max(0.0, (self.lead_ms + rng.normal(0.0, self.lead_jitter_ms)) / 1000.0)
            emit = max(0.0, e.time_s - lead)
            key = e.key if rng.random() <= self.key_accuracy else self._mutate(e.key, rng)
            preds.append(Prediction(key, emit, e.time_s, self.key_accuracy))
            if rng.random() < self.false_pos_rate:
                # a wasted speculative guess for a key that won't be requested
                preds.append(Prediction(self._mutate(e.key, rng), emit, e.time_s, 0.3))
        return preds
