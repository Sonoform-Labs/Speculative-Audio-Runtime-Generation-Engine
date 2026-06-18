from __future__ import annotations

from dataclasses import dataclass

from .cache import VariantCache
from .generators.base import Generator
from .predictor import Prediction
from .types import CacheKey, GameEvent


@dataclass(slots=True)
class EventOutcome:
    event: GameEvent
    served: str
    hit: bool
    predicted: bool
    from_prebake: bool
    lead_s: float | None
    ready_in_time: bool
    variant_id: int | None
    latency_ms: float


@dataclass(slots=True)
class SimResult:
    outcomes: list[EventOutcome]
    n_predictions: int
    n_generations: int
    n_wasted_generations: int
    gen_latency_ms: float
    cache_peak_bytes: int
    sample_rate: int = 48_000


class PredictivePipeline:
    def __init__(self, generator: Generator, cache: VariantCache | None = None, *,
                 gen_latency_ms: float | None = None,
                 authored_assets: set | None = None,
                 fallback_latency_ms: float = 1.0):
        self.generator = generator
        self.cache = cache if cache is not None else VariantCache()
        self.gen_latency_ms = (gen_latency_ms if gen_latency_ms is not None
                               else generator.latency_ms)
        self.authored = authored_assets if authored_assets is not None else {"*"}
        self.fallback_latency_ms = fallback_latency_ms

    def prebake(self, keys, variants: int = 5, pin: bool = True) -> None:
        for key in keys:
            for v in range(variants):
                self.cache.add(key, self.generator.generate(key, variant=v), pin=pin)

    def _has_authored(self, key: CacheKey) -> bool:
        return "*" in self.authored or key in self.authored

    def run(self, events: list[GameEvent], predictions: list[Prediction]) -> SimResult:
        gen_lat_s = self.gen_latency_ms / 1000.0
        requested_keys = {e.key for e in events}

        gen_events: list[tuple[float, CacheKey, int]] = []
        pred_index: dict[tuple[CacheKey, float], list[float]] = {}
        per_key_variant: dict[CacheKey, int] = {}
        n_wasted = 0
        for p in predictions:
            v = per_key_variant.get(p.key, 0)
            per_key_variant[p.key] = v + 1
            gen_events.append((p.emit_time_s + gen_lat_s, p.key, v))
            pred_index.setdefault((p.key, p.target_time_s), []).append(p.emit_time_s)
            if p.key not in requested_keys:
                n_wasted += 1
        gen_events.sort(key=lambda x: x[0])

        prebaked_keys = self.cache.pinned_keys
        events_sorted = sorted(events, key=lambda e: e.time_s)

        gi = 0
        peak_bytes = self.cache.nbytes
        outcomes: list[EventOutcome] = []
        for e in events_sorted:
            while gi < len(gen_events) and gen_events[gi][0] <= e.time_s + 1e-9:
                _, k, v = gen_events[gi]
                self.cache.add(k, self.generator.generate(k, variant=v))
                gi += 1
            peak_bytes = max(peak_bytes, self.cache.nbytes)

            emits = pred_index.get((e.key, e.time_s))
            predicted = emits is not None
            lead_s: float | None = None
            if emits:
                in_time = [em for em in emits if em + gen_lat_s <= e.time_s + 1e-9]
                if in_time:
                    lead_s = e.time_s - min(in_time)
            from_prebake = e.key in prebaked_keys

            if self.cache.has(e.key):
                clip = self.cache.get(e.key)
                outcomes.append(EventOutcome(
                    e, "cached", True, predicted, from_prebake, lead_s, True,
                    clip.variant_id if clip else None, 0.0))
            elif self._has_authored(e.key):
                outcomes.append(EventOutcome(
                    e, "fallback", False, predicted, from_prebake, lead_s, False,
                    None, self.fallback_latency_ms))
            else:
                outcomes.append(EventOutcome(
                    e, "silence", False, predicted, from_prebake, lead_s, False, None, 0.0))

        return SimResult(
            outcomes=outcomes,
            n_predictions=len(predictions),
            n_generations=len(gen_events),
            n_wasted_generations=n_wasted,
            gen_latency_ms=self.gen_latency_ms,
            cache_peak_bytes=peak_bytes,
            sample_rate=getattr(self.generator, "sample_rate", 48_000),
        )
