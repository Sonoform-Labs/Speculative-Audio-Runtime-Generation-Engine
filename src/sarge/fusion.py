from __future__ import annotations

from dataclasses import dataclass

from .types import CacheKey


@dataclass(slots=True)
class FusedProposal:
    key: CacheKey
    eta_s: float
    p_use: float
    sources: tuple


class PredictionFusion:
    def __init__(self, calibration=None, min_ev: float = 0.0, gen_cost_ms: float = 393.0,
                 latency_saved_ms: float = 400.0, owner=None):
        self.calibration = calibration
        self.min_ev = float(min_ev)
        self.gen_cost_ms = float(gen_cost_ms)
        self.latency_saved_ms = float(latency_saved_ms)
        self.owner = owner if callable(owner) else (dict(owner) if owner else {})

    def _p(self, source: str, raw: float) -> float:
        if self.calibration is None:
            return max(0.0, min(1.0, float(raw)))
        return self.calibration.calibrate(source, raw)

    def _is_owner(self, source: str, key: CacheKey) -> bool:
        if callable(self.owner):
            return bool(self.owner(source, key))
        return self.owner.get(key.event_type) == source

    def _ev(self, p_use: float) -> float:
        return p_use * self.latency_saved_ms / self.gen_cost_ms

    def fuse(self, proposals) -> list:
        by_key: dict = {}
        for source, key, eta, raw in proposals:
            p = self._p(source, raw)
            e = by_key.get(key)
            if e is None:
                by_key[key] = {"eta": eta, "p_comp": 1.0 - p, "sources": [source],
                               "owner_eta": eta if self._is_owner(source, key) else None}
            else:
                e["p_comp"] *= (1.0 - p)
                e["sources"].append(source)
                e["eta"] = min(e["eta"], eta)
                if self._is_owner(source, key):
                    e["owner_eta"] = eta
        out = []
        for key, e in by_key.items():
            p_use = 1.0 - e["p_comp"]
            if self._ev(p_use) < self.min_ev:
                continue
            eta = e["owner_eta"] if e["owner_eta"] is not None else e["eta"]
            out.append(FusedProposal(key, eta, p_use, tuple(e["sources"])))
        out.sort(key=lambda fp: self._ev(fp.p_use), reverse=True)
        return out
