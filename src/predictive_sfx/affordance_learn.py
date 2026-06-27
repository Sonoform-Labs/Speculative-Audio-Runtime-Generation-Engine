from __future__ import annotations

from collections import Counter

from .affordance import Candidate
from .types import CacheKey

_CONTINUOUS = {"footstep", "engine", "ambient", "scrape", "slide"}


class RuntimeAffordanceLearner:
    def __init__(self, min_materials_for_surface: int = 2):
        self.surfaces: set = set()
        self.event_materials: dict = {}
        self.event_count: Counter = Counter()
        self.total: int = 0
        self.min_materials_for_surface = int(min_materials_for_surface)

    def observe_context(self, ctx) -> None:
        for s in (getattr(ctx, "surfaces", None) or []):
            self.surfaces.add(s)

    def observe_event(self, key: CacheKey) -> None:
        self.event_materials.setdefault(key.event_type, Counter())[key.material] += 1
        self.event_count[key.event_type] += 1
        self.total += 1

    def _kind(self, event_type: str) -> str:
        return "continuous" if event_type in _CONTINUOUS else "impulse"

    def _surface_driven(self, event_type: str) -> bool:
        mats = set(self.event_materials.get(event_type, ()))
        return len(mats & self.surfaces) >= self.min_materials_for_surface

    def candidates(self, ctx) -> list:
        if self.total == 0:
            return []
        ctx_surfaces = list(getattr(ctx, "surfaces", None) or [])
        out = []
        for event_type, cnt in self.event_count.items():
            p = cnt / self.total
            if self._surface_driven(event_type) and ctx_surfaces:
                for surf in ctx_surfaces:
                    out.append(Candidate(CacheKey(event_type, surf, "normal"),
                                         self._kind(event_type), "learned", p))
            else:
                mat = self.event_materials[event_type].most_common(1)[0][0]
                out.append(Candidate(CacheKey(event_type, mat, "normal"),
                                     self._kind(event_type), "learned", p))
        return out
