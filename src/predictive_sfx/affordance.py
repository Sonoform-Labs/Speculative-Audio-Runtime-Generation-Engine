from __future__ import annotations

from dataclasses import dataclass

from .generators.base import Generator
from .types import CacheKey

ITEM_SOUNDS = {
    "gun": [("shoot", 0.9), ("reload", 0.5), ("empty_click", 0.2)],
    "plate": [("drop", 0.5), ("shatter", 0.5)],
}
ITEM_MATERIAL = {"gun": "metal", "plate": "ceramic"}
SURFACE_SOUNDS = [("footstep", "continuous", 0.9), ("impact", "impulse", 0.15)]


@dataclass(slots=True)
class GameContext:
    time_s: float
    equipped: str | None
    surfaces: list
    region: str = "default"


@dataclass(slots=True)
class Candidate:
    key: CacheKey
    kind: str
    source: str
    p_use: float


def candidates(ctx: GameContext) -> list:
    out = []
    if ctx.equipped in ITEM_SOUNDS:
        mat = ITEM_MATERIAL[ctx.equipped]
        for event_type, base in ITEM_SOUNDS[ctx.equipped]:
            out.append(Candidate(CacheKey(event_type, mat, "normal"), "impulse", "item", base))
    for i, surf in enumerate(ctx.surfaces):
        prox = 1.0 if i == 0 else 0.5
        for event_type, kind, base in SURFACE_SOUNDS:
            out.append(Candidate(CacheKey(event_type, surf, "normal"), kind, "surface", base * prox))
    return out


class AffordanceEngine:
    def __init__(self, generator: Generator, vram_budget_kb: float = 1600.0,
                 gpu_ms_budget: float = 3000.0, gen_cost_ms: float = 393.0,
                 latency_saved_ms: float = 400.0, tau: float = 0.05,
                 continuous_topk: int = 2, variants_per_key: int = 2):
        self.gen = generator
        self.vram_budget = int(vram_budget_kb * 1024)
        self.gpu_ms_budget = gpu_ms_budget
        self.gen_cost_ms = gen_cost_ms
        self.latency_saved_ms = latency_saved_ms
        self.tau = tau
        self.continuous_topk = continuous_topk
        self.variants_per_key = variants_per_key
        self.cache: dict = {}
        self.priority: dict = {}
        self.recent: dict = {}
        self.pinned: set = set()
        self.L = 0.0
        self.generated: set = set()
        self.fired: set = set()
        self.last_served = None
        self.stats = {"spec_gen": 0, "demand_gen": 0, "hits": 0, "fallback": 0,
                      "evicted": 0, "gpu_ms": 0.0, "peak_bytes": 0}

    def _bytes(self, clip):
        return int(getattr(clip, "nbytes", 0))

    def _used(self):
        return sum(self._bytes(c) for pool in self.cache.values() for c in pool)

    def _gate_thresh(self, gpu_pressure):
        base = self.gen_cost_ms / (self.gen_cost_ms + self.latency_saved_ms)
        return max(self.tau, base * gpu_pressure)

    def _evict_min_speculative(self):
        spec = [(self.priority.get(k, 0.0), k) for k in self.cache if k not in self.pinned]
        if not spec:
            return False
        h, k = min(spec)
        self.L = h
        del self.cache[k]
        self.priority.pop(k, None)
        self.recent.pop(k, None)
        self.stats["evicted"] += 1
        return True

    def on_context_change(self, ctx: GameContext, gpu_pressure: float = 1.0):
        cands = candidates(ctx)
        reachable = {c.key for c in cands}
        self.pinned = set(reachable)
        for k in list(self.cache):
            if k not in reachable:
                del self.cache[k]
                self.priority.pop(k, None)
                self.recent.pop(k, None)
                self.stats["evicted"] += 1

        thresh = self._gate_thresh(gpu_pressure)
        impulse_items = [c for c in cands if c.source == "item"]
        cont = sorted((c for c in cands if c.kind == "continuous"), key=lambda c: -c.p_use)[:self.continuous_topk]
        surface_other = [c for c in cands if c.source == "surface" and c.kind != "continuous"]
        pool_cands = impulse_items + cont + surface_other

        admitted = [c for c in pool_cands
                    if c.key not in self.cache
                    and ((c.source == "item" and c.p_use >= self.tau) or c.p_use >= thresh)]
        est_bytes = max(1, int(getattr(self.gen, "sample_rate", 44100) * 0.6 * 4))
        admitted.sort(key=lambda c: self.L + c.p_use * self.gen_cost_ms / est_bytes, reverse=True)

        spent = 0.0
        for c in admitted:
            pool = []
            for v in range(self.variants_per_key):
                if spent + self.gen_cost_ms > self.gpu_ms_budget:
                    break
                clip = self.gen.generate(c.key, variant=len(self.generated) * 8 + v)
                while self._used() + self._bytes(clip) > self.vram_budget:
                    if not self._evict_min_speculative():
                        break
                if self._used() + self._bytes(clip) > self.vram_budget:
                    break
                pool.append(clip)
                self.stats["spec_gen"] += 1
                self.stats["gpu_ms"] += self.gen_cost_ms
                spent += self.gen_cost_ms
            if pool:
                self.cache[c.key] = pool
                self.priority[c.key] = self.L + c.p_use * self.gen_cost_ms / max(1, self._bytes(pool[0]))
                self.generated.add(c.key)
        self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self._used())

    def on_event(self, key: CacheKey) -> str:
        self.fired.add(key)
        pool = self.cache.get(key)
        if pool:
            self.stats["hits"] += 1
            idx = (self.recent.get(key, -1) + 1) % len(pool)
            self.recent[key] = idx
            self.last_served = pool[idx]
            return "cached"
        self.stats["fallback"] += 1
        clip = self.gen.generate(key, variant=10_000)
        self.cache[key] = [clip]
        self.recent[key] = 0
        self.last_served = clip
        self.stats["demand_gen"] += 1
        self.stats["gpu_ms"] += self.gen_cost_ms
        self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self._used())
        return "fallback"

    def wasted(self):
        return len(self.generated - self.fired)


def make_session() -> list:
    T = []
    T.append((0.0, "context", GameContext(0.0, None, ["grass"], "field")))
    for t in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5):
        T.append((t, "event", CacheKey("footstep", "grass", "normal")))
    T.append((5.0, "context", GameContext(5.0, "gun", ["grass"], "field")))
    for t, et in ((6.0, "shoot"), (6.5, "shoot"), (7.0, "shoot"), (8.0, "reload"),
                  (9.0, "shoot"), (9.5, "empty_click")):
        T.append((t, "event", CacheKey(et, "metal", "normal")))
    for t in (10.0, 10.5, 11.0):
        T.append((t, "event", CacheKey("footstep", "grass", "normal")))
    T.append((12.0, "context", GameContext(12.0, "plate", ["grass"], "field")))
    T.append((14.0, "event", CacheKey("drop", "ceramic", "normal")))
    T.append((16.0, "context", GameContext(16.0, None, ["metal"], "room")))
    for t in (17.0, 17.5, 18.0, 18.5, 19.0, 19.5):
        T.append((t, "event", CacheKey("footstep", "metal", "normal")))
    T.append((18.7, "event", CacheKey("impact", "metal", "normal")))
    T.append((20.0, "event", CacheKey("explosion", "metal", "hard")))
    T.sort(key=lambda x: x[0])
    return T
