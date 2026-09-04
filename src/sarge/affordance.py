from __future__ import annotations

from dataclasses import dataclass

from .generators.base import Generator
from .generators.mock import MockGenerator
from .types import AudioClip, CacheKey

ITEM_SOUNDS = {
    "gun": [("shoot", 0.9), ("reload", 0.5), ("empty_click", 0.2)],
    "plate": [("drop", 0.5), ("shatter", 0.5)],
}
ITEM_MATERIAL = {"gun": "metal", "plate": "ceramic"}
SURFACE_SOUNDS = [("footstep", "continuous", 0.9), ("impact", "impulse", 0.15),
                  ("jump", "impulse", 0.35), ("land", "impulse", 0.4)]


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
                 continuous_topk: int = 2, variants_per_key: int = 2,
                 target_variants: int = 6, high_rate_variants: int = 8,
                 high_rate_events=None, authored=None, fallback_generator=None, learner=None,
                 router=None, realtime=False):
        self.gen = generator
        self.learner = learner
        self.router = router
        self.realtime = bool(realtime)
        self.vram_budget = int(vram_budget_kb * 1024)
        self.gpu_ms_budget = gpu_ms_budget
        self.gen_cost_ms = gen_cost_ms
        self.latency_saved_ms = latency_saved_ms
        self.tau = tau
        self.continuous_topk = continuous_topk
        self.variants_per_key = int(variants_per_key)
        self.target_variants = max(int(variants_per_key), int(target_variants))
        self.high_rate_variants = max(self.target_variants, int(high_rate_variants))
        self.high_rate_events = (set(high_rate_events) if high_rate_events is not None
                                 else {"footstep", "shoot", "gunshot"})
        self.authored = set(authored) if authored is not None else {"*"}
        self.fallback_gen = fallback_generator or MockGenerator(
            sample_rate=int(getattr(generator, "sample_rate", 48000)))
        self.cache: dict = {}
        self.priority: dict = {}
        self.recent: dict = {}
        self.fallback_recent: dict = {}
        self.want: dict = {}
        self.demand: list = []
        self.demand_set: set = set()
        self.demand_cap: int = 128
        self.inflight: set = set()
        self.eta: dict = {}
        self.grace_s: float = 0.5
        self._now: float = 0.0
        self.pinned: set = set()
        self.persistent: set = set()
        self.L = 0.0
        self.generated: set = set()
        self.fired: set = set()
        self.last_served = None
        self.stats = {"spec_gen": 0, "demand_gen": 0, "realtime": 0, "hits": 0, "fallback": 0,
                      "silence": 0, "evicted": 0, "gpu_ms": 0.0, "peak_bytes": 0}

    def _bytes(self, clip):
        return int(getattr(clip, "nbytes", 0))

    def _used(self):
        return sum(self._bytes(c) for pool in self.cache.values() for c in pool)

    def _target_variants(self, key) -> int:
        if key.event_type in self.high_rate_events:
            return self.high_rate_variants
        return self.target_variants

    def _gate_thresh(self, gpu_pressure):
        base = self.gen_cost_ms / (self.gen_cost_ms + self.latency_saved_ms)
        return max(self.tau, base * gpu_pressure)

    def _imminent(self, key) -> bool:
        e = self.eta.get(key)
        return e is not None and 0.0 <= e - self._now <= self.grace_s

    def _evict_min_speculative(self):
        spec = [(self.priority.get(k, 0.0), k) for k in self.cache
                if k not in self.pinned and not self._imminent(k)]
        if not spec:
            spec = [(self.priority.get(k, 0.0), k) for k in self.cache if k not in self.pinned]
        if not spec:
            return False
        h, k = min(spec, key=lambda t: t[0])
        self.L = h
        del self.cache[k]
        self.priority.pop(k, None)
        self.recent.pop(k, None)
        self.want.pop(k, None)
        self.eta.pop(k, None)
        self.stats["evicted"] += 1
        return True

    def _add_clips(self, batch, stat: str = "spec_gen"):
        if not batch:
            return
        clips = self.gen.generate_batch(list(batch))
        for (key, _variant), clip in zip(batch, clips):
            if clip is None:
                continue
            while self._used() + self._bytes(clip) > self.vram_budget:
                if not self._evict_min_speculative():
                    break
            if self._used() + self._bytes(clip) > self.vram_budget:
                break
            self.cache.setdefault(key, []).append(clip)
            self.generated.add(key)
            self.stats[stat] += 1
            self.stats["gpu_ms"] += self.gen_cost_ms
        self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self._used())

    def on_context_change(self, ctx: GameContext, gpu_pressure: float = 1.0, now: float | None = None):
        if now is not None:
            self._now = now
        if self.learner is not None:
            self.learner.observe_context(ctx)
        if self.realtime:
            return
        if self.learner is not None:
            cands = self.learner.candidates(ctx)
        else:
            cands = candidates(ctx)
        if self.router is not None:
            cands = [c for c in cands if self.router.is_prefetch(c.key.event_type)]
        reachable = {c.key for c in cands}
        self.pinned = set(reachable) | self.persistent
        for k in list(self.cache):
            if k not in reachable and k not in self.persistent:
                if self._imminent(k):
                    continue
                del self.cache[k]
                self.priority.pop(k, None)
                self.recent.pop(k, None)
                self.want.pop(k, None)
                self.eta.pop(k, None)
                self.stats["evicted"] += 1

        thresh = self._gate_thresh(gpu_pressure)
        cont = sorted((c for c in cands if c.kind == "continuous"), key=lambda c: -c.p_use)[:self.continuous_topk]
        other = [c for c in cands if c.kind != "continuous"]
        pool_cands = cont + other

        admitted = [c for c in pool_cands
                    if (c.source in ("item", "learned") and c.p_use >= self.tau) or c.p_use >= thresh]
        admitted.sort(key=lambda c: c.p_use, reverse=True)

        spent = 0.0
        batch = []
        for c in admitted:
            self.want[c.key] = max(self.want.get(c.key, 0), self._target_variants(c.key))
            self.priority[c.key] = self.L + c.p_use
            have = len(self.cache.get(c.key, ()))
            for v in range(have, self.variants_per_key):
                if spent + self.gen_cost_ms > self.gpu_ms_budget:
                    break
                batch.append((c.key, v))
                spent += self.gen_cost_ms
        self._add_clips(batch)

    def _has_evictable(self):
        return any(k not in self.pinned for k in self.cache)

    def _est_bytes(self):
        for pool in self.cache.values():
            if pool:
                return self._bytes(pool[0])
        return int(getattr(self.gen, "sample_rate", 44100) * 0.6 * 4)

    def deepen_step(self):
        for key in list(self.pinned):
            have = len(self.cache.get(key, ()))
            if have >= self.want.get(key, 0):
                continue
            if self._used() + self._est_bytes() > self.vram_budget and not self._has_evictable():
                continue
            self._add_clips([(key, have)])
            if len(self.cache.get(key, ())) > have:
                return True
        return False

    def plan_batch(self, max_items: int = 1):
        batch = []
        while self.demand and len(batch) < max_items:
            key = self.demand.pop(0)
            self.demand_set.discard(key)
            if key in self.cache:
                continue
            self.want[key] = max(self.want.get(key, 0), self.variants_per_key)
            self.priority[key] = self.L + 0.5
            have = len(self.cache.get(key, ()))
            for v in range(have, self.variants_per_key):
                item = (key, v)
                if item in self.inflight:
                    continue
                self.inflight.add(item)
                batch.append(item)
                if len(batch) >= max_items:
                    break
        if batch:
            return batch, "demand_gen"
        for key in list(self.pinned):
            have = len(self.cache.get(key, ()))
            if have >= self.want.get(key, 0):
                continue
            if self._used() + self._est_bytes() > self.vram_budget and not self._has_evictable():
                continue
            item = (key, have)
            if item in self.inflight:
                continue
            self.inflight.add(item)
            batch.append(item)
            if len(batch) >= max_items:
                break
        return batch, "spec_gen"

    def commit_clips(self, batch, clips, stat: str = "spec_gen"):
        for (key, _variant), clip in zip(batch, clips):
            self.inflight.discard((key, _variant))
            if clip is None:
                continue
            while self._used() + self._bytes(clip) > self.vram_budget:
                if not self._evict_min_speculative():
                    break
            if self._used() + self._bytes(clip) > self.vram_budget:
                continue
            self.cache.setdefault(key, []).append(clip)
            self.generated.add(key)
            self.stats[stat] += 1
            self.stats["gpu_ms"] += self.gen_cost_ms
        for item in batch:
            self.inflight.discard(item)
        self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self._used())

    def _set_prompts(self, prompts):
        g = getattr(self.gen, "inner", self.gen)
        pm = getattr(g, "prompts", None)
        if pm is not None and prompts:
            pm.update(prompts)

    def prewarm(self, keys, prompts=None):
        if prompts:
            self._set_prompts(prompts)
        for key in keys:
            self.pinned.add(key)
            self.persistent.add(key)
            self.want[key] = max(self.want.get(key, 0), self._target_variants(key))
            self.priority[key] = 1.0
        batch = []
        for key in keys:
            have = len(self.cache.get(key, ()))
            for v in range(have, self.variants_per_key):
                batch.append((key, v))
        self._add_clips(batch)

    def submit_predictions(self, proposals, now: float | None = None) -> dict:
        if now is not None:
            self._now = now
        spent = 0.0
        batch = []
        gated = 0
        for fp in proposals:
            if fp.eta_s * 1000.0 < self.gen_cost_ms:
                gated += 1
                continue
            self.eta[fp.key] = self._now + fp.eta_s
            self.want[fp.key] = max(self.want.get(fp.key, 0), self._target_variants(fp.key))
            self.priority[fp.key] = self.L + fp.p_use
            have = len(self.cache.get(fp.key, ()))
            for v in range(have, self.variants_per_key):
                if spent + self.gen_cost_ms > self.gpu_ms_budget:
                    break
                batch.append((fp.key, v))
                spent += self.gen_cost_ms
        self._add_clips(batch)
        return {"generated": len(batch), "lead_gated": gated}

    def register_predictions(self, proposals, now: float | None = None) -> dict:
        if now is not None:
            self._now = now
        queued = 0
        gated = 0
        for fp in proposals:
            if fp.eta_s * 1000.0 < self.gen_cost_ms:
                gated += 1
                continue
            self.eta[fp.key] = self._now + fp.eta_s
            self.want[fp.key] = max(self.want.get(fp.key, 0), self._target_variants(fp.key))
            self.priority[fp.key] = self.L + fp.p_use
            if fp.key not in self.cache and fp.key not in self.demand_set:
                self._queue_demand(fp.key)
                queued += 1
        return {"queued": queued, "lead_gated": gated}

    def _has_authored(self, key: CacheKey) -> bool:
        return "*" in self.authored or key in self.authored

    def _authored_clip(self, key: CacheKey):
        v = self.fallback_recent.get(key, -1) + 1
        self.fallback_recent[key] = v
        return self.fallback_gen.generate(key, variant=v)

    def _queue_demand(self, key: CacheKey):
        if key in self.cache or key in self.demand_set:
            return
        if len(self.demand) >= self.demand_cap:
            old = self.demand.pop(0)
            self.demand_set.discard(old)
        self.demand_set.add(key)
        self.demand.append(key)

    def drain_demand(self, max_jobs: int = 1) -> bool:
        did = False
        for _ in range(max_jobs):
            if not self.demand:
                break
            key = self.demand.pop(0)
            self.demand_set.discard(key)
            if key in self.cache:
                continue
            self.want[key] = max(self.want.get(key, 0), self.variants_per_key)
            self.priority[key] = self.L + 0.5
            before = len(self.cache.get(key, ()))
            self._add_clips([(key, v) for v in range(self.variants_per_key)], stat="demand_gen")
            if len(self.cache.get(key, ())) > before:
                did = True
        return did

    def _realtime_generate(self, key: CacheKey) -> AudioClip:
        clip = self.gen.generate(key, variant=0)
        while self._used() + self._bytes(clip) > self.vram_budget:
            if not self._evict_min_speculative():
                break
        if self._used() + self._bytes(clip) <= self.vram_budget:
            self.cache.setdefault(key, []).append(clip)
            self.generated.add(key)
            self.recent[key] = 0
        self.stats["realtime"] += 1
        self.stats["gpu_ms"] += self.gen_cost_ms
        self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self._used())
        return clip

    def on_event(self, key: CacheKey) -> str:
        self.fired.add(key)
        self.eta.pop(key, None)
        if self.learner is not None:
            self.learner.observe_event(key)
        pool = self.cache.get(key)
        if pool:
            self.stats["hits"] += 1
            idx = (self.recent.get(key, -1) + 1) % len(pool)
            self.recent[key] = idx
            self.last_served = pool[idx]
            return "cached"
        if self.realtime:
            self.last_served = self._realtime_generate(key)
            return "realtime"
        self._queue_demand(key)
        if self._has_authored(key):
            self.stats["fallback"] += 1
            self.last_served = self._authored_clip(key)
            return "fallback"
        self.stats["silence"] += 1
        self.last_served = AudioClip.silence(sample_rate=int(getattr(self.gen, "sample_rate", 48000)))
        return "silence"

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
