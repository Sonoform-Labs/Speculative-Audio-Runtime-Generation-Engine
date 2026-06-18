from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass

import numpy as np

from .types import AudioClip, CacheKey


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    inserts: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class VariantCache:
    def __init__(self, max_bytes: int | None = None, avoid_repeat: int = 2,
                 rng: np.random.Generator | None = None):
        self._store: "OrderedDict[CacheKey, list[AudioClip]]" = OrderedDict()
        self._pinned: set[CacheKey] = set()
        self._recent: dict[CacheKey, deque] = {}
        self._bytes = 0
        self.max_bytes = max_bytes
        self.avoid_repeat = max(0, int(avoid_repeat))
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self.stats = CacheStats()

    def add(self, key: CacheKey, clip: AudioClip, *, pin: bool = False) -> None:
        lst = self._store.get(key)
        if lst is None:
            lst = []
            self._store[key] = lst
            self._recent[key] = deque(maxlen=self.avoid_repeat)
        lst.append(clip)
        self._bytes += clip.nbytes
        self.stats.inserts += 1
        if pin:
            self._pinned.add(key)
        self._store.move_to_end(key)
        self._evict_if_needed()

    def has(self, key: CacheKey) -> bool:
        lst = self._store.get(key)
        return bool(lst)

    def get(self, key: CacheKey) -> AudioClip | None:
        lst = self._store.get(key)
        if not lst:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        self._store.move_to_end(key)
        recent = self._recent[key]
        candidates = [i for i in range(len(lst)) if i not in recent] or list(range(len(lst)))
        idx = int(self._rng.choice(candidates))
        recent.append(idx)
        return lst[idx]

    def variants(self, key: CacheKey) -> int:
        return len(self._store.get(key, ()))

    def _evict_if_needed(self) -> None:
        if self.max_bytes is None:
            return
        while self._bytes > self.max_bytes:
            victim = None
            for k in self._store:
                if k not in self._pinned:
                    victim = k
                    break
            if victim is None:
                break
            lst = self._store.pop(victim)
            self._recent.pop(victim, None)
            self._bytes -= sum(c.nbytes for c in lst)
            self.stats.evictions += 1

    @property
    def nbytes(self) -> int:
        return self._bytes

    @property
    def pinned_keys(self) -> set[CacheKey]:
        return set(self._pinned)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: CacheKey) -> bool:
        return self.has(key)
