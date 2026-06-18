import numpy as np

from predictive_sfx.cache import VariantCache
from predictive_sfx.types import AudioClip, CacheKey


def _clip(n=128, variant=0):
    return AudioClip(np.zeros(n, dtype=np.float32), 48_000, variant_id=variant)


def test_add_and_get_hit_miss():
    c = VariantCache()
    k = CacheKey("footstep", "wood", "normal")
    assert c.get(k) is None              # miss
    assert c.stats.misses == 1
    c.add(k, _clip(variant=0))
    assert c.has(k)
    got = c.get(k)
    assert got is not None and got.variant_id == 0
    assert c.stats.hits == 1


def test_avoid_repeat_cycles_variants():
    c = VariantCache(avoid_repeat=2)
    k = CacheKey("footstep", "stone", "hard")
    for v in range(3):
        c.add(k, _clip(variant=v))
    seen = [c.get(k).variant_id for _ in range(2)]
    # with avoid_repeat=2 the two most recent picks should differ
    assert seen[0] != seen[1]


def test_eviction_respects_pin():
    # each clip is 128 * 4 = 512 bytes; budget fits ~1 clip
    c = VariantCache(max_bytes=600)
    pinned = CacheKey("footstep", "wood", "normal")
    runtime = CacheKey("footstep", "metal", "hard")
    c.add(pinned, _clip(), pin=True)
    c.add(runtime, _clip())          # pushes over budget -> evict unpinned
    assert c.has(pinned)
    assert not c.has(runtime)
    assert c.stats.evictions >= 1


def test_nbytes_tracks_inserts():
    c = VariantCache()
    k = CacheKey("impact", "grass", "soft")
    c.add(k, _clip(n=256))
    assert c.nbytes == 256 * 4
