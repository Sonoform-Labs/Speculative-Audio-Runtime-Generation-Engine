import hashlib

from predictive_sfx.generators.cached import CachedGenerator, CACHE_SCHEMA
from predictive_sfx.generators.mock import MockGenerator
from predictive_sfx.types import CacheKey


class FakeGen(MockGenerator):
    name = "fakegen"
    model_name = "small-sfx"

    def __init__(self):
        super().__init__()
        self.steps = 3
        self.sampler_type = "euler"
        self.cfg_scale = 2.0
        self.gen_duration_s = 1.0
        self.duration_s = 0.6
        self.detail_keys = {"splash", "water"}
        self.detail_steps = 8
        self._prompt_fn = lambda key: f"a {key.event_type} on {key.material}, dry"

    def _prompt(self, key):
        return self._prompt_fn(key)

    def _steps_for(self, key):
        if key.event_type in self.detail_keys or key.material in self.detail_keys:
            return self.detail_steps
        return self.steps


def test_same_settings_same_path(tmp_path):
    g = FakeGen()
    c = CachedGenerator(g, str(tmp_path))
    k = CacheKey("hit", "metal", "normal")
    assert c._path(k, 0) == c._path(k, 0)


def test_changing_prompt_or_steps_or_sampler_misses(tmp_path):
    g = FakeGen()
    c = CachedGenerator(g, str(tmp_path))
    k = CacheKey("hit", "metal", "normal")
    base = c._path(k, 0)

    g._prompt_fn = lambda key: "a totally different enriched prompt"
    assert c._path(k, 0) != base
    g._prompt_fn = lambda key: f"a {key.event_type} on {key.material}, dry"
    assert c._path(k, 0) == base

    g.steps = 4
    assert c._path(k, 0) != base
    g.steps = 3

    g.sampler_type = "dpmpp"
    assert c._path(k, 0) != base
    g.sampler_type = "euler"
    assert c._path(k, 0) == base


def test_per_event_detail_steps_are_in_the_key(tmp_path):
    g = FakeGen()
    c = CachedGenerator(g, str(tmp_path))
    water = CacheKey("splash", "water", "normal")
    foot = CacheKey("footstep", "wood", "normal")
    assert c._key_steps(water) == 8 and c._key_steps(foot) == 3
    p_water = c._path(water, 0)
    g.detail_keys = set()
    assert c._path(water, 0) != p_water


def test_old_event_only_key_not_reused(tmp_path):
    g = FakeGen()
    c = CachedGenerator(g, str(tmp_path))
    k = CacheKey("hit", "metal", "normal")
    old_tag = f"{g.name}|{g.steps}|{g.cfg_scale}"
    old_hash = hashlib.md5(f"{old_tag}|{k}||0".encode()).hexdigest()[:16]
    new_path = c._path(k, 0)
    assert CACHE_SCHEMA not in old_tag
    assert old_hash not in new_path


def test_roundtrip_generate_and_hit(tmp_path):
    g = FakeGen()
    c = CachedGenerator(g, str(tmp_path))
    k = CacheKey("hit", "metal", "normal")
    a = c.generate(k, variant=0)
    assert a.samples.size > 0
    import os
    assert os.path.exists(c._path(k, 0))
    b = c.generate(k, variant=0)
    assert b.samples.size == a.samples.size
