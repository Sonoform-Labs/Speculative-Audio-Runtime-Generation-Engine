import numpy as np

from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, duration, steps, cfg_scale, seed, batch_size, **kw):
        self.calls.append((steps, batch_size, list(prompt)))
        rng = np.random.default_rng(seed)
        return rng.standard_normal((batch_size, 2, 44100)).astype(np.float32) * 0.1


def make():
    g = StableAudio3Generator(steps=3, detail_steps=8)
    g._model = FakeModel()
    g._ensure_loaded = lambda: None
    return g


def test_batch_gives_detail_keys_their_own_steps():
    g = make()
    items = [(CacheKey("footstep", "wood", "normal"), 0),
             (CacheKey("footstep", "water", "normal"), 0),
             (CacheKey("impact", "stone", "normal"), 0)]
    out = g.generate_batch(items)
    assert len(out) == 3 and all(c is not None for c in out)
    assert sorted((s, b) for s, b, _ in g._model.calls) == [(3, 2), (8, 1)]


def test_batch_matches_single_generate_step_choice():
    g = make()
    for key in (CacheKey("footstep", "water", "normal"),
                CacheKey("splash", "stone", "normal"),
                CacheKey("footstep", "grass", "normal")):
        g._model.calls.clear()
        g.generate_batch([(key, 0)])
        assert g._model.calls[0][0] == g._steps_for(key)


def test_batch_preserves_item_order():
    g = make()
    items = [(CacheKey("footstep", "water", "normal"), 0),
             (CacheKey("footstep", "wood", "normal"), 1),
             (CacheKey("footstep", "water", "normal"), 2)]
    out = g.generate_batch(items)
    assert [c.variant_id for c in out] == [0, 1, 2]
