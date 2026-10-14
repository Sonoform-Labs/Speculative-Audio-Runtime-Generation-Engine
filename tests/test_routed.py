import numpy as np

from sarge.generators.routed import RoutedGenerator
from sarge.types import AudioClip, CacheKey


class Fake:
    def __init__(self, tag):
        self.tag = tag
        self.calls = []
        self.sample_rate = 44100

    def generate(self, key, *, variant=0):
        self.calls.append(str(key))
        return AudioClip(np.full(10, 0.1 if self.tag == "guns" else 0.2, dtype=np.float32), 44100, variant_id=variant)

    def generate_batch(self, items):
        return [self.generate(k, variant=v) for k, v in items]


def test_guns_go_to_one_model_and_everything_else_to_the_other():
    g, r = Fake("guns"), Fake("rest")
    gen = RoutedGenerator(guns=g, rest=r, warm_guns=False)
    keys = [CacheKey("shoot", "pistol", "normal"), CacheKey("reload", "rifle", "normal"), CacheKey("break", "window", "normal"),
            CacheKey("footstep", "water", "normal"), CacheKey("splash", "water", "normal")]
    clips = [gen.generate(k) for k in keys]
    assert len(g.calls) == 2 and len(r.calls) == 3
    assert [float(c.samples[0]) for c in clips] == [np.float32(0.1)] * 2 + [np.float32(0.2)] * 3
    batch = gen.generate_batch([(k, 1) for k in keys])
    assert [b.variant_id for b in batch] == [1] * 5 and gen.fast_stats()["routed_guns"] == 4
