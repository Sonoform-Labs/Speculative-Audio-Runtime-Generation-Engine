import threading
import time

import numpy as np

from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey


class OverlapModel:
    def __init__(self):
        self.inside = 0
        self.max_inside = 0
        self.guard = threading.Lock()

    def generate(self, prompt, duration, steps, cfg_scale, seed, batch_size=1, **kw):
        with self.guard:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
        time.sleep(0.02)
        with self.guard:
            self.inside -= 1
        return np.random.default_rng(seed).standard_normal((batch_size, 2, 44100)).astype(np.float32) * 0.1


def test_realtime_and_background_never_overlap_inside_the_model():
    g = StableAudio3Generator(steps=3, detail_steps=8)
    g._model = OverlapModel()
    g._ensure_loaded = lambda: None
    key = CacheKey("footstep", "wood", "normal")

    def realtime():
        for v in range(5):
            g.generate(key, variant=v)

    def background():
        for v in range(5):
            g.generate_batch([(key, v), (CacheKey("footstep", "water", "normal"), v)])

    ts = [threading.Thread(target=realtime), threading.Thread(target=background)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert g._model.max_inside == 1
