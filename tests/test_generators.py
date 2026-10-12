import numpy as np

from sarge.generators import MockGenerator
from sarge.types import CacheKey, seed_for


def test_seed_for_is_stable_across_calls():
    k = CacheKey("footstep", "wood", "normal")
    assert seed_for(k, 0) == 515439703
    assert seed_for(k, 1) == 9405926
    assert seed_for(k, 0) != seed_for(k, 1)


def test_mock_output_is_deterministic_for_same_key_variant():
    k = CacheKey("footstep", "wood", "normal")
    a = MockGenerator().generate(k, variant=0).samples
    b = MockGenerator().generate(k, variant=0).samples
    assert np.array_equal(a, b)


def test_mock_variants_differ():
    k = CacheKey("footstep", "wood", "normal")
    a = MockGenerator().generate(k, variant=0).samples
    b = MockGenerator().generate(k, variant=1).samples
    assert not np.array_equal(a, b)


def test_natural_trim_keeps_the_tail_and_drops_the_silence():
    import numpy as np
    from sarge.generators.stable_audio import _natural
    sr = 44100
    x = np.zeros(sr, dtype=np.float32)
    t = np.arange(int(0.4 * sr))
    x[2000:2000 + len(t)] = (np.exp(-t / (0.06 * sr)) * np.sin(t * 0.3)).astype(np.float32)
    y = _natural(x, sr)
    assert 0.25 * sr < len(y) < 0.55 * sr
    assert abs(len(y) - len(x)) > 0.4 * sr
    assert np.argmax(np.abs(y)) < 0.05 * sr
