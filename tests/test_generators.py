import numpy as np

from predictive_sfx.generators import MockGenerator
from predictive_sfx.types import CacheKey, seed_for


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
