import numpy as np

from sarge.generators import MockGenerator
from sarge.loudness import normalize_loudness, rms, target_for
from sarge.types import CacheKey


def test_normalize_hits_target_rms_regardless_of_input_level():
    rng = np.random.default_rng(0)
    quiet = (rng.standard_normal(4800) * 0.01).astype(np.float32)
    loud = (rng.standard_normal(4800) * 5.0).astype(np.float32)
    for sig in (quiet, loud):
        out = normalize_loudness(sig, target_rms=0.1)
        assert abs(rms(out) - 0.1) < 1e-3


def test_peak_ceiling_enforced():
    x = np.full(1000, 0.001, dtype=np.float32)
    x[0] = 1.0
    out = normalize_loudness(x, target_rms=0.5, peak_ceiling=0.9)
    assert np.max(np.abs(out)) <= 0.9 + 1e-4


def test_intensity_targets_are_ordered():
    assert target_for("soft") < target_for("normal") < target_for("hard")


def test_two_variants_same_intensity_match_loudness():
    rng = np.random.default_rng(1)
    a = (rng.standard_normal(4800) * 0.3).astype(np.float32)
    b = (rng.standard_normal(4800) * 2.0).astype(np.float32)
    assert abs(rms(normalize_loudness(a)) - rms(normalize_loudness(b))) < 1e-3


def test_empty_signal_is_safe():
    assert normalize_loudness(np.zeros(0, dtype=np.float32)).size == 0
    assert normalize_loudness(np.zeros(100, dtype=np.float32)).max() == 0.0


def test_mock_clip_is_loudness_normalized():
    c = MockGenerator().generate(CacheKey("footstep", "wood", "normal"), variant=0)
    assert 0.0 < rms(c.samples) <= target_for("normal") + 1e-3
    assert np.max(np.abs(c.samples)) <= 0.97 + 1e-4


def test_mock_intensity_affects_loudness():
    soft = MockGenerator().generate(CacheKey("footstep", "wood", "soft"), variant=0)
    hard = MockGenerator().generate(CacheKey("footstep", "wood", "hard"), variant=0)
    assert rms(soft.samples) < rms(hard.samples)


def test_peak_normalize_keeps_the_crest():
    import numpy as np
    from sarge.loudness import normalize_peak, peak_for
    x = np.zeros(1000, dtype=np.float32)
    x[10] = 0.5
    x[20:] = 0.01
    y = normalize_peak(x, peak_for("normal"))
    assert abs(float(np.max(np.abs(y))) - 10 ** (-1 / 20)) < 1e-5
    assert abs(float(y[500] / y[10]) - 0.02) < 1e-6


def test_amplifier_adds_density_by_family_and_stays_finite():
    import numpy as np
    from sarge.amp import amplify, family_for
    from sarge.loudness import normalize_peak
    sr = 44100
    t = np.arange(int(0.5 * sr))
    x = (np.exp(-t / (0.05 * sr)) * np.sin(t * 0.21) * 0.5).astype(np.float32)
    rms = lambda y: float(np.sqrt(np.mean(normalize_peak(y) ** 2)))
    gun = amplify(x, sr, "shoot", "normal")
    step = amplify(x, sr, "footstep", "normal")
    assert np.all(np.isfinite(gun)) and np.all(np.isfinite(step))
    assert rms(gun) > rms(step) > rms(x) * 0.99
    assert family_for("bullet impact") == "impact" and family_for("shoot") == "blast" and family_for("splash") is None
