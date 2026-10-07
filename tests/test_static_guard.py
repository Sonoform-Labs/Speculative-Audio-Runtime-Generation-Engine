import time

import numpy as np
import pytest

import sarge.generators.stable_audio3 as sa3mod
from sarge.generators.stable_audio import _extract, _seed
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.loudness import normalize_loudness, target_for
from sarge.static_guard import is_static, retry_seed, static_metrics
from sarge.types import CacheKey

SR = 44_100
N = SR


def colored(rng, n, slope=0.0, hp=50.0):
    w = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    g = np.ones_like(f)
    g[1:] = f[1:] ** (-slope / 2.0)
    g[f < hp] = 0.0
    y = np.fft.irfft(np.fft.rfft(w) * g, n)
    return (y / (np.std(y) + 1e-12) * 0.1).astype(np.float32)


def hit(seed, n=N):
    rng = np.random.default_rng(seed % (2 ** 32))
    t = np.arange(n) / SR
    on = int(0.25 * SR)
    y = np.zeros(n)
    tt = t[: n - on]
    f0 = rng.uniform(180, 900)
    y[on:] = sum(np.sin(2 * np.pi * f0 * m * tt + rng.uniform(0, 6)) / m for m in (1.0, 2.7, 5.2)) * np.exp(-tt * 18)
    y[on:] += 0.3 * rng.standard_normal(n - on) * np.exp(-tt * 60)
    return (0.3 * y).astype(np.float32)


def stereo(x):
    return np.stack([x, x])


class SeedModel:
    def __init__(self, gen=None, plan=None, default="hit"):
        self.gen = gen
        self.plan = plan or {}
        self.default = default
        self.calls = []

    def make(self, seed):
        kind = self.plan.get(seed, self.default)
        if kind == "hit":
            return hit(seed)
        rng = np.random.default_rng(seed % (2 ** 32))
        if kind == "white":
            return colored(rng, N, 0.0, 50.0)
        if kind == "pink":
            return colored(rng, N, 1.0, 200.0)
        if kind == "tilt":
            return colored(rng, N, 0.5, 50.0)
        raise ValueError(kind)

    def generate(self, prompt=None, duration=1.0, steps=3, cfg_scale=2.0, seed=0, batch_size=1, **kw):
        if self.gen is not None:
            assert self.gen._lock.locked()
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        self.calls.append({"seed": seed, "steps": steps, "batch": batch_size, "prompts": prompts})
        rows = []
        for j in range(batch_size):
            s = self.plan.get(("batch", seed, j), None)
            rows.append(stereo(self.make(s if s is not None else seed + j)))
        return np.stack(rows)


def make(model=None, **kw):
    kw.setdefault("static_guard", False)
    g = StableAudio3Generator(steps=3, detail_steps=8, **kw)
    g._model = model if model is not None else SeedModel()
    if isinstance(g._model, SeedModel):
        g._model.gen = g
    g._ensure_loaded = lambda: None
    return g


def expected(g, x, key):
    return normalize_loudness(_extract(stereo(x).mean(axis=0), g.sample_rate, g.duration_s),
                              target_rms=target_for(key.intensity)).astype(np.float32)


WOOD = CacheKey("footstep", "wood", "normal")
WATER = CacheKey("footstep", "water", "normal")
GLASS = CacheKey("impact", "glass", "hard")


@pytest.mark.parametrize("dur", [0.6, 1.0, 1.5])
@pytest.mark.parametrize("seed", range(4))
def test_detector_flags_white_and_pink_noise(dur, seed):
    rng = np.random.default_rng(seed)
    for slope, hp in ((0.0, 50.0), (0.5, 50.0), (1.0, 50.0), (1.0, 200.0), (1.25, 400.0)):
        y = colored(rng, 5 * SR, slope, hp)
        x = normalize_loudness(_extract(y, SR, dur))
        bad, m = is_static(x, SR)
        assert bad, (slope, hp, m)


@pytest.mark.parametrize("seed", range(6))
def test_detector_passes_decaying_hits(seed):
    rng = np.random.default_rng(100 + seed)
    t = np.arange(int(1.2 * SR)) / SR
    burst = np.where(t >= 0.2, rng.standard_normal(len(t)) * np.exp(-(t - 0.2) * 40), 0.0).astype(np.float32)
    for y in (hit(seed, int(1.2 * SR)), burst):
        x = normalize_loudness(_extract(y, SR, 0.6))
        bad, m = is_static(x, SR)
        assert not bad, m


def test_detector_flags_hit_on_a_loud_hiss_bed():
    rng = np.random.default_rng(7)
    y = colored(rng, 2 * SR) + hit(3, 2 * SR)
    bad, m = is_static(normalize_loudness(_extract(y, SR, 0.6)), SR)
    assert bad and m["rule"] in ("hiss", "bed")


def unit_hit(seed, n=N):
    h = hit(seed, n).astype(np.float64)
    return h / np.max(np.abs(h))


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("lvl", [-14, -20, -26])
def test_detector_flags_a_hit_followed_by_steady_hiss(seed, lvl):
    rng = np.random.default_rng(300 + seed)
    on = int(0.25 * SR)
    for slope, hp in ((0.0, 50.0), (1.0, 200.0), (1.25, 400.0)):
        z = colored(rng, N, slope, hp) / 0.1 * 10 ** (lvl / 20.0) / 3.0
        z[:on] = 0.0
        x = normalize_loudness(_extract((unit_hit(seed) + z).astype(np.float32), SR, 0.6))
        bad, m = is_static(x, SR)
        assert bad, (slope, hp, m)


@pytest.mark.parametrize("amp", [10, 40, 160])
def test_detector_flags_hiss_behind_a_click(amp):
    rng = np.random.default_rng(amp)
    for slope, hp in ((0.0, 50.0), (1.0, 200.0)):
        y = colored(rng, N, slope, hp) / 0.1
        k = int(0.001 * SR)
        on = int(0.3 * SR)
        y[on:on + k] += amp * np.hanning(k) * rng.choice([-1.0, 1.0], k)
        bad, m = is_static(normalize_loudness(_extract(y.astype(np.float32), SR, 0.6)), SR)
        assert bad, (slope, hp, m)


@pytest.mark.parametrize("seed", range(4))
def test_detector_passes_hits_with_decaying_or_grainy_tails(seed):
    rng = np.random.default_rng(500 + seed)
    on = int(0.25 * SR)
    t = np.arange(N - on) / SR
    for tau in (0.05, 0.1, 0.2):
        tl = colored(rng, N, 1.0, 200.0) * 3.0
        tl[:on] = 0.0
        tl[on:] *= np.exp(-t / tau)
        x = normalize_loudness(_extract((unit_hit(seed) + tl).astype(np.float32), SR, 0.6))
        assert not is_static(x, SR)[0], tau
    gr = np.zeros(N)
    for _ in range(int(rng.integers(6, 14))):
        p = int(rng.integers(on + 2000, N - 3000))
        ln = int(rng.integers(200, 1500))
        gr[p:p + ln] += rng.standard_normal(ln) * np.hanning(ln) * rng.uniform(0.05, 0.3)
    x = normalize_loudness(_extract((unit_hit(seed) + gr).astype(np.float32), SR, 0.6))
    bad, m = is_static(x, SR)
    assert not bad, m


def test_detector_edge_cases():
    assert not is_static(np.zeros(SR, np.float32), SR)[0]
    assert not is_static(np.zeros(100, np.float32), SR)[0]
    x = hit(1)
    x[100] = np.nan
    bad, m = is_static(x, SR)
    assert bad and m["reason"] == "nonfinite"


def test_detector_only_reads_the_clip():
    x = normalize_loudness(_extract(colored(np.random.default_rng(0), 2 * SR), SR, 0.6))
    before = x.copy()
    is_static(x, SR)
    assert np.array_equal(x, before) and x.dtype == np.float32


def test_detector_is_fast():
    x = normalize_loudness(_extract(colored(np.random.default_rng(1), 3 * SR), SR, 1.5))
    for _ in range(5):
        static_metrics(x, SR)
    ts = []
    for _ in range(40):
        t0 = time.perf_counter()
        static_metrics(x, SR)
        ts.append(time.perf_counter() - t0)
    assert np.median(ts) < 0.01


def test_retry_seeds_are_deterministic_and_disjoint_from_variant_seeds():
    keys = [WOOD, WATER, GLASS, CacheKey("splash", "water", "normal")]
    variant_seeds = {_seed(k, v) for k in keys for v in range(5000)}
    assert max(variant_seeds) < 2 ** 31
    for k in keys:
        for v in range(50):
            base = _seed(k, v)
            rs = [retry_seed(base, a) for a in range(1, 6)]
            assert rs == [retry_seed(base, a) for a in range(1, 6)]
            assert len(set(rs)) == len(rs)
            assert not set(rs) & variant_seeds
            assert all(2 ** 31 <= s < 2 ** 32 for s in rs)


def test_retry_seeds_never_repeat_across_variants_of_one_key():
    keys = [CacheKey(e, m, i) for e in ("footstep", "impact", "land", "splash") for m in ("wood", "water", "grass")
            for i in ("soft", "normal")]
    for k in keys:
        seeds = [retry_seed(_seed(k, v), a) for v in range(64) for a in range(1, 6)]
        assert len(set(seeds)) == len(seeds), k


def test_guard_off_is_todays_output():
    m = SeedModel(default="white")
    g = make(m)
    c = g.generate(WOOD, variant=3)
    assert len(m.calls) == 1 and m.calls[0]["seed"] == _seed(WOOD, 3)
    assert np.array_equal(c.samples, expected(g, m.make(_seed(WOOD, 3)), WOOD))
    out = g.generate_batch([(WOOD, 0), (WATER, 1), (WOOD, 2)])
    assert len(m.calls) == 3
    assert "static_checks" not in g.fast_stats()
    assert [c.variant_id for c in out] == [0, 1, 2]


def test_clean_clip_is_returned_untouched(monkeypatch):
    seen = []

    def spy(samples, sr):
        seen.append(samples)
        return is_static(samples, sr)

    off = make(SeedModel())
    on = make(SeedModel(), static_guard=True)
    monkeypatch.setattr(sa3mod, "is_static", spy)
    a = off.generate(GLASS, variant=5)
    assert not seen
    b = on.generate(GLASS, variant=5)
    assert len(seen) == 1 and b.samples is seen[0]
    assert a.samples.tobytes() == b.samples.tobytes()
    assert len(on._model.calls) == 1
    st = on.fast_stats()
    assert st["static_checks"] == 1 and st["static_retries"] == 0 and st["static_flagged"] == 0


def test_static_first_seed_is_retried_and_clean_retry_returned():
    base = _seed(WOOD, 2)
    m = SeedModel(plan={base: "white"})
    g = make(m, static_guard=True)
    c = g.generate(WOOD, variant=2)
    seeds = [x["seed"] for x in m.calls]
    assert seeds == [base, retry_seed(base, 1)]
    assert m.calls[1]["steps"] == 3 and m.calls[1]["prompts"] == [g._prompt(WOOD)]
    assert np.array_equal(c.samples, expected(g, hit(retry_seed(base, 1)), WOOD))
    assert not is_static(c.samples, SR)[0]
    assert c.variant_id == 2
    st = g.fast_stats()
    assert st["static_flagged"] == 1 and st["static_retries"] == 1 and st["static_rescued"] == 1
    assert st["static_exhausted"] == 0 and st["static_checks"] == 2


def test_exhaustion_returns_least_flat_attempt():
    base = _seed(WATER, 0)
    plan = {base: "white", retry_seed(base, 1): "pink", retry_seed(base, 2): "tilt"}
    m = SeedModel(plan=plan)
    g = make(m, static_guard=True, max_static_retries=2)
    c = g.generate(WATER, variant=0)
    assert [x["seed"] for x in m.calls] == [base, retry_seed(base, 1), retry_seed(base, 2)]
    assert all(x["steps"] == 8 for x in m.calls)
    cands = [expected(g, m.make(s), WATER) for s in (base, retry_seed(base, 1), retry_seed(base, 2))]
    flats = [static_metrics(x, SR)["flat"] for x in cands]
    assert all(is_static(x, SR)[0] for x in cands)
    assert int(np.argmin(flats)) == 1
    assert np.array_equal(c.samples, cands[1])
    st = g.fast_stats()
    assert st["static_exhausted"] == 1 and st["static_retries"] == 2 and st["static_rescued"] == 0


def test_exhausted_nonfinite_clip_is_never_served():
    class NanModel(SeedModel):
        def make(self, seed):
            x = hit(seed)
            x[int(0.3 * SR): int(0.32 * SR)] = np.nan
            return x

    m = NanModel()
    g = make(m, static_guard=True)
    c = g.generate(WOOD, variant=0)
    assert len(m.calls) == 3
    assert np.isfinite(c.samples).all() and c.samples.dtype == np.float32
    assert g.fast_stats()["static_exhausted"] == 1


def test_zero_retries_checks_only():
    base = _seed(WOOD, 0)
    m = SeedModel(plan={base: "white"})
    g = make(m, static_guard=True, max_static_retries=0)
    c = g.generate(WOOD, variant=0)
    assert len(m.calls) == 1
    assert np.array_equal(c.samples, expected(g, m.make(base), WOOD))
    assert g.fast_stats()["static_exhausted"] == 1


def test_retries_are_deterministic_across_instances():
    base = _seed(GLASS, 4)
    plan = {base: "white", retry_seed(base, 1): "white"}
    a = make(SeedModel(plan=dict(plan)), static_guard=True)
    b = make(SeedModel(plan=dict(plan)), static_guard=True)
    ca, cb = a.generate(GLASS, variant=4), b.generate(GLASS, variant=4)
    assert [x["seed"] for x in a._model.calls] == [x["seed"] for x in b._model.calls]
    assert [x["seed"] for x in a._model.calls] == [base, retry_seed(base, 1), retry_seed(base, 2)]
    assert ca.samples.tobytes() == cb.samples.tobytes()


def test_batch_keeps_order_steps_and_untouched_items():
    items = [(WOOD, 0), (WATER, 1), (WOOD, 2), (GLASS, 3), (WATER, 4)]
    s3 = _seed(WOOD, 0)
    s8 = _seed(WATER, 1)
    plan = {("batch", s3, 1): 10 ** 6 + 1, 10 ** 6 + 1: "white",
            ("batch", s8, 1): 10 ** 6 + 2, 10 ** 6 + 2: "white"}
    off = make(SeedModel(plan=dict(plan)))
    on = make(SeedModel(plan=dict(plan)), static_guard=True)
    ref = off.generate_batch(items)
    out = on.generate_batch(items)
    assert [c.variant_id for c in out] == [0, 1, 2, 3, 4]
    calls = on._model.calls
    assert [(c["steps"], c["batch"]) for c in calls[:1]] == [(3, 3)]
    retried = {2: (WOOD, 2), 4: (WATER, 4)}
    for i, (key, v) in enumerate(items):
        if i in retried:
            assert np.array_equal(out[i].samples, expected(on, hit(retry_seed(_seed(key, v), 1)), key))
            assert is_static(ref[i].samples, SR)[0]
        else:
            assert out[i].samples.tobytes() == ref[i].samples.tobytes()
    retry_calls = [c for c in calls if c["seed"] >= 2 ** 31]
    assert sorted((c["seed"], c["steps"], c["batch"]) for c in retry_calls) == sorted(
        (retry_seed(_seed(k, v), 1), on._steps_for(k), 1) for k, v in retried.values())
    for c in retry_calls:
        k = [k for k, v in retried.values() if retry_seed(_seed(k, v), 1) == c["seed"]][0]
        assert c["prompts"] == [on._prompt(k)]
    st = on.fast_stats()
    assert st["static_checks"] == 7 and st["static_flagged"] == 2 and st["static_rescued"] == 2


def test_guard_runs_every_model_call_under_the_lock():
    base = _seed(WOOD, 1)
    m = SeedModel(plan={base: "white"})
    g = make(m, static_guard=True)
    g.generate(WOOD, variant=1)
    g.generate_batch([(WOOD, 1), (WATER, 2)])
    assert len(m.calls) >= 3
    assert not g._lock.locked()


def test_guard_is_on_by_default():
    assert StableAudio3Generator().static_guard is True
