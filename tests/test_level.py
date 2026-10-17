import hashlib
import os
import sys
import types

import numpy as np
import pytest

from sarge import loudness as L
from sarge import post
from sarge.amp import amplify
from sarge.types import CacheKey

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


def _burst(sr=48000, n=48000, tau=0.03, seed=0, ch=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    x = rng.standard_normal((n, ch) if ch > 1 else n) * 0.2
    env = np.exp(-t / tau)
    return (x * (env if ch == 1 else env[:, None])).astype(np.float32)


def _ring(sr=48000, n=48000, f=609.0, tau=0.25):
    t = np.arange(n) / sr
    x = np.sin(2 * np.pi * f * t) * np.exp(-t / tau)
    x[: int(0.002 * sr)] *= np.linspace(0, 1, int(0.002 * sr))
    return (0.5 * x).astype(np.float32)


@pytest.mark.parametrize("sr", [48000, 44100])
def test_full_scale_997_hz_sine_reads_minus_3_01_lufs(sr):
    t = np.arange(2 * sr) / sr
    assert L.momentary_max(np.sin(2 * np.pi * 997 * t), sr) == pytest.approx(-3.01, abs=0.1)


@pytest.mark.parametrize("sr", [48000, 44100])
def test_k_weighting_matches_pyloudnorm_within_a_tenth_of_a_db(sr):
    pyln = pytest.importorskip("pyloudnorm")
    from scipy.signal import freqz, sosfreqz
    f = np.array([20, 40, 80, 200, 500, 1000, 1500, 3000, 6000, 10000, 16000], dtype=float)
    _, h = sosfreqz(L._shelf_and_highpass(sr), worN=f, fs=sr)
    ref = np.ones(len(f), dtype=complex)
    for flt in pyln.Meter(sr)._filters.values():
        ref *= flt.passband_gain * freqz(flt.b, flt.a, worN=f, fs=sr)[1]
    assert np.max(np.abs(20 * np.log10(np.abs(h)) - 20 * np.log10(np.abs(ref)))) < 0.1
    x = _burst(sr, sr, 0.2, ch=2)
    meter = pyln.Meter(sr, block_size=0.4)
    k = L.kweight(x, sr)
    y = x.astype(np.float64)
    for flt in meter._filters.values():
        from scipy.signal import lfilter
        y = flt.passband_gain * lfilter(flt.b, flt.a, y, axis=0)
    a = -0.691 + 10 * np.log10(np.mean(np.sum(k ** 2, axis=1)))
    b = -0.691 + 10 * np.log10(np.mean(np.sum(y ** 2, axis=1)))
    assert a == pytest.approx(b, abs=0.1)


def test_momentary_max_pads_short_clips_and_counts_channel_power():
    sr = 48000
    x = _burst(sr, 4800)
    assert np.isfinite(L.momentary_max(x, sr))
    st = np.stack([x, x], axis=1)
    assert L.momentary_max(st, sr) == pytest.approx(L.momentary_max(x, sr) + 10 * np.log10(2), abs=1e-6)


@pytest.mark.parametrize("ch", [1, 2])
def test_limiter_never_passes_the_ceiling(ch):
    sr = 48000
    rng = np.random.default_rng(5)
    x = _burst(sr, sr, 0.2, seed=2, ch=ch) * 20.0
    idx = rng.integers(0, sr, 40)
    x[idx] = 7.0
    for ceil in (-1.0, -3.0, -0.1):
        y = L.limit(x, sr, ceil)
        assert y.dtype == np.float32 and y.shape == x.shape
        assert float(np.max(np.abs(y))) <= 10 ** (ceil / 20) + 1e-6


def test_limiter_leaves_quiet_audio_untouched_and_reports_gain_reduction():
    sr = 48000
    x = _burst(sr) * 0.5
    st = {}
    np.testing.assert_array_equal(L.limit(x, sr, stats=st), x)
    assert st["gr_db"] == 0.0
    big = x * (10 ** (12 / 20) / float(np.max(np.abs(x))))
    y = L.limit(big, sr, -1.0, stats=st)
    assert 12.0 < st["gr_db"] < 13.5
    assert np.abs(y[-2000:]).max() == pytest.approx(np.abs(big[-2000:]).max(), rel=1e-3)


def test_highpass_removes_a_dc_step():
    sr = 48000
    x = np.zeros(2 * sr, dtype=np.float32)
    x[sr // 4:] = 0.5
    y = L.dc_highpass(x, sr)
    assert np.abs(y[-sr // 2:]).max() < 1e-3
    s = L.dc_highpass(np.stack([x, -x], axis=1), sr)
    assert s.shape == (2 * sr, 2) and np.abs(s[-sr // 2:]).max() < 1e-3


def test_normalize_level_is_deterministic_and_steps_by_intensity():
    sr = 48000
    x = (np.random.default_rng(8).standard_normal((sr, 2)) * 0.05).astype(np.float32)
    out = {}
    for inten in ("soft", "normal", "hard"):
        y1, s1 = L.normalize_level(x, sr, inten)
        y2, s2 = L.normalize_level(x, sr, inten)
        np.testing.assert_array_equal(y1, y2)
        assert s1 == s2 and s1["gr_db"] <= L.MAX_GR_DB + 1e-9
        out[inten] = L.momentary_max(y1, sr)
        assert float(np.max(np.abs(y1))) <= 10 ** (L.CEILING_DBFS / 20) + 1e-6
    step = L.LUFS_BY_INTENSITY
    assert out["normal"] - out["soft"] == pytest.approx(step["normal"] - step["soft"], abs=0.01)
    assert out["hard"] - out["normal"] == pytest.approx(step["hard"] - step["normal"], abs=0.01)


def test_normalize_level_reaches_the_target_on_dense_audio_and_caps_limiting_on_sparse():
    sr = 48000
    rng = np.random.default_rng(1)
    dense = (rng.standard_normal((sr, 2)) * 0.05).astype(np.float32)
    y, st = L.normalize_level(dense, sr, "normal")
    assert L.momentary_max(y, sr) == pytest.approx(L.LUFS_BY_INTENSITY["normal"], abs=0.3)
    click = np.zeros((sr, 2), dtype=np.float32)
    click[1000:1048] = 0.9
    y, st = L.normalize_level(click, sr, "hard")
    assert st["gr_db"] <= L.MAX_GR_DB + 1e-6
    assert L.momentary_max(y, sr) < L.LUFS_BY_INTENSITY["hard"] - 3
    z, st = L.normalize_level(np.zeros((100, 2), dtype=np.float32), sr)
    assert not np.any(z) and st["gain_db"] == 0.0


def test_normalize_level_leaves_numerical_silence_alone():
    sr = 48000
    x = (np.random.default_rng(3).standard_normal(sr) * 1e-12).astype(np.float32)
    y, st = L.normalize_level(x, sr, "hard")
    np.testing.assert_array_equal(y, x)
    assert st["gain_db"] == 0.0 and st["lufs_in"] < L.SILENCE_LUFS
    q = post.process(x, sr, "impact", "normal", level="loudness")
    assert q.size == 0 or float(np.max(np.abs(q))) < 1e-9


def test_mono_is_levelled_as_dual_mono():
    sr = 48000
    rng = np.random.default_rng(4)
    m = (rng.standard_normal(sr) * 0.05).astype(np.float32)
    ym, _ = L.normalize_level(m, sr)
    ys, _ = L.normalize_level(np.stack([m, m], axis=1), sr)
    np.testing.assert_allclose(ym, ys[:, 0], atol=1e-6)


def test_amplify_defaults_are_unchanged_and_new_switches_are_additive():
    sr = 48000
    x = _ring(sr)
    for ev, inten in (("shoot", "hard"), ("break", "normal"), ("footstep", "soft"), ("door", "normal")):
        a = amplify(x, sr, ev, inten)
        np.testing.assert_array_equal(a, amplify(x, sr, ev, inten, 1.0, True, None))
        np.testing.assert_array_equal(amplify(x, sr, ev, inten, drive=False), amplify(x, sr, ev, inten, drive_cap=1.0))
    np.testing.assert_allclose(amplify(x, sr, "break", "normal", punch_scale=0.0, drive=False)[2000:], x[2000:],
                               rtol=1e-5, atol=1e-7)
    lo = amplify(x, sr, "break", "hard", drive=False, punch_scale=0.25)
    hi = amplify(x, sr, "break", "hard", drive=False)
    assert np.abs(hi).max() >= np.abs(lo).max()


def test_decay_cap_moves_the_minus_40_db_point_and_never_lengthens():
    sr = 48000
    x = _ring(sr, tau=0.25)
    t40 = post.decay_time(x, sr)
    assert t40 > 0.8
    for target in (0.08, 0.2, 0.6):
        y = post.decay_cap(x, sr, target)
        assert y.shape[0] <= x.shape[0]
        assert 0.67 * target <= post.decay_time(y, sr) <= 1.05 * target
        n = int(0.015 * sr) if target > 0.03 else int(0.5 * target * sr)
        np.testing.assert_array_equal(y[: int(0.002 * sr) + n // 2], x[: int(0.002 * sr) + n // 2])
    short = _ring(sr, tau=0.02)
    np.testing.assert_array_equal(post.decay_cap(short, sr, 0.2), short)
    st = np.stack([x, x * 0.5], axis=1)
    assert post.decay_cap(st, sr, 0.2).ndim == 2


def test_process_loudness_chain_returns_stereo_under_the_ceiling():
    sr = 48000
    x = _burst(sr, sr, 0.05)
    st = {}
    y = post.process(x, sr, "break", "normal", level="loudness", amp_mode="punch", stereo=True, stats=st)
    assert y.ndim == 2 and y.shape[1] == 2 and y.dtype == np.float32
    assert float(np.max(np.abs(y))) <= 10 ** (L.CEILING_DBFS / 20) + 1e-6
    assert {"lufs_in", "gain_db", "gr_db"} <= set(st)
    y2 = post.process(x, sr, "break", "normal", level="loudness", amp_mode="punch", stereo=True)
    np.testing.assert_array_equal(y, y2)
    shaped = post.process(_ring(sr), sr, "impact", "hard", level="loudness", amp_mode="full", stereo=True,
                          shape={"t40_s": 0.2, "punch_scale": 0.5, "drive": 1.0, "floor_db": -45.0, "lowpass_hz": 4000})
    assert post.decay_time(shaped, sr) <= 0.25


@pytest.fixture
def ws(monkeypatch):
    if SCRIPTS not in sys.path:
        monkeypatch.syspath_prepend(SCRIPTS)
    import woosh_server
    return woosh_server


def test_woosh_finish_defaults_take_the_head_path(ws):
    raw = _burst(ws.SR, ws.SR, 0.1)[:, None]
    for st in (False, True):
        np.testing.assert_array_equal(ws.finish(raw, "break", "normal", None, st), ws.finish(raw, "break", "normal", False, st))
        np.testing.assert_array_equal(ws.finish(raw, "break", "normal", "full", st), ws.finish(raw, "break", "normal", True, st))
        np.testing.assert_array_equal(ws.finish(raw, "break", "normal", True, st, "peak", None),
                                      ws.finish(raw, "break", "normal", True, st))


def test_woosh_finish_loudness_and_punch_go_through_the_post_chain(ws):
    raw = _burst(ws.SR, ws.SR, 0.1)[:, None]
    y = ws.finish(raw, "break", "hard", "punch", True, level="loudness")
    ref = post.process(np.ascontiguousarray(raw[: ws.SR - int(ws.END_CUT_S * ws.SR), 0]), ws.SR, "break", "hard",
                       level="loudness", amp_mode="punch", stereo=True)
    np.testing.assert_array_equal(y, ref)
    p = ws.finish(raw, "break", "normal", "punch", False)
    assert float(np.max(np.abs(p))) == pytest.approx(10 ** (-1 / 20), rel=1e-5)


def _sa3(**kw):
    from sarge.generators.stable_audio3 import StableAudio3Generator
    g = StableAudio3Generator.__new__(StableAudio3Generator)
    g.stereo, g.amp, g.trim, g.level, g.sample_rate, g.duration_s, g.width = True, True, "natural", "peak", 44100, 0.6, 1.0
    for k, v in kw.items():
        setattr(g, k, v)
    return g


def test_sa3_finish_routes_only_loudness_or_punch_to_the_post_chain():
    from sarge.generators.stable_audio3 import StableAudio3Generator
    a = np.stack([_burst(44100, 3 * 44100, 0.1, seed=s) for s in (1, 2)])
    key = CacheKey("shoot", "pistol", "normal")
    head = _sa3()._finish(a.copy(), key)
    assert np.array_equal(head, _sa3(amp_mode="full")._finish(a.copy(), key))
    y = _sa3(level="loudness", amp_mode="punch")._finish(a.copy(), key)
    ref = post.process(np.ascontiguousarray(a.T), 44100, "shoot", "normal", level="loudness", amp_mode="punch",
                       stereo=True, trim="natural")
    np.testing.assert_array_equal(y, ref)
    g = StableAudio3Generator(amp="punch", level="loudness")
    assert g.amp and g.amp_mode == "punch" and g.post_tag == "post:loudness:punch"
    assert StableAudio3Generator(amp=True, level="peak").post_tag == ""
    assert StableAudio3Generator().amp_mode == "off" and not StableAudio3Generator(amp="off").amp


def test_cache_path_hash_is_unchanged_without_a_post_tag(tmp_path):
    from sarge.generators.cached import CACHE_SCHEMA, CachedGenerator

    class G:
        name, model_name, sampler_type, steps, cfg_scale = "g", "m", "euler", 3, 1.0
        sample_rate = 48000
        latency_ms = 1.0

    key = CacheKey("break", "window", "normal")
    c = CachedGenerator(G(), str(tmp_path))
    sig = "|".join(str(x) for x in (CACHE_SCHEMA, "g", "m", key, "", "euler", 3, 1.0, "", "", 48000, "", "", "", 0))
    assert os.path.basename(c._path(key, 0)) == hashlib.md5(sig.encode()).hexdigest()[:16] + ".wav"
    g2 = G()
    g2.post_tag = ""
    assert CachedGenerator(g2, str(tmp_path))._path(key, 0) == c._path(key, 0)
    g2.post_tag = "post:loudness:punch"
    assert CachedGenerator(g2, str(tmp_path))._path(key, 0) != c._path(key, 0)


def test_routed_generator_exposes_the_gun_route_post_tag():
    from sarge.generators.routed import RoutedGenerator
    guns = types.SimpleNamespace(sample_rate=44100, prompts={})
    r = RoutedGenerator(guns=guns, rest=object())
    assert r.post_tag == ""
    guns.post_tag = "post:loudness:full"
    assert r.post_tag == "post:loudness:full"


@pytest.fixture
def rs(monkeypatch):
    if SCRIPTS not in sys.path:
        monkeypatch.syspath_prepend(SCRIPTS)
    import run_sidecar
    return run_sidecar


def test_start_woosh_passes_amp_mode_and_level_only_when_set(rs, monkeypatch, tmp_path):
    py = tmp_path / "python.exe"
    py.write_bytes(b"")
    cmds = []

    class Proc:
        def __init__(self, cmd, **kw):
            cmds.append(cmd)
            self.stdout = iter(["woosh server listening on 127.0.0.1:8771\n"])

        def poll(self):
            return 0

    monkeypatch.setattr(rs.subprocess, "Popen", Proc)
    monkeypatch.setattr(rs.atexit, "register", lambda f: None)
    base = dict(woosh_python=str(py), woosh_model="dflow", woosh_port=8771, stereo=True, woosh_hf_cache=str(tmp_path),
                woosh_frames=None)
    rs.start_woosh(types.SimpleNamespace(amp="full", level=None, **base))
    rs.start_woosh(types.SimpleNamespace(amp="punch", level="loudness", **base))
    rs.start_woosh(types.SimpleNamespace(amp=None, level="peak", **base))
    assert cmds[0][-2:] == ["--amp", "--stereo"] and "--level" not in cmds[0] and "full" not in cmds[0]
    assert cmds[1][-5:] == ["--amp", "punch", "--level", "loudness", "--stereo"]
    assert "--amp" not in cmds[2] and "--level" not in cmds[2]


def _main_kwargs(rs, monkeypatch, argv):
    captured = {}

    class Stop(Exception):
        pass

    def fake_serve(**kw):
        captured.update(kw)
        raise Stop()

    monkeypatch.setattr(rs, "serve", fake_serve)
    monkeypatch.setattr(rs, "check_weights", lambda name: None)
    monkeypatch.delenv("SARGE_DECODER_CKPT", raising=False)
    monkeypatch.delenv("SARGE_GEN_TIER", raising=False)
    with pytest.raises(Stop):
        rs.main(["--generator", "sa3", "--no-cache"] + argv)
    return captured


def test_sidecar_flags_default_to_head_and_switch_only_when_given(rs, monkeypatch):
    kw = _main_kwargs(rs, monkeypatch, ["--amp"])
    assert kw["gen_kwargs"]["amp"] is True and kw["gen_kwargs"]["level"] == "peak" and kw["vram_budget_kb"] == 50_000.0
    assert not {"world", "event_log", "predict_fixes", "pool_fix", "concurrent_gen", "keep_warm_ms",
                "sibling_on_miss"} & set(kw)
    kw = _main_kwargs(rs, monkeypatch, [])
    assert "amp" not in kw["gen_kwargs"]
    kw = _main_kwargs(rs, monkeypatch, ["--amp", "punch", "--level", "loudness", "--world"])
    assert kw["gen_kwargs"]["amp"] == "punch" and kw["gen_kwargs"]["level"] == "loudness"
    assert kw["world"] is True and kw["vram_budget_kb"] == 256_000.0
    kw = _main_kwargs(rs, monkeypatch, ["--world", "--vram-kb", "1000", "--short-clips", "--level", "loudness"])
    assert kw["vram_budget_kb"] == 1000.0 and kw["gen_kwargs"]["level"] == "loudness"
    kw = _main_kwargs(rs, monkeypatch, ["--short-clips"])
    assert "level" not in kw["gen_kwargs"]
    kw = _main_kwargs(rs, monkeypatch, ["--short-clips", "--level", "peak"])
    assert kw["gen_kwargs"]["level"] == "peak"
    kw = _main_kwargs(rs, monkeypatch, ["--predict-fixes", "--keep-warm-ms", "40", "--event-log", "x.jsonl"])
    assert kw["predict_fixes"] is True and kw["keep_warm_ms"] == 40.0 and kw["event_log"] == "x.jsonl"


@pytest.mark.parametrize("amp_mode", ["punch", "full", "off"])
def test_loudness_mode_carries_intensity_by_level_and_never_inverts(amp_mode):
    sr = 48000
    dense = (np.random.default_rng(9).standard_normal(sr) * 0.05).astype(np.float32)
    m = {i: L.momentary_max(post.process(dense, sr, "splash", i, level="loudness", amp_mode=amp_mode, stereo=True), sr)
         for i in ("soft", "normal", "hard")}
    assert m["normal"] - m["soft"] == pytest.approx(6.0, abs=0.05)
    assert m["hard"] - m["normal"] == pytest.approx(3.0, abs=0.05)
    for tau in (0.003, 0.02, 0.1):
        x = _burst(sr, sr, tau, seed=9)
        m = [L.momentary_max(post.process(x, sr, "break", i, level="loudness", amp_mode=amp_mode, stereo=True), sr)
             for i in ("soft", "normal", "hard")]
        assert m[0] <= m[1] + 1e-6 and m[1] <= m[2] + 1e-6
