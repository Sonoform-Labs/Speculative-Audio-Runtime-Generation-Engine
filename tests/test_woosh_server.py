import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import woosh_server as ws


class _T:
    def __init__(self, v):
        self.v = np.array(v, dtype=np.float32)

    def clone(self):
        return _T(self.v.copy())


def test_cond_cache_hit_returns_copy_of_dict():
    calls = []

    def make(p):
        calls.append(p)
        return {"cross_attn_cond": _T([1.0, 2.0]), "cross_attn_cond_mask": _T([1.0, 0.0])}

    c = ws.CondCache(4)
    a = c.get("glass", make)
    a["cfg"] = 4.5
    b = c.get("glass", make)
    assert calls == ["glass"]
    assert (c.hits, c.misses) == (1, 1)
    assert "cfg" not in b
    b["cfg"] = 1.0
    assert "cfg" not in c.get("glass", make)
    assert b["cross_attn_cond"] is not a["cross_attn_cond"]
    np.testing.assert_array_equal(b["cross_attn_cond"].v, a["cross_attn_cond"].v)


def test_cond_cache_evicts_least_recent():
    calls = []

    def make(p):
        calls.append(p)
        return {"x": _T([len(p)])}

    c = ws.CondCache(2)
    c.get("a", make)
    c.get("bb", make)
    c.get("a", make)
    c.get("ccc", make)
    assert list(c.items) == ["a", "ccc"]
    c.get("bb", make)
    assert calls == ["a", "bb", "ccc", "bb"]


def test_cond_cache_zero_size_always_makes():
    calls = []
    c = ws.CondCache(0)
    for _ in range(3):
        c.get("a", lambda p: calls.append(p) or {"x": _T([0.0])})
    assert calls == ["a", "a", "a"]
    assert not c.items and c.misses == 3 and c.hits == 0


def test_eps_torch_hands_out_buffers_in_order_and_delegates():
    class Fake:
        def ones(self, n):
            return [1] * n

    e = ws.EpsTorch(Fake(), ["e1", "e2"])
    assert e.ones(2) == [1, 1]
    assert e.randn_like("x") == "e1"
    assert e.randn_like("x") == "e2"


def test_renoise_schedule_unchanged():
    assert ws.renoise_for(4) == [0, 0.5, 0.5, 0.3]
    r = ws.renoise_for(6)
    assert r[0] == 0.0 and len(r) == 6 and r[1] == pytest.approx(0.5) and r[-1] == pytest.approx(0.3)


def test_adopted_options_are_known_and_valid():
    assert set(ws.ADOPTED) == set(ws.MODELS)
    for opts in ws.ADOPTED.values():
        assert set(opts) <= set(ws.OPTIONS)
        full = dict(ws.OPTIONS, **opts)
        assert full["graph"] in ws.GRAPHS
        assert full["precision"] in ws.PRECISIONS
        assert full["ae_precision"] in ws.AE_PRECISIONS
        assert full["time_dtype"] in ws.TIME_DTYPES
    assert ws.OPTIONS == {"cond_cache": 0, "graph": "none", "precision": "fp32", "ae_precision": "fp32",
                          "batch_cfg": False, "time_dtype": "float64", "mod_cache": False}


def test_parent_watch_fires_when_the_parent_exits():
    import subprocess
    import threading

    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)"])
    gone = threading.Event()
    t = ws.watch_parent(p.pid, on_gone=gone.set)
    assert not gone.wait(0.1)
    p.wait(timeout=30)
    assert gone.wait(10.0)
    t.join(timeout=5.0)


def test_parent_watch_fires_at_once_for_a_dead_pid():
    import subprocess
    import threading

    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait(timeout=30)
    gone = threading.Event()
    ws.watch_parent(p.pid, on_gone=gone.set)
    assert gone.wait(10.0)


def test_parent_watch_ignores_a_live_parent():
    import os
    import threading

    gone = threading.Event()
    ws.watch_parent(os.getpid(), on_gone=gone.set)
    assert not gone.wait(0.5)


def test_istft_matches_torch_istft_bitwise():
    torch = pytest.importorskip("torch")

    class Head:
        n_fft, hop_length = 960, 480
        window = torch.hann_window(960)

    g = torch.Generator().manual_seed(3)
    for frames in (101, 151):
        spec = torch.complex(torch.randn(1, 481, frames, generator=g), torch.randn(1, 481, frames, generator=g))
        ref = torch.istft(spec, 960, 480, 960, Head.window, center=True)
        out = ws.Istft(Head, frames)(spec)
        assert out.shape == ref.shape == (1, (frames - 1) * 480)
        assert torch.equal(out, ref)


def test_frames_table_names_every_family_that_reaches_woosh():
    from sarge.amp import FAMILIES
    assert set(ws.FRAMES) == {"impact", "step", "other"}
    assert set(ws.FRAMES) - {"other"} <= set(FAMILIES)
    assert all(isinstance(n, int) and n >= 101 for n in ws.FRAMES.values())


def test_parse_frames_default_single_count_and_family_pairs():
    assert ws.parse_frames(None) == ws.FRAMES and ws.parse_frames(None) is not ws.FRAMES
    assert ws.parse_frames("") == ws.FRAMES
    assert ws.parse_frames("151") == {"impact": 151, "step": 151, "other": 151}
    assert ws.parse_frames("impact=501") == dict(ws.FRAMES, impact=501)
    assert ws.parse_frames(" step = 201 ,other=301") == dict(ws.FRAMES, step=201, other=301)
    assert ws.parse_frames("blast=151")["blast"] == 151
    with pytest.raises(ValueError):
        ws.parse_frames("glass=501")


def test_frames_for_picks_the_bucket_of_the_amp_family():
    t = {"impact": 501, "step": 201, "other": 301}
    assert [ws.frames_for(e, t) for e in ("footstep", "bullet impact", "break", "punch", "jump")] == [
        201, 501, 501, 501, 201]
    assert ws.frames_for("splash", t) == 301
    assert ws.frames_for("shoot", t) == 301
    assert ws.frames_for("shoot", dict(t, blast=151)) == 151


def test_prepare_warms_each_distinct_count_once_in_ascending_order():
    class Cuda:
        def __init__(self):
            self.mem = 0

        def synchronize(self):
            pass

        def memory_reserved(self):
            return self.mem

    class G:
        def __init__(self):
            self.torch = type("T", (), {"cuda": Cuda()})()
            self.calls = []

        def __call__(self, prompt, seed, mark=None, frames=None):
            self.calls.append((prompt, seed, frames))
            self.torch.cuda.mem += frames * 2 ** 20

    g = G()
    out = ws.prepare(g, {"impact": 501, "step": 201, "other": 201})
    assert g.calls == [(ws.WARM, 0, 201), (ws.WARM, 0, 501)]
    assert sorted(out) == [201, 501]
    assert out[201]["reserved_mb"] == 201 and out[501]["reserved_mb"] == 501 and out[501]["warm_s"] >= 0.0


class _HostOnly:
    def __init__(self, torch):
        self._t = torch
        self.cuda = type("C", (), {"synchronize": staticmethod(lambda: None),
                                   "memory_reserved": staticmethod(lambda: 0)})()

    def __getattr__(self, name):
        return getattr(self._t, name)


def _stub_dflow(torch, frames=101):
    g = object.__new__(ws.Gen)
    g.torch, g.ldm, g.frames, g.plain = _HostOnly(torch), None, frames, False
    g.kind, g.graph, g.steps = "dflow", "full", 4
    g.stats, g.buckets = {"nfe": []}, {}
    g._cond = lambda p: {"c": torch.full((2,), float(len(p)))}
    captures, replays = [], []

    def capture(cond, n):
        st = {"noise": torch.zeros(1, 128, n), "eps": [torch.zeros(1, 128, n)], "cond": {"c": torch.zeros(2)},
              "out": torch.zeros(1, (n - 1) * 480)}
        g.buckets[n] = {"g": type("R", (), {"replay": staticmethod(lambda: replays.append(n))})(), "g_ae": None,
                        "st": st}
        captures.append(n)

    g._capture_dflow = capture
    return g, captures, replays


def test_dflow_request_replays_the_bucket_of_its_length_without_a_new_capture():
    torch = pytest.importorskip("torch")
    g, captures, replays = _stub_dflow(torch)
    ws.prepare(g, {"impact": 501, "step": 101, "other": 101})
    assert captures == [101, 501] and replays == [101, 501]
    out = g("glass", 7, frames=501)
    assert captures == [101, 501] and replays == [101, 501, 501]
    assert out.shape == (500 * 480, 1)
    assert torch.equal(g.buckets[501]["st"]["cond"]["c"], torch.full((2,), 5.0))
    assert torch.equal(g.buckets[101]["st"]["cond"]["c"], torch.full((2,), float(len(ws.WARM))))
    g("step", 3)
    assert captures == [101, 501] and replays == [101, 501, 501, 101]


def test_dflow_noise_is_drawn_at_the_bucket_length_from_the_request_seed():
    torch = pytest.importorskip("torch")
    g, _c, _r = _stub_dflow(torch)
    g("a", 11, frames=151)
    torch.manual_seed(11)
    assert torch.equal(g.buckets[151]["st"]["noise"], torch.randn(1, 128, 151))


def test_finish_drops_the_end_of_window_burst_before_levelling():
    t = np.arange(ws.SR) / ws.SR
    x = (0.3 * np.exp(-t * 3.0) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    x[-200:] = 0.9
    y = ws.finish(x[:, None], "break", "normal", False, False)
    assert len(y) <= ws.SR - int(ws.END_CUT_S * ws.SR)
    assert np.abs(y[: len(y) // 4]).max() > 0.8


def test_render_retries_a_silent_take_and_keeps_good_ones_untouched():
    seeds = []

    def gen(prompt, seed, frames=101):
        seeds.append(seed)
        x = np.zeros((ws.SR, 1), dtype=np.float32)
        x[-100:] = 0.5
        if len(seeds) > 1:
            x[1000:2000] = 0.3
        return x

    out = ws.render(gen, "metal step", 7, frames=101)
    assert len(seeds) == 2 and seeds[0] == 7 and seeds[1] != 7
    assert ws.body_peak(out) == np.float32(0.3)
    seeds.clear()
    ws.render(lambda p, s, frames=101: (seeds.append(s), np.full((ws.SR, 1), 0.2, dtype=np.float32))[1], "x", 3)
    assert seeds == [3]


def test_render_gives_up_after_the_retry_budget_and_returns_the_loudest():
    calls = []

    def gen(prompt, seed, frames=101):
        calls.append(seed)
        return np.full((ws.SR, 1), 0.001 * len(calls), dtype=np.float32)

    out = ws.render(gen, "x", 1)
    assert len(calls) == 1 + ws.SILENT_RETRIES
    assert ws.body_peak(out) == np.float32(0.001 * len(calls))


def test_render_judges_a_physics_take_by_what_the_chain_keeps():
    t = np.arange(ws.SR) / ws.SR
    bright = (0.3 * np.sin(2 * np.pi * 6000.0 * t) * np.minimum(1.0, t / 0.01)).astype(np.float32)[:, None]
    low = (0.3 * np.sin(2 * np.pi * 120.0 * t) * np.exp(-t / 0.1)).astype(np.float32)[:, None]
    burst = np.zeros((ws.SR, 1), dtype=np.float32)
    end, n = ws.SR - int(ws.END_CUT_S * ws.SR), int(0.002 * ws.SR)
    burst[end - n:end, 0] = 0.01 * np.sin(2 * np.pi * 2000.0 * t[:n])
    soft = {"t40_s": 0.2, "punch_scale": 0.25, "drive": 1.0, "floor_db": -45.0, "lowpass_hz": 400.0}
    assert ws.body_peak(bright, soft) < ws.SILENT_PEAK <= ws.body_peak(bright)
    assert ws.body_peak(burst, dict(soft, lowpass_hz=None)) < ws.SILENT_PEAK <= ws.body_peak(burst)
    for first in (bright, burst):
        seeds = []

        def gen(prompt, seed, frames=101):
            seeds.append(seed)
            return first if len(seeds) == 1 else low

        assert ws.render(gen, "x", 5, shape=soft) is low and len(seeds) == 2
        seeds.clear()
        assert ws.render(gen, "x", 5) is first and seeds == [5]


def test_render_key_matches_the_old_request_path_and_judges_by_the_shape_only_with_physics():
    from sarge.prompts import build
    from sarge.types import CacheKey, seed_for
    calls = []

    def gen(prompt, seed, frames=101):
        calls.append((prompt, seed, frames))
        return np.full((ws.SR, 1), 0.2, dtype=np.float32)

    frames = ws.parse_frames(None)
    k = CacheKey("impact", "metal", "hard", "default")
    ws.render_key(gen, k, 2, frames)
    assert calls == [(build("impact", "metal", "hard", "default"), seed_for(k, 2), ws.frames_for("impact", frames))]
    raw = np.full((ws.SR, 1), 0.2, dtype=np.float32)
    np.testing.assert_array_equal(ws.finish_key(raw, k, "full", True), ws.finish(raw, "impact", "hard", "full", True))
    seen = {}
    orig = ws.render

    def spy(g, prompt, seed, shape=None, **kw):
        seen["shape"] = shape
        return orig(g, prompt, seed, shape=shape, **kw)

    ws.render = spy
    try:
        ws.render_key(gen, k, 0, frames)
        assert seen["shape"] is None
        kp = CacheKey("impact", "metal", "hard", "default",
                      "bounce;a=rubber,large,inflated,ball;b=metal,huge,static,floor;c=soft;r=short,metal")
        ws.render_key(gen, kp, 0, frames)
        assert seen["shape"]["lowpass_hz"] == 400.0
    finally:
        ws.render = orig
