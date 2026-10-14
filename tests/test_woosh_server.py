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
