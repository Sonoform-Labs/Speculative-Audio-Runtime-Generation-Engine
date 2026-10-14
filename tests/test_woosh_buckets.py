import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "bench"))

import woosh_buckets as wb

SR = 48000


def _clip(seconds=1.0):
    rng = np.random.default_rng(5)
    x = np.zeros(int(seconds * SR))
    x[:int(0.1 * SR)] = rng.standard_normal(int(0.1 * SR))
    return x, rng


def test_choose_takes_the_smallest_count_at_or_under_five_percent():
    assert wb.choose({101: 0.5, 151: 0.05, 201: 0.0}) == 151
    assert wb.choose({201: 0.04, 101: 0.2, 151: 0.06}) == 201
    assert wb.choose({101: 0.0, 501: 0.0}) == 101
    assert wb.choose({101: 0.06, 501: 0.051}) is None
    assert wb.choose({101: 1 / 20, 151: 0.0}) == 101


def test_tail_marks_a_clip_still_sounding_at_the_cut():
    t = np.arange(SR) / SR
    ring = np.sin(2 * np.pi * 440 * t) * np.exp(-t / 0.5)
    r = wb.tail(ring, SR)
    assert r["truncated"] and r["trim_at_cut"] and r["content_truncated"] and r["alive_end_s"] == 1.0
    short = ring * (t < 0.3)
    r = wb.tail(short, SR)
    assert not r["truncated"] and not r["trim_at_cut"] and r["alive_end_s"] < 0.32


def test_tail_uses_the_trim_floor():
    x, _ = _clip()
    loud, quiet = x.copy(), x.copy()
    loud[int(0.1 * SR):] = 10 ** (-50 / 20)
    quiet[int(0.1 * SR):] = 10 ** (-60 / 20)
    assert wb.tail(loud, SR)["truncated"]
    assert not wb.tail(quiet, SR)["truncated"]


def test_tail_looks_only_at_the_last_50_ms_and_separates_the_edge():
    x, rng = _clip()
    n = len(x)
    early = x.copy()
    early[n - int(0.1 * SR):n - int(0.095 * SR)] = 0.1 * rng.standard_normal(int(0.005 * SR))
    assert not wb.tail(early, SR)["truncated"]
    late = x.copy()
    late[n - int(0.04 * SR):n - int(0.035 * SR)] = 0.1 * rng.standard_normal(int(0.005 * SR))
    r = wb.tail(late, SR)
    assert r["truncated"] and r["content_truncated"] and not r["edge_peak"]
    edge = x.copy()
    edge[n - int(0.005 * SR):] = 3.0 * rng.standard_normal(int(0.005 * SR))
    r = wb.tail(edge, SR)
    assert r["truncated"] and not r["content_truncated"] and r["edge_peak"]


def test_tail_reads_stereo_as_the_mono_mix_like_the_trim():
    t = np.arange(SR) / SR
    ring = np.sin(2 * np.pi * 440 * t) * np.exp(-t / 0.5)
    assert wb.tail(np.stack([ring, ring], axis=1), SR) == wb.tail(ring, SR)
    assert wb.tail(np.zeros(SR), SR)["truncated"] is False


def test_family_matches_the_routing_family():
    assert [wb.family(e) for e, _m in wb.KEYS] == ["impact", "impact", "step", "step", "step", "step", "step",
                                                   "other", "impact"]
