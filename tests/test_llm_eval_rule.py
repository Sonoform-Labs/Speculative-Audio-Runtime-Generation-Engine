import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "scripts", "bench")


def _le():
    if BENCH not in sys.path:
        sys.path.insert(0, BENCH)
    import llm_eval
    return llm_eval


def _num(U, D, G=0.0, W=0.0, A=None):
    return {"U": U, "D": D, "G": G, "W": W, "A": (U + D) / 2.0 if A is None else A, "Q": (U + D) / 2.0}


def test_pick_is_best_eligible_and_bars_are_inclusive():
    le = _le()
    num = {"llm": _num(0.0, -0.1), "wr_named": _num(0.3, 0.1, G=-0.11), "wr_texture": _num(0.1, -0.10, W=-0.15),
           "wr_title": _num(0.2, -0.11), "wr_caption": _num(0.05, 0.0, G=-0.10)}
    dec = le.decide_selection(num, {})
    assert dec["eligible"] == ["wr_texture", "wr_caption"]
    assert dec["pick"] == "wr_caption"
    assert dec["confirm"]


def test_tie_band_goes_to_all_rows_then_rejections_then_order():
    le = _le()
    num = {"llm": _num(0.0, 0.0), "wr_named": _num(0.200, 0.0, A=0.10), "wr_texture": _num(0.192, 0.0, A=0.12),
           "wr_title": _num(0.196, 0.0, A=0.12), "wr_caption": _num(0.150, 0.0, A=0.50)}
    assert le.decide_selection(num, {})["pick"] == "wr_texture"
    assert le.decide_selection(num, {"wr_texture": 2, "wr_title": 1})["pick"] == "wr_title"
    assert set(le.decide_selection(num, {})["tied"]) == {"wr_named", "wr_texture", "wr_title"}


def test_margin_over_v1_and_no_eligible_style_stop_confirmation():
    le = _le()
    num = {"llm": _num(0.10, 0.0), "wr_named": _num(0.13, 0.0), "wr_texture": _num(-1, -1), "wr_title": _num(-1, -1),
           "wr_caption": _num(-1, -1)}
    dec = le.decide_selection(num, {})
    assert dec["pick"] == "wr_named" and not dec["confirm"]
    num["wr_named"] = _num(0.14, 0.0)
    assert le.decide_selection(num, {})["confirm"]
    bad = dict(num, wr_named=_num(0.5, -0.2))
    dec = le.decide_selection(dict(bad, wr_texture=_num(0, 0, G=-0.2), wr_title=_num(0, 0, W=-0.2),
                                   wr_caption=_num(0, -0.3)), {})
    assert dec["pick"] is None and not dec["confirm"]


def test_goal_needs_unseen_lower_bound_above_zero():
    le = _le()

    def g(d, lo, n=3):
        return {"keys": list(range(n)), "cmp": {"wr_title vs phrasebook": {"delta": d, "ci": [lo, d + 0.1]}}}

    G = {"unseen": g(0.05, -0.01), "demo": g(-0.05, -0.2), "guns": g(-0.10, -0.3), "water": g(-0.10, -0.4)}
    out, meets = le.goal_verdict(G, "wr_title")
    assert not out["C1 unseen"]["pass"] and not meets
    assert out["C2 demo"]["pass"] and out["C3 guns"]["pass"] and out["C4 water"]["pass"]
    G["unseen"] = g(0.05, 0.001)
    assert le.goal_verdict(G, "wr_title")[1]
    G["demo"] = g(-0.051, -0.2)
    assert not le.goal_verdict(G, "wr_title")[1]


def test_style_arms_meet_v1_only_when_it_runs():
    le = _le()
    with_v1 = le.cmp_pairs(("phrasebook", "caption", "wr_texture", "llm"))
    assert ("wr_texture", "llm") in with_v1 and ("wr_texture", "caption") in with_v1
    assert ("wr_texture", "llm") not in le.cmp_pairs(("phrasebook", "caption", "wr_texture"))
    assert le.cmp_pairs(("phrasebook", "caption", "llm")) == (("caption", "phrasebook"), ("llm", "phrasebook"),
                                                              ("llm", "caption"))
