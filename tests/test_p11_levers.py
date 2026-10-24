import random
import sys
import threading
import types

import numpy as np
import pytest

from sarge import canon, pool as prompt_pool, prompts, scene, verify, words, writer as wr
from sarge.types import CacheKey, seed_for

from test_generic_levers import (BENCH, CAP, FakeGen, FakeLLM, FakeVerifier, LOOP, NAMES, SCRIPTS, _canon_raw,
                                 _head_module, _random_requests, _rest_cmd, _traced, _up)

CLEAN = CAP + "+clean"


@pytest.fixture
def ws(monkeypatch):
    if SCRIPTS not in sys.path:
        monkeypatch.syspath_prepend(SCRIPTS)
    import woosh_server
    return woosh_server


def _odd_names(n=60, seed=20261024):
    rng = random.Random(seed)
    letters = "bcdfghjklmnpqrstvwxz"

    def w():
        return "".join(rng.choice(letters) for _ in range(rng.randint(3, 7))) + rng.choice("aeiou")

    out = []
    for i in range(n):
        core = [w() for _ in range(rng.randint(1, 3))]
        form = i % 6
        if form == 0:
            out.append("SFX_" + "_".join(x.capitalize() for x in core) + "_%02d" % rng.randint(1, 9))
        elif form == 1:
            out.append("_".join(core) + "_" + rng.choice("LR"))
        elif form == 2:
            out.append("".join(x.capitalize() for x in core) + str(rng.randint(1, 5)))
        elif form == 3:
            out.append("Play_" + "_".join(core) + "_v%d" % rng.randint(1, 4))
        elif form == 4:
            out.append("snd-" + "-".join(core) + ".wav")
        else:
            out.append("Prop_" + "_".join(x.upper() for x in core) + "_Take_%d" % rng.randint(1, 4))
    return out


def test_clean_drops_format_engine_take_and_side_marks_and_keeps_the_rest():
    c = words.clean_phrase
    assert c("SFX_Zorbo_Krell_01") == "zorbo krell"
    assert c("Zorbo_Step_L") == "zorbo step" and c("zorbo_step_r") == "zorbo step"
    assert c("Zorbo_Krell_v2") == "zorbo krell" and c("Zorbo_Take_03") == "zorbo" and c("ZorboKrell3") == "zorbo krell"
    assert c("Play_Zorbo_Krell") == "zorbo krell" and c("A_Zorbo") == "zorbo"
    assert c("Zib_A_Zab") == "zib a zab" and c("Play") == "play" and c("A") == "a"
    assert c("Prop_Zorbo_Prefab") == "zorbo" and c("snd-zorbo-krell.wav") == "zorbo krell"
    assert c("Xk47_Zorbo") == "xk 47 zorbo" and c("Zorbo_9mm") == "zorbo 9 mm" and c("V8_Zorbo") == "v 8 zorbo"
    assert c("Zorbo_01_Krell") == "zorbo krell" and c("SFXZorboKrell") == "zorbo krell"
    assert c("Zorbo_Krell_A_02") == "zorbo krell"
    assert c("SFX_01") == "sfx 01" and c("A__B") == "a b" and c("") == "" and c(None) == ""
    assert c("Zorbo_Loop") == "zorbo loop" and c("Zorbo_Left") == "zorbo left"


def test_clean_word_sets_are_format_and_engine_words_only():
    assert words.ENGINE_WORDS <= scene.BOILER and words.SIDE_WORDS <= scene.BOILER
    assert words.FORMAT_WORDS & scene.BOILER == {"fx", "sfx"}
    judged = {"footstep", "footsteps", "step", "gunshot", "gun", "pistol", "rifle", "shotgun", "punch", "glass",
              "shatter", "door", "splash", "dog", "bark", "horn", "sword", "explosion", "click", "water", "slam",
              "rain", "metal", "wood", "stone", "gravel", "grass", "snow", "sand", "magnum", "fire", "hit", "break"}
    every = words.FORMAT_WORDS | words.ENGINE_WORDS | words.SIDE_WORDS | words.LEAD_WORDS | words.TAKE_MARKS
    assert not every & judged
    assert not every & prompts.LOOP_WORDS


def test_clean_modifier_parses_with_loop_in_either_order_and_refuses_typos():
    assert prompts.parse_template(CLEAN) == ("caption", True)
    assert prompts.template_mods(CLEAN) == {"clean"}
    assert prompts.template_mods(CAP + "+clean+loop") == {"clean", "loop"}
    assert prompts.template_mods(CAP + "+loop+clean") == {"clean", "loop"}
    assert prompts.check_style(CLEAN) == CLEAN and CLEAN not in prompts.STYLES
    for bad in ("template:+clean", "template:nope+clean", "template:caption+cleaner", "template:caption+clean+x"):
        with pytest.raises(Exception):
            prompts.check_style(bad)


def test_flags_off_prompts_match_head_on_prefixed_numbered_and_sided_names():
    head = _head_module("src/sarge/prompts.py", "sarge._head_prompts_p11")
    reqs = [(n, "") for n in _odd_names()] + [(n, m) for n, m in zip(_odd_names(20, 3), _odd_names(20, 4))]
    for name in NAMES:
        for sfx in ("", "+tt", "-tt", "+loop"):
            style = prompts.TEMPLATE + name + sfx
            for ev, mat in reqs:
                for model in ("sa3", "woosh"):
                    assert prompts.template(style, ev, mat, "hard", "indoor_small", "", model) == \
                        head.template(style, ev, mat, "hard", "indoor_small", "", model)
    for ev, mat in reqs:
        k = CacheKey(ev, mat, "normal", "default")
        assert verify.key_text(k) == verify.key_text(k, clean=False)
        assert prompts.compose("tuned", k) == head.compose("tuned", k)
        assert prompts.compose("generic", k) == head.compose("generic", k)


def test_clean_template_is_the_template_on_the_cleaned_names_for_every_template():
    reqs = [(n, "") for n in _odd_names()] + _random_requests()
    for name in NAMES:
        for sfx in ("", "+tt", "-tt"):
            style = prompts.TEMPLATE + name + sfx
            for ev, mat in reqs:
                for model in ("sa3", "woosh"):
                    want = prompts.template(style, words.clean_phrase(ev), words.clean_phrase(mat), "hard",
                                            "indoor_small", "", model)
                    assert prompts.template(style + "+clean", ev, mat, "hard", "indoor_small", "", model) == want
    t = prompts.template
    assert t(CLEAN, "SFX_Zorbo_Krell_01", "") == "A single zorbo krell sound, recorded up close."
    assert t(CAP, "SFX_Zorbo_Krell_01", "") == "A single sfx zorbo krell 01 sound, recorded up close."
    assert t(CLEAN, "Zorbo_Step_L", "Zzgrit_02", model="sa3") == \
        "TrackType: SFX, A single zzgrit zorbo step sound, recorded up close."
    assert t(CLEAN + "+loop", "SFX_Zorbo_Loop_03", "") == "A continuous zorbo sound, recorded up close."


def test_clean_runs_one_code_path_for_any_name():
    paths = set()
    for ev in _odd_names(30, 9):
        paths.add(_traced(prompts.template, CLEAN, ev, "", "normal", "default", "", "woosh",
                          files={prompts.__file__})[1])
    assert len(paths) == 1


def test_canonical_names_pass_through_the_cleaning_and_texts_follow_it():
    c = {"name": "zz hum", "material": "", "continuous": False}
    k = CacheKey("SFX_Zorbo_01", "Zzgrit_L", "normal")
    assert prompts.template(CLEAN, k.event_type, k.material, canon=c) == "A single zzgrit zz hum sound, recorded up close."
    assert verify.key_text(k, clean=True) == "zzgrit zorbo sound" and verify.key_text(k) == "zzgrit l sfx zorbo 01 sound"
    assert verify.event_text(k, clean=True) == "zorbo sound"
    assert canon.texts(c, k, CLEAN) == {"request": "zzgrit zz hum sound", "event": "zz hum sound"}
    assert canon.texts(c, k, CAP) == {"request": "zzgrit l zz hum sound", "event": "zz hum sound"}
    assert verify.style_texts(CLEAN) == {"loop": False, "clean": True}
    assert verify.style_texts("tuned") == {"loop": False, "clean": False}
    rec = canon.judge(_canon_raw("zz hum"), k, wr.key_text(k), canon.style(CLEAN))
    assert rec["woosh"] == "A single zzgrit zz hum sound, recorded up close."
    assert canon.style(CLEAN)["canon"]["template"] == CLEAN
    llm = FakeLLM(lambda m: _canon_raw("zz hum"))
    assert wr.writer_id(llm.ident(), canon.style(CLEAN)) != wr.writer_id(llm.ident(), canon.style(CAP))


def test_worker_scores_against_the_cleaned_request_when_the_style_cleans(ws):
    k = CacheKey("SFX_Zorbo_Krell_02", "", "normal")
    v = FakeVerifier(lambda i: 0.1 * i)
    up = _up(ws, 2, v)
    up.render(k, 0, CLEAN, None, None, False, "peak")
    assert v.texts == ["zorbo krell sound"] * 2
    assert [p for p, _s in up.gen.calls] == ["A single zorbo krell sound, recorded up close."] * 2
    plain = FakeVerifier(lambda i: 0.1 * i)
    up2 = _up(ws, 2, plain)
    up2.render(k, 0, CAP, None, None, False, "peak")
    assert plain.texts == ["sfx zorbo krell 02 sound"] * 2
    lp = FakeVerifier(lambda i: 0.0)
    up3 = _up(ws, 2, lp, score="event-min")
    up3.render(CacheKey("zorbo_loop", "zzgrit", "normal"), 0, LOOP, None, None, False, "peak")
    assert lp.texts[:2] == ["zzgrit zorbo sound", "zorbo sound"]


def test_sidecar_clean_style_reaches_both_routes_and_the_worker(monkeypatch):
    cmd, cap = _rest_cmd(monkeypatch, ["--prompts", CLEAN, "--woosh-upgrade", "flow", "--woosh-upgrade-solver",
                                       "midpoint:16", "--woosh-upgrade-best", "4"])
    assert cap["gen_kwargs"]["prompt_style"] == CLEAN
    i = cmd.index("--prompts")
    assert cmd[i + 1] == CLEAN


def test_small_pool_is_the_caption_and_the_canonical_caption_with_one_choice_per_key(monkeypatch, tmp_path, ws):
    made = []

    class Llama(FakeLLM):
        def __init__(self, *a, **kw):
            super().__init__(lambda m: _canon_raw("zz hum"))
            made.append(self)

    monkeypatch.setattr(wr, "LlamaBackend", Llama)
    for style in (CAP, CLEAN):
        made.clear()
        _cmd, cap = _rest_cmd(monkeypatch, ["--prompts", style, "--woosh-upgrade", "flow", "--woosh-upgrade-best",
                                            "4", "--woosh-upgrade-pool", "canon", "--writer-store",
                                            str(tmp_path / style.replace(":", "_").replace("+", "_"))])
        p = cap["prompt_pool"]
        assert "writer" not in cap and p.primary is None and [w.style for w in p.writers] == ["canon"]
        w = p.writers[0]
        assert w.spec["template"] == style and len(made) == 1
        w.setup()
        k = CacheKey("SFX_Zorbo_Krell_01", "", "normal")
        rec = w.write(k)
        w.store.put(str(k), rec)
        assert rec["woosh"] == "A single zz hum sound, recorded up close."
        assert p.prompts(k) == ["A single zz hum sound, recorded up close."]
        assert p.texts(k) == {"request": "zz hum sound", "event": "zz hum sound"}
    scores = [0.2, 0.3, 0.6, 0.7] + [0.1] * 32
    v = FakeVerifier(lambda i: scores[i] if i < len(scores) else 0.1)
    up = _up(ws, 2, v)
    k = CacheKey("SFX_Zorbo_Krell_01", "", "normal")
    base = ws.prompt_for(k, CLEAN)
    pool = ["A single zz hum sound, recorded up close."]
    texts = {"request": "zz hum sound", "event": "zz hum sound"}
    up.render(k, 0, CLEAN, None, None, False, "peak", pool=pool, texts=texts)
    assert base == "A single zorbo krell sound, recorded up close."
    assert [p for p, _s in up.gen.calls] == [base, base] + pool * 2
    assert up.choice[(str(k), (base, pool[0]))] == 1 and v.texts == ["zz hum sound"] * 4
    up.render(k, 1, CLEAN, None, None, False, "peak", pool=pool, texts=texts)
    up.render(k, 2, CLEAN, None, None, False, "peak", pool=pool, texts=texts)
    assert [p for p, _s in up.gen.calls[4:]] == pool * 4 and up.info()["pool_keys"] == 1
    assert [s for _p, s in up.gen.calls[4:6]] == [seed_for(k, verify.candidate_variant(1, j)) for j in range(2)]


def test_vram_savings_parse_and_default_off(ws, monkeypatch):
    assert ws.parse_saves("") == frozenset() and ws.parse_saves(" encoder , share ") == {"encoder", "share"}
    for bad in ("share,nope", "free"):
        with pytest.raises(ValueError):
            ws.parse_saves(bad)
    for argv in (["--upgrade-vram-save", "share"], ["--upgrade", "flow", "--upgrade-vram-save", "half"]):
        monkeypatch.setattr(sys, "argv", ["woosh_server.py"] + argv)
        with pytest.raises(SystemExit):
            ws.main()
    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=lambda: None, commit=lambda: None)
    assert up.saves == frozenset() and "vram_save" not in up.info()


def test_vram_saving_shares_the_pool_only_when_asked(ws):
    order = []

    def gen():
        g = types.SimpleNamespace(should_stop=None, share_pool=False, torch="T")
        g.capture = lambda frames: order.append(("capture", g.share_pool))
        return g

    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=gen, commit=lambda: None,
                    saves={"share", "encoder"})
    assert up.start(wait=True) == "ready" and order == [("capture", True)]
    assert up.info()["vram_save"] == ["encoder", "share"]
    order.clear()
    off = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=gen, commit=lambda: None)
    assert off.start(wait=True) == "ready" and order == [("capture", False)]


def test_graph_pool_is_the_served_pool_only_when_sharing(ws):
    class Graph:
        def pool(self):
            return ("pool", 7)

    g = ws.Gen.__new__(ws.Gen)
    g.share, g.share_pool = types.SimpleNamespace(buckets={101: None, 201: {"g": Graph()}}), False
    assert g.graph_pool() is None
    g.share_pool = True
    assert g.graph_pool() == ("pool", 7)
    g.share = types.SimpleNamespace(buckets={})
    assert g.graph_pool() is None
    g.share = None
    assert g.graph_pool() is None


def test_drop_encoder_moves_only_the_encoder_off_the_device(ws, monkeypatch):
    import torch
    monkeypatch.setattr(ws, "release_cache", lambda t: True)
    inner = torch.nn.Module()
    inner.encoder, inner.decoder = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    gen = types.SimpleNamespace(ldm=types.SimpleNamespace(autoencoder=types.SimpleNamespace(autoencoder=inner)),
                                torch=torch)
    assert ws.drop_encoder(gen) == 0
    assert all(p.is_meta for p in inner.encoder.parameters())
    assert not any(p.is_meta for p in inner.decoder.parameters())
    assert ws.drop_encoder(types.SimpleNamespace()) == 0


def test_sidecar_vram_save_flag_reaches_the_worker_only_when_set(monkeypatch):
    base = ["--prompts", CAP, "--woosh-upgrade", "flow", "--woosh-upgrade-solver", "midpoint:16"]
    cmd, _ = _rest_cmd(monkeypatch, base)
    assert "--upgrade-vram-save" not in cmd
    got, _ = _rest_cmd(monkeypatch, base + ["--woosh-upgrade-vram-save", "share, encoder"])
    assert got[len(cmd):] == ["--upgrade-vram-save", "share,encoder"]
    import run_sidecar as rs
    for bad in (["--woosh-upgrade-vram-save", "share"], base + ["--woosh-upgrade-vram-save", "free"]):
        with pytest.raises(SystemExit):
            rs.main(["--generator", "hybrid", "--no-cache", "--woosh-python", sys.executable] + bad)
