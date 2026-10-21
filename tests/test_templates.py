import argparse
import os
import random
import re
import sys
import types

import numpy as np
import pytest

from sarge import prompts, words
from sarge.generators.routed import RoutedGenerator
from sarge.types import CacheKey
from sarge.words import split_words

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
NAMES = list(prompts.TEMPLATES)
SEEN = ["shoot", "pistol", "rifle", "gravel", "footstep", "water", "glass", "dog", "bark", "window", "break", "punch",
        "body", "splash", "reload", "metal", "wood", "stone", "car", "horn", "explosion", "door", "creak", "melee"]
REQUESTS = [("shoot", "pistol"), ("shoot", "rifle"), ("break", "window"), ("bullet impact", "glass"),
            ("footstep", "gravel"), ("footstep", "water"), ("splash", "water"), ("punch", "body"),
            ("dog_bark", "dog"), ("door_creak", "wood"), ("car_horn", "car"), ("Pistol_Shot", "9mm"),
            ("melee_hit", "body"), ("water_drip", "water"), ("rain_on_roof", "metal"), ("PlayerFootstep", "snow"),
            ("grappling_hook_fire", "rope"), ("coin_pickup", "gold"), ("", "metal"), ("footstep", "")]


def _random_requests(n=30, seed=20261018):
    rng = random.Random(seed)
    letters = "abcdefghijklmnopqrstuvwxyz"
    out = []
    for i in range(n):
        pool = SEEN + ["".join(rng.choice(letters) for _ in range(rng.randint(3, 9))) for _ in range(6)]
        ev = rng.sample(pool, rng.randint(1, 3))
        mat = [w for w in rng.sample(pool, rng.randint(1, 2)) if w not in ev] or ["zzmat"]
        style = i % 3
        if style == 0:
            out.append(("_".join(ev), " ".join(mat)))
        elif style == 1:
            out.append(("".join(w.capitalize() for w in ev), "_".join(mat).upper()))
        else:
            out.append((" ".join(ev) + str(i), "-".join(mat)))
    return out


def _traced(fn, *args):
    files = {os.path.abspath(prompts.__file__), os.path.abspath(words.__file__)}
    seen = set()

    def local(frame, event, arg):
        if event == "line":
            seen.add((os.path.basename(frame.f_code.co_filename), frame.f_lineno))
        return local

    def outer(frame, event, arg):
        return local if os.path.abspath(frame.f_code.co_filename) in files else None

    old = sys.gettrace()
    sys.settrace(outer)
    try:
        out = fn(*args)
    finally:
        sys.settrace(old)
    return out, frozenset(seen)


@pytest.mark.parametrize("model", ["sa3", "woosh"])
@pytest.mark.parametrize("name", NAMES)
def test_each_template_runs_one_code_path_for_any_event_and_material(name, model):
    paths = set()
    outs = set()
    for ev, mat in _random_requests():
        out, lines = _traced(prompts.template, prompts.TEMPLATE + name, ev, mat, "normal", "default", "", model)
        paths.add(lines)
        outs.add(out)
    assert len(paths) == 1, name
    assert len([1 for f, _ in next(iter(paths)) if f == "prompts.py"]) > 10
    assert len(outs) > 20


def test_all_templates_share_the_same_code_path():
    paths = set()
    for name in NAMES:
        for model in ("sa3", "woosh"):
            for ev, mat in _random_requests(6, seed=7):
                paths.add(_traced(prompts.template, prompts.TEMPLATE + name, ev, mat, "hard", "cave", "", model)[1])
    assert len(paths) == 1


def test_the_tracer_sees_the_phrasebook_branch_on_the_event():
    a = _traced(prompts.build, "shoot", "pistol")[1]
    b = _traced(prompts.build, "footstep", "gravel")[1]
    c = _traced(prompts.build, "car_horn", "car")[1]
    assert a != b and b != c and a != c


def _fixed_words(name):
    text, _tt, space = prompts.TEMPLATES[name]
    return set(split_words(re.sub(r"\{\w+\}", " ", text))) | set(split_words(space)) | \
        set(split_words(prompts.SA3_PREFIX)) | set(split_words(" ".join(prompts.INTENSITY_ADJ.values())))


@pytest.mark.parametrize("name", NAMES)
def test_template_words_are_the_request_words_plus_fixed_structure_words(name):
    fixed = _fixed_words(name)
    for ev, mat in REQUESTS + _random_requests():
        for inten, ctx in (("normal", "default"), ("hard", "indoor_small"), ("soft", "WalkArena"), ("loud", "")):
            for model in ("sa3", "woosh"):
                out = prompts.template(prompts.TEMPLATE + name, ev, mat, inten, ctx, "", model)
                own = set(split_words(" ".join((ev, mat, inten, ctx))))
                assert set(split_words(out)) <= own | fixed, (name, ev, mat, out)


@pytest.mark.parametrize("name", NAMES)
def test_template_is_a_pure_slot_fill(name):
    rng = random.Random(name)
    letters = "bcdfghjklmnpqrstvwxz"
    style = prompts.TEMPLATE + name
    for _ in range(30):
        ev = ["".join(rng.choice(letters) for _ in range(rng.randint(4, 8))) + "o" for _ in range(rng.randint(1, 3))]
        mat = ["".join(rng.choice(letters) for _ in range(rng.randint(4, 8))) + "u" for _ in range(rng.randint(1, 2))]
        for model in ("sa3", "woosh"):
            base = prompts.template(style, "yyevo", "yymatu", "normal", "default", "", model)
            want = base.replace("Yyevo", " ".join(ev).capitalize()).replace("yyevo", " ".join(ev))
            want = want.replace("Yymatu", " ".join(mat).capitalize()).replace("yymatu", " ".join(mat))
            got = prompts.template(style, "_".join(ev), "_".join(mat), "normal", "default", "", model)
            assert got == want, (name, got, want)


@pytest.mark.parametrize("name", NAMES)
def test_template_text_is_tidy_for_empty_and_odd_fields(name):
    cases = [("", "", "", ""), ("", "metal", "normal", "default"), ("footstep", "", "hard", ""),
             ("A__B", "--", "soft", "  "), ("dog_bark", "dog", "normal", "default")] + \
        [(ev, mat, "normal", "default") for ev, mat in _random_requests()]
    for ev, mat, inten, ctx in cases:
        for model in ("sa3", "woosh"):
            out = prompts.template(prompts.TEMPLATE + name, ev, mat, inten, ctx, "", model)
            body = out[len(prompts.SA3_PREFIX):] if out.startswith(prompts.SA3_PREFIX) else out
            assert ",," not in out and " ," not in out and "  " not in out and " ." not in out, out
            assert not body.startswith(",") and not body.endswith(","), out


def test_template_examples_follow_the_written_slots():
    t = prompts.template
    assert t("template:library", "Pistol_Shot", "9mm", model="sa3") == \
        "9 mm pistol shot, single pistol shot, foley, close-distance, exterior, open air"
    assert t("template:library_ef", "shoot", "pistol") == "Shoot pistol, single shoot, foley, close-distance, exterior, open air"
    assert t("template:library_n", "footstep", "gravel") == \
        "Gravel footstep, single normal footstep, foley, close-distance, exterior, open air"
    assert t("template:tags", "shoot", "pistol", model="sa3") == \
        "TrackType: SFX, shoot, pistol, single, close-up, foley, sound effect"
    assert t("template:caption", "melee_hit", "body") == "A single body melee hit sound, recorded up close."
    assert t("template:oneshot", "car_horn", "car", "hard") == "Heavy car horn one-shot, short, isolated, close-mic"
    assert t("template:guide", "water_drip", "water", "normal", "cave", model="sa3") == \
        "TrackType: SFX, Water drip, a single short sound with a quick decay, close perspective, cave"
    assert t("template:plain", "footstep", "snow", "soft", "indoor_small") == "Light snow footstep, indoor small"
    assert t("template:library", "footstep", "snow", "hard", "indoor_small") == \
        "Snow footstep, single heavy footstep, foley, close-distance, indoor small"


def test_template_contact_words_ride_along_with_world():
    from sarge.world import CONTACT_WORDS
    phys = "bounce;a=rubber,large,inflated,ball;b=metal,huge,static,floor;c=soft;r=short,metal"
    assert prompts.template("template:caption", "impact", "metal", phys=phys) == \
        "A single metal impact sound, recorded up close, %s." % CONTACT_WORDS["soft"]
    assert prompts.template("template:library", "impact", "metal", phys=phys).endswith(
        "exterior, open air, " + CONTACT_WORDS["soft"])


def test_sa3_prefix_follows_the_template_and_the_suffix_and_never_reaches_woosh():
    t = prompts.template
    for name in NAMES:
        on = prompts.TEMPLATES[name][1]
        assert t("template:" + name, "shoot", "rifle", model="sa3").startswith(prompts.SA3_PREFIX) == on
        assert t("template:%s+tt" % name, "shoot", "rifle", model="sa3").startswith(prompts.SA3_PREFIX)
        assert not t("template:%s-tt" % name, "shoot", "rifle", model="sa3").startswith(prompts.SA3_PREFIX)
        for s in ("", "+tt", "-tt"):
            assert "TrackType" not in t("template:%s%s" % (name, s), "shoot", "rifle", model="woosh")
    assert t("template:plain+tt", "", "", "normal", "default", "", "sa3") == "TrackType: SFX"


def test_template_styles_parse_and_bad_names_are_refused():
    assert prompts.parse_template("template:caption") == ("caption", True)
    assert prompts.parse_template("template:library+tt") == ("library", True)
    assert prompts.parse_template("template:caption-tt") == ("caption", False)
    assert prompts.parse_template("tuned") is None and prompts.parse_template("generic") is None
    with pytest.raises(ValueError):
        prompts.parse_template("template:nope")
    for ok in ("tuned", "generic", "template:guide", "template:oneshot+tt"):
        assert prompts.check_style(ok) == ok
    for bad in ("template:", "template:nope", "fancy", "", "template:library+xx"):
        with pytest.raises(argparse.ArgumentTypeError):
            prompts.check_style(bad)
    assert set(prompts.STYLES) == {"tuned", "generic"} | {"template:" + n for n in NAMES}


def test_compose_keeps_tuned_and_generic_and_adds_templates():
    k = CacheKey("shoot", "pistol", "normal", "tunnel")
    assert prompts.compose("tuned", k) == prompts.build("shoot", "pistol", "normal", "tunnel")
    assert prompts.compose("generic", k, prompts.SA3_PREFIX) == "TrackType: SFX, shoot, pistol, tunnel"
    assert prompts.compose("template:caption", k) == "A single pistol shoot sound, recorded up close, tunnel."
    assert prompts.compose("template:caption", k, model="sa3") == \
        "TrackType: SFX, A single pistol shoot sound, recorded up close, tunnel."


def test_sa3_template_style_and_a_game_prompt_still_wins():
    from sarge.generators.stable_audio3 import StableAudio3Generator
    g = StableAudio3Generator(prompt_style="template:guide")
    k = CacheKey("Reload", "AK47", "hard", "tunnel")
    assert g._prompt(k) == prompts.template("template:guide", "Reload", "AK47", "hard", "tunnel", "", "sa3")
    assert g._prompt(k).startswith(prompts.SA3_PREFIX)
    g.prompts[str(k)] = "a game prompt"
    assert g._prompt(k) == "a game prompt"
    assert StableAudio3Generator(prompt_style="template:library")._prompt(CacheKey("shoot", "pistol", "normal")) == \
        "Pistol shoot, single shoot, foley, close-distance, exterior, open air"


@pytest.fixture
def ws(monkeypatch):
    if SCRIPTS not in sys.path:
        monkeypatch.syspath_prepend(SCRIPTS)
    import woosh_server
    return woosh_server


def test_woosh_worker_renders_the_template_and_a_sent_prompt_wins(ws):
    k = CacheKey("dog_bark", "dog", "normal", "yard")
    assert ws.prompt_for(k, "template:caption") == "A single dog bark sound, recorded up close, yard."
    assert ws.prompt_for(k, "template:tags+tt") == "dog bark, yard, single, close-up, foley, sound effect"
    assert ws.prompt_for(k, "template:caption", "a big dog barking twice") == "a big dog barking twice"
    assert ws.prompt_for(k) == prompts.build("dog_bark", "dog", "normal", "yard")
    calls = []

    def gen(prompt, seed, frames=101):
        calls.append(prompt)
        return np.full((ws.SR, 1), 0.2, dtype=np.float32)

    ws.render_key(gen, k, 0, ws.parse_frames(None), "template:plain")
    assert calls == ["Dog bark, yard"]


class Fake:
    def __init__(self, style=""):
        self.sample_rate = 44100
        self.prompts = {}
        self.prompt_style = style


def test_routed_cache_tag_names_the_template():
    assert RoutedGenerator(guns=Fake("detailed"), rest=Fake()).post_tag == ""
    assert RoutedGenerator(guns=Fake("generic"), rest=Fake()).post_tag == "|prompts:generic"
    assert RoutedGenerator(guns=Fake("template:caption"), rest=Fake()).post_tag == "|prompts:template:caption"
    assert RoutedGenerator(guns=Fake("template:caption-tt"), rest=Fake()).post_tag != \
        RoutedGenerator(guns=Fake("template:caption"), rest=Fake()).post_tag


@pytest.fixture
def rs(monkeypatch):
    if SCRIPTS not in sys.path:
        monkeypatch.syspath_prepend(SCRIPTS)
    import run_sidecar
    return run_sidecar


def _hybrid_kwargs(rs, monkeypatch, tmp_path, argv):
    captured = {}

    class Stop(Exception):
        pass

    def fake_serve(**kw):
        captured.update(kw)
        raise Stop()

    py = tmp_path / "python.exe"
    py.write_bytes(b"")
    monkeypatch.setattr(rs, "serve", fake_serve)
    monkeypatch.setattr(rs, "check_weights", lambda name: None)
    monkeypatch.setattr(rs, "early_woosh", lambda env=None: False)
    monkeypatch.delenv("SARGE_DECODER_CKPT", raising=False)
    with pytest.raises(Stop):
        rs.main(["--generator", "hybrid", "--no-cache", "--woosh-python", str(py)] + argv)
    return captured


def test_sidecar_template_flag_is_off_by_default_and_reaches_both_models(rs, monkeypatch, tmp_path):
    g = _hybrid_kwargs(rs, monkeypatch, tmp_path, ["--amp", "--detailed-prompts"])["gen_kwargs"]
    assert g["prompt_style"] == "detailed"
    g = _hybrid_kwargs(rs, monkeypatch, tmp_path, ["--prompts", "template:library+tt"])["gen_kwargs"]
    assert g["prompt_style"] == "template:library+tt"
    with pytest.raises(SystemExit):
        _hybrid_kwargs(rs, monkeypatch, tmp_path, ["--prompts", "template:nope"])
    py = tmp_path / "python.exe"
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
                woosh_frames=None, level=None, amp="full")
    rs.start_woosh(types.SimpleNamespace(prompts="tuned", **base))
    rs.start_woosh(types.SimpleNamespace(prompts="template:caption", **base))
    assert "--prompts" not in cmds[0]
    assert cmds[1][-2:] == ["--prompts", "template:caption"]
