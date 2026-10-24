import ast
import json
import os
import random
import subprocess
import sys
import threading
import types

import numpy as np
import pytest

from sarge import canon, pool as prompt_pool, prompts, scene, verify, words, writer as wr
from sarge.generators.remote import RemoteGenerator
from sarge.generators.routed import RoutedGenerator
from sarge.types import CacheKey, seed_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
BENCH = os.path.join(SCRIPTS, "bench")
CAP = "template:caption+tt"
LOOP = "template:caption+tt+loop"
NAMES = list(prompts.TEMPLATES)
EVAL_FILES = (os.path.join(BENCH, "llm_eval.py"), os.path.join(BENCH, "generic_eval.py"),
              os.path.join(ROOT, "data", "_gate", "lev10", "eval", "tools", "fresh4.py"),
              os.path.join(ROOT, "data", "_gate", "push09", "eval", "tools", "fresh5.py"))


def _random_requests(n=30, seed=20261023):
    rng = random.Random(seed)
    letters = "bcdfghjklmnpqrstvwxz"
    out = []
    for i in range(n):
        ev = ["".join(rng.choice(letters) for _ in range(rng.randint(3, 8))) + "o" for _ in range(rng.randint(1, 3))]
        mat = ["".join(rng.choice(letters) for _ in range(rng.randint(3, 7))) + "u" for _ in range(rng.randint(0, 2))]
        form = i % 3
        if form == 0:
            out.append(("_".join(ev), " ".join(mat)))
        elif form == 1:
            out.append(("".join(w.capitalize() for w in ev) + "_%02d" % i, "_".join(mat).upper()))
        else:
            out.append(("SFX_" + "_".join(w.capitalize() for w in ev), "-".join(mat)))
    return out


def _traced(fn, *args, files=None, **kw):
    files = files or {os.path.abspath(prompts.__file__), os.path.abspath(words.__file__),
                      os.path.abspath(canon.__file__), os.path.abspath(verify.__file__)}
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
        out = fn(*args, **kw)
    finally:
        sys.settrace(old)
    return out, frozenset(seen)


def _eval_names():
    out = set()
    for path in EVAL_FILES:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                kw = {k.arg: k.value.value for k in node.keywords
                      if isinstance(k.value, ast.Constant) and isinstance(k.value.value, str)}
                for f in ("event_type", "material"):
                    if kw.get(f):
                        out.add(words.phrase(kw[f]))
            if isinstance(node, ast.Tuple) and len(node.elts) == 2 and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
                out.update(words.phrase(e.value) for e in node.elts if e.value)
    scenes = os.path.join(ROOT, "data", "_gate", "llm", "scenes40.json")
    if os.path.isfile(scenes):
        with open(scenes, encoding="utf-8") as fh:
            for r in json.load(fh):
                out.update(words.phrase(r[f]) for f in ("event_type", "material") if r.get(f))
    return {x for x in out if x}


def _head_module(rel, name):
    r = subprocess.run(["git", "show", "HEAD:" + rel], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        pytest.skip("git show HEAD:%s failed" % rel)
    mod = types.ModuleType(name)
    mod.__package__ = "sarge"
    sys.modules[name] = mod
    exec(compile(r.stdout, rel, "exec"), mod.__dict__)
    return mod


def _canon_raw(name, material="", cont="no", source="a thing happens"):
    return json.dumps({"source": source, "name": name, "material": material, "continuous": cont})


def test_flags_off_templates_and_request_text_match_head_for_every_template():
    head = _head_module("src/sarge/prompts.py", "sarge._head_prompts")
    reqs = _random_requests() + [("rain_loop", "tin"), ("typing_loop", "plastic"), ("Footstep", "Snow_Deep"),
                                 ("", "metal"), ("footstep", ""), ("A__B", "--")]
    for name in NAMES:
        for sfx in ("", "+tt", "-tt"):
            style = prompts.TEMPLATE + name + sfx
            for ev, mat in reqs:
                for inten, ctx in (("normal", "default"), ("hard", "indoor_small")):
                    for model in ("sa3", "woosh"):
                        want = head.template(style, ev, mat, inten, ctx, "", model)
                        assert prompts.template(style, ev, mat, inten, ctx, "", model) == want
                        k = CacheKey(ev, mat, inten, ctx, "", "", "src=world;act=loop;room=small")
                        assert prompts.compose(style, k, model=model) == head.compose(style, k, model=model)
    for ev, mat in reqs:
        k = CacheKey(ev, mat, "normal", "default")
        t = head.template("template:caption", ev, mat, "normal", "default", "", "woosh")
        old = t.replace("A single ", "", 1) if t.startswith("A single ") else t
        old = old.replace(", recorded up close", "").strip().rstrip(".").strip()
        assert verify.key_text(k) == old
    assert prompts.STYLES == head.STYLES and prompts.TEMPLATES == head.TEMPLATES


def test_loop_modifier_parses_and_is_refused_on_unknown_templates():
    assert prompts.parse_template(LOOP) == ("caption", True)
    assert prompts.parse_template("template:caption+loop") == ("caption", True)
    assert prompts.parse_template("template:caption-tt+loop") == ("caption", False)
    assert prompts.template_mods(LOOP) == {"loop"} and prompts.template_mods(CAP) == frozenset()
    assert prompts.check_style(LOOP) == LOOP
    for bad in ("template:+loop", "template:nope+loop", "template:caption+loopy"):
        with pytest.raises(Exception):
            prompts.check_style(bad)
    assert LOOP not in prompts.STYLES


def test_loop_wording_follows_the_act_and_the_loop_words_only():
    t = prompts.template
    assert t(LOOP, "rain_loop", "tin") == "A continuous tin rain sound, recorded up close."
    assert t(CAP, "rain_loop", "tin") == "A single tin rain loop sound, recorded up close."
    assert t(LOOP, "rain", "tin", scene="src=world;obj=roof;act=loop;room=small") == \
        "A continuous tin rain sound, recorded up close."
    assert t(LOOP, "rain", "tin") == t(CAP, "rain", "tin")
    assert t(LOOP, "rain", "tin", canon={"name": "rain", "material": "", "continuous": True}) == \
        "A continuous tin rain sound, recorded up close."
    assert t("template:guide+loop", "amb_wind", "", model="sa3") == \
        "TrackType: SFX, Wind, a continuous steady sound, close perspective, outdoors"
    assert t("template:oneshot+loop", "engine_loop", "") == "Engine continuous, steady, close-mic"
    assert t("template:tags+loop", "hum_loop", "", model="sa3").startswith("TrackType: SFX, hum, continuous,")
    assert prompts.LOOP_WORDS == {"loop", "loops", "looped", "looping", "ambience", "ambiences", "ambient", "amb"}
    for ev, mat in _random_requests():
        for name in NAMES:
            a = t(prompts.TEMPLATE + name + "+loop", ev, mat)
            assert a == t(prompts.TEMPLATE + name, ev, mat)
            b = t(prompts.TEMPLATE + name + "+loop", ev + "_loop", mat)
            assert "single" not in words.split_words(b) and "loop" not in words.split_words(b), b


def test_loop_wording_runs_one_code_path_for_any_name():
    paths = set()
    for ev, mat in _random_requests():
        paths.add(_traced(prompts.template, LOOP, ev, mat, "normal", "default", "", "woosh")[1])
    assert len(paths) == 1
    marked = set()
    for ev, mat in _random_requests(10, seed=5):
        marked.add(_traced(prompts.template, LOOP, ev + "_loop", mat, "normal", "default", "", "woosh")[1])
        marked.add(_traced(prompts.template, LOOP, ev, mat, "normal", "default", "", "woosh",
                           "src=npc;act=loop")[1])
    assert len(marked) == 1


def test_canon_words_replace_the_event_and_keep_the_typed_material():
    t = prompts.template
    c = {"name": "dog bark", "material": "", "continuous": False}
    assert t(CAP, "ruff", "", canon=c) == "A single dog bark sound, recorded up close."
    assert t(CAP, "stomp", "stone", "hard", canon={"name": "footstep", "material": "", "continuous": False}) == \
        "A single heavy stone footstep sound, recorded up close."
    assert t(CAP, "Footstep_Stomp_X", "", canon={"name": "footstep", "material": "zzgrit", "continuous": False}) == \
        "A single zzgrit footstep sound, recorded up close."
    assert t(CAP, "ruff", "", model="sa3", canon=c) == "TrackType: SFX, A single dog bark sound, recorded up close."
    k = CacheKey("ruff", "", "normal")
    assert verify.key_text(k, c) == "dog bark sound" and verify.event_text(k, c) == "dog bark sound"
    k2 = CacheKey("lawn_step", "grass", "normal")
    c2 = {"name": "footstep", "material": "grass", "continuous": False}
    assert verify.key_text(k2, c2) == "grass footstep sound" and verify.event_text(k2, c2) == "footstep sound"
    assert verify.event_text(k2) == "lawn step sound" and verify.key_text(k2) == "grass lawn step sound"


def test_canon_prompt_names_no_evaluation_sound_and_no_judged_category_example():
    names = _eval_names()
    assert len(names) > 150
    texts = [u for u, _s in canon.EXAMPLES] + [s["name"] for _u, s in canon.EXAMPLES] + \
            [s["source"] for _u, s in canon.EXAMPLES]
    for u, s in canon.EXAMPLES:
        ev = wr._field_value(u, "event")
        assert words.phrase(ev) not in names and words.phrase(s["name"]) not in names, (u, s)
    judged = {"footstep", "footsteps", "step", "gunshot", "gun", "pistol", "rifle", "shotgun", "punch", "glass",
              "shatter", "door", "splash", "dog", "bark", "horn", "sword", "explosion", "click", "water", "slam"}
    used = set(words.split_words(" ".join(texts)))
    assert not used & judged, used & judged
    sys_words = set(words.split_words(canon.SYSTEM))
    assert not sys_words & (judged - {"water"}), sys_words & judged


def _fake_key(i, ev, mat):
    return CacheKey(ev, mat, "normal", "default", "", "", "src=npc;who=zz%do;act=use;room=open" % i if i % 2 else "")


def test_canon_judge_is_deterministic_and_validates_generic_rules():
    k = CacheKey("ruff", "", "normal", "default", "", "", "src=npc;who=dog retriever;act=use;room=open")
    text = wr.key_text(k)
    st = canon.style()
    a = canon.judge(_canon_raw("dog bark"), k, text, st)
    b = canon.judge(_canon_raw("dog bark"), k, text, st)
    assert a == b and not a.get("failed")
    assert a["canon"] == {"name": "dog bark", "material": "", "continuous": False}
    assert a["woosh"] == "A single dog bark sound, recorded up close."
    assert a["sa3"] == "TrackType: SFX, A single dog bark sound, recorded up close."
    assert a["texts"] == {"request": "dog bark sound", "event": "dog bark sound"}
    made_up = canon.judge(_canon_raw("dog bark", "granite"), k, text, st)
    assert made_up["canon"]["material"] == "" and made_up["flags"] == ["material:unsourced"]
    for bad in ("", "dog bark at night in the yard", "red dog bark", "player bark", "dog bark 2"):
        assert canon.judge(_canon_raw(bad), k, text, st).get("failed"), bad
    assert canon.judge("not json", k, text, st)["flags"] == ["format"]
    rec = canon.judge(_canon_raw("rain", "tin", "yes"), CacheKey("rain_loop", "tin", "normal"),
                      wr.key_text(CacheKey("rain_loop", "tin", "normal")), canon.style(LOOP))
    assert rec["woosh"] == "A continuous tin rain sound, recorded up close."


def test_canon_judge_runs_one_code_path_for_random_names():
    st = canon.style()
    rng = random.Random(11)
    paths, outs = set(), set()
    for i, (ev, mat) in enumerate(_random_requests()):
        k = _fake_key(i, ev, mat)
        name = "".join(rng.choice("bcdfghjklmnpqrstvwxz") for _ in range(6)) + "o hum"
        raw = _canon_raw(name, words.phrase(mat).split(" ")[0] if mat else "")
        rec, lines = _traced(canon.judge, raw, k, wr.key_text(k), st)
        paths.add(lines)
        outs.add(rec["woosh"])
        assert not rec.get("failed")
    assert len(paths) <= 2 and len(outs) == 30


class FakeLLM:
    name = "llama"
    model = "fake.gguf"
    seen_affinity = None

    def __init__(self, answer):
        self.answer = answer
        self.calls = []
        self.started = 0
        self.closed = 0

    def ident(self):
        return {"backend": "llama", "model": "sha256:fake", "runtime": "fake 1", "threads": 4, "grammar": wr.GBNF}

    def start(self):
        self.started += 1

    def close(self):
        self.closed += 1

    def pause(self):
        pass

    def resume(self):
        pass

    def complete(self, msgs, fmt=None):
        self.calls.append((msgs, fmt))
        return self.answer(msgs)


def test_canon_writer_sends_its_own_grammar_and_stores_the_canonical_prompt(tmp_path):
    llm = FakeLLM(lambda msgs: _canon_raw("dog bark"))
    w = wr.build("llama", style="canon", template=CAP, store=str(tmp_path), backend=llm)
    w.setup()
    k = CacheKey("SFX_Ruff_02", "", "normal")
    rec = w.write(k)
    msgs, fmt = llm.calls[-1]
    assert fmt is canon.FMT and fmt["gbnf"] == canon.GBNF and msgs[0]["content"] == canon.SYSTEM
    assert msgs[-1]["content"] == "event: sfx ruff 02"
    assert rec["woosh"] == "A single dog bark sound, recorded up close." and wr.ok_record(rec)
    assert rec["style"] == "canon" and rec["texts"]["request"] == "dog bark sound"
    v1 = wr.build("llama", style="v1", store=str(tmp_path), backend=FakeLLM(lambda m: "{}"))
    v1.setup()
    assert w.id != v1.id and wr.writer_id(llm.ident(), wr.style_of("v1")) == v1.id
    assert w.id == wr.writer_id(llm.ident(), canon.style(CAP))
    assert w.id != wr.writer_id(llm.ident(), canon.style(LOOP))
    assert "canon" not in wr.STYLES


def test_writer_backends_keep_the_writer_grammar_without_a_format():
    sent = []
    b = wr.LlamaBackend.__new__(wr.LlamaBackend)
    b.url, b.timeout = "http://x", 1.0
    orig = wr._post
    try:
        wr._post = lambda url, body, timeout: sent.append(body) or {"choices": [{"message": {"content": "{}"}}]}
        b.complete([{"role": "user", "content": "x"}])
        b.complete([{"role": "user", "content": "x"}], canon.FMT)
    finally:
        wr._post = orig
    assert sent[0]["grammar"] == wr.GBNF and sent[0]["max_tokens"] == wr.MAX_TOKENS
    assert sent[1]["grammar"] == canon.GBNF and sent[1]["max_tokens"] == canon.MAX_TOKENS


class FakeWriter:
    def __init__(self, style, recs, ground=None, readonly=False):
        self.style = style
        self.recs = dict(recs)
        self.ground = ground
        self.store = dict(recs)
        self.ready = threading.Event()
        self.ready.set()
        self.readonly = readonly
        self.state = "writing"
        self.wanted = []
        self.stats = {}
        self.backend = None

    def want(self, key, why="seen", eta=None):
        self.wanted.append((str(key), why))

    def lookup(self, key):
        rec = self.recs.get(str(key))
        return rec if wr.ok_record(rec) else None

    def describe(self):
        return self.style


def _rec(woosh, texts=None):
    out = {"woosh": woosh, "sa3": "TrackType: SFX, " + woosh}
    if texts:
        out["texts"] = texts
    return out


def test_prompt_pool_waits_for_every_source_and_dedupes():
    k = CacheKey("zorbo", "", "normal")
    a = FakeWriter("canon", {str(k): _rec("A single zz hum sound, recorded up close.", {"request": "zz hum sound"})})
    b = FakeWriter("named", {})
    p = prompt_pool.PromptPool([a, b], primary=a)
    assert not p.ready(k)
    p.want(k, "seen")
    assert b.wanted == [(str(k), "seen")] and a.wanted == []
    b.store[str(k)] = {"failed": True}
    assert p.ready(k) and p.prompts(k) == ["A single zz hum sound, recorded up close."]
    b.recs[str(k)] = _rec("zz hum, steady whirr")
    b.store[str(k)] = b.recs[str(k)]
    assert p.prompts(k) == ["A single zz hum sound, recorded up close.", "zz hum, steady whirr"]
    assert p.texts(k) == {"request": "zz hum sound"}
    c = FakeWriter("named", {}, readonly=True)
    assert prompt_pool.PromptPool([c]).ready(k)
    for bad in (["nope"], []):
        with pytest.raises(ValueError):
            prompt_pool.build(bad)


def test_pool_build_shares_one_llama_and_reuses_the_primary(tmp_path, monkeypatch):
    made = []

    class Llama(FakeLLM):
        def __init__(self, *a, **kw):
            super().__init__(lambda m: _canon_raw("zz hum"))
            made.append(self)

    monkeypatch.setattr(wr, "LlamaBackend", Llama)
    primary = wr.build("llama", style="canon", template=CAP, store=str(tmp_path), backend=Llama())
    p = prompt_pool.build(["canon", "named"], primary=primary, store=str(tmp_path), template=CAP)
    assert p.writers[0] is primary and len(p.owned) == 1 and p.owned[0].style == "named"
    assert isinstance(primary.backend, prompt_pool.SharedBackend) and p.owned[0].backend is primary.backend
    assert len(made) == 1
    primary.backend.start()
    p.owned[0].backend.start()
    primary.backend.close()
    assert made[0].closed == 0
    p.owned[0].backend.close()
    assert made[0].closed == 1


def test_server_sends_the_pool_and_canonical_texts_only_when_they_exist():
    from sarge.sidecar import server as srv
    k = CacheKey("zorbo", "", "normal")
    s = types.SimpleNamespace(engine=types.SimpleNamespace(gen=types.SimpleNamespace(prompts={})), writer=None,
                              writer_owned=None, prompt_pool=None)
    assert srv._upgrade_extra(s, k) == {} and not srv._pool_waits(s, k, {})
    a = FakeWriter("canon", {str(k): _rec("A single zz hum sound, recorded up close.", {"request": "zz hum sound",
                                                                                          "event": "zz hum sound"})})
    s.writer, s.writer_owned = a, {str(k)}
    assert srv._upgrade_extra(s, k) == {"texts": {"request": "zz hum sound", "event": "zz hum sound"}}
    b = FakeWriter("named", {})
    s.prompt_pool = prompt_pool.PromptPool([a, b], primary=a)
    assert srv._pool_waits(s, k, {})
    b.recs[str(k)] = b.store[str(k)] = _rec("zz hum, steady whirr")
    assert not srv._pool_waits(s, k, {})
    got = srv._upgrade_extra(s, k)
    assert got["pool"] == ["A single zz hum sound, recorded up close.", "zz hum, steady whirr"]
    s.writer_owned = set()
    assert srv._upgrade_extra(s, k, {str(k): "a game prompt"}) == {}
    assert not srv._pool_waits(s, k, {str(k): "a game prompt"})


class Sock:
    def __init__(self):
        self.sent = []


def test_remote_request_carries_scene_pool_and_texts_only_when_asked(monkeypatch):
    r = RemoteGenerator()
    got = []
    monkeypatch.setattr(r, "_ensure", lambda: None)
    monkeypatch.setattr(r, "_roundtrip", lambda req: got.append(req) or ({"sample_rate": 48000}, b""))
    k = CacheKey("rain_loop", "tin", "normal", "default", "", "", "src=world;act=loop")
    r.generate(k, variant=3)
    assert got[-1] == {"type": "gen", "event_type": "rain_loop", "material": "tin", "intensity": "normal",
                       "context": "default", "variant": 3}
    r.generate(k, variant=3, engine="flow", pool=["p1"], texts={"request": "x"})
    assert got[-1]["pool"] == ["p1"] and got[-1]["texts"] == {"request": "x"} and "scene" not in got[-1]
    r.send_scene = True
    r.generate(k, variant=3)
    assert got[-1]["scene"] == "src=world;act=loop"


class G:
    def __init__(self, style):
        self.sample_rate = 44100
        self.prompts = {}
        self.prompt_style = style


def test_routed_sends_the_scene_only_with_the_loop_rule():
    rest = RemoteGenerator()
    assert not RoutedGenerator(guns=G(CAP), rest=rest).send_scene and not rest.send_scene
    rest2 = RemoteGenerator()
    rg = RoutedGenerator(guns=G(LOOP), rest=rest2)
    assert rg.send_scene and rest2.send_scene and rg._remote().send_scene


@pytest.fixture
def ws(monkeypatch):
    if SCRIPTS not in sys.path:
        monkeypatch.syspath_prepend(SCRIPTS)
    import woosh_server
    return woosh_server


class FakeGen:
    def __init__(self):
        self.calls = []
        self.ldm = object()

    def __call__(self, prompt, seed, **kw):
        self.calls.append((prompt, seed))
        return np.full((4800 + 480 * (len(self.calls) - 1), 1), 0.5, dtype=np.float32)


class FakeVerifier:
    def __init__(self, score):
        self.fn = score
        self.texts, self.scored = [], []
        self.path, self.ms = "fake", []

    def load(self):
        return self

    def text(self, ldm, text):
        self.texts.append(text)
        return "emb:" + text

    def score(self, y, sr, temb):
        self.scored.append(temb)
        return self.fn(len(self.scored) - 1)


def _up(ws, best, verifier, score="request"):
    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=lambda: None, commit=lambda: None,
                    best=best, verifier=verifier, score=score)
    up._load_verifier()
    up.gen = FakeGen()
    return up


STEP = CacheKey("zorbo_walk", "zzgrit", "normal")


def test_pool_first_take_picks_the_prompt_by_mean_score_then_best_of_n_on_it(ws):
    scores = [0.1, 0.2, 0.3, 0.9, 0.5, 0.5, 0.4, 0.6]
    v = FakeVerifier(lambda i: scores[i] if i < len(scores) else [0.2, 0.7, 0.1, 0.3][(i - 8) % 4])
    up = _up(ws, 2, v)
    base = ws.prompt_for(STEP, CAP)
    pool = [base, "zz prompt one", "", "zz prompt two", "zz prompt one"]
    y = up.render(STEP, 0, CAP, None, None, False, "peak", pool=pool)
    prompts_used = [p for p, _s in up.gen.calls]
    seeds = [s for _p, s in up.gen.calls]
    assert prompts_used == [base, base, "zz prompt one", "zz prompt one", "zz prompt two", "zz prompt two"]
    assert seeds == [seed_for(STEP, verify.candidate_variant(0, j)) for j in range(2)] * 3
    assert up.choice[(str(STEP), (base, "zz prompt one", "zz prompt two"))] == 1
    want = ws.finish_key(np.full((4800 + 480 * 3, 1), 0.5, dtype=np.float32), STEP, None, False)
    assert np.array_equal(y, want) and up.picked == [0, 1] and up.pool_picks == {1: 1}
    up.render(STEP, 1, CAP, None, None, False, "peak", pool=pool)
    assert [p for p, _s in up.gen.calls[6:]] == ["zz prompt one"] * 2
    assert [s for _p, s in up.gen.calls[6:]] == [seed_for(STEP, verify.candidate_variant(1, j)) for j in range(2)]
    info = up.info()
    assert info["pool_keys"] == 1 and info["pool_picks"] == {1: 1} and info["candidates"] == 8
    assert v.texts == [verify.key_text(STEP)] * 8


def test_pool_choice_breaks_ties_toward_the_keys_own_prompt_and_resumes_after_preemption(ws):
    v = FakeVerifier(lambda i: 0.5)
    up = _up(ws, 2, v)
    calls = {"n": 0}
    gen = up.gen

    def flaky(prompt, seed, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ws.Preempted()
        return gen(prompt, seed, **kw)

    flaky.ldm = gen.ldm
    up.gen = flaky
    with pytest.raises(ws.Preempted):
        up.render(STEP, 0, CAP, None, None, False, "peak", pool=["zz other"])
    assert up.preempted == 1 and len(v.scored) == 2
    up.render(STEP, 0, CAP, None, None, False, "peak", pool=["zz other"])
    assert len(gen.calls) == 4 and len(v.scored) == 4 and up.pool_picks == {0: 1}


def test_event_min_scores_the_lower_of_request_and_event_words(ws):
    v = FakeVerifier(lambda i: 0.1 * i)
    up = _up(ws, 2, v, score="event-min")
    up.render(STEP, 0, CAP, None, None, False, "peak")
    assert v.texts[:2] == [verify.key_text(STEP), verify.event_text(STEP)]
    assert v.scored[0] == ("emb:" + verify.key_text(STEP), "emb:" + verify.event_text(STEP))
    up.render(STEP, 1, CAP, None, None, False, "peak", texts={"request": "zz hum sound", "event": "hum sound"})
    assert v.texts[-2:] == ["zz hum sound", "hum sound"] and up.info()["score"] == "event-min"
    plain = _up(ws, 2, FakeVerifier(lambda i: 0.0))
    plain.render(STEP, 0, CAP, None, None, False, "peak", texts={"request": "zz hum sound", "event": "x"})
    assert plain.verifier.texts == ["zz hum sound"] * 2 and "score" not in plain.info()


def test_verifier_min_over_texts_equals_the_single_score_for_one_text(ws):
    import torch
    v = ws.Verifier()
    v.sr = 32000
    e = torch.tensor([0.6, 0.8])

    class Tower:
        def get_audio_embedding_from_data(self, m, device=None, use_tensor=True):
            return e.unsqueeze(0)

    v.tower = Tower()
    y = np.zeros((4800, 2), np.float32)
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    assert v.score(y, 48000, a) == pytest.approx(0.6) and v.score(y, 48000, (a, b)) == pytest.approx(0.6)
    assert v.score(y, 48000, (b, b)) == pytest.approx(0.8)


def test_worker_flags_default_off_and_check_their_needs(ws, monkeypatch):
    for argv in (["--upgrade-score", "event-min"], ["--upgrade", "flow", "--upgrade-score", "event-min"]):
        monkeypatch.setattr(sys, "argv", ["woosh_server.py"] + argv)
        with pytest.raises(SystemExit):
            ws.main()
    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=lambda: None, commit=lambda: None)
    assert up.score_mode == "request" and up.info().keys() == {"type", "engine", "state", "error", "renders",
                                                               "preempted", "failed", "load_s", "solver"}


def _rest_cmd(monkeypatch, argv):
    if SCRIPTS not in sys.path:
        monkeypatch.syspath_prepend(SCRIPTS)
    import run_sidecar as rs
    monkeypatch.setenv("SARGE_WOOSH_EARLY", "0")
    monkeypatch.setattr(rs.prefetch, "start_hybrid", lambda *a, **k: None)
    monkeypatch.setattr(rs, "check_weights", lambda name: None)
    got, captured = [], {}

    class Stop(Exception):
        pass

    class P:
        stdout = iter(())

        def poll(self):
            return 0

        def kill(self):
            return None

    def fake_serve(**kw):
        captured.update(kw)
        raise Stop()

    monkeypatch.setattr(rs, "serve", fake_serve)
    monkeypatch.setattr(rs.subprocess, "Popen", lambda cmd, **kw: got.append(cmd) or P())
    with pytest.raises(Stop):
        rs.main(["--generator", "hybrid", "--no-cache", "--woosh-python", sys.executable] + argv)
    with pytest.raises(SystemExit):
        captured["gen_kwargs"]["start_rest"]()
    return got[0], captured


def test_sidecar_new_flags_default_off_and_reach_the_worker_only_when_set(monkeypatch, tmp_path):
    base = ["--prompts", CAP, "--woosh-upgrade", "flow", "--woosh-upgrade-solver", "midpoint:16",
            "--woosh-upgrade-best", "4", "--woosh-upgrade-vram-save", "none"]
    cmd, cap = _rest_cmd(monkeypatch, base)
    assert "--upgrade-score" not in cmd and "prompt_pool" not in cap and "writer" not in cap
    ev, _ = _rest_cmd(monkeypatch, base + ["--woosh-upgrade-score", "event-min"])
    assert ev[len(cmd):] == ["--upgrade-score", "event-min"] and ev[:len(cmd)] == cmd
    import run_sidecar as rs
    for bad in (["--prompts", CAP, "--woosh-upgrade", "flow", "--woosh-upgrade-pool", "named"],
                ["--prompts", CAP, "--woosh-upgrade", "flow", "--woosh-upgrade-score", "event-min"],
                base + ["--woosh-upgrade-pool", "nope", "--writer-store", str(tmp_path)]):
        with pytest.raises(SystemExit):
            rs.main(["--generator", "hybrid", "--no-cache", "--woosh-python", sys.executable] + bad)
    loop, cap2 = _rest_cmd(monkeypatch, ["--prompts", LOOP])
    assert loop[-2:] == ["--prompts", LOOP] and cap2["gen_kwargs"]["prompt_style"] == LOOP
    assert "canon" in rs.WRITER_STYLES


def test_pool_flag_builds_the_pool_and_hands_it_to_the_server(monkeypatch, tmp_path):
    made = []

    class Llama(FakeLLM):
        def __init__(self, *a, **kw):
            super().__init__(lambda m: _canon_raw("zz hum"))
            made.append(self)

    monkeypatch.setattr(wr, "LlamaBackend", Llama)
    base = ["--prompts", CAP, "--woosh-upgrade", "flow", "--woosh-upgrade-best", "4", "--writer", "llama",
            "--writer-style", "canon", "--writer-store", str(tmp_path), "--woosh-upgrade-pool", "canon,named"]
    _cmd, cap = _rest_cmd(monkeypatch, base)
    p = cap["prompt_pool"]
    assert p.writers[0] is cap["writer"] and [w.style for w in p.owned] == ["named"]
    assert cap["writer"].spec["template"] == CAP and len(made) == 1


def test_lever_code_has_no_event_or_material_words():
    judged = {"footstep", "footsteps", "gunshot", "pistol", "rifle", "shotgun", "punch", "shatter", "splash", "bark",
              "horn", "sword", "explosion", "rain", "dog", "metal", "wood", "stone", "gravel", "grass", "snow", "sand",
              "water", "ruff", "pummel", "magnum", "stomp", "jab"}
    for mod in (canon, prompt_pool, verify):
        src = open(mod.__file__, encoding="utf-8").read()
        tree = ast.parse(src)
        consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        if mod is canon:
            consts = [c for c in consts if c not in (canon.SYSTEM,) and c not in sum(
                ([u, s["source"], s["name"], s["material"]] for u, s in canon.EXAMPLES), [])]
        hit = set(words.split_words(" ".join(consts))) & judged
        assert not hit, (mod.__name__, hit)
    loop_src = [n.value for n in ast.walk(ast.parse(open(prompts.__file__, encoding="utf-8").read()))
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and "continuous" in n.value]
    assert not set(words.split_words(" ".join(loop_src))) & judged


def test_canon_drops_typed_material_words_the_canonical_name_already_holds():
    t = prompts.template
    c = {"name": "zzglass break", "material": "zzglass zzbottle", "continuous": False}
    assert t(CAP, "zorbo_impact", "zzglass zzbottle", canon=c) == "A single zzbottle zzglass break sound, recorded up close."
    assert t(CAP, "zorbo_impact", "zzglass zzbottle") == "A single zzglass zzbottle zorbo impact sound, recorded up close."


def test_flow_load_waits_for_room_on_the_card_only_when_asked(ws, monkeypatch):
    monkeypatch.setattr(ws, "RETRY_S", 0.0)
    seen = iter([100.0, 400.0, 900.0, 2000.0])
    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {},
                    make=lambda: types.SimpleNamespace(capture=lambda frames: {}, should_stop=None),
                    commit=lambda: None, vram_free=800, vram=lambda: next(seen))
    assert up.start(wait=True) == "ready" and next(seen) == 2000.0
    assert up.info()["vram_free_mib"] == 800
    calls = []
    off = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {},
                     make=lambda: types.SimpleNamespace(capture=lambda frames: {}, should_stop=None),
                     commit=lambda: None, vram=lambda: calls.append(1) or 0.0)
    assert off.start(wait=True) == "ready" and not calls and "vram_free_mib" not in off.info()
    blind = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {},
                       make=lambda: types.SimpleNamespace(capture=lambda frames: {}, should_stop=None),
                       commit=lambda: None, vram_free=800, vram=lambda: None)
    assert blind.start(wait=True) == "ready"
    for argv in (["--upgrade-vram-free", "500"], ["--upgrade", "flow", "--upgrade-vram-free", "-1"]):
        monkeypatch.setattr(sys, "argv", ["woosh_server.py"] + argv)
        with pytest.raises(SystemExit):
            ws.main()


def test_sidecar_vram_flag_reaches_the_worker_only_when_set(monkeypatch):
    base = ["--prompts", CAP, "--woosh-upgrade", "flow", "--woosh-upgrade-solver", "midpoint:16",
            "--woosh-upgrade-vram-save", "none"]
    cmd, _ = _rest_cmd(monkeypatch, base)
    assert "--upgrade-vram-free" not in cmd
    got, _ = _rest_cmd(monkeypatch, base + ["--woosh-upgrade-vram-free", "1500"])
    assert got[len(cmd):] == ["--upgrade-vram-free", "1500"]
    import run_sidecar as rs
    with pytest.raises(SystemExit):
        rs.main(["--generator", "hybrid", "--no-cache", "--woosh-python", sys.executable,
                 "--woosh-upgrade-vram-free", "1500"])


def test_loop_request_text_and_canon_needs_a_model(tmp_path):
    k = CacheKey("rain_loop", "tin", "normal")
    assert verify.key_text(k, loop=True) == "tin rain sound" and verify.key_text(k) == "tin rain loop sound"
    with pytest.raises(ValueError):
        wr.build("ground", style="canon", store=str(tmp_path), ground="caption")
    with pytest.raises(ValueError):
        canon.style("tuned")


def test_only_model_backends_of_the_pool_yield_to_renders():
    from sarge.sidecar import server as srv

    class B:
        def __init__(self, name):
            self.name = name

        def pause(self):
            pass

    llm, ground, mine = B("llama"), B("ground"), B("llama")
    pool = types.SimpleNamespace(owned=[types.SimpleNamespace(backend=llm), types.SimpleNamespace(backend=ground),
                                        types.SimpleNamespace(backend=llm), types.SimpleNamespace(backend=mine)])
    assert srv._pool_backends(pool, types.SimpleNamespace(backend=mine)) == [llm]
    assert srv._pool_backends(None, None) == []
