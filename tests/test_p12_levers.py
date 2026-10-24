import json
import sys
import threading
import types

import numpy as np
import pytest

from sarge import canon, pool as prompt_pool, prompts, verify, words, writer as wr
from sarge.generators.cached import CachedGenerator
from sarge.generators.routed import RoutedGenerator, firearm_words, meaning_side, side_for
from sarge.types import AudioClip, CacheKey

from test_generic_levers import (CAP, FakeLLM, FakeVerifier, FakeWriter, ROOT, SCRIPTS, _canon_raw, _rec, _rest_cmd,
                                 _up)

CLEAN = CAP + "+clean"


@pytest.fixture
def ws(monkeypatch):
    if SCRIPTS not in sys.path:
        monkeypatch.syspath_prepend(SCRIPTS)
    import woosh_server
    return woosh_server


def _head_module(rel, name, package="sarge", ref="HEAD"):
    import subprocess
    r = subprocess.run(["git", "show", ref + ":" + rel], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        pytest.skip("git show %s:%s failed" % (ref, rel))
    mod = types.ModuleType(name)
    mod.__package__ = package
    sys.modules[name] = mod
    exec(compile(r.stdout, rel, "exec"), mod.__dict__)
    return mod


def _ask(answer):
    return json.dumps({"firearm": answer})


def _answers(name, firearm):
    return lambda msgs: _ask(firearm) if msgs[0]["content"] == canon.ASK_SYSTEM else _canon_raw(name)


def test_codes_glued_to_letters_stay_codes_and_a_glued_take_at_the_end_still_goes():
    c = words.clean_phrase
    assert c("Zk98_Zorbo_Krell_R") == "zk98 zorbo krell" and c("QX15_Zorbo") == "qx15 zorbo"
    assert c("Q4B1_Zorbo_Krell") == "q4b1 zorbo krell" and c("Zorbo_Zp5") == "zorbo zp5"
    assert c("Zorbo_Zk98") == "zorbo zk98" and c("Zorbo_Zorbk17") == "zorbo zorbk17"
    assert c("Zk98Zorbo") == "zk98 zorbo" and c("ZORB47Krell") == "zorb47 krell"
    assert c("ZorboKrell3") == "zorbo krell" and c("Zorbo_Krell02_L") == "zorbo krell"
    assert c("Krell3_Zorbo") == "krell3 zorbo" and c("Zorbo_Krell_v2") == "zorbo krell"
    assert c("SFX_Zorbo_Krell_01") == "zorbo krell" and c("SFX_01") == "sfx 01"
    assert words.is_code("zk98") and words.is_code("9mm") and not words.is_code("zorbo") and not words.is_code("98")


def test_clean_template_and_request_text_keep_the_code_whole():
    k = CacheKey("Zk98_Zorbo_Krell_R", "", "normal")
    assert prompts.template(CLEAN, k.event_type, k.material) == "A single zk98 zorbo krell sound, recorded up close."
    assert prompts.template(CLEAN, k.event_type, k.material, model="sa3") == \
        "TrackType: SFX, A single zk98 zorbo krell sound, recorded up close."
    assert verify.key_text(k, clean=True) == "zk98 zorbo krell sound"
    assert prompts.template(CLEAN, "Q4B1_Zorbo", "Zk98") == "A single zk98 q4b1 zorbo sound, recorded up close."
    assert prompts.template(CLEAN, "Zorbo_Zk98", "Zk98") == "A single zorbo zk98 sound, recorded up close."


def test_code_fix_leaves_every_prompt_without_the_clean_modifier_as_at_head():
    head = _head_module("src/sarge/prompts.py", "sarge._head_prompts_p12")
    names = ["Zk98_Zorbo_Krell_R", "QX15_Zorbo", "Q4B1_Zorbo_Krell", "Zorbo_9mm", "ZorboKrell3", "Zk98Zorbo"]
    for name in prompts.TEMPLATES:
        for sfx in ("", "+tt", "-tt", "+loop"):
            style = prompts.TEMPLATE + name + sfx
            for ev in names:
                for model in ("sa3", "woosh"):
                    assert prompts.template(style, ev, "Zk98", "hard", "", "", model) == \
                        head.template(style, ev, "Zk98", "hard", "", "", model)
    hw = _head_module("src/sarge/words.py", "sarge._head_words_p12")
    for ev in names + ["Zorbo_Krell_01", "SFXZorboKrell", "zorbo-krell.wav"]:
        assert words.split_words(ev) == hw.split_words(ev) and words.phrase(ev) == hw.phrase(ev)


def test_guarded_choice_keeps_a_pool_prompt_only_at_or_above_the_keys_own_prompt():
    assert verify.choose_guarded([0.5, 0.5]) == 1 and verify.choose_guarded([0.5, 0.4]) == 0
    assert verify.choose_guarded([0.4, 0.5]) == 1 and verify.choose_guarded([0.2, 0.6, 0.6]) == 1
    assert verify.choose_guarded([0.7]) == 0 and verify.choose([0.5, 0.5]) == 0
    assert verify.guard_texts(CAP) == {"loop": False, "clean": True}
    assert verify.guard_texts(CAP + "+loop") == {"loop": True, "clean": True}
    assert verify.guard_texts(None) == {"loop": False, "clean": True}


def test_guarded_pool_scores_every_candidate_and_take_against_the_cleaned_name(ws):
    k = CacheKey("SFX_Zorbo_Krell_01", "", "normal")
    pool = ["A single zz hum sound, recorded up close."]
    texts = {"request": "zz hum sound", "event": "zz hum sound"}
    scores = [0.2, 0.3, 0.3, 0.2] + [0.1] * 32
    v = FakeVerifier(lambda i: scores[i] if i < len(scores) else 0.1)
    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=lambda: None, commit=lambda: None,
                    best=2, verifier=v, guard=True)
    up._load_verifier()
    from test_generic_levers import FakeGen
    up.gen = FakeGen()
    base = ws.prompt_for(k, CAP)
    up.render(k, 0, CAP, None, None, False, "peak", pool=pool, texts=texts)
    assert v.texts == ["zorbo krell sound"] * 4
    assert up.choice[(str(k), (base, pool[0]))] == 1 and up.info()["pool_guard"] is True
    up.render(k, 1, CAP, None, None, False, "peak", pool=pool, texts=texts)
    assert v.texts[4:] == ["zorbo krell sound"] * 2 and [p for p, _s in up.gen.calls[4:]] == pool * 2
    lower = FakeVerifier(lambda i: [0.3, 0.3, 0.29, 0.29][i] if i < 4 else 0.1)
    up2 = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=lambda: None, commit=lambda: None,
                     best=2, verifier=lower, guard=True)
    up2._load_verifier()
    up2.gen = FakeGen()
    up2.render(k, 0, CAP, None, None, False, "peak", pool=pool, texts=texts)
    assert up2.choice[(str(k), (base, pool[0]))] == 0
    plain = _up(ws, 2, FakeVerifier(lambda i: 0.1))
    assert "pool_guard" not in plain.info()
    plain.render(k, 0, CAP, None, None, False, "peak", pool=pool, texts=texts)
    assert plain.verifier.texts == ["zz hum sound"] * 4


def test_guard_flag_needs_best_of_two_on_the_worker_and_a_pool_on_the_sidecar(ws, monkeypatch, tmp_path):
    for argv in (["--upgrade", "flow", "--upgrade-pool-guard"],
                 ["--upgrade", "flow", "--upgrade-best", "1", "--upgrade-pool-guard"]):
        monkeypatch.setattr(sys, "argv", ["woosh_server.py"] + argv)
        with pytest.raises(SystemExit):
            ws.main()

    class Llama(FakeLLM):
        def __init__(self, *a, **kw):
            super().__init__(lambda m: _canon_raw("zz hum"))

    monkeypatch.setattr(wr, "LlamaBackend", Llama)
    base = ["--prompts", CLEAN, "--woosh-upgrade", "flow", "--woosh-upgrade-best", "4", "--woosh-upgrade-pool",
            "canon", "--writer-store", str(tmp_path)]
    off, _ = _rest_cmd(monkeypatch, base)
    assert "--upgrade-pool-guard" not in off
    on, _ = _rest_cmd(monkeypatch, base + ["--woosh-upgrade-pool-guard"])
    assert on[len(off):] == ["--upgrade-pool-guard"] and on[:len(off)] == off
    import run_sidecar as rs
    with pytest.raises(SystemExit):
        rs.main(["--generator", "hybrid", "--no-cache", "--woosh-python", sys.executable, "--woosh-upgrade", "flow",
                 "--woosh-upgrade-best", "4", "--woosh-upgrade-pool-guard"])


def test_firearm_question_is_a_second_call_and_changes_the_writer_id_only_when_asked():
    st = canon.style(CAP, True)
    assert canon.style(CAP)["fmt"] is canon.FMT and st["fmt"] is canon.FMT and st["ask"] is canon.ASK
    assert "ask" not in canon.style(CAP) and canon.asks_firearm(st) and not canon.asks_firearm(canon.style(CAP))
    assert canon.ident(CAP) == canon.ident(CAP, False) and "firearm" not in canon.ident(CAP)
    assert canon.ident(CAP, True)["firearm"]["question"] == canon.FIREARM
    llm = FakeLLM(lambda m: _canon_raw("zz hum"))
    assert wr.writer_id(llm.ident(), canon.style(CAP)) != wr.writer_id(llm.ident(), st)
    k = CacheKey("Zorbo_Krell", "", "normal")
    raw = _canon_raw("zz bang")
    yes = canon.judge(raw, k, wr.key_text(k), st, _ask("yes"))
    no = canon.judge(raw, k, wr.key_text(k), st, _ask("no"))
    off = canon.judge(raw, k, wr.key_text(k), canon.style(CAP))
    assert yes["canon"]["firearm"] is True and no["canon"]["firearm"] is False and yes["raw_firearm"] == _ask("yes")
    assert canon.firearm_of(yes) is True and canon.firearm_of(no) is False
    assert "firearm" not in off["canon"] and canon.firearm_of(off) is None and "raw_firearm" not in off
    for k2 in ("woosh", "sa3", "texts", "slots", "raw"):
        assert yes[k2] == no[k2] == off[k2]
    bad = canon.judge(raw, k, wr.key_text(k), st, "not json")
    assert not bad.get("failed") and "firearm:value" in bad["flags"] and canon.firearm_of(bad) is None
    assert bad["woosh"] == off["woosh"]
    assert canon.parse_ask(_ask("maybe")) == "" and canon.parse_ask(None) == ""
    assert canon.firearm_of({"failed": True, "canon": {"firearm": True}}) is None and canon.firearm_of(None) is None
    assert canon.describe(yes).endswith("(firearm)") and not canon.describe(no).endswith("(firearm)")


def test_canon_writer_asks_the_firearm_question_on_the_same_facts(tmp_path):
    llm = FakeLLM(_answers("zz bang", "yes"))
    w = wr.build("llama", style="canon", template=CAP, store=str(tmp_path), backend=llm, firearm=True)
    w.setup()
    rec = w.write(CacheKey("QX15_Zorbo", "", "normal"))
    (m1, f1), (m2, f2) = llm.calls[-2:]
    assert f1 is canon.FMT and m1[0]["content"] == canon.SYSTEM and f1["slot"] != f2["slot"]
    assert f2 is canon.ASK_FMT and m2[0]["content"] == canon.ASK_SYSTEM and m2[-1]["content"] == m1[-1]["content"]
    assert m1[-1]["content"] == "event: qx15 zorbo" and rec["canon"]["firearm"] is True
    assert len(m2) == 2 + 2 * len(canon.ASK_EXAMPLES)
    plain_llm = FakeLLM(_answers("zz bang", "yes"))
    plain = wr.build("llama", style="canon", template=CAP, store=str(tmp_path), backend=plain_llm)
    plain.setup()
    prec = plain.write(CacheKey("QX15_Zorbo", "", "normal"))
    assert plain.id != w.id and len(plain_llm.calls) == 1 and "firearm" not in prec["canon"]
    assert prec["woosh"] == rec["woosh"]


def test_firearm_question_names_no_evaluation_sound():
    import test_generic_levers as tg
    names = tg._eval_names()
    assert len(names) > 150
    for u, _a in canon.ASK_EXAMPLES:
        assert words.phrase(wr._field_value(u, "event")) not in names, u
    judged = {"footstep", "footsteps", "step", "gunshot", "gun", "pistol", "rifle", "shotgun", "punch", "glass",
              "shatter", "door", "splash", "dog", "bark", "horn", "sword", "explosion", "click", "water", "slam",
              "fire", "shot", "shoot", "reload", "magnum", "revolver", "carbine"}
    used = set(words.split_words(" ".join(u for u, _a in canon.ASK_EXAMPLES)))
    assert not used & judged, used & judged
    assert set(words.split_words(canon.ASK_SYSTEM)) & judged == {"gun"}


def test_llama_backend_keeps_one_slot_unless_the_firearm_question_needs_a_second(tmp_path, monkeypatch):
    monkeypatch.setattr(wr, "resolve_model", lambda m, dirs=None: ("m.gguf", "sha256:m"))
    monkeypatch.setattr(wr, "find_binary", lambda b=None: "llama-server.exe")
    one = wr.LlamaBackend()
    two = wr.LlamaBackend(slots=2)
    c1, c2 = one.command(), two.command()
    assert c1[c1.index("-np") + 1] == "1" and c1[c1.index("-c") + 1] == str(wr.CTX)
    assert c2[c2.index("-np") + 1] == "2" and c2[c2.index("-c") + 1] == str(2 * wr.CTX)
    one._runtime = two._runtime = "fake"
    assert "slots" not in one.ident() and two.ident()["slots"] == 2
    sent = []
    monkeypatch.setattr(wr, "_post", lambda url, body, timeout: sent.append(body) or
                        {"choices": [{"message": {"content": "{}"}}]})
    for b in (one, two):
        b.url, b.timeout = "http://x", 1.0
        b.complete([{"role": "user", "content": "x"}], canon.FMT)
        b.complete([{"role": "user", "content": "x"}], canon.ASK_FMT)
        b.complete([{"role": "user", "content": "x"}])
    assert ["id_slot" in x for x in sent[:3]] == [False] * 3
    assert [x.get("id_slot") for x in sent[3:]] == [0, 1, None]
    monkeypatch.setattr(wr, "LlamaBackend", lambda *a, **kw: types.SimpleNamespace(slots=kw.get("slots"), name="llama"))
    assert wr.build("llama", style="canon", template=CAP, store=str(tmp_path), firearm=True).backend.slots == 2
    assert wr.build("llama", style="canon", template=CAP, store=str(tmp_path)).backend.slots == 1
    p = prompt_pool.build(["canon"], store=str(tmp_path), template=CAP, firearm=True, route_sources=("canon",))
    assert p.writers[0].backend.inner.slots == 2
    q = prompt_pool.build(["canon"], store=str(tmp_path), template=CAP)
    assert q.writers[0].backend.inner.slots == 1


def test_meaning_route_takes_the_writer_answer_else_whole_words_of_the_cleaned_name():
    shot = CacheKey("SFX_Zorbo_Shot_02", "", "normal")
    clicker = CacheKey("Zorbo_Clicker", "", "normal")
    camp = CacheKey("Campfire_Zorbo", "", "normal")
    gun = CacheKey("Zorbo", "Zk98_Rifle", "normal")
    assert side_for(shot, "families") == "guns" and side_for(shot, "meaning") == "guns"
    assert side_for(clicker, "families") == "guns" and side_for(clicker, "meaning") == "rest"
    assert side_for(camp, "families") == "guns" and side_for(camp, "meaning") == "rest"
    assert side_for(gun, "meaning") == "guns" and firearm_words(gun) and not firearm_words(camp)
    assert side_for(shot, "meaning", meaning=lambda k: False) == "rest"
    assert side_for(clicker, "meaning", meaning=lambda k: True) == "guns"
    assert side_for(shot, "meaning", meaning=lambda k: None) == "guns"

    def broken(k):
        raise RuntimeError("store gone")

    assert meaning_side(shot, broken) == "guns" and meaning_side(camp, broken) == "rest"


class Fake:
    def __init__(self, tag):
        self.tag = tag
        self.calls = []
        self.sample_rate = 44100

    def generate(self, key, *, variant=0):
        self.calls.append(str(key))
        return AudioClip(np.full(10, 0.1 if self.tag == "guns" else 0.2, dtype=np.float32), 44100, variant_id=variant)

    def generate_batch(self, items):
        return [self.generate(k, variant=v) for k, v in items]


def test_routed_generator_records_sides_and_reports_keys_whose_answer_moved_them():
    answers = {}
    g = RoutedGenerator(guns=Fake("guns"), rest=Fake("rest"), warm_guns=False, route="meaning")
    g.meaning = lambda k: answers.get(str(k))
    a, b = CacheKey("Zorbo_Bang", "", "normal"), CacheKey("Zorbo_Shot", "", "normal")
    g.generate(a)
    g.generate_batch([(b, 0), (b, 1)])
    assert g.made == {a: "rest", b: "guns"} and g.flips() == [] and g.route_tag(a) == "side:rest"
    answers[str(a)] = True
    answers[str(b)] = True
    assert g.flips() == [a]
    assert g.settle(a) == "guns" and g.flips() == [] and g.fast_stats()["routed_flips"] == 1
    answers[str(b)] = False
    assert g.flips() == [b] and g.route_tag(b) == "side:rest"
    fam = RoutedGenerator(guns=Fake("guns"), rest=Fake("rest"), warm_guns=False)
    fam.generate(a)
    assert fam.made == {} and fam.flips() == [] and fam.route_tag(a) == "" and "routed_flips" not in fam.fast_stats()
    assert "route:meaning" in g.post_tag


def test_disk_cache_names_the_side_only_under_the_meaning_route(tmp_path):
    answers = {}
    g = RoutedGenerator(guns=Fake("guns"), rest=Fake("rest"), warm_guns=False, route="meaning")
    g.meaning = lambda k: answers.get(str(k))
    c = CachedGenerator(g, str(tmp_path / "m"))
    k = CacheKey("Zorbo_Bang", "", "normal")
    rest_path = c._path(k, 0)
    answers[str(k)] = True
    assert c._path(k, 0) != rest_path
    fam = RoutedGenerator(guns=Fake("guns"), rest=Fake("rest"), warm_guns=False)
    cf = CachedGenerator(fam, str(tmp_path / "f"))
    head = _head_module("src/sarge/generators/cached.py", "sarge.generators._head_cached_p12", "sarge.generators")
    hf = head.CachedGenerator(fam, str(tmp_path / "f"))
    for key in (k, CacheKey("shoot", "pistol", "normal"), CacheKey("footstep", "stone", "loud", "tunnel")):
        for v in (0, 3):
            assert cf._path(key, v) == hf._path(key, v)


class Eng:
    def __init__(self, gen):
        self.gen = gen
        self.cache, self.recent, self.claims, self.inflight, self.stats = {}, {}, set(), set(), {}


def _server(gen, upgrade=None):
    from sarge.sidecar import server as srv
    s = types.SimpleNamespace(engine=Eng(gen), upgrade=upgrade, prompt_rev={}, event_log=None, writer=None,
                              upgrade_done={}, upgrade_stage={})
    srv._upgrade_init(s)
    return s


def test_a_moved_key_drops_its_takes_or_keeps_them_for_the_flow_upgrade():
    from sarge.sidecar import server as srv
    answers = {}
    g = RoutedGenerator(guns=Fake("guns"), rest=Fake("rest"), warm_guns=False, route="meaning")
    g.meaning = lambda k: answers.get(str(k))
    a, b = CacheKey("Zorbo_Bang", "", "normal"), CacheKey("Zorbo_Shot", "", "normal")
    s = _server(g)
    s.engine.cache[a] = [g.generate(a)]
    s.engine.cache[b] = [g.generate(b)]
    assert not srv._route_step(s) and g.made == {a: "rest", b: "guns"}
    answers[str(a)] = True
    s.engine.inflight.add((a, 1))
    assert not srv._route_step(s) and a in s.engine.cache
    s.engine.inflight.clear()
    assert srv._route_step(s) and a not in s.engine.cache and s.prompt_rev[a] == 1
    assert s.engine.stats["route_flips"] == 1 and not srv._route_step(s) and g.made[a] == "guns"
    g.generate(a)
    assert g.guns.calls[-1] == str(a)
    answers[str(b)] = False
    up = _server(g, upgrade="flow")
    held = [AudioClip(np.zeros(4, np.float32), 44100)]
    up.engine.cache[b] = held
    assert srv._route_step(up) and up.engine.cache[b] is held and b not in up.prompt_rev and g.made[b] == "rest"
    g.made[b] = "guns"
    s2 = _server(g)
    s2.engine.cache[b] = [AudioClip(np.zeros(4, np.float32), 44100)]
    assert srv._route_step(s2) and b not in s2.engine.cache and s2.prompt_rev[b] == 1


def test_meaning_sources_are_the_writers_that_ask_the_firearm_question():
    from sarge.sidecar import server as srv
    k = CacheKey("Zorbo_Bang", "", "normal")
    ask = FakeWriter("canon", {str(k): dict(_rec("A single zz bang sound, recorded up close."),
                                            canon={"name": "zz bang", "material": "", "continuous": False,
                                                   "firearm": True})})
    ask.spec = canon.style(CAP, True)
    plain = FakeWriter("canon", {})
    plain.spec = canon.style(CAP)
    s = types.SimpleNamespace(writer=plain, prompt_pool=prompt_pool.PromptPool([ask], route=[ask]))
    got = srv._meaning_sources(s)
    assert got == [ask]
    assert srv._meaning_lookup(got)(k) is True and srv._meaning_lookup(got)(CacheKey("x", "", "normal")) is None


def test_route_only_writers_feed_the_route_and_never_the_prompt_pool():
    k = CacheKey("zorbo", "", "normal")
    a = FakeWriter("canon", {str(k): _rec("A single zz hum sound, recorded up close.", {"request": "zz hum sound"})})
    p = prompt_pool.PromptPool([a], route=[a])
    assert p.prompts(k) == [] and p.texts(k) is None and p.ready(k) and p.route_only(a)
    assert "(route only)" in p.describe()
    b = FakeWriter("named", {str(k): _rec("zz hum, steady whirr")})
    q = prompt_pool.PromptPool([a, b], route=[a])
    assert q.prompts(k) == ["zz hum, steady whirr"] and q.offered() == [b]


def test_sidecar_meaning_route_builds_one_firearm_canon_writer(monkeypatch, tmp_path):
    made = []

    class Llama(FakeLLM):
        def __init__(self, *a, **kw):
            super().__init__(_answers("zz bang", "yes"))
            made.append(self)

    monkeypatch.setattr(wr, "LlamaBackend", Llama)
    _cmd, cap = _rest_cmd(monkeypatch, ["--prompts", CAP, "--route", "meaning", "--writer-store", str(tmp_path / "a")])
    p = cap["prompt_pool"]
    assert cap["gen_kwargs"]["route"] == "meaning" and "writer" not in cap and len(made) == 1
    assert [w.style for w in p.writers] == ["canon"] and p.route_only(p.writers[0])
    assert canon.asks_firearm(p.writers[0].spec)
    made.clear()
    _cmd, cap = _rest_cmd(monkeypatch, ["--prompts", CAP, "--woosh-upgrade", "flow", "--woosh-upgrade-best", "4",
                                        "--woosh-upgrade-pool", "canon", "--route", "meaning", "--writer-store",
                                        str(tmp_path / "b")])
    p = cap["prompt_pool"]
    assert len(p.writers) == 1 and not p.route_only(p.writers[0]) and canon.asks_firearm(p.writers[0].spec)
    made.clear()
    _cmd, cap = _rest_cmd(monkeypatch, ["--prompts", CAP, "--writer", "llama", "--writer-style", "canon", "--route",
                                        "meaning", "--writer-store", str(tmp_path / "c")])
    assert "prompt_pool" not in cap and canon.asks_firearm(cap["writer"].spec)
    made.clear()
    _cmd, cap = _rest_cmd(monkeypatch, ["--prompts", CAP, "--woosh-upgrade", "flow", "--woosh-upgrade-best", "4",
                                        "--woosh-upgrade-pool", "canon", "--writer-store", str(tmp_path / "d")])
    assert not canon.asks_firearm(cap["prompt_pool"].writers[0].spec)
    _cmd, cap = _rest_cmd(monkeypatch, ["--prompts", CAP, "--route", "words"])
    assert "prompt_pool" not in cap and cap["gen_kwargs"]["route"] == "words"


def test_serve_hands_the_meaning_lookup_to_the_router_only_under_the_meaning_route(monkeypatch):
    from sarge.sidecar import server as srv
    k = CacheKey("Zorbo_Bang", "", "normal")
    ask = FakeWriter("canon", {str(k): dict(_rec("A single zz bang sound, recorded up close."),
                                            canon={"name": "zz bang", "material": "", "continuous": False,
                                                   "firearm": True})})
    ask.spec = canon.style(CAP, True)
    ask.start = ask.stop = lambda *a: None
    pool = prompt_pool.PromptPool([ask], route=[ask])
    for mode, want in (("meaning", "guns"), ("families", "rest")):
        g = RoutedGenerator(guns=Fake("guns"), rest=Fake("rest"), warm_guns=False, route=mode)
        from sarge.affordance import AffordanceEngine
        eng = AffordanceEngine(g, realtime=True, vram_budget_kb=200_000, authored=set())
        s = srv.serve(engine=eng, host="127.0.0.1", port=0, warmup=False, prompt_pool=pool)
        try:
            assert g.route(k) == want and s.route_meaning == (mode == "meaning")
            assert (s.prompt_rev is not None) == (mode == "meaning")
        finally:
            s.shutdown()
            s.server_close()


def test_canon_writer_reads_model_codes_whole_only_when_it_cleans_names_or_routes():
    cp = words.code_phrase
    assert cp("QX15_Zorbo_Single") == "qx15 zorbo single" and cp("Zk98Zorbo_Krell_R") == "zk98 zorbo krell r"
    assert cp("Zorbo_Krell_02") == "zorbo krell 02" and cp("ZorboKrell3") == "zorbo krell 3"
    for n in ("SFX_Zorbo_Krell_01", "zorbo-krell.wav", "Zorbo Krell", "PlayZorbo_L", ""):
        assert cp(n) == words.phrase(n)
    k = CacheKey("QX15_Zorbo_Single", "Zk98", "normal")
    assert wr.key_text(k) == "event: qx 15 zorbo single\nmaterial: zk 98"
    assert wr.key_text(k, True) == "event: qx15 zorbo single\nmaterial: zk98"
    assert not canon.style(CAP)["codes"] and canon.style(CLEAN)["codes"] and canon.style(CAP, True)["codes"]
    head = _head_module("src/sarge/canon.py", "sarge._head_canon_p12", ref="d1c77f0")
    assert canon.ident(CAP) == head.ident(CAP) and canon.ident(CLEAN) != head.ident(CLEAN)
    llm = FakeLLM(lambda msgs: _canon_raw("zz bang"))
    w = wr.build("llama", style="canon", template=CLEAN, store="unused", backend=llm)
    w.spec = canon.style(CLEAN)
    w.write(k)
    assert llm.calls[-1][0][-1]["content"] == "event: qx15 zorbo single\nmaterial: zk98"
    plain = FakeLLM(lambda msgs: _canon_raw("zz bang"))
    w2 = wr.build("llama", style="canon", template=CAP, store="unused", backend=plain)
    w2.write(k)
    assert plain.calls[-1][0][-1]["content"] == "event: qx 15 zorbo single\nmaterial: zk 98"
