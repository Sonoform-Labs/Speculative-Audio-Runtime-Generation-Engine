import glob
import json
import os
import random
import string
import sys

import numpy as np
import pytest

from sarge import ground as gd
from sarge import writer as w
from sarge.types import CacheKey

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
BENCH = os.path.join(SCRIPTS, "bench")
ROUND2 = os.path.join(ROOT, "data", "_gate", "llm", "eval", "wstyles")
REAL = os.environ.get("SARGE_CAPTION_INDEX", gd.DEFAULT_INDEX)
FILLER = ("a", "the", "on", "and", "of", "crisp", "airy")
CAPTIONS = ("a crisp qqqq on the zzzz", "the zzzz and a crisp airy qqqq", "airy qqqq of zzzz",
            "a man talks while a qqqq rings", "a crisp airy zzzz", "the qqqq and the zzzz",
            "a loud qqqq on the zzzz", "qqqq zzzz then a crisp qqqq")


def _bench():
    if BENCH not in sys.path:
        sys.path.insert(0, BENCH)
    import llm_eval
    import generic_eval
    return llm_eval, generic_eval


class CountEmbedder:
    def encode(self, texts):
        v = np.array([[1.0, len(w._words(t)) / 10.0, 0.5] for t in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def one(self, text):
        return self.encode([text])[0]

    def ident(self):
        return {"embedder": "count"}


def _index(folder, captions):
    os.makedirs(folder, exist_ok=True)
    emb = CountEmbedder().encode(captions).astype(np.float16)
    with open(os.path.join(folder, gd.FILES["captions"]), "w", encoding="utf-8") as fh:
        json.dump({"text": list(captions), "corpus": ["fake"] * len(captions), "src": [""] * len(captions)}, fh)
    np.save(os.path.join(folder, gd.FILES["emb"]), emb)
    with open(os.path.join(folder, gd.FILES["meta"]), "w", encoding="utf-8") as fh:
        json.dump({"digest": "fake", "count": len(captions)}, fh)
    return folder


def _grounder(mode, folder, captions=CAPTIONS):
    g = gd.Grounder(mode, _index(folder, captions))
    g.index = gd.CaptionIndex(folder)
    g.embedder = CountEmbedder()
    g._ident = {"ground": gd.VERSION, "mode": mode, "index": "fake"}
    return g


class EchoBackend:
    name = "fake"

    def __init__(self):
        self.calls = []

    def ident(self):
        return {"backend": "fake", "model": "m", "runtime": "t", "threads": 4}

    def start(self):
        pass

    def complete(self, msgs):
        self.calls.append(msgs)
        ev = msgs[-1]["content"].split("\n")[0].partition(": ")[2]
        return json.dumps({"source": ev.capitalize() + " sound", "action": "one %s heard here" % ev,
                           "sound": "crisp attack, airy tail"})

    def close(self):
        pass


def test_flags_off_keeps_every_round_two_writer_id_and_judgment():
    stores = sorted(glob.glob(os.path.join(ROUND2, "writer", "prompts_*.json")))
    if len(stores) < 5:
        pytest.skip("round-two stores not on this machine")
    style_of_id = {}
    n = 0
    for p in stores:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        recs = data["records"]
        styles = {r.get("style", w.DEFAULT_STYLE) for r in recs.values()}
        assert len(styles) == 1
        style = styles.pop()
        assert w.writer_id(data["ident"], style) == data["writer_id"]
        style_of_id[data["writer_id"][:12]] = style
        for k, rec in recs.items():
            key = w.key_of(rec["key"])
            assert str(key) == k
            got = w.judge(rec["raw"], key, rec["text"], style)
            for f in ("woosh", "sa3", "slots", "flags", "failed"):
                assert got.get(f) == rec.get(f), (p, k, f)
            n += 1
    assert style_of_id["07d28eff9ea0"] == "v1" and style_of_id["39800c4737a4"] == "named" and n >= 500


def test_flags_off_writer_builds_and_messages_are_unchanged(tmp_path):
    assert w.build("off", ground="caption") is None
    pw = w.PromptWriter(EchoBackend(), store_dir=str(tmp_path), log=lambda m: None, style="named")
    assert pw.ground is None and pw.spec is w.STYLES["named"]
    pw.setup()
    k = CacheKey("kettle_boil", "steel", "normal")
    rec = pw.write(k)
    text = w.key_text(k)
    assert pw.backend.calls[0] == w.messages(text, "named")
    assert "ground" not in rec and rec["style"] == "named"
    assert pw.id == w.writer_id(EchoBackend().ident(), "named")
    with pytest.raises(ValueError):
        w.build("llama", ground="caption")
    with pytest.raises(ValueError):
        w.build("ground", ground="anchors")
    with pytest.raises(ValueError):
        w.build("ground")


def test_llm_eval_prompts_and_scenarios_match_round_two():
    p = os.path.join(ROUND2, "scenarios.json")
    if not os.path.isfile(p):
        pytest.skip("round-two scenario file not on this machine")
    le, _ge = _bench()
    with open(p, encoding="utf-8") as fh:
        old = {r["sid"]: r for r in json.load(fh)["rows"]}
    rows = le.build_scenarios("all")
    assert [r["sid"] for r in rows] == list(old)
    for r in rows:
        o = old[r["sid"]]
        assert r["key"] == o["key"] and r["model"] == o["model"] and r["text"] == o["text"]
        k = w.key_of(r["key"])
        assert le.phrase_prompt(k, r["model"]) == o["prompts"]["phrasebook"]
        assert le.caption_prompt(k, r["model"]) == o["prompts"]["caption"]


@pytest.fixture
def rs(monkeypatch):
    if SCRIPTS not in sys.path:
        monkeypatch.syspath_prepend(SCRIPTS)
    import run_sidecar
    return run_sidecar


def _main_kwargs(rs, monkeypatch, argv):
    captured = {}

    class Stop(Exception):
        pass

    def fake_serve(**kw):
        captured.update(kw)
        raise Stop()

    monkeypatch.setattr(rs, "serve", fake_serve)
    monkeypatch.setattr(rs, "check_weights", lambda name: None)
    with pytest.raises(Stop):
        rs.main(["--generator", "sa3", "--no-cache"] + argv)
    return captured


def test_sidecar_ground_flags_default_off(rs, monkeypatch, tmp_path):
    assert tuple(rs.WRITER_GROUND) == gd.MODES
    assert "writer" not in _main_kwargs(rs, monkeypatch, [])
    kw = _main_kwargs(rs, monkeypatch, ["--writer", "ollama", "--writer-store", str(tmp_path)])
    assert kw["writer"].ground is None
    kw = _main_kwargs(rs, monkeypatch, ["--writer", "ground", "--writer-ground", "caption", "--writer-store",
                                        str(tmp_path), "--writer-ground-index", str(tmp_path / "idx")])
    assert kw["writer"].ground.mode == "caption" and "ground caption" in kw["writer"].describe()
    assert kw["writer"].ground.folder == str(tmp_path / "idx")
    kw = _main_kwargs(rs, monkeypatch, ["--writer", "ollama", "--writer-style", "named", "--writer-ground",
                                        "anchors", "--writer-store", str(tmp_path)])
    assert kw["writer"].ground.mode == "anchors" and kw["writer"].style == "named"
    for argv in (["--writer-ground", "caption"], ["--writer", "ground"], ["--writer", "ground", "--writer-ground",
                                                                         "anchors"],
                 ["--writer", "ollama", "--writer-ground", "words"]):
        with pytest.raises(SystemExit):
            _main_kwargs(rs, monkeypatch, argv + ["--writer-store", str(tmp_path)])


def test_retrieval_filters_ranks_and_composes(tmp_path):
    caps = ("a man talks while a horn honks", "a car horn honks", "a loud car horn honks twice",
            "the car horn blares and honks", "wind blows", "a horn honks then a dog barks")
    k = CacheKey("car_horn", "", "normal", scene="src=npc;who=driver;obj=car sedan;act=use;room=open")
    text = w.key_text(k)
    g = _grounder("caption", str(tmp_path / "c"), caps)
    ranked = g.ranked(k, text)
    kept = [g.index.text[h["i"]] for h in ranked]
    assert "a man talks while a horn honks" not in kept and "a loud car horn honks twice" not in kept
    assert kept.index("a car horn honks") < kept.index("a horn honks then a dog barks")
    spec = g.style("v1")
    raw, info = g.answer(k, text, None, spec)
    rec = w.judge(raw, k, text, spec)
    assert rec["woosh"] == "Car horn, a car horn honks, outdoors"
    assert rec["sa3"] == "TrackType: SFX, Car horn, a car horn honks, a single sound recorded up close, outdoors."
    assert info["mode"] == "caption" and info["hits"][0][4] == "a car horn honks"
    gw = _grounder("words", str(tmp_path / "w"), caps)
    raw, info = gw.answer(k, text, None, gw.style("v1"))
    rec = w.judge(raw, k, text, gw.style("v1"))
    assert rec["slots"]["source"] == "Car horn" and "honks" in rec["slots"]["sound"]
    assert not {"man", "talks", "loud", "twice", "then"} & set(w._words(rec["woosh"]))


def test_anchor_mode_shows_captions_and_judges_on_the_facts_alone(tmp_path):
    caps = ("a loud car horn honks", "a car horn honks", "the car horn blares", "a horn beeps")
    g = _grounder("anchors", str(tmp_path / "a"), caps)
    be = EchoBackend()
    pw = w.PromptWriter(be, store_dir=str(tmp_path / "s"), log=lambda m: None, style="named", ground=g)
    pw.setup()
    assert pw.spec["name"] == "named+anchors" and pw.spec["system"].endswith(gd.ANCHOR_RULE)
    assert all(gd.ANCHOR_HEAD in u for u, _s in pw.spec["examples"])
    k = CacheKey("car_horn", "", "normal")
    rec = pw.write(k)
    user = be.calls[0][-1]["content"]
    assert user.startswith(w.key_text(k) + "\n" + gd.ANCHOR_HEAD)
    assert "- a car horn honks" in user and "loud" not in user
    assert rec["text"] == w.key_text(k) and rec["ground"]["mode"] == "anchors"
    assert pw.id != w.writer_id(be.ident(), "named")
    assert rec["style"] == "named+anchors"


def test_retrieval_is_deterministic(tmp_path):
    caps = tuple("a %s %s on the floor" % (a, b) for a in ("crisp", "airy", "dull", "bright")
                 for b in ("tap", "knock", "ring", "hum", "click"))
    g = _grounder("caption", str(tmp_path / "d"), caps)
    keys = [CacheKey(e, m, "normal") for e, m in (("tap", "wood"), ("ring", "metal"), ("hum", ""), ("click", "glass"))]
    first = [g.ranked(k, w.key_text(k)) for k in keys]
    again = [g.ranked(k, w.key_text(k)) for k in reversed(keys)][::-1]
    assert first == again
    for r in first:
        assert [h["i"] for h in r] == [h["i"] for h in sorted(r, key=lambda h: (-h["score"], h["i"]))]
    s = g.index.scores(g.query(w.key_text(keys[0])))
    assert g.index.top(g.query(w.key_text(keys[0])), 5) == g.index.top(g.query(w.key_text(keys[0])), 5)
    assert len(set(np.round(s, 6))) < len(s)


FIXED = FILLER + ("qqqq", "zzzz", "sound", "heard", "here", "attack", "tail", "talks", "rings", "loud", "keeper",
                  "lantern", "medium", "indoors", "single", "recorded", "close")
CLASSES = (gd.SPEECH, gd.PEOPLE, gd.MULTI, gd.COUNT, w.BANNED, w.BANNED_OPEN, w.COLLIDE, w.CONTROLLER)


def _eval_words():
    le, ge = _bench()
    words = set()
    for et, mat in list(ge.DEMO) + list(ge.UNSEEN) + list(ge.FRESH):
        words.update(w._words(et) + w._words(mat))
    for f in le.FRESH + le.FRESH3:
        words.update(w._words(f["event_type"]) + w._words(f["material"]))
        sc = f.get("scene") or {}
        words.update(w._words(" ".join(str(sc.get(x, "")) for x in ("obj", "who"))))
    with open(le.SCENES, encoding="utf-8") as fh:
        for r in json.load(fh):
            words.update(w._words(r["event_type"]) + w._words(r["material"]))
    return words


def _substitutes(n_random=40):
    from sarge import scene
    rng = random.Random(5)
    rand = ["".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 9))) for _ in range(n_random)]
    out, skipped = [], []
    for x in sorted(x for x in _eval_words() if x.isalpha() and len(x) >= 4) + rand:
        if (x in w.STOP or x in out or x in scene.ACTS or any(rx.fullmatch(x) for rx in CLASSES)
                or any(w._near(x, f) or w._near(f, x) for f in FIXED)):
            skipped.append(x)
            continue
        out.append(x)
    return out, skipped


class NameEcho(EchoBackend):
    def complete(self, msgs):
        self.calls.append(msgs)
        f = gd.facts(msgs[-1]["content"])
        ev, mat = f.get("event", ""), f.get("material", "")
        return json.dumps({"source": "%s %s" % (mat.capitalize(), ev), "action": "one %s heard here" % ev,
                           "sound": "crisp attack, airy tail"})


def _traced(fn):
    lines = set()
    target = os.path.abspath(gd.__file__)

    def tracer(frame, event, arg):
        if os.path.abspath(frame.f_code.co_filename) == target:
            if event == "line":
                lines.add(frame.f_lineno)
            return tracer
        return None

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        out = fn()
    finally:
        sys.settrace(old)
    return out, frozenset(lines)


def test_no_grounding_mode_branches_on_the_event_or_the_material(tmp_path):
    subs, skipped = _substitutes()
    assert len(subs) > 150
    real = [x for x in skipped if x in _eval_words()]
    assert len(real) < 40, real
    rng = random.Random(3)
    pairs = [(e, rng.choice([m for m in subs if m != e and not w._near(m, e) and not w._near(e, m)])) for e in subs]
    sc = "src=npc;who=keeper;obj=lantern;act=use;room=medium"

    def run(mode, ev, mat, folder):
        caps = tuple(c.replace("qqqq", ev).replace("zzzz", mat) for c in CAPTIONS)
        g = _grounder(mode, folder, caps)
        k = CacheKey(ev, mat, "normal", scene=sc)
        text = w.key_text(k)
        spec = g.style("named" if mode == "anchors" else "v1")
        be = NameEcho()
        raw, info = g.answer(k, text, be, spec)
        rec = w.judge(raw, k, text, spec)
        return {"woosh": rec.get("woosh"), "sa3": rec.get("sa3"), "flags": rec.get("flags"),
                "hits": [h[0] for h in info["hits"]], "user": be.calls[0][-1]["content"] if be.calls else ""}

    for mode in ("caption", "words", "anchors"):
        ref, ref_lines = _traced(lambda: run(mode, "qqqq", "zzzz", str(tmp_path / ("ref_" + mode))))
        assert ref["flags"] == [] and ref["hits"]
        for i, (ev, mat) in enumerate(pairs):
            got, got_lines = _traced(lambda: run(mode, ev, mat, str(tmp_path / ("%s_%d" % (mode, i)))))
            assert got_lines == ref_lines, (mode, ev, mat)
            assert got["hits"] == ref["hits"], (mode, ev, mat)
            for f in ("woosh", "sa3", "user"):
                want = ref[f].replace("qqqq", ev).replace("Zzzz", mat.capitalize()).replace("zzzz", mat)
                assert got[f] == want, (mode, ev, mat, f)


def test_ground_code_names_no_eval_scenario():
    le, ge = _bench()
    ev = set()
    for et, mat in list(ge.DEMO) + list(ge.UNSEEN) + list(ge.FRESH):
        ev.update(w._words(et) + w._words(mat))
    for f in le.FRESH + le.FRESH3:
        ev.update(w._words(f["event_type"]) + w._words(f["material"]))
    lists = " ".join((gd.SPEECH.pattern, gd.PEOPLE.pattern, gd.MULTI.pattern, gd.COUNT.pattern, gd.ANCHOR_HEAD,
                      gd.ANCHOR_RULE))
    assert not set(w._words(lists)) & {x for x in ev if len(x) >= 4}


def test_new_fresh_scenarios_reuse_no_earlier_event():
    le, ge = _bench()
    assert len(le.FRESH3) == 30 and len({f["id"] for f in le.FRESH3}) == 30
    assert sum(1 for f in le.FRESH3 if f.get("scene")) == 15
    before = {" ".join(w._words(f["event_type"])) for f in le.FRESH}
    before |= {" ".join(w._words(et)) for et, _m in list(ge.DEMO) + list(ge.UNSEEN) + list(ge.FRESH)}
    with open(le.SCENES, encoding="utf-8") as fh:
        before |= {" ".join(w._words(r["event_type"])) for r in json.load(fh)}
    now = {" ".join(w._words(f["event_type"])) for f in le.FRESH3}
    assert not now & before
    types = sorted(f["type"] for f in le.FRESH3)
    assert types.count("footstep") == 4 and types.count("glass_break") == 3 and sum(
        t.startswith("gunshot") for t in types) == 5 and sum(1 for f in le.FRESH3 if f.get("water")) == 2
    rows = le.build_scenarios("fresh3")
    assert all(r["model"] == "sa3" for r in rows if r["type"].startswith("gunshot"))
    assert {r["sid"] for r in rows}.isdisjoint({r["sid"] for r in le.build_scenarios("known")})


def test_round_three_rule_compares_ground_arms_with_named():
    le, _ge = _bench()
    arms = ("phrasebook", "caption", "wr_named", "rg_caption")
    assert ("rg_caption", "wr_named") in le.cmp_pairs(arms)
    assert ("rg_caption", "wr_named") not in le.cmp_pairs(("phrasebook", "caption", "rg_caption"))

    def num(U, D, G=0.0, W=0.0):
        return {"U": U, "D": D, "G": G, "W": W, "A": (U + D) / 2.0, "Q": (U + D) / 2.0}

    n = {"wr_named": num(0.05, -0.1), "rg_anchor": num(0.10, -0.08), "rg_words": num(0.2, -0.2),
         "rg_caption": num(0.08, -0.06, W=-0.15)}
    dec = le.decide_selection(n, {}, le.GROUND_ARMS, "wr_named")
    assert dec["eligible"] == ["rg_anchor", "rg_caption"] and dec["pick"] == "rg_anchor" and dec["confirm"]
    assert dec["ref"] == "wr_named"
    n["rg_anchor"] = num(0.06, -0.08)
    n["rg_caption"] = num(0.06, -0.08)
    dec = le.decide_selection(n, {}, le.GROUND_ARMS, "wr_named")
    assert dec["pick"] == "rg_anchor" and not dec["confirm"]
    assert "ref" not in le.decide_selection(dict(n, llm=num(0.0, 0.0)), {}, le.GROUND_ARMS)


def _real():
    if not os.path.isfile(os.path.join(REAL, gd.FILES["meta"])):
        pytest.skip("caption index not built on this machine")
    return REAL


def test_real_index_holds_no_caption_tied_to_a_judge_reference():
    folder = _real()
    with open(os.path.join(folder, "exclusions.json"), encoding="utf-8") as fh:
        ex = json.load(fh)
    with open(os.path.join(folder, gd.FILES["captions"]), encoding="utf-8") as fh:
        caps = json.load(fh)
    ids, texts = set(ex["ids"]), set(ex["texts"])
    refs = set()
    for p in (os.path.join(ROOT, "data", "refs_heldout", "manifest.json"),
              os.path.join(ROOT, "data", "refs", "manifest.json")):
        with open(p, encoding="utf-8") as fh:
            for c in json.load(fh)["clips"]:
                if c["dataset"] == "foleyset":
                    refs.add("fs:%s" % c["freesound_id"])
                elif c["dataset"] == "esc50":
                    refs.add("fs:%s" % c["esc_src"])
    for f in glob.glob(os.path.join(ROOT, "data", "esc50", "**", "*.wav"), recursive=True):
        refs.add("fs:%s" % os.path.basename(f).split("-")[1])
    assert refs and refs <= ids
    assert not ids & set(caps["src"])
    assert not texts & {gd.caption_key(t) for t in caps["text"]}
    assert len(caps["text"]) == len(caps["src"]) == len(caps["corpus"])


def test_real_index_retrieval_is_deterministic_across_threads():
    folder = _real()
    keys = [CacheKey("car_horn", "car", "normal"), CacheKey("footstep", "gravel", "normal"),
            CacheKey("shoot", "pistol", "normal"), CacheKey("splash", "water", "normal")]
    out = []
    for threads in (4, 1, 4):
        g = gd.Grounder("caption", folder, threads=threads)
        g.start()
        out.append([[h["i"] for h in g.hits(k, w.key_text(k), 8)] for k in keys])
    assert out[0] == out[1] == out[2]
