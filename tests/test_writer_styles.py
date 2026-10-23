import ast
import json
import os
import random
import string
import sys

import pytest

from sarge import scene, writer as w
from sarge.types import CacheKey

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
BENCH = os.path.join(SCRIPTS, "bench")
NEW = ("named", "texture", "title", "caption")
V1_IDENT = {"backend": "llama", "grammar": w.GBNF,
            "model": "sha256:3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd",
            "runtime": "version: 0.6.0-dev (build 1, commit b42b7e6d3) 22920", "threads": 4}
FIELDS = {"event", "material", "made", "by", "object", "touches", "contact", "action", "force", "room", "source",
          "sound"}
PROTOCOL = set(scene.SOURCES) | set(scene.ACTS) | set(scene.ROOMS) | set(w.STOP) | FIELDS


def _bench():
    if BENCH not in sys.path:
        sys.path.insert(0, BENCH)
    import llm_eval
    import generic_eval
    return llm_eval, generic_eval


def _ast_strings(path, names):
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            out += [n.value for n in ast.walk(node.value) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    return out


def eval_scenes(fresh=True):
    le, ge = _bench()
    out = []
    for et, mat in list(ge.DEMO) + list(ge.UNSEEN) + list(ge.FRESH):
        out.append({"event": et, "material": mat, "names": "", "touch": ""})
    for f in le.FRESH if fresh else ():
        sc = f.get("scene") or {}
        out.append({"event": f["event_type"], "material": f["material"],
                    "names": " ".join(str(sc.get(x, "")) for x in ("obj", "who")), "touch": ""})
    if os.path.isfile(le.SCENES):
        with open(le.SCENES, encoding="utf-8") as fh:
            for r in json.load(fh):
                names = [str(r["scene"].get(x, "")) for x in ("obj", "who")]
                phys = r.get("phys") if isinstance(r.get("phys"), dict) else {}
                body = {b: str(phys[b].get("name", "")) for b in ("a", "b") if isinstance(phys.get(b), dict)}
                out.append({"event": r["event_type"], "material": r["material"],
                            "names": " ".join(names + [body.get("a", "")]), "touch": body.get("b", "")})
    probe = _ast_strings(os.path.join(BENCH, "generic_probe.py"), {"DEMO", "UNSEEN"})
    vocab = _ast_strings(os.path.join(BENCH, "vocab_eval.py"), {"KEYS", "SURFACES"})
    for s in probe + vocab:
        out.append({"event": s, "material": "", "names": "", "touch": ""})
    return out


def eval_words():
    ws = set()
    for r in eval_scenes():
        for k in ("event", "material", "names", "touch"):
            ws.update(w._words(r[k]))
    return ws


def _phrase(text):
    return " ".join(w._words(text))


def _example_parts(text):
    vals = {}
    for line in text.split("\n"):
        k, _, v = line.partition(": ")
        vals[k] = v
    return _phrase(vals.get("event", "")), set(w._words(vals.get("material", "") + " " + vals.get("object", "") + " "
                                                      + vals.get("made by", ""))) - PROTOCOL


def test_default_style_is_v1_and_keeps_the_shipped_writer_id():
    assert w.DEFAULT_STYLE == "v1" and w.style_of(None) is w.STYLES["v1"]
    assert w.writer_id(V1_IDENT) == w.writer_id(V1_IDENT, "v1") == "07d28eff9ea0d691f824b91fac736642e6649295"
    st = w.STYLES["v1"]
    assert st["system"] is w.SYSTEM and st["examples"] is w.EXAMPLES and st["banned"] is w.BANNED
    assert st["ex_words"] == w.EX_WORDS and st["leak"] == w.LEAK and not st["named"] and not st["collide"]
    text = w.key_text(CacheKey("door_slam", "metal", "hard"))
    assert w.messages(text) == w.messages(text, "v1")
    assert w.messages(text)[0]["content"] == w.SYSTEM and len(w.messages(text)) == 2 + 2 * len(w.EXAMPLES)
    pw = w.PromptWriter(None, log=lambda m: None)
    assert pw.style == "v1" and w.build("off") is None


def test_every_style_has_its_own_writer_id_and_a_bad_name_fails():
    ids = {w.writer_id(V1_IDENT, s) for s in w.STYLES}
    assert len(ids) == len(w.STYLES)
    with pytest.raises(ValueError):
        w.style_of("louder")
    with pytest.raises(ValueError):
        w.build("llama", style="louder")


def test_v1_judging_is_unchanged_by_the_style_layer():
    cases = [
        (CacheKey("horn_honk", "", "normal", scene="src=player;obj=car hatchback;act=use;room=open"),
         {"source": "Car horn", "action": "a player's car horn emits a sharp blast", "sound": "loud, brief honk"}),
        (CacheKey("impact", "rubber", "normal"),
         {"source": "Ball bounce", "action": "a large inflated rubber ball bounces on metal",
          "sound": "hollow thud, soft bounce"}),
        (CacheKey("explode", "", "hard", scene="src=world;obj=barrel explosive;act=break;room=open"),
         {"source": "Barrel break", "action": "a wooden barrel ruptures violently",
          "sound": "sharp crack, heavy thud"}),
    ]
    for k, slots in cases:
        raw, text = json.dumps(slots), w.key_text(k)
        assert w.judge(raw, k, text) == w.judge(raw, k, text, "v1") == w.judge(raw, k, text, w.STYLES["v1"])
    k, slots = cases[2]
    rec = w.judge(json.dumps(slots), k, w.key_text(k))
    assert rec["woosh"].startswith("Barrel break, a wooden barrel ruptures violently, sharp crack, heavy thud")
    big = w.judge(json.dumps(cases[1][1]), cases[1][0], w.key_text(cases[1][0]))
    assert big["slots"]["sound"] == "soft bounce"
    kept = w.judge(json.dumps(cases[1][1]), cases[1][0], w.key_text(cases[1][0]), "named")
    assert kept["slots"]["sound"] == "hollow thud, soft bounce"


def test_naming_rule_keeps_the_request_words_in_front():
    k = CacheKey("explode", "", "hard", scene="src=world;obj=barrel explosive;act=break;room=open")
    raw = json.dumps({"source": "Barrel break", "action": "a wooden barrel ruptures violently",
                      "sound": "sharp crack, deep boom"})
    rec = w.judge(raw, k, w.key_text(k), "named")
    assert rec["flags"] == ["unnamed:source"] and rec["slots"]["source"] == "Explode, Barrel break"
    assert rec["sa3"].startswith("TrackType: SFX, Explode, Barrel break, a wooden barrel ruptures violently")
    ok = w.judge(json.dumps({"source": "Barrel explosion", "action": "a barrel blows apart", "sound": "deep boom"}), k,
                 w.key_text(k), "named")
    assert ok["flags"] == [] and not ok["repaired"]
    g = CacheKey("glass_shatter", "", "normal")
    gr = w.judge(json.dumps({"source": "Glass breaking", "action": "a piece of glass striking a surface",
                             "sound": "sharp crack"}), g, w.key_text(g), "texture")
    assert gr["slots"]["source"] == "Glass shatter, Glass breaking"
    s = CacheKey("shoot", "pistol", "normal")
    sr = w.judge(json.dumps({"source": "Pistol shot", "action": "a pistol fires once", "sound": "sharp crack"}), s,
                 w.key_text(s), "named")
    assert sr["slots"]["source"] == "Pistol shoot, Pistol shot"
    f = CacheKey("footstep", "wood", "normal")
    assert w.unnamed("Wooden steps", w.key_text(f)) == ["footstep"]
    assert w.unnamed("Wood footsteps", w.key_text(f)) == [] and w.unnamed("a foot on wood", w.key_text(f)) == []
    p = CacheKey("PlayerFootstep", "snow", "normal")
    assert w.name_words(w.key_text(p)) == ["snow", "footstep"] and w.name_phrase(w.key_text(p)) == "snow footstep"
    c = w.judge(json.dumps({"source": "Glass shatter", "action": "a piece of glass breaking apart",
                            "sound": "bright crackle"}), g, w.key_text(g), "caption")
    assert c["flags"] == ["unnamed:action"] and c["slots"]["action"] == "glass shatter, a piece of glass breaking apart"
    assert c["woosh"] == "glass shatter, a piece of glass breaking apart, bright crackle"


def test_collision_words_leave_the_sound_field_unless_the_request_has_them():
    k = CacheKey("footstep", "wood", "normal")
    raw = json.dumps({"source": "Wood footstep", "action": "a foot on a wooden floor",
                      "sound": "solid thud, dull impact"})
    for s in ("texture", "title", "caption"):
        rec = w.judge(raw, k, w.key_text(k), s)
        assert not rec.get("failed") and rec["slots"]["sound"] == "" and "collide:thud" in rec["flags"]
    assert w.judge(raw, k, w.key_text(k), "texture")["woosh"] == "Wood footstep, a foot on a wooden floor"
    assert w.judge(raw, k, w.key_text(k), "title")["woosh"] == "Wood footstep"
    assert w.judge(raw, k, w.key_text(k), "named")["slots"]["sound"] == "solid thud, dull impact"
    assert w.judge(raw, k, w.key_text(k))["slots"]["sound"] == "solid thud, dull impact"
    mixed = json.dumps({"source": "Gravel footstep", "action": "a foot on loose gravel",
                        "sound": "crunchy scrape, sharp impact"})
    kg = CacheKey("footstep", "gravel", "normal")
    assert w.judge(mixed, kg, w.key_text(kg), "texture")["slots"]["sound"] == "crunchy scrape"
    kb = CacheKey("bullet impact", "glass", "normal")
    keep = w.judge(json.dumps({"source": "Glass bullet impact", "action": "a bullet through a glass pane",
                               "sound": "bright impact, tinkle"}), kb, w.key_text(kb), "texture")
    assert keep["flags"] == [] and keep["slots"]["sound"] == "bright impact, tinkle"


def test_style_forms_on_both_routes():
    k = CacheKey("rain_loop", "tin", "normal", scene="src=world;obj=roof tin shack;act=loop;room=small")
    slots = {"source": "Tin rain", "action": "steady rain on a tin roof", "sound": "patter, bright drumming"}
    assert w.compose(slots, k, "woosh") == w.compose(slots, k, "woosh", "texture") == (
        "Tin rain, steady rain on a tin roof, patter, bright drumming, in a small room")
    assert w.compose(slots, k, "woosh", "title") == "Tin rain, patter, bright drumming, in a small room"
    assert w.compose(slots, k, "woosh", "caption") == (
        "steady rain on a tin roof, patter, bright drumming, in a small room")
    assert w.compose(slots, k, "sa3", "title") == w.compose(slots, k, "sa3")
    assert w.compose(slots, k, "sa3", "caption") == (
        "TrackType: SFX, Tin rain. Steady rain on a tin roof, patter, bright drumming, a single sound recorded up "
        "close, in a small room.")
    assert w.compose(dict(slots, sound=""), k, "woosh", "title") == "Tin rain, in a small room"


def _substitutes(n_random=60):
    excluded = set(PROTOCOL) | set(w._words(_case("qqqq", "zzzz", "")[1]))
    for st in w.STYLES.values():
        excluded |= set(st["leak"])
    rng = random.Random(7)
    rand = ["".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 9))) for _ in range(n_random)]
    real = sorted(x for x in eval_words() if x.isalpha() and len(x) >= 4)
    out, skipped = [], []
    for x in real + rand:
        if (x in excluded or w.BANNED.fullmatch(x) or w.COLLIDE.fullmatch(x) or w.CONTROLLER.fullmatch(x)
                or x in out):
            skipped.append(x)
            continue
        out.append(x)
    return out, skipped


def _case(ev, mat, style_scene):
    k = CacheKey(ev, mat, "normal", scene=style_scene)
    raw = json.dumps({"source": "%s %s" % (mat.capitalize(), ev), "action": "a %s %s heard up close" % (mat, ev),
                      "sound": "crisp attack, airy %s tail" % ev})
    return k, raw


def _traced(fn):
    lines = set()
    target = os.path.abspath(w.__file__)

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


def test_no_style_branches_on_the_event_or_the_material():
    subs, skipped = _substitutes()
    assert len(subs) > 150
    assert all(x in PROTOCOL or w.BANNED.fullmatch(x) or w.COLLIDE.fullmatch(x) or w.CONTROLLER.fullmatch(x)
               or any(x in st["leak"] for st in w.STYLES.values()) for x in skipped)
    rng = random.Random(11)
    pairs = [(e, rng.choice([m for m in subs if m != e])) for e in subs]
    sc = "src=npc;who=keeper;obj=lantern;act=use;room=medium"
    for style in w.STYLES:
        ref_k, ref_raw = _case("qqqq", "zzzz", sc)
        ref, ref_lines = _traced(lambda: w.judge(ref_raw, ref_k, w.key_text(ref_k), style))
        for ev, mat in pairs:
            k, raw = _case(ev, mat, sc)
            got, got_lines = _traced(lambda: w.judge(raw, k, w.key_text(k), style))
            assert got_lines == ref_lines, (style, ev, mat)
            for f in ("woosh", "sa3"):
                assert got[f] == ref[f].replace("qqqq", ev).replace("Zzzz", mat.capitalize()).replace("zzzz", mat), (
                    style, ev, mat, f)
            assert got["flags"] == ref["flags"] == []


def test_style_texts_and_new_examples_name_no_eval_scenario():
    ev = eval_words()
    for name in NEW:
        st = w.STYLES[name]
        assert not (set(w._words(st["system"])) & ev) - PROTOCOL, name
        for text, slots in st["examples"]:
            flags = w.check(slots, text, CacheKey("x", "", "normal"), st)
            assert all(f.startswith("copy:") for f in flags), (name, text)
    for text, slots in w.NEW_EXAMPLES:
        ins = set()
        for line in text.split("\n"):
            ins.update(w._words(line.partition(": ")[2]))
        assert not (ins & ev) - PROTOCOL, text
        outs = set(w._words(" ".join(slots.values())))
        assert not (outs & ev) - PROTOCOL, slots
    assert all(w.STYLES[s]["examples"] is w.NEW_EXAMPLES for s in ("texture", "title", "caption"))
    assert w.STYLES["named"]["examples"] is w.EXAMPLES


def test_no_style_example_is_an_eval_scene():
    scenes = [(_phrase(r["event"]), set(w._words(r["material"] + " " + r["names"])) - PROTOCOL) for r in eval_scenes()]
    events = {e for e, _n in scenes}
    for name, st in w.STYLES.items():
        for text, _slots in st["examples"]:
            ev, nouns = _example_parts(text)
            assert not any(ev == e and nouns & n for e, n in scenes), (name, text)
            if name in ("texture", "title", "caption"):
                assert ev not in events, (name, text)
    le, _ge = _bench()
    assert len(le.FRESH) == 30 and len({f["id"] for f in le.FRESH}) == 30
    known = {_phrase(r["event"]) for r in eval_scenes(fresh=False)}
    assert not known & {_phrase(f["event_type"]) for f in le.FRESH}


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


def test_sidecar_writer_style_flag_defaults_to_v1(rs, monkeypatch, tmp_path):
    assert tuple(rs.WRITER_STYLES) == tuple(w.STYLES) + (w.CANON_STYLE,)
    kw = _main_kwargs(rs, monkeypatch, ["--writer", "ollama", "--writer-store", str(tmp_path)])
    assert kw["writer"].style == "v1"
    kw = _main_kwargs(rs, monkeypatch, ["--writer", "ollama", "--writer-style", "texture", "--writer-store",
                                        str(tmp_path)])
    assert kw["writer"].style == "texture" and "style texture" in kw["writer"].describe()
    with pytest.raises(SystemExit):
        _main_kwargs(rs, monkeypatch, ["--writer", "ollama", "--writer-style", "louder"])
    assert not {"writer", "scene"} & set(_main_kwargs(rs, monkeypatch, ["--writer-style", "caption"]))


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
                           "sound": "dull thud, airy tail"})

    def close(self):
        pass


def test_a_style_writes_its_own_store_with_its_own_messages(tmp_path):
    k = CacheKey("kettle_boil", "steel", "normal")
    recs = {}
    for style in ("v1", "texture"):
        be = EchoBackend()
        pw = w.PromptWriter(be, store_dir=str(tmp_path), log=lambda m: None, style=style)
        pw.setup()
        rec = pw.write(k)
        recs[style] = (pw.id, rec, be.calls[0])
        assert pw.store.path.endswith("prompts_%s.json" % pw.id[:12])
    assert recs["v1"][0] != recs["texture"][0]
    assert recs["v1"][2][0]["content"] == w.SYSTEM and recs["texture"][2][0]["content"] == w.STYLES["texture"]["system"]
    assert "style" not in recs["v1"][1] and recs["texture"][1]["style"] == "texture"
    assert recs["v1"][1]["slots"]["sound"] == "dull thud, airy tail"
    assert recs["texture"][1]["slots"]["sound"] == "airy tail"
