import hashlib
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import zlib

import numpy as np
import pytest

from sarge import scene, writer as w
from sarge.affordance import AffordanceEngine
from sarge.generators.cached import CachedGenerator
from sarge.generators.mock import MockGenerator
from sarge.sidecar import SfxClient, serve, wire
from sarge.sidecar.server import engine_lock
from sarge.sidecar.wire import MsgType
from sarge.types import AudioClip, CacheKey, seed_for
from sarge.world import bucket

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
BALL = {"shape": "sphere", "ext": [1, 1, 1], "dyn": 1, "m": 1.0, "pm": "psfx_bouncy", "bnc": 0.8,
        "name": "AutoCaptureBall_0"}
TILE = {"shape": "box", "ext": [10, 0.2, 10], "dyn": 0, "pm": "psfx_bouncy", "bnc": 0.8, "tag": "metal",
        "name": "Surface_metal"}
PHYS = {"a": BALL, "b": TILE, "n": [0, 1, 0], "vn": 9.4, "vt": 0.3}


def code(prompt):
    return 0.001 + (zlib.crc32(str(prompt).encode()) % 997) / 1000.0


class PromptGen(MockGenerator):
    def __init__(self, delay=0.0):
        super().__init__()
        self.prompts = {}
        self.seen = []
        self.delay = delay

    def generate(self, key, *, variant=0, prompt=None):
        p = prompt or self.prompts.get(str(key), "")
        self.seen.append((str(key), variant, p))
        if self.delay:
            time.sleep(self.delay)
        return AudioClip(np.full(480, code(p), dtype=np.float32), self.sample_rate, variant_id=variant)


class RoutedPromptGen(PromptGen):
    def route(self, key):
        return "guns" if key.event_type == "shoot" else "rest"


def fake_reply(text):
    ev = text.split("\n")[0].partition(": ")[2]
    return json.dumps({"source": ev.capitalize() + " sound", "action": "one %s happening right here" % ev,
                       "sound": "sharp click, short tail"})


class FakeBackend:
    name = "fake"

    def __init__(self, reply=fake_reply, model="fake-model", delay=0.0):
        self.reply = reply
        self.model = model
        self.delay = delay
        self.calls = []
        self.started = 0
        self.seen_affinity = None

    def ident(self):
        return {"backend": "fake", "model": self.model, "runtime": "test", "threads": 4}

    def start(self):
        self.started += 1

    def complete(self, msgs):
        self.calls.append(msgs[-1]["content"])
        if self.delay:
            time.sleep(self.delay)
        return self.reply(msgs[-1]["content"])

    def close(self):
        pass


def wait_until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def _writer(tmp_path, backend=None, **kw):
    return w.PromptWriter(backend if backend is not None else FakeBackend(), store_dir=str(tmp_path),
                          log=lambda m: None, **kw)


def _srv(gen, writer=None, **kw):
    eng = AffordanceEngine(gen, realtime=True, vram_budget_kb=200_000, authored=set())
    srv = serve(engine=eng, host="127.0.0.1", port=0, warmup=False, deepen=True, writer=writer, **kw)
    if writer is not None:
        writer.start()
    return srv


def _stop(srv):
    srv.deepening = False
    if srv.writer is not None:
        srv.writer.stop()
    srv.shutdown()
    srv.server_close()


def test_scene_names_are_cleaned_the_same_for_every_game():
    assert scene.name_words("Barrel_Rusty_02") == "barrel rusty"
    assert scene.name_words("Cube (12)") == ""
    assert scene.name_words("Lightsaber_Blue") == "lightsaber"
    assert scene.name_words("Barrel_Explosive_Red") == "barrel explosive"
    assert scene.name_words("GrapplingHook_Gun") == "grappling hook gun"
    assert scene.name_words("SM_Prop_Car_Door_FL_LOD0 (Clone)") == "car door fl"
    assert scene.name_words("a|b;c=d") == "c d"
    assert scene.name_words("x" * 60 + "Door") == "x" * 48


def test_scene_quantises_to_a_fixed_order_key_with_enums_only():
    s = scene.quantise({"room": "open", "act": "hit", "obj": "Club_Wooden", "who": "Orc_Brute_02", "src": "npc"})
    assert s == "src=npc;who=orc brute;obj=club wooden;act=hit;room=open"
    assert scene.quantise({"src": "player", "who": "Hero", "obj": "Pistol_01", "act": "use"}) == \
        "src=player;obj=pistol;act=use"
    assert scene.quantise({"src": "boss", "act": "explode", "room": "tiny", "obj": ""}) == ""
    assert scene.quantise("src=npc") == "" and scene.quantise(None) == ""
    assert scene.parse(s) == {"src": "npc", "who": "orc brute", "obj": "club wooden", "act": "hit", "room": "open"}


def test_scene_physics_takes_object_names_and_kind_from_phys():
    s = scene.quantise({"src": "physics", "obj": "Ignored", "act": "slide", "room": "small"}, PHYS, "impact")
    assert s == "src=physics;obj=auto capture ball;on=surface metal;act=bounce;room=small"
    brk = scene.quantise({"src": "projectile"}, {"a": {"shape": "box", "ext": [1, 1, 0.05], "dyn": 0,
                                                      "name": "Window_03"}, "kind": "break"}, "break")
    assert brk == "src=projectile;obj=window;act=break"


def test_cache_key_without_scene_is_unchanged_and_seed_ignores_scene():
    plain = CacheKey("shoot", "pistol", "normal")
    assert str(plain) == "shoot|pistol|normal|default"
    assert seed_for(plain) == int(hashlib.md5(b"shoot|pistol|normal|default").hexdigest()[:8], 16) & 0x7FFFFFFF
    full = CacheKey("impact", "metal", "hard", "cave", phys="hit;a=x", cls="weapon", scene="src=npc;act=hit")
    assert str(full) == "impact|metal|hard|cave|hit;a=x|class=weapon|scene=src=npc;act=hit"
    base = CacheKey("impact", "metal", "hard", "cave", phys="hit;a=x")
    for v in (0, 1, 5):
        assert seed_for(full, v) == seed_for(base, v) == seed_for(CacheKey("impact", "metal", "hard", "cave",
                                                                           phys="hit;a=x", scene="src=ui"), v)
    assert CacheKey("a", "b", "c") != CacheKey("a", "b", "c", scene="src=ui")


def test_cached_path_moves_only_with_a_prompt_override(tmp_path):
    g = PromptGen()
    c = CachedGenerator(g, str(tmp_path))
    k = CacheKey("dog_bark", "dog", "normal")
    head = c._path(k, 0)
    assert c._path(k, 0, None) == head and c._path(k, 0, "") == head
    assert c._path(k, 0, "a dog barking") != head
    assert c._path(CacheKey("dog_bark", "dog", "normal", scene="src=npc"), 0) != head
    clip = c.generate(k, variant=0, prompt="a dog barking")
    assert g.seen[-1] == (str(k), 0, "a dog barking") and clip.samples[0] == pytest.approx(code("a dog barking"))
    g.prompts[str(k)] = "a dog barking"
    assert c._path(k, 0) == c._path(k, 0, "a dog barking")
    c.generate(k, variant=0)
    assert len(g.seen) == 1


def test_writer_input_reads_only_the_key_and_is_deterministic():
    k = CacheKey("melee_hit", "body", "hard", scene=scene.quantise(
        {"src": "npc", "who": "Orc_Brute", "obj": "Club_Wooden", "act": "hit", "room": "open"}))
    text = w.key_text(k)
    assert text == ("event: melee hit\nmaterial: body\nmade by: npc (orc brute)\nobject: club wooden\naction: hit\n"
                    "force: hard\nroom: open")
    assert w.key_text(w.key_of(w.key_fields(k))) == text
    assert json.dumps(w.messages(text)) == json.dumps(w.messages(w.key_text(k)))
    assert w.key_text(CacheKey("shoot", "pistol", "normal")) == "event: shoot\nmaterial: pistol"
    assert w.key_text(CacheKey("dog_bark", "dog", "normal")) == "event: dog bark"
    assert w.key_text(CacheKey("shoot", "rifle", "normal", cls="weapon")).endswith("class: weapon")


def test_writer_input_drops_size_and_build_from_physics():
    k = CacheKey("impact", "", "hard", phys=bucket(PHYS, "impact"),
                 scene=scene.quantise({"src": "physics", "room": "small"}, PHYS, "impact"))
    text = w.key_text(k)
    assert "large" not in text and "inflated" not in text and "huge" not in text and "static" not in text
    assert text == ("event: impact\nmade by: physics\nobject: auto capture ball, rubber ball\n"
                    "touches: surface metal, metal floor\ncontact: soft\naction: bounce\nforce: hard\nroom: small")
    assert w.key_text(CacheKey("impact", "metal", "hard", phys=bucket(PHYS, "impact"))) == (
        "event: impact\nmaterial: metal\nobject: rubber ball\ntouches: metal floor\ncontact: soft\naction: bounce\n"
        "force: hard")


def test_writer_id_changes_with_model_runtime_or_threads():
    base = {"backend": "llama", "model": "sha256:aa", "runtime": "v1", "threads": 4}
    assert w.writer_id(base) == w.writer_id(dict(base))
    for k, v in (("model", "sha256:bb"), ("runtime", "v2"), ("threads", 2), ("backend", "ollama")):
        assert w.writer_id(dict(base, **{k: v})) != w.writer_id(base)


def test_judge_accepts_clean_output_and_composes_both_routes():
    k = CacheKey("rain_loop", "tin", "normal", scene="src=world;obj=roof tin shack;act=loop;room=small")
    text = w.key_text(k)
    raw = json.dumps({"source": "Tin roof rain", "action": "steady rain hitting a tin roof",
                      "sound": "patter, metallic drumming"})
    rec = w.judge(raw, k, text)
    assert not rec.get("failed") and rec["flags"] == [] and not rec["repaired"]
    assert rec["woosh"] == "Tin roof rain, steady rain hitting a tin roof, patter, metallic drumming, in a small room"
    assert rec["sa3"] == ("TrackType: SFX, Tin roof rain, steady rain hitting a tin roof, patter, metallic drumming, "
                          "a single sound recorded up close, in a small room.")
    kp = CacheKey("impact", "", "hard", phys=bucket(PHYS, "impact"))
    rp = w.judge(json.dumps({"source": "ball bounce", "action": "a rubber ball striking a metal floor",
                             "sound": "dull thud"}), kp, w.key_text(kp))
    assert rp["woosh"] == "ball bounce, a rubber ball striking a metal floor, dull thud, a soft muffled bump"
    assert rp["sa3"].startswith("TrackType: SFX, Ball bounce,") and rp["sa3"].endswith(
        "a soft muffled bump, a single sound recorded up close.")
    kc = CacheKey("footstep", "stone", "normal", "cave")
    assert w.compose({"source": "Stone step", "action": "a boot on stone", "sound": "click"}, kc, "woosh").endswith(
        ", cave")


def test_judge_repairs_banned_leaked_and_copied_words_deterministically():
    k = CacheKey("horn_honk", "", "normal", scene="src=player;obj=car hatchback;act=use;room=open")
    rec = w.judge(json.dumps({"source": "Car horn", "action": "a player's car horn emits a sharp blast",
                              "sound": "loud, brief honk"}), k, w.key_text(k))
    assert rec["repaired"] and rec["flags"] == ["banned:loud", "banned:player"]
    assert rec["slots"] == {"source": "Car horn", "action": "a car horn emits a sharp blast", "sound": "brief honk"}
    kp = CacheKey("punch", "body", "normal", scene="src=player;obj=fist;act=hit")
    rp = w.judge(json.dumps({"source": "Fist punch", "action": "a fist hitting against a wooden desk drawer",
                             "sound": "dull thud"}), kp, w.key_text(kp))
    assert "leak:desk" in rp["flags"] and "leak:drawer" in rp["flags"]
    assert rp["slots"]["action"] == "a fist hitting"
    rq = w.judge(json.dumps({"source": "Punch", "action": "a fist hits a wooden desk", "sound": "hard thud"}), kp,
                 w.key_text(kp))
    assert rq["slots"]["action"] == "a fist hits"
    rs_ = w.judge(json.dumps({"source": "Metal footstep", "action": "a player steps on a metal floor",
                              "sound": "metallic clank"}), CacheKey("footstep", "metal", "normal", scene="src=player"),
                  w.key_text(CacheKey("footstep", "metal", "normal", scene="src=player")))
    assert rs_["slots"]["action"] == "steps on a metal floor"
    rh = w.judge(json.dumps({"source": "Gun reload", "action": "a player reloads a semi-automatic rifle",
                             "sound": "click, pitter-patter"}), CacheKey("reload", "rifle", "normal"),
                 w.key_text(CacheKey("reload", "rifle", "normal")))
    assert rh["slots"]["action"] == "reloads a semi-automatic rifle" and rh["slots"]["sound"] == "click, pitter-patter"
    rl = w.judge(json.dumps({"source": "Car horn", "action": "a vehicle emits a loud, sustained blast",
                             "sound": "continuous honk"}), CacheKey("car_horn", "car", "normal"),
                 w.key_text(CacheKey("car_horn", "car", "normal")))
    assert rl["slots"]["action"] == "a vehicle emits a sustained blast"
    rc = w.judge(json.dumps({"source": "Lid open", "action": "a lid pulled open on its runners",
                             "sound": "wooden slide, soft rumble"}), CacheKey("open", "lid", "normal"),
                 w.key_text(CacheKey("open", "lid", "normal")))
    assert "copy:pulled open on its" in rc["flags"] and "copy:wooden slide soft rumble" in rc["flags"]
    assert rc["failed"] and rc["after"] == ["unrepairable"]
    rc2 = w.judge(json.dumps({"source": "Lid open", "action": "a lid pulled open on its runners",
                              "sound": "dull clack"}), CacheKey("open", "lid", "normal"),
                  w.key_text(CacheKey("open", "lid", "normal")))
    assert rc2["repaired"] and rc2["slots"]["action"] == "a lid"
    assert w.repair({"source": "Lid open", "action": "a lid pulled open on its runners", "sound": "dull clack"},
                    ["copy:pulled open on its"], "event: open\nmaterial: lid")["action"] == "a lid"
    again = w.judge(json.dumps({"source": "Car horn", "action": "a player's car horn emits a sharp blast",
                                "sound": "loud, brief honk"}), k, w.key_text(k))
    assert again == rec
    big = w.judge(json.dumps({"source": "Ball bounce", "action": "a large inflated rubber ball bounces on metal",
                              "sound": "hollow thud, soft bounce"}), CacheKey("impact", "rubber", "normal"),
                  w.key_text(CacheKey("impact", "rubber", "normal")))
    assert big["slots"]["action"] == "a rubber ball bounces on metal" and big["slots"]["sound"] == "soft bounce"


def test_judge_loop_check_looks_inside_one_slot_only():
    k = CacheKey("growl", "", "normal", scene="src=npc;who=troll cave boss;room=large")
    rec = w.judge(json.dumps({"source": "Troll cave boss growl", "action": "a deep growl from a troll cave boss",
                              "sound": "rumbling growl"}), k, w.key_text(k))
    assert not rec.get("failed") and rec["flags"] == []
    looped = w.judge(json.dumps({"source": "Troll growl", "action": "a troll growl and a troll growl and more",
                                 "sound": "rumbling growl"}), k, w.key_text(k))
    assert "loop:action" in looped["flags"]


def test_judge_rejects_unparseable_ungrounded_or_overlong_output():
    k = CacheKey("door_slam", "metal", "hard")
    t = w.key_text(k)
    assert w.judge("not json", k, t)["flags"] == ["format"]
    assert w.judge(json.dumps({"source": "", "action": "x y", "sound": "z"}), k, t)["failed"]
    bad = w.judge(json.dumps({"source": "Thunder", "action": "rain falls over the valley", "sound": "rumble"}), k, t)
    assert bad["failed"] and "ungrounded" in bad["flags"]
    long = w.judge(json.dumps({"source": "Door slam", "action": " ".join(["metal door slams shut"] * 30),
                               "sound": "clang"}), k, t)
    assert long["failed"] and "long:woosh" in long["flags"]
    dig = w.judge(json.dumps({"source": "Door slam", "action": "a metal door slams 3 times", "sound": "clang"}), k, t)
    assert dig["repaired"] and dig["slots"]["action"] == "a metal door slams times"


def test_store_keeps_the_first_text_and_survives_a_reload(tmp_path):
    st = w.PromptStore.for_writer(str(tmp_path), "ab" * 20, {"model": "m"})
    assert st.put("k1", {"woosh": "first", "sa3": "first."}) and not st.put("k1", {"woosh": "second", "sa3": "x"})
    assert st.flush()
    assert os.path.basename(st.path) == "prompts_" + ("ab" * 6) + ".json"
    again = w.PromptStore.for_writer(str(tmp_path), "ab" * 20)
    assert again.get("k1")["woosh"] == "first"
    other = w.PromptStore.for_writer(str(tmp_path), "cd" * 20)
    assert other.get("k1") is None and other.path != st.path
    file_only = w.PromptStore(st.path)
    assert file_only.meta["writer_id"] == "ab" * 20 and file_only.get("k1")["woosh"] == "first"
    assert not os.path.exists(st.path + ".tmp")


def test_writer_queue_orders_fired_then_predicted_by_eta_then_manifest(tmp_path):
    pw = _writer(tmp_path)
    pw.setup()
    keys = {n: CacheKey(n, "x", "normal") for n in ("m", "p2", "p1", "s")}
    assert pw.want(keys["m"], "manifest") == "queued"
    assert pw.want(keys["p2"], "predict", 2.0) == "queued"
    assert pw.want(keys["p1"], "predict", 0.4) == "queued"
    assert pw.want(keys["s"], "seen") == "queued"
    assert pw.want(keys["m"], "seen") == "queued"
    order = [pw._pop(0.01)[1].event_type for _ in range(4)]
    assert order == ["s", "m", "p1", "p2"]
    assert pw._heap == [] or all(pw._best.get(k) is None for _r, _n, k in pw._heap)
    pw2 = _writer(tmp_path, when=("seen",))
    pw2.setup()
    assert pw2.want(keys["p1"], "predict", 0.1) is None and pw2.want(keys["s"], "seen") == "queued"


def test_server_plays_the_fallback_first_then_swaps_in_the_written_prompt(tmp_path):
    gen = PromptGen()
    pw = _writer(tmp_path)
    srv = _srv(gen, pw)
    k = CacheKey("dog_bark", "dog", "normal")
    try:
        with SfxClient(*srv.server_address) as c:
            m1, s1 = c.event("dog_bark", "dog")
            assert m1["served"] == "realtime" and s1[0] == pytest.approx(code("")) and "prompt_rev" not in m1
            assert wait_until(lambda: srv.prompt_rev.get(k) == 1)
            want = pw.lookup(k)["woosh"]
            assert want == "Dog bark sound, one dog bark happening right here, sharp click, short tail"
            m2, s2 = c.event("dog_bark", "dog")
            assert m2["served"] == "cached" and m2["prompt_rev"] == 1 and s2[0] == pytest.approx(code(want))
            m3, _ = c.event("dog_bark", "dog", have=[0])
            assert m3.get("meta_only") is not True
            m4, _ = c.event("dog_bark", "dog", have=[0], have_rev=1)
            assert m4.get("meta_only") is True and m4["prompt_rev"] == 1
    finally:
        _stop(srv)
    assert [s for s in gen.seen if s[0] == str(k)] == [(str(k), 0, ""), (str(k), 0, want)]
    assert pw.store.get(str(k))["key"] == w.key_fields(k)


def test_a_predicted_key_gets_its_prompt_before_the_first_hit_and_keeps_its_seed(tmp_path):
    gen = PromptGen()
    pw = _writer(tmp_path)
    srv = _srv(gen, pw)
    k = CacheKey("horn_honk", "car", "normal")
    try:
        pw.want(k, "manifest")
        assert wait_until(lambda: gen.prompts.get(str(k)))
        with SfxClient(*srv.server_address) as c:
            m, s = c.event("horn_honk", "car")
        assert m["served"] == "realtime" and "prompt_rev" not in m and s[0] == pytest.approx(code(gen.prompts[str(k)]))
    finally:
        _stop(srv)
    assert gen.seen == [(str(k), 0, gen.prompts[str(k)])]
    assert seed_for(k) == seed_for(CacheKey("horn_honk", "car", "normal", scene="src=player"))


def test_written_prompts_reach_each_route_in_its_own_form(tmp_path):
    gen = RoutedPromptGen()
    pw = _writer(tmp_path)
    srv = _srv(gen, pw)
    gun, step = CacheKey("shoot", "pistol", "normal"), CacheKey("footstep", "stone", "normal")
    try:
        pw.want(gun, "manifest")
        pw.want(step, "manifest")
        assert wait_until(lambda: str(gun) in gen.prompts and str(step) in gen.prompts)
    finally:
        _stop(srv)
    assert gen.prompts[str(gun)] == pw.lookup(gun)["sa3"] and gen.prompts[str(gun)].startswith("TrackType: SFX, ")
    assert gen.prompts[str(step)] == pw.lookup(step)["woosh"] and not gen.prompts[str(step)].startswith("TrackType")


def test_a_game_prompt_always_wins_over_the_writer(tmp_path):
    gen = PromptGen()
    pw = _writer(tmp_path)
    srv = _srv(gen, pw)
    a, b = CacheKey("splash", "water", "normal"), CacheKey("click", "plastic", "normal")
    try:
        with SfxClient(*srv.server_address) as c:
            c.event("splash", "water", prompt="a big splash")
            c.manifest([{"event_type": "splash", "material": "water"}])
            time.sleep(0.3)
            assert gen.prompts[str(a)] == "a big splash" and str(a) not in pw.store
            assert not any("splash" in t for t in pw.backend.calls)
            pw.want(b, "manifest")
            assert wait_until(lambda: str(b) in gen.prompts)
            c.event("click", "plastic")
            m, s = c.event("click", "plastic", prompt="a tiny click")
            assert gen.prompts[str(b)] == "a tiny click" and s[0] == pytest.approx(code("a tiny click"))
            assert m["prompt_rev"] == 1
            pw._deliver(b, pw.lookup(b))
            assert gen.prompts[str(b)] == "a tiny click"
    finally:
        _stop(srv)


def test_store_prefills_known_keys_at_startup_and_readonly_never_calls_the_model(tmp_path):
    be = FakeBackend()
    first = _writer(tmp_path, be)
    first.setup()
    k = CacheKey("ice_crack", "ice", "normal")
    rec = first.write(k)
    first.store.put(str(k), rec)
    first.store.flush()
    gen = PromptGen()
    be2 = FakeBackend()
    ro = _writer(tmp_path, be2, readonly=True)
    srv = _srv(gen, ro)
    try:
        assert wait_until(lambda: gen.prompts.get(str(k)) == rec["woosh"])
        with SfxClient(*srv.server_address) as c:
            c.event("ice_crack", "ice")
            c.event("new_sound", "ice")
        time.sleep(0.2)
    finally:
        _stop(srv)
    assert be2.calls == [] and be2.started == 0 and gen.seen[0] == (str(k), 0, rec["woosh"])
    shipped = w.build("store", store=first.store.path, log=lambda m: None)
    shipped.setup()
    assert shipped.lookup(k)["woosh"] == rec["woosh"] and shipped.backend is None


def test_next_miss_mode_drops_template_takes_and_swap_waits_for_renders_in_flight(tmp_path):
    gen = PromptGen()
    pw = _writer(tmp_path, FakeBackend(delay=0.3))
    srv = _srv(gen, pw, writer_swap="next-miss")
    k = CacheKey("thud", "wood", "normal")
    try:
        with SfxClient(*srv.server_address) as c:
            c.event("thud", "wood")
            assert wait_until(lambda: srv.prompt_rev.get(k) == 1)
            with engine_lock:
                assert k not in srv.engine.cache
            m, s = c.event("thud", "wood")
            assert m["served"] == "realtime" and s[0] == pytest.approx(code(pw.lookup(k)["woosh"]))
    finally:
        _stop(srv)
    gen2 = PromptGen(delay=0.4)
    pw2 = _writer(tmp_path / "b")
    srv2 = _srv(gen2, pw2)
    k2 = CacheKey("creak", "wood", "normal")
    try:
        with engine_lock:
            srv2.engine.inflight.add((k2, 1))
        with SfxClient(*srv2.server_address) as c:
            c.event("creak", "wood")
            assert wait_until(lambda: pw2.lookup(k2) is not None)
            time.sleep(0.3)
            assert srv2.prompt_rev.get(k2, 0) == 0 and k2 in srv2.writer_pending
            with engine_lock:
                srv2.engine.inflight.discard((k2, 1))
            srv2.deepen_wake.set()
            assert wait_until(lambda: srv2.prompt_rev.get(k2) == 1, 5.0)
    finally:
        _stop(srv2)


def test_scene_rides_on_the_wire_only_under_the_flag_and_predictions_take_the_last_scene(tmp_path):
    body = {"src": "npc", "who": "Orc_Brute", "obj": "Club", "act": "hit", "room": "open"}
    gen = PromptGen()
    srv = _srv(gen, scene=True, predict_fixes=True)
    try:
        with SfxClient(*srv.server_address) as c:
            m, _ = c.event("attack", "flesh", scene=body)
            assert m["scene_key"] == "src=npc;who=orc brute;obj=club;act=hit;room=open"
            m2, _ = c.event("attack", "flesh", scene=body, have=[0], have_scene=m["scene_key"])
            assert m2.get("meta_only") is True
            m3, _ = c.event("attack", "flesh", scene=body, have=[0])
            assert m3.get("meta_only") is not True
            c.predict([{"source": "aim", "event_type": "attack", "material": "flesh", "eta_s": 1.0},
                       {"source": "aim", "event_type": "attack", "material": "stone", "eta_s": 1.0}])
        time.sleep(0.3)
        with engine_lock:
            keys = set(srv.engine.cache) | set(srv.engine.demand) | {k for k, _v in srv.engine.inflight}
        flesh = [k for k in keys if k.material == "flesh"]
        assert flesh and all(k.scene for k in flesh)
        assert [k.scene for k in keys if k.material == "stone"] == [""]
    finally:
        _stop(srv)
    gen2 = PromptGen()
    srv2 = _srv(gen2)
    try:
        with SfxClient(*srv2.server_address) as c:
            m, _ = c.event("attack", "flesh", scene=body)
        assert "scene_key" not in m and gen2.seen[0][0] == "attack|flesh|normal|default"
    finally:
        _stop(srv2)


def _frames(srv, script):
    import socket
    out = []
    with socket.create_connection(srv.server_address, timeout=5) as s:
        for mt, body in script:
            s.sendall(wire.pack_frame(mt, body))
            rt, rb, pcm = wire.read_frame(s)
            out.append((rt, json.dumps(rb, sort_keys=True), pcm))
    return out


def test_flags_off_replies_are_byte_identical_with_or_without_the_new_fields(tmp_path):
    hello = (MsgType.HELLO, {"type": "hello", "proto": 1, "pcm": "f32le"})
    ev = {"type": "event", "event_type": "footstep", "material": "stone", "intensity": "normal",
          "context": "default"}
    sc = {"src": "player", "act": "step", "room": "small"}
    plain = [hello, (MsgType.EVENT, ev), (MsgType.EVENT, dict(ev, have=[0], have_phys="")),
             (MsgType.EVENT, dict(ev, event_type="impact", phys=PHYS))]
    extra = [hello, (MsgType.EVENT, dict(ev, scene=sc)),
             (MsgType.EVENT, dict(ev, have=[0], have_phys="", have_scene="", have_rev=0, scene=sc)),
             (MsgType.EVENT, dict(ev, event_type="impact", phys=PHYS, scene=sc))]
    runs = []
    for script, kw in ((plain, {}), (extra, {}), (plain, {"scene": True})):
        gen = MockGenerator()
        eng = AffordanceEngine(gen, realtime=True, vram_budget_kb=200_000, authored=set())
        srv = serve(engine=eng, host="127.0.0.1", port=0, warmup=False, **kw)
        try:
            runs.append(_frames(srv, script))
        finally:
            srv.shutdown()
            srv.server_close()
    assert runs[0] == runs[1] == runs[2]
    assert json.loads(runs[0][2][1]).get("meta_only") is True


def test_manifest_message_is_answered_with_or_without_a_writer(tmp_path):
    gen = PromptGen()
    pw = _writer(tmp_path)
    srv = _srv(gen, pw)
    keys = [{"event_type": "door_creak", "material": "wood"}, {"event_type": "shoot", "material": "rifle"}]
    try:
        with SfxClient(*srv.server_address) as c:
            ack = c.manifest(keys)
            assert ack == {"type": "manifest_ack", "keys": 2, "known": 0, "queued": 2, "writer": True}
            assert wait_until(lambda: len(gen.prompts) == 2)
            assert c.manifest(keys)["known"] == 2
            with pytest.raises(Exception):
                c.manifest([{"event_type": "x"}])
    finally:
        _stop(srv)
    srv2 = _srv(PromptGen())
    try:
        with SfxClient(*srv2.server_address) as c:
            assert c.manifest(keys) == {"type": "manifest_ack", "keys": 2, "known": 0, "queued": 0, "writer": False}
            assert c.stats().get("writer") is None
    finally:
        _stop(srv2)


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


def test_sidecar_writer_and_scene_flags_default_off(rs, monkeypatch, tmp_path):
    kw = _main_kwargs(rs, monkeypatch, [])
    assert not {"scene", "writer", "writer_swap"} & set(kw)
    st = w.PromptStore(str(tmp_path / "prompts_x.json"), "x" * 40)
    st.put("a|b|normal|default", {"woosh": "a", "sa3": "a."})
    st.flush()
    kw = _main_kwargs(rs, monkeypatch, ["--scene", "--writer", "store", "--writer-store", st.path,
                                        "--writer-swap", "next-miss"])
    assert kw["scene"] is True and kw["writer_swap"] == "next-miss" and kw["writer"].backend is None
    assert kw["writer_yield"] is True
    kw = _main_kwargs(rs, monkeypatch, ["--writer", "store", "--writer-store", st.path, "--no-writer-yield"])
    assert kw["writer_yield"] is False and "scene" not in kw
    with pytest.raises(SystemExit):
        _main_kwargs(rs, monkeypatch, ["--writer", "store", "--writer-store", str(tmp_path / "missing.json")])
    with pytest.raises(SystemExit):
        _main_kwargs(rs, monkeypatch, ["--writer", "llama", "--writer-model", str(tmp_path / "none.gguf"),
                                       "--writer-store", str(tmp_path)])


def test_llama_backend_runs_on_cpu_only_with_capped_threads(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 64)
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"")
    be = w.LlamaBackend(str(model), str(binary), port=11599, threads=4)
    cmd = be.command()
    for a, b in (("-ngl", "0"), ("--device", "none"), ("-t", "4"), ("-tb", "4"), ("-np", "1"), ("-c", "2048"),
                 ("--threads-http", "2"), ("--host", "127.0.0.1")):
        assert cmd[cmd.index(a) + 1] == b
    env = w.cpu_env(4)
    assert env["CUDA_VISIBLE_DEVICES"] == "-1" and env["OMP_NUM_THREADS"] == "4"
    assert be.digest.startswith("quick-sha256:") and be.digest == w.quick_digest(str(model))
    blob = tmp_path / ("sha256-" + "e" * 64)
    blob.write_bytes(b"x")
    assert w.resolve_model(str(blob)) == (str(blob), "sha256:" + "e" * 64)


def test_resolve_model_finds_an_ollama_tag(tmp_path):
    root = tmp_path / "models"
    man = root / "manifests" / "registry.ollama.ai" / "library" / "gemma4"
    man.mkdir(parents=True)
    digest = "sha256:" + "a" * 64
    (man / "e2b-it-qat").write_text(json.dumps({"layers": [
        {"mediaType": "application/vnd.ollama.image.license", "digest": "sha256:" + "b" * 64},
        {"mediaType": "application/vnd.ollama.image.model", "digest": digest}]}))
    (root / "blobs").mkdir()
    (root / "blobs" / ("sha256-" + "a" * 64)).write_bytes(b"x")
    path, got = w.resolve_model("gemma4:e2b-it-qat", [str(root)])
    assert got == digest and path.endswith("sha256-" + "a" * 64)
    with pytest.raises(FileNotFoundError):
        w.resolve_model("qwen3:0.6b", [str(root)])


class _Capture(http.server.BaseHTTPRequestHandler):
    bodies = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        _Capture.bodies.append((self.path, body))
        out = json.dumps({"choices": [{"message": {"content": '{"source": "a", "action": "b c", "sound": "d"}'}}],
                          "message": {"content": '{"source": "a", "action": "b c", "sound": "d"}'}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        out = json.dumps({"status": "ok", "version": "test"}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_backends_ask_for_deterministic_cpu_decoding(tmp_path):
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Capture)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"")
    try:
        _Capture.bodies.clear()
        msgs = w.messages(w.key_text(CacheKey("bark", "", "normal")))
        lb = w.LlamaBackend(str(model), str(binary), port=port)
        assert json.loads(lb.complete(msgs))["source"] == "a"
        ob = w.OllamaBackend("gemma4:e2b-it-qat", port=port)
        assert json.loads(ob.complete(msgs))["sound"] == "d"
    finally:
        httpd.shutdown()
    (p1, b1), (p2, b2) = _Capture.bodies
    assert p1 == "/v1/chat/completions" and b1["temperature"] == 0 and b1["top_k"] == 1 and b1["seed"] == 7
    assert b1["grammar"] == w.GBNF and b1["max_tokens"] == 96 and b1["messages"] == msgs
    assert p2 == "/api/chat" and b2["format"] == w.SCHEMA and b2["stream"] is False
    o = b2["options"]
    assert o["num_gpu"] == 0 and o["num_thread"] == 4 and o["temperature"] == 0 and o["top_k"] == 1 and o["seed"] == 7


class PausingBackend(FakeBackend):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.depth = 0
        self.log = []

    def pause(self):
        self.depth += 1
        self.log.append(("pause", self.depth))

    def resume(self):
        self.log.append(("resume", self.depth))
        self.depth -= 1


def test_writer_yields_the_cpu_while_the_sidecar_renders(tmp_path):
    gen = PromptGen()
    be = PausingBackend()
    pw = _writer(tmp_path, be)
    srv = _srv(gen, pw)
    try:
        assert isinstance(srv.engine.gen, w.Yielding) and srv.engine.gen.prompts is gen.prompts
        with SfxClient(*srv.server_address) as c:
            c.event("knock", "wood")
            c.event("knock", "wood")
        assert wait_until(lambda: srv.prompt_rev.get(CacheKey("knock", "wood", "normal")) == 1)
    finally:
        _stop(srv)
    assert be.depth == 0 and be.log[:2] == [("pause", 1), ("resume", 1)] and len(be.log) == 4
    plain = _srv(PromptGen(), _writer(tmp_path / "b", PausingBackend()), writer_yield=False)
    try:
        assert not isinstance(plain.engine.gen, w.Yielding)
    finally:
        _stop(plain)
    other = _srv(PromptGen(), _writer(tmp_path / "c"))
    try:
        assert not isinstance(other.engine.gen, w.Yielding)
    finally:
        _stop(other)


@pytest.mark.skipif(os.name != "nt", reason="windows process suspend")
def test_llama_backend_pause_nests_and_always_resumes(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    be = w.LlamaBackend(str(model), sys.executable)
    be.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.0)"], creationflags=0x08000000)
    try:
        t = time.perf_counter()
        be.pause()
        be.pause()
        be.resume()
        time.sleep(1.5)
        assert be.proc.poll() is None and be.pauses == 1
        be.resume()
        be.resume()
        assert be.proc.wait(20) == 0 and time.perf_counter() - t > 1.5
    finally:
        if be.proc.poll() is None:
            be.proc.kill()


@pytest.mark.skipif(os.name != "nt", reason="windows job objects")
def test_confine_pins_a_spawned_child_and_resumes_it():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)"], creationflags=0x4 | 0x08000000)
    try:
        job, mask = w.confine(p, 0x3)
        assert mask == 0x3
        assert p.wait(20) == 0
    finally:
        if p.poll() is None:
            p.kill()
