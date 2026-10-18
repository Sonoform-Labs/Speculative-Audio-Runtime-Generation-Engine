import os
import sys
import types

import numpy as np
import pytest

from sarge import prompts
from sarge.affordance import AffordanceEngine
from sarge.generators.mock import MockGenerator
from sarge.generators.remote import RemoteGenerator
from sarge.generators.routed import RoutedGenerator, side_for, weapon_words
from sarge.sidecar import SfxClient, serve
from sarge.sidecar.server import _alias, engine_lock
from sarge.types import AudioClip, CacheKey, seed_for
from sarge.words import names, split_words

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
PROBE = ["campfire", "camera_snapshot", "magic_missile", "ice_crack", "peacock_call", "keyboard_click", "pistol",
         "cannon", "laser_zap", "rocket_launch", "gunshot"]


def test_split_words_handles_snake_camel_digits_and_case():
    assert split_words("Footstep_Concrete") == ["footstep", "concrete"]
    assert split_words("PlayerFootstep") == ["player", "footstep"]
    assert split_words("HTTPServer") == ["http", "server"]
    assert split_words("Fire1") == ["fire", "1"]
    assert split_words("M4A1") == ["m", "4", "a", "1"]
    assert split_words("iron boot  footstep") == ["iron", "boot", "footstep"]
    assert split_words(None) == [] and split_words("") == []
    assert names(["land"], split_words("IronBootLanding")) and not names(["land"], ["landmine"])
    assert names(["bullet", "impact"], split_words("BigBulletImpacts"))


def test_generic_prompt_is_built_from_the_request_alone():
    g = prompts.generic
    assert g("footstep", "gravel") == "footstep, gravel"
    assert g("grappling_hook", "rope", "hard", "indoor_small") == "grappling hook, rope, hard, indoor small"
    assert g("dog_bark", "dog") == "dog bark"
    assert g("DoorCreak", "Wood", "normal", "WalkArena") == "door creak, wood, walk arena"
    assert g("footstep", "ice", "normal", "cave") == "footstep, ice, cave"
    assert g("break", "window", "normal", "default") == "break, window"
    assert g("shoot", "pistol", prefix=prompts.SA3_PREFIX) == "TrackType: SFX, shoot, pistol"
    assert g("", "", prefix=prompts.SA3_PREFIX) == "TrackType: SFX"
    assert g("impact", "psfx_bouncy") == "impact, psfx bouncy"


def test_generic_prompt_never_adds_words_the_request_lacks():
    req = [("lightsaber", "plasma", "hard", "spaceship"), ("explosion", "grenade", "soft", "tunnel"),
           ("splash", "lava", "normal", "default"), ("equip", "rain boots", "normal", "default"),
           ("kick", "door", "normal", "default"), ("shoot", "rifle", "loud", "outdoor_open"),
           ("footstep", "", "normal", "default")]
    for ev, mat, inten, ctx in req:
        out = prompts.generic(ev, mat, inten, ctx)
        allowed = set(split_words(" ".join((ev, mat, inten, ctx))))
        assert set(split_words(out)) <= allowed, (ev, mat, out)
        assert "knock" not in out and "outdoors" not in out and "exterior" not in out


def test_generic_prompt_adds_only_the_validated_contact_words_with_world():
    phys = "bounce;a=rubber,large,inflated,ball;b=metal,huge,static,floor;c=soft;r=short,metal"
    from sarge.world import CONTACT_WORDS
    assert prompts.generic("impact", "metal", phys=phys) == "impact, metal, " + CONTACT_WORDS["soft"]
    brk = "break;a=glass,medium,static,pane;r=ringing,glass"
    assert prompts.generic("break", "window", phys=brk) == "break, window"
    assert prompts.contact_words("not a bucket") == ""


def test_tuned_style_is_the_old_phrasebook_exactly():
    for k in (CacheKey("shoot", "pistol", "normal", "tunnel"), CacheKey("footstep", "gravel", "soft"),
              CacheKey("impact", "metal", "hard", "default", "hit;a=rubber,small,solid,ball;b=metal,huge,static,floor;c=crisp;r=short,metal")):
        assert prompts.compose("tuned", k) == prompts.build(k.event_type, k.material, k.intensity, k.context, k.phys)


def test_sa3_generic_style_uses_the_vendor_prefix_and_a_game_prompt_wins():
    from sarge.generators.stable_audio3 import StableAudio3Generator
    g = StableAudio3Generator(prompt_style="generic")
    k = CacheKey("Reload", "AK47", "hard", "tunnel")
    assert g._prompt(k) == "TrackType: SFX, reload, ak 47, hard, tunnel"
    g.prompts[str(k)] = "a game prompt"
    assert g._prompt(k) == "a game prompt"
    assert StableAudio3Generator(prompt_style="detailed")._prompt(CacheKey("shoot", "pistol", "normal")) == \
        prompts.build("shoot", "pistol")


@pytest.fixture
def ws(monkeypatch):
    if SCRIPTS not in sys.path:
        monkeypatch.syspath_prepend(SCRIPTS)
    import woosh_server
    return woosh_server


def test_woosh_prompt_follows_the_style_and_a_sent_prompt_wins(ws):
    k = CacheKey("dog_bark", "dog", "normal", "yard")
    assert ws.prompt_for(k) == prompts.build("dog_bark", "dog", "normal", "yard")
    assert ws.prompt_for(k, "generic") == "dog bark, yard"
    assert ws.prompt_for(k, "generic", "a big dog barking twice") == "a big dog barking twice"
    assert ws.prompt_for(k, "tuned", "  ") == prompts.build("dog_bark", "dog", "normal", "yard")
    calls = []

    def gen(prompt, seed, frames=101):
        calls.append(prompt)
        return np.full((ws.SR, 1), 0.2, dtype=np.float32)

    ws.render_key(gen, k, 0, ws.parse_frames(None), "generic", "custom")
    ws.render_key(gen, k, 0, ws.parse_frames(None), "generic")
    assert calls == ["custom", "dog bark, yard"]


def test_remote_generator_sends_a_registered_prompt():
    r = RemoteGenerator()
    sent = []
    r._ensure = lambda: None
    r._roundtrip = lambda req, pcm=b"": (sent.append(dict(req)), ({"sample_rate": 48000, "channels": 1},
                                                                    np.zeros(4, "<f4").tobytes()))[1]
    k = CacheKey("splash", "water", "normal")
    r.generate(k)
    r.prompts[str(k)] = "a heavy rock dropped into a pond"
    r.generate(k)
    assert "prompt" not in sent[0] and sent[1]["prompt"] == "a heavy rock dropped into a pond"


class Fake:
    def __init__(self, tag):
        self.tag = tag
        self.calls = []
        self.sample_rate = 44100
        self.prompts = {}

    def generate(self, key, *, variant=0):
        self.calls.append((str(key), self.prompts.get(str(key), "")))
        return AudioClip(np.full(10, 0.1 if self.tag == "guns" else 0.2, dtype=np.float32), 44100, variant_id=variant)

    def generate_batch(self, items):
        return [self.generate(k, variant=v) for k, v in items]


def test_routed_generator_shares_one_prompt_table_with_both_routes():
    g, r = Fake("guns"), Fake("rest")
    gen = RoutedGenerator(guns=g, rest=r, warm_guns=False)
    assert gen.prompts is g.prompts is r.prompts
    gen.prompts[str(CacheKey("break", "window", "normal"))] = "game window prompt"
    gen.generate(CacheKey("break", "window", "normal"))
    assert r.calls == [("break|window|normal|default", "game window prompt")]


def _routes(name, material="", cls=""):
    k = CacheKey(name, material, "normal", cls=cls)
    return {m: side_for(k, m) for m in ("families", "words", "class", "single")} | {
        "single sa3": side_for(k, "single", single="guns")}


def test_route_modes_on_the_audit_names():
    table = {n: _routes(n) for n in PROBE}
    assert [n for n in PROBE if table[n]["families"] == "guns"] == [
        "campfire", "camera_snapshot", "magic_missile", "ice_crack", "peacock_call", "keyboard_click", "gunshot"]
    assert [n for n in PROBE if table[n]["words"] == "guns"] == ["pistol", "cannon", "gunshot"]
    assert all(table[n]["class"] == "rest" for n in PROBE)
    assert all(table[n]["single"] == "rest" and table[n]["single sa3"] == "guns" for n in PROBE)
    assert side_for(CacheKey("laser_zap", "", "normal", cls="Weapon".lower()), "class") == "guns"
    assert side_for(CacheKey("campfire", "", "normal", cls="ambience"), "class") == "rest"


def test_word_route_keeps_the_demo_gun_keys_on_the_gun_model():
    guns = [("shoot", "pistol"), ("shoot", "rifle"), ("reload", "rifle"), ("empty_click", "pistol"),
            ("equip", "rifle"), ("WeaponFire", "AK47"), ("Fire1", "")]
    rest = [("break", "window"), ("bullet impact", "glass"), ("footstep", "gravel"), ("splash", "water"),
            ("punch", "body"), ("equip", "iron boots"), ("take off", "iron boots"), ("shotgun_shell_drop", "")]
    assert all(weapon_words(CacheKey(e, m, "normal")) for e, m in guns)
    assert [(e, m) for e, m in rest if weapon_words(CacheKey(e, m, "normal"))] == [("shotgun_shell_drop", "")]


def test_routed_generator_modes_and_costs():
    for mode, want in (("families", ("guns", "rest")), ("words", ("guns", "rest")), ("class", ("rest", "rest")),
                       ("single", ("rest", "rest"))):
        gen = RoutedGenerator(guns=Fake("guns"), rest=Fake("rest"), warm_guns=False, route=mode, gun_cost_ms=150.0,
                              rest_cost_ms=40.0)
        got = (gen.route(CacheKey("shoot", "pistol", "normal")), gen.route(CacheKey("break", "window", "normal")))
        assert got == want, mode
        assert gen.cost_ms(CacheKey("shoot", "pistol", "normal")) == (150.0 if got[0] == "guns" else 40.0)
    gen = RoutedGenerator(guns=Fake("guns"), rest=Fake("rest"), warm_guns=False, route="class")
    assert gen.route(CacheKey("shoot", "pistol", "normal", cls="weapon")) == "guns"
    assert RoutedGenerator(guns=Fake("g"), rest=Fake("r"), route="single", single_route="guns").route(
        CacheKey("splash", "water", "normal")) == "guns"
    with pytest.raises(ValueError):
        RoutedGenerator(guns=Fake("g"), rest=Fake("r"), route="nope")
    assert RoutedGenerator(guns=Fake("g"), rest=Fake("r")).post_tag == ""
    assert RoutedGenerator(guns=Fake("g"), rest=Fake("r"), route="words").post_tag == "|route:words"


def test_class_never_moves_the_seed_and_shows_in_the_key():
    a, b = CacheKey("shoot", "pistol", "normal"), CacheKey("shoot", "pistol", "normal", cls="weapon")
    assert str(b) == "shoot|pistol|normal|default|class=weapon" and a != b
    assert seed_for(a, 3) == seed_for(b, 3)
    p = CacheKey("impact", "metal", "hard", "default", "hit;a=x", "weapon")
    assert str(p).endswith("|hit;a=x|class=weapon") and seed_for(p) == seed_for(CacheKey("impact", "metal", "hard",
                                                                                           "default", "hit;a=x"))


class PromptGen(MockGenerator):
    def __init__(self):
        super().__init__()
        self.prompts = {}
        self.seen = []

    def generate(self, key, *, variant=0):
        self.seen.append((str(key), self.prompts.get(str(key), "")))
        return super().generate(key, variant=variant)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _srv(gen=None, **kw):
    eng = AffordanceEngine(gen or PromptGen(), realtime=True, vram_budget_kb=200_000, authored=set())
    return serve(engine=eng, host="127.0.0.1", port=0, warmup=False, **kw)


def _stop(srv):
    srv.shutdown()
    srv.server_close()


def _queued(srv):
    with engine_lock:
        return set(srv.engine.demand) | set(srv.engine.cache) | {k for k, _v in srv.engine.inflight}


def test_a_prompt_on_the_event_reaches_the_generator_and_a_new_one_rerenders():
    gen = PromptGen()
    srv = _srv(gen)
    try:
        with SfxClient(*srv.server_address) as c:
            c.event("dog_bark", "dog")
            m1, _ = c.event("dog_bark", "dog", prompt="a small dog yapping")
            m2, _ = c.event("dog_bark", "dog", prompt="a small dog yapping")
            m3, _ = c.event("dog_bark", "dog", prompt="a big dog growling")
            m4, _ = c.event("dog_bark", "dog")
    finally:
        _stop(srv)
    assert [p for _k, p in gen.seen] == ["", "a small dog yapping", "a big dog growling"]
    assert [m["served"] for m in (m1, m2, m3, m4)] == ["realtime", "cached", "realtime", "cached"]


def test_prebake_and_event_prompts_reach_the_woosh_route():
    g, r = Fake("guns"), RemoteGenerator()
    sent = []
    r._ensure = lambda: None
    r._roundtrip = lambda req, pcm=b"": (sent.append(dict(req)), ({"sample_rate": 48000, "channels": 1},
                                                                    np.zeros(8, "<f4").tobytes()))[1]
    gen = RoutedGenerator(guns=g, rest=r, warm_guns=False)
    eng = AffordanceEngine(gen, realtime=True, vram_budget_kb=200_000, authored=set(), variants_per_key=1)
    srv = serve(engine=eng, host="127.0.0.1", port=0, warmup=False)
    try:
        with SfxClient(*srv.server_address) as c:
            c.prewarm([{"event_type": "splash", "material": "water", "prompt": "pebble into a pond"}])
            c.event("door_creak", "wood", prompt="old door creaking open slowly")
            c.event("shoot", "pistol", prompt="a suppressed pistol shot")
    finally:
        _stop(srv)
    assert {(q["event_type"], q.get("prompt")) for q in sent} == {("splash", "pebble into a pond"),
                                                                   ("door_creak", "old door creaking open slowly")}
    assert g.calls == [("shoot|pistol|normal|default", "a suppressed pistol shot")]


def test_class_reaches_the_key_only_when_the_server_reads_it():
    for read in (False, True):
        gen = PromptGen()
        srv = _srv(gen, read_class=read)
        try:
            with SfxClient(*srv.server_address) as c:
                c.event("laser_zap", "", cls="Weapon")
        finally:
            _stop(srv)
        assert gen.seen[0][0] == ("laser_zap||normal|default|class=weapon" if read else "laser_zap||normal|default")


def test_predictions_take_the_last_fired_class_of_their_event():
    srv = _srv(read_class=True, predict_fixes=True)
    try:
        with SfxClient(*srv.server_address) as c:
            c.event("Reload", "AK47", cls="weapon")
            c.predict([{"source": "equip", "event_type": "Reload", "material": "AK47", "eta_s": 2.0}])
            c.predict([{"source": "equip", "event_type": "shoot", "material": "AK47", "eta_s": 2.0, "class": "weapon"}])
        q = _queued(srv)
    finally:
        _stop(srv)
    assert CacheKey("Reload", "AK47", "normal", cls="weapon") in q
    assert CacheKey("shoot", "AK47", "normal", cls="weapon") in q


def test_alias_matches_split_words_in_any_case():
    s = types.SimpleNamespace(fired_at={"Footstep_Concrete": 3, "PlayerFootstep": 2, "landmine": 9})
    assert _alias(s, "footstep") == "Footstep_Concrete"
    assert _alias(types.SimpleNamespace(fired_at={"PlayerFootstep": 2}), "footstep") == "PlayerFootstep"
    assert _alias(types.SimpleNamespace(fired_at={"IronBootLanding": 1, "landmine": 4}), "land") == "IronBootLanding"
    assert _alias(types.SimpleNamespace(fired_at={"landmine": 4}), "land") == "land"
    assert _alias(types.SimpleNamespace(fired_at={"bullet impact": 1}), "Bullet_Impact") == "bullet impact"


def test_motion_prediction_without_a_name_takes_the_event_that_lands_on_its_surface():
    clock = Clock()
    srv = _srv(predict_fixes=True, clock=clock)
    try:
        with SfxClient(*srv.server_address) as c:
            ack = c.predict([{"source": "motion", "event_type": "", "material": "Grass", "eta_s": 0.6}])
            assert ack["queued"] == 0
            clock.t += 0.5
            c.event("Footstep_Grass", "Grass")
            clock.t += 0.2
            c.predict([{"source": "motion", "event_type": "", "material": "Stone", "eta_s": 0.6}])
        q = _queued(srv)
    finally:
        _stop(srv)
    assert CacheKey("Footstep_Grass", "Stone", "normal") in q
    assert not any(k.event_type == "" for k in q)


def _walk(c, clock, n, surface, step="PlayerStep"):
    for _ in range(n):
        c.predict([{"source": "motion", "event_type": "", "material": surface, "eta_s": 0.6}])
        clock.t += 0.4
        c.event(step, surface)


def test_aim_prediction_without_a_name_learns_the_impact_while_steps_stay_with_motion():
    clock = Clock()
    srv = _srv(predict_fixes=True, clock=clock)
    try:
        with SfxClient(*srv.server_address) as c:
            c.predict([{"source": "aim", "event_type": "", "material": "wood", "eta_s": 0.5}])
            _walk(c, clock, 4, "wood")
            clock.t += 1.0
            c.event("WeaponFire", "rifle")
            c.event("BulletHit", "wood")
            clock.t += 1.0
            _walk(c, clock, 4, "wood")
            clock.t += 0.3
            c.predict([{"source": "aim", "event_type": "", "material": "metal", "eta_s": 0.5}])
            c.predict([{"source": "motion", "event_type": "", "material": "stone", "eta_s": 0.6}])
        q = _queued(srv)
        aim = dict(srv.kind_score["aim"])
    finally:
        _stop(srv)
    assert CacheKey("BulletHit", "metal", "normal") in q and CacheKey("PlayerStep", "stone", "normal") in q
    assert set(aim) == {"BulletHit"}
    assert not any(k.event_type == "PlayerStep" and k.material == "metal" for k in q)


def test_a_rapid_name_never_confirms_a_source_other_than_motion():
    clock = Clock()
    srv = _srv(predict_fixes=True, clock=clock)
    try:
        with SfxClient(*srv.server_address) as c:
            c.predict([{"source": "aim", "event_type": "", "material": "metal", "eta_s": 0.5}])
            c.event("Hit", "metal")
            for _ in range(5):
                clock.t += 0.1
                c.event("Spark", "metal")
        score = dict(srv.kind_score["aim"])
    finally:
        _stop(srv)
    assert score == {"Hit": pytest.approx(0.8), "Spark": 1.0}


def test_after_expands_only_to_events_that_landed_on_two_aimed_surfaces():
    clock = Clock()
    srv = _srv(predict_fixes=True, clock=clock)

    def aim(c, mat, **kw):
        c.predict([dict({"source": "aim", "event_type": "BulletHit", "material": mat, "eta_s": 0.5,
                         "after": "shoot"}, **kw)])

    try:
        with SfxClient(*srv.server_address) as c:
            aim(c, "metal")
            for _ in range(3):
                clock.t += 1.0
                c.event("shoot", "metal")
                c.event("BulletHit", "metal")
            clock.t += 1.0
            c.event("shoot", "metal")
            c.event("reload", "metal")
            clock.t += 1.0
            aim(c, "wood")
            q1 = _queued(srv)
            clock.t += 1.0
            c.event("shoot", "metal")
            c.event("BulletHit", "wood")
            clock.t += 1.0
            aim(c, "glass")
        q2 = _queued(srv)
        targets = {e: len(t) for e, t in srv.kind_targets["aim"].items()}
    finally:
        _stop(srv)
    assert CacheKey("BulletHit", "wood", "normal") in q1
    assert not any(k.event_type == "reload" and k.material in ("wood", "glass") for k in q2)
    assert CacheKey("BulletHit", "glass", "normal") in q2
    assert "shoot" not in targets and targets["BulletHit"] == 2 and targets["reload"] == 1


def test_without_predict_fixes_an_unnamed_prediction_still_resolves_or_drops():
    clock = Clock()
    srv = _srv(clock=clock)
    try:
        with SfxClient(*srv.server_address) as c:
            assert c.predict([{"source": "motion", "event_type": "", "material": "sand", "eta_s": 0.6}])["queued"] == 0
            clock.t += 0.5
            c.event("step", "sand")
            assert c.predict([{"source": "motion", "event_type": "", "material": "mud", "eta_s": 0.6}])["queued"] == 1
        assert CacheKey("step", "mud", "normal") in _queued(srv)
    finally:
        _stop(srv)


def test_motion_eta_wins_for_any_name():
    from sarge.fusion import PredictionFusion
    f = PredictionFusion(owner=lambda source, key: source == "motion")
    k = CacheKey("Footstep_Concrete", "concrete", "normal")
    out = f.fuse([("intent", k, 0.2, 0.5), ("motion", k, 0.6, 0.9)])
    assert out[0].eta_s == 0.6
    assert PredictionFusion(owner={"footstep": "motion"}).fuse([("intent", k, 0.2, 0.5), ("motion", k, 0.6, 0.9)])[0].eta_s == 0.2


def _burst(sr, tau, attack_s=0.0005, tone=None, seed=1):
    t = np.arange(int(1.5 * sr)) / sr
    rng = np.random.default_rng(seed)
    src = rng.standard_normal(t.size) if tone is None else np.sin(2 * np.pi * tone * t)
    return (0.5 * src * np.minimum(1.0, t / attack_s) * np.exp(-t / tau)).astype(np.float32)


def test_auto_amp_reads_the_audio_and_ignores_the_event_name():
    from sarge.amp import amplify, auto_params
    sr = 48000
    x = _burst(sr, 0.15)
    a = amplify(x, sr, "shoot", "normal", auto=True)
    b = amplify(x, sr, "footstep", "normal", auto=True)
    c = amplify(x, sr, "grappling_hook", "normal", auto=True)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    crack = auto_params(_burst(sr, 0.15), sr)
    swell = auto_params(_burst(sr, 0.4, attack_s=0.2), sr)
    ring = auto_params(_burst(sr, 0.3, tone=880.0), sr)
    assert crack["punch_db"] > 5.0 and swell["punch_db"] < 3.0
    assert crack["drive"] > 1.8 and ring["drive"] < 1.2
    assert crack["noisy"] > 0.9 and ring["noisy"] < 0.1


def test_auto_saturation_never_lengthens_the_decay():
    from sarge.amp import amplify, gated_drive
    from sarge.post import decay_time
    sr = 48000
    for tau in (0.03, 0.15, 0.6):
        x = _burst(sr, tau).astype(np.float64)
        y = gated_drive(x, sr, 2.4)
        assert decay_time(y, sr) <= decay_time(x, sr) * 1.02 + 0.002
        plain = np.tanh(2.4 * x / np.max(np.abs(x))) / np.tanh(2.4) * np.max(np.abs(x))
        assert decay_time(plain, sr) > decay_time(x, sr)
        assert np.sqrt(np.mean(y[:int(0.02 * sr)] ** 2)) > np.sqrt(np.mean(x[:int(0.02 * sr)] ** 2))
    z = amplify(_burst(sr, 0.15), sr, "x", "normal", auto=True)
    assert decay_time(z, sr) <= decay_time(amplify(_burst(sr, 0.15), sr, "x", "normal", auto=True, drive=False), sr) \
        * 1.02 + 0.002


def test_auto_width_follows_the_decay():
    from sarge.amp import AUTO, auto_width
    sr = 48000
    lo, hi = AUTO["width"]
    assert auto_width(_burst(sr, 0.01), sr) == pytest.approx(lo)
    assert auto_width(_burst(sr, 0.5), sr) == pytest.approx(hi)
    from sarge.stereo import widen
    x = _burst(sr, 0.5)
    np.testing.assert_array_equal(widen(x, sr, "footstep", auto=True), widen(x, sr, "shoot", auto=True))
    assert not np.array_equal(widen(x, sr, "footstep"), widen(x, sr, "shoot"))


def test_auto_mode_through_both_finishing_chains():
    from sarge.post import process
    sr = 48000
    x = _burst(sr, 0.2)
    p = process(x, sr, "anything", "normal", amp_mode="auto", stereo=True)
    q = process(x, sr, "other", "normal", amp_mode="auto", stereo=True)
    np.testing.assert_array_equal(p, q)
    assert p.ndim == 2 and np.max(np.abs(p)) == pytest.approx(10 ** (-1 / 20), abs=1e-3)


def test_woosh_finish_auto_ignores_the_event(ws):
    x = _burst(ws.SR, 0.2)[:, None]
    a = ws.finish(x, "shoot", "normal", "auto", True)
    b = ws.finish(x, "splash", "normal", "auto", True)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(ws.finish(x, "shoot", "normal", "full", True), ws.finish(x, "splash", "normal", "full", True))


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


def test_sidecar_new_flags_default_off(rs, monkeypatch, tmp_path):
    kw = _hybrid_kwargs(rs, monkeypatch, tmp_path, ["--amp", "--detailed-prompts"])
    g = kw["gen_kwargs"]
    assert g["prompt_style"] == "detailed" and g["amp"] is True and "route" not in g
    assert "read_class" not in kw and kw["rapid_hz"] == 1.5
    kw = _hybrid_kwargs(rs, monkeypatch, tmp_path, ["--prompts", "generic", "--route", "words", "--amp", "auto"])
    g = kw["gen_kwargs"]
    assert g["prompt_style"] == "generic" and g["route"] == "words" and g["amp"] == "auto"
    kw = _hybrid_kwargs(rs, monkeypatch, tmp_path, ["--route", "class", "--gun-class", "weapon,firearm"])
    assert kw["read_class"] is True and kw["gen_kwargs"]["gun_classes"] == ("weapon", "firearm")
    kw = _hybrid_kwargs(rs, monkeypatch, tmp_path, ["--route", "single", "--single-model", "sa3", "--rapid-hz", "0"])
    assert kw["gen_kwargs"]["single_route"] == "guns" and kw["rapid_hz"] == 0.0


def test_start_woosh_passes_generic_prompts_and_auto_amp_only_when_set(rs, monkeypatch, tmp_path):
    py = tmp_path / "python.exe"
    py.write_bytes(b"")
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
                woosh_frames=None, level=None)
    rs.start_woosh(types.SimpleNamespace(amp="full", prompts="tuned", **base))
    rs.start_woosh(types.SimpleNamespace(amp="auto", prompts="generic", **base))
    assert cmds[0][-2:] == ["--amp", "--stereo"] and "--prompts" not in cmds[0]
    assert cmds[1][-5:] == ["--amp", "auto", "--stereo", "--prompts", "generic"]
