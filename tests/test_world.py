import hashlib
import json
import math

import pytest

from sarge import world
from sarge.generators.cached import CachedGenerator
from sarge.generators.mock import MockGenerator
from sarge.prompts import build
from sarge.types import CacheKey, seed_for

BALL = {"shape": "sphere", "ext": [0.6, 0.6, 0.6], "m": 1.0, "dyn": 1, "pm": "sarge_bouncy", "bnc": 1.0,
        "rm": "Default-Material", "name": "AutoCaptureBall_0"}
BOTTLE = {"shape": "capsule", "ext": [0.15, 0.5, 0.15], "m": 0.4, "dyn": 1, "name": "glass bottle"}
BARREL = {"shape": "capsule", "ext": [0.6, 0.9, 0.6], "m": 8.0, "dyn": 1, "name": "metal barrel"}
CRATE = {"shape": "box", "ext": [0.8, 0.8, 0.8], "m": 4.0, "dyn": 1, "name": "wooden crate"}
MARBLE = {"shape": "sphere", "ext": [0.016] * 3, "m": 0.005, "dyn": 1, "name": "glass marble", "bnc": 0.3}
TENNIS = {"shape": "sphere", "ext": [0.067] * 3, "m": 0.057, "dyn": 1, "pm": "rubber", "bnc": 0.75}
BASKET = {"shape": "sphere", "ext": [0.24] * 3, "m": 0.62, "dyn": 1, "pm": "rubber", "bnc": 0.8}
PANE = {"shape": "box", "ext": [1.6, 2.0, 0.08], "dyn": 0, "name": "window", "tag": "window"}


def floor(mat):
    return {"shape": "box", "ext": [10, 0.2, 10], "dyn": 0, "tag": mat, "pm": "sarge_bouncy", "bnc": 1.0,
            "name": "Surface_" + mat}


def phys(a, b, vn, vt=0.1):
    return {"a": a, "b": b, "vn": vn, "vt": vt, "n": [0.0, 1.0, 0.0]}


def test_bucket_is_deterministic_and_carries_the_ball_on_metal_physics():
    p = phys(BALL, floor("metal"), 12.0)
    k = world.bucket(p, "impact")
    assert k == world.bucket(json.loads(json.dumps(p)), "impact")
    assert k == "bounce;a=rubber,large,inflated,ball;b=metal,huge,static,floor;c=soft;r=short,metal"
    s = world.shape(k)
    assert s["t40_s"] == 0.2 and s["punch_scale"] == 0.25 and s["lowpass_hz"] == 400.0


@pytest.mark.parametrize("a,b,v,lo,hi", [
    (MARBLE, floor("stone"), 1.0, 25e-6, 100e-6),
    (TENNIS, floor("concrete"), 15.0, 4.6e-3 / 2, 5.4e-3 * 2),
    (BASKET, floor("wood"), 5.0, 10e-3 / 2, 15e-3 * 2),
])
def test_contact_time_is_within_2x_of_measured_values(a, b, v, lo, hi):
    tau, _model = world.contact_time(world.describe(a), world.describe(b), v)
    assert lo <= tau <= hi


def test_basketball_and_tennis_use_the_pneumatic_model_and_the_marble_hertz():
    assert world.contact_time(world.describe(BASKET), world.describe(floor("wood")))[1] == "pneumatic"
    assert world.contact_time(world.describe(MARBLE), world.describe(floor("stone")))[1] == "hertz"


@pytest.mark.parametrize("a,b", [(BALL, floor("metal")), (BOTTLE, floor("glass")), (BARREL, floor("gravel")),
                                 (CRATE, BARREL), (MARBLE, floor("stone"))])
def test_bucket_is_stable_across_a_tenfold_speed_range(a, b):
    keys = {world.bucket(phys(a, b, v), "impact") for v in (1.0, 2.0, 4.0, 7.0, 10.0)}
    assert len(keys) == 1


def test_intensity_takes_the_client_value_else_normal_speed_thresholds():
    assert world.intensity_of({"vn": 20.0}, "soft") == "soft"
    assert world.intensity_of({"vn": 1.0}) == "soft"
    assert world.intensity_of({"vn": 3.0}) == "normal"
    assert world.intensity_of({"vn": 6.0}) == "hard"


def test_material_ladder_order():
    assert world.material_of({"tag": "metal", "name": "wooden thing"}, None, 0.0) == ("metal", "tag")
    assert world.material_of({"pm": "Steel (Instance)", "name": "wooden"}, None, 0.0) == ("steel", "pm")
    assert world.material_of({"name": "WoodenCrate_03"}, None, 0.0) == ("wood", "name")
    assert world.material_of({"name": "Bottles"}, None, 0.0) == ("glass", "name")
    assert world.material_of({"name": "TinyRock"}, None, 0.0) == ("stone", "name")
    assert world.material_of({"name": "candy"}, None, 0.0) == ("", "unknown")
    assert world.material_of({"name": "thing"}, 9000.0, 0.7) == ("rubber", "bounciness")
    assert world.material_of({"name": "thing"}, 9000.0, 0.0) == ("metal", "density")
    assert world.material_of({}, None, 0.0) == ("", "unknown")


def test_inflated_hollow_and_static_builds():
    assert world.describe(BALL)["build"] == "inflated"
    assert world.describe(BOTTLE)["build"] == "hollow"
    assert world.describe(floor("metal"))["build"] == "static"
    assert world.describe(dict(CRATE, m=400.0))["build"] == "solid"


def test_ring_classes_follow_the_physics():
    dead = world.bucket(phys(BALL, floor("grass"), 5.0))
    assert dead.endswith("r=dead,rubber")
    assert world.bucket(phys(CRATE, BARREL, 4.0)).endswith("r=long,metal")
    assert ";c=sharp;" in world.bucket(phys(MARBLE, floor("stone"), 1.0))


def test_break_kind_from_event_words_or_explicit_kind():
    k = world.bucket({"a": PANE}, "break")
    assert k == world.bucket({"a": PANE, "kind": "break"}, "whatever")
    assert k.startswith("break;a=glass,") and k.endswith("r=long,glass")
    assert world.shape(k)["drive"] is None


def test_missing_or_garbage_fields_still_bucket():
    k = world.bucket({"a": {"shape": "sphere", "ext": ["x", None, 1e999], "m": "?"}})
    assert world.parse(k)["kind"] in world.KINDS
    with pytest.raises(ValueError):
        world.bucket("not a dict")
    with pytest.raises(ValueError):
        world.parse("impact|metal")


def test_slides_are_logged_and_served_as_the_same_bucket():
    hit = phys(CRATE, floor("stone"), 3.0, vt=0.1)
    slide = phys(CRATE, floor("stone"), 0.3, vt=4.0)
    assert world.analyse(slide)[1]["motion"] == "slide"
    assert world.analyse(hit)[1]["motion"] == "impact"
    assert world.bucket(hit) == world.bucket(slide)


def test_shape_numbers_per_ring_and_contact_class():
    for ring, t40 in world.T40_S.items():
        s = world.shape("hit;a=wood,small,solid,box;b=stone,huge,static,floor;c=crisp;r=%s,wood" % ring)
        assert s["t40_s"] == t40
        assert s["punch_scale"] == 0.8 and s["lowpass_hz"] is None
        assert s["drive"] == (1.0 if ring in ("dead", "short") else None)
        assert s["floor_db"] == (-45.0 if ring in ("dead", "short") else -55.0)


def test_prompt_words_name_the_real_objects():
    k = world.bucket(phys(BALL, floor("metal"), 12.0))
    p = build("impact", "metal", "hard", "default", phys=k)
    assert "rubber ball" in p and "metal floor" in p and "bouncing hard" in p
    assert "steel struck" not in p and "clang" not in p
    p = build("glass bottle hit", "gravel", "normal", "default", phys=world.bucket(phys(BOTTLE, floor("gravel"), 2)))
    assert p.startswith("Glass bottle impact, a dull thud, a glass bottle hitting the gravel")
    p = build("wooden crate hit", "gravel", "soft", "tunnel", phys=world.bucket(phys(CRATE, floor("gravel"), 1)))
    assert "wood wooden" not in p and "gently hitting" in p and "tunnel" in p
    p = build("break", "window", "normal", "indoor_small", phys=world.bucket({"a": PANE}, "break"))
    assert p.startswith("Glass pane break") and "shards" in p and "small room" in p


VOCAB = [(e, m, i, c) for e in ("footstep", "iron boot footstep", "land", "iron boot landing", "jump", "shoot",
                                "reload", "empty_click", "equip", "take off", "holster", "bullet impact", "break",
                                "impact", "glass bottle hit", "metal barrel hit", "wooden crate hit", "splash",
                                "punch", "kick", "click", "drop")
         for m in ("grass", "stone", "metal", "wood", "glass", "gravel", "water", "window", "rifle", "pistol",
                   "iron boots", "sarge_bouncy", "carpet")
         for i in ("soft", "normal", "hard")
         for c in ("default", "arena", "indoor_small", "indoor_large", "tunnel")]


def test_empty_phys_leaves_prompt_key_string_seed_and_cache_path_unchanged(tmp_path):
    cached = CachedGenerator(MockGenerator(), str(tmp_path))
    for e, m, i, c in VOCAB[::7]:
        k = CacheKey(e, m, i, c)
        k2 = CacheKey(e, m, i, c, "")
        assert build(e, m, i, c) == build(e, m, i, c, phys="")
        assert str(k2) == "%s|%s|%s|%s" % (e, m, i, c)
        assert seed_for(k2, 3) == seed_for(k, 3)
        assert k2 == k and hash(k2) == hash(k)
        assert cached._path(k2, 1) == cached._path(k, 1)


def test_head_seed_value_is_pinned():
    k = CacheKey("footstep", "grass", "normal", "default")
    base = int(hashlib.md5(b"footstep|grass|normal|default").hexdigest()[:8], 16)
    assert seed_for(k, 0) == base & 0x7FFFFFFF


def test_phys_key_is_a_distinct_cache_entry():
    a = CacheKey("impact", "metal", "hard")
    b = CacheKey("impact", "metal", "hard", "default", "bounce;a=rubber,large,inflated,ball;b=metal,huge,static,"
                 "floor;c=soft;r=short,metal")
    assert a != b and str(b).startswith(str(a) + "|bounce;")
    assert seed_for(a) != seed_for(b)


def test_replay_of_a_demo_session_stays_small_and_does_not_flicker():
    import random
    rng = random.Random(0)
    floors = ["grass", "stone", "metal", "wood", "glass", "gravel"]
    pairs = [(BALL, floor(f), "impact", f) for f in floors[:4]] + [
        (BOTTLE, floor("glass"), "glass bottle hit", "glass"), (BOTTLE, floor("gravel"), "glass bottle hit", "gravel"),
        (BARREL, floor("gravel"), "metal barrel hit", "gravel"), (CRATE, floor("gravel"), "wooden crate hit", "gravel"),
        (CRATE, BARREL, "wooden crate hit", "sarge_bouncy"), (CRATE, CRATE, "wooden crate hit", "sarge_bouncy"),
        (BALL, BALL, "impact", "sarge_bouncy")]
    keys, flips, steps = set(), 0, 0
    for a, b, ev, mat in pairs:
        prev = None
        for _ in range(40):
            v = rng.uniform(0.7, 25.0)
            p = phys(a, b, v, vt=rng.uniform(0.0, 3.0))
            k = world.bucket(p, ev)
            keys.add((ev, mat, world.intensity_of(p), k))
            if prev is not None:
                steps += 1
                flips += k != prev
            prev = k
    assert len(keys) <= 33
    assert flips <= 0.02 * steps


def test_prompt_never_mentions_a_bare_object_word_twice():
    for a, b in ((BALL, floor("metal")), (BOTTLE, BARREL), (CRATE, CRATE), (MARBLE, floor("stone"))):
        p = build("impact", "x", "normal", "default", phys=world.bucket(phys(a, b, 3.0)))
        words = p.replace(",", "").split()
        assert all(words[j] != words[j + 1] for j in range(len(words) - 1))


def test_contact_time_is_finite_for_all_static_pairs():
    tau, _ = world.contact_time(world.describe(PANE), world.describe(floor("stone")))
    assert math.isfinite(tau) and tau > 0
