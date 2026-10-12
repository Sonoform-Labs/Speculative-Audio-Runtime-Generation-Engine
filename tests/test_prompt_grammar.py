from sarge.prompts import build, space_of


def test_measured_winners_come_out_verbatim():
    assert build("break", "window") == (
        "Close perspective glass break, a window pane smashed with a hard crack, dense shatter of glass, "
        "high-pitched tinkle and chime of shards settling on paving stones outdoors")
    assert build("shoot", "pistol").startswith("Gunshot, 9mm pistol, single shot, sharp cracking report")
    assert "two quick metallic clicks" in build("shoot", "rifle")


def test_space_comes_from_the_request_context():
    assert space_of("default") == "outdoor_open" and space_of("tunnel") == "tunnel" and space_of("room") == "indoor_small"
    out = build("shoot", "pistol", "normal", "default")
    tun = build("shoot", "pistol", "normal", "tunnel")
    assert out != tun and "tunnel" in tun and "open field" in out


def test_every_unity_key_gets_a_detailed_prompt():
    keys = [("footstep", "gravel"), ("footstep", "water"), ("land", "stone"), ("jump", "wood"), ("reload", "rifle"),
            ("empty_click", "pistol"), ("equip", "iron boots"), ("bullet impact", "metal barrel"),
            ("glass bottle hit", "stone"), ("unknown thing", "plastic")]
    for ev, mat in keys:
        p = build(ev, mat)
        assert len(p.split()) >= 10 and " on pistol," not in p


def test_generator_uses_the_grammar_when_asked():
    from sarge.generators.stable_audio3 import StableAudio3Generator
    from sarge.types import CacheKey
    g = StableAudio3Generator(prompt_style="detailed")
    k = CacheKey("break", "window", "normal", "tunnel")
    assert g._prompt(k) == build("break", "window", "normal", "tunnel")
    assert StableAudio3Generator()._prompt(k).startswith("a break on window")
