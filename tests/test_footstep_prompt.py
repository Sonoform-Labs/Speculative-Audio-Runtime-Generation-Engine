from sarge.generators.stable_audio import build_prompt


def test_footstep_prompt_always_names_the_intensity():
    assert build_prompt("footstep", "metal", "normal") == (
        "a normal metallic footstep on metal, a clear distinct footstep, dry, isolated sound effect")


def test_footstep_prompt_keeps_other_intensities():
    assert build_prompt("footstep", "wood", "heavy").startswith("a heavy wooden footstep on wood")


def test_footstep_prompt_defaults_missing_intensity_to_normal():
    assert build_prompt("footstep", "stone", "").startswith("a normal stone footstep on stone")


def test_impact_prompts_unchanged():
    assert build_prompt("impact", "glass", "normal") == (
        "a impact on glass, sharp shattering glass shards, dry, isolated sound effect")
