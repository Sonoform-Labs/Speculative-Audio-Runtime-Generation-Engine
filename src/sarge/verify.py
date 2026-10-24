import re

from .prompts import CLEAN_MOD, LOOP_MOD, template, template_mods

CAPTION = "template:caption"
CAND_STRIDE = 4096
SCORES = ("request", "event-min")


def _strip(t):
    t = re.sub(r"^A (single|continuous) ", "", t)
    t = re.sub(r", recorded up close", "", t)
    return t.strip().rstrip(".").strip()


def request_text(event, material="", intensity="normal", context="default", phys="", canon=None, scene="",
                 loop=False, clean=False):
    style = CAPTION + (LOOP_MOD if loop else "") + (CLEAN_MOD if clean else "")
    return _strip(template(style, event, material, intensity, context, phys, "woosh", scene, canon))


def key_text(key, canon=None, loop=False, clean=False):
    return request_text(key.event_type, key.material, key.intensity, key.context, key.phys, canon,
                        getattr(key, "scene", ""), loop, clean)


def event_text(key, canon=None, loop=False, clean=False):
    c = dict(canon, material="") if isinstance(canon, dict) else None
    return request_text(key.event_type, "", key.intensity, key.context, key.phys, c, getattr(key, "scene", ""), loop,
                        clean)


def style_texts(style):
    mods = template_mods(style)
    return {"loop": "loop" in mods, "clean": "clean" in mods}


def candidate_variant(variant, j):
    return int(variant) + CAND_STRIDE * int(j)


def pick(scored):
    return max(scored, key=lambda x: (x[0], -x[1]))


def choose(means):
    return max(range(len(means)), key=lambda i: (means[i], -i))


def choose_guarded(means):
    return max(range(len(means)), key=lambda i: (means[i], i > 0, -i))


def guard_texts(style):
    return dict(style_texts(style) if style else {"loop": False}, clean=True)
