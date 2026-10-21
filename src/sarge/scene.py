from __future__ import annotations

import re

SOURCES = ("player", "npc", "physics", "projectile", "world", "ui")
ACTS = ("hit", "bounce", "break", "slide", "roll", "step", "use", "loop")
ROOMS = ("open", "small", "medium", "large", "corridor")
MAX_CHARS = 48
MAX_WORDS = 3
BOILER = frozenset(("cube", "sphere", "capsule", "cylinder", "plane", "quad", "clone", "instance", "prefab", "lod",
                    "mesh", "variant", "gameobject", "object", "geo", "sm", "fx", "sfx", "prop", "go", "new", "a", "b",
                    "l", "r"))
COLOURS = frozenset(("red", "blue", "green", "yellow", "black", "white", "orange", "purple", "violet", "pink",
                     "grey", "gray", "brown", "cyan", "magenta", "teal", "crimson", "scarlet", "golden"))
FIELDS = ("src", "who", "obj", "on", "act", "room")


def name_words(text, n=MAX_WORDS):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text or "")[:MAX_CHARS])
    ws = [w for w in re.split(r"[^a-z]+", s.lower()) if w and w not in BOILER and w not in COLOURS]
    return " ".join(ws[:n])


def _enum(value, allowed):
    v = str(value or "").strip().lower()
    return v if v in allowed else ""


def physics_act(phys, event=""):
    from .world import describe, kind_of
    try:
        return kind_of(phys, describe(phys.get("a")), event)
    except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError, AttributeError):
        return ""


def _body_name(phys, side):
    b = phys.get(side) if isinstance(phys, dict) else None
    return name_words(b.get("name")) if isinstance(b, dict) else ""


def quantise(scene, phys=None, event=""):
    if not isinstance(scene, dict):
        return ""
    src = _enum(scene.get("src"), SOURCES)
    out = []
    if src:
        out.append(("src", src))
    if src == "npc":
        who = name_words(scene.get("who"))
        if who:
            out.append(("who", who))
    if isinstance(phys, dict):
        obj, on = _body_name(phys, "a"), _body_name(phys, "b")
        act = physics_act(phys, event)
    else:
        obj, on = name_words(scene.get("obj")), ""
        act = _enum(scene.get("act"), ACTS)
    if obj:
        out.append(("obj", obj))
    if on:
        out.append(("on", on))
    if act:
        out.append(("act", act))
    room = _enum(scene.get("room"), ROOMS)
    if room:
        out.append(("room", room))
    return ";".join("%s=%s" % kv for kv in out)


def parse(text):
    out = {}
    for part in str(text or "").split(";"):
        name, eq, value = part.partition("=")
        if eq and name in FIELDS:
            out[name] = value
    return out
