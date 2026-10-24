from __future__ import annotations

import json

from .words import split_words

VERSION = "sarge-canon-1"
TEMPLATE = "template:caption+tt"
SLOTS = ("source", "name", "material", "continuous")
MAX_NAME_WORDS = 4
MAX_TOKENS = 64
SYSTEM = (
    "You name sounds for a sound effect library. You get facts about one moment in a game: the event name a "
    "developer typed, and sometimes the material, what made the sound, the objects, the action and the room. "
    "Developers type odd names: slang, sound words, made-up words, verbs for what a character does, file names with "
    "numbers and prefixes. Work out which everyday sound the developer means and give it the name a sound library "
    "would use.\n"
    "Answer with four fields.\n"
    "source: 3 to 10 plain words saying what happens in the moment that makes the sound, in everyday words.\n"
    "name: 1 to 4 plain words, the standard name a sound library gives that sound: the thing that makes it, then "
    "the usual word for the sound, for example shark bite, cat meow or ceiling fan hum. Never copy an odd, slang or "
    "made-up word into the name: replace it with the standard word.\n"
    "material: the material or surface word from the facts that the sound happens on or is made of, or empty.\n"
    "continuous: yes when the sound goes on steadily, such as weather, wind, engines, machines, fire, flowing water "
    "or a loop; no for a single event.\n"
    "Rules.\n"
    "Use what you know about games, about slang and about how real things sound.\n"
    "Ignore numbers, file prefixes such as sfx, snd or fx, and engine words such as cube, prefab, clone or mesh.\n"
    "Leave out who controls the action (player, enemy, npc), and leave out size, force, loudness, distance and the "
    "room.\n"
    "Never add an object or a material that the facts neither mention nor imply."
)
EXAMPLES = (
    ("event: chomp 2\nmade by: npc (shark)",
     {"source": "a shark snaps its jaws shut on something", "name": "shark bite", "material": "", "continuous": "no"}),
    ("event: yowl\nmade by: npc (cat tabby)\naction: use\nroom: open",
     {"source": "a cat calls out loudly", "name": "cat meow", "material": "", "continuous": "no"}),
    ("event: flutter 01\nmade by: npc (pigeon)\naction: use\nroom: open",
     {"source": "a pigeon beats its wings as it takes off", "name": "wing flap", "material": "", "continuous": "no"}),
    ("event: scrape\nmaterial: wood\nmade by: player\nobject: chair\naction: slide\nroom: small",
     {"source": "a chair is dragged across a wooden floor", "name": "chair scrape", "material": "wood",
      "continuous": "no"}),
    ("event: amb wind loop\nmade by: world\naction: loop\nroom: open",
     {"source": "wind blows steadily outdoors", "name": "wind", "material": "", "continuous": "yes"}),
    ("event: whirr\nmade by: world\nobject: fan ceiling\naction: loop\nroom: medium",
     {"source": "a ceiling fan spins and its motor runs", "name": "ceiling fan hum", "material": "",
      "continuous": "yes"}),
    ("event: smooch", {"source": "two people kiss", "name": "kiss", "material": "", "continuous": "no"}),
    ("event: munch munch\nmade by: npc (goat)",
     {"source": "a goat chews its food", "name": "goat chewing", "material": "", "continuous": "no"}),
    ("event: sfx page flip 04\nmaterial: paper",
     {"source": "a paper page is turned over", "name": "page turn", "material": "paper", "continuous": "no"}),
)
GBNF = r'''root ::= "{\"source\": \"" sr "\", \"name\": \"" nm "\", \"material\": \"" mt "\", \"continuous\": \"" yn "\"}"
sr ::= [a-z] [a-z -]{0,59}
nm ::= [a-z] [a-z -]{0,39}
mt ::= [a-z -]{0,30}
yn ::= "yes" | "no"
'''
SCHEMA = {"type": "object", "properties": {
    "source": {"type": "string", "maxLength": 60},
    "name": {"type": "string", "maxLength": 40},
    "material": {"type": "string", "maxLength": 30},
    "continuous": {"type": "string", "enum": ["yes", "no"]}}, "required": list(SLOTS)}
FMT = {"gbnf": GBNF, "schema": SCHEMA, "max_tokens": MAX_TOKENS}


def parse_slots(raw):
    try:
        d = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    out = {k: " ".join(str(d.get(k) or "").split()).strip(" ,.;").lower() for k in SLOTS}
    return out if out["name"] else None


def check(slots, text):
    from . import writer as wr
    flags = []
    name = split_words(slots.get("name"))
    if not name or len(name) > MAX_NAME_WORDS:
        flags.append("name:size")
    for w in name:
        if not w.isalpha() or len(w) < 2:
            flags.append("name:word:" + w)
        elif wr.BANNED_OPEN.fullmatch(w) or wr.CONTROLLER.fullmatch(w):
            flags.append("name:banned:" + w)
    inw = wr._input_words(text)
    if any(not any(wr._near(m, x) for x in inw) for m in split_words(slots.get("material"))):
        flags.append("material:unsourced")
    if slots.get("continuous") not in ("yes", "no"):
        flags.append("continuous:value")
    return flags


def canon_of(slots, flags=()):
    keep = "material:unsourced" not in flags
    return {"name": " ".join(split_words(slots["name"])),
            "material": " ".join(split_words(slots["material"])) if keep else "",
            "continuous": slots.get("continuous") == "yes"}


def fatal(flags):
    return [f for f in flags if not f.startswith("material:")]


def compose(canon, key, model, template=TEMPLATE):
    from .prompts import template as fill
    return fill(template, key.event_type, key.material, key.intensity, key.context, key.phys, model,
                getattr(key, "scene", ""), canon)


def texts(canon, key, template=TEMPLATE):
    from .verify import event_text, key_text, style_texts
    m = style_texts(template)
    return {"request": key_text(key, canon, **m), "event": event_text(key, canon, **m)}


def judge(raw, key, text, spec):
    slots = parse_slots(raw)
    if slots is None:
        return {"failed": True, "flags": ["format"], "raw": raw}
    flags = check(slots, text)
    if fatal(flags):
        return {"failed": True, "flags": flags, "slots": slots, "raw": raw}
    c = canon_of(slots, flags)
    tpl = spec.get("template") or TEMPLATE
    return {"slots": slots, "flags": flags, "repaired": bool(flags), "canon": c, "woosh": compose(c, key, "woosh", tpl),
            "sa3": compose(c, key, "sa3", tpl), "texts": texts(c, key, tpl), "raw": raw}


def ident(template=TEMPLATE):
    return {"version": VERSION, "gbnf": GBNF, "schema": SCHEMA, "max_tokens": MAX_TOKENS, "template": template,
            "max_name_words": MAX_NAME_WORDS}


def style(template=TEMPLATE):
    from . import writer as wr
    from .prompts import parse_template
    if parse_template(template) is None:
        raise ValueError("the canon writer needs a template:NAME prompt style, got %r" % (template,))
    return {"name": "canon", "system": SYSTEM, "examples": EXAMPLES, "named": "", "form": "canon",
            "banned": wr.BANNED_OPEN, "collide": False, "ex_words": (), "leak": frozenset(), "kind": "canon",
            "template": template, "fmt": FMT, "canon": ident(template)}


def describe(rec):
    c = rec.get("canon") if isinstance(rec, dict) else None
    if not c:
        return ""
    return "%s%s%s" % (c["name"], " / " + c["material"] if c["material"] else "",
                       " (continuous)" if c["continuous"] else "")
