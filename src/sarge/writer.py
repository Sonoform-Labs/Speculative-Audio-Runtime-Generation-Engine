from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request

from . import scene as scenes
from .types import CacheKey
from .words import split_words

VERSION = "sarge-writer-1"
SLOTS = ("source", "action", "sound")
SYSTEM = (
    "You write the text for a sound effect generator. You get facts about one moment in a game: the event name a "
    "developer typed, what made it happen, the objects involved with their materials, how they touched, and the "
    "space. Describe what a microphone right next to the sound would record.\n"
    "Answer with three fields.\n"
    "source: 2 to 5 words naming what makes the sound, like a sound library title.\n"
    "action: 4 to 14 words saying what physically happens to which objects and materials.\n"
    "sound: 2 to 6 words describing the sound itself, the way a sound designer would.\n"
    "Rules.\n"
    "Names were typed by developers. Read them as hints and turn them into plain nouns; ignore numbers and engine "
    "words such as cube, prefab, clone, mesh or fx.\n"
    "Use what you know about how real things sound.\n"
    "Describe one occurrence. For a continuous sound such as weather, an engine, a fire or a machine, describe its "
    "steady texture.\n"
    "Write only about the sound and what produces it: no colours, light, smells, feelings, story, music or intentions.\n"
    "Leave out size, weight, loudness, distance, echo, reverb and duration, because the engine sets those.\n"
    "Never add an object or an event that the facts do not mention.\n"
    "Leave out who controls the action (player, enemy, npc); keep the kind of creature or object that makes the sound."
)
EXAMPLES = (
    ("event: impact\nmade by: physics\nobject: tin can, metal cylinder\ntouches: counter, stone floor\n"
     "contact: crisp\naction: hit\nforce: hard\nroom: small",
     {"source": "Tin can impact", "action": "an empty tin can knocked onto a stone counter",
      "sound": "tinny clank, light rattle"}),
    ("event: croak\nmade by: npc (frog)\naction: use\nroom: open",
     {"source": "Frog croak", "action": "a single croak from a frog", "sound": "throaty ribbit, wet rasp"}),
    ("event: wind loop\nmade by: world\naction: loop\nroom: open",
     {"source": "Wind gusts", "action": "steady wind blowing across open ground and dry grass",
      "sound": "airy whoosh, soft rustle"}),
    ("event: open\nmade by: player\nobject: desk drawer\naction: use\nroom: small",
     {"source": "Drawer open", "action": "a wooden desk drawer pulled open on its runners",
      "sound": "wooden slide, soft rumble"}),
)
STYLE_SYSTEM = (
    "You write the text for a sound effect generator. You get facts about one moment in a game: the event name a "
    "developer typed, what made it happen, the objects involved with their materials, how they touched, and the "
    "space. The event name says which sound the developer wants, and the other facts only add detail to it. Describe "
    "what a microphone right next to the sound would record.\n"
    "Answer with three fields.\n"
    "source: 2 to 5 words naming the sound like a sound library title. Keep the words of the event name in it, and add "
    "the object or the material when that helps.\n"
    "{action}\n"
    "{sound}\n"
    "Rules.\n"
    "Never rename the event or describe a different event from the one it names.\n"
    "Names were typed by developers. Read them as hints and turn them into plain nouns; ignore numbers and editor "
    "words such as cube, prefab, clone, mesh or fx.\n"
    "Use what you know about how real things sound.{texture}\n"
    "Describe one occurrence. For a continuous sound, describe its steady texture.\n"
    "Write only about the sound and what produces it: no colours, light, smells, feelings, story, music or "
    "intentions.\n"
    "Leave out size, weight, loudness, distance, echo, reverb and duration, because the game sets those.\n"
    "Never add an object or an event that the facts do not mention.\n"
    "Leave out who controls the action (player, enemy, npc); keep the kind of animal, being or object that makes "
    "the sound."
)
ACTION_PLAIN = ("action: 4 to 14 words saying what physically happens to which objects and materials during that "
                "event.")
ACTION_CAPTION = ("action: 5 to 14 words, one plain caption of the kind a sound library prints under a clip, saying "
                  "what makes the sound and what it does, using the words of the event name.")
SOUND_PLAIN = "sound: 2 to 6 words describing the sound itself, the way a sound designer would."
SOUND_CHARACTER = ("sound: 2 to 6 words on the character of the sound itself, the way a sound designer hears it: "
                   "how it starts, its pitch and texture, and how it ends.")
TEXTURE_RULE = (" In the sound field, say what sets this sound apart; avoid generic collision words such as thud, "
                "bang, slap or bump, which only say that something was struck, and use each word once.")
NEW_EXAMPLES = (
    ("event: zipper\nmaterial: nylon",
     {"source": "Nylon zipper", "action": "a nylon zipper drawn quickly along its teeth",
      "sound": "fast ratcheting buzz, gritty rasp"}),
    ("event: purr\nmade by: npc (cat tabby)\naction: use\nroom: small",
     {"source": "Cat purr", "action": "a tabby cat purring steadily while it rests",
      "sound": "low rhythmic rumble, soft flutter"}),
    ("event: kettle whistle\nmade by: world\nobject: kettle enamel\naction: loop\nroom: medium",
     {"source": "Kettle whistle", "action": "an enamel kettle whistling as steam escapes its spout",
      "sound": "rising shrill tone, airy steam"}),
    ("event: quill scratch\nmade by: player\nobject: quill pen\naction: use\nroom: small",
     {"source": "Quill scratch", "action": "a quill pen scratching across a sheet of parchment",
      "sound": "dry papery scritch, thin squeak"}),
)
SCHEMA = {"type": "object", "properties": {
    "source": {"type": "string", "maxLength": 40},
    "action": {"type": "string", "maxLength": 90},
    "sound": {"type": "string", "maxLength": 60}}, "required": list(SLOTS)}
GBNF = r'''root ::= "{\"source\": \"" src "\", \"action\": \"" act "\", \"sound\": \"" snd "\"}"
src ::= [^"\\\n]{1,40}
act ::= [^"\\\n]{1,90}
snd ::= [^"\\\n]{1,60}
'''
SEED = 7
MAX_TOKENS = 96
CTX = 2048
BATCH = 512
ROOM_WORDS = {"open": "outdoors", "small": "in a small room", "medium": "indoors", "large": "in a large hall",
              "corridor": "in a corridor"}
SA3_FRAME = "a single sound recorded up close"
SA3_PREFIX = "TrackType: SFX, "
BUDGET = {"woosh": 70, "sa3": 128}
WHEN = ("seen", "predict", "manifest")
PRIORITY = {"seen": 0, "predict": 1, "manifest": 2}
UNSET = ("", "default", "normal")
DEFAULT_MODEL = "gemma4:e2b-it-qat"
DEFAULT_PORT = 11540
DEFAULT_AFFINITY = 0xEFF
MODEL_DIRS = (r"F:\OllamaModels", r"G:\sarge\llm\ollama_models", os.path.join(os.path.expanduser("~"), ".ollama",
                                                                                 "models"))
BINARIES = (r"F:\Ollama\lib\ollama\llama-server.exe",)
STOP = frozenset(("the", "and", "with", "from", "into", "onto", "over", "its", "a", "an", "of", "on", "in", "at",
                  "by", "to", "single", "sound", "one", "out", "off", "that", "this", "as", "is", "are", "being",
                  "against", "across"))
PREPS = frozenset(("against", "on", "onto", "into", "in", "from", "across", "over", "off", "at", "inside", "under",
                   "through", "with", "of", "by", "near", "beside", "upon"))
TRAIL = frozenset(("a", "an", "the", "and", "or", "of", "with", "on", "onto", "into", "in", "from", "at", "by", "to",
                   "against", "across", "over", "off", "its", "his", "her", "their"))
BANNED_HEAD = (r"red|blue|green|yellow|black|white|orange|purple|grey|gray|golden|shadow\w*|glow\w*|shin\w*|"
               r"sparkl\w*|spark|sparks|flash\w*|smell\w*|scent\w*|aroma\w*|loud\w*|quiet\w*|deafening|distant|"
               r"distance|echo\w*|reverb\w*|huge|tiny|massive|enormous|large|small|big|inflated|")
BANNED_TAIL = r"music\w*|melod\w*|player|players|enemy|enemies|npc|npcs"
BANNED = re.compile("(" + BANNED_HEAD + r"hollow|static|" + BANNED_TAIL + ")")
BANNED_OPEN = re.compile("(" + BANNED_HEAD + BANNED_TAIL + ")")
CONTROLLER = re.compile(r"(player|players|enemy|enemies|npc|npcs)")
COLLIDE = re.compile(r"(thud|thuds|thudding|thump|thumps|thumping|bang|bangs|banging|slap|slaps|slapping|smack|smacks|"
                     r"whack|whacks|thwack|thwacks|bump|bumps|bumping|impact|impacts|collision|collisions)")
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\sA-Za-z0-9]")
_APPROX = re.compile(r"[A-Za-z]+|[0-9]|[^\sA-Za-z0-9]")


def _words(text):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text or ""))
    return [w for w in re.split(r"[^a-z0-9]+", s.lower()) if w]


def _phrase(text):
    return " ".join(split_words(text))


def _bucket(phys):
    if not phys:
        return None
    from .world import parse
    try:
        return parse(phys)
    except ValueError:
        return None


def _body(field, name=""):
    mat, _size, _build, noun = (list(field) + ["", "", "", ""])[:4]
    phrase = " ".join(w for w in ("" if mat == "object" else mat, noun) if w)
    return ([name] if name else []) + ([phrase] if phrase and phrase != name else [])


def key_text(key):
    sc = scenes.parse(key.scene)
    ev = _phrase(key.event_type)
    lines = ["event: " + (ev or "sound")]
    mat = _phrase(key.material)
    if mat and not all(w in ev.split() for w in mat.split()):
        lines.append("material: " + mat)
    src = sc.get("src", "")
    if src:
        lines.append("made by: " + src + (" (%s)" % sc["who"] if sc.get("who") else ""))
    if key.cls:
        lines.append("class: " + _phrase(key.cls))
    b = _bucket(key.phys)
    obj, on = sc.get("obj", ""), sc.get("on", "")
    if b is not None:
        lines.append("object: " + ", ".join(_body(b["a"], obj)))
        if "b" in b:
            lines.append("touches: " + ", ".join(_body(b["b"], on)))
        if "c" in b:
            lines.append("contact: " + b["c"][0])
    else:
        if obj:
            lines.append("object: " + obj)
        if on:
            lines.append("touches: " + on)
    act = sc.get("act") or (b["kind"] if b is not None else "")
    if act:
        lines.append("action: " + act)
    inten = str(key.intensity or "").strip().lower()
    if inten not in UNSET:
        lines.append("force: " + _phrase(inten))
    room = sc.get("room", "")
    if room:
        lines.append("room: " + room)
    return "\n".join(lines)


def messages(text, style=None):
    st = style_of(style)
    out = [{"role": "system", "content": st["system"]}]
    for user, slots in st["examples"]:
        out.append({"role": "user", "content": user})
        out.append({"role": "assistant", "content": json.dumps(slots)})
    out.append({"role": "user", "content": text})
    return out


def _values(text):
    return [line.partition(": ")[2] for line in str(text or "").split("\n")]


def _input_words(text):
    return set(_words(" ".join(_values(text))))


def _ngrams(ws, n):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def _leak_words(examples=EXAMPLES):
    from .prompts import EVENT_VERBS
    from .world import MATERIALS, NOUN_MATERIAL
    vocab = (set(scenes.SOURCES) | set(scenes.ACTS) | set(scenes.ROOMS) | set(MATERIALS) | set(EVENT_VERBS)
             | {"sharp", "crisp", "dull", "soft", "hard", "ball", "box", "pane", "plate", "board", "rod", "cylinder",
                "floor", "wall", "ground", "wheel", "panel"})
    out = set()
    for text, _slots in examples:
        for line in text.split("\n"):
            name, _, value = line.partition(": ")
            if name in ("event", "object", "touches", "made by"):
                out.update(w for w in _words(value) if w not in vocab and w not in NOUN_MATERIAL or w in ("can",))
    return frozenset(w for w in out if len(w) > 2 and w not in vocab)


def _ex_words(examples):
    return tuple(_words(slots[k]) for _text, slots in examples for k in SLOTS)


EX_WORDS = _ex_words(EXAMPLES)
LEAK = _leak_words()
DEFAULT_STYLE = "v1"
STYLE_RULES = "named-1"


def _style(name, system, examples, named="", form="list", banned=BANNED, collide=False):
    return {"name": name, "system": system, "examples": examples, "named": named, "form": form, "banned": banned,
            "collide": collide, "ex_words": _ex_words(examples), "leak": _leak_words(examples)}


STYLES = {
    "v1": dict(_style("v1", SYSTEM, EXAMPLES), ex_words=EX_WORDS, leak=LEAK),
    "named": _style("named", STYLE_SYSTEM.format(action=ACTION_PLAIN, sound=SOUND_PLAIN, texture=""), EXAMPLES,
                    "source", "list", BANNED_OPEN),
    "texture": _style("texture", STYLE_SYSTEM.format(action=ACTION_PLAIN, sound=SOUND_CHARACTER, texture=TEXTURE_RULE),
                      NEW_EXAMPLES, "source", "list", BANNED_OPEN, True),
    "title": _style("title", STYLE_SYSTEM.format(action=ACTION_PLAIN, sound=SOUND_CHARACTER, texture=TEXTURE_RULE),
                    NEW_EXAMPLES, "source", "title", BANNED_OPEN, True),
    "caption": _style("caption", STYLE_SYSTEM.format(action=ACTION_CAPTION, sound=SOUND_CHARACTER,
                                                     texture=TEXTURE_RULE), NEW_EXAMPLES, "action", "caption",
                      BANNED_OPEN, True),
}


def style_of(style=None):
    if isinstance(style, dict):
        return style
    name = style or DEFAULT_STYLE
    if name not in STYLES:
        raise ValueError("unknown writer style %r (choose from %s)" % (name, ", ".join(STYLES)))
    return STYLES[name]


def _field_value(text, name):
    for line in str(text or "").split("\n"):
        k, _, v = line.partition(": ")
        if k == name:
            return v
    return ""


def _request(text):
    return (_field_value(text, "material") + " " + _field_value(text, "event")).strip()


def name_words(text, banned=BANNED_OPEN):
    out = []
    for w in _words(_request(text)):
        if (len(w) >= 3 and w not in STOP and w not in scenes.ACTS and not any(ch.isdigit() for ch in w)
                and not banned.fullmatch(w) and not CONTROLLER.fullmatch(w) and w not in out):
            out.append(w)
    return out


def name_phrase(text):
    return " ".join(w for w in _request(text).split() if not CONTROLLER.fullmatch(w.lower()))


def _near(a, b):
    if a == b:
        return True
    s, lo = sorted((a, b), key=len)
    return (len(s) >= 4 and s in lo) or (len(s) >= 5 and a[:5] == b[:5])


def unnamed(slot, text, banned=BANNED_OPEN):
    have = _words(_clean(slot))
    return [w for w in name_words(text, banned) if not any(_near(w, x) for x in have)]


def _clean(text):
    return " ".join(str(text or "").replace("\n", " ").split()).strip(" ,.;")


def parse_slots(raw, optional=()):
    try:
        d = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    out = {k: _clean(d.get(k)) if isinstance(d.get(k), str) else "" for k in SLOTS}
    return out if all(v for k, v in out.items() if k not in optional) else None


def tail_words(key):
    from .prompts import contact_words
    contact = contact_words(key.phys)
    room = ROOM_WORDS.get(scenes.parse(key.scene).get("room", ""), "")
    if not room and str(key.context or "").strip().lower() not in UNSET:
        room = _phrase(key.context)
    return contact, room


def compose(slots, key, route, style=None):
    form = style_of(style)["form"]
    src, act, snd = (_clean(slots.get(k)) for k in SLOTS)
    contact, room = tail_words(key)
    if route == "sa3":
        if form == "caption":
            body = (act[:1].upper() + act[1:], snd, contact, SA3_FRAME, room)
            return SA3_PREFIX + src[:1].upper() + src[1:] + ". " + ", ".join(p for p in body if p) + "."
        parts = (src[:1].upper() + src[1:], act, snd, contact, SA3_FRAME, room)
        return SA3_PREFIX + ", ".join(p for p in parts if p) + "."
    if form == "title":
        return ", ".join(p for p in (src, snd, contact, room) if p)
    if form == "caption":
        return ", ".join(p for p in (act, snd, contact, room) if p)
    return ", ".join(p for p in (src, act, snd, contact, room) if p)


def approx_tokens(text):
    n = 2
    for t in _APPROX.findall(str(text or "")):
        n += 1 + (len(t) - 1) // 4 if t.isalpha() else 1
    return n


def _unit(tok):
    t = tok.lower()
    return t[:-2] if t.endswith("'s") else t


def _banned(tok, rx=BANNED):
    return bool(rx.fullmatch(_unit(tok)))


def check(slots, text, key, style=None):
    st = style_of(style)
    flags = []
    inw = _input_words(text)
    for name in SLOTS:
        if not _clean(slots.get(name)):
            if not (name == "sound" and st["collide"]) and name not in st.get("optional", ()):
                flags.append("empty:" + name)
            continue
        ws = _words(slots[name])
        grams = list(zip(ws, ws[1:], ws[2:]))
        if len(grams) != len(set(grams)):
            flags.append("loop:" + name)
        if name == "action" and len(ws) < 2:
            flags.append("short:action")
    allw = _words(" ".join(_clean(slots.get(k)) for k in SLOTS))
    for name in SLOTS:
        mine = _ngrams(_words(slots.get(name)), 4)
        for ex in st["ex_words"]:
            for g in sorted(mine & _ngrams(ex, 4)):
                if not set(g) <= inw:
                    flags.append("copy:" + " ".join(g))
    for w in sorted((set(allw) & st["leak"]) - inw):
        flags.append("leak:" + w)
    head = _words(_clean(slots.get("source")) + " " + _clean(slots.get("action")))
    content = [w for w in head if w not in STOP and len(w) >= 3]
    if not any(any(w[:5] == x[:5] for x in inw if len(x) >= 3) for w in content):
        flags.append("ungrounded")
    toks = [t for k in SLOTS for t in _TOKEN.findall(_clean(slots.get(k)))]
    for t in sorted({_unit(t) for t in toks if _banned(t, st["banned"])}):
        if t not in inw or CONTROLLER.fullmatch(t):
            flags.append("banned:" + t)
    for w in sorted({w for w in allw if any(ch.isdigit() for ch in w) and w not in inw}):
        flags.append("digits:" + w)
    if st["collide"]:
        for t in sorted({_unit(t) for t in _TOKEN.findall(_clean(slots.get("sound"))) if COLLIDE.fullmatch(_unit(t))}):
            if t not in inw:
                flags.append("collide:" + t)
    if st["named"] and _clean(slots.get(st["named"])) and unnamed(slots.get(st["named"]), text, st["banned"]):
        flags.append("unnamed:" + st["named"])
    for route, limit in BUDGET.items():
        if approx_tokens(compose(slots, key, route, st)) > limit:
            flags.append("long:" + route)
    out = []
    for f in flags:
        if f not in out:
            out.append(f)
    return out


def _fix_articles(toks):
    out = list(toks)
    for i, t in enumerate(out[:-1]):
        if t.lower() in ("a", "an"):
            nxt = out[i + 1]
            if nxt[:1].isalpha():
                want = "an" if nxt[:1].lower() in "aeiou" else "a"
                out[i] = want if t.islower() else want.capitalize()
    return out


def _detok(toks):
    s = ""
    glue = False
    for t in toks:
        if not s:
            s = t
        elif t in (",", ".", ";", ":", "!", "?"):
            s += t
        elif t in ("-", "/") or glue:
            s += t
        else:
            s += " " + t
        glue = t in ("-", "/")
    return s


def _tidy(toks):
    toks = [t for i, t in enumerate(toks)
            if not (t == "," and (i == 0 or toks[i - 1] == "," or toks[i - 1].lower() in TRAIL))]
    while toks and (toks[-1] in (",", ";") or toks[-1].lower() in TRAIL):
        toks.pop()
    while toks and (toks[0] in (",", ";") or toks[0].lower() in ("and", "or", "with")):
        toks.pop(0)
    return _clean(_detok(_fix_articles(toks)))


ARTICLES = frozenset(("a", "an", "the"))


def _cut_phrase(toks, marks, to_end=False):
    lo, hi = min(marks), max(marks)
    start = lo if to_end else None
    art = None
    j = lo - 1
    while j >= 0 and toks[j] not in (",", ";"):
        if toks[j].lower() in PREPS:
            start = j
            break
        if art is None and toks[j].lower() in ARTICLES:
            art = j
        j -= 1
    if start is None and art is not None:
        return toks[:art] + toks[hi + 1:]
    if start is None:
        return [t for k, t in enumerate(toks) if k not in marks]
    end = hi + 1
    while end < len(toks) and toks[end] not in (",", ";"):
        end += 1
    return toks[:start] + toks[end:]


def _items(toks):
    items, cur = [], []
    for i, t in enumerate(toks):
        if t == ",":
            items.append(cur)
            cur = []
        else:
            cur.append(i)
    items.append(cur)
    return items


def _span(toks, words):
    idx = [(w, i) for i, t in enumerate(toks) for w in _words(t)]
    ws = [w for w, _i in idx]
    n = len(words)
    for j in range(len(ws) - n + 1):
        if ws[j:j + n] == list(words):
            return set(range(idx[j][1], idx[j + n - 1][1] + 1))
    return set()


def _remove(slot, name, hit=None, span=None, cut=True):
    toks = _TOKEN.findall(slot)
    marks = set().union(*[_span(toks, s) for s in span]) if span else {k for k, t in enumerate(toks) if hit(t)}
    if not marks:
        return slot
    if name == "sound":
        out = []
        for it in _items(toks):
            if set(it) & marks:
                continue
            if out:
                out.append(",")
            out.extend(toks[k] for k in it)
        return _tidy(out)
    if name == "action" and cut:
        for group in _groups(sorted(marks), toks):
            toks = _cut_phrase(toks, group, bool(span)) if group else toks
            break
        return _tidy(toks)
    drop = set(marks)
    for k in marks:
        possessive = toks[k].lower().endswith("'s")
        if CONTROLLER.fullmatch(_unit(toks[k])) and not possessive and k > 0 and toks[k - 1].lower() in ARTICLES:
            drop.add(k - 1)
    return _tidy([t for k, t in enumerate(toks) if k not in drop])


def _groups(marks, toks):
    out, cur = [], []
    for k in marks:
        if cur and any(toks[j] in (",", ";") for j in range(cur[-1], k)):
            out.append(set(cur))
            cur = []
        cur.append(k)
    if cur:
        out.append(set(cur))
    return out


def _unloop(slot):
    toks = _TOKEN.findall(slot)
    low = [_unit(t) for t in toks]
    seen = {}
    for i in range(len(low) - 2):
        g = tuple(low[i:i + 3])
        if g in seen and seen[g] + 3 <= i:
            return _tidy(toks[:i] + toks[i + 3:])
        seen.setdefault(g, i)
    return slot


def repair(slots, flags, text, style=None):
    st = style_of(style)
    rx = st["banned"]
    s = {k: _clean(slots.get(k)) for k in SLOTS}
    inw = _input_words(text)
    spans = [f.partition(":")[2].split() for f in flags if f.startswith("copy:")]
    if spans:
        for name in SLOTS:
            s[name] = _remove(s[name], name, span=spans)
    for f in flags:
        kind, _, arg = f.partition(":")
        if kind == "loop":
            s[arg] = _unloop(s[arg])
        elif kind == "copy":
            continue
        elif kind == "leak":
            for name in SLOTS:
                s[name] = _remove(s[name], name, lambda t, _w=arg: _w in _words(t))
        elif kind == "banned":
            for name in SLOTS:
                s[name] = _remove(s[name], name, lambda t: _banned(t, rx) and (
                    _unit(t) not in inw or bool(CONTROLLER.fullmatch(_unit(t)))), cut=False)
        elif kind == "digits":
            for name in SLOTS:
                s[name] = _remove(s[name], name, lambda t, _w=arg: _w in _words(t), cut=False)
        elif kind == "collide":
            s["sound"] = _remove(s["sound"], "sound", lambda t, _w=arg: _unit(t) == _w)
        elif kind == "unnamed" and arg in SLOTS:
            head = name_phrase(text)
            if head and s[arg]:
                s[arg] = (head[:1].upper() + head[1:] if arg == "source" else head) + ", " + s[arg]
        else:
            return None
    return s if all(v for k, v in s.items() if not (k == "sound" and st["collide"])
                    and k not in st.get("optional", ())) else None


REPAIR_PASSES = 3


def judge(raw, key, text, style=None):
    st = style_of(style)
    slots = parse_slots(raw, st.get("optional", ()))
    if slots is None:
        return {"failed": True, "flags": ["format"], "raw": raw}
    flags = check(slots, text, key, st)
    rec = {"slots": slots, "flags": flags, "repaired": False}
    if flags:
        fixed, after = slots, flags
        for _ in range(REPAIR_PASSES):
            fixed = repair(fixed, after, text, st)
            after = check(fixed, text, key, st) if fixed is not None else ["unrepairable"]
            if not after or fixed is None:
                break
        if after:
            return {"failed": True, "flags": flags, "after": after, "slots": slots, "raw": raw}
        rec = {"slots": fixed, "orig": slots, "flags": flags, "repaired": True}
    rec["woosh"] = compose(rec["slots"], key, "woosh", st)
    rec["sa3"] = compose(rec["slots"], key, "sa3", st)
    rec["raw"] = raw
    return rec


def key_fields(key):
    return {"event_type": key.event_type, "material": key.material, "intensity": key.intensity,
            "context": key.context, "phys": key.phys, "cls": key.cls, "scene": key.scene}


def key_of(fields):
    return CacheKey(**{k: str(fields.get(k, "")) for k in ("event_type", "material", "intensity", "context", "phys",
                                                           "cls", "scene")})


def writer_id(ident, style=None):
    st = style_of(style)
    payload = {"version": VERSION, "system": st["system"], "examples": st["examples"], "schema": SCHEMA, "seed": SEED,
               "max_tokens": MAX_TOKENS, "temperature": 0, "top_k": 1, "ctx": CTX, "batch": BATCH,
               "room_words": ROOM_WORDS, "sa3_frame": SA3_FRAME, "sa3_prefix": SA3_PREFIX, "budget": BUDGET,
               "ident": ident}
    if st["name"] != DEFAULT_STYLE:
        payload.update(style=st["name"], named=st["named"], form=st["form"], banned=st["banned"].pattern,
                       collide=COLLIDE.pattern if st["collide"] else "", rules=STYLE_RULES)
        if st.get("optional"):
            payload["optional"] = list(st["optional"])
    from .world import CONTACT_WORDS
    payload["contact_words"] = CONTACT_WORDS
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def model_dirs():
    env = os.environ.get("OLLAMA_MODELS")
    return ([env] if env else []) + [d for d in MODEL_DIRS if d != env]


def quick_digest(path):
    h = hashlib.sha256()
    size = os.path.getsize(path)
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 22))
        if size > 1 << 23:
            f.seek(size - (1 << 22))
            h.update(f.read(1 << 22))
    return "quick-sha256:" + h.hexdigest()


def resolve_model(spec, dirs=None):
    s = str(spec or "")
    if os.path.isfile(s):
        base = os.path.basename(s)
        if base.startswith("sha256-") and len(base) == 71:
            return s, "sha256:" + base[7:]
        return s, quick_digest(s)
    name, _, tag = s.partition(":")
    tag = tag or "latest"
    parts = name.split("/")
    rel = parts if len(parts) > 1 and "." in parts[0] else ["registry.ollama.ai"] + (
        parts if len(parts) > 1 else ["library"] + parts)
    for root in dirs if dirs is not None else model_dirs():
        man = os.path.join(root, "manifests", *rel, tag)
        if not os.path.isfile(man):
            continue
        try:
            with open(man, encoding="utf-8") as fh:
                layers = json.load(fh).get("layers", [])
        except (OSError, ValueError):
            continue
        for layer in layers:
            if layer.get("mediaType") == "application/vnd.ollama.image.model":
                digest = str(layer.get("digest", ""))
                blob = os.path.join(root, "blobs", digest.replace(":", "-"))
                if os.path.isfile(blob):
                    return blob, digest
    raise FileNotFoundError("writer model %r: not a file and not an Ollama tag under %s" % (
        s, ", ".join(dirs if dirs is not None else model_dirs())))


def find_binary(spec=None):
    for c in ([spec] if spec else []) + [os.environ.get("SARGE_LLAMA_SERVER", "")] + list(BINARIES):
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError("llama-server not found (--writer-bin or SARGE_LLAMA_SERVER)")


def _post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8") or "{}")


def cpu_env(threads):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1",
               GGML_VK_VISIBLE_DEVICES="-1", OMP_NUM_THREADS=str(threads))
    return env


def confine(proc, mask):
    if os.name != "nt":
        return None, None
    import ctypes
    from ctypes import wintypes as w

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", w.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]

    class Io(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class Ext(ctypes.Structure):
        _fields_ = [("Basic", Basic), ("Io", Io), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    nt = ctypes.WinDLL("ntdll")
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
    k.CreateJobObjectW.restype = w.HANDLE
    k.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    k.SetInformationJobObject.restype = w.BOOL
    k.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    k.AssignProcessToJobObject.restype = w.BOOL
    k.SetProcessAffinityMask.argtypes = [w.HANDLE, ctypes.c_size_t]
    k.SetProcessAffinityMask.restype = w.BOOL
    k.SetPriorityClass.argtypes = [w.HANDLE, w.DWORD]
    k.SetPriorityClass.restype = w.BOOL
    k.GetProcessAffinityMask.argtypes = [w.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    k.GetProcessAffinityMask.restype = w.BOOL
    nt.NtResumeProcess.argtypes = [w.HANDLE]
    h = w.HANDLE(int(proc._handle))
    job = k.CreateJobObjectW(None, None)
    ok = False
    if job:
        info = Ext()
        info.Basic.LimitFlags = 0x2000 | 0x10 | 0x20
        info.Basic.Affinity = int(mask)
        info.Basic.PriorityClass = 0x4000
        ok = bool(k.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))) and bool(
            k.AssignProcessToJobObject(job, h))
    if not ok:
        k.SetProcessAffinityMask(h, int(mask))
        k.SetPriorityClass(h, 0x4000)
    nt.NtResumeProcess(h)
    got, sysm = ctypes.c_size_t(), ctypes.c_size_t()
    k.GetProcessAffinityMask(h, ctypes.byref(got), ctypes.byref(sysm))
    return (job if ok else None), got.value


def _nt_call(name, proc):
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes as w
    fn = getattr(ctypes.WinDLL("ntdll"), name)
    fn.argtypes = [w.HANDLE]
    fn.restype = ctypes.c_long
    return fn(w.HANDLE(int(proc._handle))) == 0


class _Gate:
    def __init__(self, backend):
        self.backend = backend

    def __enter__(self):
        self.backend.pause()
        return self

    def __exit__(self, *exc):
        self.backend.resume()
        return False


class Yielding:
    def __init__(self, inner, backend):
        self._g = inner
        self._gate = _Gate(backend)

    def __getattr__(self, name):
        return getattr(self._g, name)

    def generate(self, key, *, variant=0, prompt=None):
        with self._gate:
            if prompt:
                return self._g.generate(key, variant=variant, prompt=prompt)
            return self._g.generate(key, variant=variant)

    def generate_batch(self, items):
        with self._gate:
            return self._g.generate_batch(items)


class LlamaBackend:
    name = "llama"

    def __init__(self, model="", binary=None, port=DEFAULT_PORT, threads=4, affinity=DEFAULT_AFFINITY,
                 host="127.0.0.1", timeout=120.0, load_timeout=600.0, log_path=None):
        self.model, self.digest = resolve_model(model or DEFAULT_MODEL)
        self.binary = find_binary(binary)
        self.port = int(port)
        self.threads = int(threads)
        self.affinity = int(affinity)
        self.host = host
        self.timeout = float(timeout)
        self.load_timeout = float(load_timeout)
        self.log_path = log_path
        self.url = "http://%s:%d" % (host, self.port)
        self.proc = None
        self.job = None
        self.seen_affinity = None
        self._runtime = None
        self._pause_lock = threading.Lock()
        self._paused = 0
        self.pauses = 0

    def pause(self):
        with self._pause_lock:
            self._paused += 1
            p = self.proc
            if self._paused == 1 and p is not None and p.poll() is None:
                if _nt_call("NtSuspendProcess", p):
                    self.pauses += 1

    def resume(self):
        with self._pause_lock:
            if self._paused == 0:
                return
            self._paused -= 1
            p = self.proc
            if self._paused == 0 and p is not None and p.poll() is None:
                _nt_call("NtResumeProcess", p)

    def runtime(self):
        if self._runtime is None:
            try:
                r = subprocess.run([self.binary, "--version"], env=cpu_env(self.threads), capture_output=True,
                                   text=True, timeout=60, creationflags=0x08000000 if os.name == "nt" else 0)
                lines = [ln.strip() for ln in (r.stdout + "\n" + r.stderr).splitlines() if "version" in ln]
                self._runtime = lines[0] if lines else "unknown"
            except (OSError, subprocess.SubprocessError):
                self._runtime = "unknown"
            st = os.stat(self.binary)
            self._runtime += " %d" % st.st_size
        return self._runtime

    def ident(self):
        return {"backend": self.name, "model": self.digest, "runtime": self.runtime(), "threads": self.threads,
                "grammar": GBNF}

    def command(self):
        return [self.binary, "--model", self.model, "--host", self.host, "--port", str(self.port), "-ngl", "0",
                "--device", "none", "-t", str(self.threads), "-tb", str(self.threads), "-np", "1", "-c", str(CTX),
                "-b", str(BATCH), "-ub", str(BATCH), "--jinja", "--no-webui", "--threads-http", "2",
                "--cache-ram", "0"]

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        log = open(self.log_path, "a", encoding="utf-8") if self.log_path else subprocess.DEVNULL
        flags = (0x4 | 0x4000 | 0x08000000) if os.name == "nt" else 0
        self.proc = subprocess.Popen(self.command(), env=cpu_env(self.threads), stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, creationflags=flags)
        if log is not subprocess.DEVNULL:
            log.close()
        self.job, self.seen_affinity = confine(self.proc, self.affinity)
        deadline = time.monotonic() + self.load_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited with code %s" % self.proc.returncode)
            try:
                status, body = _get(self.url + "/health", 2.0)
                if status == 200 and body.get("status") == "ok":
                    return
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(0.25)
        raise TimeoutError("llama-server did not get ready in %.0f s" % self.load_timeout)

    def complete(self, msgs):
        body = {"messages": msgs, "temperature": 0, "top_k": 1, "seed": SEED, "max_tokens": MAX_TOKENS,
                "grammar": GBNF, "chat_template_kwargs": {"enable_thinking": False}}
        out = _post(self.url + "/v1/chat/completions", body, self.timeout)
        return out["choices"][0]["message"].get("content") or ""

    def close(self):
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            try:
                p.terminate()
                p.wait(10)
            except (OSError, subprocess.SubprocessError):
                pass
        job, self.job = self.job, None
        if job:
            import ctypes
            ctypes.WinDLL("kernel32").CloseHandle(job)


class OllamaBackend:
    name = "ollama"

    def __init__(self, model="", port=DEFAULT_PORT, threads=4, host="127.0.0.1", timeout=120.0, load_timeout=120.0):
        self.model = model or DEFAULT_MODEL
        try:
            self.digest = resolve_model(self.model)[1]
        except FileNotFoundError:
            self.digest = "tag:" + self.model
        self.port = int(port)
        self.threads = int(threads)
        self.url = "http://%s:%d" % (host, self.port)
        self.timeout = float(timeout)
        self.load_timeout = float(load_timeout)
        self.think = True

    def runtime(self):
        try:
            return "ollama " + str(_get(self.url + "/api/version", 5.0)[1].get("version", "unknown"))
        except (urllib.error.URLError, OSError, ValueError):
            return "ollama unknown"

    def ident(self):
        self.start()
        return {"backend": self.name, "model": self.digest, "runtime": self.runtime(), "threads": self.threads,
                "schema": SCHEMA}

    def start(self):
        deadline = time.monotonic() + self.load_timeout
        while True:
            try:
                _get(self.url + "/api/version", 2.0)
                return
            except (urllib.error.URLError, OSError, ValueError):
                if time.monotonic() >= deadline:
                    raise
            time.sleep(0.5)

    def complete(self, msgs):
        body = {"model": self.model, "messages": msgs, "stream": False, "format": SCHEMA, "keep_alive": "30m",
                "options": {"temperature": 0, "top_k": 1, "seed": SEED, "num_thread": self.threads, "num_gpu": 0,
                            "num_predict": MAX_TOKENS, "num_ctx": CTX, "num_batch": BATCH}}
        if self.think:
            body["think"] = False
        try:
            out = _post(self.url + "/api/chat", body, self.timeout)
        except urllib.error.HTTPError as e:
            if not self.think or e.code != 400:
                raise
            self.think = False
            body.pop("think", None)
            out = _post(self.url + "/api/chat", body, self.timeout)
        return out.get("message", {}).get("content", "")

    def close(self):
        pass


class PromptStore:
    def __init__(self, path, writer_id="", ident=None, readonly=False, debounce_s=1.0):
        self.path = path
        self.readonly = bool(readonly)
        self.debounce_s = float(debounce_s)
        self.meta = {"version": VERSION, "writer_id": writer_id, "ident": ident or {}}
        self.records = {}
        self.lock = threading.Lock()
        self._timer = None
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                data = {}
            if isinstance(data, dict) and (not writer_id or data.get("writer_id") == writer_id):
                self.records = dict(data.get("records") or {})
                if not writer_id:
                    self.meta.update({k: data.get(k) for k in ("version", "writer_id", "ident") if k in data})

    @classmethod
    def for_writer(cls, folder, writer_id, ident=None, readonly=False):
        return cls(os.path.join(folder, "prompts_%s.json" % writer_id[:12]), writer_id, ident, readonly)

    def get(self, k):
        return self.records.get(k)

    def __contains__(self, k):
        return k in self.records

    def __len__(self):
        return len(self.records)

    def items(self):
        with self.lock:
            return list(self.records.items())

    def put(self, k, rec):
        with self.lock:
            if k in self.records:
                return False
            self.records[k] = rec
            if not self.readonly and self._timer is None:
                self._timer = threading.Timer(self.debounce_s, self.flush)
                self._timer.daemon = True
                self._timer.start()
        return True

    def flush(self):
        if self.readonly:
            return False
        with self.lock:
            self._timer = None
            data = dict(self.meta, records=dict(self.records))
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        for attempt in range(10):
            try:
                os.replace(tmp, self.path)
                return True
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        return False


def ok_record(rec):
    return isinstance(rec, dict) and not rec.get("failed") and bool(rec.get("woosh")) and bool(rec.get("sa3"))


class PromptWriter:
    def __init__(self, backend=None, store_dir=os.path.join(".cache", "writer"), when=WHEN, readonly=False,
                 store_file=None, log=None, queue_cap=4096, retry_s=30.0, style=None, ground=None):
        self.backend = backend
        self.spec = style_of(style)
        self.style = self.spec["name"]
        self.ground = ground
        self.store_dir = store_dir
        self.store_file = store_file
        self.when = frozenset(when)
        self.readonly = bool(readonly) or backend is None
        self.log = log or (lambda msg: print(msg, flush=True))
        self.queue_cap = int(queue_cap)
        self.retry_s = float(retry_s)
        self.on_written = None
        self.on_record = None
        self.store = None
        self.id = ""
        self.state = "new"
        self.ready = threading.Event()
        self.idle = threading.Event()
        self._heap = []
        self._best = {}
        self._seq = itertools.count()
        self._cv = threading.Condition()
        self._stop = False
        self._thread = None
        self.stats = {"queued": 0, "written": 0, "failed": 0, "repaired": 0, "errors": 0, "delivered": 0}
        self.times_ms = []

    def describe(self):
        if self.backend is None:
            return "store %s" % (self.store_file or self.store_dir)
        return "%s %s%s%s%s" % (self.backend.name, os.path.basename(getattr(self.backend, "model", "")),
                                "" if self.style == DEFAULT_STYLE else " style " + self.style,
                                "" if self.ground is None else " ground " + self.ground.mode,
                                " (read-only)" if self.readonly else "")

    def setup(self):
        if self.store_file:
            self.store = PromptStore(self.store_file, readonly=True)
            self.id = str(self.store.meta.get("writer_id") or "")
        else:
            ident = self.backend.ident()
            if self.ground is not None:
                self.spec = self.ground.style(self.spec)
                ident = dict(ident, ground=self.ground.ident())
            self.id = writer_id(ident, self.spec)
            self.store = PromptStore.for_writer(self.store_dir, self.id, ident, readonly=self.readonly)
        self.ready.set()
        return self.store

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="sarge-writer", daemon=True)
            self._thread.start()
        return self._thread

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self.store is not None and not self.store.readonly:
            self.store.flush()
        if self.backend is not None:
            self.backend.close()

    def lookup(self, key):
        st = self.store
        rec = st.get(str(key)) if st is not None else None
        return rec if ok_record(rec) else None

    def want(self, key, why="seen", eta=None):
        if why not in self.when:
            return None
        k = str(key)
        st = self.store
        if st is not None and k in st:
            return "known" if ok_record(st.get(k)) else "failed"
        if st is not None and self.readonly:
            return None
        prio = PRIORITY.get(why, 2)
        rank = (prio, float(eta) if eta is not None and prio == 1 else 0.0)
        with self._cv:
            best = self._best.get(k)
            if best is not None and best[0] <= rank:
                return "queued"
            if best is None and len(self._best) >= self.queue_cap:
                return None
            self._best[k] = (rank, key)
            heapq.heappush(self._heap, (rank, next(self._seq), k))
            self.stats["queued"] += 1
            self.idle.clear()
            self._cv.notify()
        return "queued"

    def _pop(self, timeout=0.5):
        with self._cv:
            while not self._stop:
                while self._heap:
                    rank, _n, k = heapq.heappop(self._heap)
                    best = self._best.get(k)
                    if best is None or best[0] != rank:
                        continue
                    del self._best[k]
                    return k, best[1]
                self.idle.set()
                self._cv.wait(timeout)
            return None

    def _deliver(self, key, rec):
        cb = self.on_written
        if cb is not None and ok_record(rec):
            try:
                cb(key, rec)
                self.stats["delivered"] += 1
            except Exception as e:
                self.log("[writer] delivery failed for %s: %s" % (key, e))

    def prefill(self):
        n = 0
        for k, rec in self.store.items():
            if ok_record(rec):
                try:
                    key = key_of(rec.get("key") or {})
                except TypeError:
                    continue
                if str(key) == k:
                    self._deliver(key, rec)
                    n += 1
        return n

    def write(self, key):
        text = key_text(key)
        t0 = time.perf_counter()
        info = None
        if self.ground is not None:
            raw, info = self.ground.answer(key, text, self.backend, self.spec)
        else:
            raw = self.backend.complete(messages(text, self.spec))
        ms = (time.perf_counter() - t0) * 1000.0
        rec = judge(raw, key, text, self.spec)
        rec.update(key=key_fields(key), text=text, ms=round(ms, 1), t=round(time.time(), 3), attempt=1)
        if self.style != DEFAULT_STYLE:
            rec["style"] = self.style
        if info is not None:
            rec["ground"] = info
            rec["style"] = self.spec["name"]
        self.times_ms = (self.times_ms + [ms])[-200:]
        return rec

    def _run(self):
        try:
            self.setup()
        except Exception as e:
            self.state = "failed: %s" % e
            self.log("[writer] off: %s" % e)
            self.ready.set()
            self._drain_only()
            return
        n = self.prefill()
        self.log("[writer] id %s, store %s: %d prompts loaded, %d served" % (
            self.id[:12], getattr(self.store, "path", ""), len(self.store), n))
        if not self.readonly:
            t0 = time.perf_counter()
            try:
                self.backend.start()
            except Exception as e:
                self.state = "backend failed: %s" % e
                self.log("[writer] backend failed, serving the store only: %s" % e)
                self.readonly = True
            else:
                self.log("[writer] %s ready in %.1f s (affinity %s)" % (
                    self.describe(), time.perf_counter() - t0,
                    hex(self.backend.seen_affinity) if getattr(self.backend, "seen_affinity", None) else "n/a"))
        self.state = "serving" if self.readonly else "writing"
        errors = 0
        while True:
            item = self._pop()
            if item is None:
                return
            k, key = item
            rec = self.store.get(k)
            if rec is not None:
                if ok_record(rec):
                    self._deliver(key, rec)
                continue
            if self.readonly:
                continue
            try:
                rec = self.write(key)
            except Exception as e:
                errors += 1
                self.stats["errors"] += 1
                self.log("[writer] request failed for %s: %s" % (key, e))
                self.want(key, "seen" if errors < 3 else "manifest")
                with self._cv:
                    self._cv.wait(min(self.retry_s, 0.5 * 2 ** errors))
                if errors >= 3:
                    try:
                        self.backend.close()
                        self.backend.start()
                        errors = 0
                    except Exception as e2:
                        self.log("[writer] backend restart failed, serving the store only: %s" % e2)
                        self.readonly = True
                continue
            errors = 0
            self.store.put(k, rec)
            if rec.get("failed"):
                self.stats["failed"] += 1
                self.log("[writer] rejected %s: %s" % (k, ", ".join(rec.get("after") or rec.get("flags") or [])))
            else:
                self.stats["written"] += 1
                if rec.get("repaired"):
                    self.stats["repaired"] += 1
                self.log("[writer] %s in %.0f ms: %s" % (k, rec["ms"], rec["woosh"]))
                self._deliver(key, rec)
            cb = self.on_record
            if cb is not None:
                try:
                    cb(key, rec)
                except Exception:
                    pass

    def _drain_only(self):
        while self._pop() is not None:
            pass


def build(kind, model="", binary=None, port=DEFAULT_PORT, threads=4, affinity=DEFAULT_AFFINITY,
          store=os.path.join(".cache", "writer"), when=WHEN, readonly=False, log_path=None, log=None, style=None,
          ground=None, ground_index=None, ground_k=None):
    style_of(style)
    if kind in (None, "", "off"):
        return None
    if kind == "store":
        return PromptWriter(None, store_file=store, when=when, readonly=True, log=log)
    grounder = None
    if ground not in (None, "", "off"):
        from .ground import Grounder
        grounder = Grounder(ground, ground_index, threads=threads, k=ground_k)
    if kind == "ground":
        from .ground import GroundBackend
        if grounder is None or grounder.mode == "anchors":
            raise ValueError("--writer ground writes without a model and needs --writer-ground words or caption")
        backend = GroundBackend(grounder)
    elif grounder is not None and grounder.mode != "anchors":
        raise ValueError("--writer-ground %s needs no model: use --writer ground" % grounder.mode)
    elif kind == "llama":
        backend = LlamaBackend(model, binary, port, threads, affinity, log_path=log_path)
    elif kind == "ollama":
        backend = OllamaBackend(model, port, threads)
    else:
        raise ValueError("unknown writer backend %r (off, llama, ollama, store, ground)" % kind)
    return PromptWriter(backend, store_dir=store, when=when, readonly=readonly, log=log, style=style, ground=grounder)
