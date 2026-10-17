from __future__ import annotations

import math
import re

MATERIALS = {
    "steel": (200e9, 7850, 0.29, 3), "metal": (200e9, 7850, 0.29, 3), "iron": (170e9, 7200, 0.26, 3),
    "aluminium": (70e9, 2700, 0.33, 3), "aluminum": (70e9, 2700, 0.33, 3), "brass": (100e9, 8500, 0.34, 3),
    "copper": (120e9, 8900, 0.34, 3),
    "glass": (70e9, 2500, 0.22, 2), "ceramic": (100e9, 2400, 0.22, 2), "porcelain": (100e9, 2400, 0.22, 2),
    "stone": (40e9, 2500, 0.2, 1), "concrete": (30e9, 2400, 0.2, 1), "brick": (15e9, 1900, 0.2, 1),
    "ice": (9e9, 917, 0.33, 1), "wood": (11e9, 600, 0.3, 1),
    "plastic": (3e9, 1100, 0.38, 0), "rubber": (5e7, 1100, 0.49, 0),
    "gravel": (1e8, 1600, 0.3, 0), "dirt": (5e7, 1500, 0.3, 0), "sand": (3e7, 1600, 0.3, 0),
    "grass": (2e7, 1200, 0.3, 0), "snow": (1e7, 400, 0.3, 0), "mud": (5e6, 1700, 0.45, 0),
    "carpet": (1e7, 300, 0.3, 0), "cloth": (5e6, 300, 0.3, 0), "flesh": (1e6, 1050, 0.49, 0),
    "water": (2e6, 1000, 0.49, 0),
}
UNKNOWN = (1e9, 1000.0, 0.3, 1)
NOUN_MATERIAL = {
    "window": "glass", "pane": "glass", "windshield": "glass", "mirror": "glass", "bottle": "glass", "jar": "glass",
    "vase": "ceramic", "cup": "ceramic", "mug": "ceramic", "barrel": "metal", "drum": "metal", "can": "metal",
    "pipe": "metal", "girder": "metal", "locker": "metal", "anvil": "iron", "bell": "metal", "gong": "metal",
    "rail": "metal", "chain": "metal", "sword": "steel", "helmet": "metal", "crate": "wood", "plank": "wood",
    "log": "wood", "branch": "wood", "board": "wood", "pallet": "wood", "fence": "wood", "bench": "wood",
    "chair": "wood", "table": "wood", "rock": "stone", "boulder": "stone", "pebble": "stone", "slab": "stone",
    "pillar": "stone", "statue": "stone", "tire": "rubber", "tyre": "rubber", "balloon": "rubber",
    "pillow": "cloth", "cushion": "cloth", "mattress": "cloth", "sack": "cloth", "bag": "cloth",
    "body": "flesh", "corpse": "flesh", "ragdoll": "flesh",
}
KINDS = ("hit", "bounce", "break")
BREAK_WORDS = ("break", "smash", "shatter")
RING = ("dead", "short", "ringing", "long")
T40_S = {"dead": 0.08, "short": 0.2, "ringing": 0.6, "long": 1.5}
PUNCH = {"sharp": 1.0, "crisp": 0.8, "dull": 0.5, "soft": 0.25}
SOFT_LOWPASS_HZ = 400.0
DRIVE_CAP = 1.0
P0 = 5.0e4
REF_SPEED = 3.0
SOFT_VN = 1.5
HARD_VN = 6.0
GROUND = {"shape": "box", "ext": [10.0, 0.2, 10.0], "dyn": 0}
GROUNDS = {"grass", "gravel", "dirt", "sand", "snow", "mud", "carpet", "water"}
ADJECTIVE = {"wood": "wooden"}
SIZE_WORD = {"tiny": "", "small": "", "medium": "", "large": "", "huge": ""}
BUILD_WORDS = {"inflated": "", "hollow": ""}
PANEL = {"glass": "pane", "metal": "plate", "steel": "plate", "iron": "plate", "wood": "board", "stone": "slab",
         "concrete": "slab"}
CONTACT_WORDS = {"sharp": "a sharp bright click", "crisp": "a crisp hard knock", "dull": "a dull thud",
                 "soft": "a soft muffled bump"}
RING_WORDS = {"dead": "", "short": "", "ringing": "", "long": ""}
FRAGMENTS = {"glass": "bright tinkling shards", "ceramic": "clinking shards", "porcelain": "clinking shards",
             "wood": "cracking splinters", "stone": "gritty chips and debris", "concrete": "gritty chips and debris",
             "brick": "gritty chips and debris", "plastic": "clattering plastic pieces"}
BREAK_TAIL = {"dead": "{f} dropping", "short": "{f} scattering briefly", "ringing": "{f} scattering and settling",
              "long": "{f} scattering and settling with a long tail"}


def _num(x, default=0.0):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def tokens(text):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text or ""))
    return [t for t in re.split(r"[^a-z]+", s.lower()) if t]


MATERIAL_SUFFIXES = ("", "s", "es", "en", "y", "ic", "lic", "ed", "ish")
NOUN_SUFFIXES = ("", "s", "es")


def _word_in(toks, table, suffixes):
    for t in toks:
        for w in table:
            if t.startswith(w) and t[len(w):] in suffixes:
                return w
    return ""


def _has_word(phrase, word):
    return bool(word) and _word_in(tokens(phrase), (word,), MATERIAL_SUFFIXES) == word


def material_of(d, rho, bnc):
    for src in ("tag", "pm", "rm", "name"):
        w = _word_in(tokens(d.get(src)), MATERIALS, MATERIAL_SUFFIXES)
        if w:
            return w, src
    for src in ("tag", "name", "pm", "rm"):
        w = _word_in(tokens(d.get(src)), NOUN_MATERIAL, NOUN_SUFFIXES)
        if w:
            return NOUN_MATERIAL[w], src
    if bnc >= 0.6:
        return "rubber", "bounciness"
    if rho is None:
        return "", "unknown"
    if rho > 5000:
        return "metal", "density"
    if rho > 1800:
        return "stone", "density"
    if rho > 800:
        return "plastic", "density"
    return "wood", "density"


def volume(shape, ext):
    a, b, c = sorted(ext)
    if shape == "sphere":
        return math.pi / 6.0 * c ** 3
    if shape in ("capsule", "cylinder"):
        return math.pi / 4.0 * a * b * c
    if shape == "box":
        return a * b * c
    return 0.5 * a * b * c


def size_class(size):
    return ("tiny" if size < 0.05 else "small" if size < 0.2 else "medium" if size < 0.6 else
            "large" if size < 2.0 else "huge")


def shape_noun(shape, ext, mat):
    lo, hi = min(ext), max(ext)
    if shape == "sphere":
        return "ball"
    if shape == "terrain":
        return "ground"
    if shape == "box":
        return PANEL.get(mat, "panel") if lo < 0.15 * hi else "box"
    if shape in ("capsule", "cylinder"):
        return "rod" if hi > 3.0 * lo else "cylinder"
    if shape == "wheel":
        return "wheel"
    return "object"


def describe(d):
    d = d if isinstance(d, dict) else {}
    shape = str(d.get("shape") or "mesh").lower()
    raw = d.get("ext")
    ext = [abs(_num(v)) for v in raw][:3] if isinstance(raw, (list, tuple)) and len(raw) >= 3 else [0.3, 0.3, 0.3]
    ext = [float("%.3g" % max(v, 1e-3)) for v in ext]
    dyn = bool(_num(d.get("dyn")))
    m = _num(d.get("m"))
    bnc = _num(d.get("bnc"))
    vol = volume(shape, ext)
    rho = m / vol if dyn and m > 0 and vol > 0 else None
    mat, src = material_of(d, rho, bnc)
    E, dens, nu, ring = MATERIALS.get(mat, UNKNOWN)
    inflated = (rho is not None and rho < 0.5 * dens and shape in ("sphere", "capsule")
                and (mat in ("rubber", "plastic") or bnc >= 0.5))
    hollow = rho is not None and rho < 0.25 * dens
    size = max(ext)
    build = "static" if not dyn else "inflated" if inflated else "hollow" if hollow else "solid"
    if shape == "sphere":
        R = size / 2.0
    elif shape in ("capsule", "cylinder"):
        R = min(ext) / 2.0
    else:
        R = max(0.005, 0.25 * min(ext))
    return dict(mat=mat, src=src, E=E, nu=nu, dens=dens, ring=ring, rho=rho, bnc=bnc, dyn=dyn,
                m=m if dyn and m > 0 else math.inf, shape=shape, ext=ext, size=size, sc=size_class(size),
                build=build, R=R, noun=shape_noun(shape, ext, mat))


def surface_noun(b, normal):
    if b["shape"] == "terrain":
        return "ground"
    if isinstance(normal, (list, tuple)) and len(normal) >= 3:
        n = [_num(v) for v in normal[:3]]
        mag = math.sqrt(sum(v * v for v in n))
        if mag > 1e-6:
            return "floor" if abs(n[1]) / mag >= 0.5 else "wall"
    return "floor" if b["ext"][1] <= min(b["ext"][0], b["ext"][2]) else "wall"


def contact_time(a, b, v=REF_SPEED):
    inv = sum(1.0 / x["m"] for x in (a, b) if math.isfinite(x["m"]))
    ms = 1.0 / inv if inv > 0 else 1.0
    for x in (a, b):
        if x["build"] == "inflated":
            k = 2.0 * math.pi * x["R"] * P0
            return math.pi * math.sqrt(ms / k), "pneumatic"
    estar = 1.0 / ((1.0 - a["nu"] ** 2) / a["E"] + (1.0 - b["nu"] ** 2) / b["E"])
    if a["dyn"] and b["dyn"]:
        rstar = 1.0 / (1.0 / a["R"] + 1.0 / b["R"])
    elif b["dyn"]:
        rstar = b["R"]
    elif a["dyn"]:
        rstar = a["R"]
    else:
        rstar = min(a["R"], b["R"])
    return 2.87 * (ms * ms / (rstar * estar * estar * max(v, 0.05))) ** 0.2, "hertz"


def contact_class(tau):
    return "sharp" if tau < 2.5e-4 else "crisp" if tau < 1e-3 else "dull" if tau < 4e-3 else "soft"


def ring_class(a, b, tau):
    best, who = None, ""
    for x, y in ((a, b), (b, a)):
        r = x["ring"]
        if not x["dyn"]:
            r -= 1
        if x["build"] == "hollow" and x["ring"] >= 2:
            r += 1
        if x["dyn"] and not y["dyn"] and y["ring"] == 0:
            r -= 1
        if x["sc"] == "tiny":
            r -= 1
        if best is None or r > best:
            best, who = r, x["mat"]
    if tau > 4e-3:
        best -= 1
    return RING[max(0, min(3, best))], who or "object"


def kind_of(phys, a, event=""):
    k = str(phys.get("kind") or "").lower()
    if k in KINDS:
        return k
    e = str(event or "").lower()
    if any(w in e for w in BREAK_WORDS):
        return "break"
    return "bounce" if a["bnc"] >= 0.5 else "hit"


def motion_of(phys):
    vn, vt = abs(_num(phys.get("vn"))), abs(_num(phys.get("vt")))
    if vt > 0.5 and _num(phys.get("roll")) > 0.8:
        return "roll"
    if vt > 0.5 and vt > 2.0 * vn:
        return "slide"
    return "impact"


def intensity_of(phys, given=None):
    if given in ("soft", "normal", "hard"):
        return given
    vn = abs(_num((phys or {}).get("vn"), _num((phys or {}).get("v"))))
    return "soft" if vn < SOFT_VN else "hard" if vn >= HARD_VN else "normal"


def _field(x, build=None, noun=None):
    return "%s,%s,%s,%s" % (x["mat"] or "object", x["sc"], build or x["build"], noun or x["noun"])


def analyse(phys, event=""):
    if not isinstance(phys, dict):
        raise ValueError("phys must be an object")
    a = describe(phys.get("a"))
    kind = kind_of(phys, a, event)
    if kind == "break":
        ring = RING[max(0, min(3, a["ring"] + (1 if a["ring"] >= 2 else 0)))]
        key = "break;a=%s;r=%s,%s" % (_field(a), ring, a["mat"] or "object")
        return key, dict(kind=kind, motion=motion_of(phys), a_mat=a["mat"], a_src=a["src"], a_rho=a["rho"])
    b = describe(phys.get("b") if isinstance(phys.get("b"), dict) else GROUND)
    tau_ref, model = contact_time(a, b, REF_SPEED)
    tau, _ = contact_time(a, b, abs(_num(phys.get("vn"), REF_SPEED)) or REF_SPEED)
    ring, ringer = ring_class(a, b, tau_ref)
    bfield = _field(b, noun=surface_noun(b, phys.get("n")) if not b["dyn"] else None)
    key = "%s;a=%s;b=%s;c=%s;r=%s,%s" % (kind, _field(a), bfield, contact_class(tau_ref), ring, ringer)
    return key, dict(kind=kind, motion=motion_of(phys), a_mat=a["mat"], a_src=a["src"], a_rho=a["rho"],
                     b_mat=b["mat"], b_src=b["src"], tau_ms=round(tau * 1000.0, 3),
                     tau_ref_ms=round(tau_ref * 1000.0, 3), fc_hz=round(1.0 / tau), model=model)


def bucket(phys, event=""):
    return analyse(phys, event)[0]


def parse(key):
    parts = str(key or "").split(";")
    if not parts or parts[0] not in KINDS:
        raise ValueError("not a physics bucket: %r" % key)
    out = {"kind": parts[0]}
    for p in parts[1:]:
        name, _, val = p.partition("=")
        out[name] = val.split(",")
    return out


def shape(key):
    p = parse(key)
    ring = p["r"][0]
    c = p["c"][0] if "c" in p else ""
    return {"t40_s": T40_S[ring], "punch_scale": PUNCH.get(c, 1.0),
            "drive": DRIVE_CAP if ring in ("dead", "short") else None,
            "floor_db": -45.0 if ring in ("dead", "short") else -55.0,
            "lowpass_hz": SOFT_LOWPASS_HZ if c == "soft" else None}


def _article(phrase):
    return ("an " if phrase[:1] in "aeiou" else "a ") + phrase


def _object(mat, size, build, noun):
    words = [SIZE_WORD.get(size, ""), BUILD_WORDS.get(build, "")]
    if mat and not _has_word(noun, mat):
        words.append(mat)
    words.append(noun)
    return _article(" ".join(w for w in words if w))


def prompt_words(key, noun="", intensity="normal"):
    p = parse(key)
    kind = p["kind"]
    am, asz, abuild, anoun = (p["a"] + ["", "", "", ""])[:4]
    am = "" if am == "object" else am
    obj_noun = noun or anoun
    obj = _object(am, asz, abuild if abuild != "static" else "", obj_noun)
    head_noun = obj_noun if (not am or _has_word(obj_noun, am)) else "%s %s" % (am, obj_noun)
    ring, ringer = p["r"][0], p["r"][1]
    if kind == "break":
        frag = FRAGMENTS.get(am, "metal pieces clattering" if MATERIALS.get(am, UNKNOWN)[3] >= 3 else
                             "%s fragments" % (am or "broken"))
        verb = {"soft": "cracking and breaking", "hard": "smashed hard"}.get(intensity, "smashed")
        return {"kind": kind, "head": ("%s break" % head_noun).capitalize(), "action": "%s %s" % (obj, verb),
                "contact": "a hard crack and a dense shatter", "ring": BREAK_TAIL[ring].format(f=frag)}
    bm, bsz, bbuild, bnoun = (p["b"] + ["", "", "", ""])[:4]
    bm = "" if bm == "object" else bm
    if bbuild == "static" and bm in GROUNDS and bnoun in ("floor", "ground"):
        target = "the %s" % bm
    elif bbuild == "static":
        target = "the ground" if bnoun == "ground" else _article(" ".join(w for w in (ADJECTIVE.get(bm, bm), bnoun) if w))
    else:
        target = _object(bm, bsz, bbuild, bnoun)
    if kind == "bounce":
        verb = {"soft": "gently bouncing on %s", "hard": "bouncing hard on %s"}.get(intensity, "bouncing on %s")
    else:
        verb = {"soft": "gently hitting %s", "hard": "hitting %s hard"}.get(intensity, "hitting %s")
    ringer = "" if ringer == "object" else ringer
    return {"kind": kind, "head": ("%s %s" % (head_noun, "bounce" if kind == "bounce" else "impact")).capitalize(),
            "action": "%s %s" % (obj, verb % target), "contact": CONTACT_WORDS[p["c"][0]],
            "ring": " ".join(RING_WORDS[ring].format(m=ringer).split())}
