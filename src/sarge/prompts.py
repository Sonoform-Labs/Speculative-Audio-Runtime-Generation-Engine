from __future__ import annotations

SPACES = {
    "outdoor_open": {"foley": "exterior, open air", "gun": "exterior, open field, echo rolling away into the distance",
                     "in": "outdoors", "at": "in an open field", "tail": "with a long outdoor echo rolling away",
                     "floor": "paving stones", "tag": "exterior"},
    "indoor_small": {"foley": "inside a small room, short tight reflections off nearby walls",
                     "gun": "fired inside a small concrete room, harsh short slapback off nearby walls, interior",
                     "in": "inside a small room", "at": "inside a small concrete room",
                     "tail": "with a harsh short slapback off nearby walls", "floor": "a tiled floor",
                     "tag": "interior, small room"},
    "indoor_large": {"foley": "inside a large empty hall, light echoing reverb tail",
                     "gun": "interior, large empty warehouse, long echoing reverb tail",
                     "in": "inside a large empty hall", "at": "inside a large empty warehouse",
                     "tail": "with a long echoing reverb tail", "floor": "a concrete floor",
                     "tag": "interior, large hall"},
    "tunnel": {"foley": "inside a long concrete tunnel, hard echo ringing down the corridor",
               "gun": "inside a long concrete tunnel, hard echo ringing down the corridor",
               "in": "inside a long concrete tunnel", "at": "inside a long concrete tunnel",
               "tail": "with a hard echo ringing down the corridor", "floor": "the concrete",
               "tag": "interior, concrete tunnel"},
}
SPACE_ALIASES = {"": "outdoor_open", "default": "outdoor_open", "arena": "outdoor_open", "field": "outdoor_open",
                 "walkarena": "outdoor_open", "outdoor": "outdoor_open", "open": "outdoor_open",
                 "small_room": "indoor_small", "room": "indoor_small", "indoor": "indoor_small",
                 "hall": "indoor_large", "warehouse": "indoor_large", "corridor": "tunnel"}

GUNS = {
    "pistol": {"fire": "Gunshot, 9mm pistol, single shot, sharp cracking report with a metallic slide clack, "
                       "shooter perspective, close, {gun}",
               "noun": "pistol", "reload_head": "Pistol reload",
               "reload": "magazine slapped into the grip with a hard click, slide racked forward with a crisp metallic clack",
               "draw": "9mm pistol drawn from a leather holster, quick leather slide, light metallic click of the safety",
               "holster": "9mm pistol slid into a leather holster, soft leather rub and a snap"},
    "rifle": {"fire": "Single M4 rifle shot {at}, two quick metallic clicks of the trigger and bolt, then a sharp "
                      "ringing crack {tail}, close perspective",
              "noun": "rifle", "reload_head": "Assault rifle reload",
              "reload": "magazine slapped home with a hard click, bolt released forward with a crisp metallic clack",
              "draw": "M4 carbine lifted off a sling, cloth swish, plastic and metal clatter, light tap on the charging handle",
              "holster": "M4 carbine slung over the shoulder, nylon sling swish and a soft metallic clack"},
    "_other": {"fire": "Gunshot, {name}, single shot, sharp crack with a metallic mechanical clack, "
                       "shooter perspective, close, {gun}",
               "noun": "gun", "reload_head": "Gun reload",
               "reload": "magazine slapped home with a hard click and a crisp metallic clack",
               "draw": "{name} picked up, quick grip rustle and a light metallic click",
               "holster": "{name} put away, soft leather rub and a light metallic click"},
}
SURFACE_WORDS = {"metal", "stone", "wood", "grass", "gravel", "glass", "water", "snow", "sand", "carpet", "mud", "dirt"}

STEPS = {
    "stone": {"head": "Footstep on stone tiles", "action": "single {shoe} step", "shoe": "hard-heeled shoe",
              "step": "sharp clicky heel strike and a short gritty scuff", "persp": "close perspective foley",
              "surface": "stone tiles", "land": "a scrape of grit", "under": "a hard knock on stone"},
    "gravel": {"head": "Boots Gravel Footstep", "action": "single {shoe} step on loose gravel, heel and toe",
               "shoe": "boot", "step": "coarse crunch with grinding pebbles and grit",
               "persp": "close perspective foley", "surface": "loose gravel",
               "land": "a crunch of scattering pebbles", "under": "a gritty crunch of gravel"},
    "wood": {"head": "Single footstep, {wood}", "action": "{shoe} heel thud on planks", "shoe": "shoe",
             "step": "resonant hollow wood tone, small creak", "persp": "foley, close-distance",
             "surface": "wooden planks", "land": "a hollow knock of the planks", "under": "a hollow knock on wood"},
    "metal": {"head": "Metal footstep", "action": "single {shoe} step", "shoe": "",
              "step": "steel, clang, ring, rattle, hollow, boot, metal floor", "persp": "foley, close perspective",
              "surface": "a steel deck plate", "land": "a ringing metallic clang", "under": "a ringing clang of steel"},
    "grass": {"head": "Single footstep on grass", "action": "soft heel and toe", "shoe": "",
              "step": "gentle swish of grass blades, light earthy thud", "persp": "close up",
              "surface": "grass", "land": "a soft rustle of grass", "under": "a soft thump on grass"},
    "glass": {"head": "Boots glass footstep", "action": "one {shoe} step onto a floor of shattered glass",
              "shoe": "heavy", "step": "crunchy grinding shards with tiny bright pings", "persp": "close up foley",
              "surface": "broken glass shards", "land": "a crunch of broken shards",
              "under": "a crunch of broken glass"},
    "snow": {"head": "Footstep on packed snow", "action": "single {shoe} step, heel and toe", "shoe": "boot",
             "step": "squeaky crunch of compressed snow", "persp": "close perspective foley",
             "surface": "packed snow", "land": "a squeaky crunch of snow", "under": "a muffled thump in snow"},
    "sand": {"head": "Footstep on sand", "action": "single {shoe} step into soft sand, heel and toe", "shoe": "boot",
             "step": "muffled grainy crunch with sliding grit", "persp": "close perspective foley",
             "surface": "soft sand", "land": "a soft spray of sand", "under": "a soft thud in sand"},
    "mud": {"head": "Footstep in mud", "action": "single {shoe} step into watery mud, heel and toe", "shoe": "boot",
            "step": "wet squelch and squidgy suction as the sole lifts", "persp": "close perspective foley",
            "surface": "watery mud", "land": "a wet squelch of mud", "under": "a wet splat in mud"},
    "dirt": {"head": "Footstep on a dirt path", "action": "single {shoe} step, heel and toe", "shoe": "boot",
             "step": "dull crunch of loose soil and a soft earthy thud", "persp": "close perspective foley",
             "surface": "a dirt path", "land": "a dull crunch of soil", "under": "a dull thud on dirt"},
    "carpet": {"head": "Footstep on carpet", "action": "single {shoe} step on carpet over a wooden floor",
               "shoe": "shoe", "step": "muted soft thud with a faint fabric brush", "persp": "close perspective foley",
               "surface": "carpet", "land": "a muted padded thump", "under": "a muted thump on carpet"},
}
BALLISTIC = {
    "glass": ("glass", "punching through a window pane", "hard crack, crisp shatter, high-pitched tinkling fragments", True),
    "glass bottle": ("glass", "shattering a glass bottle", "sharp snap and burst of shattering glass, bright shards tinkling down", True),
    "window": ("glass", "punching through a window pane", "hard crack, crisp shatter, high-pitched tinkling fragments", True),
    "metal": ("metal", "striking a steel plate", "sharp metallic ping with a ringing ricochet whine", True),
    "metal barrel": ("metal", "striking a steel barrel", "hard clank and a boomy ringing bong", True),
    "wood": ("wood", "punching into a wooden crate", "splintering thwack, cracking wood chips and splinters", False),
    "wooden crate": ("wood", "punching into a wooden crate", "splintering thwack, cracking wood chips and splinters", False),
    "stone": ("stone", "chipping a stone wall", "hard crack with gritty chips and debris", False),
    "gravel": ("gravel", "kicking up loose gravel", "dull crunch with a spray of pebbles", False),
    "grass": ("turf", "thudding into the turf", "soft thump with a spray of soil and torn grass", False),
    "dirt": ("dirt", "hitting packed earth", "dull thud with a spray of dirt", False),
    "sand": ("sand", "hitting soft sand", "soft thud with a spray of sand", False),
    "snow": ("snow", "hitting packed snow", "muffled powdery thump", False),
    "mud": ("mud", "hitting wet mud", "wet splat with mud spatter", False),
    "carpet": ("carpet", "hitting a carpeted floor", "dull padded thump", False),
}
PROPS = {"glass bottle": ("Glass bottle impact", "glass bottle", "bright glassy clink and a hollow ring"),
         "metal barrel": ("Metal barrel impact", "steel barrel", "boomy ringing bong of the steel drum"),
         "wooden crate": ("Wooden crate impact", "wooden crate", "hollow crate knock and a dull woody thud")}
BREAKS = {"window": ("glass", "a window pane", "glass", "high-pitched tinkle and chime of shards"),
          "glass": ("glass", "a glass pane", "glass", "high-pitched tinkle and chime of shards"),
          "glass bottle": ("glass", "a glass bottle", "glass", "high-pitched tinkle and chime of shards"),
          "wood": ("wood", "a wooden crate", "splintering wood", "cracking planks and scattering wood chips"),
          "wooden crate": ("wood", "a wooden crate", "splintering wood", "cracking planks and scattering wood chips")}
INT = {"soft": "light", "hard": "heavy"}


def space_of(context):
    c = (context or "").strip().lower()
    if c in SPACES:
        return c
    return SPACE_ALIASES.get(c, "outdoor_open")


def _gun(material):
    g = GUNS.get(material)
    if g is not None:
        return g, material
    name = "handgun" if material in SURFACE_WORDS or not material else material
    return GUNS["_other"], name


def _join(*parts):
    return ", ".join(p for p in parts if p)


def _surface(material):
    return STEPS.get(material) or {"head": "Footstep on %s" % material, "action": "single {shoe} step, heel and toe",
                                   "shoe": "boot", "step": "short %s footstep texture" % material,
                                   "persp": "close perspective foley", "surface": "a %s surface" % material,
                                   "land": "a short %s thud" % material, "under": "a short knock on %s" % material}


def _wood_word(sp):
    return "wooden boardwalk" if sp == "outdoor_open" else "wooden floorboards"


def step(material, intensity, S, iron=False):
    if material == "water":
        lead = "Heavy iron boot wading step" if iron else "Wading step"
        return ("%s through shallow water %s, a boot plunging in with a gulping splash, swirling slosh and bubbling, "
                "scattered droplets falling, recorded close" % (lead, S["in"]))
    s = _surface(material)
    word = INT.get(intensity, "")
    if iron:
        shoe = "heavy iron boot"
    elif s["shoe"] == "heavy":
        shoe = word or "heavy"
    else:
        shoe = ("%s %s" % (word, s["shoe"])).strip()
    action = s["action"]
    if "{shoe}" in action:
        action = action.replace("{shoe}", shoe)
    elif shoe:
        action = "single %s step, %s" % (shoe, action)
    action = " ".join(action.split())
    head = "Iron boot footstep on %s" % s["surface"] if iron else s["head"].replace("{wood}", _wood_word(S["key"]))
    detail = ("hard clank of iron, " + s["step"]) if iron else s["step"]
    return _join(head, action, detail, s["persp"], S["foley"])


def land(material, intensity, S, iron=False):
    if material == "water":
        return ("Landing feet first in shallow water %s, a heavy splash with swirling slosh and bubbling, scattered "
                "droplets falling, recorded close" % S["in"])
    s = _surface(material)
    who = "in heavy iron boots " if iron else ""
    thud = "%s impact thud" % (INT.get(intensity) or "weighty")
    body = "a hard clank of iron" if iron else "a brief clothing rustle"
    return _join("Landing on both feet %sonto %s" % (who, s["surface"]),
                 "%s with %s and %s" % (thud, body, s["land"]), "close perspective", S["foley"])


def jump(material, intensity, S, iron=False):
    s = _surface(material)
    if material == "water":
        return ("Push-off step out of shallow water %s, both feet lifting with a quick splash, slosh and dripping "
                "droplets, recorded close" % S["in"])
    shoe = "heavy iron boots" if iron else "both feet"
    return _join("Push-off step on %s" % s["surface"], "%s pushing off with a quick scuff" % shoe, s["step"],
                 s["persp"], S["foley"])


def ballistic(material, S):
    if material == "water":
        return ("Bullet hitting water %s, a round striking the water surface, quick sharp plop and splash with a "
                "spray of droplets, close perspective" % S["in"])
    noun, hit, det, hi = BALLISTIC.get(material, (material, "striking a %s surface" % material,
                                                  "%s impact, short sharp knock" % material, False))
    return _join("Bullet impact on %s" % noun, "a round %s" % hit, det, "close-distance", S["foley"],
                 "96kHz" if hi else "")


def prop_hit(obj, material, intensity, S):
    head, noun, knock = PROPS.get(obj, ("%s impact" % obj.capitalize(), obj, "%s knock" % obj))
    s = _surface(material)
    if material == "water":
        return "A %s dropping into water %s, a heavy plop and splash with droplets and bubbles, close perspective foley" % (
            noun, S["in"])
    word = INT.get(intensity, "")
    who = ("a %s %s" % (word, noun)) if word else ("a %s" % noun)
    return _join(head, "%s knocking against %s" % (who, s["surface"]), "%s with %s" % (knock, s["under"]),
                 "foley, close-distance", S["foley"])


def breaking(material, S):
    noun, obj, mass, det = BREAKS.get(material, (material, "a %s object" % material, material,
                                                 "%s fragments scattering" % material))
    return "Close perspective %s break, %s smashed with a hard crack, dense shatter of %s, %s settling on %s %s" % (
        noun, obj, mass, det, S["floor"], S["in"])


def build(event, material, intensity="normal", context="default"):
    ev = (event or "").strip().lower()
    mat = (material or "").strip().lower()
    sk = space_of(context)
    S = dict(SPACES[sk], key=sk)
    if ev in ("shoot", "fire"):
        g, name = _gun(mat)
        return g["fire"].format(name=name, gun=S["gun"], at=S["at"], tail=S["tail"])
    if ev == "reload":
        g, name = _gun(mat)
        return "%s, %s, gun handling foley recorded close %s" % (g["reload_head"], g["reload"], S["in"])
    if ev in ("empty_click", "dry_fire"):
        g, name = _gun(mat)
        return ("Empty gun click, %s out of ammo, trigger pulled on an empty chamber, a light metallic tick with a "
                "faint spring rattle, close-distance, %s, foley" % (g["noun"], S["tag"]))
    if ev in ("equip", "draw", "holster", "take off"):
        if "boot" in mat:
            act = "heavy iron boots pulled off, straps loosened, metal plates clanking against each other" \
                if ev == "take off" else "heavy iron boots pulled on, leather straps buckled, metal plates clanking"
            return _join("Armor foley", act, "close-distance", S["foley"])
        g, name = _gun(mat)
        if ev == "holster" or ev == "take off":
            return _join("Weapon holster foley", g["holster"].format(name=name), "close-distance", S["foley"])
        return _join("Weapon draw foley", g["draw"].format(name=name), "close-distance", S["foley"])
    if ev in ("footstep", "iron boot footstep"):
        return step(mat, intensity, S, iron=ev.startswith("iron"))
    if ev in ("land", "iron boot landing"):
        return land(mat, intensity, S, iron=ev.startswith("iron"))
    if ev == "jump":
        return jump(mat, intensity, S)
    if ev == "bullet impact":
        return ballistic(mat, S)
    if ev.endswith(" hit") and ev[:-4] in PROPS:
        return prop_hit(ev[:-4], mat, intensity, S)
    if ev in ("break", "smash", "shatter"):
        return breaking(mat, S)
    if ev in ("splash", "drip", "pour") or mat == "water":
        return ("Big water splash, a heavy plop into deep water with a gulping bloop, a spray of droplets and bubbles "
                "rising, close perspective foley %s" % S["in"])
    if ev in ("punch", "kick"):
        if ev == "kick":
            return _join("Body Being Kicked", "single kick", "boot thumping into flesh, heavy thud and cloth slap",
                         "foley, close-distance", S["foley"])
        return _join("Face/Body Being Punched", "single punch", "knuckles smacking flesh, fleshy slap and low thud",
                     "foley, close-distance", S["foley"])
    if ev in ("click", "press", "switch"):
        return _join("Clicky button", "one click of a hard plastic button",
                     "bright sharp click with a tiny spring rattle", "foley, close-distance", S["foley"])
    s = _surface(mat)
    word = INT.get(intensity, "")
    return _join("%s %s" % (mat.capitalize(), ev), "single %s%s on %s" % (word + " " if word else "", ev, s["surface"]),
                 s["under"], "foley, close-distance", S["foley"])
