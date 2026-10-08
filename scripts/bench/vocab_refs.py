import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import csv
import hashlib
import json
import re
import time
import wave
from collections import Counter, defaultdict
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from sarge.generators.stable_audio import _extract
from sarge.loudness import normalize_loudness, target_for
from sarge.types import AudioClip

ROOT = os.path.dirname(os.path.dirname(BENCH))
SR = 44100
INTENSITY = "normal"
LENGTHS = {"0.6": 1.0, "1.5": 2.0}
FADE_S = 0.05
CLIP_LEVEL = 0.999
DATA = r"D:\datasets\sarge_refs"
C3GD = os.path.join(DATA, "c3gd", "C3GD")
HEAR = os.path.join(DATA, "hear", "tasks", "gunshot_triangulation-v1.0-full")
FOLEY = os.path.join(DATA, "foleyset", "FoleySet")
ESC = os.path.join(ROOT, "data", "esc50")
OUT_DIR = os.path.join("data", "refs")
HELDOUT_DIR = os.path.join("data", "refs_heldout")
HELDOUT_JSON = os.path.join(BENCH, "heldout_sources.json")
TRAIN_MIC = ("oh_farm", "DAVE", "")
POOLS = {
    "train": "training and anchor clips: FoleySet split train from uploaders on the training side, C3GD split train "
             "on DAVE at oh_farm from guns on the training side",
    "val": "the same sources in FoleySet split val and C3GD split val on DAVE at oh_farm, for early stopping only",
    "heldout": "every clip of a held-out source, any split and any mic; the held-out reference sets come only from here",
    "unused": "training-side sources outside the two pools above: their FoleySet and C3GD split test clips and their "
              "C3GD clips on other mics",
}
AUDIO_RULE = ("FoleySet holds byte-identical audio under different names, uploaders and sub-categories; a FoleySet "
              "reference clip is dropped when its PCM md5 matches a file from an uploader on the training side "
              "(held-out builds), a file of another sub-category, or an earlier clip of the same set")

LICENSES = {
    "c3gd": {"license": "CC BY 4.0", "source": "C3GD v0.2.0, Zenodo record 22286299",
             "cite": "Gurny and Quinn, Descriptor: Certus Caliber Classification Gunshot Dataset (C3GD), arXiv:2606.18135"},
    "hear": {"license": "CC0", "source": "HEAR 2021 gunshot_triangulation v1.0 full 48000, Zenodo record 5887964",
             "cite": "HEAR 2021 benchmark task gunshot_triangulation"},
    "foleyset": {"license": "CC BY 4.0", "source": "FoleySet, Zenodo record 20735877 (CC0 Freesound clips)",
                 "cite": "Wang and Lerch, FoleySet: A Multi-Level Human-Annotated Foley Sound Dataset, arXiv:2606.25980"},
    "esc50": {"license": "CC BY-NC 3.0", "source": "ESC-50 (github karolpiczak/ESC-50), evaluation only",
              "cite": "Piczak, ESC: Dataset for Environmental Sound Classification, ACM MM 2015"},
}

CLOSE_MICS = [("oh_farm", "DAVE", ""), ("nj_farm", "ALICE", "left"), ("ny_gravel_pit", "ALICE", "left")]
FORMS = {"pistol": "c3gd_pistol", "rifle": "c3gd_rifle", "shotgun": "c3gd_shotgun"}

SETS = {
    "c3gd_pistol": {"type": "gunshot_pistol", "family": "gunshot", "dataset": "c3gd"},
    "c3gd_rifle": {"type": "gunshot_rifle", "family": "gunshot", "dataset": "c3gd"},
    "c3gd_shotgun": {"type": "gunshot_shotgun", "family": "gunshot", "dataset": "c3gd"},
    "hear_far": {"type": "gunshot_far", "family": "gunshot", "dataset": "hear"},
    "fs_gunmech": {"type": "gun_mech", "family": "gun_mech", "dataset": "foleyset", "sub": "GunMechanism"},
    "fs_bulletdrop": {"type": "shell_drop", "family": "shell", "dataset": "foleyset", "sub": "BulletDrop"},
    "esc_fireworks": {"type": "explosion", "family": "explosion", "dataset": "esc50", "cls": "fireworks"},
    "fs_sword": {"type": "sword", "family": "sword", "dataset": "foleyset", "sub": "Sword"},
    "fs_dooropen": {"type": "door_open", "family": "door", "dataset": "foleyset", "sub": "DoorOpenClose"},
    "fs_doorsqueak": {"type": "door_open", "family": "door", "dataset": "foleyset", "sub": "DoorSqueak"},
    "esc_door_wood_creaks": {"type": "door_open", "family": "door", "dataset": "esc50", "cls": "door_wood_creaks"},
    "fs_doorslam": {"type": "door_slam", "family": "door", "dataset": "foleyset", "sub": "DoorSlam"},
    "fs_glass": {"type": "glass_break", "family": "glass", "dataset": "foleyset", "sub": "GlassBreak"},
    "esc_glass_breaking": {"type": "glass_break", "family": "glass", "dataset": "esc50", "cls": "glass_breaking"},
    "fs_mouseclick": {"type": "ui_click", "family": "click", "dataset": "foleyset", "sub": "MouseButtonClick"},
    "fs_button": {"type": "ui_click", "family": "click", "dataset": "foleyset", "sub": "ButtonPress"},
    "esc_mouse_click": {"type": "ui_click", "family": "click", "dataset": "esc50", "cls": "mouse_click"},
    "fs_punch": {"type": "punch", "family": "punch", "dataset": "foleyset", "sub": "PunchHit"},
    "esc_car_horn": {"type": "horn", "family": "horn", "dataset": "esc50", "cls": "car_horn"},
    "esc_dog": {"type": "dog_bark", "family": "dog", "dataset": "esc50", "cls": "dog"},
    "fs_splash": {"type": "splash", "family": "water", "dataset": "foleyset", "sub": "Splash"},
    "fs_step": {"type": "footstep", "family": "footstep", "dataset": "foleyset", "sub": "SingleStep"},
    "esc_footsteps": {"type": "footstep_esc", "family": "footstep", "dataset": "esc50", "cls": "footsteps"},
    "esc_crackling_fire": {"type": None, "family": "fire", "dataset": "esc50", "cls": "crackling_fire"},
    "esc_wind": {"type": None, "family": "wind", "dataset": "esc50", "cls": "wind"},
    "esc_water_drops": {"type": None, "family": "water", "dataset": "esc50", "cls": "water_drops"},
    "esc_door_wood_knock": {"type": None, "family": "knock", "dataset": "esc50", "cls": "door_wood_knock"},
}
TYPES = {
    "gunshot_pistol": {"key": True, "approx": False, "note": "C3GD close mics, form_factor pistol"},
    "gunshot_rifle": {"key": True, "approx": False, "note": "C3GD close mics, form_factor rifle"},
    "gunshot_shotgun": {"key": False, "approx": False, "note": "C3GD close mics, form_factor shotgun"},
    "gunshot_far": {"key": False, "approx": False, "note": "HEAR far mics, cross-check only"},
    "gun_mech": {"key": True, "approx": False, "note": "FoleySet GunMechanism, real firearms only"},
    "shell_drop": {"key": False, "approx": False, "note": "FoleySet BulletDrop, casings falling"},
    "explosion": {"key": True, "approx": True, "note": "ESC-50 fireworks stand in for a grenade explosion"},
    "sword": {"key": True, "approx": True, "note": "FoleySet Sword is mostly clashes and draws, few swings"},
    "door_open": {"key": True, "approx": False,
                  "note": "FoleySet DoorOpenClose open-only, DoorSqueak with no closing, ESC-50 door_wood_creaks"},
    "door_slam": {"key": True, "approx": False, "note": "FoleySet DoorSlam"},
    "glass_break": {"key": True, "approx": False, "note": "FoleySet GlassBreak without ceramics, ESC-50 glass_breaking"},
    "ui_click": {"key": True, "approx": False, "note": "FoleySet MouseButtonClick and ButtonPress, ESC-50 mouse_click"},
    "punch": {"key": True, "approx": False, "note": "FoleySet PunchHit, foley punches without layered designs"},
    "horn": {"key": True, "approx": True, "note": "ESC-50 car_horn stands in for a truck horn"},
    "dog_bark": {"key": True, "approx": False, "note": "ESC-50 dog"},
    "splash": {"key": False, "approx": False, "note": "FoleySet Splash"},
    "footstep": {"key": True, "approx": False, "note": "FoleySet SingleStep, per-surface subsets from original-tags"},
    "footstep_esc": {"key": False, "approx": False, "note": "ESC-50 footsteps (walking), cross-check only"},
}

NOT_TYPE = {
    "fs_gunmech": {
        "443763": "no firearm named (tags: retractable, sound, weapon)",
        "627223": "designed sound effect",
        "659933": "play gun",
        "427254": "source object not named ('gun mechanism or something')",
        "500293": "made with a Baikal MP-654K for a shotgun shell sound",
        "125381": "Nerf gun",
        "647556": "made with the cover of a remote",
        "530225": "processed sound effect, source not named",
        "674742": "mix of airsoft recordings",
        "593888": "airsoft MP5",
        "725397": "airsoft guns",
        "262598": "replica M14",
        "523087": "made with stapler springs",
    },
    "fs_punch": {
        "185714": "a zipper recording",
        "726620": "layered vendor effects, heavily processed",
        "399183": "layered from meat, whip, bass and crack",
        "348242": "a whipped glove layered with a kick drum",
        "144015": "layered with a bass drum, paper and another upload",
        "488720": "layered from many sounds",
        "241969": "about ten layers of slaps and pounding on wood",
        "439747": "combined from several punch recordings",
        "209393": "made from unnamed sounds the uploader had",
        "424144": "built from several sounds",
        "789210": "a cartoon punch recreation from egg shells and a bone crack",
        "395025": "several sandbag drops added together",
    },
    "fs_glass": {
        "735850": "a porcelain mug", "735851": "a porcelain mug", "735852": "a porcelain mug",
        "735853": "a porcelain mug", "735854": "a porcelain mug",
        "566446": "plates and mugs", "566447": "plates and mugs", "566448": "plates and mugs",
        "566450": "plates and mugs", "566451": "plates and mugs", "566452": "plates and mugs",
        "566453": "plates and mugs",
        "735759": "glass and ceramics mixed",
        "554367": "tiles",
        "534534": "a falling cup",
        "540970": "two recordings mixed with a synthetic tone",
    },
    "fs_dooropen": {
        "183529": "an ice-maker lid",
        "667201": "a call to prayer heard through a train door",
    },
    "fs_doorsqueak": {
        "214873": "squeaking household objects, no door named",
        "214880": "squeaking household objects, no door named",
        "214882": "squeaking household objects, no door named",
        "590148": "creaking wood for a rocking chair",
    },
    "fs_doorslam": {
        "448909": "no door named (tags: bang, slams, tap)",
        "483192": "the uploader says it could be a door or a window",
        "646454": "a fist punching a door",
    },
    "fs_button": {
        "503639": "tagged footstep, floor, ground, step, walk",
        "403692": "a ring binder",
        "729999": "film foley, source not named",
        "452839": "a nail clipper",
        "376961": "a squeezed plastic bottle",
        "680003": "source not named ('forgot exactly what it is')",
        "815844": "a designed menu sound effect",
    },
    "fs_splash": {
        "51147": "wet footsteps, the same recording as an ESC-50 footsteps clip",
    },
}
DOOR_FILTER = {"fs_dooropen": "open-only", "fs_doorsqueak": "no closing"}
NOT_USED = {
    "Magazine": "all 114 clips are one Joao_Janz foley pack tagged book, paper, pen, cover and writing, so these are "
                "paper magazines",
}
SURFACES = ["wood", "stone", "grass", "metal", "gravel"]
SURF_TERMS = {
    "wood": {"wood", "wooden", "hardwood", "floorboard", "board", "plank", "parquet", "creak", "creaking", "staircase",
             "stairs"},
    "stone": {"stone", "concrete", "tile", "tiles", "marble", "pavement", "asphalt", "ceramic", "cement", "brick",
              "cobblestone"},
    "grass": {"grass", "grassy", "dry-grass", "lawn"},
    "metal": {"metal", "metallic", "steel", "iron", "grate", "grating"},
    "gravel": {"gravel"},
    "dirt": {"dirt", "earth", "mud", "sludge", "soil", "tierra"},
    "snow": {"snow", "ice", "frozen", "frost"},
    "leaves": {"leaves", "leaf", "leave", "underbrush", "bush", "bracken", "sticks"},
    "carpet": {"carpet"},
    "sand": {"sand"},
    "linoleum": {"linoleum"},
    "water": {"water", "squelch"},
    "litter": {"rubbish", "dump", "paper", "styrofoam"},
}
OPEN_RE = re.compile(r"\bopen(s|ed|ing)?\b|door-open", re.I)
CLOSE_RE = re.compile(r"\bclos(e|es|ed|ing)\b|\bshut\w*|\bslam\w*|door-clos", re.I)
SWORD_KINDS = {
    "draw": re.compile(r"\b(draw|draws|drawn|drawing|unsheath\w*|sheath\w*|re-sheath\w*|scabbard|pull\w*)\b", re.I),
    "clash": re.compile(r"\b(clash\w*|hit|hits|hitting|clang\w*|clank\w*|strike|striking|impact|collide\w*|collision|"
                        r"block|ting|sparring|fight\w*|duel)\b", re.I),
}
SWING_RE = re.compile(r"\b(swing\w*|swoosh|whoosh|wush|swish)\b", re.I)
ESC_NAME = re.compile(r"^(\d+)-(\d+)-([A-Z])-(\d+)\.wav$")
HEAR_NAME = re.compile(r"^(mic\d+)_(shot\d+)_(fid\d+)\.wav$")


def rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def to_sr(x, sr):
    x = np.asarray(x, dtype=np.float64)
    if sr == SR:
        return x.astype(np.float32)
    g = gcd(SR, int(sr))
    return resample_poly(x, SR // g, int(sr) // g).astype(np.float32)


def read(path):
    c = AudioClip.from_wav(path)
    x = c.samples
    return x, int(c.sample_rate)


def pad_to(x, n):
    if len(x) >= n:
        return x, False
    y = np.asarray(x, dtype=np.float32).copy()
    fo = min(int(FADE_S * SR), len(y) // 2)
    if fo > 0:
        y[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return np.pad(y, (0, n - len(y))), True


def process(x, keep, gen):
    y, padded = pad_to(x, int(round(gen * SR)))
    z = normalize_loudness(_extract(y, SR, keep), target_rms=target_for(INTENSITY))
    return np.asarray(z, dtype=np.float32), padded


def spaced(items, k):
    if k is None or len(items) <= k:
        return list(items)
    idx = np.unique(np.round(np.linspace(0, len(items) - 1, k)).astype(int))
    return [items[i] for i in idx]


def cap_groups(rows, group, order, k):
    by = defaultdict(list)
    for r in rows:
        by[group(r)].append(r)
    out = []
    for g in sorted(by):
        out += spaced(sorted(by[g], key=order), k)
    return out


def cap_surfaces(rows, k):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["username"]][r["surface"]].append(r)
    out = []
    for u in sorted(by):
        groups = {s: sorted(v, key=lambda r: r["name"]) for s, v in by[u].items()}
        quota, left = Counter(), k
        for tier in ([s for s in SURFACES if s in groups], sorted(s for s in groups if s not in SURFACES)):
            while left > 0 and any(quota[s] < len(groups[s]) for s in tier):
                for s in tier:
                    if left > 0 and quota[s] < len(groups[s]):
                        quota[s] += 1
                        left -= 1
        for s in sorted(groups):
            out += spaced(groups[s], quota[s]) if quota[s] else []
    return out


def door_ok(sname, r):
    text = r["original-tags"] + " " + r["description"]
    if CLOSE_RE.search(text):
        return False
    return DOOR_FILTER[sname] != "open-only" or bool(OPEN_RE.search(text))


def words(r):
    return {t.strip().lower() for t in r["original-tags"].split(",") if t.strip()}


def c3gd_meta():
    rows = list(csv.DictReader(open(os.path.join(C3GD, "metadata.csv"), encoding="utf-8")))
    plat = {r["platform_id"]: r for r in csv.DictReader(open(os.path.join(C3GD, "metadata", "platforms.csv"),
                                                              encoding="utf-8"))}
    return rows, plat


def mic_of(r):
    return r["event_id"], r["mic_id"], r["channel_orientation"]


def c3gd_clips(args, report, allow=None):
    rows, plat = c3gd_meta()
    locs = list(csv.DictReader(open(os.path.join(C3GD, "metadata", "microphone_locations.csv"), encoding="utf-8")))
    data = os.path.join(C3GD, "data")
    check, picked = {}, []
    for ev, mic, ch in CLOSE_MICS:
        d = sorted({float(r["distance_m"]) for r in locs if r["mic_name"] == mic and r["event_name"] == ev})
        rs = [r for r in rows if r["event_id"] == ev and r["mic_id"] == mic and r["channel_orientation"] == ch]
        miss = [r["filename"] for r in rs if not os.path.isfile(os.path.join(data, r["filename"]))]
        if len(d) != 1 or d[0] > 7.0:
            raise SystemExit("close mic %s at %s has distances %s in microphone_locations.csv" % (mic, ev, d))
        if miss:
            raise SystemExit("%d C3GD files missing under %s, first %s" % (len(miss), data, miss[0]))
        nopl = sorted({r["platform_id"] for r in rs if r["platform_id"] not in plat})
        if nopl:
            raise SystemExit("platforms missing from platforms.csv: %s" % nopl)
        forms = Counter(plat[r["platform_id"]]["form_factor"] for r in rs)
        check["%s|%s|%s" % (ev, mic, ch or "mono")] = {"distance_m": d[0], "files": len(rs), "forms": dict(forms)}
        for r in rs:
            ff = plat[r["platform_id"]]["form_factor"]
            if allow is not None and r["platform_id"] not in allow:
                continue
            if ff in FORMS:
                picked.append((FORMS[ff], r, d[0], ff))
            else:
                report["skipped"].append({"dataset": "c3gd", "id": r["datum_id"], "reason": "form_factor %s" % ff})
    out = []
    for sname in FORMS.values():
        rs = [p for p in picked if p[0] == sname]
        sel = cap_groups(rs, lambda p: p[1]["platform_id"],
                         lambda p: (p[1]["event_id"], p[1]["file_id"], int(p[1]["clip_id"])), args.cap_gun)
        report["capped"][sname] = {"before": len(rs), "after": len(sel)}
        for _s, r, dist, ff in sel:
            out.append({"set": sname, "id": "c3gd_%s" % r["datum_id"], "dataset": "c3gd",
                        "src": os.path.join(data, r["filename"]), "cluster": "c3gd:%s" % r["platform_id"],
                        "recording": "c3gd:%s:%s:%s" % (r["event_id"], r["file_id"], r["clip_id"]),
                        "stratum": "%s:%s" % (r["event_id"], r["mic_id"]), "platform": r["platform_id"],
                        "form_factor": ff, "caliber": r["class_name"], "cartridge": r["cartridge_id"],
                        "event": r["event_id"], "mic": r["mic_id"], "channel": r["channel_orientation"] or "mono",
                        "distance_m": dist, "split": r["split"], "meta_duration_s": float(r["duration_secs"])})
    report["c3gd_close_mics"] = check
    return out


def hear_clips(args, report):
    out = []
    folds = sorted(f for f in os.listdir(os.path.join(HEAR, "48000")) if f.startswith("fold"))
    for fold in folds:
        labels = json.load(open(os.path.join(HEAR, "%s.json" % fold), encoding="utf-8"))
        for name in sorted(labels):
            m = HEAR_NAME.match(name)
            p = os.path.join(HEAR, "48000", fold, name)
            if not m or not os.path.isfile(p):
                raise SystemExit("unexpected HEAR file %s in %s" % (name, fold))
            out.append({"set": "hear_far", "id": "hear_%s" % name[:-4], "dataset": "hear", "src": p,
                        "cluster": "hear:%s" % m.group(2), "recording": "hear:%s:%s" % (m.group(2), m.group(1)),
                        "stratum": m.group(1), "mic": m.group(1), "shot": m.group(2), "fid": m.group(3), "fold": fold,
                        "label": labels[name]})
    report["capped"]["hear_far"] = {"before": len(out), "after": len(out)}
    return out


def esc_clips(sname, cls, users, used, report, block=None):
    out = []
    names = sorted(n for n in os.listdir(os.path.join(ESC, cls)) if ESC_NAME.match(n))
    for name in names:
        m = ESC_NAME.match(name)
        src = m.group(2)
        if block is not None and users.get(src) in block:
            report["excluded"].append({"dataset": "esc50", "set": sname, "id": name, "freesound_id": src,
                                       "user": users[src], "reason": "Freesound uploader on the training side"})
            continue
        if (SETS[sname]["type"], src) in used:
            report["excluded"].append({"dataset": "esc50", "set": sname, "id": name, "freesound_id": src,
                                       "reason": "same Freesound recording as a FoleySet clip of this type"})
            continue
        c = {"set": sname, "id": "esc_%s" % name[:-4], "dataset": "esc50", "src": os.path.join(ESC, cls, name),
             "cluster": "fs:%s" % users[src] if src in users else "esc:%s" % src,
             "recording": "esc:%s:%s" % (src, m.group(3)), "stratum": "esc50", "esc_class": cls,
             "fold": int(m.group(1)), "esc_src": src, "take": m.group(3)}
        if src in users:
            c["username"] = users[src]
        out.append(c)
    report["capped"][sname] = {"before": len(names), "after": len(out)}
    return out


def surface_of(r):
    w = words(r)
    hit = sorted(s for s, terms in SURF_TERMS.items() if w & terms)
    if len(hit) == 1:
        return hit[0]
    return "ambiguous:" + "+".join(hit) if hit else "none"


def sword_kinds(r):
    text = r["original-tags"] + " " + r["description"]
    k = [n for n, rx in SWORD_KINDS.items() if rx.search(text)]
    if SWING_RE.search(r["description"]):
        k.append("swing")
    return k or ["other"]


def foley_rows():
    rows = list(csv.DictReader(open(os.path.join(FOLEY, "metadata", "foleyset_metadata.csv"), encoding="utf-8")))
    for r in rows:
        r["path"] = os.path.join(FOLEY, "audio", r["name"])
    return rows


def foley_eligible(sname, spec, rows, report):
    sub = [r for r in rows if r["sub-category"] == spec["sub"]]
    keep = []
    for r in sub:
        if "ai-generated" in words(r):
            report["excluded"].append({"dataset": "foleyset", "set": sname, "id": r["name"], "user": r["username"],
                                       "reason": "ai-generated"})
        elif r["freesound-id"] in NOT_TYPE.get(sname, {}):
            report["excluded"].append({"dataset": "foleyset", "set": sname, "id": r["name"], "user": r["username"],
                                       "freesound_id": r["freesound-id"],
                                       "reason": NOT_TYPE[sname][r["freesound-id"]]})
        elif sname in DOOR_FILTER and not door_ok(sname, r):
            report["filtered"][sname] = report["filtered"].get(sname, 0) + 1
        elif not os.path.isfile(r["path"]):
            raise SystemExit("FoleySet audio missing: %s" % r["path"])
        else:
            keep.append(r)
    return sub, keep


def pcm_md5(path):
    try:
        with wave.open(path, "rb") as w:
            d = w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        d = open(path, "rb").read()
    return hashlib.md5(d).hexdigest()


def audio_index(rows, held):
    md5 = {r["name"]: pcm_md5(r["path"]) for r in rows}
    subs, sides = defaultdict(set), defaultdict(set)
    for r in rows:
        subs[md5[r["name"]]].add(r["sub-category"])
        sides[md5[r["name"]]].add(held is not None and r["username"] in held)
    train = {md5[r["name"]] for r in rows if held is not None and r["username"] not in held}
    n = Counter(md5.values())
    return {"md5": md5, "subs": subs, "train": train, "groups": sum(k > 1 for k in n.values()),
            "files_in_groups": sum(k for k in n.values() if k > 1),
            "groups_across_labels": sum(len(subs[h]) > 1 for h, k in n.items() if k > 1),
            "groups_across_pools": sum(len(sides[h]) > 1 for h in n)}


def audio_filter(sname, keep, audio, report):
    seen, out = set(), []
    for r in keep:
        h = audio["md5"][r["name"]]
        why = ("audio identical to a FoleySet file from an uploader on the training side" if h in audio["train"] else
               "audio identical to a FoleySet file of another sub-category" if len(audio["subs"][h]) > 1 else
               "audio identical to an earlier clip of this set" if h in seen else None)
        seen.add(h)
        if why:
            report["excluded"].append({"dataset": "foleyset", "set": sname, "id": r["name"], "user": r["username"],
                                       "freesound_id": r["freesound-id"], "reason": why})
        else:
            out.append(r)
    return out


def foley_set(sname, spec, rows, args, report, allow=None, audio=None):
    if allow is not None:
        rows = [r for r in rows if r["username"] in allow]
    sub, keep = foley_eligible(sname, spec, rows, report)
    if audio is not None:
        keep = audio_filter(sname, keep, audio, report)
    one = [r for r in keep if r["Oneshot-Multi"] == "One-shot"]
    use_one = len(one) >= args.min_oneshot and len({r["username"] for r in one}) >= args.min_oneshot_sources
    pool = one if use_one else keep
    if sname == "fs_step":
        for r in pool:
            r["surface"] = surface_of(r)
        sel = cap_surfaces(pool, args.cap_uploader)
    else:
        sel = cap_groups(pool, lambda r: r["username"], lambda r: r["name"], args.cap_uploader)
    report["capped"][sname] = {"before": len(sub), "eligible": len(keep), "oneshot": len(one),
                               "rule": "one-shot only" if use_one else "all (multi-shot cut at the loudest window)",
                               "after": len(sel)}
    out = []
    for r in sel:
        c = {"set": sname, "id": "fs_%s" % r["name"][:-4], "dataset": "foleyset", "src": r["path"],
             "cluster": "fs:%s" % r["username"], "recording": "fs:%s" % r["freesound-id"],
             "stratum": "oneshot" if r["Oneshot-Multi"] == "One-shot" else "multishot",
             "oneshot": r["Oneshot-Multi"] == "One-shot", "username": r["username"], "freesound_id": r["freesound-id"],
             "url": r["url"], "sub_category": r["sub-category"], "tags": r["original-tags"]}
        if sname == "fs_step":
            c["surface"] = r["surface"]
        if sname == "fs_sword":
            c["sword_kind"] = sword_kinds(r)
        out.append(c)
    return out


def split_by(rows, key, field="split"):
    out = defaultdict(set)
    for r in rows:
        out[key(r)].add(r[field])
    return out


def combos(by):
    return dict(sorted(Counter("+".join(sorted(v)) for v in by.values()).items()))


def overlap(rows, crow):
    fu = split_by(rows, lambda r: r["username"])
    dave = [r for r in crow if mic_of(r) == TRAIN_MIC]
    return {
        "foleyset": {"uploaders": len(fu), "uploaders_by_splits": combos(fu),
                     "uploaders_in_test_and_another_split": sum("test" in v and len(v) > 1 for v in fu.values()),
                     "recordings_by_splits": combos(split_by(rows, lambda r: r["freesound-id"]))},
        "c3gd": {"guns_by_splits": combos(split_by(crow, lambda r: r["platform_id"])),
                 "events_by_splits": combos(split_by(crow, lambda r: r["event_id"])),
                 "cartridges_by_splits": combos(split_by(crow, lambda r: r["cartridge_id"])),
                 "recording_files_by_splits": combos(split_by(crow, lambda r: (r["event_id"], r["file_id"]))),
                 "shots_by_splits_across_mics": combos(split_by(crow, lambda r: (r["event_id"], r["file_id"],
                                                                                 r["clip_id"]))),
                 "train_mic_files_by_splits": combos(split_by(dave, lambda r: r["file_id"])),
                 "guns_per_event": {e: len({r["platform_id"] for r in crow if r["event_id"] == e})
                                    for e in sorted({r["event_id"] for r in crow})},
                 "events_per_gun": combos(split_by(crow, lambda r: r["platform_id"], "event_id"))},
    }


def fs_pool(r, held):
    if r["username"] in held:
        return "heldout"
    return {"train": "train", "val": "val"}.get(r["split"], "unused")


def gun_pool(r, held):
    if r["platform_id"] in held:
        return "heldout"
    if mic_of(r) != TRAIN_MIC:
        return "unused"
    return {"train": "train", "val": "val"}.get(r["split"], "unused")


def gun_order(plat, p):
    return plat[p]["form_factor"], plat[p]["caliber"], p


def assign(args, rows):
    crow, plat = c3gd_meta()
    plat = {**{r["platform_id"]: {"form_factor": "unknown", "caliber": "", "missing": True} for r in crow}, **plat}
    fsp = split_by(rows, lambda r: r["username"])
    elig, per = {}, defaultdict(Counter)
    for sname, spec in SETS.items():
        if spec["dataset"] == "foleyset":
            elig[sname] = foley_eligible(sname, spec, rows, {"excluded": [], "filtered": {}})[1]
            for r in elig[sname]:
                per[r["username"]][sname] += 1
    big = sorted(u for u, c in per.items() if max(c.values()) > args.cap_uploader)
    held = {u for u, s in fsp.items() if "test" in s and u not in big}
    seen = defaultdict(list)
    for p in sorted({r["platform_id"] for r in crow if mic_of(r) == TRAIN_MIC}, key=lambda p: gun_order(plat, p)):
        seen[plat[p]["form_factor"]].append(p)
    room = {p for ps in seen.values() for p in ps}
    gh = {p for p in plat if p not in room}
    gh |= {p for ps in seen.values() if len(ps) > 1 for i, p in enumerate(ps) if i % args.gun_holdout_every == 0}
    close = set(CLOSE_MICS)
    guns = {}
    for p in sorted(plat, key=lambda p: gun_order(plat, p)):
        rs = [r for r in crow if r["platform_id"] == p]
        guns[p] = {"pool": "heldout" if p in gh else "train", "form_factor": plat[p]["form_factor"],
                   "caliber": plat[p]["caliber"], "in_platforms_csv": not plat[p].get("missing"),
                   "events": sorted({r["event_id"] for r in rs}),
                   "on_train_mic": p in room,
                   "room": ("training room" if p in room else "never on the training mic") if p in gh else None,
                   "clips": dict(sorted(Counter(gun_pool(r, gh) for r in rs).items())),
                   "close_mic_clips": sum(mic_of(r) in close for r in rs)}
    ups = {}
    for u in sorted(fsp):
        ups[u] = {"pool": "heldout" if u in held else "train", "splits": sorted(fsp[u])}
        if u in big and "test" in fsp[u]:
            ups[u]["kept_for_training"] = "more than %d eligible clips in %s" % (
                args.cap_uploader, ", ".join(s for s, n in sorted(per[u].items()) if n > args.cap_uploader))
        if per.get(u):
            ups[u]["sets"] = {s: dict(sorted(Counter(fs_pool(r, held) for r in elig[s] if r["username"] == u).items()))
                              for s in sorted(per[u])}
    step = SETS["fs_step"]["sub"]
    cat = {r["category"] for r in rows if r["sub-category"] == step}
    other = {}
    for sub in sorted({r["sub-category"] for r in rows if r["category"] in cat} - {step}):
        rs = [r for r in rows if r["sub-category"] == sub and "ai-generated" not in words(r)]
        other[sub] = {s: tally((fs_pool(r, held), "fs:%s" % r["username"]) for r in rs if surface_of(r) == s)
                      for s in SURFACES}
    return {"fs_heldout": held, "fs_train_side": set(fsp) - held, "fs_big": big, "gun_heldout": gh, "guns": guns,
            "uploaders": ups, "elig": elig, "crow": crow, "plat": plat, "overlap": overlap(rows, crow),
            "steps_other": other, "audio": audio_index(rows, held)}


def tally(items):
    out = defaultdict(lambda: {"clips": 0, "sources": set()})
    for pool, cluster in items:
        out[pool]["clips"] += 1
        out[pool]["sources"].add(cluster)
    return {p: {"clips": v["clips"], "sources": len(v["sources"])} for p, v in sorted(out.items())}


def pool_rows(pools):
    by = {}
    close = set(CLOSE_MICS) | {TRAIN_MIC}
    for sname, rs in pools["elig"].items():
        by[sname] = [(fs_pool(r, pools["fs_heldout"]), "fs:%s" % r["username"], r) for r in rs]
    for ff, sname in FORMS.items():
        by[sname] = [(gun_pool(r, pools["gun_heldout"]), "c3gd:%s" % r["platform_id"], r) for r in pools["crow"]
                     if pools["plat"][r["platform_id"]]["form_factor"] == ff and mic_of(r) in close]
    return by


def thin(n, k, args):
    why = []
    if n < args.min_heldout_clips:
        why.append("%d clips < %d" % (n, args.min_heldout_clips))
    if k < args.min_heldout_sources:
        why.append("%d sources < %d" % (k, args.min_heldout_sources))
    return why


def heldout_info(pools, clips, built, report, args):
    by = pool_rows(pools)
    types = {}
    for t in TYPES:
        ss = [s for s, spec in SETS.items() if spec["type"] == t]
        cand = [(p, c) for s in ss for p, c, _r in by.get(s, [])]
        cs = [c for c in clips if c["set"] in ss]
        n, k = len(cs), len({c["cluster"] for c in cs})
        why = thin(n, k, args)
        types[t] = {"sets": ss, "pools_before_selection": tally(cand),
                    "heldout_built": {"clips": n, "sources": k,
                                      "by_dataset": dict(sorted(Counter(c["dataset"] for c in cs).items()))},
                    "thin": bool(why), "thin_reason": "; ".join(why)}
    surf = {}
    steps = by.get("fs_step", [])
    for s in SURFACES:
        cs = [c for c in clips if c["set"] == "fs_step" and c["surface"] == s]
        n, k = len(cs), len({c["cluster"] for c in cs})
        why = thin(n, k, args)
        surf[s] = {"pools_before_selection": tally((p, c) for p, c, r in steps if surface_of(r) == s),
                   "heldout_built": {"clips": n, "sources": k}, "material_check_built": s in built,
                   "thin": bool(why), "thin_reason": "; ".join(why)}
    train = {}
    for sname, rs in by.items():
        train[sname] = []
        for p, c, r in rs:
            if p not in ("train", "val"):
                continue
            path = r["path"] if "path" in r else os.path.join(C3GD, "data", r["filename"])
            e = {"path": path.replace(os.sep, "/"), "cluster": c, "pool": p}
            if sname == "fs_step":
                e["surface"] = surface_of(r)
            train[sname].append(e)
    esc = [c for c in clips if c["dataset"] == "esc50"]
    return {
        "meta": {"built_by": "scripts/bench/vocab_refs.py --heldout", "refs": args.out_dir.replace(os.sep, "/"),
                 "manifest": os.path.join(args.out_dir, "manifest.json").replace(os.sep, "/"),
                 "cluster_format": "fs:<FoleySet username>, c3gd:<C3GD platform_id>, esc:<ESC-50 Freesound source id>, "
                                   "hear:<HEAR shot id>",
                 "trainable": "a FoleySet clip trains or anchors only when foleyset.uploaders[username].pool is train "
                              "and its split is train (val for early stopping); a C3GD clip only when "
                              "c3gd.guns[platform_id].pool is train, its mic is DAVE at oh_farm and its split is train "
                              "(val for early stopping); ESC-50 and HEAR never; train_clips lists the reference-set "
                              "clips that pass",
                 "route_json": "a route folder for vocab_eval.py --route NAME=DIR holds <event>-<material>_NN.wav for "
                               "seeds 00 to 19 at 0.6 or 1.5 s and a route.json with clusters (every source it trained "
                               "on or anchored from, in cluster_format; empty for none) and optionally sources (file "
                               "paths), steps, prompts {event|material: prompt} and note; vocab_eval stops when a "
                               "cluster or file is held out",
                 "rules": {
                     "foleyset": "an uploader is held out when FoleySet puts any of its clips in split test, unless it "
                                 "holds more than %d eligible clips in one reference set (the per-uploader cap of the "
                                 "eval, so the eval could take only %d of them); a held-out uploader gives every clip "
                                 "to the held-out pool, the rest train on split train; %s" % (
                                     args.cap_uploader, args.cap_uploader, AUDIO_RULE),
                     "c3gd": "a gun (platform_id) is held out when it was never recorded on %s at %s, or when its "
                             "position in its form factor's list of guns on that mic, sorted by caliber and "
                             "platform_id, is a multiple of %d counting from 0 (a form factor with one gun keeps it for "
                             "training); a held-out gun gives every clip on every mic to the held-out pool" % (
                                 TRAIN_MIC[1], TRAIN_MIC[0], args.gun_holdout_every),
                     "esc50": "CC BY-NC, never trained on; an ESC-50 clip is dropped from the held-out sets when "
                              "FoleySet names its Freesound uploader on the training side",
                     "hear": "CC0 far mics, never in the training pool, so all of it is held out"},
                 "pools": POOLS, "train_mic": "|".join(x or "mono" for x in TRAIN_MIC),
                 "thin_rule": "fewer than %d clips or fewer than %d sources in the built held-out set" % (
                     args.min_heldout_clips, args.min_heldout_sources),
                 "args": dict(vars(args))},
        "overlap": pools["overlap"],
        "types": types, "surfaces": surf,
        "thin_types": sorted(t for t, v in types.items() if v["thin"]),
        "thin_surfaces": [s for s, v in surf.items() if v["thin"]],
        "surfaces_in_other_footstep_subcategories": {
            "note": "multi-shot FoleySet recordings outside every reference set; counts by pool and surface from "
                    "original-tags with the same one-surface rule", "counts": pools["steps_other"]},
        "foleyset": {"heldout_uploaders": sorted(pools["fs_heldout"]),
                     "kept_for_training": [u for u in pools["fs_big"] if "test" in pools["uploaders"][u]["splits"]],
                     "uploaders": pools["uploaders"]},
        "c3gd": {"heldout_guns": sorted(pools["gun_heldout"], key=lambda p: gun_order(pools["plat"], p)),
                 "train_guns": [p for p, g in pools["guns"].items() if g["pool"] == "train"], "guns": pools["guns"]},
        "esc50": {"kept": len(esc), "uploader_known_heldout": sum(c["cluster"].startswith("fs:") for c in esc),
                  "uploader_unknown": sum(not c["cluster"].startswith("fs:") for c in esc),
                  "dropped_training_uploader": [e for e in report["excluded"]
                                                if e["reason"] == "Freesound uploader on the training side"]},
        "hear": {"kept": sum(c["dataset"] == "hear" for c in clips)},
        "audio_duplicates": {**report.get("audio_duplicates", {}), "rule": AUDIO_RULE,
                             "dropped": [e for e in report["excluded"] if e["reason"].startswith("audio identical")]},
        "train_clips": train,
    }


def heldout_summary(info):
    W = 132
    print()
    print("=" * W)
    print(" HELD-OUT POOLS  (%s)" % info["meta"]["thin_rule"])
    ov = info["overlap"]
    print("   FoleySet uploaders by split: %s; Freesound recordings by split: %s" % (
        ov["foleyset"]["uploaders_by_splits"], ov["foleyset"]["recordings_by_splits"]))
    print("   C3GD guns by split %s, events %s, recording files %s, the same shot on different mics %s" % (
        ov["c3gd"]["guns_by_splits"], ov["c3gd"]["events_by_splits"], ov["c3gd"]["recording_files_by_splits"],
        ov["c3gd"]["shots_by_splits_across_mics"]))
    fo, cg = info["foleyset"], info["c3gd"]
    print("   FoleySet: %d held-out uploaders, %d kept for training although they touch test: %s" % (
        len(fo["heldout_uploaders"]), len(fo["kept_for_training"]), ", ".join(fo["kept_for_training"])))
    print("   C3GD: %d training guns on the training mic: %s" % (len(cg["train_guns"]), ", ".join(cg["train_guns"])))
    for p in cg["heldout_guns"]:
        g = cg["guns"][p]
        if g["close_mic_clips"]:
            print("     held out %-24s %-8s %-18s %-26s %s" % (p, g["form_factor"], g["caliber"], g["room"],
                                                             ",".join(g["events"])))
    print("   ESC-50: %d kept (%d with a held-out FoleySet uploader, %d with an uploader FoleySet does not name), %d "
          "dropped for a training-side uploader; HEAR %d" % (
              info["esc50"]["kept"], info["esc50"]["uploader_known_heldout"], info["esc50"]["uploader_unknown"],
              len(info["esc50"]["dropped_training_uploader"]), info["hear"]["kept"]))
    ad = info["audio_duplicates"]
    print("   FoleySet byte-identical audio: %d groups over %d files, %d across sub-categories, %d across the two sides; "
          "held-out clips dropped: %s" % (ad.get("groups", 0), ad.get("files_in_groups", 0),
                                          ad.get("groups_across_labels", 0), ad.get("groups_across_pools", 0),
                                          ", ".join("%s %d" % x for x in sorted(Counter(
                                              "%s (%s)" % (e["set"], e["reason"][len("audio identical to "):])
                                              for e in ad["dropped"]).items())) or "none"))
    print("-" * W)
    print("   %-16s %18s %18s %18s %18s   %s" % ("type", "train", "val", "held-out pool", "held-out built", "flag"))
    for t, v in info["types"].items():
        pb = v["pools_before_selection"]
        cells = ["%5d clips %3d src" % (pb.get(p, {}).get("clips", 0), pb.get(p, {}).get("sources", 0))
                 for p in ("train", "val", "heldout")]
        hb = v["heldout_built"]
        print("   %-16s %s %s %s %5d clips %3d src   %s" % (t, cells[0], cells[1], cells[2], hb["clips"], hb["sources"],
                                                          "THIN (%s)" % v["thin_reason"] if v["thin"] else ""))
    for s, v in info["surfaces"].items():
        pb = v["pools_before_selection"]
        cells = ["%5d clips %3d src" % (pb.get(p, {}).get("clips", 0), pb.get(p, {}).get("sources", 0))
                 for p in ("train", "val", "heldout")]
        hb = v["heldout_built"]
        print("   %-16s %s %s %s %5d clips %3d src   %s%s" % (
            "footstep " + s, cells[0], cells[1], cells[2], hb["clips"], hb["sources"],
            "THIN (%s)" % v["thin_reason"] if v["thin"] else "",
            "" if v["material_check_built"] else "  no material check"))
    for sub, by in info["surfaces_in_other_footstep_subcategories"]["counts"].items():
        print("   %s (multi-shot, unused) held-out pool per surface: %s" % (sub, ", ".join(
            "%s %d clips %d src" % (s, v.get("heldout", {}).get("clips", 0), v.get("heldout", {}).get("sources", 0))
            for s, v in by.items())))
    print("   THIN: %s; footstep surfaces %s" % (", ".join(info["thin_types"]) or "none",
                                                ", ".join(info["thin_surfaces"]) or "none"))
    print("=" * W)


def build_list(args, report, pools=None):
    rows = foley_rows()
    users = {r["freesound-id"]: r["username"] for r in rows}
    audio = pools["audio"] if pools else audio_index(rows, None) if args.dedupe_audio else None
    clips = c3gd_clips(args, report, pools and pools["gun_heldout"]) + hear_clips(args, report)
    for sname, spec in SETS.items():
        if spec["dataset"] == "foleyset":
            clips += foley_set(sname, spec, rows, args, report, pools and pools["fs_heldout"], audio)
    used = {(SETS[c["set"]]["type"], c["freesound_id"]) for c in clips if c["dataset"] == "foleyset"}
    for sname, spec in SETS.items():
        if spec["dataset"] == "esc50":
            clips += esc_clips(sname, spec["cls"], users, used, report, pools and pools["fs_train_side"])
    report["esc_uploaders"] = sum(c["dataset"] == "esc50" and c["cluster"].startswith("fs:") for c in clips)
    if audio is not None:
        report["audio_duplicates"] = {k: audio[k] for k in ("groups", "files_in_groups", "groups_across_labels",
                                                            "groups_across_pools")}
    for sub, why in NOT_USED.items():
        n = sum(r["sub-category"] == sub for r in rows)
        report["not_used"][sub] = {"clips": n, "reason": why}
    return clips


def materialize(clips, args, report):
    t0 = time.perf_counter()
    kept = []
    for i, c in enumerate(clips):
        x, sr = read(c["src"])
        c["src_sr"], c["src_frames"] = sr, int(len(x))
        c["orig_s"] = len(x) / float(sr)
        c["clipped_frac"] = float(np.mean(np.abs(x) >= CLIP_LEVEL)) if len(x) else 0.0
        c["peak_s"] = float(np.argmax(np.abs(x)) / float(sr)) if len(x) else 0.0
        c["license"] = LICENSES[c["dataset"]]["license"]
        y = to_sr(x, sr)
        if len(y) == 0 or float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) < 1e-6:
            report["excluded"].append({"dataset": c["dataset"], "set": c["set"], "id": c["id"], "reason": "silent"})
            continue
        c["files"], c["padded"] = {}, {}
        for w, gen in LENGTHS.items():
            z, padded = process(y, float(w), gen)
            p = os.path.join(ROOT, args.out_dir, c["set"], w, "%s.wav" % c["id"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            AudioClip(z, SR).to_wav(p)
            c["files"][w] = rel(p)
            c["padded"][w] = bool(padded)
            c["n_%s" % w] = int(len(z))
        c["src"] = c["src"].replace(os.sep, "/")
        kept.append(c)
        if (i + 1) % 250 == 0:
            print("  processed %d/%d clips  %.1f s" % (i + 1, len(clips), time.perf_counter() - t0), flush=True)
    return kept


def prune(clips, args):
    keep = {os.path.normcase(os.path.join(ROOT, *f.split("/"))) for c in clips for f in c["files"].values()}
    gone = []
    for s in SETS:
        for w in LENGTHS:
            d = os.path.join(ROOT, args.out_dir, s, w)
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                p = os.path.join(d, name)
                if name.endswith(".wav") and os.path.normcase(p) not in keep:
                    os.remove(p)
                    gone.append(rel(p))
    print("removed %d stale wav files from earlier builds" % len(gone))
    return gone


def surfaces(clips, args):
    step = [c for c in clips if c["set"] == "fs_step"]
    built, dropped, counts = {}, {}, {}
    for s in sorted({c["surface"] for c in step}):
        cs = [c for c in step if c["surface"] == s]
        counts[s] = {"clips": len(cs), "sources": len({c["cluster"] for c in cs})}
    for s in SURFACES:
        cs = [c for c in step if c["surface"] == s]
        n, k = len(cs), len({c["cluster"] for c in cs})
        if n >= args.min_surface_clips and k >= args.min_surface_sources:
            built[s] = {"ids": [c["id"] for c in cs], "clips": n, "sources": k,
                        "by_source": dict(Counter(c["username"] for c in cs))}
        else:
            dropped[s] = {"clips": n, "sources": k, "by_source": dict(Counter(c["username"] for c in cs)),
                          "reason": "needs %d clips from %d uploaders" % (args.min_surface_clips,
                                                                         args.min_surface_sources)}
    return built, dropped, counts


def type_table(clips):
    out = {}
    for t, info in TYPES.items():
        ss = [s for s, spec in SETS.items() if spec["type"] == t]
        fam = {SETS[s]["family"] for s in ss}
        cs = [c for c in clips if c["set"] in ss]
        out[t] = {**info, "family": fam.pop(), "sets": ss, "clips": len(cs), "sources": len({c["cluster"] for c in cs})}
    return out


def summary(clips, report, built, dropped, scounts, types):
    W = 132
    print()
    print("=" * W)
    print(" REFERENCE SETS  (mono, %d Hz, onset window by sarge _extract, normalize_loudness at target_for('%s') = %.2f RMS;"
          " sources shorter than the generation length are faded over %d ms and zero-padded first)"
          % (SR, INTENSITY, target_for(INTENSITY), int(FADE_S * 1000)))
    print("=" * W)
    print(" %-22s %-16s %-10s %-12s %5s %7s %8s %8s %8s %8s  %s" % (
        "set", "type", "family", "license", "clips", "sources", "pad0.6", "pad1.5", "oneshot", "clipped", "rule"))
    for sname, spec in SETS.items():
        cs = [c for c in clips if c["set"] == sname]
        cap = report["capped"].get(sname, {})
        print(" %-22s %-16s %-10s %-12s %5d %7d %8d %8d %8s %8d  %s" % (
            sname, spec["type"] or "(negative)", spec["family"], LICENSES[spec["dataset"]]["license"], len(cs),
            len({c["cluster"] for c in cs}), sum(c["padded"]["0.6"] for c in cs), sum(c["padded"]["1.5"] for c in cs),
            str(sum(c.get("oneshot", False) for c in cs)) if spec["dataset"] == "foleyset" else "-",
            sum(c["clipped_frac"] > 0.001 for c in cs), cap.get("rule", "")))
    print(" every set has the same clips at 0.6 s (generation 1.0 s) and 1.5 s (generation 2.0 s): %d clips per length,"
          " %d wav files" % (len(clips), 2 * len(clips)))
    print("-" * W)
    print(" TYPES")
    for t, info in types.items():
        print("   %-16s family %-10s %4d clips %3d sources  %s%s" % (
            t, info["family"], info["clips"], info["sources"], info["note"], "   (approximation)" if info["approx"] else ""))
    print("-" * W)
    print(" C3GD close mics (microphone_locations.csv distance, files, form factors before the per-gun cap of %s)"
          % report["args"]["cap_gun"])
    for k, v in report["c3gd_close_mics"].items():
        print("   %-26s %4.0f m  %4d files  %s" % (k, v["distance_m"], v["files"], v["forms"]))
    fr = defaultdict(Counter)
    pk = defaultdict(list)
    for c in clips:
        if c["dataset"] == "c3gd":
            fr[c["stratum"]][(c["src_sr"], c["src_frames"])] += 1
            pk[c["stratum"]].append(c["peak_s"])
    for k in sorted(fr):
        print("   %-26s header rate and frames %s  median peak at %.3f s" % (k, dict(fr[k]), float(np.median(pk[k]))))
    for s in ("c3gd_pistol", "c3gd_rifle", "c3gd_shotgun"):
        cs = [c for c in clips if c["set"] == s]
        by = Counter(c["platform"] for c in cs)
        print("   %-14s %d guns: %s" % (s, len(by), ", ".join("%s %d" % x for x in sorted(by.items()))))
    print("-" * W)
    print(" FoleySet choices (prefer One-shot when >= %d one-shot clips from >= %d uploaders; cap %s per uploader, spread"
          " over named surfaces first for SingleStep)" % (report["args"]["min_oneshot"],
                                                          report["args"]["min_oneshot_sources"],
                                                          report["args"]["cap_uploader"]))
    for sname, v in report["capped"].items():
        if sname.startswith("fs_"):
            print("   %-14s sub-category %4d  eligible %4d  one-shot %3d  kept %4d  %s%s" % (
                sname, v["before"], v["eligible"], v["oneshot"], v["after"], v["rule"],
                "  (%d dropped by the %s door filter)" % (report["filtered"][sname], DOOR_FILTER[sname])
                if sname in report["filtered"] else ""))
    print("   ESC-50 clips whose Freesound source is in FoleySet: %d clustered by the FoleySet uploader, %d dropped as"
          " the same recording as a FoleySet clip of their type" % (
              report["esc_uploaders"], sum(e["dataset"] == "esc50" and e["reason"].startswith("same Freesound")
                                           for e in report["excluded"])))
    sk = Counter("+".join(c["sword_kind"]) for c in clips if c["set"] == "fs_sword")
    print("   fs_sword content by tags and description: %s" % ", ".join("%s %d" % x for x in sk.most_common()))
    for sub, v in report["not_used"].items():
        print("   %s (%d clips) not used: %s" % (sub, v["clips"], v["reason"]))
    ex = Counter(e["reason"] for e in report["excluded"])
    print("   excluded: %s" % (", ".join("%s %d" % x for x in ex.most_common()) or "none"))
    print("-" * W)
    print(" FOOTSTEP SURFACES from SingleStep original-tags (one surface group named, else none or ambiguous)")
    for s, v in sorted(scounts.items(), key=lambda kv: -kv[1]["clips"]):
        print("   %-34s %4d clips %3d uploaders%s" % (s, v["clips"], v["sources"], "   BUILT" if s in built else ""))
    for s, v in dropped.items():
        print("   %-8s not built: %d clips from %d uploaders (%s)" % (s, v["clips"], v["sources"], v["reason"]))
    print("=" * W)


def parse(argv):
    ap = argparse.ArgumentParser(description="Build processed real reference sets for the SARGE vocabulary eval.")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--sources-json", default=HELDOUT_JSON)
    ap.add_argument("--gun-holdout-every", type=int, default=2)
    ap.add_argument("--min-heldout-clips", type=int, default=30)
    ap.add_argument("--min-heldout-sources", type=int, default=8)
    ap.add_argument("--dedupe-audio", action="store_true")
    ap.add_argument("--cap-gun", type=int, default=20)
    ap.add_argument("--cap-uploader", type=int, default=12)
    ap.add_argument("--min-oneshot", type=int, default=30)
    ap.add_argument("--min-oneshot-sources", type=int, default=10)
    ap.add_argument("--min-surface-clips", type=int, default=8)
    ap.add_argument("--min-surface-sources", type=int, default=3)
    args = ap.parse_args(argv)
    args.dedupe_audio = bool(args.dedupe_audio or args.heldout)
    if args.out_dir is None:
        args.out_dir = HELDOUT_DIR if args.heldout else OUT_DIR
    if args.gun_holdout_every < 1:
        raise SystemExit("--gun-holdout-every takes 1 or more")
    if args.heldout and os.path.normcase(os.path.abspath(os.path.join(ROOT, args.out_dir))) == \
            os.path.normcase(os.path.abspath(os.path.join(ROOT, OUT_DIR))):
        raise SystemExit("--heldout would overwrite %s, the full reference build" % OUT_DIR)
    return args


def main(argv=None):
    args = parse(argv)
    t0 = time.perf_counter()
    report = {"args": vars(args), "capped": {}, "excluded": [], "skipped": [], "filtered": {}, "not_used": {}}
    pools = assign(args, foley_rows()) if args.heldout else None
    clips = build_list(args, report, pools)
    print("selected %d clips, processing at %s s" % (len(clips), " and ".join(LENGTHS)), flush=True)
    clips = materialize(clips, args, report)
    built, dropped, scounts = surfaces(clips, args)
    types = type_table(clips)
    summary(clips, report, built, dropped, scounts, types)
    info = heldout_info(pools, clips, built, report, args) if pools else None
    if info:
        heldout_summary(info)
    man = {"meta": {"root": ROOT, "sr": SR, "intensity": INTENSITY, "target_rms": target_for(INTENSITY),
                    "lengths": {w: {"keep_s": float(w), "gen_s": g} for w, g in LENGTHS.items()},
                    "processing": "AudioClip.from_wav mono, scipy resample_poly to 44100 Hz, sources shorter than the "
                                  "generation length get a %d ms linear fade at their end and zero padding to it, then "
                                  "sarge.generators.stable_audio._extract to the kept length and "
                                  "sarge.loudness.normalize_loudness at target_for('%s'), saved as 16 bit wav"
                                  % (int(FADE_S * 1000), INTENSITY),
                    "datasets": LICENSES, "args": vars(args), "seconds": time.perf_counter() - t0},
           "sets": {s: {**spec, "clips": sum(c["set"] == s for c in clips),
                        "sources": len({c["cluster"] for c in clips if c["set"] == s}),
                        "license": LICENSES[spec["dataset"]]["license"], "selection": report["capped"].get(s, {})}
                    for s, spec in SETS.items()},
           "types": types,
           "surfaces": built, "surfaces_dropped": dropped, "surface_counts": scounts,
           "c3gd_close_mics": report["c3gd_close_mics"],
           "excluded": report["excluded"], "skipped": report["skipped"], "not_used": report["not_used"],
           "filtered": report["filtered"], "esc_fs_uploader_clusters": report["esc_uploaders"],
           "stale_removed": prune(clips, args),
           "clips": clips}
    if "audio_duplicates" in report:
        man["audio_duplicates"] = {**report["audio_duplicates"], "rule": AUDIO_RULE,
                                   "dropped": [e for e in report["excluded"] if e["reason"].startswith("audio identical")]}
    if info:
        man["meta"]["heldout"] = {"sources_json": rel(args.sources_json), "rules": info["meta"]["rules"],
                                  "pools": POOLS, "thin_types": info["thin_types"],
                                  "clusters": {"foleyset": sorted("fs:%s" % u for u in pools["fs_heldout"]),
                                               "c3gd": sorted("c3gd:%s" % g for g in pools["gun_heldout"])}}
        with open(args.sources_json, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=1)
        print("saved -> %s" % args.sources_json)
    p = os.path.join(ROOT, args.out_dir, "manifest.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print("saved -> %s  (%.1f s)" % (p, time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
