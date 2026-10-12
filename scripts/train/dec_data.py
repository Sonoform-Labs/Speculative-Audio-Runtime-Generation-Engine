import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
BENCH = os.path.join(SCRIPTS, "bench")
for _p in (BENCH, SCRIPTS, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap

import argparse
import csv
import hashlib
import json
import pickle
import re
import time
import wave
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt

import dec_common as dc
from vocab_refs import AUDIO_RULE, ESC_NAME, SURF_TERMS, surface_of

DATA = r"D:\datasets\sarge_refs"
FOLEY = os.path.join(DATA, "foleyset", "FoleySet")
C3GD = os.path.join(DATA, "c3gd", "C3GD")
HEAR = os.path.join(DATA, "hear", "tasks", "gunshot_triangulation-v1.0-full")
ANCHOR_MD5 = os.path.join(dc.ROOT, "data", "_gate", "anchor", "audio_md5.json")
NEW_MD5 = os.path.join(dc.OUT, "audio_md5.json")
SCAN = os.path.join(dc.OUT, "scan_v1.pkl")
HEAR_NAME = re.compile(r"^(mic\d+)_(shot\d+)_(fid\d+)\.wav$")
TRAIN_MIC = ("oh_farm", "DAVE", "mono")
NCC_DUP = 0.8
EVAL_Q = 40
GUN_EVAL_MICS = {("oh_farm", "DAVE", "mono"): "close", ("nj_farm", "ALICE", "left"): "close",
                 ("ny_gravel_pit", "ALICE", "left"): "close", ("nj_farm_2", "lilypad_SAMSON", "mono"): "clean",
                 ("oh_farm", "ERIN", "mono"): "clean", ("nj_farm", "JON", "mono"): "clean",
                 ("ny_gravel_pit", "JON", "mono"): "clean"}
CAP_EVAL_UP = 4
STEP_SURF = {"gravel": "step_gravel", "stone": "step_stone", "wood": "step_wood", "metal": "step_metal"}
PRIORITY = ["step_gravel", "step_metal", "step_stone", "step_wood", "glass_ice", "impact_break", "splash_drip",
            "click_mech", "metal_ring", "door_drawer", "step_other", "texture", "human_steady"]
EXPECT_PRE = {"train": 4158, "val": 617}
EXPECT = {"train": 4127, "val": 598, "eval": 750}
EXPECT_ESC_DROP = {"train": 31, "val": 19}
PLAN_STEPS = dc.PLAN_STEPS
PLAN_REAL = dc.PLAN_REAL
MAX_P95 = dc.MAX_PASSES
RULES = r"""HS = scripts/bench/heldout_sources.json
HELD_USERS = HS.foleyset.heldout_uploaders ∪ {u : HS.foleyset.uploaders[u].pool != "train"}        # 418
HELD_GUNS  = HS.c3gd.heldout_guns ∪ {g : HS.c3gd.guns[g].pool != "train"}                          # 43
TRAIN_GUNS = guns − HELD_GUNS                                                                       # 9
VAL_GUNS   = last of each form's TRAIN_GUNS sorted (caliber, platform_id), forms with >= 3 guns     # sig_P320, lmt_ar10
ESC_IDS    = fsids parsed by vocab_refs.ESC_NAME under data/esc50/** and data/real_footsteps/**    # 349
ESC_USERS  = FoleySet usernames owning a fsid in ESC_IDS                                           # 13
EVAL       = data/refs_heldout/manifest.json clips                                                 # 1225
md5(f)     = md5(wave.readframes(all)), raw file bytes on wave error; cache data/_gate/anchor/audio_md5.json
             keyed normcase(path) with [size, int(mtime)]
hard_clip(f) = (max run of samples == max(x) or == min(x)) >= 6 and peak >= 0.25
               or (run of |x| >= 0.999) >= 3
bucket(r)  = C3GD: "gunshot_"+form | HEAR: gunshot_far
           | Footstep rows: first of metal, gravel, stone, wood hit by vocab_refs.SURF_TERMS on the tag set, else step_other
           | else SUB_BUCKET[sub-category]

pool(r):
  FoleySet:
    "ai-generated" in tags                  -> excluded
    user in HELD_USERS                      -> heldout
    split == test                           -> unused
    fsid in ESC_IDS                         -> excluded
    user in ESC_USERS and not R3            -> excluded
    hard_clip(r) and R4                     -> excluded
    user in VAL_USERS                       -> val
    split == val and not R1                 -> unused
    else                                    -> train
  C3GD:
    gun in HELD_GUNS                        -> heldout
    (event,mic,ch) != (oh_farm,DAVE,mono) and not (R2 and event == oh_farm and mic == ERIN) -> unused
    split == test                           -> unused
    mic != DAVE and {split of same (event,file_id,clip_id) on every mic} != {split} -> unused
    gun in VAL_GUNS                         -> val
    split == val                            -> unused
    else                                    -> train
  HEAR                                      -> heldout

VAL_USERS = {u : every FoleySet row of u has split val} − HELD_USERS                                # 113
for b in [step_gravel, step_metal, step_stone, step_wood, glass_ice, impact_break, splash_drip,
          click_mech, metal_ring, door_drawer, step_other, texture, human_steady]:
  rows = surviving FoleySet rows of b; target = ceil(0.10 * len(rows))
  for u in uploaders(rows) sorted by md5(u.encode()).hexdigest():
    stop if val_rows(b) >= target and val_uploaders(b) >= 2
    skip if u in VAL_USERS or u in HS.foleyset.kept_for_training or count_b(u) > max(2, 0.25*len(rows))
    VAL_USERS.add(u)

BLOCK = {md5(r) : pool in {heldout, unused}, minus C3GD training-gun off-mic clips}
      ∪ {md5(r) : excluded for an ESC reason} ∪ {md5(c.src) : c in EVAL}
for r in val   sorted by id: drop if md5 in BLOCK or already seen in val
for r in train sorted by id: drop if md5 in BLOCK or md5 in VAL_MD5 or already seen in train
near-dup: candidates = 10 ms log-RMS envelope r > 0.95 (durations within ±30-80 ms, plus a first-1 s pass)
          against held-out and val FoleySet clips; drop train clip if waveform NCC at 11.025 kHz >= 0.8"""
POST_RULE = ("v2 post-split rule: after the rules above, every train and val clip whose uploader is in ESC_USERS is "
             "dropped (31 train and 19 val clips); every other assignment stays fixed")
ATTRIBUTION = ("Training audio: FoleySet (Wang and Lerch, Zenodo record 20735877, CC BY 4.0, built from CC0 Freesound "
               "clips) and C3GD v0.2.0 (Gurny and Quinn, Zenodo record 22286299, CC BY 4.0). Evaluation only: HEAR 2021 "
               "gunshot_triangulation (Zenodo record 5887964, CC0). Never Sonniss, never ESC-50.")
SOS = {}


def runs(mask):
    if not mask.any():
        return 0
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def run_spans(mask, k):
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(s, e) if b - a >= k]


def sos_for(sr):
    if sr not in SOS:
        SOS[sr] = butter(4, 4000.0, "highpass", fs=sr, output="sos")
    return SOS[sr]


def stats(x, sr):
    info = {"read_sr": int(sr), "read_ch": int(x.shape[1]), "frames": int(x.shape[0])}
    mono = x.mean(axis=1).astype(np.float64)
    if mono.size == 0:
        return info, np.zeros(0, np.float16)
    pk = float(np.max(np.abs(x)))
    r = float(np.sqrt(np.mean(mono * mono)))
    info["peak"] = pk
    info["rms_db"] = float(20 * np.log10(r + 1e-12))
    info["clipped"] = float(np.mean(np.abs(x) >= 0.999))
    info["dc"] = float(np.mean(mono))
    h = sosfilt(sos_for(sr), mono)
    e_all = float(np.sum(mono * mono))
    info["hf_frac"] = float(np.sum(h * h) / e_all) if e_all > 0 else float("nan")
    info["hfk_full"] = dc.kurt(h)
    s, e = dc.window(mono, sr, 0.6)
    info["hfk_06"] = dc.kurt(h[s:e])
    seg = mono[s:e]
    rs = float(np.sqrt(np.mean(seg * seg)))
    info["crest06_db"] = float(20 * np.log10((float(np.max(np.abs(seg))) + 1e-12) / (rs + 1e-12)))
    info["peak_s"] = float(np.argmax(np.abs(mono)) / sr)
    fr = max(1, sr // 100)
    nf = mono.size // fr
    if nf > 0:
        env = np.sqrt(np.mean(mono[: nf * fr].reshape(nf, fr) ** 2, axis=1))
        top = float(env.max())
        info["active_s"] = float(np.sum(env >= top * 10 ** (-40 / 20)) / 100.0) if top > 0 else 0.0
        info["onsets"] = int(np.sum((env[1:] > 4 * np.maximum(env[:-1], top * 1e-3)) & (env[1:] > top * 0.1)))
        fp = np.log10(env + 1e-5).astype(np.float16)
    else:
        info["active_s"] = 0.0
        info["onsets"] = 0
        fp = np.zeros(0, np.float16)
    return info, fp


def raw_md5(path):
    try:
        with wave.open(path, "rb") as w:
            d = w.readframes(w.getnframes())
    except Exception:
        d = open(path, "rb").read()
    return hashlib.md5(d).hexdigest()


class Md5Cache:
    def __init__(self):
        self.anchor = dc.load_json(ANCHOR_MD5) if os.path.isfile(ANCHOR_MD5) else {}
        self.new = dc.load_json(NEW_MD5) if os.path.isfile(NEW_MD5) else {}
        self.hit = 0
        self.miss = 0
        self.dirty = False

    def get(self, path):
        k = os.path.normcase(os.path.abspath(path))
        st = os.stat(path)
        for c in (self.new, self.anchor):
            ent = c.get(k)
            if ent and ent[0] == st.st_size and ent[1] == int(st.st_mtime):
                self.hit += 1
                return ent[2]
        self.miss += 1
        h = raw_md5(path)
        self.new[k] = [st.st_size, int(st.st_mtime), h]
        self.dirty = True
        return h

    def save(self):
        if self.dirty:
            dc.dump_json(self.new, NEW_MD5)
            self.dirty = False


def foley_items():
    rows = list(csv.DictReader(open(os.path.join(FOLEY, "metadata", "foleyset_metadata.csv"), encoding="utf-8")))
    out = []
    for r in rows:
        out.append({"dataset": "foleyset", "id": "fs_%s" % r["name"][:-4], "path": os.path.join(FOLEY, "audio", r["name"]),
                    "name": r["name"], "split": r["split"], "category": r["category"], "sub": r["sub-category"],
                    "oneshot": r["Oneshot-Multi"] == "One-shot", "fsid": r["freesound-id"], "user": r["username"],
                    "tags": r["original-tags"]})
    return out


def c3gd_items():
    rows = list(csv.DictReader(open(os.path.join(C3GD, "metadata.csv"), encoding="utf-8")))
    plat = {r["platform_id"]: r for r in csv.DictReader(open(os.path.join(C3GD, "metadata", "platforms.csv"),
                                                              encoding="utf-8"))}
    out = []
    for r in rows:
        p = plat.get(r["platform_id"], {})
        out.append({"dataset": "c3gd", "id": "c3gd_%s" % r["datum_id"], "path": os.path.join(C3GD, "data", r["filename"]),
                    "split": r["split"], "gun": r["platform_id"], "form": p.get("form_factor", "unknown"),
                    "caliber": r["class_name"], "event": r["event_id"], "mic": r["mic_id"],
                    "channel": r["channel_orientation"] or "mono", "file_id": r["file_id"], "clip_id": r["clip_id"]})
    return out


def hear_items():
    out = []
    base = os.path.join(HEAR, "48000")
    for fold in sorted(os.listdir(base)):
        d = os.path.join(base, fold)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            m = HEAR_NAME.match(name)
            if not m:
                continue
            out.append({"dataset": "hear", "id": "hear_%s" % name[:-4], "path": os.path.join(d, name), "split": fold,
                        "mic": m.group(1), "shot": m.group(2), "fold": fold})
    return out


def scan_file(path, md5c):
    hdr = sf.info(path)
    info = {"sr": int(hdr.samplerate), "ch": int(hdr.channels), "hframes": int(hdr.frames), "subtype": hdr.subtype}
    info["md5"] = md5c.get(path)
    x64, sr = sf.read(path, dtype="float64", always_2d=True)
    x32 = x64.astype(np.float32)
    st, fp = stats(x32, sr)
    info.update(st)
    info["dur"] = info["frames"] / float(info["read_sr"])
    m64 = x64.mean(axis=1) if x64.shape[1] > 1 else x64[:, 0]
    info["run_fs"] = runs(np.abs(m64) >= 0.999)
    xi, _ = sf.read(path, dtype="int32")
    if xi.ndim > 1:
        xi = xi[:, 0]
    if xi.size:
        info["rail_run"] = max(runs(xi == int(xi.max())), runs(xi == int(xi.min())))
    else:
        info["rail_run"] = 0
    return info, fp


def scan_all(items, rescan=False):
    old = {}
    if os.path.isfile(SCAN) and not rescan:
        with open(SCAN, "rb") as fh:
            old = pickle.load(fh)
    md5c = Md5Cache()
    new, fps = {}, []
    t0 = time.perf_counter()
    reused = 0
    for i, it in enumerate(items):
        p = it["path"]
        if not os.path.isfile(p):
            raise SystemExit("missing audio: %s" % p)
        st = os.stat(p)
        k = os.path.normcase(os.path.abspath(p))
        ent = old.get(k)
        if ent and ent[0] == st.st_size and ent[1] == int(st.st_mtime):
            info, fp = ent[2], ent[3]
            reused += 1
        else:
            info, fp = scan_file(p, md5c)
        new[k] = (st.st_size, int(st.st_mtime), info, fp)
        it.update(info)
        fps.append(fp)
        if (i + 1) % 2000 == 0:
            print("scan %d/%d %.1f s reused %d md5 hit %d miss %d" % (i + 1, len(items), time.perf_counter() - t0, reused,
                                                                     md5c.hit, md5c.miss), flush=True)
            md5c.save()
    md5c.save()
    os.makedirs(dc.OUT, exist_ok=True)
    with open(SCAN + ".tmp", "wb") as fh:
        pickle.dump(new, fh)
    os.replace(SCAN + ".tmp", SCAN)
    print("scan done %d items %.1f s reused %d md5 hit %d miss %d" % (len(items), time.perf_counter() - t0, reused,
                                                                      md5c.hit, md5c.miss), flush=True)
    return fps, md5c


def bucket_of(r):
    if r["dataset"] == "c3gd":
        return "gunshot_" + r["form"]
    if r["dataset"] == "hear":
        return "gunshot_far"
    if r["category"] == "Footstep":
        for s in ("metal", "gravel", "stone", "wood"):
            if s in r["surface_any"]:
                return STEP_SURF[s]
        return "step_other"
    return dc.SUB_BUCKET[r["sub"]]


def words(tags):
    return {t.strip().lower() for t in tags.split(",") if t.strip()}


def hid(u):
    return hashlib.md5(u.encode("utf-8")).hexdigest()


def esc_ids():
    ids = {}
    for d in (os.path.join(dc.ROOT, "data", "esc50"), os.path.join(dc.ROOT, "data", "real_footsteps")):
        for base, _ds, files in os.walk(d):
            for f in files:
                m = ESC_NAME.match(f)
                if m:
                    ids.setdefault(m.group(2), os.path.join(base, f))
    return ids


def ncc(pa, pb):
    a, _sa = sf.read(pa, dtype="float64")
    b, _sb = sf.read(pb, dtype="float64")
    a = a.mean(axis=1) if a.ndim > 1 else a
    b = b.mean(axis=1) if b.ndim > 1 else b
    a = resample_poly(a, 1, 4)
    b = resample_poly(b, 1, 4)
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    x = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    return float(np.max(np.abs(x)) / (na * nb))


def _zmat(rs, fps, idx, L):
    m = np.stack([fps[idx[r["id"]]][:L].astype(np.float64) for r in rs])
    m = m - m.mean(axis=1, keepdims=True)
    s = m.std(axis=1, keepdims=True)
    return m / np.maximum(s, 1e-6), (s[:, 0] > 1e-3)


def head_dupes(train, other, fps, idx, thr):
    L = 100
    tr = [r for r in train if len(fps[idx[r["id"]]]) >= L]
    ot = [r for r in other if len(fps[idx[r["id"]]]) >= L]
    A, va = _zmat(tr, fps, idx, L)
    B, vb = _zmat(ot, fps, idx, L)
    out = []
    for i0 in range(0, len(tr), 512):
        C = A[i0:i0 + 512] @ B.T / L
        C[~va[i0:i0 + 512], :] = 0
        C[:, ~vb] = 0
        for i, j in zip(*np.nonzero(C > thr)):
            out.append((tr[i0 + i], ot[j], float(C[i, j]), L))
    return out


def fingerprint_dupes(train, other, fps, idx, thr_long=0.98):
    flagged = []
    tr = sorted(train, key=lambda r: r["dur"])
    ot = sorted(other, key=lambda r: r["dur"])
    od = np.array([r["dur"] for r in ot])
    step = 0.05
    b = tr[0]["dur"] if tr else 0
    while tr and b <= tr[-1]["dur"] + 1e-9:
        grp = [r for r in tr if b <= r["dur"] < b + step]
        b += step
        if not grp:
            continue
        lo, hi = grp[0]["dur"] - 0.03, grp[-1]["dur"] + 0.03
        cand = [ot[i] for i in np.flatnonzero((od >= lo) & (od <= hi))]
        if not cand:
            continue
        L = int(min(min(len(fps[idx[r["id"]]]) for r in grp), min(len(fps[idx[r["id"]]]) for r in cand)))
        if L < 25:
            continue
        A, va = _zmat(grp, fps, idx, L)
        B, vb = _zmat(cand, fps, idx, L)
        C = A @ B.T / L
        C[~va, :] = 0
        C[:, ~vb] = 0
        thr = thr_long if L >= 50 else 0.995
        for i, j in zip(*np.nonzero(C > thr)):
            if grp[i]["id"] != cand[j]["id"]:
                flagged.append((grp[i], cand[j], float(C[i, j]), L))
    return flagged


def assign(items, fps, args):
    for r in items:
        r["flat"] = bool((r["rail_run"] >= 6 and r.get("peak", 0) >= 0.25) or r["run_fs"] >= 3)
    idx = {r["id"]: i for i, r in enumerate(items)}
    by_path = {dc.norm(r["path"]): r for r in items}
    hs = dc.load_json(dc.HELDOUT_JSON)
    guns = hs["c3gd"]["guns"]
    big = set(hs["foleyset"]["kept_for_training"])
    held_users, held_guns = dc.held_sources(hs)
    train_guns = sorted(set(guns) - held_guns)
    plat = {g: (v["form_factor"], v["caliber"], g) for g, v in guns.items()}
    by_form = defaultdict(list)
    for g in sorted(train_guns, key=lambda g: plat[g]):
        by_form[plat[g][0]].append(g)
    val_guns = {gs[-1] for gs in by_form.values() if len(gs) >= 3}
    esc = esc_ids()
    fs_rows = [r for r in items if r["dataset"] == "foleyset"]
    owner = {}
    for r in fs_rows:
        owner.setdefault(r["fsid"], r["user"])
    esc_users = {owner[i] for i in esc if i in owner}
    user_splits = defaultdict(set)
    for r in fs_rows:
        user_splits[r["user"]].add(r["split"])
    val_users = {u for u, s in user_splits.items() if s == {"val"} and u not in held_users}
    for r in items:
        r["cluster"] = ("fs:%s" % r["user"]) if r["dataset"] == "foleyset" else (
            "c3gd:%s" % r["gun"] if r["dataset"] == "c3gd" else "hear:%s" % r["shot"])
        if r["dataset"] == "foleyset":
            r["surface"] = surface_of({"original-tags": r["tags"]}) if r["category"] == "Footstep" else None
            w = words(r["tags"])
            r["surface_any"] = sorted(s for s in ("gravel", "stone", "wood", "metal") if w & SURF_TERMS[s]) \
                if r["category"] == "Footstep" else []
        r["bucket"] = bucket_of(r)
    shot_splits = defaultdict(set)
    for r in items:
        if r["dataset"] == "c3gd":
            r["cshot"] = "%s:%s:%s" % (r["event"], r["file_id"], r["clip_id"])
            shot_splits[r["cshot"]].add(r["split"])
    oh_mics = [m for m in args.oh_mics.split(",") if m]

    def side_pool(r):
        if r["dataset"] == "foleyset":
            if "ai-generated" in words(r["tags"]):
                return "excluded", "ai-generated tag"
            if r["user"] in held_users:
                return "heldout", "held-out uploader"
            if r["split"] == "test":
                return "unused", "split test of a training-side uploader"
            if r["fsid"] in esc:
                return "excluded", "same Freesound recording as an ESC-50 file"
            if r["user"] in esc_users and not args.esc_uploaders_ok:
                return "excluded", "uploader of an ESC-50 file"
            if args.drop_flat_fs and r["flat"]:
                return "excluded", "hard-clipped (symmetric rails)"
            return "side", ""
        if r["dataset"] == "c3gd":
            mic = (r["event"], r["mic"], r["channel"])
            if r["gun"] in held_guns:
                return "heldout", "held-out gun"
            if mic != TRAIN_MIC and not (r["event"] == "oh_farm" and r["mic"] in oh_mics):
                return "unused", "training gun off the training mic"
            if r["split"] == "test":
                return "unused", "split test of a training gun"
            if mic != TRAIN_MIC and shot_splits[r["cshot"]] != {r["split"]}:
                return "unused", "shot split differs across mics"
            return "side", ""
        return "heldout", "HEAR is held out"

    for r in items:
        r["pool"], r["why"] = side_pool(r)
    side = [r for r in fs_rows if r["pool"] == "side"]
    vset = set(val_users)
    bucket_rows = defaultdict(list)
    for r in side:
        if not (r["split"] == "val" and r["user"] not in vset and not args.val_split_train):
            bucket_rows[r["bucket"]].append(r)
    picked_for = {}
    for b in PRIORITY:
        rs = bucket_rows[b]
        n = len(rs)
        cnt = Counter(r["user"] for r in rs)
        target = max(1, int(np.ceil(0.10 * n)))
        cur = [r for r in rs if r["user"] in vset]
        srcs = {r["user"] for r in cur}
        for u in sorted(cnt, key=hid):
            if len(cur) >= target and len(srcs) >= 2:
                break
            if u in vset or u in big or cnt[u] > max(2, 0.25 * n):
                continue
            vset.add(u)
            picked_for[u] = b
            cur += [r for r in rs if r["user"] == u]
            srcs.add(u)
    for r in items:
        if r["pool"] != "side":
            continue
        if r["dataset"] == "foleyset":
            if r["user"] in vset:
                r["pool"], r["why"] = "val", ("uploader with split val only" if r["user"] in val_users else
                                              "validation uploader picked for %s" % picked_for[r["user"]])
            elif r["split"] == "val" and not args.val_split_train:
                r["pool"], r["why"] = "unused", "split val of an uploader that trains"
            else:
                r["pool"], r["why"] = "train", "split %s, training-side uploader" % r["split"]
        else:
            if r["gun"] in val_guns:
                r["pool"], r["why"] = "val", "validation gun"
            elif r["split"] == "val":
                r["pool"], r["why"] = "unused", "split val of a training gun"
            else:
                r["pool"], r["why"] = "train", "split train, training gun"
    man = dc.load_json(dc.REFS_HELDOUT)
    md5c = Md5Cache()

    def md5s(clips):
        out = set()
        for c in clips:
            p = dc.norm(c["src"])
            out.add(by_path[p]["md5"] if p in by_path else md5c.get(c["src"]))
        return out

    eval_md5 = md5s(man["clips"])
    md5c.save()
    owners = defaultdict(set)
    for r in items:
        if r["pool"] in ("heldout", "unused") and r["why"] != "training gun off the training mic":
            owners[r["md5"]].add(r["pool"])
        elif r["pool"] == "excluded" and r["why"].startswith(("same Freesound", "uploader of an ESC")):
            owners[r["md5"]].add("excluded")
    for h in eval_md5:
        owners[h].add("eval")
    seen, val_md5 = set(), set()
    for r in sorted([r for r in items if r["pool"] == "val"], key=lambda r: r["id"]):
        if r["md5"] in owners:
            r["pool"], r["why"] = "dropped", "val audio identical to a %s clip" % "/".join(sorted(owners[r["md5"]]))
        elif r["md5"] in seen:
            r["pool"], r["why"] = "dropped", "val audio identical to an earlier val clip"
        else:
            seen.add(r["md5"])
            val_md5.add(r["md5"])
    seen = set()
    for r in sorted([r for r in items if r["pool"] == "train"], key=lambda r: r["id"]):
        if r["md5"] in owners:
            r["pool"], r["why"] = "dropped", "train audio identical to a %s clip" % "/".join(sorted(owners[r["md5"]]))
        elif r["md5"] in val_md5:
            r["pool"], r["why"] = "dropped", "train audio identical to a val clip"
        elif r["md5"] in seen:
            r["pool"], r["why"] = "dropped", "train audio identical to an earlier train clip"
        else:
            seen.add(r["md5"])
    train = [r for r in items if r["pool"] == "train"]
    other = [r for r in items if r["dataset"] == "foleyset" and r["pool"] in ("heldout", "val")]
    trf = [r for r in train if r["dataset"] == "foleyset"]
    t0 = time.perf_counter()
    flagged = fingerprint_dupes(trf, other, fps, idx, 0.95)
    flagged += head_dupes(trf, other, fps, idx, 0.95)
    cand = {}
    for a, b, c, L in flagged:
        k = (a["id"], b["id"])
        if k not in cand or cand[k][2] < c:
            cand[k] = (a, b, c, L)
    print("near-dup candidates %d (%.1f s); waveform ncc..." % (len(cand), time.perf_counter() - t0), flush=True)
    fl = []
    for (_ia, _ib), (a, b, c, L) in sorted(cand.items()):
        fl.append({"train": a["id"], "train_user": a["user"], "other": b["id"], "other_user": b["user"],
                   "other_pool": b["pool"], "env_r": round(c, 4), "frames": L, "ncc": round(ncc(a["path"], b["path"]), 4)})
    near = {p["train"]: p for p in fl if p["ncc"] >= NCC_DUP}
    for r in train:
        r["near_dupe"] = r["id"] in near
        if r["near_dupe"]:
            r["pool"], r["why"] = "dropped", "train audio a near-duplicate of a %s clip (waveform ncc >= %.1f)" % (
                near[r["id"]]["other_pool"], NCC_DUP)
    print("near-dup pass %.1f s, dropped %d" % (time.perf_counter() - t0, len(near)), flush=True)
    held = [r for r in items if r["pool"] == "heldout"]
    side_md5 = {r["md5"] for r in items if r["dataset"] == "foleyset" and r["user"] not in held_users}
    md5_subs = defaultdict(set)
    for r in items:
        if r["dataset"] == "foleyset":
            md5_subs[r["md5"]].add(r.get("sub"))
    dec_eval = []
    seen = set()
    for b in sorted({r["bucket"] for r in held if r["dataset"] == "foleyset"}):
        rs = sorted([r for r in held if r["bucket"] == b and r["dataset"] == "foleyset"],
                    key=lambda r: (r["flat"], hid(r["id"])))
        per = Counter()
        n = 0
        for r in rs:
            if (n >= EVAL_Q or r["md5"] in seen or per[r["user"]] >= CAP_EVAL_UP or r["md5"] in side_md5
                    or len(md5_subs[r["md5"]]) > 1):
                continue
            seen.add(r["md5"])
            per[r["user"]] += 1
            n += 1
            dec_eval.append((b, "real", r))
    for form in ("pistol", "rifle", "shotgun", "other"):
        for cls in ("close", "clean"):
            rs = [r for r in held if r["dataset"] == "c3gd" and GUN_EVAL_MICS.get((r["event"], r["mic"], r["channel"])) == cls
                  and (r["form"] == form or (form == "other" and r["form"] not in ("pistol", "rifle", "shotgun")))]
            rs = sorted(rs, key=lambda r: hid(r["id"]))
            per = Counter()
            n = 0
            for r in rs:
                k = (r["gun"], r["mic"], r["channel"], r["event"])
                if n >= EVAL_Q or r["md5"] in seen or per[k] >= CAP_EVAL_UP:
                    continue
                seen.add(r["md5"])
                per[k] += 1
                n += 1
                dec_eval.append(("gunshot_" + form, cls, r))
    for r in sorted([r for r in held if r["dataset"] == "hear"], key=lambda r: r["id"]):
        dec_eval.append(("gunshot_far", "far", r))
    block = set(owners)
    info = {"held_users": held_users, "held_guns": held_guns, "val_guns": sorted(val_guns),
            "train_guns": [g for g in train_guns if g not in val_guns], "esc_ids": len(esc), "esc_users": esc_users,
            "val_users": len(vset), "val_users_split_val_only": len(val_users), "picked_for": picked_for,
            "eval_md5": eval_md5, "block": block, "near": fl, "near_dropped": len(near), "refs_heldout_clips": len(man["clips"]),
            "eval_paths": {dc.norm(c["src"]) for c in man["clips"]}}
    return dec_eval, info


def check_csv(items, path):
    rows = {r["id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    mine = {r["id"]: r for r in items}
    missing = sorted(set(rows) - set(mine))
    extra = sorted(set(mine) - set(rows))
    pool = sorted(i for i in set(rows) & set(mine) if rows[i]["pool"] != mine[i]["pool"])
    buck = sorted(i for i in set(rows) & set(mine) if rows[i]["bucket"] != mine[i]["bucket"])
    rep = {"csv": path, "rows": len(rows), "missing_ids": len(missing), "extra_ids": len(extra),
           "pool_diffs": len(pool), "bucket_diffs": len(buck),
           "examples": {"missing": missing[:5], "extra": extra[:5],
                        "pool": [(i, rows[i]["pool"], mine[i]["pool"], mine[i]["why"]) for i in pool[:5]],
                        "bucket": [(i, rows[i]["bucket"], mine[i]["bucket"]) for i in buck[:5]]}}
    print("CHECK %s" % json.dumps(rep), flush=True)
    return rep


def check_eval_csv(dec_eval, path):
    rows = [(r["id"], r["bucket"], r["subset"]) for r in csv.DictReader(open(path, encoding="utf-8"))]
    mine = [(r["id"], b, c) for b, c, r in dec_eval]
    rep = {"csv": path, "rows": len(rows), "mine": len(mine), "id_diffs": len(set(r[0] for r in rows) ^ set(m[0] for m in mine)),
           "row_diffs": len(set(rows) ^ set(mine)), "order_equal": rows == mine}
    print("CHECK-EVAL %s" % json.dumps(rep), flush=True)
    return rep


def clip_weights(recs):
    med = {}
    by = defaultdict(list)
    for r in recs:
        if r["set"] == "train" and np.isfinite(r["hfk_06"]):
            by[r["bucket"]].append(r["hfk_06"])
    for b, v in by.items():
        med[b] = float(np.median(v))
    for r in recs:
        k = r["hfk_06"]
        m = med.get(r["bucket"])
        f = 1.0 if (m is None or not np.isfinite(k) or m <= 0) else float(np.clip(np.sqrt(k / m), 0.5, 2.0))
        r["weight"] = round(min(r["active_s"], 3.0) * f, 5)
    return med


def summarize(rs):
    k = np.array([r.get("hfk_06", np.nan) for r in rs], float)
    return {"clips": len(rs), "hours": round(sum(r["dur"] for r in rs) / 3600.0, 3),
            "active_h": round(sum(r["active_s"] for r in rs) / 3600.0, 3), "sources": len({r["cluster"] for r in rs}),
            "hfk06_med": round(float(np.nanmedian(k)), 1) if len(rs) else None,
            "flat": int(sum(r["flat"] for r in rs))}


def record_of(r, setname, subset=None):
    lic = dc.LICENSES[r["dataset"]]["license"]
    out = {"id": r["id"], "dataset": r["dataset"], "set": setname, "pool": r["pool"], "why": r["why"],
           "bucket": r["bucket"], "cluster": r["cluster"], "split": r.get("split"), "sub": r.get("sub"),
           "form": r.get("form"), "event": r.get("event"), "mic": r.get("mic"), "channel": r.get("channel"),
           "dur": round(float(r["dur"]), 5), "active_s": float(r["active_s"]),
           "hfk_06": float(r["hfk_06"]) if np.isfinite(r.get("hfk_06", np.nan)) else float("nan"),
           "flat": bool(r["flat"]), "md5": r["md5"], "path": r["path"], "src_sr": int(r["read_sr"]), "license": lic}
    if subset is not None:
        out["subset"] = subset
        out["half"] = dc.half_of(r["cluster"])
    return out


def cmd_manifest(args):
    t0 = time.perf_counter()
    items = foley_items() + c3gd_items() + hear_items()
    print("items %d (foleyset %d, c3gd %d, hear %d)" % (len(items), sum(r["dataset"] == "foleyset" for r in items),
                                                       sum(r["dataset"] == "c3gd" for r in items),
                                                       sum(r["dataset"] == "hear" for r in items)), flush=True)
    fps, _m = scan_all(items, args.rescan)
    dec_eval, info = assign(items, fps, args)
    pre = Counter(r["pool"] for r in items)
    print("pre-drop pools %s" % dict(sorted(pre.items())), flush=True)
    checks = {}
    if args.check:
        checks["check"] = check_csv(items, args.check)
    if args.check_eval:
        checks["check_eval"] = check_eval_csv(dec_eval, args.check_eval)
    fail = []
    if args.check and any(checks["check"][k] for k in ("missing_ids", "extra_ids", "pool_diffs", "bucket_diffs")):
        fail.append("check differs from %s" % args.check)
    if args.check_eval and (checks["check_eval"]["row_diffs"] or checks["check_eval"]["id_diffs"]):
        fail.append("check-eval differs from %s" % args.check_eval)
    for k, v in EXPECT_PRE.items():
        if pre.get(k, 0) != v and args.strict:
            fail.append("pre-drop %s %d != %d" % (k, pre.get(k, 0), v))
    if fail:
        raise SystemExit("manifest STOP: %s" % "; ".join(fail))
    esc_users = info["esc_users"]
    esc_drop = Counter()
    for r in items:
        if r["pool"] in ("train", "val") and r["dataset"] == "foleyset" and r["user"] in esc_users:
            esc_drop[r["pool"]] += 1
            r["pool"], r["why"] = "dropped", "uploader of an ESC-50 recording (v2 post-split rule)"
    train = sorted([r for r in items if r["pool"] == "train"], key=lambda r: r["id"])
    val = sorted([r for r in items if r["pool"] == "val"], key=lambda r: r["id"])
    recs = [record_of(r, "train") for r in train] + [record_of(r, "val") for r in val] + \
           [record_of(r, "eval", subset=c) | {"bucket": b, "in_refs_heldout": dc.norm(r["path"]) in info["eval_paths"]}
            for b, c, r in dec_eval]
    counts = Counter(r["set"] for r in recs)
    print("esc-uploader drop %s -> final %s" % (dict(esc_drop), dict(counts)), flush=True)
    if args.strict:
        bad = [k for k, v in EXPECT.items() if counts.get(k, 0) != v] + \
              ["esc_%s" % k for k, v in EXPECT_ESC_DROP.items() if esc_drop.get(k, 0) != v]
        if bad:
            raise SystemExit("manifest STOP: counts %s esc drop %s differ from expected %s / %s" % (
                dict(counts), dict(esc_drop), EXPECT, EXPECT_ESC_DROP))
    med = clip_weights(recs)
    tv = [r for r in recs if r["set"] == "train"]
    passes = dc.expected_passes(tv, PLAN_STEPS, PLAN_REAL)
    for m, v in passes.items():
        print("  passes %-13s %s" % (m, v), flush=True)
    worst = max(passes.items(), key=lambda kv: kv[1]["passes_p95"])
    by_bucket = defaultdict(list)
    for r in recs:
        by_bucket[(r["set"], r["bucket"])].append(r)
    meta = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "built_by": "scripts/train/dec_data.py manifest",
            "rules": RULES, "post_split_rule": POST_RULE, "audio_rule": AUDIO_RULE,
            "flags": {"oh_mics": args.oh_mics, "val_split_train": args.val_split_train,
                      "drop_flat_fs": args.drop_flat_fs, "esc_uploaders_ok": args.esc_uploaders_ok},
            "heldout_sources_md5": dc.file_md5(dc.HELDOUT_JSON), "snapshot": dc.SNAP_ID, "licenses": dc.LICENSES,
            "attribution": ATTRIBUTION, "val_guns": info["val_guns"], "train_guns": info["train_guns"],
            "held_users": len(info["held_users"]), "held_guns": len(info["held_guns"]), "esc_ids": info["esc_ids"],
            "esc_users": sorted(esc_users), "esc_users_dropped": sorted(esc_users), "esc_drop": dict(esc_drop),
            "val_users": info["val_users"], "val_users_split_val_only": info["val_users_split_val_only"],
            "block_md5": sorted(info["block"]), "refs_heldout_md5": sorted(info["eval_md5"]),
            "refs_heldout_clips": info["refs_heldout_clips"], "near_dupes": {"candidates": len(info["near"]),
                                                                             "dropped": info["near_dropped"],
                                                                             "ncc_ge_0.5": [p for p in info["near"] if p["ncc"] >= 0.5]},
            "pool_reasons": {ds: {"%s | %s" % k: v for k, v in sorted(Counter((r["pool"], r["why"]) for r in items
                                                                              if r["dataset"] == ds).items())}
                             for ds in ("foleyset", "c3gd", "hear")},
            "pre_drop_pools": dict(pre), "counts": dict(counts), "checks": checks, "hfk06_bucket_median": med,
            "passes": {"steps": PLAN_STEPS, "real_per_batch": PLAN_REAL, "by_bucket": passes, "max_p95": MAX_P95,
                       "rule": "clip weight min(active_s, 3) x clamp(sqrt(hfk_06 / bucket median), 0.5, 2); source cap "
                               "max(10%%, 1.5 / n_sources, 0.5 x active-second share); clip cap so that expected passes "
                               "(draws x min(1, crop / active_s)) stay <= %.0f at the planned run" % MAX_P95},
            "buckets": {"%s|%s" % k: summarize(v) for k, v in sorted(by_bucket.items())},
            "eval_halves": dict(Counter(r["half"] for r in recs if r["set"] == "eval")),
            "eval_subsets": dict(Counter("%s|%s" % (r["bucket"], r["subset"]) for r in recs if r["set"] == "eval"))}
    man = {"meta": meta, "records": recs}
    rep = dc.audit(man)
    hs_cl = {"fs:" + u for u in info["held_users"]} | {"c3gd:" + g for g in info["held_guns"]}
    tvr = [r for r in recs if r["set"] in ("train", "val")]
    ev = [r for r in recs if r["set"] == "eval"]
    overlap = {"train_val_clusters_held_out": len({r["cluster"] for r in tvr} & hs_cl),
               "train_val_clusters_esc_users": len({r["cluster"] for r in tvr} & {"fs:" + u for u in esc_users}),
               "train_val_md5_in_block": sum(r["md5"] in info["block"] for r in tvr),
               "train_val_md5_in_refs_heldout": sum(r["md5"] in info["eval_md5"] for r in tvr),
               "train_val_paths_in_refs_heldout": sum(dc.norm(r["path"]) in info["eval_paths"] for r in tvr),
               "eval_md5_in_train_val": len({r["md5"] for r in ev} & {r["md5"] for r in tvr}),
               "eval_clusters_in_train_val": len({r["cluster"] for r in ev} & {r["cluster"] for r in tvr}),
               "train_val_md5_duplicates": len(tvr) - len({r["md5"] for r in tvr})}
    meta["audit"] = rep
    meta["overlap"] = overlap
    print("AUDIT %s" % json.dumps(rep), flush=True)
    print("OVERLAP %s" % json.dumps(overlap), flush=True)
    if any(overlap.values()):
        raise SystemExit("manifest STOP: overlap %s" % overlap)
    dc.dump_json(man, dc.MANIFEST)
    for s in ("train", "val", "eval"):
        rs = [r for r in recs if r["set"] == s]
        print("%-5s %5d clips %.3f h active %.3f h sources %d" % (
            s, len(rs), sum(r["dur"] for r in rs) / 3600, sum(r["active_s"] for r in rs) / 3600,
            len({r["cluster"] for r in rs})), flush=True)
    print("wrote %s (%.1f s)" % (dc.MANIFEST, time.perf_counter() - t0), flush=True)
    if worst[1]["passes_p95"] > MAX_P95:
        raise SystemExit("manifest STOP: bucket %s expected passes p95 %.1f > %.0f for %d steps x %d real crops" % (
            worst[0], worst[1]["passes_p95"], MAX_P95, PLAN_STEPS, PLAN_REAL))
    return 0


def read_mono(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    return x.mean(axis=1), int(sr)


def half_len_out(up, down):
    return int(np.ceil(10 * max(up, down) / down))


def clip_runs_of(path, sr):
    xi, _ = sf.read(path, dtype="int32", always_2d=True)
    xi = xi[:, 0].astype(np.int64)
    if not xi.size:
        return []
    hi, lo = int(xi.max()), int(xi.min())
    q = 0.25 * 2 ** 31
    m = np.zeros(xi.size, bool)
    if abs(hi) >= q:
        m |= xi == hi
    if abs(lo) >= q:
        m |= xi == lo
    spans = run_spans(m, 6) + run_spans(np.abs(xi) >= 0.999 * 2 ** 31, 3)
    if not spans:
        return []
    if sr == dc.SR:
        r, pad = 1.0, 22
    else:
        g = np.gcd(dc.SR, sr)
        up, down = dc.SR // g, sr // g
        r, pad = up / down, 22 + half_len_out(up, down)
    out = []
    for a, b in sorted(spans):
        a2 = max(0, int(np.floor(a * r)) - pad)
        b2 = int(np.ceil(b * r)) + pad
        if out and a2 <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b2)
        else:
            out.append([a2, b2])
    return out


def prep_audio(path):
    x, sr = read_mono(path)
    if sr != dc.SR:
        g = np.gcd(dc.SR, sr)
        x = resample_poly(x, dc.SR // g, sr // g)
    x = x - x.mean()
    if x.size > 32:
        x = sosfiltfilt(butter(2, 10.0, "highpass", fs=dc.SR, output="sos"), x)
    return x.astype(np.float32), sr


def cmd_cache(args):
    man = dc.load_json(dc.MANIFEST)
    dc.audit(man)
    os.makedirs(dc.CACHE, exist_ok=True)
    index = {"meta": {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "sr": dc.SR, "manifest_md5": dc.file_md5(dc.MANIFEST),
                      "prep": "mono mean, resample_poly(147, 160) for 48 kHz, mean removed, 10 Hz 2nd-order high-pass "
                              "(sosfiltfilt); onsets = loudest 5 ms plus up to 7 high-band flux peaks (n_fft 256, hop 64, "
                              ">= 0.3 x max, >= 50 ms apart); clip_runs found at the source rate (rail runs >= 6 or "
                              "|x| >= 0.999 runs >= 3), mapped to 44.1 kHz and widened by 22 samples plus the resampler "
                              "half-length", "limit": args.limit}}
    t0 = time.perf_counter()
    for s in ("train", "val", "eval"):
        rs = [r for r in man["records"] if r["set"] == s]
        if args.limit:
            rs = rs[:args.limit]
        path = os.path.join(dc.CACHE, "%s.f32" % s)
        ents = []
        off = 0
        with open(path + ".tmp", "wb") as fh:
            for i, r in enumerate(rs):
                x, sr = prep_audio(r["path"])
                runs_ = clip_runs_of(r["path"], sr) if r["flat"] else []
                ons = dc.flux_onsets(x)
                pk = float(np.max(np.abs(x))) if x.size else 0.0
                x.tofile(fh)
                ent = {"id": r["id"], "off": off, "n": int(x.size), "bucket": r["bucket"], "mix": dc.mix_of(r["bucket"]),
                       "cluster": r["cluster"], "dataset": r["dataset"], "flat": r["flat"],
                       "adv": not (r["flat"] and r["dataset"] == "c3gd"), "weight": r.get("weight", 1.0),
                       "active_s": r["active_s"], "hfk_06": r["hfk_06"], "onsets": ons, "clip_runs": runs_,
                       "peak": pk, "md5": r["md5"], "sub": r.get("sub")}
                if s == "eval":
                    ent["subset"] = r["subset"]
                    ent["half"] = r["half"]
                ents.append(ent)
                off += int(x.size)
                if (i + 1) % 1000 == 0:
                    print("cache %s %d/%d %.1f s" % (s, i + 1, len(rs), time.perf_counter() - t0), flush=True)
        os.replace(path + ".tmp", path)
        index[s] = ents
        print("cache %-5s %5d clips %8.3f h %7.1f MB  flat %d  clip-run clips %d  no-adv %d" % (
            s, len(ents), off / dc.SR / 3600, off * 4 / 1e6, sum(e["flat"] for e in ents),
            sum(bool(e["clip_runs"]) for e in ents), sum(not e["adv"] for e in ents)), flush=True)
    dc.dump_json(index, os.path.join(dc.CACHE, "index.json"))
    print("wrote %s (%.1f s)" % (dc.CACHE, time.perf_counter() - t0), flush=True)
    return 0


def parse(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--check", default=None)
    m.add_argument("--check-eval", default=None)
    m.add_argument("--oh-mics", default="ERIN")
    m.add_argument("--val-split-train", action=argparse.BooleanOptionalAction, default=True)
    m.add_argument("--drop-flat-fs", action=argparse.BooleanOptionalAction, default=True)
    m.add_argument("--esc-uploaders-ok", action=argparse.BooleanOptionalAction, default=True)
    m.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    m.add_argument("--rescan", action="store_true")
    c = sub.add_parser("cache")
    c.add_argument("--limit", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    for a in ("check", "check_eval"):
        if getattr(args, a, None):
            setattr(args, a, dc.at_root(getattr(args, a)))
    if args.cmd == "manifest":
        return cmd_manifest(args)
    return cmd_cache(args)


if __name__ == "__main__":
    raise SystemExit(main())
