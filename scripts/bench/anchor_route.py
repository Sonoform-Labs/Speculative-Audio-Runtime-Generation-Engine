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
import time
import wave
from collections import Counter, defaultdict

import numpy as np
from scipy.io import wavfile

import vocab_eval as ve
from sarge.generators.stable_audio import _extract, _seed, build_prompt
from sarge.loudness import normalize_loudness, rms, target_for
from sarge.static_guard import is_static, retry_seed
from sarge.types import AudioClip, CacheKey
from vocab_eval import CURRENT, INTENSITY, KEYS, SETTINGS, SR, dedupe, free_gpu, gen_info, gen_paths, key_name, rel, \
    shape_rows, spaced, wav
from vocab_refs import C3GD, ESC, ESC_NAME, FOLEY, FORMS, LICENSES, SETS as REF_SETS, pad_to, surface_of, to_sr, words
from validate_panel import _resample
from ab_clap import CLAP_NAME
from water_judge import _release, _torch, pct

ROOT = ve.ROOT
CUR = {k: SETTINGS[CURRENT][k] for k in ("steps", "detail_steps", "gen_duration_s", "duration_s")}
GEN_S, KEEP_S = float(CUR["gen_duration_s"]), float(CUR["duration_s"])
N_GEN, N_KEEP = int(round(GEN_S * SR)), int(round(KEEP_S * SR))
OUT_DIR = os.path.join("data", "_gate", "anchor")
HELDOUT = os.path.join("scripts", "bench", "heldout_sources.json")
ESC_DIRS = [ESC, os.path.join(ROOT, "data", "real_footsteps")]
VERSION = "anchor_route_v1"
METRICS = ("clap", "mel", "wave", "env")
SHAPE_SHOW = ("attack_ms", "tail_ms", "floor_db", "decay100_db")
N_FFT, HOP, N_MELS = 1024, 441, 64
NFFT_W = 1 << int(np.ceil(np.log2(2 * N_KEEP)))
CAUTION = ("research only: an audit flagged EA patent US12420192B2 as mapping onto an init_audio anchor-bank route, so"
           " nothing here ships before a freedom-to-operate read")
POOL_WORDS = ("pool", "train", "training", "anchor", "anchors", "library", "fit", "allow", "allowed", "include")
HELD_PREFIX = ("held", "holdout", "eval", "test", "val", "exclu", "block", "reserved")
MANIFEST_KEYS = ("manifest", "manifest_path", "refs_manifest", "heldout_manifest", "eval_manifest")
FIELD_TOKENS = {"user": "user", "users": "user", "username": "user", "usernames": "user", "uploader": "user",
                "uploaders": "user", "freesound": "fsid", "fsid": "fsid", "fsids": "fsid", "platform": "gun",
                "platforms": "gun", "gun": "gun", "guns": "gun", "cluster": "cluster", "clusters": "cluster",
                "source": "cluster", "sources": "cluster", "recording": "rec", "recordings": "rec"}
NAN = float("nan")


def level(x):
    return normalize_loudness(x, target_rms=target_for(INTENSITY))


def safe(s, n=None):
    s = str(s).encode("ascii", "replace").decode("ascii")
    return s if n is None or len(s) <= n else s[: n - 2] + ".."


def med(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else NAN


def qt(v, p):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, p)) if v.size else NAN


def dist(v):
    return {"median": med(v), "p10": qt(v, 10), "p90": qt(v, 90), "n": int(np.isfinite(np.asarray(v, float)).sum())}


def share(mask):
    m = np.asarray(mask, dtype=float)
    return float(m.mean()) if m.size else NAN


def sync():
    t = sys.modules.get("torch")
    if t is not None and getattr(t, "cuda", None) is not None and t.cuda.is_available():
        t.cuda.synchronize()


def unit(e):
    e = np.asarray(e, dtype=np.float32)
    return e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-12)


def read_audio(path):
    try:
        with wave.open(path, "rb") as w:
            sw = w.getsampwidth()
        if sw == 2:
            c = AudioClip.from_wav(path)
            return c.samples, int(c.sample_rate)
    except Exception:
        pass
    sr, a = wavfile.read(path)
    a = np.asarray(a)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32767.0
    elif a.dtype == np.int32:
        a = a.astype(np.float32) / 2147483648.0
    elif a.dtype == np.uint8:
        a = (a.astype(np.float32) - 128.0) / 128.0
    else:
        a = a.astype(np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1)
    return np.ascontiguousarray(a, dtype=np.float32), int(sr)


def prepare(item):
    x, sr = read_audio(item["src"])
    y = to_sr(x, sr)
    if len(y) == 0 or rms(y) < 1e-6:
        return None
    z, padded = pad_to(y, N_GEN)
    anchor = np.asarray(level(_extract(z, SR, GEN_S)), dtype=np.float32)
    kept = np.asarray(level(_extract(anchor, SR, KEEP_S)), dtype=np.float32)
    if len(anchor) != N_GEN or len(kept) != N_KEEP:
        raise SystemExit("%s processed to %d and %d samples, expected %d and %d" % (
            item["id"], len(anchor), len(kept), N_GEN, N_KEEP))
    return {"anchor": anchor, "kept": kept, "padded": bool(padded), "src_s": len(x) / float(sr)}


def foley_table():
    rows = list(csv.DictReader(open(os.path.join(FOLEY, "metadata", "foleyset_metadata.csv"), encoding="utf-8")))
    out = []
    for r in rows:
        out.append({"id": "fs_%s" % r["name"][:-4], "dataset": "foleyset", "src": os.path.join(FOLEY, "audio", r["name"]),
                    "name": r["name"], "split": r["split"], "cluster": "fs:%s" % r["username"], "user": r["username"],
                    "recording": "fs:%s" % r["freesound-id"], "fsid": r["freesound-id"], "label": r["sub-category"],
                    "category": r["category"], "oneshot": r["Oneshot-Multi"] == "One-shot",
                    "ai": "ai-generated" in words(r),
                    "surface": surface_of(r) if r["category"] == "Footstep" else None,
                    "license": LICENSES["foleyset"]["license"]})
    return out


def c3gd_table():
    rows = list(csv.DictReader(open(os.path.join(C3GD, "metadata.csv"), encoding="utf-8")))
    plat = {r["platform_id"]: r for r in csv.DictReader(open(os.path.join(C3GD, "metadata", "platforms.csv"),
                                                              encoding="utf-8"))}
    out = []
    for r in rows:
        out.append({"id": "c3gd_%s" % r["datum_id"], "dataset": "c3gd", "src": os.path.join(C3GD, "data", r["filename"]),
                    "filename": r["filename"], "datum": r["datum_id"], "split": r["split"],
                    "cluster": "c3gd:%s" % r["platform_id"], "gun": r["platform_id"], "file_id": r["file_id"],
                    "recording": "c3gd:%s:%s:%s" % (r["event_id"], r["file_id"], r["clip_id"]),
                    "label": plat.get(r["platform_id"], {}).get("form_factor", "unknown"), "category": "gunshot",
                    "caliber": r["class_name"], "mic": r["mic_id"], "event": r["event_id"],
                    "channel": r["channel_orientation"] or "mono", "ai": False, "surface": None, "oneshot": True,
                    "license": LICENSES["c3gd"]["license"]})
    return out


def clip_like(x):
    if isinstance(x, dict):
        return any(k in x for k in ("src", "path", "file", "wav", "name", "filename", "datum_id", "id"))
    return isinstance(x, str) and (x.lower().endswith(".wav") or x.startswith(("fs_", "c3gd_")))


def field_of(kl):
    if kl in ("file_id", "file_ids", "fileid", "fileids"):
        return "file"
    if kl in ("freesound_id", "freesound_ids", "freesound-id"):
        return "fsid"
    for tok in reversed(kl.split("_")):
        if tok in FIELD_TOKENS:
            return FIELD_TOKENS[tok]
    return None


def add_src(bucket, f, x, ds):
    if isinstance(x, int) and f not in ("fsid", "file"):
        return
    x = str(x)
    if x.startswith("fs:"):
        bucket["fsid" if f == "rec" else "user"].add(x[3:])
    elif x.startswith("c3gd:"):
        bucket["crec" if f == "rec" else "gun"].add(x if f == "rec" else x[5:])
    elif x.startswith(("esc:", "hear:")):
        bucket["other"].add(x)
    elif f == "cluster":
        if ds in (None, "foleyset"):
            bucket["user"].add(x)
        if ds in (None, "c3gd"):
            bucket["gun"].add(x)
    elif f == "rec":
        bucket["fsid" if ds != "c3gd" else "crec"].add(x)
    else:
        bucket[f].add(x)


def scalar(x):
    return isinstance(x, (str, int)) and not isinstance(x, bool)


def walk(node, mode, ds, fctx, spec):
    if isinstance(node, dict):
        if fctx and node and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in node.values()):
            for x in node:
                add_src(spec[mode], fctx, x, ds)
            return
        for k, v in node.items():
            kl = str(k).lower().replace("-", "_").replace(" ", "_")
            if kl in MANIFEST_KEYS and isinstance(v, str):
                spec["manifest"] = v
                continue
            m = mode
            if kl in POOL_WORDS or kl.endswith("_pool") or kl.startswith("train"):
                m = "allow"
            elif kl.startswith(HELD_PREFIX):
                m = "held"
            d = "foleyset" if kl in ("foleyset", "foley", "fs") else ("c3gd" if kl == "c3gd" else ds)
            walk(v, m, d, field_of(kl) or fctx, spec)
    elif isinstance(node, list):
        if node and all(scalar(x) for x in node):
            if fctx:
                for x in node:
                    add_src(spec[mode], fctx, x, ds)
            elif all(clip_like(x) for x in node):
                spec["clips"][mode] += node
            return
        if node and all(clip_like(x) for x in node):
            spec["clips"][mode] += node
        for x in node:
            walk(x, mode, ds, None, spec)
    elif scalar(node) and fctx:
        add_src(spec[mode], fctx, node, ds)


def schema_spec(obj, p):
    if not isinstance(obj, dict):
        return None
    fs, c3 = obj.get("foleyset"), obj.get("c3gd")
    if not (isinstance(fs, dict) and isinstance(fs.get("uploaders"), dict) and isinstance(c3, dict) and
            isinstance(c3.get("guns"), dict)):
        return None
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    ups, guns = fs["uploaders"], c3["guns"]

    def pool_of(v):
        return v.get("pool") if isinstance(v, dict) else None
    held, allow = defaultdict(set), defaultdict(set)
    held["user"] = set(fs.get("heldout_uploaders") or []) | {u for u, v in ups.items() if pool_of(v) != "train"}
    allow["user"] = {u for u, v in ups.items() if pool_of(v) == "train"} - held["user"]
    held["gun"] = set(c3.get("heldout_guns") or []) | {g for g, v in guns.items() if pool_of(v) != "train"}
    allow["gun"] = {g for g, v in guns.items() if pool_of(v) == "train"} - held["gun"]
    tm = str(meta.get("train_mic") or "").split("|")
    tm = tm if len(tm) == 3 and all(tm[:2]) else None
    return {"path": rel(p), "held": held, "allow": allow, "clips": {"held": [], "allow": []},
            "manifest": meta.get("manifest") if isinstance(meta.get("manifest"), str) else None,
            "top_keys": list(obj)[:40], "strict": True, "train_mic": tm,
            "rule": "%s trainable rule: FoleySet split train from the %d uploaders whose pool is train, C3GD split train"
                    " on %s from the %d guns whose pool is train" % (
                        rel(p), len(allow["user"]), "|".join(tm) if tm else "the training mic", len(allow["gun"]))}


def held_spec(path):
    p = os.path.join(ROOT, path)
    if not os.path.isfile(p):
        return None
    obj = json.load(open(p, encoding="utf-8"))
    sch = schema_spec(obj, p)
    if sch is not None:
        return sch
    spec = {"path": rel(p), "held": defaultdict(set), "allow": defaultdict(set), "clips": {"held": [], "allow": []},
            "manifest": None, "top_keys": list(obj)[:40] if isinstance(obj, dict) else type(obj).__name__}
    walk(obj, "held", None, None, spec)
    found = sum(len(v) for v in spec["held"].values()) + sum(len(v) for v in spec["allow"].values()) + \
        len(spec["clips"]["held"]) + len(spec["clips"]["allow"])
    if not found:
        raise SystemExit("%s holds no source, recording or clip list this script recognises; top-level keys %s" % (
            rel(p), spec["top_keys"]))
    return spec


def lookup(tables):
    by = {}
    for t in tables:
        for r in t:
            by[r["id"]] = r
            by[os.path.normcase(os.path.abspath(r["src"]))] = r
            if r["dataset"] == "foleyset":
                by[r["name"]] = r
            else:
                by[r["filename"]] = r
                by["datum:%s" % r["datum"]] = r
    return by


def resolve(e, by):
    if isinstance(e, dict):
        for k in ("src", "path", "file", "wav"):
            if isinstance(e.get(k), str):
                p = e[k] if os.path.isabs(e[k]) else os.path.join(ROOT, e[k])
                r = by.get(os.path.normcase(os.path.abspath(p))) or by.get(os.path.basename(e[k]))
                if r:
                    return r
        for k in ("id", "name", "filename"):
            if isinstance(e.get(k), str) and e[k] in by:
                return by[e[k]]
        if "datum_id" in e:
            return by.get("datum:%s" % e["datum_id"])
        return None
    p = e if os.path.isabs(e) else os.path.join(ROOT, e)
    return by.get(e) or by.get(os.path.normcase(os.path.abspath(p))) or by.get(os.path.basename(e))


def fallback_spec(fs, c3, args):
    held = defaultdict(set)
    held["user"] = {r["user"] for r in fs if r["split"] not in args.foley_splits}
    cand = [r for r in c3 if r["split"] in args.c3gd_splits and r["mic"] in args.c3gd_mics
            and r["event"] in args.c3gd_events]
    by_form = defaultdict(set)
    for r in cand:
        by_form[r["label"]].add(r["gun"])
    if args.c3gd_holdout_every > 0:
        for form in sorted(by_form):
            gs = sorted(by_form[form])
            held["gun"] |= {g for i, g in enumerate(gs) if i % args.c3gd_holdout_every == args.c3gd_holdout_every - 1}
    return {"path": None, "held": held, "allow": defaultdict(set), "clips": {"held": [], "allow": []}, "manifest": None,
            "top_keys": None,
            "rule": "fallback: FoleySet uploaders with any clip outside %s are held out; C3GD guns at position %d, %d, ..."
                    " of each form factor in sorted order are held out" % (
                        "/".join(args.foley_splits), args.c3gd_holdout_every, 2 * args.c3gd_holdout_every)}


def load_manifest(path):
    p = os.path.join(ROOT, path)
    if not os.path.isfile(p):
        return None, rel(p)
    return json.load(open(p, encoding="utf-8")), rel(p)


def manifest_index(man):
    recs, by_type = set(), defaultdict(set)
    if not man:
        return recs, by_type
    stype = {s: v.get("type") for s, v in man.get("sets", {}).items()}
    for c in man.get("clips", []):
        recs.add(c.get("recording"))
        if c.get("esc_src"):
            recs.add("fs:%s" % c["esc_src"])
        if c.get("dataset") == "c3gd" and c.get("id"):
            recs.add("id:%s" % c["id"])
        t = stype.get(c.get("set"))
        if t:
            by_type[t].add(c.get("cluster"))
    return recs, by_type


def held_reason(r, held):
    if r["id"] in held["ids"]:
        return "held-out clip"
    if r["dataset"] == "foleyset":
        if r["user"] in held["user"]:
            return "held-out uploader"
        if r["fsid"] in held["fsid"]:
            return "held-out recording"
    else:
        if r["gun"] in held["gun"]:
            return "held-out gun"
        if r["file_id"] in held["file"]:
            return "held-out C3GD file"
        if r["recording"] in held["crec"]:
            return "held-out recording"
    return None


def audio_hash(path, cache):
    k = os.path.normcase(os.path.abspath(path))
    st = os.stat(path)
    ent = cache.get(k)
    if ent and ent[0] == st.st_size and ent[1] == int(st.st_mtime):
        return ent[2]
    try:
        with wave.open(path, "rb") as w:
            d = w.readframes(w.getnframes())
    except Exception:
        d = open(path, "rb").read()
    h = hashlib.md5(d).hexdigest()
    cache[k] = [st.st_size, int(st.st_mtime), h]
    return h


def load_hashes(path):
    p = os.path.join(ROOT, path)
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}


def save_hashes(path, cache):
    p = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)


def dedupe_audio(pool, fs, c3, held, man, args):
    cache = load_hashes(args.hash_cache)
    n0 = len(cache)
    t0 = time.perf_counter()
    fsh = {r["id"]: audio_hash(r["src"], cache) for r in fs if os.path.isfile(r["src"])}
    mics = {(r["mic"], r["event"]) for r in pool if r["dataset"] == "c3gd"} | {
        (m, e) for m in args.c3gd_mics for e in args.c3gd_events}
    c3h = {r["id"]: audio_hash(r["src"], cache) for r in c3 if (r["mic"], r["event"]) in mics and os.path.isfile(r["src"])}
    man_h = set()
    for c in (man or {}).get("clips", []):
        if c.get("dataset") in ("foleyset", "c3gd") and c.get("src") and os.path.isfile(c["src"]):
            man_h.add(audio_hash(c["src"], cache))
    if len(cache) != n0:
        save_hashes(args.hash_cache, cache)
    ids = {r["id"] for r in pool}
    blocked = set(man_h)
    for r in fs:
        if r["id"] in fsh and r["id"] not in ids and (r["split"] not in args.foley_splits or held_reason(r, held)):
            blocked.add(fsh[r["id"]])
    for r in c3:
        if r["id"] in c3h and r["id"] not in ids and (r["split"] not in args.c3gd_splits or held_reason(r, held)):
            blocked.add(c3h[r["id"]])
    hof = {**fsh, **c3h}
    keep, seen, excl = [], set(), Counter()
    for r in sorted(pool, key=lambda r: r["id"]):
        h = hof.get(r["id"])
        if h in blocked:
            excl["audio identical to a held-out or eval clip"] += 1
        elif h in seen:
            excl["audio identical to another pool clip"] += 1
        else:
            seen.add(h)
            r["hash"] = h
            keep.append(r)
    groups = defaultdict(list)
    for r in fs:
        if r["id"] in fsh:
            groups[fsh[r["id"]]].append(r)
    dup = [g for g in groups.values() if len(g) > 1]
    cross = [g for g in dup if len({r["label"] for r in g}) > 1]
    in_cross = {r["id"] for g in cross for r in g}
    by_set = Counter(c["set"] for c in (man or {}).get("clips", []) if c.get("dataset") == "foleyset" and
                     c.get("id") in in_cross)
    audit = {"foleyset_files": len(fsh), "distinct_audio": len(groups), "duplicate_groups": len(dup),
             "files_in_duplicate_groups": sum(len(g) for g in dup), "cross_label_groups": len(cross),
             "files_in_cross_label_groups": len(in_cross),
             "cross_label_by_sub": dict(Counter(r["label"] for g in cross for r in g).most_common()),
             "eval_refs_in_cross_label_groups": dict(by_set), "hash_seconds": time.perf_counter() - t0,
             "examples": [[(r["id"], r["label"], safe(r["user"]), r["split"]) for r in g] for g in cross[:5]]}
    return keep, excl, audit


def esc_sources():
    ids = {}
    for d in ESC_DIRS:
        for base, _ds, files in os.walk(d):
            for f in sorted(files):
                m = ESC_NAME.match(f)
                if m:
                    ids.setdefault(m.group(2), rel(os.path.join(base, f)))
    return ids


def contract(args, spec):
    if not spec.get("strict"):
        return
    want = {"foley_splits": ["train"], "c3gd_splits": ["train"]}
    tm = spec.get("train_mic")
    if tm:
        want["c3gd_events"], want["c3gd_mics"] = [tm[0]], [tm[1]]
    bad = {k: list(getattr(args, k)) for k, v in want.items() if list(getattr(args, k)) != v}
    if bad:
        raise SystemExit("%s lets a clip anchor only from %s; this run asks for %s" % (spec["path"], want, bad))


def norm(p):
    return os.path.normcase(os.path.abspath(p))


def leak_check(pool, man):
    clips = (man or {}).get("clips", [])
    held = ((man or {}).get("meta") or {}).get("heldout") or {}
    hc = {c for v in (held.get("clusters") or {}).values() for c in v}
    used = {r["cluster"] for r in pool}
    return {"clusters_in_refs": sorted(used & {c.get("cluster") for c in clips}),
            "sources_in_refs": sorted({norm(r["src"]) for r in pool} & {norm(c["src"]) for c in clips if c.get("src")}),
            "clusters_held_out": sorted(used & hc), "recordings_in_refs": sorted(
                {r["recording"] for r in pool} & {c.get("recording") for c in clips}),
            "heldout_manifest": bool(held)}


def build_pool(args, spec, man):
    fs, c3 = foley_table(), c3gd_table()
    by = lookup([fs, c3])
    fallback = spec is None
    if fallback:
        spec = fallback_spec(fs, c3, args)
    contract(args, spec)
    held = defaultdict(set, {k: set(v) for k, v in spec["held"].items()})
    for e in spec["clips"]["held"]:
        r = resolve(e, by)
        if r:
            held["ids"].add(r["id"])
    unresolved = 0
    excl = Counter()
    if spec["clips"]["allow"]:
        cand, seen = [], set()
        for e in spec["clips"]["allow"]:
            r = resolve(e, by)
            if r is None:
                unresolved += 1
            elif isinstance(e, dict) and e.get("pool") not in (None, "train"):
                excl["listed under pool %s" % e.get("pool")] += 1
            elif r["split"] not in (args.foley_splits if r["dataset"] == "foleyset" else args.c3gd_splits):
                excl["split %s outside the anchor splits" % r["split"]] += 1
            elif r["id"] not in seen:
                seen.add(r["id"])
                cand.append(r)
        rule = "explicit clip list in %s" % spec["path"]
    else:
        tm = spec.get("train_mic")
        cand = [r for r in fs if r["split"] in args.foley_splits] + \
               [r for r in c3 if r["split"] in args.c3gd_splits and r["mic"] in args.c3gd_mics and
                r["event"] in args.c3gd_events and (not tm or r["channel"] == tm[2])]
        allow = spec["allow"]
        strict = bool(spec.get("strict"))
        if strict or allow.get("user"):
            cand = [r for r in cand if r["dataset"] != "foleyset" or r["user"] in allow["user"]]
        if strict or allow.get("gun"):
            cand = [r for r in cand if r["dataset"] != "c3gd" or r["gun"] in allow["gun"]]
        rule = spec.get("rule") or ("FoleySet %s and C3GD %s %s at %s, minus the held-out sources in %s" % (
            "/".join(args.foley_splits), "/".join(args.c3gd_splits), "/".join(args.c3gd_mics),
            "/".join(args.c3gd_events), spec["path"]))
    recs, _by_type = manifest_index(man)
    esc = esc_sources()
    owner = {}
    for r in fs:
        owner.setdefault(r["fsid"], r["user"])
    esc_users = {owner[i] for i in esc if i in owner}
    pool = []
    for r in cand:
        why = None
        if os.path.normcase(os.path.abspath(r["src"])).startswith(os.path.normcase(os.path.abspath(ESC))):
            why = "ESC-50 is evaluation only"
        elif r["ai"]:
            why = "tagged ai-generated"
        elif r["dataset"] == "foleyset" and r["fsid"] in esc:
            why = "same Freesound recording as an ESC-50 file"
        elif r["dataset"] == "foleyset" and r["user"] in esc_users:
            why = "uploader of an ESC-50 file"
        else:
            why = held_reason(r, held)
        if why is None and (r["recording"] in recs or "id:%s" % r["id"] in recs):
            why = "recording in the eval manifest"
        if why is None and not os.path.isfile(r["src"]):
            why = "audio file missing"
        if why:
            excl[why] += 1
        else:
            pool.append(r)
    pool, dx, audit = dedupe_audio(pool, fs, c3, held, man, args)
    excl.update(dx)
    leak = leak_check(pool, man)
    if leak["heldout_manifest"] and any(v for k, v in leak.items() if k != "heldout_manifest"):
        raise SystemExit("the anchor pool shares sources with the held-out references: %s" % json.dumps(
            {k: v[:10] for k, v in leak.items() if k != "heldout_manifest"}))
    if args.limit_library:
        pool = spaced(sorted(pool, key=lambda r: r["id"]), args.limit_library)
    info = {"rule": rule, "fallback": fallback, "heldout_file": spec["path"], "heldout_top_keys": spec["top_keys"],
            "audit": audit, "esc_files": len(esc), "esc_uploaders": sorted(esc_users),
            "leak": {k: (v if isinstance(v, bool) else len(v)) for k, v in leak.items()},
            "held_counts": {k: len(v) for k, v in held.items()},
            "allow_counts": {k: len(v) for k, v in spec["allow"].items()},
            "explicit_clips": len(spec["clips"]["allow"]), "unresolved_clips": unresolved,
            "candidates": len(cand), "excluded": dict(excl), "limit_library": args.limit_library,
            "held_guns": sorted(held["gun"]) if len(held["gun"]) <= 60 else len(held["gun"])}
    return pool, info


class Clap:
    def __init__(self, args):
        self.torch, self.dev = _torch(args.threads)
        from transformers import ClapModel, ClapProcessor
        self.m = ClapModel.from_pretrained(CLAP_NAME).to(self.dev).eval()
        self.pr = ClapProcessor.from_pretrained(CLAP_NAME)
        self.bs = 4 * args.bs

    def audio(self, clips):
        out = []
        for i in range(0, len(clips), self.bs):
            xs = [_resample(np.asarray(x, dtype=np.float32), SR, 48000) for x in clips[i: i + self.bs]]
            a = self.pr(audio=xs, sampling_rate=48000, return_tensors="pt")
            inp = {k: v.to(self.dev) for k, v in a.items() if k in ("input_features", "is_longer")}
            with self.torch.no_grad():
                o = self.m.get_audio_features(**inp)
            out.append(unit(getattr(o, "pooler_output", o).float().cpu().numpy()))
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    def text(self, texts):
        t = self.pr(text=list(texts), return_tensors="pt", padding=True)
        inp = {k: v.to(self.dev) for k, v in t.items() if k in ("input_ids", "attention_mask")}
        with self.torch.no_grad():
            o = self.m.get_text_features(**inp)
        return unit(getattr(o, "pooler_output", o).float().cpu().numpy())

    def close(self, tag):
        del self.m, self.pr
        _release(self.torch)
        free_gpu(tag)


def fingerprint():
    return hashlib.md5(json.dumps([VERSION, CLAP_NAME, SR, GEN_S, KEEP_S, INTENSITY, target_for(INTENSITY)]).encode()
                       ).hexdigest()[:12]


def load_cache(path):
    p = os.path.join(ROOT, path)
    if not os.path.isfile(p):
        return {}, set()
    z = np.load(p, allow_pickle=False)
    if str(z["fingerprint"]) != fingerprint():
        print("library cache %s has fingerprint %s, current %s: rebuilding" % (rel(p), z["fingerprint"], fingerprint()))
        return {}, set()
    return {str(i): e for i, e in zip(z["ids"], z["emb"])}, {str(s) for s in z["silent"]}


def save_cache(path, have, silent):
    p = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    ids = sorted(have)
    np.savez(p, ids=np.array(ids), emb=np.stack([have[i] for i in ids]) if ids else np.zeros((0, 512), np.float32),
             silent=np.array(sorted(silent)), fingerprint=np.array(fingerprint()))


def embed_library(pool, clap, args):
    have, silent = load_cache(args.library_cache)
    todo = [r for r in pool if r["id"] not in have and r["id"] not in silent]
    print("library: %d clips in the pool, %d cached embeddings, %d to embed" % (
        len(pool), sum(r["id"] in have for r in pool), len(todo)), flush=True)
    t0 = time.perf_counter()
    chunk = max(64, 8 * clap.bs)
    done = 0
    for i in range(0, len(todo), chunk):
        part = todo[i: i + chunk]
        kept, ids = [], []
        for r in part:
            p = prepare(r)
            if p is None:
                silent.add(r["id"])
                continue
            kept.append(p["kept"])
            ids.append(r["id"])
        e = clap.audio(kept)
        for j, iid in enumerate(ids):
            have[iid] = e[j]
        done += len(part)
        el = time.perf_counter() - t0
        print("  library %5d/%d  %7.1f s  eta %6.0f s" % (done, len(todo), el, el / done * (len(todo) - done)), flush=True)
        if (i // chunk) % 8 == 7:
            save_cache(args.library_cache, have, silent)
    if todo:
        save_cache(args.library_cache, have, silent)
    lib = [r for r in pool if r["id"] in have]
    return lib, np.stack([have[r["id"]] for r in lib]).astype(np.float32), sorted(r["id"] for r in pool if r["id"] in silent)


def type_labels(t, fs_cat):
    if t is None:
        return set(), set()
    subs = {s["sub"] for s in REF_SETS.values() if s["type"] == t and s["dataset"] == "foleyset"}
    forms = {f for f, sname in FORMS.items() if REF_SETS[sname]["type"] == t}
    cats = {fs_cat[s] for s in subs if s in fs_cat} | ({"gunshot"} if forms else set())
    return subs | forms, cats


def retrieve(clap, lib, emb, keys, args, by_type):
    fs_cat = {r["label"]: r["category"] for r in lib if r["dataset"] == "foleyset"}
    out = {}
    clap.text(["warm up"])
    for et, mat, t in keys:
        k = key_name(et, mat)
        prompt = build_prompt(et, mat, INTENSITY)
        tms, sms = [], []
        for _ in range(max(1, args.text_reps)):
            sync()
            t0 = time.perf_counter()
            q = clap.text([prompt])[0]
            sync()
            t1 = time.perf_counter()
            s = emb @ q
            idx = np.argpartition(-s, args.topk)[: args.topk] if len(s) > args.topk else np.arange(len(s))
            idx = idx[np.argsort(-s[idx])]
            t2 = time.perf_counter()
            tms.append((t1 - t0) * 1000.0)
            sms.append((t2 - t1) * 1000.0)
        labs, cats = type_labels(t, fs_cat)
        top = []
        for j in idx:
            r = lib[int(j)]
            top.append({"id": r["id"], "cos": float(s[j]), "dataset": r["dataset"], "label": r["label"],
                        "category": r["category"], "cluster": r["cluster"], "split": r["split"],
                        "surface": r["surface"], "oneshot": r["oneshot"], "lib_index": int(j),
                        "label_match": bool(r["label"] in labs) if t else None,
                        "category_match": bool(r["category"] in cats) if t else None,
                        "surface_match": bool(r["surface"] == mat) if et == "footstep" else None,
                        "shares_eval_source": bool(r["cluster"] in by_type.get(t, set())) if t else None})
        order = np.argsort(-s)
        out[k] = {"prompt": prompt, "type": t, "text_ms": med(tms), "search_ms": med(sms), "top": top,
                  "label_share": share([x["label_match"] for x in top]) if t else None,
                  "category_share": share([x["category_match"] for x in top]) if t else None,
                  "surface_share": share([x["surface_match"] for x in top]) if et == "footstep" else None,
                  "eval_source_share": share([x["shares_eval_source"] for x in top]) if t else None,
                  "cos_k": float(s[order[min(len(order) - 1, args.topk - 1)]]),
                  "cos_median_library": float(np.median(s)),
                  "sources": len({x["cluster"] for x in top})}
    return out


def plan(args):
    st = {CURRENT: {**CUR, "route": "plain"},
          "raw": {**CUR, "steps": 0, "detail_steps": 0, "route": "raw", "note": "the retrieved anchor itself at the kept"
                                                                                " length, no model"},
          "replay": {**CUR, "route": "replay", "init_noise_level": 0.0, "schedule": "package",
                     "note": "init_audio at noise 0.0, the encode and decode chain alone"}}
    for sch in args.schedules:
        for lv in args.levels:
            st[setting_name(lv, sch)] = {**CUR, "route": "anchor", "init_noise_level": float(lv), "schedule": sch}
    return st


def setting_name(lv, sch):
    return "anc_n%g%s" % (lv, "" if sch == "package" else "_sc")


class Scaled:
    def __init__(self, base, sigma):
        self.base = base
        self.sigma = float(sigma)

    def shift(self, t, seq_len):
        return self.base.shift(t / self.sigma, seq_len) * self.sigma


def shift_for(g, cfg):
    if cfg.get("schedule") != "scaled":
        return None
    return Scaled(g._model.model.sampling_dist_shift, cfg["init_noise_level"])


def schedule_of(g, steps, sigma, shift):
    try:
        from stable_audio_3.inference.sampling import build_schedule
        sam = g._model
        ds = shift if shift is not None else sam.model.sampling_dist_shift
        n = int(sam._adapt_sample_size([{"prompt": "", "seconds_total": g.gen_duration_s}], 5292032,
                                       g.duration_padding_s) // sam.model.pretransform.downsampling_ratio)
        eff = -(-int(g.gen_duration_s * g.sample_rate) // int(sam.model.pretransform.downsampling_ratio))
        t = build_schedule(steps=steps, sigma_max=sigma, dist_shift=ds, effective_seq_len=eff, fallback_seq_len=n,
                           include_endpoint=True, device="cpu")
        return {"sigmas": [round(float(v), 4) for v in np.asarray(t.reshape(-1)[: steps + 1].tolist())],
                "dist_shift": type(ds).__name__, "latent_len": n}
    except Exception as e:
        return {"error": repr(e)[:300]}


def gen_anchor(g, key, variant, anchor, sigma, shift, stats, guard=True):
    prompt = g._prompt(key)
    steps = g._steps_for(key)
    base = _seed(key, variant)

    def replay():
        import torch
        sam = g._model
        n = int(sam._adapt_sample_size([{"prompt": "", "seconds_total": g.gen_duration_s}], 5292032,
                                       g.duration_padding_s))
        with g._lock, torch.inference_mode():
            z, _ = sam._encode_audio_input((int(g.sample_rate), np.ascontiguousarray(anchor, dtype=np.float32)), n)
            audio = sam.model.pretransform.decode(z)
        return g._finish(g._numpy(audio), key)

    def once(seed):
        if sigma <= 0.0:
            return replay()
        kw = g._inputs([prompt])
        if shift is not None:
            kw["dist_shift"] = shift
        with g._lock:
            audio = g._model.generate(duration=g.gen_duration_s, steps=steps, cfg_scale=g.cfg_scale, seed=seed,
                                      init_audio=(int(g.sample_rate), np.ascontiguousarray(anchor, dtype=np.float32)),
                                      init_noise_level=float(sigma), **kw)
        return g._finish(g._numpy(audio), key)

    mono = once(base)
    if not np.isfinite(mono).all():
        stats["nonfinite"] += 1
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    if not (guard and g.static_guard) or sigma <= 0.0:
        return mono
    bad, m = is_static(mono, g.sample_rate)
    stats["static_checks"] += 1
    if not bad:
        return mono
    stats["static_flagged"] += 1
    best, best_flat = mono, m["flat"]
    for attempt in range(1, g.max_static_retries + 1):
        stats["static_retries"] += 1
        cand = once(retry_seed(base, attempt))
        bad, m = is_static(cand, g.sample_rate)
        stats["static_checks"] += 1
        if not bad:
            stats["static_rescued"] += 1
            return cand
        if m["flat"] < best_flat:
            best, best_flat = cand, m["flat"]
    stats["static_exhausted"] += 1
    return np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)


def fast_delta(a, b):
    out = {}
    for k, v in b.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(a.get(k, 0), (int, float)):
            out[k] = v - a.get(k, 0)
        elif isinstance(v, list):
            out[k] = v[len(a.get(k, []) or []):] if isinstance(a.get(k), list) else v
        else:
            out[k] = v
    return out


def stamp(args, keys, assign, settings, man_rel):
    return {"version": VERSION, "seeds": args.seeds, "topk": args.topk, "keys": [key_name(e, m) for e, m, _t in keys],
            "assign": assign, "settings": settings, "fingerprint": fingerprint(), "manifest": man_rel}


def write_routes(args, keys, assign, settings, rows):
    out = {}
    aids = dedupe([a for et, mat, _t in keys for a in assign[key_name(et, mat)]])
    for s, cfg in settings.items():
        if cfg["route"] == "plain":
            continue
        info = {"clusters": sorted({rows[a]["cluster"] for a in aids}), "sources": sorted(rows[a]["src"] for a in aids),
                "steps": int(cfg["steps"]), "prompts": {key_name(et, mat): build_prompt(et, mat, INTENSITY)
                                                        for et, mat, _t in keys},
                "anchors": {key_name(et, mat): assign[key_name(et, mat)] for et, mat, _t in keys},
                "route": cfg["route"], "init_noise_level": cfg.get("init_noise_level"),
                "schedule": cfg.get("schedule"), "fingerprint": fingerprint(),
                "note": "%s; %s" % (cfg.get("note") or "init_audio anchor route", CAUTION)}
        p = os.path.join(ROOT, args.gen_dir, s, "route.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=1)
        out[s] = rel(p)
    return out


def reusable(args, keys, assign, settings):
    p = os.path.join(ROOT, args.gen_dir, "settings.json")
    if not args.reuse or not os.path.isfile(p):
        return set()
    old = json.load(open(p, encoding="utf-8"))
    if old.get("assign") != assign or old.get("seeds") != args.seeds or old.get("fingerprint") != fingerprint():
        print("--reuse: settings.json in %s was written for other anchors or seeds, so every setting is regenerated" %
              rel(os.path.dirname(p)))
        return set()
    ok = set()
    for s, cfg in settings.items():
        if old.get("settings", {}).get(s) != cfg:
            continue
        if all(os.path.isfile(p) for et, mat, _t in keys for p in gen_paths(args, s, et, mat)):
            ok.add(s)
    return ok


def write_raw(args, keys, assign, anchors):
    for et, mat, _t in keys:
        for v, p in enumerate(gen_paths(args, "raw", et, mat)):
            AudioClip(anchors[assign[key_name(et, mat)][v]]["kept"], SR).to_wav(p)


def generate(args, keys, assign, anchors, settings, skip):
    info, runs, scheds = {"loaded": False}, {}, {}
    need = [s for s, c in settings.items() if c["route"] != "raw" and s not in skip]
    if not need:
        print("generator: every setting reused from %s" % rel(os.path.join(ROOT, args.gen_dir)), flush=True)
        return info, runs, scheds
    g = ve.make_generator()
    g.ensure_ready()
    if int(g.sample_rate) != SR:
        raise SystemExit("generator sample rate %s, expected %d" % (g.sample_rate, SR))
    for a, v in CUR.items():
        setattr(g, a, v)
    info = gen_info(g)
    cks = {key_name(et, mat): CacheKey(et, mat, INTENSITY) for et, mat, _t in keys}
    for k, ck in cks.items():
        if g._prompt(ck) != build_prompt(ck.event_type, ck.material, INTENSITY):
            raise SystemExit("prompt for %s differs from build_prompt" % ck)
        if g._steps_for(ck) != CUR["steps"]:
            raise SystemExit("steps for %s = %d, expected %d" % (ck, g._steps_for(ck), CUR["steps"]))
    print("generator %s" % json.dumps(info, default=str), flush=True)
    k0 = next(iter(cks.values()))
    a0 = anchors[assign[key_name(k0.event_type, k0.material)][0]]["anchor"]
    g.generate(k0, variant=0)
    gen_anchor(g, k0, 0, a0, 0.5, None, Counter(), guard=False)
    for s in need:
        cfg = settings[s]
        os.makedirs(os.path.join(ROOT, args.gen_dir, s), exist_ok=True)
        shift = shift_for(g, cfg)
        if cfg["route"] in ("anchor", "replay"):
            scheds[s] = schedule_of(g, CUR["steps"], cfg["init_noise_level"], shift)
        f0 = dict(g.fast_stats()) if hasattr(g, "fast_stats") else {}
        st0 = dict(getattr(g, "static_stats", {}) or {})
        stats, ms = Counter(), defaultdict(list)
        t_set = time.perf_counter()
        for et, mat, _t in keys:
            k = key_name(et, mat)
            ck = cks[k]
            ps = gen_paths(args, s, et, mat)
            done = {}
            for v, p in enumerate(ps):
                aid = assign[k][v]
                if cfg["route"] == "replay" and aid in done:
                    AudioClip(done[aid], SR).to_wav(p)
                    continue
                sync()
                t0 = time.perf_counter()
                if cfg["route"] == "plain":
                    y = g.generate(ck, variant=v).samples
                else:
                    y = gen_anchor(g, ck, v, anchors[aid]["anchor"], cfg["init_noise_level"], shift, stats)
                sync()
                ms[k].append((time.perf_counter() - t0) * 1000.0)
                y = np.asarray(y, dtype=np.float32)
                if len(y) != N_KEEP:
                    raise SystemExit("%s %s variant %d has %d samples, expected %d" % (s, k, v, len(y), N_KEEP))
                AudioClip(y, SR).to_wav(p)
                done[aid] = y
        st1 = dict(getattr(g, "static_stats", {}) or {})
        if cfg["route"] == "plain":
            stats = Counter({a: int(st1.get(a, 0)) - int(st0.get(a, 0)) for a in st1})
        fs = fast_delta(f0, dict(g.fast_stats()) if hasattr(g, "fast_stats") else {})
        allms = [x for v in ms.values() for x in v]
        runs[s] = {"ms_median": med(allms), "ms_p90": qt(allms, 90), "ms_max": float(max(allms)) if allms else NAN,
                   "ms_by_key": {k: med(v) for k, v in ms.items()}, "clips_timed": len(allms),
                   "static": dict(stats), "fast": fs, "seconds": time.perf_counter() - t_set}
        print("generated %-14s %d keys x %d seeds  %.1f s  median %.1f ms  p90 %.1f ms  %s  %s" % (
            s, len(keys), args.seeds, runs[s]["seconds"], runs[s]["ms_median"], runs[s]["ms_p90"],
            " ".join("%s %s" % kv for kv in sorted(stats.items())),
            " ".join("%s %s" % (k, fs[k]) for k in ("cond_misses", "graphs", "replays", "eager", "capture_failures",
                                                   "hoisted", "plain", "slots", "fills", "cache_hits",
                                                   "slot_failures", "hoist_capture_failures") if k in fs)), flush=True)
    del g
    free_gpu("generator")
    return info, runs, scheds


def envelope(x):
    win, hop = max(1, int(0.005 * SR)), max(1, int(0.001 * SR))
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, np.float64) ** 2)])
    s = np.arange(0, len(x) - win + 1, hop)
    return np.sqrt(np.maximum(c[s + win] - c[s], 0.0) / win)


def mel_bank():
    def hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    def mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    pts = hz(np.linspace(mel(20.0), mel(16000.0), N_MELS + 2))
    f = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    fb = np.zeros((N_MELS, len(f)))
    for i in range(N_MELS):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0.0, None)
    return fb


MEL = mel_bank()
HANN = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)


def logmel(x):
    x = np.asarray(x, np.float64)
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    fr = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP] * HANN
    m = 10.0 * np.log10(np.abs(np.fft.rfft(fr, axis=1)) ** 2 @ MEL.T + 1e-12)
    return np.maximum(m, m.max() - 80.0)


def zs(a):
    a = np.asarray(a, np.float64)
    return (a - a.mean()) / (a.std() + 1e-12)


class Feats:
    def __init__(self, x):
        x = np.asarray(x, np.float64)
        self.norm = max(float(np.sqrt(np.sum(x * x))), 1e-12)
        self.W = np.fft.rfft(x, NFFT_W).astype(np.complex64)
        self.env = zs(envelope(x))
        self.mel = zs(logmel(x))


def lagcos(a, b, lag):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    best = -1.0
    for l in range(-lag, lag + 1):
        x = a[l:] if l >= 0 else a[: n + l]
        y = b[: n - l] if l >= 0 else b[-l:]
        q = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
        if q > 0:
            best = max(best, float(np.sum(x * y)) / q)
    return best


def wave_ncc(a, b, lag):
    c = np.fft.irfft(a.W * np.conj(b.W), NFFT_W)
    return float(np.max(np.abs(np.concatenate([c[: lag + 1], c[NFFT_W - lag:]]))) / (a.norm * b.norm))


def pair(fa, fb, ea, eb, lags):
    return {"clap": float(np.dot(ea, eb)), "mel": lagcos(fa.mel, fb.mel, lags["mel"]),
            "wave": wave_ncc(fa, fb, lags["wave"]), "env": lagcos(fa.env, fb.env, lags["env"])}


def load_outputs(args, keys, settings):
    clips = {}
    for s in settings:
        clips[s] = {}
        for et, mat, _t in keys:
            xs = [wav(p) for p in gen_paths(args, s, et, mat)]
            bad = sorted({len(x) for x in xs if len(x) != N_KEEP})
            if bad:
                raise SystemExit("%s %s clips have lengths %s, expected %d" % (s, key_name(et, mat), bad, N_KEEP))
            clips[s][key_name(et, mat)] = xs
    return clips


def embed_outputs(clap, clips, keys, assign, settings):
    emb = {}
    t0 = time.perf_counter()
    n = 0
    for s, cfg in settings.items():
        if cfg["route"] == "raw":
            continue
        emb[s] = {}
        for et, mat, _t in keys:
            k = key_name(et, mat)
            if cfg["route"] == "replay":
                first = {}
                for v, aid in enumerate(assign[k]):
                    first.setdefault(aid, v)
                vs = sorted(first.values())
                e = clap.audio([clips[s][k][v] for v in vs])
                m = {assign[k][v]: e[j] for j, v in enumerate(vs)}
                emb[s][k] = np.stack([m[aid] for aid in assign[k]])
                n += len(vs)
            else:
                emb[s][k] = clap.audio(clips[s][k])
                n += len(clips[s][k])
    print("clap  outputs %7.1f s for %d clips" % (time.perf_counter() - t0, n), flush=True)
    return emb


def nearest(sv, lib, aid, rec):
    for j in np.argsort(-sv):
        r = lib[int(j)]
        if r["id"] != aid and r["recording"] != rec:
            return int(j)
    return None


def copy_measure(clips, emb, lib, lib_emb, keys, assign, settings, args):
    lags = {"wave": int(args.max_lag_ms / 1000.0 * SR), "env": int(round(args.max_lag_ms)),
            "mel": max(1, int(round(args.max_lag_ms / 1000.0 * SR / HOP)))}
    idx = {r["id"]: i for i, r in enumerate(lib)}
    anchor_settings = [s for s, c in settings.items() if c["route"] in ("anchor", "plain")]
    rows, pairs, near = [], [], {}
    t0 = time.perf_counter()
    for et, mat, _t in keys:
        k = key_name(et, mat)
        aids = dedupe(assign[k])
        af = {aid: Feats(clips["raw"][k][assign[k].index(aid)]) for aid in aids}
        ae = {aid: lib_emb[idx[aid]] for aid in aids}
        for s in anchor_settings + ["replay"]:
            F = [Feats(x) for x in clips[s][k]]
            E = emb[s][k]
            sims = E @ lib_emb.T
            vs = range(len(F)) if s != "replay" else sorted({assign[k].index(a) for a in aids})
            for v in vs:
                aid = assign[k][v]
                m = pair(F[v], af[aid], E[v], ae[aid], lags)
                sv = sims[v]
                ai = idx[aid]
                rank = int(1 + np.sum(sv > sv[ai]))
                other = np.delete(sv, ai)
                nn = int(np.argmax(sv))
                rows.append({"setting": s, "key": k, "v": v, "anchor": aid, **m, "anchor_rank": rank,
                             "anchor_margin": float(sv[ai] - other.max()) if other.size else NAN,
                             "nn_id": lib[nn]["id"], "nn_cos": float(sv[nn]), "nn_is_anchor": bool(nn == ai)})
            if s == "replay":
                continue
            for i in range(len(F)):
                for j in range(i + 1, len(F)):
                    m = pair(F[i], F[j], E[i], E[j], lags)
                    kind = "plain" if s == CURRENT else ("same" if assign[k][i] == assign[k][j] else "diff")
                    pairs.append({"setting": s, "key": k, "kind": kind, **m})
        for i, a in enumerate(aids):
            for b in aids[i + 1:]:
                pairs.append({"setting": "raw", "key": k, "kind": "anchors", **pair(af[a], af[b], ae[a], ae[b], lags)})
        for aid in aids:
            if aid in near:
                continue
            j = nearest(lib_emb @ ae[aid], lib, aid, lib[idx[aid]]["recording"])
            if j is None:
                continue
            p = prepare(lib[j])
            if p is None:
                continue
            m = pair(af[aid], Feats(p["kept"]), ae[aid], lib_emb[j], lags)
            near[aid] = {"neighbor": lib[j]["id"], "same_source": lib[j]["cluster"] == lib[idx[aid]]["cluster"], **m}
    print("copy  metrics %7.1f s for %d output rows and %d pairs" % (time.perf_counter() - t0, len(rows), len(pairs)),
          flush=True)
    return rows, pairs, near, lags


def summarize_copy(rows, pairs, near, settings, keys):
    def sel(**kw):
        return [r for r in rows if all(r[a] == b for a, b in kw.items())]

    rep = sel(setting="replay")
    plain = sel(setting=CURRENT)
    rep_lo = {m: qt([r[m] for r in rep], 10) for m in METRICS}
    plain_hi = {m: qt([r[m] for r in plain], 90) for m in METRICS}
    nb = list(near.values())
    out = {"thresholds": {"replay_p10": rep_lo, "plain_p90": plain_hi},
           "nulls": {"replay": {m: dist([r[m] for r in rep]) for m in METRICS},
                     "plain": {m: dist([r[m] for r in plain]) for m in METRICS},
                     "real_neighbor": {m: dist([r[m] for r in nb]) for m in METRICS},
                     "real_neighbor_same_source": share([r["same_source"] for r in nb])},
           "by_setting": {}, "by_key": {}, "variants": {}}

    def block(rs):
        return {**{m: dist([r[m] for r in rs]) for m in METRICS},
                "nn_is_anchor": share([r["nn_is_anchor"] for r in rs]),
                "anchor_rank_median": med([r["anchor_rank"] for r in rs]),
                "nn_cos": dist([r["nn_cos"] for r in rs]),
                "replay_like": share([r["clap"] >= rep_lo["clap"] and r["mel"] >= rep_lo["mel"] for r in rs]),
                "plain_like": share([r["clap"] <= plain_hi["clap"] and r["mel"] <= plain_hi["mel"] for r in rs]),
                "n": len(rs)}

    for s, cfg in settings.items():
        if cfg["route"] == "raw":
            continue
        out["by_setting"][s] = block(sel(setting=s))
        for et, mat, _t in keys:
            k = key_name(et, mat)
            out["by_key"].setdefault(k, {})[s] = block(sel(setting=s, key=k))
    for s in list(settings) + ["raw"]:
        ps = [p for p in pairs if p["setting"] == s]
        if not ps:
            continue
        out["variants"][s] = {kind: {**{m: dist([p[m] for p in ps if p["kind"] == kind]) for m in METRICS},
                                     "n": sum(p["kind"] == kind for p in ps)}
                              for kind in ("same", "diff", "plain", "anchors") if any(p["kind"] == kind for p in ps)}
    return out


def shape_summary(clips, keys, settings):
    out = {}
    for et, mat, _t in keys:
        k = key_name(et, mat)
        out[k] = {}
        for s in settings:
            rs = shape_rows(clips[s][k])
            out[k][s] = {f: med([r[f] for r in rs]) for f in SHAPE_SHOW + ("crest_db", "low500")}
    return out


def static_summary(clips, keys, settings):
    out = {}
    for s in settings:
        rules = Counter()
        n = 0
        for et, mat, _t in keys:
            for x in clips[s][key_name(et, mat)]:
                bad, m = is_static(x, SR)
                n += 1
                if bad:
                    rules[m["rule"]] += 1
        out[s] = {"flagged": sum(rules.values()) / float(max(1, n)), "rules": dict(rules), "n": n}
    return out


def build_listen(clips, keys, assign, settings, args):
    gap, kgap = np.zeros(int(0.25 * SR), np.float32), np.zeros(int(0.8 * SR), np.float32)
    anc = [s for s, c in settings.items() if c["route"] == "anchor"]
    seq, order = [], []
    for et, mat, _t in keys:
        k = key_name(et, mat)
        for v in range(min(args.listen_seeds, args.seeds)):
            names = ["raw", "replay"] + anc + [CURRENT]
            for s in names:
                seq += [clips[s][k][v], gap]
            order.append("%s v%d [%s] anchor %s" % (k, v, " ".join(names), assign[k][v]))
            seq.append(gap)
        seq.append(kgap)
    y = np.concatenate(seq)
    y = y / max(1e-9, float(np.abs(y).max())) * 0.9
    p = os.path.join(ROOT, args.listen)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    wavfile.write(p, SR, (y * 32767).astype(np.int16))
    return p, order


def f2(x, w=5, d=2):
    return ("%*.*f" % (w, d, x)) if x == x else " " * (w - 3) + "nan"


def report(pinfo, lib, silent, ret, settings, scheds, runs, copy, shapes, static, keys, man_path, listen, order, args):
    W = 160
    anc = [s for s, c in settings.items() if c["route"] == "anchor"]
    print()
    print("=" * W)
    print(" ANCHOR ROUTE   %s" % CAUTION)
    print("=" * W)
    print(" POOL  %s" % pinfo["rule"])
    print("   candidates %d, kept %d (silent %d), excluded %s" % (
        pinfo["candidates"], len(lib), len(silent), ", ".join("%s %d" % kv for kv in sorted(pinfo["excluded"].items()))
        or "none"))
    print("   held-out sets %s  allow sets %s  explicit clips %d (unresolved %d)  eval manifest %s" % (
        pinfo["held_counts"], pinfo["allow_counts"], pinfo["explicit_clips"], pinfo["unresolved_clips"], man_path))
    lk = pinfo["leak"]
    print("   sources shared with the %s references: clusters %d, recordings %d, files %d, held-out clusters %d;"
          " ESC-50 files %d, their FoleySet uploaders %d" % (
              "held-out" if lk["heldout_manifest"] else "eval", lk["clusters_in_refs"], lk["recordings_in_refs"],
              lk["sources_in_refs"], lk["clusters_held_out"], pinfo["esc_files"], len(pinfo["esc_uploaders"])))
    by_ds = Counter((r["dataset"], r["split"]) for r in lib)
    print("   library %s; %d sources" % (", ".join("%s %s %d" % (a, b, n) for (a, b), n in sorted(by_ds.items())),
                                         len({r["cluster"] for r in lib})))
    au = pinfo["audit"]
    print("   FoleySet audit by PCM md5: %d files, %d distinct; %d groups repeat one recording under several rows (%d"
          " files), %d of them mix sub-categories (%d files: %s); eval references inside mixed groups: %s" % (
              au["foleyset_files"], au["distinct_audio"], au["duplicate_groups"], au["files_in_duplicate_groups"],
              au["cross_label_groups"], au["files_in_cross_label_groups"],
              ", ".join("%s %d" % kv for kv in list(au["cross_label_by_sub"].items())[:8]),
              ", ".join("%s %d" % kv for kv in sorted(au["eval_refs_in_cross_label_groups"].items())) or "none"))
    print("-" * W)
    print(" RETRIEVAL  CLAP text embedding of the SARGE prompt against the CLAP audio embedding of every library clip at"
          " %.1f s; top %d by cosine; anchor of variant v = rank v mod %d" % (KEEP_S, args.topk, args.topk))
    print("   match shares are for reading only: label = FoleySet sub-category or C3GD form factor of a real reference"
          " set of the key's type, cat = its FoleySet category, surf = surface named in the tags, eval = anchor shares"
          " an uploader or gun with that type's eval references")
    for et, mat, t in keys:
        k = key_name(et, mat)
        r = ret[k]
        print("   %-19s text %5.1f ms search %5.2f ms  label %s cat %s%s eval %s  cos %.3f..%.3f (library median %.3f)"
              "  %d sources" % (
                  k, r["text_ms"], r["search_ms"], pct(r["label_share"]) if t else "    -",
                  pct(r["category_share"]) if t else "    -",
                  (" surf %s" % pct(r["surface_share"])) if r["surface_share"] is not None else "",
                  pct(r["eval_source_share"]) if t else "    -", r["top"][0]["cos"], r["cos_k"],
                  r["cos_median_library"], r["sources"]))
        print("   %19s %s" % ("", "  ".join("%s %s/%s %.3f" % (safe(x["id"]), safe(x["label"], 18),
                                                              safe(x["cluster"].split(":", 1)[1], 16), x["cos"])
                                         for x in r["top"])))
    print("-" * W)
    print(" SCHEDULES  sigma per sampler step as stable_audio_3 build_schedule makes it for %d steps" % CUR["steps"])
    for s in [x for x in settings if x in scheds]:
        sc = scheds[s]
        print("   %-12s %s" % (s, ("%s  %s" % (sc["sigmas"], sc["dist_shift"])) if "sigmas" in sc else sc.get("error")))
    print("-" * W)
    th = copy["thresholds"]
    print(" COPYING  each output against its anchor at %.1f s: clap = CLAP audio cosine, mel = log-mel pattern"
          " correlation, wave = waveform correlation, env = 5 ms envelope correlation, best lag within +-%g ms" % (
              KEEP_S, args.max_lag_ms))
    print("   nulls: replay = encode and decode of the anchor (what pure replay scores), plain = production generation"
          " of the same key and seed against the same anchor, neighbour = nearest other library recording to the"
          " anchor")
    for name, d in copy["nulls"].items():
        if isinstance(d, dict):
            print("   %-14s %s" % (name, "  ".join("%s %s [%s,%s]" % (m, f2(d[m]["median"]), f2(d[m]["p10"]),
                                                                          f2(d[m]["p90"])) for m in METRICS)))
    print("   replay-like = clap >= %.3f and mel >= %.3f (replay p10); plain-like = clap <= %.3f and mel <= %.3f (plain"
          " p90)" % (th["replay_p10"]["clap"], th["replay_p10"]["mel"], th["plain_p90"]["clap"], th["plain_p90"]["mel"]))
    print("   %-12s %s  %8s %8s %6s %8s %8s" % ("setting", "  ".join("%-20s" % m for m in METRICS), "nn=anch",
                                               "rank", "nn_cos", "replay", "plain"))
    for s in anc + [CURRENT, "replay"]:
        b = copy["by_setting"].get(s)
        if not b:
            continue
        print("   %-12s %s  %8s %8.0f %6.3f %8s %8s" % (
            s, "  ".join("%s [%s,%s]   " % (f2(b[m]["median"]), f2(b[m]["p10"]), f2(b[m]["p90"])) for m in METRICS),
            pct(b["nn_is_anchor"]), b["anchor_rank_median"], b["nn_cos"]["median"], pct(b["replay_like"]),
            pct(b["plain_like"])))
    print("   per key, median clap / mel against the anchor, then replay-like share")
    for et, mat, _t in keys:
        k = key_name(et, mat)
        bk = copy["by_key"][k]
        print("   %-19s %s" % (k, "  ".join("%s %s/%s %s" % (s, f2(bk[s]["clap"]["median"]), f2(bk[s]["mel"]["median"]),
                                                             pct(bk[s]["replay_like"]).strip())
                                           for s in anc + [CURRENT, "replay"] if s in bk)))
    print("-" * W)
    print(" VARIANTS  pairs of the %d seeds of one key: same = both from one anchor, diff = two different anchors,"
          " anchors = the retrieved real clips against each other" % args.seeds)
    for s, by in copy["variants"].items():
        print("   %-12s %s" % (s, "   ".join("%s n%-4d %s" % (kind, d["n"], " ".join("%s %s" % (m, f2(d[m]["median"]))
                                                                                    for m in METRICS))
                                            for kind, d in by.items())))
    print("-" * W)
    print(" SHAPE  medians per key (attack ms 10-90%, tail ms within 30 dB, floor dB, level 100 ms after the peak);"
          " raw is the real anchor")
    cols = ["raw", "replay"] + anc + [CURRENT]
    print("   %-19s %s" % ("", "  ".join("%-27s" % s for s in cols)))
    for et, mat, _t in keys:
        k = key_name(et, mat)
        print("   %-19s %s" % (k, "  ".join("%5.1f %5.0f %6.1f %6.1f     " % tuple(shapes[k][s][f] for f in SHAPE_SHOW)
                                           for s in cols)))
    print("   static guard verdict on the saved clips (raw = real anchors, so a flag there means the guard objects to"
          " real room tone): %s" % "  ".join("%s %s%s" % (s, pct(static[s]["flagged"]).strip(), (" (%s)" % ", ".join(
              "%s %d" % kv for kv in sorted(static[s]["rules"].items()))) if static[s]["rules"] else "") for s in cols))
    print("-" * W)
    print(" LATENCY  per clip, batch 1, after warm-up; anchor clips include the anchor encode, sampling, decode,"
          " _finish and the static guard")
    for s in [CURRENT, "replay"] + anc:
        r = runs.get(s)
        if not r:
            print("   %-12s reused files, no timing" % s)
            continue
        print("   %-12s median %6.1f ms  p90 %6.1f  max %6.1f  over %d clips  %s" % (
            s, r["ms_median"], r["ms_p90"], r["ms_max"], r["clips_timed"],
            " ".join("%s %s" % kv for kv in sorted(r["static"].items()))))
    tm = med([ret[key_name(e, m)]["text_ms"] for e, m, _t in keys])
    sm = med([ret[key_name(e, m)]["search_ms"] for e, m, _t in keys])
    print("   retrieval per new key: CLAP text %.1f ms + search %.2f ms over %d clips (median over keys); later clips of"
          " the key reuse the anchor list" % (tm, sm, len(lib)))
    for s in anc:
        r = runs.get(s)
        if r and runs.get(CURRENT):
            print("   %-12s steady %.1f ms per clip (+%.1f vs plain), first clip of a new key %.1f ms" % (
                s, r["ms_median"], r["ms_median"] - runs[CURRENT]["ms_median"], r["ms_median"] + tm + sm))
    print("=" * W)
    print(" LISTEN %s  per key and seed: %s" % (rel(listen), " | ".join(order[:4]) + (" | ..." if len(order) > 4 else "")))
    print(" SCORE with the validated judges: python scripts/bench/anchor_route.py score --gen-dir %s" % args.gen_dir)
    print("=" * W)


def parse(argv):
    ap = argparse.ArgumentParser(description="Anchor route probe: retrieve real anchors by CLAP for each SARGE key,"
                                             " generate from them through init_audio, and measure copying and latency.")
    ap.add_argument("--heldout", default=HELDOUT)
    ap.add_argument("--manifest", default=os.path.join("data", "refs", "manifest.json"))
    ap.add_argument("--foley-splits", nargs="+", default=["train"])
    ap.add_argument("--c3gd-splits", nargs="+", default=["train"])
    ap.add_argument("--c3gd-mics", nargs="+", default=["DAVE"])
    ap.add_argument("--c3gd-events", nargs="+", default=["oh_farm"])
    ap.add_argument("--c3gd-holdout-every", type=int, default=3)
    ap.add_argument("--keys", nargs="+", default=None)
    ap.add_argument("--levels", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    ap.add_argument("--schedules", nargs="+", default=["package"], choices=["package", "scaled"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit-library", type=int, default=None)
    ap.add_argument("--gen-dir", default=OUT_DIR)
    ap.add_argument("--library-cache", default=os.path.join(OUT_DIR, "library_clap.npz"))
    ap.add_argument("--hash-cache", default=os.path.join(OUT_DIR, "audio_md5.json"))
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--max-lag-ms", type=float, default=50.0)
    ap.add_argument("--text-reps", type=int, default=5)
    ap.add_argument("--listen", default=os.path.join("data", "_gate", "listen", "anchor_route.wav"))
    ap.add_argument("--listen-seeds", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "bench", "out_anchor_route.json"))
    args = ap.parse_args(argv)
    if any(not 0.0 < lv < 1.0 for lv in args.levels):
        raise SystemExit("noise levels sit strictly between 0 and 1, got %s" % args.levels)
    if args.topk < 1 or args.seeds < 1:
        raise SystemExit("--topk and --seeds start at 1")
    return args


def select_keys(args):
    if not args.keys:
        return list(KEYS)
    want = set(args.keys)
    keys = [k for k in KEYS if key_name(k[0], k[1]) in want]
    miss = want - {key_name(k[0], k[1]) for k in keys}
    if miss:
        raise SystemExit("unknown keys %s; vocab_eval.KEYS holds %s" % (sorted(miss), [key_name(e, m) for e, m, _t in KEYS]))
    return keys


def run(argv=None):
    args = parse(argv)
    t0 = time.perf_counter()
    print(CAUTION, flush=True)
    spec = held_spec(args.heldout)
    man_path = os.path.join(ROOT, spec["manifest"]) if spec and spec.get("manifest") else args.manifest
    man, man_rel = load_manifest(man_path)
    if man is None and spec and spec.get("manifest"):
        raise SystemExit("%s names the eval manifest %s, which is missing" % (spec["path"], man_rel))
    pool, pinfo = build_pool(args, spec, man)
    _recs, by_type = manifest_index(man)
    keys = select_keys(args)
    print("pool: %d clips (%s), %.1f s" % (len(pool), pinfo["rule"], time.perf_counter() - t0), flush=True)
    clap = Clap(args)
    lib, lib_emb, silent = embed_library(pool, clap, args)
    if len(lib) < args.topk:
        raise SystemExit("library holds %d clips, fewer than --topk %d" % (len(lib), args.topk))
    ret = retrieve(clap, lib, lib_emb, keys, args, by_type)
    clap.close("clap library")
    assign = {k: [r["top"][v % args.topk]["id"] for v in range(args.seeds)] for k, r in ret.items()}
    need = dedupe([a for k in assign for a in assign[k]])
    idx = {r["id"]: i for i, r in enumerate(lib)}
    anchors = {}
    for aid in need:
        p = prepare(lib[idx[aid]])
        if p is None:
            raise SystemExit("anchor %s came back silent" % aid)
        anchors[aid] = p
    settings = plan(args)
    skip = reusable(args, keys, assign, settings)
    write_raw(args, keys, assign, anchors)
    ginfo, runs, scheds = generate(args, keys, assign, anchors, settings, skip)
    p = os.path.join(ROOT, args.gen_dir, "settings.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(stamp(args, keys, assign, settings, man_rel), fh, indent=1)
    route_files = write_routes(args, keys, assign, settings, {aid: lib[idx[aid]] for aid in need})
    clips = load_outputs(args, keys, settings)
    clap = Clap(args)
    emb = embed_outputs(clap, clips, keys, assign, settings)
    clap.close("clap outputs")
    rows, pairs, near, lags = copy_measure(clips, emb, lib, lib_emb, keys, assign, settings, args)
    copy = summarize_copy(rows, pairs, near, settings, keys)
    shapes = shape_summary(clips, keys, settings)
    static = static_summary(clips, keys, settings)
    listen, order = build_listen(clips, keys, assign, settings, args)
    report(pinfo, lib, silent, ret, settings, scheds, runs, copy, shapes, static, keys, man_rel, listen, order, args)
    out = {"meta": {"root": ROOT, "args": vars(args), "sr": SR, "intensity": INTENSITY, "current": CURRENT, "cfg": CUR,
                    "version": VERSION, "fingerprint": fingerprint(), "clap": CLAP_NAME, "caution": CAUTION,
                    "generator": ginfo, "schedules": scheds, "lags": lags, "manifest": man_rel,
                    "seconds": time.perf_counter() - t0},
           "pool": {**pinfo, "library": len(lib), "silent": silent,
                    "by_dataset": {"%s|%s" % k: n for k, n in Counter((r["dataset"], r["split"]) for r in lib).items()},
                    "sources": len({r["cluster"] for r in lib}),
                    "eval_source_overlap": {t: sum(r["cluster"] in cl for r in lib) for t, cl in by_type.items()}},
           "retrieval": ret, "assign": assign,
           "anchors": {aid: {k: v for k, v in lib[idx[aid]].items()} | {"padded": anchors[aid]["padded"],
                                                                         "src_s": anchors[aid]["src_s"]}
                       for aid in need},
           "settings": settings, "runs": runs, "copy": copy, "neighbors": near, "shape": shapes, "static": static,
           "rows": rows,
           "listen": {"path": rel(listen), "order": order},
           "files": {s: rel(os.path.join(ROOT, args.gen_dir, s)) for s in settings}, "route_json": route_files}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("saved -> %s" % args.out)
    return 0


def score(argv=None):
    ap = argparse.ArgumentParser(description="Score anchor route clips with vocab_eval's validated judges.")
    ap.add_argument("--gen-dir", default=OUT_DIR)
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "bench", "out_anchor_vocab.json"))
    ap.add_argument("--listen", default=os.path.join("data", "_gate", "listen", "anchor_vocab.wav"))
    args, rest = ap.parse_known_args(argv)
    p = os.path.join(ROOT, args.gen_dir, "settings.json")
    if not os.path.isfile(p):
        raise SystemExit("no settings.json in %s; run anchor_route.py first" % rel(os.path.dirname(p)))
    st = json.load(open(p, encoding="utf-8"))
    if ve.CURRENT not in st["settings"]:
        raise SystemExit("settings.json lacks the current setting %s, which vocab_eval compares against" % ve.CURRENT)
    routes = [s for s, c in st["settings"].items() if c.get("route") != "plain"]
    miss = [s for s in routes if not os.path.isfile(os.path.join(ROOT, args.gen_dir, s, "route.json"))]
    if miss:
        raise SystemExit("no route.json in %s; rerun anchor_route.py with --reuse to write them" % miss)
    want = set(st["keys"])
    ve.KEYS = [k for k in ve.KEYS if key_name(k[0], k[1]) in want]
    extra = [] if "--seeds" in rest else ["--seeds", str(st["seeds"])]
    if "--manifest" not in rest and "--refs" not in rest and st.get("manifest"):
        extra += ["--manifest", st["manifest"]]
    if "--settings" not in rest:
        extra += ["--settings", ve.CURRENT]
    for s in routes:
        extra += ["--route", "%s=%s" % (s, os.path.join(args.gen_dir, s))]
    print("scoring %s plus routes %s x %d keys from %s with vocab_eval" % (
        ve.CURRENT, ", ".join(routes), len(ve.KEYS), rel(os.path.dirname(p))), flush=True)
    return ve.main(["--reuse", "--gen-dir", args.gen_dir, "--out", args.out, "--listen", args.listen] + extra + rest)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "score":
        return score(argv[1:])
    if argv and argv[0] == "run":
        argv = argv[1:]
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
