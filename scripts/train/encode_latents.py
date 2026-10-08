import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.path.dirname(HERE), "bench")
for _p in (BENCH, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap

import argparse
import csv
import html
import json
import re
import time
import warnings
from collections import Counter, defaultdict
from math import gcd

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from sarge.loudness import PEAK_CEILING, target_for

ROOT = os.path.dirname(os.path.dirname(HERE))
SR = 44100
MODEL = "small-sfx"
DATA = r"D:\datasets\sarge_refs"
FOLEY = os.path.join(DATA, "foleyset", "FoleySet")
C3GD = os.path.join(DATA, "c3gd", "C3GD")
ESC_DIRS = [os.path.join(ROOT, "data", "esc50"), os.path.join(ROOT, "data", "real_footsteps")]
SOURCES = os.path.join(BENCH, "heldout_sources.json")
OUT = os.path.join("data", "lora", "real")
STYLES = ("natural", "sarge")
SUFFIX = "dry, isolated sound effect"
INTENSITY = "normal"
FADE_IN = 0.005
FADE_OUT = 0.05
SILENT_RMS = 1e-6
CLIP_LEVEL = 0.999
LICENSES = {
    "foleyset": {"license": "CC BY 4.0", "source": "FoleySet, Zenodo record 20735877 (CC0 Freesound clips)",
                 "cite": "Wang and Lerch, FoleySet: A Multi-Level Human-Annotated Foley Sound Dataset, arXiv:2606.25980"},
    "c3gd": {"license": "CC BY 4.0", "source": "C3GD v0.2.0, Zenodo record 22286299",
             "cite": "Gurny and Quinn, Descriptor: Certus Caliber Classification Gunshot Dataset (C3GD), arXiv:2606.18135"},
}
POOL_RULES = {
    "foleyset": "a FoleySet clip trains when foleyset.uploaders[username].pool of the sources file is train, its "
                "uploader is absent from foleyset.heldout_uploaders, and FoleySet puts it in split train (split val "
                "goes to the val pool); clips tagged ai-generated, clips of a Freesound recording that ESC-50 also "
                "holds, and every clip of an uploader that FoleySet names for the Freesound recording of an ESC-50 "
                "clip under data/esc50 or data/real_footsteps stay out; ESC-50 file names carry only the recording "
                "id, so the uploaders of ESC-50 recordings that FoleySet lacks stay unknown",
    "c3gd": "a C3GD clip trains when c3gd.guns[platform_id].pool of the sources file is train, it was recorded on the "
            "training mic named by meta.train_mic, and C3GD puts it in split train (split val goes to the val pool)",
    "esc50": "never used",
    "hear": "never used",
}
CAPTION_RULES = {
    "natural": "FoleySet: the uploader description with markup and links removed, whole sentences until at least "
               "--min-words words, cut at --max-words; a description under two words falls back to the sub-category "
               "phrase and the first --max-tags tags. C3GD: '<caliber> <form factor> firing, recorded <distance> m "
               "away' from metadata.csv, platforms.csv and microphone_locations.csv",
    "sarge": "'a <sub-category phrase>, <top tag>, dry, isolated sound effect' (the tag part dropped when no tag "
             "qualifies), the same suffix every production prompt from build_prompt ends with. C3GD uses '<form "
             "factor> gunshot' as the sub-category and the caliber as the tag",
    "top_tag": "the clip tag carried by the most training clips of its sub-category, among tags with at least "
               "--min-tag-df clips from --min-tag-users uploaders in that sub-category, at most --max-tag-share of the "
               "sub-category, at most --max-cat-share of its FoleySet category, present in at most --max-tag-spread "
               "of all sub-categories, and sharing no stem of three or more letters with the sub-category or "
               "category names; ties go to the alphabetically first tag",
}
ESC_NAME = re.compile(r"^(\d+)-(\d+)-([A-Z])-(\d+)\.wav$")
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
URL = re.compile(r"https?://\S+|www\.\S+", re.I)
MARKUP = re.compile(r"<[^>]+>")
SPACE = re.compile(r"\s+")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def rel(p):
    try:
        return os.path.relpath(p, ROOT).replace(os.sep, "/")
    except ValueError:
        return os.path.abspath(p).replace(os.sep, "/")


def at_root(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def load_sources(path):
    p = at_root(path)
    if not os.path.isfile(p):
        raise SystemExit("%s is missing; build the held-out split first with scripts/bench/vocab_refs.py --heldout"
                         % rel(p))
    s = json.load(open(p, encoding="utf-8"))
    ev, mic, ch = s["meta"]["train_mic"].split("|")
    ups, guns = s["foleyset"].get("uploaders"), s["c3gd"].get("guns")
    return {"users": set(s["foleyset"]["heldout_uploaders"]), "guns": set(s["c3gd"]["heldout_guns"]),
            "train_users": {u for u, v in ups.items() if v.get("pool") == "train"} if ups else None,
            "train_guns": ({g for g, v in guns.items() if v.get("pool") == "train"} if guns else
                           set(s["c3gd"].get("train_guns", []))),
            "mic": (ev, mic, "" if ch == "mono" else ch), "path": rel(p), "rules": s["meta"].get("rules", {}),
            "trainable": s["meta"].get("trainable", "")}


def tag_list(text):
    out = []
    for t in text.split(","):
        t = t.strip().lower()
        if t and t not in out:
            out.append(t)
    return out


def phrase(label):
    return SPACE.sub(" ", CAMEL.sub(" ", label).replace("/", " ").replace("_", " ").replace("-", " ")).strip().lower()


def article(p):
    return ("an " if p[:1] in "aeiou" else "a ") + p


def clean(text, min_words, max_words):
    t = MARKUP.sub(" ", html.unescape(text or ""))
    t = SPACE.sub(" ", URL.sub(" ", t)).strip()
    out = []
    for sent in SENTENCE.split(t):
        out += sent.split(" ")
        if len(out) >= min_words:
            break
    return " ".join(out[:max_words]).strip()


def esc_sources():
    ids = {}
    for d in ESC_DIRS:
        for root, _ds, fs in os.walk(d):
            for f in sorted(fs):
                m = ESC_NAME.match(f)
                if m:
                    ids.setdefault(m.group(2), rel(os.path.join(root, f)))
    return ids


def fs_pool(r, src):
    u = r["username"]
    if u in src["users"] or (src["train_users"] is not None and u not in src["train_users"]):
        return "heldout"
    return {"train": "train", "val": "val"}.get(r["split"], "unused")


def gun_pool(r, src):
    if r["platform_id"] in src["guns"] or r["platform_id"] not in src["train_guns"]:
        return "heldout"
    if (r["event_id"], r["mic_id"], r["channel_orientation"]) != src["mic"]:
        return "unused"
    return {"train": "train", "val": "val"}.get(r["split"], "unused")


def foley_entries(src, args, report):
    rows = list(csv.DictReader(open(os.path.join(FOLEY, "metadata", "foleyset_metadata.csv"), encoding="utf-8")))
    esc = esc_sources()
    users = {}
    for r in rows:
        users.setdefault(r["freesound-id"], r["username"])
    esc_users = {users[i] for i in esc if i in users}
    report["esc_overlap"] = {"esc_files": len(esc), "freesound_ids_in_foleyset": sum(i in users for i in esc),
                             "uploaders": sorted(esc_users)}
    out = []
    for r in rows:
        pool = fs_pool(r, src)
        report["pools"]["foleyset"][pool] += 1
        if pool not in args.pools:
            continue
        tags = tag_list(r["original-tags"])
        why = None
        if "ai-generated" in tags:
            why = "ai-generated"
        elif r["freesound-id"] in esc:
            why = "same Freesound recording as ESC-50 file %s" % esc[r["freesound-id"]]
        elif r["username"] in esc_users:
            why = "uploader of an ESC-50 evaluation clip"
        if why:
            report["excluded"].append({"dataset": "foleyset", "id": r["name"], "pool": pool, "user": r["username"],
                                       "freesound_id": r["freesound-id"], "reason": why})
            continue
        path = os.path.join(FOLEY, "audio", r["name"])
        if not os.path.isfile(path):
            raise SystemExit("FoleySet audio missing: %s" % path)
        out.append({"id": "fs_%s" % r["name"][:-4], "dataset": "foleyset", "src": path, "pool": pool,
                    "label": r["sub-category"], "category": r["category"], "sub_phrase": phrase(r["sub-category"]),
                    "tags": tags, "description": r["description"], "cluster": "fs:%s" % r["username"],
                    "recording": "fs:%s" % r["freesound-id"], "username": r["username"], "url": r["url"],
                    "freesound_id": r["freesound-id"], "oneshot": r["Oneshot-Multi"] == "One-shot",
                    "split": r["split"], "license": LICENSES["foleyset"]["license"]})
    return out


def c3gd_entries(src, args, report):
    rows = list(csv.DictReader(open(os.path.join(C3GD, "metadata.csv"), encoding="utf-8")))
    plat = {r["platform_id"]: r for r in csv.DictReader(open(os.path.join(C3GD, "metadata", "platforms.csv"),
                                                              encoding="utf-8"))}
    locs = list(csv.DictReader(open(os.path.join(C3GD, "metadata", "microphone_locations.csv"), encoding="utf-8")))
    ev, mic, _ch = src["mic"]
    dist = sorted({float(r["distance_m"]) for r in locs if r["mic_name"] == mic and r["event_name"] == ev})
    if len(dist) != 1:
        raise SystemExit("training mic %s at %s has distances %s in microphone_locations.csv" % (mic, ev, dist))
    out = []
    for r in rows:
        pool = gun_pool(r, src)
        report["pools"]["c3gd"][pool] += 1
        if pool not in args.pools:
            continue
        ff = plat[r["platform_id"]]["form_factor"] if r["platform_id"] in plat else "unknown"
        path = os.path.join(C3GD, "data", r["filename"])
        if not os.path.isfile(path):
            raise SystemExit("C3GD audio missing: %s" % path)
        out.append({"id": "c3gd_%s" % r["datum_id"], "dataset": "c3gd", "src": path, "pool": pool,
                    "label": "%s gunshot" % ff, "category": "gunshot", "sub_phrase": "%s gunshot" % ff,
                    "tags": [r["class_name"].strip().lower()], "description": "",
                    "cluster": "c3gd:%s" % r["platform_id"],
                    "recording": "c3gd:%s:%s:%s" % (r["event_id"], r["file_id"], r["clip_id"]),
                    "platform": r["platform_id"], "form_factor": ff, "caliber": r["class_name"].strip(),
                    "cartridge": r["cartridge_id"], "event": r["event_id"], "mic": r["mic_id"], "distance_m": dist[0],
                    "split": r["split"], "license": LICENSES["c3gd"]["license"]})
    return out


def spaced(items, k):
    if k is None or len(items) <= k:
        return list(items)
    idx = np.unique(np.round(np.linspace(0, len(items) - 1, k)).astype(int))
    return [items[i] for i in idx]


def build_pool(args, report):
    src = load_sources(args.sources)
    entries = foley_entries(src, args, report) + c3gd_entries(src, args, report)
    entries.sort(key=lambda e: (args.pools.index(e["pool"]), e["dataset"], e["id"]))
    if args.limit:
        entries = [e for p in args.pools for e in spaced([x for x in entries if x["pool"] == p], args.limit)]
    return src, entries


def tag_stats(entries):
    lab = defaultdict(lambda: {"n": 0, "df": Counter(), "users": defaultdict(set)})
    cat = defaultdict(lambda: {"n": 0, "df": Counter()})
    spread = defaultdict(set)
    for e in entries:
        if e["dataset"] != "foleyset" or e["pool"] != "train":
            continue
        s, c = lab[e["label"]], cat[e["category"]]
        s["n"] += 1
        c["n"] += 1
        for t in e["tags"]:
            s["df"][t] += 1
            s["users"][t].add(e["username"])
            c["df"][t] += 1
            spread[t].add(e["label"])
    return {"label": lab, "category": cat, "spread": spread, "labels": max(1, len(lab))}


def stems(e):
    out = set()
    for s in (e["label"], e["category"], e["sub_phrase"], phrase(e["category"])):
        out |= {w for w in re.split(r"[^a-z0-9]+", s.lower()) if len(w) >= 3}
    return out


def related(t, ws):
    tt = re.sub(r"[^a-z0-9]+", "", t)
    return len(tt) < 3 or any(tt in w or w in tt for w in ws)


def top_tag(e, st, args):
    s, c = st["label"].get(e["label"]), st["category"].get(e["category"])
    if s is None or c is None:
        return ""
    ws = stems(e)
    best = None
    for t in e["tags"]:
        if related(t, ws) or len(st["spread"][t]) > args.max_tag_spread * st["labels"]:
            continue
        df = s["df"][t]
        if (df < args.min_tag_df or len(s["users"][t]) < args.min_tag_users or df > args.max_tag_share * s["n"] or
                c["df"][t] > args.max_cat_share * c["n"]):
            continue
        if best is None or (-df, t) < best:
            best = (-df, t)
    return best[1].replace("-", " ") if best else ""


def sarge_caption(sub, tag):
    head = article(sub)
    return "%s, %s, %s" % (head, tag, SUFFIX) if tag else "%s, %s" % (head, SUFFIX)


def caption(e, st, args):
    if e["dataset"] == "c3gd":
        tag = e["tags"][0]
        nat = "%s firing, recorded %g m away" % (article("%s %s" % (e["caliber"], e["form_factor"])), e["distance_m"])
    else:
        tag = top_tag(e, st, args)
        nat = clean(e["description"], args.min_words, args.max_words)
        if len(nat.split()) < 2:
            nat = ", ".join([e["sub_phrase"]] + [t.replace("-", " ") for t in e["tags"][:args.max_tags]])
    return {"natural": nat, "sarge": sarge_caption(e["sub_phrase"], tag)}, tag


def label_weights(entries, mode):
    n = Counter(e["label"] for e in entries)
    for e in entries:
        k = n[e["label"]]
        e["weight"] = 1.0 if mode == "none" else (1.0 / np.sqrt(k) if mode == "sqrt" else 1.0 / k)


def read_audio(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr, a = wavfile.read(path)
    if a.dtype == np.int16:
        x = a.astype(np.float32) / 32767.0
    elif a.dtype == np.int32:
        x = a.astype(np.float32) / 2147483647.0
    elif a.dtype == np.uint8:
        x = (a.astype(np.float32) - 128.0) / 128.0
    else:
        x = a.astype(np.float32)
    x = x[:, None] if x.ndim == 1 else x
    return np.ascontiguousarray(x.T), int(sr)


def to_sr(x, sr):
    if sr == SR:
        return np.asarray(x, dtype=np.float32)
    g = gcd(SR, int(sr))
    return np.stack([resample_poly(np.asarray(c, dtype=np.float64), SR // g, int(sr) // g) for c in x]).astype(
        np.float32)


def stereo(x):
    return np.repeat(x, 2, axis=0) if x.shape[0] == 1 else np.ascontiguousarray(x[:2])


def cut(x, n, preroll):
    if x.shape[1] <= n:
        seg = x.copy()
        fo = min(int(FADE_OUT * SR), seg.shape[1] // 2)
        if fo > 0:
            seg[:, -fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
        return seg, 0
    mono = x.mean(axis=0).astype(np.float64)
    win = max(1, int(0.005 * SR))
    energy = np.convolve(mono ** 2, np.ones(win) / win, mode="same")
    start = max(0, int(np.argmax(energy)) - int(preroll * SR))
    end = min(x.shape[1], start + n)
    start = max(0, end - n)
    seg = x[:, start:end].copy()
    fi = min(int(FADE_IN * SR), seg.shape[1] // 2)
    fo = min(int(FADE_OUT * SR), seg.shape[1] // 2)
    if fi > 0:
        seg[:, :fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 0:
        seg[:, -fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return seg, start


def prep(x, sr, n_gen, preroll):
    y = stereo(to_sr(x, sr))
    seg, start = cut(y, n_gen, preroll)
    out = np.zeros((2, n_gen), dtype=np.float32)
    out[:, :seg.shape[1]] = seg
    mono = out.mean(axis=0).astype(np.float64)
    r = float(np.sqrt(np.mean(mono * mono)))
    info = {"src_sr": int(sr), "src_frames": int(x.shape[1]), "src_channels": int(x.shape[0]),
            "content_samples": int(seg.shape[1]), "window_start_s": round(start / float(SR), 4),
            "clipped_frac": float(np.mean(np.abs(x) >= CLIP_LEVEL)) if x.size else 0.0,
            "padded": bool(seg.shape[1] < n_gen)}
    if r < SILENT_RMS:
        return None, info
    g = target_for(INTENSITY) / r
    pk = float(np.max(np.abs(out))) * g
    if pk > PEAK_CEILING:
        g *= PEAK_CEILING / pk
    info["gain_db"] = round(float(20.0 * np.log10(g)), 2)
    return (out * np.float32(g)).astype(np.float32), info


def load_model(device=None, half=True):
    import torch
    from stable_audio_3 import StableAudioModel
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return StableAudioModel.from_pretrained(MODEL, device=dev, model_half=half)


def sizes(sam, gen_s, pad_s):
    n_audio = int(sam._adapt_sample_size([{"seconds_total": gen_s}], sam.model_config["sample_size"], pad_s))
    ds = int(sam.model.pretransform.downsampling_ratio)
    return n_audio, n_audio // ds, ds


def encode_audio(sam, windows, n_audio):
    import torch
    pt = sam.model.pretransform
    x = torch.zeros((len(windows), 2, n_audio), dtype=torch.float32)
    for i, w in enumerate(windows):
        x[i, :, :w.shape[1]] = torch.from_numpy(w)
    x = x.to(str(sam.device)).to(next(pt.parameters()).dtype)
    with torch.inference_mode():
        z = pt.encode(x)
    return z.to(torch.float16).cpu()


def encode_text(sam, prompts, gen_s, bs):
    import torch
    cond = sam.model.conditioner
    dev = str(sam.device)
    toks, starts, lens, sec, width = [], [], [], None, None
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        with torch.inference_mode():
            out = cond([{"prompt": p, "seconds_total": gen_s} for p in chunk], dev)
        emb, mask = out["prompt"]
        width = int(emb.shape[1])
        m = mask.bool().cpu().numpy()
        for j, p in enumerate(chunk):
            pos = np.flatnonzero(m[j])
            if not len(pos) or pos[-1] - pos[0] + 1 != len(pos):
                raise SystemExit("prompt mask is not one contiguous run for %r" % p)
            toks.append(emb[j, int(pos[0]):int(pos[-1]) + 1].to(torch.float16).cpu())
            starts.append(int(pos[0]))
            lens.append(int(len(pos)))
        if sec is None:
            sec = out["seconds_total"][0][0, 0].to(torch.float16).cpu()
    pad = cond.conditioners["prompt"].padding_embedding.detach().to(torch.float16).cpu()
    return {"tokens": toks, "starts": starts, "lengths": lens, "pad": pad, "seconds": sec, "width": width}


def pack_text(enc, index):
    import torch
    off = np.concatenate([[0], np.cumsum(enc["lengths"])]).astype(np.int64)
    return {"tokens": torch.cat(enc["tokens"], 0).contiguous(), "offsets": torch.from_numpy(off),
            "starts": torch.tensor(enc["starts"], dtype=torch.int64), "index": torch.tensor(index, dtype=torch.int64),
            "pad": enc["pad"].contiguous(), "seconds": enc["seconds"].contiguous(),
            "width": torch.tensor([enc["width"]], dtype=torch.int64)}


def text_tables(sam, kept, styles, gen_s, bs):
    out, uniq = {}, {}
    for style in styles:
        ps, pos = [], {}
        for e in kept:
            p = e["captions"][style]
            if p not in pos:
                pos[p] = len(ps)
                ps.append(p)
        t0 = time.perf_counter()
        enc = encode_text(sam, ps, gen_s, bs)
        out[style] = pack_text(enc, [pos[e["captions"][style]] for e in kept])
        uniq[style] = {"unique": len(ps), "tokens_mean": float(np.mean(enc["lengths"])),
                       "tokens_max": int(max(enc["lengths"])), "truncated": int(sum(n >= enc["width"] for n in
                                                                                      enc["lengths"])),
                       "seconds": round(time.perf_counter() - t0, 1)}
        print("text %-8s %5d unique captions  mean %.1f tokens  %.1f s" % (
            style, len(ps), uniq[style]["tokens_mean"], uniq[style]["seconds"]), flush=True)
    return out, uniq


def save_cache(out_dir, kept, latents, texts, meta, report):
    import torch
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    content = torch.tensor([e["content_frames"] for e in kept], dtype=torch.int32)
    save_file({"latents": latents.contiguous(), "content": content}, os.path.join(out_dir, "latents.safetensors"))
    for style, t in texts.items():
        save_file(t, os.path.join(out_dir, "text_%s.safetensors" % style))
    for i, e in enumerate(kept):
        e["row"] = i
    keep = ("row", "id", "dataset", "pool", "label", "category", "cluster", "recording", "license", "weight",
            "captions", "top_tag", "username", "url", "freesound_id", "oneshot", "split", "platform", "form_factor",
            "caliber", "cartridge", "event", "mic", "distance_m", "content_samples", "content_frames",
            "window_start_s", "src_sr", "src_frames", "src_channels", "clipped_frac", "padded", "gain_db", "src",
            "audio", "seed", "batch_pos")
    clips = [{k: (os.path.abspath(e[k]).replace(os.sep, "/") if k == "src" else e[k]) for k in keep if k in e}
             for e in kept]
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "report": report, "clips": clips}, fh, indent=1)
    with open(os.path.join(out_dir, "attribution.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "dataset", "license", "uploader_or_gun", "url", "freesound_id", "cite"])
        for e in kept:
            if e["dataset"] in LICENSES:
                w.writerow([e["id"], e["dataset"], e["license"], e.get("username") or e.get("platform", ""),
                            e.get("url", ""), e.get("freesound_id", ""), LICENSES[e["dataset"]]["cite"]])


def encode_pool(sam, entries, args, n_gen, n_audio, ds, report, source=None):
    kept, lat = [], []
    t0 = time.perf_counter()
    batch, ents = [], []

    def flush():
        if batch:
            lat.append(encode_audio(sam, batch, n_audio))
            kept.extend(ents)
            batch.clear()
            ents.clear()

    for i, e in enumerate(entries):
        x, sr = source(e) if source is not None else read_audio(e["src"])
        w, info = prep(x, sr, n_gen, args.preroll)
        e.update(info)
        if w is None:
            report["excluded"].append({"dataset": e["dataset"], "id": e["id"], "pool": e["pool"], "reason": "silent"})
            continue
        e["content_frames"] = int(-(-e["content_samples"] // ds))
        batch.append(w)
        ents.append(e)
        if len(batch) >= args.batch:
            flush()
        if (i + 1) % 500 == 0:
            print("  encoded %d/%d clips  %.1f s" % (i + 1, len(entries), time.perf_counter() - t0), flush=True)
    flush()
    import torch
    z = torch.cat(lat, 0) if lat else torch.zeros((0, 0, 0), dtype=torch.float16)
    return kept, z, time.perf_counter() - t0


def summary(kept, report, uniq, meta):
    W = 132
    print()
    print("=" * W)
    print(" LORA TRAINING CACHE  %s  (%d clips, latents %s, %.1f s windows in %d samples, preroll %.0f ms)" % (
        meta["out"], len(kept), meta["latent_shape"], meta["gen_s"], meta["n_audio"], 1000 * meta["preroll_s"]))
    print("=" * W)
    for ds_name, pc in report["pools"].items():
        print("   %-9s metadata rows by pool: %s" % (ds_name, ", ".join("%s %d" % x for x in sorted(pc.items()))))
    for pool in meta["pools"]:
        cs = [e for e in kept if e["pool"] == pool]
        by = Counter(e["dataset"] for e in cs)
        print("   %-6s %5d clips  %s  sources %d  labels %d  padded %d" % (
            pool, len(cs), ", ".join("%s %d" % x for x in sorted(by.items())), len({e["cluster"] for e in cs}),
            len({e["label"] for e in cs}), sum(e["padded"] for e in cs)))
    tr = [e for e in kept if e["pool"] == "train"]
    lab = Counter(e["label"] for e in tr)
    w = defaultdict(float)
    for e in tr:
        w[e["label"]] += e["weight"]
    tw = max(sum(w.values()), 1e-12)
    print("   train labels (clips, sampling share): %s" % ", ".join(
        "%s %d %.1f%%" % (k, n, 100 * w[k] / tw) for k, n in lab.most_common(16)))
    fs = [e for e in tr if e["dataset"] == "foleyset"]
    print("   FoleySet train clips with a top tag: %d of %d" % (sum(bool(e["top_tag"]) for e in fs), len(fs)))
    ex = Counter(x["reason"] if not x["reason"].startswith("same Freesound") else "same Freesound recording as ESC-50"
                 for x in report["excluded"])
    print("   excluded: %s" % (", ".join("%s %d" % x for x in ex.most_common()) or "none"))
    eo = report.get("esc_overlap")
    if eo:
        print("   ESC-50: %d recordings under data/esc50 and data/real_footsteps, %d of them in FoleySet, %d uploaders "
              "known from those and excluded, the rest unknown" % (eo["esc_files"], eo["freesound_ids_in_foleyset"],
                                                                  len(eo["uploaders"])))
    for style, u in uniq.items():
        print("   text %-8s %d unique, mean %.1f tokens, max %d, truncated at the conditioner length %d" % (
            style, u["unique"], u["tokens_mean"], u["tokens_max"], u["truncated"]))
    print("-" * W)
    for e in spaced(tr, 10):
        print("   %-12s %-22s natural: %s" % (e["id"], e["label"][:22], e["captions"]["natural"][:84]))
        print("   %-12s %-22s sarge:   %s" % ("", "", e["captions"]["sarge"][:84]))
    print("=" * W)


def parse(argv):
    ap = argparse.ArgumentParser(description="Encode the real training pool (FoleySet split train and C3GD split "
                                             "train on the training mic, held-out sources removed) to SA3 latents "
                                             "and cached prompt embeddings.")
    ap.add_argument("--sources", default=SOURCES)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pools", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--styles", nargs="+", default=list(STYLES), choices=list(STYLES))
    ap.add_argument("--gen-s", type=float, default=1.0)
    ap.add_argument("--pad-s", type=float, default=1.0)
    ap.add_argument("--preroll", type=float, default=0.03)
    ap.add_argument("--balance", default="sqrt", choices=["none", "sqrt", "flat"])
    ap.add_argument("--min-words", type=int, default=8)
    ap.add_argument("--max-words", type=int, default=40)
    ap.add_argument("--max-tags", type=int, default=4)
    ap.add_argument("--min-tag-df", type=int, default=3)
    ap.add_argument("--min-tag-users", type=int, default=2)
    ap.add_argument("--max-tag-share", type=float, default=0.5)
    ap.add_argument("--max-cat-share", type=float, default=0.2)
    ap.add_argument("--max-tag-spread", type=float, default=0.4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--text-bs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--captions-only", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    t0 = time.perf_counter()
    report = {"pools": {"foleyset": Counter(), "c3gd": Counter()}, "excluded": []}
    src, entries = build_pool(args, report)
    st = tag_stats(entries)
    for e in entries:
        e["captions"], e["top_tag"] = caption(e, st, args)
    label_weights([e for e in entries if e["pool"] == "train"], args.balance)
    for e in entries:
        e.setdefault("weight", 1.0)
    print("pool: %d clips (%s) from %s" % (len(entries), ", ".join("%s %d" % x for x in sorted(
        Counter(e["pool"] for e in entries).items())), src["path"]), flush=True)
    report["pools"] = {k: dict(v) for k, v in report["pools"].items()}
    out_dir = at_root(args.out)
    meta = {"model": MODEL, "sr": SR, "gen_s": args.gen_s, "pad_s": args.pad_s, "preroll_s": args.preroll,
            "styles": args.styles, "pools": args.pools, "balance": args.balance, "sources": src["path"],
            "pool_rules": POOL_RULES, "caption_rules": CAPTION_RULES, "sources_rules": src["rules"],
            "licenses": LICENSES, "audio": "real",
            "processing": "scipy wavfile, resample_poly to 44100 Hz, mono sources doubled to stereo, the loudest "
                          "window of the generation length found as sarge _extract finds it (5 ms energy, %.0f ms "
                          "preroll, 5 ms fade in, 50 ms fade out; shorter sources get the 50 ms fade and zero "
                          "padding), gain to %.2f RMS of the mono mix with a %.2f peak ceiling, zero padding to the "
                          "production audio length, pretransform.encode in the production precision" % (
                              1000 * args.preroll, target_for(INTENSITY), PEAK_CEILING),
            "args": vars(args), "out": rel(out_dir)}
    if args.captions_only:
        for e in entries:
            e.update({"content_samples": 0, "content_frames": 0, "padded": False})
        meta.update({"latent_shape": None, "n_audio": 0})
        summary(entries, report, {}, meta)
        return 0
    sam = load_model()
    n_audio, frames, ds = sizes(sam, args.gen_s, args.pad_s)
    n_gen = int(round(args.gen_s * SR))
    print("model %s on %s, audio %d samples -> %d latent frames (ratio %d), window %d samples" % (
        MODEL, sam.device, n_audio, frames, ds, n_gen), flush=True)
    kept, z, secs = encode_pool(sam, entries, args, n_gen, n_audio, ds, report)
    if z.shape[-1] != frames:
        raise SystemExit("latents have %d frames, expected %d" % (z.shape[-1], frames))
    texts, uniq = text_tables(sam, kept, args.styles, args.gen_s, args.text_bs)
    meta.update({"n_audio": n_audio, "latent_frames": frames, "downsampling_ratio": ds,
                 "latent_shape": list(z.shape), "seconds_audio": round(secs, 1), "text": uniq,
                 "seconds": round(time.perf_counter() - t0, 1)})
    save_cache(out_dir, kept, z, texts, meta, report)
    summary(kept, report, uniq, meta)
    print("saved -> %s  (%.1f s)" % (out_dir, time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
