import os
import sys

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import csv
import glob
import hashlib
import json
import re
import shutil
import time

import numpy as np

from sarge import ground as gd
from sarge import writer as wr

ROOT = os.path.dirname(os.path.dirname(BENCH))
SRC = r"G:\sarge\captions"
OUT = os.path.join(SRC, "index")
EMBEDDER = os.path.join(SRC, "embedder", "bge-small-en-v1.5")
FOLEYSET = r"D:\datasets\sarge_refs\foleyset\FoleySet"
REF_MANIFESTS = (os.path.join(ROOT, "data", "refs_heldout", "manifest.json"),
                 os.path.join(ROOT, "data", "refs", "manifest.json"))
ESC_DIRS = (os.path.join(ROOT, "data", "esc50"), os.path.join(ROOT, "data", "real_footsteps"))
CORPORA = {
    "clotho": {"name": "Clotho v2.1 captions (Zenodo 4783391)", "licence": "Tampere University licence: "
               "non-commercial research with attribution (captions); audio under Freesound CC licences",
               "use": "research only"},
    "audiocaps": {"name": "AudioCaps captions (github cdjkim/audiocaps, dataset/)", "licence": "MIT licence file; "
                  "README: free to use for academic purposes only", "use": "research only"},
    "wc_fsd": {"name": "WavCaps FreeSound captions (hf cvssp/WavCaps json_files/FreeSound)", "licence":
               "dataset card cc-by-4.0, with the card's rule: only academic uses; source clips under Freesound "
               "licences", "use": "research only"},
    "wc_bbc": {"name": "WavCaps BBC Sound Effects captions (hf cvssp/WavCaps)", "licence": "dataset card cc-by-4.0, "
               "academic use only; descriptions from BBC Sound Effects (RemArc licence: personal, educational, "
               "research)", "use": "research only"},
    "wc_sb": {"name": "WavCaps SoundBible captions (hf cvssp/WavCaps)", "licence": "dataset card cc-by-4.0, academic "
              "use only; SoundBible clips under their own licences", "use": "research only"},
    "foleyset": {"name": "FoleySet uploader descriptions (Zenodo 20735877, on disk)", "licence": "FoleySet CC BY 4.0; "
                 "the Freesound clips are CC0", "use": "open"},
}
SKIPPED = {"wc_as": "WavCaps AudioSet_SL: machine captions written from AudioSet labels of 10 s clips, mostly several "
                    "sources at once; left out",
           "esc50": "ESC-50: class labels only, no captions; its clips feed the judges and every one of them is "
                    "excluded by Freesound id",
           "hear": "HEAR gunshot triangulation: no captions; judge references",
           "c3gd": "C3GD: no captions; judge references",
           "sonniss": "Sonniss GDC tracklist on D:\\datasets\\sonniss: file names under a royalty-free audio licence "
                      "that says nothing about the text; left out"}
URLISH = re.compile(r"(https?:|www\.|\.com\b|\.wav\b|\.mp3\b|\.flac\b|@|#|\bfreesound\b|\bzoom\b|\bmic\b)", re.I)


def norm(text):
    return " ".join(str(text or "").replace("\u00a0", " ").split()).strip()


def usable(text):
    t = norm(text)
    if not t or any(ord(ch) > 126 or ord(ch) < 32 for ch in t):
        return None
    if URLISH.search(t) or re.search(r"[0-9]", t) or re.search(r"[\[\]{}<>|_=*~^\\]", t):
        return None
    ws = re.findall(r"[A-Za-z]+", t)
    if not 3 <= len(ws) <= 25:
        return None
    letters = [ch for ch in t if ch.isalpha()]
    if sum(ch.isupper() for ch in letters) > 0.5 * len(letters):
        return None
    return t


def read_csv(path):
    with open(path, encoding="utf-8", errors="surrogateescape", newline="") as fh:
        return list(csv.DictReader(fh))


def raw_rows(src):
    out = []
    meta = {}
    for split in ("development", "validation", "evaluation"):
        for r in read_csv(os.path.join(src, "clotho", "clotho_metadata_%s.csv" % split)):
            meta[(split, r["file_name"])] = r
        for r in read_csv(os.path.join(src, "clotho", "clotho_captions_%s.csv" % split)):
            m = meta.get((split, r["file_name"]), {})
            for j in range(1, 6):
                out.append({"corpus": "clotho", "text": r.get("caption_%d" % j, ""), "src": "fs:%s" % m.get("sound_id", ""),
                            "user": m.get("manufacturer", ""), "ref": "%s/%s#%d" % (split, r["file_name"], j)})
    for split in ("train", "val", "test"):
        for r in read_csv(os.path.join(src, "audiocaps", "%s.csv" % split)):
            out.append({"corpus": "audiocaps", "text": r["caption"], "src": "yt:%s" % r["youtube_id"], "user": "",
                        "ref": "%s/%s" % (split, r["audiocap_id"])})
    for name, corpus, prefix in (("fsd_final.json", "wc_fsd", "fs"), ("bbc_final.json", "wc_bbc", "bbc"),
                                 ("sb_final.json", "wc_sb", "sb")):
        with open(os.path.join(src, "wavcaps", name), encoding="utf-8") as fh:
            data = json.load(fh)["data"]
        for r in data:
            out.append({"corpus": corpus, "text": r.get("caption", ""), "src": "%s:%s" % (prefix, r.get("id", "")),
                        "user": r.get("author", "") if corpus == "wc_fsd" else "", "ref": str(r.get("id", "")),
                        "desc": r.get("description", "")})
    for r in read_csv(os.path.join(FOLEYSET, "metadata", "foleyset_metadata.csv")):
        out.append({"corpus": "foleyset", "text": r.get("description", ""), "src": "fs:%s" % r.get("freesound-id", ""),
                    "user": r.get("username", ""), "ref": r.get("name", "")})
    return out


def esc_id(name):
    parts = os.path.basename(name).split("-")
    return parts[1] if len(parts) >= 4 and parts[1].isdigit() else None


def pcm_md5(path):
    import struct
    h = hashlib.md5()
    with open(path, "rb") as fh:
        head = fh.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            h.update(head + fh.read())
            return h.hexdigest()
        while True:
            ch = fh.read(8)
            if len(ch) < 8:
                break
            name, size = ch[:4], struct.unpack("<I", ch[4:])[0]
            body = fh.read(size + (size & 1))
            if name in (b"fmt ", b"data"):
                h.update(name + body[:size])
    return h.hexdigest()


def cmd_exclusions(a):
    t0 = time.perf_counter()
    ids, why, users, refs = set(), {}, set(), set()

    def add(fid, reason, ref=True):
        if fid and str(fid).isdigit():
            k = "fs:%s" % fid
            ids.add(k)
            why.setdefault(k, reason)
            if ref:
                refs.add(k)

    fs_meta = {r["name"]: r for r in read_csv(os.path.join(FOLEYSET, "metadata", "foleyset_metadata.csv"))}
    ref_files = []
    for p in REF_MANIFESTS:
        with open(p, encoding="utf-8") as fh:
            m = json.load(fh)
        tag = os.path.basename(os.path.dirname(p))
        for c in m["clips"]:
            if c["dataset"] == "foleyset":
                add(c.get("freesound_id"), "%s reference clip %s" % (tag, c["id"]))
                if c.get("username"):
                    users.add(c["username"])
                ref_files.append(c["src"])
            elif c["dataset"] == "esc50":
                add(c.get("esc_src") or esc_id(c["src"]), "%s reference clip %s" % (tag, c["id"]))
        for c in m.get("excluded") or []:
            add(c.get("freesound_id"), "%s excluded list %s" % (tag, c.get("id")), False)
        for c in (m.get("audio_duplicates") or {}).get("dropped") or []:
            add(c.get("freesound_id"), "%s audio duplicate %s" % (tag, c.get("id")))
    n_esc = 0
    for d in ESC_DIRS:
        for f in glob.glob(os.path.join(d, "**", "*.wav"), recursive=True):
            fid = esc_id(f)
            if fid:
                n_esc += 1
                add(fid, "ESC-50 file %s" % os.path.relpath(f, ROOT))
    with open(os.path.join(a.src, "wavcaps", "blacklist_exclude_ub8k_esc50_vggsound.json"), encoding="utf-8") as fh:
        bl = json.load(fh)
    for fid in bl.get("FreeSound", []):
        add(str(fid), "WavCaps blacklist (overlap with UrbanSound8K, ESC-50, VGGSound)", False)
    ref_hash = {}
    for f in sorted(set(ref_files)):
        p = f if os.path.isfile(f) else os.path.join(FOLEYSET, "audio", os.path.basename(f))
        ref_hash.setdefault(pcm_md5(p), []).append(os.path.basename(p))
    dup = []
    for f in sorted(glob.glob(os.path.join(FOLEYSET, "audio", "*.wav"))):
        h = pcm_md5(f)
        if h in ref_hash and os.path.basename(f) not in ref_hash[h]:
            r = fs_meta.get(os.path.basename(f), {})
            add(r.get("freesound-id"), "FoleySet file %s has the PCM of reference %s" % (
                os.path.basename(f), ",".join(ref_hash[h])))
            if r.get("username"):
                users.add(r["username"])
            dup.append([os.path.basename(f), ref_hash[h]])
    rows = raw_rows(a.src)
    by_src = {}
    for r in rows:
        by_src.setdefault(r["src"], []).append(r)
    esc_users = set()
    for k in sorted(refs):
        for r in by_src.get(k, []):
            if r.get("user"):
                esc_users.add(r["user"])
    texts = set()
    for k in ids:
        for r in by_src.get(k, []):
            for t in (r.get("text"), r.get("desc")):
                if norm(t):
                    texts.add(gd.caption_key(t))
    for c in fs_meta.values():
        if "fs:%s" % c.get("freesound-id") in ids:
            texts.add(gd.caption_key(c.get("description")))
    texts.discard("")
    out = {"ids": sorted(ids), "ref_ids": sorted(refs), "why": why, "users": sorted(users | esc_users),
           "ref_users": sorted(users), "id_users": sorted(esc_users), "texts": sorted(texts), "pcm_duplicates": dup,
           "esc_files": n_esc,
           "foleyset_ref_files": len(set(ref_files)), "seconds": time.perf_counter() - t0,
           "sources": {"manifests": [os.path.relpath(p, ROOT) for p in REF_MANIFESTS],
                       "esc_dirs": [os.path.relpath(d, ROOT) for d in ESC_DIRS]}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("exclusions: %d freesound ids, %d uploaders, %d texts, %d PCM duplicates, %.1f s -> %s" % (
        len(ids), len(out["users"]), len(texts), len(dup), out["seconds"], a.out), flush=True)
    return 0


def real_src(src):
    return src.startswith("fs:") and src[3:].isdigit()


def user_srcs(rows, users):
    low = {u.lower() for u in users}
    return {r["src"] for r in rows if real_src(r["src"]) and r.get("user") and r["user"].lower() in low}


def select(rows, ex):
    ids, texts = set(ex["ids"]), set(ex["texts"])
    users = {u.lower() for u in ex["users"]}
    bad = user_srcs(rows, ex["users"])
    stats = {c: {"raw": 0, "kept": 0, "unusable": 0, "excluded_id": 0, "excluded_user": 0, "excluded_text": 0,
                 "duplicate": 0} for c in CORPORA}
    seen = {}
    kept = []
    for r in rows:
        s = stats[r["corpus"]]
        s["raw"] += 1
        t = usable(r["text"])
        if t is None:
            s["unusable"] += 1
            continue
        if r["src"] in ids or (r["src"].startswith("fs:") and not real_src(r["src"])):
            s["excluded_id"] += 1
            continue
        if r["src"] in bad or (r.get("user") and r["user"].lower() in users):
            s["excluded_user"] += 1
            continue
        key = gd.caption_key(t)
        if key in texts:
            s["excluded_text"] += 1
            continue
        if key in seen:
            s["duplicate"] += 1
            continue
        seen[key] = len(kept)
        s["kept"] += 1
        kept.append({"text": t, "corpus": r["corpus"], "src": r["src"]})
    return kept, stats


def cmd_build(a):
    t0 = time.perf_counter()
    with open(a.exclusions, encoding="utf-8") as fh:
        ex = json.load(fh)
    rows = raw_rows(a.src)
    kept, stats = select(rows, ex)
    print("captions: %d raw, %d kept" % (len(rows), len(kept)), flush=True)
    for c, s in stats.items():
        print("  %-10s %s" % (c, s), flush=True)
    if a.dry:
        return 0
    os.makedirs(a.out, exist_ok=True)
    emb_dir = os.path.join(a.out, gd.EMBEDDER_DIR)
    if os.path.isdir(emb_dir):
        shutil.rmtree(emb_dir)
    shutil.copytree(a.embedder, emb_dir)
    e = gd.Embedder(emb_dir, a.threads)
    order = sorted(range(len(kept)), key=lambda i: (len(kept[i]["text"]), i))
    emb = np.zeros((len(kept), 384), dtype=np.float16)
    t1 = time.perf_counter()
    for s in range(0, len(order), a.batch):
        idx = order[s:s + a.batch]
        emb[idx] = e.encode([kept[i]["text"] for i in idx]).astype(np.float16)
        if (s // a.batch) % 200 == 0:
            print("  embedded %d of %d, %.0f s" % (s + len(idx), len(kept), time.perf_counter() - t1), flush=True)
    codes = {c: i for i, c in enumerate(CORPORA)}
    caps = {"text": [k["text"] for k in kept], "corpus": [k["corpus"] for k in kept],
            "src": [k["src"] for k in kept], "codes": codes}
    cp = os.path.join(a.out, gd.FILES["captions"])
    with open(cp, "w", encoding="utf-8") as fh:
        json.dump(caps, fh, ensure_ascii=True, separators=(",", ":"))
    ep = os.path.join(a.out, gd.FILES["emb"])
    np.save(ep, emb)
    xp = os.path.join(a.out, "exclusions.json")
    if os.path.abspath(xp) != os.path.abspath(a.exclusions):
        shutil.copyfile(a.exclusions, xp)
    digest = hashlib.sha256((gd.sha256_file(cp) + gd.sha256_file(ep)).encode()).hexdigest()
    meta = {"version": gd.VERSION, "digest": digest, "count": len(kept), "dims": int(emb.shape[1]),
            "dtype": "float16", "embedder": e.ident(), "corpora": {c: dict(CORPORA[c], **stats[c]) for c in CORPORA},
            "skipped": SKIPPED, "exclusions": {"file": "exclusions.json", "sha256": gd.sha256_file(xp),
                                               "ids": len(ex["ids"]), "users": len(ex["users"]),
                                               "texts": len(ex["texts"])},
            "filters": {"ascii": True, "words": [3, 25], "digits": False, "url_or_gear_words": URLISH.pattern,
                        "dedupe": "lower-case alphanumeric key, first kept in corpus order"},
            "build_s": time.perf_counter() - t0, "embed_s": time.perf_counter() - t1}
    with open(os.path.join(a.out, gd.FILES["meta"]), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    print("index %s: %d captions, digest %s, %.0f s" % (a.out, len(kept), digest[:16], meta["build_s"]), flush=True)
    return 0


def cmd_overlap(a):
    idx = gd.CaptionIndex(a.index)
    with open(os.path.join(a.index, "exclusions.json"), encoding="utf-8") as fh:
        ex = json.load(fh)
    caps = json.load(open(os.path.join(a.index, gd.FILES["captions"]), encoding="utf-8"))
    ids, users = set(ex["ids"]), set(ex["users"])
    texts = set(ex["texts"])
    src_hits = [i for i, s in enumerate(caps["src"]) if s in ids]
    text_hits = [i for i, t in enumerate(caps["text"]) if gd.caption_key(t) in texts]
    rows = raw_rows(SRC)
    bad = user_srcs(rows, users)
    user_hits = [i for i, s in enumerate(caps["src"]) if s in bad]
    e = gd.Embedder(os.path.join(a.index, gd.EMBEDDER_DIR), a.threads)
    refs = set(ex.get("ref_ids") or [])
    ref_only = set()
    by_src = {}
    for r in rows:
        by_src.setdefault(r["src"], []).append(r)
    for k in refs:
        for r in by_src.get(k, []):
            for t in (r.get("text"), r.get("desc")):
                if norm(t):
                    ref_only.add(gd.caption_key(t))
    ref_only &= texts
    ref_texts = sorted(texts)
    near = []
    best = []
    for s in range(0, len(ref_texts), 256):
        q = e.encode(ref_texts[s:s + 256])
        sc = np.zeros((q.shape[0], idx.emb.shape[0]), dtype=np.float32)
        for c in range(0, idx.emb.shape[0], gd.CHUNK):
            sc[:, c:c + gd.CHUNK] = q @ idx.emb[c:c + gd.CHUNK].astype(np.float32).T
        j = sc.argmax(axis=1)
        for r, (jj, v) in enumerate(zip(j, sc[np.arange(sc.shape[0]), j])):
            best.append(float(v))
            if v >= a.near:
                near.append([ref_texts[s + r], idx.text[int(jj)], round(float(v), 4), idx.corpus[int(jj)],
                             ref_texts[s + r] in ref_only])
    best = np.array(best) if best else np.zeros(1)
    out = {"index": a.index, "count": len(caps["text"]), "excluded_ids": len(ids), "excluded_users": len(users),
           "excluded_texts": len(texts), "index_rows_with_excluded_id": len(src_hits),
           "index_rows_with_excluded_text": len(text_hits), "index_rows_from_excluded_uploader": len(user_hits),
           "ref_text_nearest_cos": {"p50": float(np.median(best)), "p95": float(np.percentile(best, 95)),
                                    "max": float(best.max())},
           "near_threshold": a.near, "reference_clip_texts": len(ref_only),
           "reference_clip_texts_near": sum(1 for n in near if n[4]), "near": sorted(near, key=lambda x: -x[2])}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("overlap: %d rows, excluded id %d, excluded text %d, excluded uploader %d; %d of %d excluded texts have "
          "an index caption at cos >= %.2f (nearest p50 %.3f, max %.3f); texts of the reference clips themselves: "
          "%d of %d" % (
              out["count"], len(src_hits), len(text_hits), len(user_hits), len(near), len(ref_texts), a.near,
              out["ref_text_nearest_cos"]["p50"], out["ref_text_nearest_cos"]["max"],
              out["reference_clip_texts_near"], len(ref_only)), flush=True)
    for n in [x for x in out["near"] if x[4]][:a.show]:
        print("   %.3f  reference clip text: %s  ||  index (%s): %s" % (n[2], n[0], n[3], n[1]))
    return 0 if not (src_hits or text_hits or user_hits) else 1


def cmd_bench(a):
    t0 = time.perf_counter()
    g = gd.Grounder("caption", a.index, threads=a.threads)
    g.start()
    load_s = time.perf_counter() - t0
    texts = ["event: car horn", "event: footstep\nmaterial: gravel", "event: shoot\nmaterial: pistol",
             "event: impact\nmade by: physics\nobject: crate wooden\ntouches: floor stone\naction: hit\nroom: small",
             "event: splash\nmaterial: water"]
    g.query(texts[0])
    emb_ms, ret_ms, rank_ms = [], [], []
    for _ in range(a.reps):
        for t in texts:
            req, _full = gd.query_texts(t)
            s = time.perf_counter()
            g.embedder.one(req)
            emb_ms.append((time.perf_counter() - s) * 1000)
            q = g.query(t)
            s = time.perf_counter()
            g.index.top(q, gd.POOL)
            ret_ms.append((time.perf_counter() - s) * 1000)
            s = time.perf_counter()
            g.ranked(None, t)
            rank_ms.append((time.perf_counter() - s) * 1000)
    rss = None
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 1e6
    except ImportError:
        pass
    out = {"threads": a.threads, "load_s": load_s, "captions": len(g.index.text),
           "embed_one_ms": {"p50": float(np.median(emb_ms)), "p95": float(np.percentile(emb_ms, 95))},
           "top_pool_ms": {"p50": float(np.median(ret_ms)), "p95": float(np.percentile(ret_ms, 95))},
           "ranked_total_ms": {"p50": float(np.median(rank_ms)), "p95": float(np.percentile(rank_ms, 95))},
           "rss_mb": rss, "index_bytes": {f: os.path.getsize(os.path.join(a.index, f)) for f in gd.FILES.values()},
           "embedder_bytes": sum(os.path.getsize(p) for p in glob.glob(os.path.join(a.index, gd.EMBEDDER_DIR, "**", "*"),
                                                                    recursive=True) if os.path.isfile(p))}
    print(json.dumps(out, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="caption index for the grounded prompt writer: exclusions of every "
                                             "caption tied to a judge reference, build, overlap check, CPU bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("exclusions")
    p.add_argument("--src", default=SRC)
    p.add_argument("--out", default=os.path.join(OUT, "exclusions.json"))
    p = sub.add_parser("build")
    p.add_argument("--src", default=SRC)
    p.add_argument("--out", default=OUT)
    p.add_argument("--exclusions", default=os.path.join(OUT, "exclusions.json"))
    p.add_argument("--embedder", default=EMBEDDER)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--dry", action="store_true", help="count the captions that pass, embed nothing")
    p = sub.add_parser("overlap")
    p.add_argument("--index", default=OUT)
    p.add_argument("--out", required=True)
    p.add_argument("--near", type=float, default=0.95)
    p.add_argument("--show", type=int, default=12)
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("bench")
    p.add_argument("--index", default=OUT)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return {"exclusions": cmd_exclusions, "build": cmd_build, "overlap": cmd_overlap, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
