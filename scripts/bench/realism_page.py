import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import vocab_eval as ve

import argparse
import hashlib
import html
import json
import time

import numpy as np

import realism_feats as rf
import realism_refs as rr

ROOT = ve.ROOT
SR = rr.SR
TARGET_LUFS = -23.0
PEAK_MAX = 0.98
REAL_PER_KEY = 3
REPEATS = 12
PAIRS_PER_FAMILY = 10
LABELS = {"shoot|pistol": "pistol shot", "shoot|rifle": "rifle shot", "break|window": "window breaking",
          "bullet_impact|glass": "bullet hitting glass", "footstep|gravel": "footstep on gravel",
          "footstep|stone": "footstep on stone", "footstep|wood": "footstep on wood", "footstep|metal": "footstep on metal",
          "footstep|water": "footstep in water", "splash|water": "water splash", "punch|body": "punch to the body",
          "reload|rifle": "rifle reload"}


def seed_of(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def wav_write(p, x):
    from scipy.io import wavfile
    os.makedirs(os.path.dirname(p), exist_ok=True)
    wavfile.write(p, SR, np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16))


def loudness_match(clips):
    gains = []
    for x in clips:
        lu = rf.integrated_lufs(x)
        gains.append(10 ** ((TARGET_LUFS - lu) / 20.0) if np.isfinite(lu) and lu > -70 else 1.0)
    peaks = [float(np.max(np.abs(x))) * g if len(x) else 0.0 for x, g in zip(clips, gains)]
    top = max(peaks) if peaks else 1.0
    trim = min(1.0, PEAK_MAX / top) if top > 0 else 1.0
    return [x * g * trim for x, g in zip(clips, gains)], TARGET_LUFS + 20 * np.log10(trim)


def real_clips(key):
    pool = rr.KEYS[key]["pool"]
    if pool is None:
        return []
    spec = rr.POOLS[pool]
    if spec["src"] == "heldout":
        man = rr.man_load(rr.HELDOUT)
        _m, sets, meta = ve.load_refs(rr.ns_refs(rr.HELDOUT), {})
        base = rr.heldout_specs(pool, man, sets, meta)
        w1 = rr.w1_real_specs(pool, base)
    elif spec["src"] == "cache_v1":
        base = rr.footstep_specs(pool)
        w1 = []
        idx = rr.eval_index()
        ents = {e["id"]: e for e in idx["eval"]}
        for s in base:
            e = ents[s.meta["entry"]]
            a, b = s.meta["seg"]

            def make(e=e, a=a, b=b):
                return rr.chain(rr.crop_audio(e, a, b), SR, 1.0)
            w1.append(rr.Spec("w1:" + s.uid, "real", None, pool, s.cluster, "1.0", False, s.fam, s.wet, dict(s.meta),
                              make))
    else:
        base, _man = rr.refs_file_specs(pool)
        w1 = rr.w1_real_specs(pool, base)
    pick = rr.spaced_by_source(w1, REAL_PER_KEY, cluster=lambda s: s.cluster)
    return [(s.uid, s.cluster, s.make()["x5"]) for s in pick]


def css():
    return """:root{--bg:#fafaf8;--fg:#1c1c1a;--mute:#6b6b66;--line:#ddddd8;--acc:#2f6fdb;--card:#ffffff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ececea;--mute:#9a9a95;
--line:#33332f;--acc:#7aa7ff;--card:#1f1f1d}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ececea;--mute:#9a9a95;--line:#33332f;--acc:#7aa7ff;--card:#1f1f1d}
*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px;
max-width:1100px;margin-inline:auto}h1{font-size:22px;margin:0 0 6px}h2{font-size:17px;margin:28px 0 8px;
border-bottom:1px solid var(--line);padding-bottom:4px}.m{color:var(--mute);font-size:13px}
.clip{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;margin:8px 0;display:grid;
grid-template-columns:minmax(0,220px) 1fr 1fr;gap:12px;align-items:center}audio{width:100%;height:32px}
label{display:block;font-size:13px;color:var(--mute)}input[type=range]{width:100%}.v{font-variant-numeric:tabular-nums;
color:var(--fg);font-weight:600}.unset .v{color:var(--mute);font-weight:400}button{background:var(--acc);color:#fff;
border:0;border-radius:6px;padding:9px 16px;font-size:15px;cursor:pointer}.bar{position:sticky;top:0;background:var(--bg);
padding:8px 0;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2}
@media (max-width:640px){.clip{grid-template-columns:1fr}}"""


def build_phase2(args):
    bd = args.bakeoff if os.path.isabs(args.bakeoff) else os.path.join(ROOT, args.bakeoff)
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    models = args.models or sorted(d for d in os.listdir(os.path.join(bd, "raw"))
                                   if os.path.isfile(os.path.join(bd, "raw", d, "gen.json")) and
                                   os.path.isdir(os.path.join(bd, "routes", d)))
    keys = args.keys or list(rr.KEYS)
    slots = [int(s) for s in args.slots]
    items = []
    for m in models:
        raw = os.path.join(bd, "raw", m)
        for key in keys:
            for v in slots:
                sp = rr.raw_specs(m, raw, max(slots) + 1, [key])
                sp = [s for s in sp if s.meta["seed"] == v]
                if not sp:
                    continue
                items.append({"kind": "gen", "key": key, "model": m, "seed": v, "x": sp[0].make()["x5"]})
    for key in keys:
        for uid, cl, x in real_clips(key):
            items.append({"kind": "real", "key": key, "model": "real", "seed": None, "uid": uid, "cluster": cl, "x": x})
    rng = np.random.default_rng(seed_of("page|" + args.out))
    rep = rng.choice(len(items), min(REPEATS, len(items)), replace=False)
    for j in rep:
        it = dict(items[j])
        it["repeat_of"] = int(j)
        items.append(it)
    xs, level = loudness_match([it["x"] for it in items])
    order = rng.permutation(len(items))
    ids = {}
    for n, j in enumerate(order):
        ids[int(j)] = "c%03d" % (n + 1)
    key_rows = []
    for j, (it, x) in enumerate(zip(items, xs)):
        cid = ids[j]
        wav_write(os.path.join(out, "a", cid + ".wav"), x)
        key_rows.append({"id": cid, "kind": it["kind"], "key": it["key"], "model": it["model"], "seed": it["seed"],
                         "uid": it.get("uid"), "repeat_of": ids[it["repeat_of"]] if "repeat_of" in it else None})
    by_key = {}
    for r in key_rows:
        by_key.setdefault(r["key"], []).append(r["id"])
    key_order = list(rng.permutation(sorted(by_key)))
    secs = []
    for k in key_order:
        cs = sorted(by_key[k], key=lambda c: seed_of(c + args.out))
        cells = "".join(
            "<div class='clip unset' data-id='%s'><div><b>%s</b><audio controls preload='none' src='a/%s.wav'></audio>"
            "</div><div><label>realism <span class='v' id='r_%s'>-</span></label><input type='range' min='0' max='100' "
            "step='1' value='50' data-id='%s' data-ax='realism'></div><div><label>quality <span class='v' id='q_%s'>-"
            "</span></label><input type='range' min='0' max='100' step='1' value='50' data-id='%s' data-ax='quality'>"
            "</div></div>" % (c, c, c, c, c, c, c) for c in cs)
        secs.append("<h2>%s</h2><p class='m'>%d clips. Realism: could this be a real recording of %s? Quality: how "
                    "clean and full does it sound, whatever its realism?</p>%s" % (
                        html.escape(LABELS.get(k, k)), len(cs), html.escape(LABELS.get(k, k)), cells))
    page = PAGE % {"css": css(), "n": len(key_rows), "secs": "".join(secs), "name": html.escape(os.path.basename(out)),
                   "lufs": "%.1f" % level}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    meta = {"page": os.path.basename(out), "built": time.strftime("%Y-%m-%d %H:%M:%S"), "models": models, "keys": keys,
            "slots": slots, "loudness_lufs": level, "repeats": REPEATS, "real_per_key": REAL_PER_KEY,
            "format": "W1.0 judge format: chain P step-5 copy, mono 44.1 kHz, BS.1770 integrated loudness matched",
            "clips": key_rows}
    with open(os.path.join(out, "page_key.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print("wrote %s (%d clips, %d gen, %d real, %d repeats, loudness %.1f LUFS)" % (
        ve.rel(os.path.join(out, "index.html")), len(key_rows), sum(r["kind"] == "gen" and not r["repeat_of"]
                                                                    for r in key_rows),
        sum(r["kind"] == "real" and not r["repeat_of"] for r in key_rows), sum(bool(r["repeat_of"]) for r in key_rows),
        level))
    return 0


PAGE = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Realism Ratings</title><style>%(css)s</style></head><body>
<h1>Realism ratings</h1>
<p class='m'>%(n)d clips, blind: model names and real recordings are hidden and shuffled; a few clips appear twice.
Every clip is 1.0 s, mono, loudness matched to %(lufs)s LUFS, so set your volume once and leave it. Headphones. Move both
sliders for every clip; a slider you have not touched stays unrated. Ratings save in this browser as you go.</p>
<div class='bar'><button id='dl'>Download ratings</button><span class='m' id='cnt'></span></div>
%(secs)s
<p class='m'>When done, download the ratings and run realism_page.py --import on the file.</p>
<script>
const PAGE = "%(name)s";
const KEY = "realism_ratings_" + PAGE;
let st = {};
try { st = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { st = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} }
function show() {
  let done = 0, total = 0;
  document.querySelectorAll(".clip").forEach(c => {
    const id = c.dataset.id, s = st[id] || {};
    total += 1;
    if (s.realism != null && s.quality != null) { done += 1; c.classList.remove("unset"); }
    document.getElementById("r_" + id).textContent = s.realism != null ? s.realism : "-";
    document.getElementById("q_" + id).textContent = s.quality != null ? s.quality : "-";
  });
  document.getElementById("cnt").textContent = done + " of " + total + " rated";
}
document.querySelectorAll("input[type=range]").forEach(r => {
  const s = st[r.dataset.id] || {};
  if (s[r.dataset.ax] != null) r.value = s[r.dataset.ax];
  r.addEventListener("input", () => {
    st[r.dataset.id] = st[r.dataset.id] || {};
    st[r.dataset.id][r.dataset.ax] = Number(r.value);
    save(); show();
  });
});
document.getElementById("dl").addEventListener("click", () => {
  const out = {page: PAGE, exported: new Date().toISOString(), ratings: Object.entries(st).map(([id, v]) =>
    ({id, realism: v.realism != null ? v.realism : null, quality: v.quality != null ? v.quality : null}))};
  const b = new Blob([JSON.stringify(out, null, 1)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = "ear_ratings_" + PAGE + ".json"; a.click();
});
show();
</script></body></html>"""


def build_ab(args):
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    route = args.route.partition("=")
    base = args.baseline.partition("=")
    fams = args.families or ["gun", "glass", "footstep", "water"]
    pick = []
    for fam in fams:
        keys = [k for k, v in rr.KEYS.items() if v["family"] == fam and v["status"] != "proxy"]
        n = 0
        rng = np.random.default_rng(seed_of("ab|" + fam + args.out))
        while n < PAIRS_PER_FAMILY:
            k = keys[n % len(keys)]
            et, mat = k.split("|")
            v = int(rng.integers(0, args.seeds))
            pa = os.path.join(route[2] if os.path.isabs(route[2]) else os.path.join(ROOT, route[2]),
                              "%s-%s_%02d.wav" % (et, mat, v))
            pb = os.path.join(base[2] if os.path.isabs(base[2]) else os.path.join(ROOT, base[2]),
                              "%s-%s_%02d.wav" % (et, mat, v))
            n += 1
            if os.path.isfile(pa) and os.path.isfile(pb):
                pick.append({"family": fam, "key": k, "seed": v, "route": pa, "base": pb,
                             "route_left": bool(rng.integers(0, 2))})
    xs = []
    for p in pick:
        for side in ("route", "base"):
            x, sr = rr.read_any(p[side])
            xs.append(rr.resample(x, sr))
    xs, level = loudness_match(xs)
    rows, secs = [], []
    for i, p in enumerate(pick):
        xr, xb = xs[2 * i], xs[2 * i + 1]
        left, right = (xr, xb) if p["route_left"] else (xb, xr)
        pid = "p%02d" % (i + 1)
        wav_write(os.path.join(out, "a", pid + "_A.wav"), left)
        wav_write(os.path.join(out, "a", pid + "_B.wav"), right)
        rows.append({"id": pid, "family": p["family"], "key": p["key"], "seed": p["seed"],
                     "A": route[0] if p["route_left"] else base[0], "B": base[0] if p["route_left"] else route[0]})
        secs.append("<div class='clip' data-id='%s'><div><b>%s</b> %s</div><div><audio controls preload='none' "
                    "src='a/%s_A.wav'></audio><label><input type='radio' name='%s' value='A'> A is more real</label>"
                    "</div><div><audio controls preload='none' src='a/%s_B.wav'></audio><label><input type='radio' "
                    "name='%s' value='B'> B is more real</label><label><input type='radio' name='%s' value='same'> no "
                    "difference</label></div></div>" % (pid, pid, html.escape(LABELS.get(p["key"], p["key"])), pid, pid,
                                                        pid, pid, pid))
    page = AB_PAGE % {"css": css(), "n": len(rows), "secs": "".join(secs), "name": html.escape(os.path.basename(out))}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    with open(os.path.join(out, "page_key.json"), "w", encoding="utf-8") as fh:
        json.dump({"page": os.path.basename(out), "route": route[0], "baseline": base[0], "pairs": rows,
                   "loudness_lufs": level}, fh, indent=1)
    print("wrote %s (%d pairs)" % (ve.rel(os.path.join(out, "index.html")), len(rows)))
    return 0


AB_PAGE = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Blind Pairs</title><style>%(css)s</style></head><body><h1>Blind pairs</h1>
<p class='m'>%(n)d pairs, sides shuffled, loudness matched. Pick the clip that could more plausibly be a real recording.</p>
<div class='bar'><button id='dl'>Download answers</button></div>%(secs)s
<script>
const PAGE = "%(name)s";
document.getElementById("dl").addEventListener("click", () => {
  const ans = [];
  document.querySelectorAll(".clip").forEach(c => {
    const s = c.querySelector("input:checked");
    ans.push({id: c.dataset.id, choice: s ? s.value : null});
  });
  const b = new Blob([JSON.stringify({page: PAGE, answers: ans}, null, 1)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = "ab_answers_" + PAGE + ".json"; a.click();
});
</script></body></html>"""


def do_import(args):
    src = args.import_file if os.path.isabs(args.import_file) else os.path.join(ROOT, args.import_file)
    with open(src, encoding="utf-8") as fh:
        got = json.load(fh)
    pd = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    with open(os.path.join(pd, "page_key.json"), encoding="utf-8") as fh:
        key = json.load(fh)
    by = {c["id"]: c for c in key["clips"]}
    rows = []
    for r in got.get("ratings", []):
        c = by.get(r["id"])
        if c is None or r.get("realism") is None:
            continue
        rows.append({**c, "realism": r["realism"], "quality": r.get("quality")})
    dest = os.path.join(ROOT, "data", "_gate", "judges", "ear_ratings.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump({"page": key["page"], "imported": time.strftime("%Y-%m-%d %H:%M:%S"), "slots": key["slots"],
                   "ratings": rows}, fh, indent=1)
    print("wrote %s with %d ratings" % (ve.rel(dest), len(rows)))
    return 0


def parse(argv):
    ap = argparse.ArgumentParser(description="Blind rating pages for the SARGE realism judges")
    ap.add_argument("--bakeoff", default=os.path.join("data", "_gate", "bakeoff"))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--keys", nargs="+", default=None)
    ap.add_argument("--slots", nargs="+", default=["02", "05"])
    ap.add_argument("--out", default=os.path.join("data", "_gate", "judges", "page_v1"))
    ap.add_argument("--ab", action="store_true")
    ap.add_argument("--route", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--import", dest="import_file", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    if args.import_file:
        return do_import(args)
    if args.ab:
        if not args.route or not args.baseline:
            raise SystemExit("--ab needs --route NAME=DIR and --baseline NAME=DIR")
        return build_ab(args)
    return build_phase2(args)


if __name__ == "__main__":
    raise SystemExit(main())
