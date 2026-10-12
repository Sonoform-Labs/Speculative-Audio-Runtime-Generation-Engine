import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import html
import json
import shutil

import numpy as np

import vocab_eval as ve
from sarge.loudness import normalize_loudness, target_for
from sarge.types import AudioClip, CacheKey

ROOT = os.path.dirname(os.path.dirname(BENCH))
GATE = os.path.join(ROOT, "data", "_gate")
OUT = os.path.join(GATE, "listen", "ear_test")
KEYS = [("shoot", "pistol"), ("shoot", "rifle"), ("break", "window"), ("footstep", "wood"), ("footstep", "stone"),
        ("footstep", "gravel"), ("footstep", "metal"), ("punch", "body")]
WATER = [("footstep", "water"), ("splash", "water")]
FOLEY_AUDIO = os.path.join("D:" + os.sep, "datasets", "sarge_refs", "foleyset", "FoleySet", "audio")
COLS = [("full", "Real recording, full length (up to 3 s)"), ("real", "Real recording, cut to 0.6 s like the engine"),
        ("varied", "Real recording, randomly varied"),
        ("decoder_s", "Real recording through the small model's decoder only"),
        ("decoder_l", "Real recording through the medium model's decoder only"),
        ("anchored", "Neural, started from that recording"), ("engine", "Engine today (text only)")]
CFG = {"sampler_type": "euler", "steps": 3, "skip_first_step": True, "compile_mode": "default", "compile_decode": True,
       "graph_decode": True, "hoist_rope": True, "fp16_accumulation": True, "graph_text": True, "unfuse_bias": True,
       "triton_gemm": True, "decode_prefix": True, "fast_load": True}


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-gen", action="store_true")
    return ap.parse_args(argv)


def norm(x):
    return normalize_loudness(np.asarray(x, dtype=np.float32), target_rms=target_for("normal")).astype(np.float32)


def shelf(x, sr, hz, db, high):
    from scipy.signal import butter, sosfilt
    sos = butter(1, hz, btype="highpass" if high else "lowpass", fs=sr, output="sos")
    band = sosfilt(sos, x)
    return x + (10.0 ** (db / 20.0) - 1.0) * band


def vary(x, sr, rng):
    from scipy.signal import resample_poly
    ratio = 2.0 ** (rng.uniform(-1.0, 1.0) / 12.0)
    up = 1000
    down = int(round(up * ratio))
    y = resample_poly(x.astype(np.float64), up, down)
    y = shelf(y, sr, 300.0, rng.uniform(-3.0, 3.0), False)
    y = shelf(y, sr, 4000.0, rng.uniform(-3.0, 3.0), True)
    n = len(x)
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)


def hf_share(x, sr=44100, hz=5000.0):
    x = np.asarray(x, dtype=np.float64)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(p[f >= hz].sum() / max(p.sum(), 1e-20))


HF = {}


def save(path, x):
    y = norm(x)
    HF[os.path.basename(path)] = hf_share(y)
    AudioClip(y, ve.SR).to_wav(path)


def name(et, mat, v):
    return "%s-%s_%02d.wav" % (et, mat, v)


def main(argv=None):
    args = parse(argv)
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    rng = np.random.default_rng(7)
    rows = []
    anchors = json.load(open(os.path.join(GATE, "anchor", "raw", "route.json"), encoding="utf-8"))["anchors"]
    for et, mat in KEYS:
        cells = {c: [] for c, _ in COLS}
        for v in range(args.seeds):
            aid = anchors.get("%s|%s" % (et, mat), [None] * (v + 1))[v]
            if aid and aid.startswith("fs_") and os.path.isfile(os.path.join(FOLEY_AUDIO, aid[3:] + ".wav")):
                full = AudioClip.from_wav(os.path.join(FOLEY_AUDIO, aid[3:] + ".wav")).samples
                start = max(0, int(np.argmax(np.abs(full))) - int(0.03 * ve.SR))
                seg = np.asarray(full[start:start + 3 * ve.SR], dtype=np.float32).copy()
                fo = min(int(0.05 * ve.SR), len(seg) // 2)
                seg[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
                fn = "full_%s" % name(et, mat, v)
                save(os.path.join(OUT, fn), seg)
                cells["full"].append(fn)
            src = {"real": os.path.join(GATE, "anchor", "raw", name(et, mat, v)),
                   "decoder_s": os.path.join(GATE, "ae_roundtrip", "same_s", name(et, mat, v)),
                   "decoder_l": os.path.join(GATE, "ae_roundtrip", "same_l", name(et, mat, v)),
                   "anchored": os.path.join(GATE, "anchor", "anc_n0.5", name(et, mat, v)),
                   "engine": os.path.join(GATE, "speed_routes", "deflt214_pre", name(et, mat, v))}
            if not all(os.path.isfile(p) for p in src.values()):
                continue
            real = ve.wav(src["real"])
            for col, x in (("real", real), ("varied", vary(real, ve.SR, rng)), ("decoder_s", ve.wav(src["decoder_s"])),
                           ("decoder_l", ve.wav(src["decoder_l"])), ("anchored", ve.wav(src["anchored"])),
                           ("engine", ve.wav(src["engine"]))):
                fn = "%s_%s" % (col, name(et, mat, v))
                save(os.path.join(OUT, fn), x)
                cells[col].append(fn)
        rows.append({"key": "%s %s" % (et, mat), "cells": cells})
    water_refs = sorted(os.listdir(os.path.join(ROOT, "data", "refs_heldout", "fs_splash", "0.6")))[:args.seeds]
    gen = None
    if not args.no_gen:
        from sarge.generators.stable_audio3 import StableAudio3Generator
        gen = StableAudio3Generator(**CFG)
        gen.ensure_ready()
    for et, mat in WATER:
        cells = {c: [] for c, _ in COLS}
        for v, ref in enumerate(water_refs):
            real = ve.wav(os.path.join(ROOT, "data", "refs_heldout", "fs_splash", "0.6", ref))
            for col, x in (("real", real), ("varied", vary(real, ve.SR, rng))):
                fn = "%s_%s" % (col, name(et, mat, v))
                save(os.path.join(OUT, fn), x)
                cells[col].append(fn)
            if gen is not None:
                c = gen.generate(CacheKey(et, mat, "normal"), variant=v)
                fn = "engine_%s" % name(et, mat, v)
                save(os.path.join(OUT, fn), c.samples)
                cells["engine"].append(fn)
        rows.append({"key": "%s %s" % (et, mat), "cells": cells,
                     "note": "real column is a FoleySet splash (no real water footsteps on disk); no anchored version"})
    head = "".join("<th>%s</th>" % html.escape(t) for _, t in COLS)
    body = []
    for r in rows:
        tds = []
        for c, _ in COLS:
            files = r["cells"][c]
            if not files:
                tds.append("<td class='na'>none on disk</td>")
                continue
            share = float(np.median([HF[f] for f in files])) * 100.0
            tds.append("<td>" + "".join("<audio controls preload='none' src='%s'></audio>" % html.escape(f)
                                        for f in files) + "<div class='note'>energy above 5 kHz: %.1f%%</div></td>" % share)
        note = "<div class='note'>%s</div>" % html.escape(r["note"]) if r.get("note") else ""
        body.append("<tr><th class='key'>%s%s</th>%s</tr>" % (html.escape(r["key"]), note, "".join(tds)))
    page = """<!doctype html><html><head><meta charset='utf-8'><title>SARGE Ear Test</title>
<style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mute:#6b6b66;--line:#ddddd8}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ececea;--mute:#9a9a95;--line:#33332f}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:24px}
table{border-collapse:collapse;width:100%%}th,td{border-bottom:1px solid var(--line);padding:10px;vertical-align:top;text-align:left}
td audio{display:block;width:210px;height:32px;margin:4px 0}.key{white-space:nowrap}.note,.na{color:var(--mute);font-size:13px}
p{max-width:780px;color:var(--mute)}
</style></head><body><h1>SARGE ear test</h1>
<p>Each row is one sound. Columns, left to right: a real recording; the same recording with random pitch (up to one semitone) and EQ tilt
(up to 3 dB), the way games vary samples; the neural model started from that recording at noise 0.5 (the anchor route); and the
engine today from the text prompt alone. The two decoder columns pass the real recording through the autoencoder and nothing
else, so whatever detail they lose, no generator built on that decoder can produce. Every clip is 0.6 s at the same loudness;
the number under each cell is the median share of energy above 5 kHz, where glass tinkle, gravel grain and water detail live.</p>
<table><tr><th>Sound</th>%s</tr>%s</table></body></html>""" % (head, "".join(body))
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    json.dump({"rows": rows, "hf_share": HF}, open(os.path.join(OUT, "rows.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "index.html"), "rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
