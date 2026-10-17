import os
import sys

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.dirname(BENCH))

import _bootstrap

import argparse
import hashlib
import html
import json
import shutil

import numpy as np
from scipy.signal import resample_poly

import woosh_server as ws
from sarge import loudness as L
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import AudioClip, CacheKey

ROOT = os.path.dirname(os.path.dirname(BENCH))
SWEEP = os.path.join(ROOT, "data", "_gate", "opt", "woosh_buckets", "sweep", "101")
GUNS = os.path.join(ROOT, "data", "_gate", "opt", "medium", "_work", "raw", "m1_prod")
OUT = os.path.join(ROOT, "data", "_gate", "feel", "loudness_page")
BALL = os.path.join(ROOT, "data", "_gate", "feel", "served_ball_metal_hard.wav")
SR = 48000
VOLUME = {"footstep": 0.45, "shoot": 1.0, "reload": 0.6, "equip": 0.6, "empty_click": 0.6, "break": 1.0,
          "bullet impact": 0.8, "splash": 0.8, "punch": 0.8}
ARMS = {"H": "HEAD chain (peak level, amp with saturation)", "L": "loudness level + amp punch",
        "LD": "loudness level + amp with saturation"}
WOOSH_KEYS = (("break-window", "break", "window"), ("bullet_impact-glass", "bullet impact", "glass"),
              ("footstep-gravel", "footstep", "gravel"), ("footstep-stone", "footstep", "stone"),
              ("footstep-wood", "footstep", "wood"), ("footstep-metal", "footstep", "metal"),
              ("footstep-water", "footstep", "water"), ("splash-water", "splash", "water"), ("punch-body", "punch", "body"))
GUN_KEYS = (("shoot-pistol", "shoot", "pistol"), ("shoot-rifle", "shoot", "rifle"), ("reload-rifle", "reload", "rifle"),
            ("equip-rifle", "equip", "rifle"), ("empty_click-pistol", "empty_click", "pistol"))


def woosh(raw, event, intensity, arm, stats=None):
    if arm == "H":
        return ws.finish(raw, event, intensity, "full", True)
    y = ws.finish(raw, event, intensity, "punch" if arm == "L" else "full", True, level="loudness")
    if stats is not None:
        a = raw[:, 0] if raw.ndim == 2 and raw.shape[1] == 1 else raw
        from sarge import post
        post.process(np.ascontiguousarray(a[:max(1, len(a) - int(ws.END_CUT_S * SR))], dtype=np.float32), SR, event,
                     intensity, level="loudness", amp_mode="punch" if arm == "L" else "full", stereo=True, stats=stats)
    return y


def guns(raw, event, material, intensity, arm, stats=None):
    level, amp = {"H": ("peak", True), "L": ("loudness", "punch"), "LD": ("loudness", True)}[arm]
    g = StableAudio3Generator(trim="natural", level=level, amp=amp, stereo=True)
    g.sample_rate = 44100
    y = g._finish(np.ascontiguousarray(raw.T), CacheKey(event, material, intensity))
    if stats is not None and arm != "H":
        from sarge import post
        post.process(np.ascontiguousarray(raw), 44100, event, intensity, level="loudness",
                     amp_mode="punch" if arm == "L" else "full", stereo=True, stats=stats)
    return resample_poly(y, 160, 147, axis=0).astype(np.float32)


def stereo(y):
    return np.stack([y, y], axis=1) if y.ndim == 1 else y


def describe(y, st=None):
    pk = 20 * np.log10(max(float(np.max(np.abs(y))), 1e-9))
    m = L.momentary_max(stereo(y), SR)
    s = "M400 %.1f LUFS, PLR %.1f dB" % (m, pk - m)
    if st and "gr_db" in st:
        s += ", limiter %.1f dB" % st["gr_db"]
    return s


def gain(d, vol, mind):
    return vol if mind is None else vol * mind / max(mind, d)


def mix(parts, length_s):
    out = np.zeros((int(length_s * SR), 2), dtype=np.float32)
    for t, y, g in parts:
        y = stereo(y) * np.float32(g)
        i = int(t * SR)
        n = min(len(y), len(out) - i)
        out[i:i + n] += y[:n]
    return out


def order_for(rid, arms):
    h = int(hashlib.md5(rid.encode()).hexdigest(), 16)
    arms = list(arms)
    out = []
    while arms:
        out.append(arms.pop(h % len(arms)))
        h //= 7
    return out


def write(rid, arm, y):
    name = hashlib.md5(("sarge-level-" + rid + arm).encode()).hexdigest()[:12] + ".wav"
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / np.float32(peak)
    AudioClip(np.ascontiguousarray(y, dtype=np.float32), SR).to_wav(os.path.join(OUT, "a", name))
    return name


def clip_rows(seeds):
    rows = []
    for stem, event, mat in WOOSH_KEYS:
        for prompt, inten in (("normal", "normal"), ("loud", "hard")):
            for seed in seeds:
                raw = np.load(os.path.join(SWEEP, "%s-%s_%02d.npy" % (stem, prompt, seed)))
                rid = "c-%s-%s-%d" % (stem, inten, seed)
                arms = {}
                for arm in ARMS:
                    st = {}
                    y = woosh(raw, event, inten, arm, st)
                    vol = VOLUME[event]
                    arms[arm] = (y * np.float32(vol), "%s; clip %s; game volume %.2f" % (ARMS[arm], describe(y, st), vol))
                rows.append((rid, "%s on %s, %s" % (event, mat, inten), "Woosh render seed %d" % seed, arms))
    for stem, event, mat in GUN_KEYS:
        for seed in seeds:
            raw = np.load(os.path.join(GUNS, "%s_%02d.npy" % (stem, seed)))
            rid = "g-%s-%d" % (stem, seed)
            arms = {}
            for arm in ARMS:
                st = {}
                y = guns(raw, event, mat, "normal", arm, st)
                vol = VOLUME[event]
                arms[arm] = (y * np.float32(vol), "%s; clip %s; game volume %.2f" % (ARMS[arm], describe(y, st), vol))
            rows.append((rid, "%s %s, normal" % (event, mat), "SA3 medium render seed %d" % seed, arms))
    return rows


def game_rows():
    rows = []
    win = np.load(os.path.join(SWEEP, "break-window-normal_00.npy"))
    glass = np.load(os.path.join(SWEEP, "bullet_impact-glass-normal_00.npy"))
    pistol = np.load(os.path.join(GUNS, "shoot-pistol_00.npy"))
    rifle = np.load(os.path.join(GUNS, "shoot-rifle_00.npy"))
    mins = (("H", None, "HEAD chain, HEAD playback (3D, min distance 1 m)"),
            ("N4", 4.0, "loudness + punch, mix v2 min distance 4 m"),
            ("N8", 8.0, "loudness + punch, mix v2 min distance 8 m"),
            ("N12", 12.0, "loudness + punch, mix v2 min distance 12 m"))

    def two(arm):
        return "H" if arm == "H" else "L"

    for d in (5.5, 9.6, 17.8):
        rid = "m-window-%g" % d
        arms = {}
        for arm, mind, lab in mins:
            g = gain(d, 1.0, 1.0 if mind is None else mind)
            w = woosh(win, "break", "normal", two(arm))
            p = guns(pistol, "shoot", "pistol", "normal", two(arm))
            arms[arm] = (mix([(0.2, p, 1.0), (0.2, w, g)], 2.8),
                         "%s; window %+.1f dB at %.1f m, pistol 2D at gun volume 1" % (lab, 20 * np.log10(g), d))
        rows.append((rid, "pistol shot breaks a window %.1f m away" % d, "both start in the same frame", arms))
    for d in (3.0, 16.0):
        rid = "m-glassfloor-%g" % d
        arms = {}
        for arm, mind, lab in mins:
            g = gain(d, 0.8, 1.0 if mind is None else mind)
            b = woosh(glass, "bullet impact", "normal", two(arm))
            r = guns(rifle, "shoot", "rifle", "normal", two(arm))
            arms[arm] = (mix([(0.2, r, 1.0), (0.2, b, g)], 1.8),
                         "%s; bullet on glass %+.1f dB at %.0f m, rifle 2D" % (lab, 20 * np.log10(g), d))
        rows.append((rid, "rifle shot hits the glass floor %.0f m away" % d, "both start in the same frame", arms))
    if os.path.isfile(BALL):
        served = AudioClip.from_wav(BALL, keep_channels=True).samples
        relevel, st = L.normalize_level(served, SR, "hard")
        arms = {}
        for arm, mind, lab in mins:
            ball = served if arm == "H" else relevel
            g = 1.0 if mind is None else gain(7.0, 1.0, mind)
            p = guns(pistol, "shoot", "pistol", "normal", two(arm))
            parts = [(0.2 + 3.9 * i, ball, g) for i in range(4)] + [(0.2 + 3.9 * 1.5, p, 1.0)]
            arms[arm] = (mix(parts, 13.6), "%s; ball %s (%s), %+.1f dB at 7 m, pistol between hits 2 and 3" % (
                lab, "served clip as heard" if arm == "H" else "re-levelled served clip, hard", describe(ball, None if arm == "H" else st),
                20 * np.log10(g)))
        rows.append(("m-ball-metal", "four ball-on-metal hits, hard, 3.9 s apart, one pistol shot",
                     "the served clip from the session; it has no raw render", arms))
    steps = [np.load(os.path.join(SWEEP, "footstep-%s-normal_%02d.npy" % (m, s))) for m, s in
             (("stone", 0), ("stone", 1), ("wood", 0), ("wood", 1), ("gravel", 0), ("gravel", 1))]
    arms = {}
    for arm in ARMS:
        parts = [(0.2 + 0.45 * i, woosh(r, "footstep", "normal", arm), VOLUME["footstep"]) for i, r in enumerate(steps)]
        arms[arm] = (mix(parts, 3.6), "%s; footsteps 0.45 s apart at footstep volume 0.45, 2D" % ARMS[arm])
    rows.append(("m-steps", "six footsteps: stone, wood, gravel", "player source, 2D", arms))
    return rows


PAGE = """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Loudness blind test</title><style>
:root{--bg:#f7f7f4;--card:#ffffff;--fg:#1d1d1b;--mute:#66665f;--line:#dcdcd5;--acc:#2a5bd7;--good:#1f7a3d;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141413;--card:#1d1d1b;--fg:#ecece8;--mute:#a0a09a;
--line:#34342f;--acc:#8fb0ff;--good:#6fd08f;--bad:#ff8a80}}
:root[data-theme=dark]{--bg:#141413;--card:#1d1d1b;--fg:#ecece8;--mute:#a0a09a;--line:#34342f;--acc:#8fb0ff;--good:#6fd08f;--bad:#ff8a80}
*{box-sizing:border-box}html,body{overflow-x:hidden}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:24px 16px 96px}
main{max-width:860px;margin:0 auto}h1{font-size:1.5rem;margin:0 0 8px}h2{font-size:1.1rem;margin:32px 0 8px}
p{color:var(--mute);margin:0 0 12px}.row{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0}
.row h3{font-size:1rem;margin:0}.row .sub{color:var(--mute);font-size:.85rem;margin:0 0 8px}
.arms{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%%,190px),1fr));gap:10px}
.arm{border:1px solid var(--line);border-radius:8px;padding:8px;min-width:0}.lab{font-weight:600}
audio{width:100%%;height:36px;display:block;margin:6px 0}
.pick{display:flex;gap:6px}button{font:inherit;padding:4px 10px;border:1px solid var(--line);background:transparent;color:var(--fg);
border-radius:6px;cursor:pointer}button.best{border-color:var(--good);color:var(--good);font-weight:600}
button.worst{border-color:var(--bad);color:var(--bad);font-weight:600}.rev{font-size:.8rem;color:var(--acc);margin-top:6px;overflow-wrap:anywhere}
.bar{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px 16px;display:flex;
gap:8px;flex-wrap:wrap;align-items:center;justify-content:center}#tally{color:var(--mute);font-size:.85rem;flex-basis:100%%;text-align:center}
</style></head><body><main><h1>Loudness blind test</h1>
<p>Every row plays the same generated sound through different finishing chains, at the level the game would play it:
the per-sound volume from the demo and, in the game rows, the distance gain of the 3D source. Letters are shuffled per row.
Mark the best and the worst version in each row. Picks stay in this browser; Reveal shows which chain is which with its
loudness numbers, and Download picks saves them as a file.</p>
<p>Chains: the current one (peak level per intensity, amplifier with saturation), loudness levelling with the attack-only
amplifier, and loudness levelling with the full amplifier. The game rows add the playback change: a pooled 3D source with a
larger minimum distance (4, 8 or 12 m) for windows, bullet hits and balls.</p>
%s</main><div class='bar'><button id='reveal'>Reveal</button><button id='dl'>Download picks</button><button id='clear'>Clear picks</button>
<div id='tally'></div></div><script>
const KEY=%s;const SK='sarge_loudness_page_v1';let store={};
try{store=JSON.parse(localStorage.getItem(SK)||'{}')}catch(e){store={}}
function save(){try{localStorage.setItem(SK,JSON.stringify(store))}catch(e){}}
function paint(){document.querySelectorAll('.row').forEach(r=>{const p=store[r.dataset.id]||{};
r.querySelectorAll('.arm').forEach(a=>{const l=a.dataset.l;a.querySelector('[data-k=best]').classList.toggle('best',p.best===l);
a.querySelector('[data-k=worst]').classList.toggle('worst',p.worst===l);});});
const n=Object.values(store).filter(p=>p.best||p.worst).length;document.getElementById('tally').textContent=n+' of '+Object.keys(KEY).length+' rows picked';}
document.querySelectorAll('.row').forEach(r=>{r.querySelectorAll('.arm').forEach(a=>{a.querySelectorAll('button').forEach(b=>{
b.onclick=()=>{const id=r.dataset.id,k=b.dataset.k,l=a.dataset.l;const p=store[id]||{};p[k]=p[k]===l?undefined:l;
if(k==='best'&&p.worst===l)p.worst=undefined;if(k==='worst'&&p.best===l)p.best=undefined;store[id]=p;save();paint();};});});});
document.getElementById('reveal').onclick=()=>{const t={};document.querySelectorAll('.row').forEach(r=>{const id=r.dataset.id,p=store[id]||{};
r.querySelectorAll('.arm').forEach(a=>{const k=KEY[id][a.dataset.l];a.querySelector('.rev').textContent=k.arm+': '+k.note;
t[k.arm]=t[k.arm]||{best:0,worst:0};if(p.best===a.dataset.l)t[k.arm].best++;if(p.worst===a.dataset.l)t[k.arm].worst++;});});
document.getElementById('tally').textContent=Object.entries(t).map(([k,v])=>k+' best '+v.best+' worst '+v.worst).join('   ');};
document.getElementById('dl').onclick=()=>{const out=Object.keys(KEY).map(id=>{const p=store[id]||{};
return {row:id,best:p.best?KEY[id][p.best].arm:null,worst:p.worst?KEY[id][p.worst].arm:null,letters:KEY[id]};});
const b=new Blob([JSON.stringify({page:'loudness',picks:out},null,1)],{type:'application/json'});const u=URL.createObjectURL(b);
const a=document.createElement('a');a.href=u;a.download='loudness_picks.json';document.body.appendChild(a);a.click();a.remove();
setTimeout(()=>URL.revokeObjectURL(u),1000);};
document.getElementById('clear').onclick=()=>{store={};save();paint();document.querySelectorAll('.rev').forEach(e=>e.textContent='');};
paint();</script></body></html>"""


def section(title, rows, key):
    out = ["<h2>%s</h2>" % html.escape(title)]
    for rid, label, sub, arms in rows:
        letters = "ABCD"
        cells = []
        key[rid] = {}
        for i, arm in enumerate(order_for(rid, arms)):
            y, note = arms[arm]
            name = write(rid, arm, y)
            key[rid][letters[i]] = {"arm": arm, "note": note}
            cells.append("<div class='arm' data-l='%s'><span class='lab'>%s</span><audio controls preload='none' src='a/%s'>"
                         "</audio><div class='pick'><button data-k='best'>best</button><button data-k='worst'>worst</button>"
                         "</div><div class='rev'></div></div>" % (letters[i], letters[i], name))
        out.append("<section class='row' data-id='%s'><h3>%s</h3><p class='sub'>%s</p><div class='arms'>%s</div></section>" % (
            rid, html.escape(label), html.escape(sub), "".join(cells)))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="blind page: current finishing chain against loudness levelling, CPU only")
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "a"))
    key = {}
    body = section("In the game", game_rows(), key) + section("Single clips", clip_rows(seeds), key)
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(PAGE % (body, json.dumps(key)))
    print("%d rows, %d clips -> %s" % (len(key), sum(len(v) for v in key.values()), os.path.join(OUT, "index.html")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
