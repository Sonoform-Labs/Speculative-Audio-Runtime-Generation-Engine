import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import glob
import hashlib
import html
import json
import shutil

import numpy as np

import vocab_eval as ve
from sarge.amp import amplify, family_for
from sarge.loudness import normalize_peak, peak_for
from sarge.types import AudioClip

ROOT = os.path.dirname(os.path.dirname(BENCH))
SRC = os.path.join(ROOT, "data", "_gate", "listen", "present")
OUT = os.path.join(ROOT, "data", "_gate", "listen", "amp_ab")


def db(v):
    return 20.0 * np.log10(max(float(v), 1e-9))


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "a"))
    rows, key, stats = [], {}, []
    for f in sorted(glob.glob(os.path.join(SRC, "*.wav"))):
        stem = os.path.basename(f)[:-4]
        event, rest = stem.split("-", 1)
        x = ve.wav(f)
        before = normalize_peak(x, peak_for("normal"))
        after = normalize_peak(amplify(x, ve.SR, event, "normal"), peak_for("normal"))
        stats.append({"clip": stem, "family": family_for(event), "rms_db_before": db(np.sqrt(np.mean(before ** 2))),
                      "rms_db_after": db(np.sqrt(np.mean(after ** 2)))})
        h = int(hashlib.md5(stem.encode()).hexdigest(), 16)
        order = [("before", before), ("after", after)] if h % 2 else [("after", after), ("before", before)]
        names = []
        for tag, y in order:
            nm = hashlib.md5((stem + tag).encode()).hexdigest()[:10] + ".wav"
            AudioClip(y, ve.SR).to_wav(os.path.join(OUT, "a", nm))
            names.append(nm)
        key[stem] = order[1][0] == "after"
        rows.append("<tr data-id='%s'><td>%s</td><td><audio controls preload='none' src='a/%s'></audio></td>"
                    "<td><audio controls preload='none' src='a/%s'></audio></td><td><button data-v='a'>A</button>"
                    "<button data-v='same'>same</button><button data-v='b'>B</button></td><td class='rev'></td></tr>"
                    % (stem, html.escape(stem.replace("-", " ").replace("_", " #")), names[0], names[1]))
    page = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Amplifier A/B</title><style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mute:#6b6b66;--line:#ddddd8;--acc:#2b5fd9}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ececea;--mute:#9a9a95;--line:#33332f;--acc:#8fb0ff}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:24px 16px;max-width:900px}
table{border-collapse:collapse;width:100%%}td,th{border-bottom:1px solid var(--line);padding:8px;text-align:left}
audio{width:220px;height:32px}button{font:inherit;padding:4px 10px;margin:0 2px;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px}
button.on{border-color:var(--acc);color:var(--acc);font-weight:600}p{color:var(--mute);max-width:720px}.rev{color:var(--acc);font-weight:600}
</style></head><body><h1>Amplifier A/B</h1>
<p>Each row is one generated sound, played with and without the amplifier stage (attack emphasis plus saturation, set by
sound type: strongest on gunshots, lighter on impacts, light on footsteps). Both versions hit the same peak level, so
any difference you hear is punch and density, not volume. Pick the one that sounds better, then press Reveal.</p>
<table><tr><th>Sound</th><th>A</th><th>B</th><th>Better</th><th></th></tr>%s</table>
<p><button id='reveal'>Reveal</button> <span id='tally'></span></p>
<script>
const KEY=%s;let store={};try{store=JSON.parse(localStorage.getItem('amp_ab')||'{}')}catch(e){}
function save(){try{localStorage.setItem('amp_ab',JSON.stringify(store))}catch(e){}}
document.querySelectorAll('tr[data-id]').forEach(tr=>{const id=tr.dataset.id;tr.querySelectorAll('button').forEach(b=>{
 if(store[id]===b.dataset.v)b.classList.add('on');b.onclick=()=>{store[id]=b.dataset.v;save();
 tr.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));};});});
document.getElementById('reveal').onclick=()=>{let a=0,b=0,s=0;document.querySelectorAll('tr[data-id]').forEach(tr=>{
 const id=tr.dataset.id,bAfter=KEY[id];tr.querySelector('.rev').textContent=bAfter?'B = amplified':'A = amplified';
 const p=store[id];if(!p)return;if(p==='same')s++;else if((p==='b')===bAfter)a++;else b++;});
 document.getElementById('tally').textContent='amplified picked '+a+', plain picked '+b+', same '+s;};
</script></body></html>""" % ("".join(rows), json.dumps(key))
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
    for s in stats:
        print("%-22s %-9s rms %.1f -> %.1f dB at the same peak" % (s["clip"], s["family"], s["rms_db_before"], s["rms_db_after"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
