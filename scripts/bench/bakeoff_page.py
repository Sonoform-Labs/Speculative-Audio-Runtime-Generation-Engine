import argparse
import hashlib
import html
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "data", "_gate", "bakeoff")
MODELS = {
    "small_prod": ("SA3 small-sfx, today's engine", "0.022 s", "Stability Community"),
    "medium_rec": ("SA3 medium, recommended pingpong 8 steps, CFG 1", "0.26 s compiled", "Stability Community"),
    "medium_e8": ("SA3 medium, euler 8 steps", "0.49 s", "Stability Community"),
    "moss_100_w10": ("MOSS-SoundEffect v2, 100 steps", "19.7 s", "Apache-2.0"),
    "woosh_flow": ("Sony Woosh Flow", "6.3 s", "CC-BY-NC, reference only"),
}
KEYS = ["shoot-pistol", "shoot-rifle", "break-window", "bullet_impact-glass", "footstep-gravel", "footstep-stone",
        "footstep-wood", "footstep-metal", "footstep-water", "splash-water", "punch-body", "reload-rifle"]


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--seed", default="bake1")
    return ap.parse_args(argv)


def h(s):
    return hashlib.md5(s.encode()).hexdigest()


def main(argv=None):
    a = parse(argv)
    out = os.path.join(BASE, "page")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "a"))
    key, sections = {}, []
    letters = "ABCDEFGHIJ"
    for k in KEYS:
        models = sorted(MODELS, key=lambda m: h(a.seed + k + m))
        rows = []
        for i, m in enumerate(models):
            clips = []
            for v in range(a.takes):
                src = os.path.join(BASE, "listen", m, "%s_%02d.wav" % (k, v))
                if not os.path.isfile(src):
                    continue
                nm = h(a.seed + k + m + str(v))[:10] + ".wav"
                shutil.copy(src, os.path.join(out, "a", nm))
                clips.append("<audio controls preload='none' src='a/%s'></audio>" % nm)
            key["%s|%s" % (k, letters[i])] = m
            rows.append("<tr data-k='%s' data-l='%s'><th>%s</th><td class='clips'>%s</td><td>"
                        "<button data-v='best'>best</button><button data-v='worst'>worst</button></td>"
                        "<td class='rev'></td></tr>" % (k, letters[i], letters[i], "".join(clips)))
        sections.append("<h2>%s</h2><table>%s</table><p><button class='none' data-k='%s'>none sound right</button></p>"
                        % (html.escape(k.replace("-", " ").replace("_", " ")), "".join(rows), k))
    page = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Model Bake-off</title><style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mute:#6b6b66;--line:#ddddd8;--acc:#2b5fd9;--bad:#c2410c}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ececea;--mute:#9a9a95;--line:#33332f;--acc:#8fb0ff;--bad:#fb923c}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:24px 16px;max-width:1100px}
table{border-collapse:collapse;width:100%%}td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:middle}
.clips{display:flex;flex-wrap:wrap;gap:6px}audio{width:210px;height:32px}
button{font:inherit;padding:3px 10px;margin:0 2px;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;cursor:pointer}
button.on{border-color:var(--acc);color:var(--acc);font-weight:600}button.on[data-v='worst'],button.none.on{border-color:var(--bad);color:var(--bad)}
p{color:var(--mute);max-width:820px}.rev{color:var(--acc);font-size:13px}h2{margin-top:28px;text-transform:capitalize}
</style></head><body><h1>Model bake-off</h1>
<p>Use headphones. Every option is the same detailed prompt rendered by a different model, three takes each, in stereo,
peak-levelled to the same height. Letters are shuffled per sound. Mark the best and the worst option for each sound, or
"none sound right". Press Reveal at the end for the model names, speed per clip and licence.</p>
%s
<p><button id='reveal'>Reveal</button> <button id='dl'>Download picks</button> <span id='tally'></span></p>
<script>
const KEY=%s, INFO=%s;
let store={};try{store=JSON.parse(localStorage.getItem('bakeoff1')||'{}')}catch(e){}
function save(){try{localStorage.setItem('bakeoff1',JSON.stringify(store))}catch(e){}}
function paint(){document.querySelectorAll('tr[data-k]').forEach(tr=>{const k=tr.dataset.k,l=tr.dataset.l,s=store[k]||{};
 tr.querySelectorAll('button').forEach(b=>b.classList.toggle('on',s[b.dataset.v]===l));});
 document.querySelectorAll('button.none').forEach(b=>b.classList.toggle('on',!!(store[b.dataset.k]||{}).none));}
document.querySelectorAll('tr[data-k] button').forEach(b=>b.onclick=()=>{const tr=b.closest('tr'),k=tr.dataset.k;
 store[k]=store[k]||{};store[k][b.dataset.v]=tr.dataset.l;store[k].none=false;save();paint();});
document.querySelectorAll('button.none').forEach(b=>b.onclick=()=>{const k=b.dataset.k;store[k]={none:!(store[k]||{}).none};save();paint();});
document.getElementById('reveal').onclick=()=>{const best={},worst={};
 document.querySelectorAll('tr[data-k]').forEach(tr=>{const m=KEY[tr.dataset.k+'|'+tr.dataset.l],i=INFO[m];
  tr.querySelector('.rev').textContent=i[0]+' | '+i[1]+' per clip | '+i[2];});
 Object.entries(store).forEach(([k,s])=>{if(s.best){const m=KEY[k+'|'+s.best];best[m]=(best[m]||0)+1;}
  if(s.worst){const m=KEY[k+'|'+s.worst];worst[m]=(worst[m]||0)+1;}});
 document.getElementById('tally').textContent='best: '+JSON.stringify(best)+'  worst: '+JSON.stringify(worst);};
document.getElementById('dl').onclick=()=>{const x=document.createElement('a');
 x.href=URL.createObjectURL(new Blob([JSON.stringify({picks:store,key:KEY},null,1)],{type:'application/json'}));
 x.download='bakeoff_picks.json';x.click();};
paint();
</script></body></html>""" % ("".join(sections), json.dumps(key), json.dumps(MODELS))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    print("wrote %s (%d sounds, %d options each)" % (os.path.join(out, "index.html"), len(KEYS), len(MODELS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
