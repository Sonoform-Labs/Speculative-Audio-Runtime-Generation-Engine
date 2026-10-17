import argparse
import base64
import hashlib
import html
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import world_render as wr
from sarge.types import AudioClip

ROOT = HERE.parent.parent
OUT = ROOT / "data" / "_gate" / "feel" / "world_page"
TAKES = 3
LETTERS = "ABC"
ARMS = {
    "head": "HEAD: prompt from the event and material words, peak level, amplifier with saturation",
    "world": "--world: physics bucket prompt and the bucket's decay cap, attack and low-pass, peak level",
    "world_loud": "--world --level loudness: the same render as --world, levelled by 400 ms loudness",
}
LABELS = {
    "ball-metal": "Rubber ball bouncing on the metal tile (hard, 12 to 25 m/s)",
    "ball-stone": "Rubber ball bouncing on the stone tile (hard)",
    "ball-wood": "Rubber ball bouncing on the wood tile (hard)",
    "ball-grass": "Rubber ball bouncing on the grass tile (hard)",
    "ball-glass": "Rubber ball bouncing on the glass tile (hard)",
    "bottle-glass": "Glass bottle dropping onto the glass tile (hard)",
    "bottle-gravel": "Glass bottle landing on gravel (normal)",
    "barrel-gravel": "Metal barrel dropping onto gravel (hard)",
    "crate-gravel": "Wooden crate landing on gravel (normal)",
    "crate-barrel": "Wooden crate landing on the metal barrel (normal)",
    "window-break": "Window pane shot and breaking (normal)",
}
VOLUME = 1.0


def h(s):
    return hashlib.md5(s.encode()).hexdigest()


def facts(y):
    st = wr.level_stats(y)
    return {k: round(float(st[k]), 3) for k in ("lufs", "t40", "len", "peak_db", "crest_db")}


def build(out, seed):
    if out.is_dir():
        shutil.rmtree(out)
    (out / "a").mkdir(parents=True)
    meta = wr.load_json(wr.OUT / "demo_meta.json")
    key, info, blocks, n_clips = {}, {}, [], 0
    for name, ev, mat, inten, phys in wr.DEMO:
        sid = name
        arms = sorted(ARMS, key=lambda m: h(seed + sid + m))
        rows = []
        for i, arm in enumerate(arms):
            clips, nums = [], []
            for v in range(TAKES):
                y = wr.finished(name, arm, v) * np.float32(VOLUME)
                nm = h(seed + sid + arm + str(v))[:12] + ".wav"
                AudioClip(np.ascontiguousarray(y, dtype=np.float32), wr.SR).to_wav(str(out / "a" / nm))
                n_clips += 1
                nums.append(facts(y))
                clips.append("<audio controls preload='none' src='a/%s' aria-label='%s take %d'></audio>"
                             % (nm, LETTERS[i], v + 1))
            key["%s|%s" % (sid, LETTERS[i])] = arm
            k = wr.demo_key(name, arm)
            src = meta["pairs"][name]["arms"]["world" if wr.ARMS[arm][1] else "head"]
            med = {f: float(np.median([n[f] for n in nums])) for f in nums[0]}
            info["%s|%s" % (sid, LETTERS[i])] = (
                "%s | takes 1-3: M400 %s LUFS, -40 dB after %s s, length %s s | median M400 %.1f LUFS, -40 dB after "
                "%.2f s | prompt: %s" % (
                    ARMS[arm], " / ".join("%.1f" % n["lufs"] for n in nums), " / ".join("%.2f" % n["t40"] for n in nums),
                    " / ".join("%.2f" % n["len"] for n in nums), med["lufs"], med["t40"], src["prompt"]))
            info["%s|%s|nums" % (sid, LETTERS[i])] = {"takes": nums, "median": med, "phys": k.phys}
            rows.append("<div class='opt' data-s='%s' data-l='%s'><div class='let'>%s</div>"
                        "<div class='takes'>%s</div><div class='btns'><button data-v='best'>best</button>"
                        "<button data-v='worst'>worst</button></div><div class='rev'></div></div>"
                        % (sid, LETTERS[i], LETTERS[i], "".join(clips)))
        blocks.append("<section class='snd' data-s='%s'><h3>%s</h3>%s<p class='row'><button class='none' "
                      "data-s='%s'>none sound right</button><span class='st'></span></p></section>"
                      % (sid, html.escape(LABELS.get(name, name)), "".join(rows), sid))
    payload = base64.b64encode(json.dumps({"key": key, "info": info, "arms": ARMS}, ensure_ascii=True).encode()).decode()
    page = TEMPLATE.replace("%SECTIONS%", "".join(blocks)).replace("%DATA%", payload).replace(
        "%NSOUNDS%", str(len(wr.DEMO))).replace("%STORE%", "sarge_world_page_" + seed)
    with open(out / "index.html", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    with open(out.parent / (out.name + "_key.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"seed": seed, "key": key, "info": info}, fh, indent=1)
    return len(wr.DEMO), n_clips


TEMPLATE = """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Physics World Listening</title><style>
:root{--bg:#f7f7f4;--fg:#1c1c1a;--mute:#66665f;--line:#dcdcd5;--card:#ffffff;--acc:#2456c9;--bad:#b4400c;--ok:#2f7d32}
@media (prefers-color-scheme: dark){:root:not([data-theme='light']){--bg:#151514;--fg:#ececea;--mute:#9c9c96;--line:#34342f;--card:#1d1d1b;--acc:#8fb0ff;--bad:#fb923c;--ok:#7bc47f}}
:root[data-theme='dark']{--bg:#151514;--fg:#ececea;--mute:#9c9c96;--line:#34342f;--card:#1d1d1b;--acc:#8fb0ff;--bad:#fb923c;--ok:#7bc47f}
*{box-sizing:border-box}
html{background:var(--bg);overflow-x:hidden}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif;margin:0 auto;padding:24px 16px 110px;max-width:1080px;overflow-x:hidden}
h1{font-size:24px;margin:0 0 8px}
h3{font-size:16px;margin:0 0 6px}
p{color:var(--mute);max-width:760px;margin:6px 0}
.snd{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:14px 0}
.opt{display:grid;grid-template-columns:2rem minmax(0,1fr) auto;gap:6px 12px;align-items:center;padding:8px 0;border-top:1px solid var(--line)}
.let{font-weight:700;font-size:17px}
.takes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;min-width:0}
audio{width:100%;min-width:0;height:34px}
.btns{display:flex;gap:6px}
button{font:inherit;font-size:14px;padding:4px 12px;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;cursor:pointer}
button:hover{border-color:var(--mute)}
button.on{border-color:var(--acc);color:var(--acc);font-weight:600}
button.on[data-v='worst'],button.none.on{border-color:var(--bad);color:var(--bad)}
.rev{grid-column:1/-1;color:var(--acc);font-size:13px;display:none;overflow-wrap:anywhere}
body.revealed .rev{display:block}
.row{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:10px 0 0}
.st{font-size:13px;color:var(--mute)}.st.done{color:var(--ok)}
.bar{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px 16px}
.bar div{max-width:1080px;margin:0 auto;display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center}
#prog{font-size:14px;color:var(--mute);margin-right:auto}
#tally{margin:24px 0}#tally table{border-collapse:collapse;width:100%;font-size:14px}
#tally th,#tally td{text-align:left;border-bottom:1px solid var(--line);padding:5px 8px;vertical-align:top}
#tally td.n{text-align:right;font-variant-numeric:tabular-nums}
.wrap{overflow-x:auto}
@media (max-width:640px){.opt{grid-template-columns:2rem minmax(0,1fr)}.btns{grid-column:2}.takes{grid-template-columns:minmax(0,1fr)}h1{font-size:21px}}
</style></head><body>
<h1>Physics world: blind listening</h1>
<p>Use headphones. Each row is one collision from the demo arena, rendered by the Woosh worker the way the game would ask
for it, and finished three ways. Every option plays three takes, the variants the game rotates through. Letters are
shuffled per row.</p>
<p>Levels are what the game plays: every option in a row goes out at the same playback gain (volume 1, the source at its
minimum distance), so a louder or quieter option is louder or quieter in the game too. Mark the best and the worst option
in each row, or press "none sound right". Picks stay in this browser. Reveal names each option with its loudness, decay
and length; Download saves your picks with the key.</p>
%SECTIONS%
<section id='tally'></section>
<div class='bar'><div><span id='prog'></span><button id='reveal'>Reveal</button><button id='dl'>Download picks</button><button id='clear'>Clear picks</button></div></div>
<script>
const D=JSON.parse(atob('%DATA%'));
const SK='%STORE%', N=%NSOUNDS%;
let store={};
try{store=JSON.parse(localStorage.getItem(SK)||'{}')||{}}catch(e){store={}}
let shown=false;
try{shown=localStorage.getItem(SK+'_rev')==='1'}catch(e){}
function save(){try{localStorage.setItem(SK,JSON.stringify(store))}catch(e){}}
function done(s){return !!(s&&(s.none||(s.best&&s.worst)))}
function paint(){
 document.querySelectorAll('.opt').forEach(o=>{const s=store[o.dataset.s]||{};
  o.querySelectorAll('button').forEach(b=>b.classList.toggle('on',s[b.dataset.v]===o.dataset.l));});
 document.querySelectorAll('section.snd').forEach(sec=>{const s=store[sec.dataset.s]||{};
  sec.querySelector('button.none').classList.toggle('on',!!s.none);
  const st=sec.querySelector('.st');
  st.textContent=s.none?'none sound right':(s.best||s.worst)?('best '+(s.best||'-')+', worst '+(s.worst||'-')):'';
  st.classList.toggle('done',done(s));});
 const n=Object.values(store).filter(done).length;
 document.getElementById('prog').textContent=n+' of '+N+' rows rated';
 if(shown)reveal();}
document.querySelectorAll('.opt button').forEach(b=>b.onclick=()=>{const o=b.closest('.opt'),s=o.dataset.s,l=o.dataset.l,v=b.dataset.v;
 const cur=Object.assign({},store[s]||{});cur.none=false;
 if(cur[v]===l){delete cur[v]}else{cur[v]=l;const w=v==='best'?'worst':'best';if(cur[w]===l)delete cur[w];}
 store[s]=cur;save();paint();});
document.querySelectorAll('button.none').forEach(b=>b.onclick=()=>{const s=b.dataset.s;
 store[s]=(store[s]||{}).none?{}:{none:true};save();paint();});
function arm(s,l){return D.key[s+'|'+l]}
function reveal(){
 document.body.classList.add('revealed');
 document.querySelectorAll('.opt').forEach(o=>{o.querySelector('.rev').textContent=D.info[o.dataset.s+'|'+o.dataset.l]||'';});
 const t={};Object.keys(D.arms).forEach(m=>{t[m]={best:0,worst:0}});let none=0;
 Object.entries(store).forEach(([s,v])=>{if(v.none){none++;return}
  const b=arm(s,v.best),w=arm(s,v.worst);if(t[b])t[b].best++;if(t[w])t[w].worst++;});
 let h='<h2>Tally</h2><div class="wrap"><table><tr><th>option</th><th>best</th><th>worst</th><th>what it is</th></tr>';
 Object.entries(D.arms).forEach(([m,d])=>{h+='<tr><td>'+m+'</td><td class="n">'+t[m].best+'</td><td class="n">'+t[m].worst+'</td><td>'+d+'</td></tr>';});
 h+='<tr><td>none sound right</td><td class="n" colspan="2">'+none+'</td><td></td></tr>';
 document.getElementById('tally').innerHTML=h+'</table></div>';
 shown=true;try{localStorage.setItem(SK+'_rev','1')}catch(e){}}
document.getElementById('reveal').onclick=reveal;
document.getElementById('dl').onclick=()=>{const picks={};
 Object.entries(store).forEach(([s,v])=>{picks[s]={best:v.best||null,worst:v.worst||null,none:!!v.none,
  best_option:v.best?arm(s,v.best):null,worst_option:v.worst?arm(s,v.worst):null}});
 const x=document.createElement('a');
 x.href=URL.createObjectURL(new Blob([JSON.stringify({page:'world',saved:new Date().toISOString(),revealed:shown,picks:picks,key:D.key},null,1)],{type:'application/json'}));
 x.download='world_picks.json';document.body.appendChild(x);x.click();x.remove();};
document.getElementById('clear').onclick=()=>{if(!confirm('Clear every pick on this page?'))return;
 store={};shown=false;save();try{localStorage.removeItem(SK+'_rev')}catch(e){}
 document.body.classList.remove('revealed');document.getElementById('tally').innerHTML='';paint();};
paint();
</script></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="blind page: HEAD against --world and --world --level loudness on the demo pairs")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", default="world1")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    n_sounds, n_clips = build(Path(a.out), a.seed)
    print("wrote %s (%d rows, %d clips)" % (Path(a.out) / "index.html", n_sounds, n_clips), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
