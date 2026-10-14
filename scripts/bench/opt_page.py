import argparse
import base64
import hashlib
import html
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OPT = ROOT / "data" / "_gate" / "opt"
TAKES = 3
LETTERS = "ABCDEFGHIJ"

GROUPS = [
    {"id": "guns", "family": "medium", "ref": "m0_eager",
     "title": "Guns: SA3 medium",
     "blurb": "Gunshots, reloads, clicks and equips. The production setting runs at 146 ms per clip.",
     "arms": [("m_p1t", "production default: pingpong 8 steps, cfg 1, 3 s generation, 1 s padding, compiled "
                        "(bit-identical to the shipped m_final)"),
              ("m0_eager", "the ear-approved bake-off recipe before the speed work: eager, 6 s padding"),
              ("m_s6", "candidate: 6 pingpong steps"),
              ("m_g2p", "candidate: 2.0 s generation (cuts some tails)"),
              ("m_g15p", "candidate: 1.5 s generation (cuts most tails)")]},
    {"id": "dflow", "family": "woosh", "ref": "w0_base",
     "title": "Everything else: Woosh DFlow (the hybrid default)",
     "blurb": "Glass, water, footsteps and hits. The production setting runs at 28 ms per clip.",
     "arms": [("w_final", "production default: DFlow 4 steps, 1.0 s, bf16, CUDA graph"),
              ("w0_base", "before the speed work: DFlow 4 steps, 1.0 s, fp32 eager"),
              ("w_s4f151", "candidate: 4 steps, 1.5 s render"),
              ("w_s6f101", "candidate: 6 steps, 1.0 s render"),
              ("w_s6f151", "candidate: 6 steps, 1.5 s render"),
              ("w_s8f101", "candidate: 8 steps, 1.0 s render"),
              ("w_s8f151", "candidate: 8 steps, 1.5 s render")]},
    {"id": "flow", "family": "woosh", "ref": "f0_base",
     "title": "Woosh Flow (the slower option, --woosh-model flow)",
     "blurb": "The same nine sounds from the Flow model. Its production setting runs at about 1 s per clip.",
     "arms": [("f_final", "production default for --woosh-model flow: adaptive dopri5, bf16, batched CFG"),
              ("f0_base", "before the speed work: adaptive dopri5, fp32 eager"),
              ("f_rk4_8", "candidate: fixed rk4, 8 steps (32 evaluations)"),
              ("f_rk4_16", "candidate: fixed rk4, 16 steps (64 evaluations)"),
              ("f_mid_16", "candidate: fixed midpoint, 16 steps (32 evaluations)"),
              ("f_mid_32", "candidate: fixed midpoint, 32 steps (64 evaluations)")]},
]
GUN_KEYS = ["shoot-pistol", "shoot-rifle", "reload-rifle", "empty_click-pistol", "equip-rifle"]
REST_KEYS = ["break-window", "bullet_impact-glass", "footstep-gravel", "footstep-stone", "footstep-wood",
             "footstep-metal", "footstep-water", "splash-water", "punch-body"]


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def h(s):
    return hashlib.md5(s.encode()).hexdigest()


def place(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def arm_facts(group, arm):
    fam = OPT / group["family"]
    res = load_json(fam / "results.json")["arms"]
    r = res.get(arm, {})
    a = res.get(arm + "_a")
    ms = r.get("ms_per_clip")
    cls = r.get("class") or "-"
    ratio = None
    cp = fam / "_work" / ("compare_%s.json" % group["ref"])
    if arm == group["ref"]:
        ratio = 0.0
    elif cp.is_file():
        c = load_json(cp)["arms"]
        ratio = (c.get(arm) or c.get(arm + "_a") or {}).get("g1_logmel", {}).get("ratio_to_seed")
    out = {"arm": arm, "class": cls, "ms_rendered": ms, "ms_ship": a.get("ms_per_clip") if a else ms,
           "precision_rendered": (r.get("config") or {}).get("precision"),
           "precision_ship": ((a or {}).get("config") or {}).get("precision") or (r.get("config") or {}).get("precision"),
           "ratio_vs_ref": ratio, "ref": group["ref"], "vram_reserved_gb": r.get("vram_peak_reserved_gb")}
    return out


def label(f, desc):
    ms = f["ms_ship"]
    sp = "%.0f ms per clip" % ms if ms is not None else "speed not measured"
    if f["ms_ship"] != f["ms_rendered"] and f["ms_rendered"] is not None:
        sp += " at %s (heard as rendered at %s, %.0f ms)" % (f["precision_ship"], f["precision_rendered"],
                                                              f["ms_rendered"])
    mv = ("log-mel moved %.2f of a new seed from %s" % (f["ratio_vs_ref"], f["ref"])
          if f["ratio_vs_ref"] is not None else "no log-mel number")
    return "%s | %s | %s | gate %s | %s" % (f["arm"], sp, mv, f["class"], desc)


def build(out, seed):
    if out.is_dir():
        shutil.rmtree(out)
    (out / "a").mkdir(parents=True)
    key, info, facts, sections, n_clips, n_sounds = {}, {}, {}, [], 0, 0
    for g in GROUPS:
        keys = GUN_KEYS if g["id"] == "guns" else REST_KEYS
        for arm, desc in g["arms"]:
            f = arm_facts(g, arm)
            facts[arm] = dict(f, desc=desc, group=g["id"])
            info[arm] = label(f, desc)
        blocks = []
        for k in keys:
            sid = "%s|%s" % (g["id"], k)
            arms = sorted([a for a, _d in g["arms"]], key=lambda m: h(seed + sid + m))
            rows = []
            for i, m in enumerate(arms):
                clips = []
                for v in range(TAKES):
                    src = OPT / g["family"] / m / ("%s_%02d.wav" % (k, v))
                    if not src.is_file():
                        raise SystemExit("missing %s" % src)
                    nm = h(seed + sid + m + str(v))[:12] + ".wav"
                    place(src, out / "a" / nm)
                    n_clips += 1
                    clips.append("<audio controls preload='none' src='a/%s' aria-label='%s take %d'></audio>"
                                 % (nm, LETTERS[i], v + 1))
                key["%s|%s" % (sid, LETTERS[i])] = m
                rows.append("<div class='opt' data-s='%s' data-l='%s'><div class='let'>%s</div>"
                            "<div class='takes'>%s</div><div class='btns'><button data-v='best'>best</button>"
                            "<button data-v='worst'>worst</button></div><div class='rev'></div></div>"
                            % (sid, LETTERS[i], LETTERS[i], "".join(clips)))
            n_sounds += 1
            name = k.replace("-", " ").replace("_", " ")
            blocks.append("<section class='snd' data-s='%s'><h3>%s</h3>%s<p class='row'><button class='none' "
                          "data-s='%s'>none sound right</button><span class='st'></span></p></section>"
                          % (sid, html.escape(name), "".join(rows), sid))
        sections.append("<h2 id='%s'>%s</h2><p>%s %d options per sound, %d takes each.</p>%s"
                        % (g["id"], html.escape(g["title"]), html.escape(g["blurb"]), len(g["arms"]), TAKES,
                           "".join(blocks)))
    payload = base64.b64encode(json.dumps({"key": key, "info": info, "facts": facts,
                                           "groups": {g["id"]: g["title"] for g in GROUPS}},
                                          ensure_ascii=True).encode()).decode()
    page = TEMPLATE.replace("%SECTIONS%", "".join(sections)).replace("%DATA%", payload).replace(
        "%NSOUNDS%", str(n_sounds)).replace("%STORE%", "sarge_opt_page_" + seed)
    with open(out / "index.html", "w", encoding="utf-8") as fh:
        fh.write(page)
    with open(out.parent / (out.name + "_key.json"), "w", encoding="utf-8") as fh:
        json.dump({"seed": seed, "key": key, "facts": facts}, fh, indent=1)
    return n_sounds, n_clips


TEMPLATE = """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Speed Candidates Listening</title><style>
:root{--bg:#f7f7f4;--fg:#1c1c1a;--mute:#66665f;--line:#dcdcd5;--card:#ffffff;--acc:#2456c9;--bad:#b4400c;--ok:#2f7d32}
@media (prefers-color-scheme: dark){:root:not([data-theme='light']){--bg:#151514;--fg:#ececea;--mute:#9c9c96;--line:#34342f;--card:#1d1d1b;--acc:#8fb0ff;--bad:#fb923c;--ok:#7bc47f}}
:root[data-theme='dark']{--bg:#151514;--fg:#ececea;--mute:#9c9c96;--line:#34342f;--card:#1d1d1b;--acc:#8fb0ff;--bad:#fb923c;--ok:#7bc47f}
*{box-sizing:border-box}
html{background:var(--bg)}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif;margin:0 auto;padding:24px 16px 96px;max-width:1080px}
h1{font-size:24px;margin:0 0 8px}h2{font-size:20px;margin:40px 0 4px;padding-top:8px;border-top:2px solid var(--line)}
h3{font-size:16px;margin:0 0 6px;text-transform:capitalize}
p{color:var(--mute);max-width:760px;margin:6px 0}
nav{display:flex;flex-wrap:wrap;gap:8px 16px;margin:12px 0}nav a{color:var(--acc)}
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
.rev{grid-column:1/-1;color:var(--acc);font-size:13px;display:none}
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
<h1>Speed candidates: blind listening</h1>
<p>Use headphones. Every sound below comes from the same prompt and the same three seeds, rendered by different settings
and finished the way the game gets it (natural trim, amplifier, stereo widening, peak level). Each sound offers the
production setting, the setting before the speed work, and the candidates that changed the sound too much to adopt
without your ear. Letters are shuffled per sound.</p>
<p>For each sound mark the best and the worst option, or press "none sound right". Picks stay in this browser. Reveal shows
each setting, its speed per clip and how far it moved from the setting before the speed work (log-mel distance as a
fraction of a new seed). Download saves your picks with the key.</p>
<nav><a href='#guns'>Guns (SA3 medium)</a><a href='#dflow'>Woosh DFlow</a><a href='#flow'>Woosh Flow</a><a href='#tally'>Tally</a></nav>
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
 document.getElementById('prog').textContent=n+' of '+N+' sounds rated';
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
 document.querySelectorAll('.opt').forEach(o=>{o.querySelector('.rev').textContent=D.info[arm(o.dataset.s,o.dataset.l)]||'';});
 const t={};
 Object.keys(D.facts).forEach(m=>{t[m]={best:0,worst:0}});
 const none={};
 Object.entries(store).forEach(([s,v])=>{const g=s.split('|')[0];
  if(v.none){none[g]=(none[g]||0)+1;return}
  const b=arm(s,v.best),w=arm(s,v.worst);if(t[b])t[b].best++;if(t[w])t[w].worst++;});
 let h='<h2>Tally</h2><div class="wrap"><table><tr><th>group</th><th>setting</th><th>best</th><th>worst</th><th>ms per clip</th><th>what it is</th></tr>';
 Object.keys(D.groups).forEach(g=>{
  Object.entries(D.facts).filter(([m,f])=>f.group===g).forEach(([m,f])=>{
   h+='<tr><td>'+g+'</td><td>'+m+'</td><td class="n">'+t[m].best+'</td><td class="n">'+t[m].worst+'</td><td class="n">'+
    (f.ms_ship==null?'-':Math.round(f.ms_ship))+'</td><td>'+f.desc+'</td></tr>';});
  h+='<tr><td>'+g+'</td><td>none sound right</td><td class="n" colspan="2">'+(none[g]||0)+'</td><td></td><td></td></tr>';});
 document.getElementById('tally').innerHTML=h+'</table></div>';
 shown=true;try{localStorage.setItem(SK+'_rev','1')}catch(e){}}
document.getElementById('reveal').onclick=reveal;
document.getElementById('dl').onclick=()=>{const picks={};
 Object.entries(store).forEach(([s,v])=>{picks[s]={best:v.best||null,worst:v.worst||null,none:!!v.none,
  best_setting:v.best?arm(s,v.best):null,worst_setting:v.worst?arm(s,v.worst):null}});
 const x=document.createElement('a');
 x.href=URL.createObjectURL(new Blob([JSON.stringify({saved:new Date().toISOString(),revealed:shown,picks:picks,key:D.key,facts:D.facts},null,1)],{type:'application/json'}));
 x.download='opt_picks.json';document.body.appendChild(x);x.click();x.remove();};
document.getElementById('clear').onclick=()=>{if(!confirm('Clear every pick on this page?'))return;
 store={};shown=false;save();try{localStorage.removeItem(SK+'_rev')}catch(e){}
 document.body.classList.remove('revealed');document.getElementById('tally').innerHTML='';paint();};
paint();
</script></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="blind listening page for every G2 speed candidate (Woosh and SA3 medium)")
    ap.add_argument("--out", default=str(OPT / "page"))
    ap.add_argument("--seed", default="opt1")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    n_sounds, n_clips = build(Path(a.out), a.seed)
    print("wrote %s (%d sounds, %d clips)" % (Path(a.out) / "index.html", n_sounds, n_clips), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
