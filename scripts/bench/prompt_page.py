import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.isfile(os.path.join(HERE, "_bootstrap.py")):
    HERE = r"D:\Python\SAGE\scripts\bench"
ROOT = os.path.dirname(os.path.dirname(HERE))

import argparse
import base64
import json
import random
import secrets
import shutil
import time

OUT = os.path.join(ROOT, "data", "_gate", "prompt_search")
SEEDS = 4
ORDER = ["shoot|pistol", "shoot|rifle", "reload|rifle", "empty_click|pistol", "equip|rifle",
         "break|window", "bullet impact|glass", "bullet impact|metal", "bullet impact|wood",
         "footstep|stone", "footstep|wood", "footstep|gravel", "footstep|metal", "footstep|grass",
         "footstep|glass", "footstep|water", "land|stone", "splash|water", "punch|body", "click|button"]
TITLES = {"shoot|pistol": "Pistol shot", "shoot|rifle": "Rifle shot", "reload|rifle": "Rifle reload",
          "empty_click|pistol": "Pistol empty click", "equip|rifle": "Rifle equip", "break|window": "Window break",
          "bullet impact|glass": "Bullet hit on glass", "bullet impact|metal": "Bullet hit on metal",
          "bullet impact|wood": "Bullet hit on wood", "footstep|stone": "Footstep on stone",
          "footstep|wood": "Footstep on wood", "footstep|gravel": "Footstep on gravel",
          "footstep|metal": "Footstep on metal", "footstep|grass": "Footstep on grass",
          "footstep|glass": "Footstep on glass", "footstep|water": "Footstep in water",
          "land|stone": "Landing on stone", "splash|water": "Water splash", "punch|body": "Punch to the body",
          "click|button": "Button click"}
GROUPS = {"shoot": "Guns", "reload": "Guns", "empty_click": "Guns", "equip": "Guns", "break": "Glass and impacts",
          "bullet impact": "Glass and impacts", "footstep": "Footsteps", "land": "Footsteps", "splash": "Water",
          "punch": "Body and UI", "click": "Body and UI"}
HINTS = {"shoot|pistol": "A sharp crack with a slide clack. Judge the character; loudness comes later from the amplifier.",
         "shoot|rifle": "The crack plus the quick mechanical clicks of a real rifle shot.",
         "reload|rifle": "Magazine in, bolt or charging handle: mechanical, no shot.",
         "empty_click|pistol": "One small metallic click of a gun with no ammo.",
         "equip|rifle": "Picking up and shouldering a rifle: cloth, strap, metal parts, no shot.",
         "break|window": "The crack, then the high pings and tinkle of falling shards.",
         "bullet impact|glass": "A bullet going through glass: snap plus bright shards.",
         "bullet impact|metal": "A bullet hitting steel: ping or clang with a ring.",
         "bullet impact|wood": "A bullet hitting wood: thwack plus splinters.",
         "footstep|stone": "Hard heel click on stone, a little grit.",
         "footstep|wood": "Does it sound like wood planks, distinct from stone and gravel?",
         "footstep|gravel": "Grainy crunch of loose stones.",
         "footstep|metal": "A boot on a metal floor: clang and ring.",
         "footstep|grass": "Soft step with a rustle of grass.",
         "footstep|glass": "Stepping on broken glass: crunch with small bright pings.",
         "footstep|water": "A step in shallow water: splash detail, slosh, droplets.",
         "land|stone": "Both feet landing on stone after a jump.",
         "splash|water": "Something dropping into water: splash body, droplets, bubbles.",
         "punch|body": "A fist hitting a body: meaty thud and slap.",
         "click|button": "A single button press."}
MODES = {"vocab": "held-out AST, PANNs and CLAP identity judges (fraction of the real-recording rate)",
         "water": "validated water judge (fraction of the real-recording rate)",
         "shape": "shape only, compared with the nearest real sound type (no validated identity judge)",
         "unranked": "no judge and no matching real reference; the two new options were picked by hand"}
UNRANKED_PICKS = {
    "bullet impact|metal": {
        "library_0": "Title-shaped head (Bullet Hit, Metal) that mirrors real Freesound titles, steel-plate ping with a "
                     "ringing ricochet, bright spectrum (31% of energy above 4 kHz) at a mid audio-quality score (6.43).",
        "descriptive_2": "The brightest metal option (63% of energy above 4 kHz) with a 130 ms ringing tail, which is the "
                         "ping a listener expects; descriptive_1 and descriptive_3 render as dull barrel bongs."},
    "bullet impact|wood": {
        "library_0": "Real Freesound title shape (Bullet - Wood Hit) and the top library audio-quality score (7.35), "
                     "short thwack with a splinter tail.",
        "descriptive_3": "Picked for contrast: descriptive_0 is close to identical wording to library_0, so the fence "
                         "board thwack with splinter crackle gives the ear a real alternative (audio quality 7.06)."},
}
SPACE_KEYS = ["shoot|pistol", "shoot|rifle", "footstep|stone", "break|window"]
SPACES = ["outdoor_open", "indoor_small", "indoor_large", "tunnel"]
SPACE_NAMES = {"outdoor_open": "Open field", "indoor_small": "Small room", "indoor_large": "Large hall",
               "tunnel": "Concrete tunnel"}
STYLES = ["library", "descriptive"]


def fname(key, v):
    et, mat = key.split("|")
    return "%s-%s_%02d.wav" % (et.replace(" ", "_"), mat, v)


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


class Copier:
    def __init__(self, dst):
        self.dst = dst
        self.used = set()
        self.map = {}

    def __call__(self, src):
        if not os.path.isfile(src):
            raise SystemExit("missing clip %s" % src)
        while True:
            name = secrets.token_hex(5) + ".wav"
            if name not in self.used:
                break
        self.used.add(name)
        shutil.copy2(src, os.path.join(self.dst, name))
        self.map[name] = os.path.relpath(src, OUT)
        return name


def stat_line(r, n):
    parts = []
    if r.get("rank"):
        parts.append("rank %d of %d" % (r["rank"], n))
    if r.get("cons") is not None:
        ci = r.get("cons_ci") or [None, None]
        s = "judges %.2f of real" % r["cons"]
        if ci[0] is not None:
            s += " [%.2f, %.2f]" % (ci[0], ci[1])
        parts.append(s)
    if r.get("shape_in_range") is not None:
        parts.append("shape %.2f" % r["shape_in_range"])
    if r.get("pq") is not None:
        parts.append("PQ %.2f" % r["pq"])
    if r.get("material") is not None:
        parts.append("CLAP material %.2f" % r["material"])
    if r.get("static_rate"):
        parts.append("static guard flags %d%%" % round(100 * r["static_rate"]))
    top = r.get("clap_identity_top")
    if top:
        parts.append("CLAP top label: %s on %d of 12" % (top[0][0], top[0][1]))
    return " · ".join(parts)


def choose(key, entry):
    if key in UNRANKED_PICKS:
        return ["cur"] + list(UNRANKED_PICKS[key]), UNRANKED_PICKS[key]
    ranked = [r["slot"] for r in sorted((r for r in entry["ranking"] if r.get("rank")), key=lambda r: r["rank"])
              if r["slot"] != "cur"]
    return ["cur"] + ranked[:2], {}


def build(args):
    R = load_json(os.path.join(OUT, "ranking.json"))
    SP = load_json(os.path.join(OUT, "spaces", "prompts.json"))
    SS = load_json(os.path.join(OUT, "spaces", "space_stats.json"))
    page = os.path.join(OUT, "page")
    adir = os.path.join(page, "a")
    if os.path.isdir(adir):
        shutil.rmtree(adir)
    os.makedirs(adir)
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    rng = random.Random(seed)
    cp = Copier(adir)
    sections, manifest = [], {"seed": seed, "made": time.strftime("%Y-%m-%d %H:%M"), "keys": {}, "spaces": {}}
    for key in ORDER:
        entry = R["keys"][key]
        rows = {r["slot"]: r for r in entry["ranking"]}
        n = len([r for r in entry["ranking"] if r.get("rank")])
        slots, why = choose(key, entry)
        order = slots[:]
        rng.shuffle(order)
        opts = []
        for i, slot in enumerate(order):
            r = rows[slot]
            files = [cp(os.path.join(OUT, "natural", slot, fname(key, v))) for v in range(SEEDS)]
            opts.append({"letter": "ABC"[i], "slot": slot, "style": r["style"], "is_current": slot == "cur",
                         "prompt": r["prompt"], "stats": stat_line(r, n), "why": why.get(slot, ""),
                         "files": files})
        et = key.split("|")[0]
        sections.append({"key": key, "title": TITLES[key], "group": GROUPS[et], "hint": HINTS[key],
                         "mode": entry["mode"], "mode_text": MODES[entry["mode"]], "options": opts})
        manifest["keys"][key] = {o["letter"]: {"slot": o["slot"], "prompt": o["prompt"],
                                               "files": {f: cp.map[f] for f in o["files"]}} for o in opts}
    spaces = []
    for key in SPACE_KEYS:
        for style in STYLES:
            stem = fname(key, 0)[:-7]
            cells = []
            for sp in SPACES:
                rec = SP["prompts"][style][sp][key]
                st = SS[style][stem][sp]
                files = [cp(os.path.join(OUT, "spaces", style, sp, f)) for f in rec["files"][:SEEDS]]
                info = "tail to -20 dB %d ms · length %d ms" % (round(st["tail20_ms"]), round(st["dur_ms"]))
                if st.get("static"):
                    info += " · static guard flags %d%%" % round(100 * st["static"])
                cells.append({"space": sp, "name": SPACE_NAMES[sp], "prompt": rec["prompt"],
                              "from_slot": rec.get("from_slot"), "info": info, "files": files})
                manifest["spaces"].setdefault(key, {}).setdefault(style, {})[sp] = {
                    "prompt": rec["prompt"], "files": {f: cp.map[f] for f in files}}
            spaces.append({"key": key, "title": TITLES[key], "style": style, "cells": cells})
    data = {"made": manifest["made"], "seed": seed, "seeds": SEEDS, "sections": sections, "spaces": spaces}
    blob = base64.b64encode(json.dumps(data, ensure_ascii=False).encode("utf-8")).decode("ascii")
    with open(os.path.join(page, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(HTML.replace("@@DATA@@", blob))
    with open(os.path.join(page, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    total = sum(os.path.getsize(os.path.join(adir, f)) for f in os.listdir(adir))
    print("page %s  keys %d  space blocks %d  clips %d  %.1f MB  seed %d" % (
        os.path.join(page, "index.html"), len(sections), len(spaces), len(cp.used), total / 1e6, seed))
    for s in sections:
        print("  %-20s %-8s %s" % (s["key"], s["mode"], "  ".join("%s=%s" % (o["letter"], o["slot"]) for o in s["options"])))
    return 0


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prompt Blind Test</title>
<style>
:root{--bg:#f7f6f2;--card:#ffffff;--fg:#1d1c1a;--mute:#6a6862;--line:#dedcd5;--acc:#2456c9;--acc-bg:#e7eefc;--pick:#1d7a46;--pick-bg:#e3f3ea;--warn:#a2551b;--play:#c24a1f}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#151514;--card:#1f1f1d;--fg:#ebeae6;--mute:#9b9992;--line:#34332f;--acc:#8fb0ff;--acc-bg:#1f2a44;--pick:#6fd39b;--pick-bg:#173325;--warn:#e9a46c;--play:#ff8a5c}}
:root[data-theme="dark"]{--bg:#151514;--card:#1f1f1d;--fg:#ebeae6;--mute:#9b9992;--line:#34332f;--acc:#8fb0ff;--acc-bg:#1f2a44;--pick:#6fd39b;--pick-bg:#173325;--warn:#e9a46c;--play:#ff8a5c}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:0 16px 64px}
.bar{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line)}
.bar .in{max-width:1040px;margin:0 auto;padding:10px 16px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.bar h1{font-size:17px;margin:0 auto 0 0}
.prog{color:var(--mute);font-variant-numeric:tabular-nums}
button{font:inherit;color:var(--fg);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer;min-height:36px}
button:hover{border-color:var(--acc)}
button:focus-visible,input:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
button.primary{background:var(--acc);border-color:var(--acc);color:var(--card)}
button.playing{border-color:var(--play);color:var(--play);font-weight:600}
.intro{color:var(--mute);max-width:720px;margin:20px 0 8px}
.intro b{color:var(--fg)}
.group{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:36px 0 0}
section.key{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0}
section.key.done{border-color:var(--pick)}
.kh{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 12px}
.kh h2{font-size:18px;margin:0}
.kh .state{font-size:13px;color:var(--pick);font-weight:600}
.hint{color:var(--mute);margin:4px 0 12px}
.opts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(240px,100%),1fr));gap:10px}
.opt{border:1px solid var(--line);border-radius:10px;padding:12px;display:flex;flex-direction:column;gap:10px}
.opt.picked{border-color:var(--pick);background:var(--pick-bg)}
.ol{font-size:22px;font-weight:700;line-height:1}
.takes{display:flex;flex-wrap:wrap;gap:6px}
.takes button{min-width:40px;padding:6px 8px}
.pick{align-self:flex-start}
.opt.picked .pick{background:var(--pick);border-color:var(--pick);color:var(--card)}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:12px}
.row .lab{color:var(--mute);font-size:13px}
.none.on{background:var(--warn);border-color:var(--warn);color:var(--card)}
.note{width:100%;margin-top:10px;font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.rev{border-top:1px dashed var(--line);padding-top:8px;font-size:13px}
.rev .tag{display:inline-block;font-size:12px;font-weight:600;border-radius:999px;padding:1px 8px;margin-right:6px;background:var(--acc-bg);color:var(--acc)}
.rev .tag.cur{background:var(--pick-bg);color:var(--pick)}
.rev q{display:block;quotes:none;margin:4px 0;color:var(--fg)}
.rev .st,.rev .why,.mode{color:var(--mute)}
.mode{font-size:13px;margin:10px 0 0}
[hidden]{display:none!important}
h2.sec{font-size:20px;margin:48px 0 4px}
.sp{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0}
.sp h3{font-size:16px;margin:0 0 6px}
.grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr));gap:10px;margin-top:10px}
.cell{border:1px solid var(--line);border-radius:10px;padding:12px;display:flex;flex-direction:column;gap:8px}
.cell .sn{font-weight:700}
.cell .inf{font-size:13px;color:var(--mute);font-variant-numeric:tabular-nums}
.cell q{quotes:none;font-size:13px;color:var(--mute)}
footer{color:var(--mute);font-size:13px;margin-top:40px}
@media (max-width:520px){.bar h1{font-size:15px;width:100%}.kh h2{font-size:17px}section.key,.sp{padding:12px}}
</style>
</head>
<body>
<div class="bar"><div class="in"><h1>Prompt blind test</h1><span class="prog" id="prog"></span><button id="dl" class="primary">Download picks</button><button id="revall">Reveal all</button></div></div>
<main>
<p class="intro">Each sound below has <b>three prompt wordings</b>, shuffled and labelled A, B and C. One of them is the prompt the game uses today. Every option has the same four takes (take 1 of A, B and C share a seed), at the production setting and level. Pick the option that sounds <b>most like the real thing</b>, or press <b>None sound right</b>. Reveal shows which one is today's prompt and the full text of each. Picks stay in this browser; Download picks saves them as a file.</p>
<div id="keys"></div>
<h2 class="sec">Spaces (not blind)</h2>
<p class="intro">The same sound requested in four rooms: open field, small room, large hall and concrete tunnel. Each block is one prompt style. The open-field column uses that style's best measured prompt, the other three columns use the room wording written for this test, so the text around the room words can differ a little. Use the take buttons in the top row to hear one take across all four rooms in a row. Tunnel clips can hit the 1.0 s generation cap, which cuts their tails.</p>
<div id="spaces"></div>
<footer id="foot"></footer>
</main>
<script id="d" type="application/octet-stream">@@DATA@@</script>
<script>
(function(){
var D=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(document.getElementById('d').textContent.trim()),function(c){return c.charCodeAt(0)})));
var SK='sarge_prompt_blind_'+D.seed;
var st={picks:{},notes:{}};
try{var s=JSON.parse(localStorage.getItem(SK)||'null');if(s&&s.picks){st.picks=s.picks;st.notes=s.notes||{}}}catch(e){}
function save(){try{localStorage.setItem(SK,JSON.stringify(st))}catch(e){}}
var au=new Audio(),queue=[],timer=null,active=null,activeList=null;
function mark(b){if(active)active.classList.remove('playing');active=b;if(b)b.classList.add('playing')}
function stop(){clearTimeout(timer);queue=[];au.onended=null;au.pause();mark(null);activeList=null}
function next(){var it=queue.shift();if(!it){mark(null);activeList=null;return}mark(it.b);au.src='a/'+it.f;au.onended=function(){timer=setTimeout(next,it.gap)};var p=au.play();if(p&&p.catch)p.catch(function(){})}
function play(list,owner){if(activeList===owner&&owner){stop();return}stop();activeList=owner;queue=list.slice();next()}
function el(t,c,txt){var e=document.createElement(t);if(c)e.className=c;if(txt!=null)e.textContent=txt;return e}
function takeRow(files,label,gap){var w=el('div','takes');var all=el('button',null,label||'Play all');var bs=[];
 files.forEach(function(f,i){var b=el('button',null,String(i+1));b.title='Take '+(i+1);b.onclick=function(){play([{f:f,b:b,gap:0}],b)};bs.push(b)});
 all.onclick=function(){play(files.map(function(f,i){return{f:f,b:bs[i],gap:gap||350}}),all)};
 w.appendChild(all);bs.forEach(function(b){w.appendChild(b)});return w}
var keysEl=document.getElementById('keys'),lastGroup=null,secs=[];
D.sections.forEach(function(S){
 if(S.group!==lastGroup){keysEl.appendChild(el('div','group',S.group));lastGroup=S.group}
 var sec=el('section','key');var kh=el('div','kh');kh.appendChild(el('h2',null,S.title));var state=el('span','state');kh.appendChild(state);sec.appendChild(kh);
 sec.appendChild(el('p','hint',S.hint));
 var grid=el('div','opts');var cards=[];
 S.options.forEach(function(o){var c=el('div','opt');c.appendChild(el('div','ol',o.letter));c.appendChild(takeRow(o.files));
  var pb=el('button','pick','Most real');pb.onclick=function(){st.picks[S.key]=st.picks[S.key]===o.letter?null:o.letter;save();refresh()};c.appendChild(pb);
  var rv=el('div','rev');rv.hidden=true;var tg=el('span','tag'+(o.is_current?' cur':''),o.is_current?'today\'s prompt':o.style+' '+o.slot.split('_')[1]);rv.appendChild(tg);
  rv.appendChild(el('q',null,o.prompt));rv.appendChild(el('div','st',o.stats||'no measurements'));if(o.why)rv.appendChild(el('div','why','Why this option: '+o.why));
  c.appendChild(rv);cards.push({o:o,c:c,rv:rv});grid.appendChild(c)});
 sec.appendChild(grid);
 var row=el('div','row');var none=el('button','none','None sound right');none.onclick=function(){st.picks[S.key]=st.picks[S.key]==='none'?null:'none';save();refresh()};row.appendChild(none);
 var lab=el('span','lab','Same take, A then B then C:');row.appendChild(lab);
 var cmpW=el('div','takes');var ordered=cards.slice().sort(function(a,b){return a.o.letter<b.o.letter?-1:1});
 for(var t=0;t<D.seeds;t++){(function(t){var b=el('button',null,String(t+1));b.title='Take '+(t+1)+' of A, B and C';b.onclick=function(){play(ordered.map(function(x){return{f:x.o.files[t],b:b,gap:500}}),b)};cmpW.appendChild(b)})(t)}
 row.appendChild(cmpW);
 var rb=el('button',null,'Reveal');rb.onclick=function(){var h=!cards[0].rv.hidden;cards.forEach(function(x){x.rv.hidden=h});mode.hidden=h;rb.textContent=h?'Reveal':'Hide'};row.appendChild(rb);
 sec.appendChild(row);
 var mode=el('p','mode','Ranked by '+S.mode_text+'.');mode.hidden=true;sec.appendChild(mode);
 var note=el('input','note');note.type='text';note.placeholder='Optional note (what is wrong or right)';note.value=st.notes[S.key]||'';note.oninput=function(){st.notes[S.key]=note.value;save()};sec.appendChild(note);
 keysEl.appendChild(sec);
 secs.push({S:S,sec:sec,cards:cards,none:none,state:state,rb:rb,mode:mode})});
function refresh(){var n=0;secs.forEach(function(x){var p=st.picks[x.S.key];if(p)n++;x.sec.classList.toggle('done',!!p);x.none.classList.toggle('on',p==='none');
 x.cards.forEach(function(c){c.c.classList.toggle('picked',p===c.o.letter)});x.state.textContent=p==='none'?'none sound right':p?'picked '+p:''});
 document.getElementById('prog').textContent=n+' / '+secs.length+' picked'}
refresh();
document.getElementById('revall').onclick=function(){var anyHidden=secs.some(function(x){return x.cards[0].rv.hidden});secs.forEach(function(x){x.cards.forEach(function(c){c.rv.hidden=!anyHidden});x.mode.hidden=!anyHidden;x.rb.textContent=anyHidden?'Hide':'Reveal'});this.textContent=anyHidden?'Hide all':'Reveal all'};
document.getElementById('dl').onclick=function(){var out={page:'SARGE prompt blind test',made:D.made,seed:D.seed,saved:new Date().toISOString(),picks:{},summary:{current:0,library:0,descriptive:0,none:0,unpicked:0}};
 secs.forEach(function(x){var p=st.picks[x.S.key]||null;var opts={};x.S.options.forEach(function(o){opts[o.letter]={slot:o.slot,prompt:o.prompt}});
  var o=x.S.options.filter(function(o){return o.letter===p})[0];
  out.picks[x.S.key]={pick:p,slot:o?o.slot:null,is_current:o?o.is_current:null,prompt:o?o.prompt:null,note:st.notes[x.S.key]||'',options:opts};
  if(!p)out.summary.unpicked++;else if(p==='none')out.summary.none++;else if(o.is_current)out.summary.current++;else out.summary[o.style]++});
 var blob=new Blob([JSON.stringify(out,null,1)],{type:'application/json'});var a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='prompt_picks.json';document.body.appendChild(a);a.click();setTimeout(function(){URL.revokeObjectURL(a.href);a.remove()},500)};
var spEl=document.getElementById('spaces');
D.spaces.forEach(function(B){var box=el('div','sp');box.appendChild(el('h3',null,B.title+', '+B.style+' style'));
 var row=el('div','row');row.appendChild(el('span','lab','Same take in all four rooms:'));var w=el('div','takes');
 for(var t=0;t<D.seeds;t++){(function(t){var b=el('button',null,String(t+1));b.title='Take '+(t+1)+' in each room';b.onclick=function(){play(B.cells.map(function(c){return{f:c.files[t],b:b,gap:600}}),b)};w.appendChild(b)})(t)}
 row.appendChild(w);box.appendChild(row);
 var g=el('div','grid4');B.cells.forEach(function(c){var cell=el('div','cell');cell.appendChild(el('div','sn',c.name));cell.appendChild(el('div','inf',c.info));cell.appendChild(takeRow(c.files));cell.appendChild(el('q',null,c.prompt));g.appendChild(cell)});
 box.appendChild(g);spEl.appendChild(box)});
document.getElementById('foot').textContent='Built '+D.made+' · shuffle seed '+D.seed+' · '+D.sections.length+' sounds, 3 options, '+D.seeds+' takes each';
})();
</script>
</body>
</html>
"""


def parse(argv):
    ap = argparse.ArgumentParser(description="Blind listening page for the prompt search")
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None):
    return build(parse(argv))


if __name__ == "__main__":
    raise SystemExit(main())
