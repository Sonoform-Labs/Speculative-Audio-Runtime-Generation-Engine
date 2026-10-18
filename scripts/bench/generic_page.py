import os
import sys

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import hashlib
import html
import json
import shutil

import numpy as np

from sarge.generators.routed import side_for
from sarge.prompts import SA3_PREFIX, build, generic
from sarge.types import AudioClip, CacheKey

ROOT = os.path.dirname(os.path.dirname(BENCH))
OUT = os.path.join(ROOT, "data", "_gate", "listen", "generic_ab")
ARMS = {"prompts": ("Prompts", "--prompts generic: the prompt is built from the request alone (event and material words, "
                               "the intensity and context words), SA3 gets the vendor's TrackType: SFX prefix. Same "
                               "models and amplifier as the shipped scene."),
        "amp": ("Amplifier", "--amp auto: attack emphasis, saturation and stereo width set from the clip itself, "
                             "saturation held to the body. Same prompts and models as the shipped scene."),
        "all": ("Everything", "--prompts generic --route words --amp auto together: generic prompts, guns picked by whole "
                              "words, audio-derived amplifier.")}
MODEL = {44100: "SA3 medium", 48000: "Woosh"}


def load(d):
    idx = json.load(open(os.path.join(d, "index.json"), encoding="utf-8"))
    out = {}
    for r in idx["rows"]:
        if "file" in r:
            out[json.dumps(r["key"])] = (np.load(os.path.join(d, r["file"])), int(r["sr"]))
    return out


def label(k):
    et, mat, inten, ctx = k[:4]
    parts = [et + (" on " + mat if mat else "")]
    if inten != "normal":
        parts.append(inten)
    if ctx != "default":
        parts.append(ctx)
    if len(k) > 5 and k[5]:
        parts.append("game prompt: " + k[5])
    return ", ".join(parts)


def prompt_of(arm, k, sr):
    if len(k) > 5 and k[5]:
        return k[5]
    if arm in ("head", "amp"):
        return build(*k[:4])
    return generic(*k[:4], prefix=SA3_PREFIX if sr == 44100 else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description="blind page: the shipped flags against each generic flag set")
    ap.add_argument("--head", required=True)
    ap.add_argument("--arm", action="append", default=[], help="name=dir, name one of %s" % ", ".join(ARMS))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    head = load(a.head)
    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.makedirs(os.path.join(a.out, "a"))
    secret, sections, stats = {}, [], []
    for spec in a.arm:
        name, d = spec.split("=", 1)
        title, blurb = ARMS[name]
        other = load(d)
        rows = []
        for kj in sorted(set(head) & set(other), key=lambda s: json.loads(s)):
            k = json.loads(kj)
            if len(k) > 5 and k[5]:
                continue
            rid = hashlib.md5((name + kj).encode()).hexdigest()[:12]
            pair = [("head", head[kj]), (name, other[kj])]
            if int(rid, 16) % 2:
                pair.reverse()
            files = []
            for tag, (y, sr) in pair:
                fn = hashlib.md5((rid + tag).encode()).hexdigest()[:12] + ".wav"
                AudioClip(np.ascontiguousarray(y, dtype=np.float32), sr).to_wav(os.path.join(a.out, "a", fn))
                files.append(fn)
            secret[rid] = {"b_is_new": pair[1][0] == name,
                           "a": "%s, %s: %s" % ("shipped" if pair[0][0] == "head" else "generic", MODEL.get(pair[0][1][1], "?"),
                                                prompt_of(pair[0][0], k, pair[0][1][1])),
                           "b": "%s, %s: %s" % ("shipped" if pair[1][0] == "head" else "generic", MODEL.get(pair[1][1][1], "?"),
                                                prompt_of(pair[1][0], k, pair[1][1][1]))}
            stats.append({"arm": name, "key": k, "head_model": MODEL.get(head[kj][1]), "new_model": MODEL.get(other[kj][1]),
                          "routed_differently": head[kj][1] != other[kj][1]})
            rows.append("<tr data-id='%s'><td>%s</td><td><audio controls preload='none' src='a/%s'></audio></td>"
                        "<td><audio controls preload='none' src='a/%s'></audio></td><td class='pick'>"
                        "<button data-v='a'>A</button><button data-v='same'>same</button><button data-v='b'>B</button>"
                        "</td></tr><tr class='revrow' data-for='%s'><td colspan='4' class='rev'></td></tr>"
                        % (rid, html.escape(label(k)), files[0], files[1], rid))
        sections.append("<section data-arm='%s'><h2>%s</h2><p>%s</p><table><tr><th>Sound</th><th>A</th><th>B</th>"
                        "<th>Better</th></tr>%s</table><p class='tally'></p></section>"
                        % (name, html.escape(title), html.escape(blurb), "".join(rows)))
    page = PAGE.replace("__SECTIONS__", "".join(sections)).replace("__KEY__", json.dumps(secret))
    with open(os.path.join(a.out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print("%d pairs in %d sections -> %s" % (len(secret), len(sections), os.path.join(a.out, "index.html")))
    return 0


PAGE = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Generic runtime blind test</title><style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mute:#6b6b66;--line:#ddddd8;--acc:#2b5fd9}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ececea;--mute:#9a9a95;--line:#33332f;--acc:#8fb0ff}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:24px 16px;max-width:960px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:middle}
tr.revrow td{border-bottom:1px solid var(--line);padding:0 8px 8px;font-size:13px}tr.revrow td:empty{display:none}
audio{width:200px;height:32px;max-width:40vw}button{font:inherit;padding:4px 10px;margin:0 2px;border:1px solid var(--line);
background:transparent;color:var(--fg);border-radius:6px}button.on{border-color:var(--acc);color:var(--acc);font-weight:600}
p{color:var(--mute);max-width:760px}.rev{color:var(--acc)}.tally{font-weight:600;color:var(--fg)}
section{margin:32px 0}td.pick{white-space:nowrap}
</style></head><body><h1>Generic runtime blind test</h1>
<p>Each row is one request rendered twice: once with the shipped scene's flags and once with a generic flag set.
A and B are shuffled per row. Pick the one that sounds more like the named sound, or same, then press Reveal for the
section. Reveal shows which side was which, the model that rendered it and the prompt it got. Picks stay in this browser.</p>
__SECTIONS__
<p><button id='reveal'>Reveal all</button></p>
<script>
const KEY=__KEY__;let store={};try{store=JSON.parse(localStorage.getItem('generic_ab')||'{}')}catch(e){}
function save(){try{localStorage.setItem('generic_ab',JSON.stringify(store))}catch(e){}}
document.querySelectorAll('tr[data-id]').forEach(tr=>{const id=tr.dataset.id;tr.querySelectorAll('button').forEach(b=>{
 if(store[id]===b.dataset.v)b.classList.add('on');b.onclick=()=>{store[id]=b.dataset.v;save();
 tr.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));};});});
document.getElementById('reveal').onclick=()=>{document.querySelectorAll('section').forEach(sec=>{let n=0,h=0,s=0;
 sec.querySelectorAll('tr[data-id]').forEach(tr=>{const id=tr.dataset.id,k=KEY[id];
  sec.querySelector("tr.revrow[data-for='"+id+"'] td").textContent='A = '+k.a+'  |  B = '+k.b;
  const p=store[id];if(!p)return;if(p==='same')s++;else if((p==='b')===k.b_is_new)n++;else h++;});
 sec.querySelector('.tally').textContent='generic picked '+n+', shipped picked '+h+', same '+s;});};
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
