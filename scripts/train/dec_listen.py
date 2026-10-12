import argparse
import hashlib
import html
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN_KEYS = ["shoot_pistol", "shoot_rifle", "break_window", "footstep_gravel", "footstep_stone", "footstep_wood",
            "footstep_metal", "footstep_water", "splash_water", "punch_body", "slam_door", "click_button", "bark_dog"]
REAL_BUCKETS = ["gunshot_pistol", "gunshot_rifle", "glass_ice", "impact_break", "step_gravel", "step_stone",
                "step_wood", "splash_drip"]


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variants", type=int, default=5)
    ap.add_argument("--real", type=int, default=3)
    return ap.parse_args(argv)


def code(s):
    return hashlib.md5(s.encode()).hexdigest()[:10]


def main(argv=None):
    args = parse(argv)
    src = os.path.join(ROOT, "data", "decoder_ft", "eval", args.tag, "listen")
    out = os.path.join(src, "page")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "a"))
    pairs, key = [], {}
    for k in GEN_KEYS:
        for v in range(args.variants):
            files = {n: os.path.join(src, "gen", k, "v%02d_%s.wav" % (v, n)) for n in ("base", "new")}
            if not all(os.path.isfile(f) for f in files.values()):
                continue
            pid = "%s_%d" % (k, v)
            order = ["base", "new"] if int(code(args.tag + pid), 16) % 2 == 0 else ["new", "base"]
            names = []
            for n in order:
                nm = code(args.tag + pid + n) + ".wav"
                shutil.copy(files[n], os.path.join(out, "a", nm))
                names.append(nm)
            pairs.append({"id": pid, "label": k.replace("_", " "), "a": names[0], "b": names[1]})
            key[pid] = order[1] == "new"
    real = []
    for b in REAL_BUCKETS:
        d = os.path.join(src, "real", b)
        if not os.path.isdir(d):
            continue
        stems = sorted({f.rsplit("_", 1)[0] for f in os.listdir(d) if f.endswith("_ref.wav")})[:args.real]
        for st in stems:
            row = {}
            for n in ("ref", "base", "new"):
                f = os.path.join(d, "%s_%s.wav" % (st, n))
                if os.path.isfile(f):
                    nm = "r_%s_%s_%s.wav" % (b, st, n)
                    shutil.copy(f, os.path.join(out, "a", nm))
                    row[n] = nm
            real.append({"bucket": b.replace("_", " "), "files": row})
    rows = []
    for p in pairs:
        rows.append("<tr data-id='%s'><td>%s</td><td><audio controls preload='none' src='a/%s'></audio></td>"
                    "<td><audio controls preload='none' src='a/%s'></audio></td><td class='pick'>"
                    "<button data-v='a'>A</button><button data-v='same'>same</button><button data-v='b'>B</button>"
                    "</td><td class='rev'></td></tr>" % (p["id"], html.escape(p["label"]), p["a"], p["b"]))
    rrows = []
    for r in real:
        cells = "".join("<td><audio controls preload='none' src='a/%s'></audio></td>" % r["files"][n]
                        if n in r["files"] else "<td></td>" for n in ("ref", "base", "new"))
        rrows.append("<tr><td>%s</td>%s</tr>" % (html.escape(r["bucket"]), cells))
    page = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Decoder A/B</title><style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mute:#6b6b66;--line:#ddddd8;--acc:#2b5fd9}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ececea;--mute:#9a9a95;--line:#33332f;--acc:#8fb0ff}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:24px 16px;max-width:1000px}
table{border-collapse:collapse;width:100%%;margin:12px 0 32px}td,th{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:middle}
audio{width:220px;height:32px}button{font:inherit;padding:4px 10px;margin:0 2px;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;cursor:pointer}
button.on{border-color:var(--acc);color:var(--acc);font-weight:600}p,.mute{color:var(--mute);max-width:760px}.rev{color:var(--acc);font-weight:600}
</style></head><body><h1>Decoder A/B</h1>
<p>Part 1 is blind. Each row is one sound the engine generated, decoded twice: once by the stock decoder and once by the
fine-tuned decoder (%s). A and B are shuffled per row. Pick the one that sounds more like a real recording, or "same".
Your picks save in this browser. Press Reveal when you are done.</p>
<table><tr><th>Sound</th><th>A</th><th>B</th><th>More real</th><th></th></tr>%s</table>
<button id='reveal'>Reveal</button> <button id='dl'>Download picks</button> <span id='tally' class='mute'></span>
<h2>Part 2: real recordings through each decoder</h2>
<p>Held-out real recordings the decoders never trained on. The first column is the recording; the next two are that
recording encoded and decoded by the stock and the fine-tuned decoder. Whatever the stock column loses, no generated
sound can have.</p>
<table><tr><th>Sound</th><th>Real recording</th><th>Stock decoder</th><th>Fine-tuned decoder</th></tr>%s</table>
<script>
const KEY=%s, TAG=%s;
const store=(()=>{try{return JSON.parse(localStorage.getItem('ab_'+TAG)||'{}')}catch(e){return {}}})();
function save(){try{localStorage.setItem('ab_'+TAG,JSON.stringify(store))}catch(e){}}
document.querySelectorAll('tr[data-id]').forEach(tr=>{const id=tr.dataset.id;
  tr.querySelectorAll('button').forEach(b=>{if(store[id]===b.dataset.v)b.classList.add('on');
    b.onclick=()=>{store[id]=b.dataset.v;save();tr.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));};});});
document.getElementById('reveal').onclick=()=>{let n=0,o=0,s=0;
  document.querySelectorAll('tr[data-id]').forEach(tr=>{const id=tr.dataset.id,bNew=KEY[id];
    tr.querySelector('.rev').textContent=bNew?'B = fine-tuned':'A = fine-tuned';const p=store[id];
    if(!p)return; if(p==='same')s++; else if((p==='b')===bNew)n++; else o++;});
  document.getElementById('tally').textContent='fine-tuned picked '+n+', stock picked '+o+', same '+s;};
document.getElementById('dl').onclick=()=>{const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify({tag:TAG,picks:store,key:KEY},null,1)],{type:'application/json'}));
  a.download='decoder_ab_'+TAG+'.json';a.click();};
</script></body></html>""" % (html.escape(args.tag), "".join(rows), "".join(rrows), json.dumps(key), json.dumps(args.tag))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    print("wrote %s: %d blind pairs, %d real rows" % (os.path.join(out, "index.html"), len(pairs), len(real)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
