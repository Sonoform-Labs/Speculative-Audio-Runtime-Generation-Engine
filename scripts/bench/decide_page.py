import argparse
import base64
import hashlib
import html
import json
import os
import random
import re
import shutil
import sys
import time

BENCH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BENCH))
GATE = os.path.join(ROOT, "data", "_gate")
OUT = os.path.join(GATE, "decide")
CONFIRM = os.path.join(GATE, "llm", "eval", "wstyles", "confirm.json")
SOURCES = {
    "loudness": (os.path.join(GATE, "feel", "loudness_page"), None),
    "world": (os.path.join(GATE, "feel", "world_page"), os.path.join(GATE, "feel", "world_page_key.json")),
    "llm2": (os.path.join(GATE, "llm2", "page"), os.path.join(GATE, "llm2", "page_key.json")),
}
MIN_FREE = 20 * 1024 ** 3

SECTIONS = (
    ("loud", "Loudness levelling: on or off",
     "Each option is the same moment mixed the way the game would play it. One option is the current chain: peak level "
     "per intensity, amplifier with saturation, and a 3D source with a 1 m minimum distance. The others level by 400 ms "
     "loudness with the attack-only amplifier and play the far sound on a pooled 3D source with a 4, 8 or 12 m minimum "
     "distance. The gun plays in 2D at gun volume 1 in every option."),
    ("world", "World model: on or off",
     "One option builds the prompt from the event and material words. One turns the physics block into a bucket prompt "
     "with the bucket's decay cap, attack and low-pass. The third levels that same render by 400 ms loudness. Each option "
     "plays three takes, the variants the game rotates through."),
    ("prompt", "Prompt source: phrasebook, caption template or writer",
     "Three prompt sources for the same request on the same seeds and the shipped models: the tuned phrasebook, the "
     "universal caption template and the small CPU writer under its latest rules. Four takes each. The first nine rows "
     "are where the judges split, contradict the request or reward the wrong sound; the last six have no validated "
     "judge, so the ear is the only verdict."),
)

ROWS = (
    ("L1", "loud", "loudness", "m-window-5.5", "Pistol shot breaks a window 5.5 m away",
     "both sounds start in the same frame",
     "Settles levelling for glass heard right behind a gun at close range, and which minimum distance keeps the glass "
     "audible without covering the shot."),
    ("L2", "loud", "loudness", "m-window-17.8", "Pistol shot breaks a window 17.8 m away",
     "both sounds start in the same frame",
     "Settles levelling for a window far away: whether the current distance roll-off buries it, and which minimum "
     "distance still reads as far."),
    ("L3", "loud", "loudness", "m-ball-metal", "Four hard ball hits on metal, 3.9 s apart, one pistol shot between hits 2 and 3",
     "the clip the session served; the ball sits 7 m from the listener",
     "Settles levelling for the ball: the served clip as the session heard it against the same clip re-levelled for a "
     "hard hit, with a pistol shot for scale."),
    ("W1", "world", "world", "ball-metal", "Rubber ball bouncing on the metal tile (hard, 12 to 25 m/s)",
     "one collision from the demo arena, three takes per option",
     "Settles the world model on its headline case: the word builder asks for a ringing steel clang when a rubber ball "
     "meets metal, the world model asks for a rubber ball bounce and caps the decay. The third option adds loudness "
     "levelling on top of the world model."),
    ("W2", "world", "world", "crate-gravel", "Wooden crate landing on gravel (normal)",
     "one collision from the demo arena, three takes per option",
     "Settles whether the bucket's decay cap and dull-thud prompt suit a crate on gravel, against the longer hollow knock "
     "with a gravel crunch that the word builder asks for."),
    ("P1", "prompt", "llm2", "g_door_creak_wood", "Door creak", "",
     "The only valid door judge passes a knock as a door, and the phrasebook asks for a hollow knock on wood "
     "(docs/generic.md), so its top score may be the judge's blind spot. The ear decides whether a creak beats a knock."),
    ("P2", "prompt", "llm2", "g_water_drip_water", "Water drip", "",
     "Only the water judge scores this row, and it rewards any water. The phrasebook answers a drip with its big splash "
     "sentence and gets a perfect water score."),
    ("P3", "prompt", "llm2", "g_splash_water", "Water splash", "",
     "The water judge gives the phrasebook a perfect score while CLAP's top label on those clips is a magic fire spell; "
     "the writer's wet slap turns into a punch for CLAP."),
    ("P4", "prompt", "llm2", "g_footstep_wood", "Footstep on wood", "",
     "PANNs carries the phrasebook's wide lead, while CLAP's nearest label is a punch for all three sources on half the "
     "seeds or more. The ear decides which one sounds like a step on wood."),
    ("P5", "prompt", "llm2", "g_punch_body", "Punch to the body", "",
     "A demo key where PANNs fails the writer on most seeds while CLAP's nearest label for the writer's takes is still a "
     "punch. The ear settles which judge to trust."),
    ("P6", "prompt", "llm2", "f_beer_bottle_break", "Beer bottle breaking", "a request no earlier run had seen",
     "A fresh request. The writer adds a liquid splash nobody asked for; AST and PANNs side with the phrasebook's crunch "
     "of glass."),
    ("P7", "prompt", "llm2", "g_car_horn_car", "Car horn", "",
     "The phrasebook's fallback frame asks for a knock on a car and never made a horn. This row checks that the generic "
     "sources sound like a horn to a person."),
    ("P8", "prompt", "llm2", "f_woof_shepherd", "Dog woof", "a request no earlier run had seen",
     "A fresh request with no material, so the phrasebook frame reads 'on a  surface'. Caption and writer both pass "
     "the judges."),
    ("P9", "prompt", "llm2", "f_gunfire_revolver", "Revolver fired by an npc", "a request no earlier run had seen",
     "A fresh gun name on SA3 and the row where the judges split hardest: CLAP hears a grenade in most caption takes "
     "while PANNs hears a gunshot."),
    ("P10", "prompt", "llm2", "g_lightsaber_swing_plasma", "Lightsaber swing", "",
     "No validated judge exists for this sound; the page is the only verdict."),
    ("P11", "prompt", "llm2", "g_grappling_hook_fire_rope", "Grappling hook firing", "",
     "No validated judge exists for this sound; the page is the only verdict. The writer's prompt invents a burning "
     "rope, so this row also shows what an invention costs."),
    ("P12", "prompt", "llm2", "b_ball_metal", "Rubber ball bouncing on metal (hard)", "",
     "No validated judge exists for the balls. Every option goes through the same world-model chain, and the phrasebook "
     "option is the world model's bucket prompt."),
    ("P13", "prompt", "llm2", "b_ball_stone", "Rubber ball bouncing on stone (hard)", "",
     "No validated judge exists for the balls; the page is the only verdict."),
    ("P14", "prompt", "llm2", "b_ball_wood", "Rubber ball bouncing on wood (hard)", "",
     "No validated judge exists for the balls; the page is the only verdict."),
    ("P15", "prompt", "llm2", "b_ball_grass", "Rubber ball bouncing on grass (hard)", "",
     "No validated judge exists for the balls; the page is the only verdict."),
)

ARM_LABEL = {
    "H": "current chain: peak level, amplifier with saturation, 3D min distance 1 m",
    "N4": "loudness levelling, attack-only amplifier, pooled 3D min distance 4 m",
    "N8": "loudness levelling, attack-only amplifier, pooled 3D min distance 8 m",
    "N12": "loudness levelling, attack-only amplifier, pooled 3D min distance 12 m",
    "head": "word builder (HEAD)",
    "world": "--world",
    "world_loud": "--world --level loudness",
    "phrasebook": "phrasebook (shipped default)",
    "caption": "caption template (template:caption+tt)",
    "wr_named": "writer, style named (Gemma 4 E2B on the CPU)",
}
JUDGE_NAME = {"ast": "AST", "panns": "PANNs", "clap": "CLAP"}
ARM_ORDER = ("phrasebook", "caption", "wr_named")


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


def read(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def parse_loudness(d):
    t = read(os.path.join(d, "index.html"))
    m = re.search(r"const KEY=(\{.*?\});const SK=", t, re.S)
    key = json.loads(m.group(1))
    rows = {}
    for sec in re.finditer(r"<section class='row' data-id='([^']+)'>(.*?)</section>", t, re.S):
        rid, body = sec.group(1), sec.group(2)
        opts = {}
        for arm in re.finditer(r"<div class='arm' data-l='([A-Z])'>(.*?)<div class='rev'></div></div>", body, re.S):
            opts[arm.group(1)] = re.findall(r"src='a/([^']+)'", arm.group(2))
        rows[rid] = {L: {"arms": [key[rid][L]["arm"]], "files": f, "note": key[rid][L]["note"]} for L, f in opts.items()}
    return rows


def parse_snd(d, kp):
    t = read(os.path.join(d, "index.html"))
    k = json.load(open(kp, encoding="utf-8"))
    rows = {}
    for sec in re.finditer(r"<section class='snd' data-s='([^']+)'>(.*?)</section>", t, re.S):
        sid, body = sec.group(1), sec.group(2)
        facts = re.search(r"<p class='facts'>([^<]*)</p>", body)
        head = re.search(r"<h3>([^<]*)</h3>", body)
        opts = {}
        for o in re.finditer(r"<div class='opt' data-s='[^']+' data-l='([A-Z])'>(.*?)<div class='rev'></div></div>", body, re.S):
            L = o.group(1)
            arms = k["key"]["%s|%s" % (sid, L)]
            opts[L] = {"arms": arms if isinstance(arms, list) else [arms],
                       "files": re.findall(r"src='a/([^']+)'", o.group(2)),
                       "note": k["info"].get("%s|%s" % (sid, L), "")}
        rows[sid] = {"opts": opts, "request": html.unescape(head.group(1)) if head else "",
                     "facts": html.unescape(facts.group(1)) if facts else ""}
    return rows


def ci(v, c):
    return "%.2f [%.2f, %.2f]" % (v, c[0], c[1])


def judged_range(note, n):
    m = re.search(r"variants (\d+)-\d+", note)
    first = int(m.group(1)) if m else 0
    return "variants %d to %d (%d seeds)" % (first, first + n - 1, n)


def short(a):
    return ARM_LABEL.get(a, a).split(" (")[0].split(",")[0]


def mcn(cmp, a, b):
    c = cmp.get("%s vs %s" % (a, b))
    if not c:
        return ""
    parts = ["%s %+d/-%d" % (JUDGE_NAME.get(j, j), v["gain"], v["loss"]) for j, v in sorted(c.get("mcnemar", {}).items())]
    return "%s minus %s %+.2f [%+.2f, %+.2f]%s" % (short(a), short(b), c["delta"], c["ci"][0], c["ci"][1],
                                                  (" (seeds gained/lost " + ", ".join(parts) + ")") if parts else "")


def judge_text(conf, sid, arms, note):
    out = []
    for sec, name in (("per", "Category identity"), ("water", "Water judge")):
        r = conf[sec].get(sid)
        if not r:
            continue
        n = max(c["n"] for c in conf["clips"][sid].values())
        arms = [a for a in ARM_ORDER if a in arms] + [a for a in arms if a not in ARM_ORDER]
        scores = ", ".join("%s %s" % (short(a), ci(r["cons"][a], r["cons_ci"][a])) for a in arms)
        judges = "/".join(JUDGE_NAME.get(j, j) for j in sorted(r["judges"]))
        cmps = "; ".join(x for x in (mcn(r["cmp"], "caption", "phrasebook"), mcn(r["cmp"], "wr_named", "phrasebook"),
                                     mcn(r["cmp"], "wr_named", "caption")) if x)
        out.append("%s on %s, valid judges %s: %s. Paired deltas: %s." % (name, judged_range(note, n), judges, scores, cmps))
    if not out:
        out.append("No validated judge for this sound in either round.")
    return " ".join(out)


def option_note(src, conf, sid, opt):
    if src == "loudness":
        return opt["note"].split("; ", 1)[-1]
    if src != "llm2":
        return opt["note"]
    parts = opt["note"].split(" | ")
    r = conf["per"].get(sid)
    heard = ""
    if r and "identity" in r:
        labs = []
        for a in opt["arms"]:
            labs += ["%s %d" % (lab, c) for lab, c in r["identity"].get(a, [])[:2]]
        if labs:
            heard = "CLAP top labels: " + ", ".join(labs)
    if len(parts) >= 5:
        rest = parts[1:3] + ([heard] if heard else []) + parts[3:]
        return " | ".join(rest)
    return opt["note"] + (" | " + heard if heard else "")


def place(src_path, dst_path):
    try:
        os.link(src_path, dst_path)
        return "link"
    except OSError:
        if shutil.disk_usage(os.path.dirname(dst_path)).free - os.path.getsize(src_path) < MIN_FREE:
            raise SystemExit("refusing to copy %s: D: would drop under 20 GB free" % src_path)
        shutil.copy2(src_path, dst_path)
        return "copy"


def file_md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def merged_options(src, conf, sid, opts):
    groups = {}
    for L in sorted(opts):
        o = opts[L]
        paths = [os.path.join(SOURCES[src][0], "a", fn) for fn in o["files"]]
        for p in paths:
            if not os.path.isfile(p):
                raise SystemExit("missing source clip %s" % p)
        sig = tuple(file_md5(p) for p in paths)
        g = groups.setdefault(sig, {"arms": [], "notes": [], "source_letters": [], "paths": paths, "md5": list(sig)})
        g["arms"] += o["arms"]
        g["notes"].append(option_note(src, conf, sid, o))
        g["source_letters"].append(L)
    out = []
    for g in groups.values():
        if len(g["notes"]) > 1:
            print("merged %s options %s: identical audio" % (sid, "+".join(g["source_letters"])))
        g["note"] = " || ".join(dict.fromkeys(g["notes"]))
        g["sig"] = "+".join(sorted(g["arms"]))
        g["same"] = len(g["arms"]) > 1
        out.append(g)
    flat = [d for s in out for d in s["md5"]]
    if len(flat) != len(set(flat)):
        print("warning: %s repeats a take inside or across options" % sid)
    return out


def balanced_orders(seed, sec, rows_opts):
    rng = random.Random(md5(seed + "|" + sec))
    sizes = {len(v) for v in rows_opts.values()}
    sigsets = {tuple(sorted(o["sig"] for o in v)) for v in rows_opts.values()}
    orders = {}
    if len(sizes) == 1 and len(sigsets) == 1:
        canon = list(next(iter(sigsets)))
        m = len(canon)
        rids = list(rows_opts)
        for b in range(0, len(rids), m):
            block = rids[b:b + m]
            base = canon[:]
            rng.shuffle(base)
            shifts = list(range(m))
            rng.shuffle(shifts)
            for rid, k in zip(block, shifts):
                seq = base[k:] + base[:k]
                by_sig = {o["sig"]: o for o in rows_opts[rid]}
                orders[rid] = [by_sig[s] for s in seq]
    else:
        for rid, v in rows_opts.items():
            orders[rid] = sorted(v, key=lambda o: md5(seed + "|" + rid + "|" + o["sig"]))
    return orders


def build(out, seed):
    conf = json.load(open(CONFIRM, encoding="utf-8"))
    parsed = {"loudness": parse_loudness(SOURCES["loudness"][0]),
              "world": parse_snd(*SOURCES["world"]),
              "llm2": parse_snd(*SOURCES["llm2"])}
    a_dir = os.path.join(out, "a")
    if os.path.isdir(a_dir):
        shutil.rmtree(a_dir)
    os.makedirs(a_dir)
    letters = "ABCDEF"
    payload_rows, key_rows, blocks, how, per_letter = {}, [], {s[0]: [] for s in SECTIONS}, {"link": 0, "copy": 0}, {}
    merged, facts_of = {}, {}
    for rid, sec, src, sid, title, sub, why in ROWS:
        if src == "loudness":
            opts = parsed[src][sid]
            facts_of[rid] = sub
        else:
            entry = parsed[src][sid]
            opts = entry["opts"]
            facts_of[rid] = sub
            if src == "llm2":
                bits = ["request: " + entry["request"]] + ([entry["facts"]] if entry["facts"] else []) + ([sub] if sub else [])
                facts_of[rid] = "; ".join(bits)
        merged.setdefault(sec, {})[rid] = merged_options(src, conf, sid, opts)
    orders = {}
    for sec in merged:
        orders.update(balanced_orders(seed, sec, merged[sec]))
    for n_row, (rid, sec, src, sid, title, sub, why) in enumerate(ROWS, 1):
        facts = facts_of[rid]
        order = orders[rid]
        all_arms = [a for o in order for a in o["arms"]]
        row_payload = {"section": sec, "title": title, "why": why, "letters": {}}
        if src == "llm2":
            row_payload["judge"] = judge_text(conf, sid, all_arms, order[0]["note"])
        key_row = {"id": rid, "section": sec, "title": title, "decides": why, "source": {"page": src, "row": sid},
                   "letters": {}}
        opt_html = []
        for i, o in enumerate(order):
            new_l = letters[i]
            per_letter.setdefault(o["sig"], {}).setdefault(new_l, 0)
            per_letter[o["sig"]][new_l] += 1
            names, files = [], []
            for t, sp in enumerate(o["paths"]):
                nm = md5("%s|%s|%s|%d" % (seed, rid, o["sig"], t))[:12] + ".wav"
                how[place(sp, os.path.join(a_dir, nm))] += 1
                names.append(nm)
                files.append({"decide": "a/" + nm, "source": sp, "md5": o["md5"][t]})
            label = " = ".join(ARM_LABEL.get(a, a) for a in o["arms"]) + (" (the same audio)" if o["same"] else "")
            note = o["note"]
            row_payload["letters"][new_l] = {"arms": o["arms"], "label": label, "note": note}
            key_row["letters"][new_l] = {"arms": o["arms"], "label": label, "note": note,
                                         "source_letters": o["source_letters"], "files": files}
            takes = "".join("<button class='tk' data-src='a/%s' aria-label='%s take %d'>%s</button>"
                            % (nm, new_l, t + 1, "play" if len(names) == 1 else str(t + 1)) for t, nm in enumerate(names))
            if len(names) > 1:
                takes += "<button class='all' aria-label='%s all takes'>all</button>" % new_l
            opt_html.append("<div class='opt' data-s='%s' data-l='%s'><div class='let'>%s</div><div class='takes'>%s</div>"
                            "<div class='btns'><button data-v='best'>best</button><button data-v='worst'>worst</button>"
                            "</div><div class='rev'></div></div>" % (rid, new_l, new_l, takes))
        if src == "llm2":
            key_row["judge"] = row_payload["judge"]
        payload_rows[rid] = row_payload
        key_rows.append(key_row)
        blocks[sec].append("<section class='snd' data-s='%s'><div class='hd'><span class='num'>%d</span><h3>%s</h3></div>"
                           "<p class='facts'>%s</p>%s<p class='row'><button class='none' data-s='%s'>none sound right</button>"
                           "<span class='st'></span></p><div class='why'></div></section>"
                           % (rid, n_row, html.escape(title), html.escape(facts), "".join(opt_html), rid))
    body, nav = [], []
    for sec, head, blurb in SECTIONS:
        nums = [i for i, r in enumerate(ROWS, 1) if r[1] == sec]
        span = ("row %d" % nums[0] if len(nums) == 1 else "rows %d and %d" % (nums[0], nums[1]) if len(nums) == 2
                else "rows %d to %d" % (nums[0], nums[-1]))
        nav.append("<a href='#s-%s'>%s, %s</a>" % (sec, html.escape(head.split(":")[0]), span))
        body.append("<h2 id='s-%s'>%s</h2><p>%s</p>%s" % (sec, html.escape(head), html.escape(blurb), "".join(blocks[sec])))
    body.insert(0, "<nav>%s</nav>" % "".join(nav))
    data = {"seed": seed, "rows": payload_rows, "order": [r[0] for r in ROWS],
            "sections": {s[0]: s[1] for s in SECTIONS}, "labels": ARM_LABEL}
    b64 = base64.b64encode(json.dumps(data, ensure_ascii=True).encode()).decode()
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    counts = sorted({len(v) for v in orders.values()})
    span = words[counts[0]] if len(counts) == 1 else "%s to %s" % (words[counts[0]], words[counts[-1]])
    if len(counts) == 2 and counts[1] == counts[0] + 1:
        span = "%s or %s" % (words[counts[0]], words[counts[1]])
    page = (TEMPLATE.replace("%BODY%", "".join(body)).replace("%DATA%", b64).replace("%N%", str(len(ROWS)))
            .replace("%OPTS%", span)
            .replace("%STORE%", "sarge_decide_page_" + seed))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    key = {"seed": seed, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "page": os.path.join(out, "index.html"),
           "store_key": "sarge_decide_page_" + seed, "confirm_report": CONFIRM,
           "sources": {k: {"page": v[0], "key": v[1]} for k, v in SOURCES.items()},
           "letter_counts": per_letter, "rows": key_rows}
    with open(os.path.join(out, "decide_key.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(key, fh, indent=1)
    return len(ROWS), how, per_letter


TEMPLATE = """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>SARGE Decision Listening</title><style>
:root{--bg:#f7f7f4;--fg:#1c1c1a;--mute:#66665f;--line:#dcdcd5;--card:#ffffff;--acc:#2456c9;--accbg:rgba(36,86,201,.16);--bad:#b4400c;--ok:#2f7d32}
@media (prefers-color-scheme: dark){:root:not([data-theme='light']){--bg:#151514;--fg:#ececea;--mute:#9c9c96;--line:#34342f;--card:#1d1d1b;--acc:#8fb0ff;--accbg:rgba(143,176,255,.2);--bad:#fb923c;--ok:#7bc47f}}
:root[data-theme='dark']{--bg:#151514;--fg:#ececea;--mute:#9c9c96;--line:#34342f;--card:#1d1d1b;--acc:#8fb0ff;--accbg:rgba(143,176,255,.2);--bad:#fb923c;--ok:#7bc47f}
*{box-sizing:border-box}
html{background:var(--bg);overflow-x:hidden}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif;margin:0 auto;padding:24px 16px 120px;max-width:1000px;overflow-x:hidden}
h1{font-size:24px;margin:0 0 8px}
h2{font-size:19px;margin:32px 0 4px}
h3{font-size:16px;margin:0;overflow-wrap:anywhere}
p{color:var(--mute);max-width:760px;margin:6px 0}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;margin:10px 0 0;font-size:14px}
nav a{color:var(--acc)}
.snd{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:14px 0}
.hd{display:flex;gap:10px;align-items:baseline}
.num{color:var(--mute);font-size:13px;font-variant-numeric:tabular-nums;min-width:1.6rem}
.facts{font-size:13px;margin:2px 0 6px;overflow-wrap:anywhere}
.opt{display:grid;grid-template-columns:2rem minmax(0,1fr) auto;gap:6px 12px;align-items:center;padding:8px 0;border-top:1px solid var(--line)}
.let{font-weight:700;font-size:17px}
.takes{display:flex;flex-wrap:wrap;gap:6px;min-width:0}
.btns{display:flex;gap:6px}
button{font:inherit;font-size:14px;padding:5px 12px;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;cursor:pointer}
button:hover{border-color:var(--mute)}
button.tk{min-width:48px;background:linear-gradient(90deg,var(--accbg) var(--p,0%),transparent var(--p,0%))}
button.tk.heard{color:var(--mute)}
button.tk.playing,button.all.playing{border-color:var(--acc);color:var(--acc)}
button.on{border-color:var(--acc);color:var(--acc);font-weight:600}
button.on[data-v='worst'],button.none.on{border-color:var(--bad);color:var(--bad)}
.rev{grid-column:1/-1;color:var(--acc);font-size:13px;display:none;overflow-wrap:anywhere}
.why{display:none;font-size:13px;color:var(--fg);border-top:1px dashed var(--line);margin-top:10px;padding-top:8px;overflow-wrap:anywhere}
.why b{font-weight:600}
body.revealed .rev,body.revealed .why{display:block}
.row{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:10px 0 0}
.st{font-size:13px;color:var(--mute)}.st.done{color:var(--ok)}
.bar{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px 16px}
.bar div{max-width:1000px;margin:0 auto;display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center}
#prog{font-size:14px;color:var(--mute);margin-right:auto}
#tally{margin:24px 0}
#tally table{border-collapse:collapse;width:100%;font-size:14px;margin:6px 0 18px}
#tally th,#tally td{text-align:left;border-bottom:1px solid var(--line);padding:5px 8px;vertical-align:top;overflow-wrap:anywhere}
#tally td.n{text-align:right;font-variant-numeric:tabular-nums;width:4.5rem}
@media (max-width:640px){.opt{grid-template-columns:2rem minmax(0,1fr)}.btns{grid-column:2}h1{font-size:21px}#tally td.n{width:3rem}}
</style></head><body>
<h1>Decision listening</h1>
<p>This page holds %N% rows taken from the six pending blind pages, picked to settle three open decisions: loudness
levelling, the world model and the prompt source. Use headphones. Every row plays one moment through %OPTS% options, with the
letters shuffled per row. Press a take number to play it (press it again to stop) or "all" to hear every take of that
option in order; played takes turn grey.</p>
<p>Inside a row every option plays at the gain the game would give it, so a louder option is louder in the game too.
Mark the best and the worst option, or press "none sound right". Picks stay in this browser. Reveal names every option
with its judge scores and says why the row is here; Download picks saves a JSON file for the session to read.</p>
%BODY%
<section id='tally'></section>
<div class='bar'><div><span id='prog'></span><button id='reveal'>Reveal</button><button id='dl'>Download picks</button><button id='clear'>Clear picks</button></div></div>
<script>
const D=JSON.parse(atob('%DATA%'));
const SK='%STORE%',N=%N%;
let store={};
try{store=JSON.parse(localStorage.getItem(SK)||'{}')||{}}catch(e){store={}}
let shown=false;
try{shown=localStorage.getItem(SK+'_rev')==='1'}catch(e){}
function save(){try{localStorage.setItem(SK,JSON.stringify(store))}catch(e){}}
function done(s){return !!(s&&(s.none||(s.best&&s.worst)))}
const player=new Audio();player.preload='auto';
let cur=null,queue=[],raf=0,gap=0;
function tick(){if(cur&&player.duration){cur.style.setProperty('--p',Math.min(100,100*player.currentTime/player.duration)+'%')}raf=requestAnimationFrame(tick)}
function stop(){clearTimeout(gap);queue=[];player.pause();cancelAnimationFrame(raf);
 document.querySelectorAll('.playing').forEach(b=>b.classList.remove('playing'));
 if(cur){cur.style.setProperty('--p','0%')}cur=null}
function start(b){if(cur){cur.style.setProperty('--p','0%');cur.classList.remove('playing')}
 cur=b;b.classList.add('playing');player.src=b.dataset.src;player.currentTime=0;
 const p=player.play();if(p&&p.catch)p.catch(()=>{});cancelAnimationFrame(raf);raf=requestAnimationFrame(tick)}
player.addEventListener('ended',()=>{if(cur){cur.classList.add('heard');cur.classList.remove('playing');cur.style.setProperty('--p','0%')}
 if(queue.length){const nx=queue.shift();gap=setTimeout(()=>start(nx),350)}else{cancelAnimationFrame(raf);
 document.querySelectorAll('button.all.playing').forEach(b=>b.classList.remove('playing'));cur=null}});
document.querySelectorAll('button.tk').forEach(b=>b.onclick=()=>{if(cur===b&&!player.paused){stop();return}stop();start(b)});
document.querySelectorAll('button.all').forEach(b=>b.onclick=()=>{const was=b.classList.contains('playing');stop();if(was)return;
 const tk=[...b.closest('.takes').querySelectorAll('button.tk')];b.classList.add('playing');queue=tk.slice(1);start(tk[0])});
function paint(){
 document.querySelectorAll('.opt').forEach(o=>{const s=store[o.dataset.s]||{};
  o.querySelectorAll('.btns button').forEach(b=>b.classList.toggle('on',s[b.dataset.v]===o.dataset.l));});
 document.querySelectorAll('section.snd').forEach(sec=>{const s=store[sec.dataset.s]||{};
  sec.querySelector('button.none').classList.toggle('on',!!s.none);
  const st=sec.querySelector('.st');
  st.textContent=s.none?'none sound right':(s.best||s.worst)?('best '+(s.best||'-')+', worst '+(s.worst||'-')):'';
  st.classList.toggle('done',done(s));});
 const n=D.order.filter(r=>done(store[r])).length;
 document.getElementById('prog').textContent=n+' of '+N+' rows rated';
 if(shown)reveal();}
document.querySelectorAll('.btns button').forEach(b=>b.onclick=()=>{const o=b.closest('.opt'),s=o.dataset.s,l=o.dataset.l,v=b.dataset.v;
 const c=Object.assign({},store[s]||{});c.none=false;
 if(c[v]===l){delete c[v]}else{c[v]=l;const w=v==='best'?'worst':'best';if(c[w]===l)delete c[w];}
 store[s]=c;save();paint();});
document.querySelectorAll('button.none').forEach(b=>b.onclick=()=>{const s=b.dataset.s;
 store[s]=(store[s]||{}).none?{}:{none:true};save();paint();});
function arms(r,l){const x=D.rows[r]&&D.rows[r].letters[l];return x?x.arms:null}
function esc(t){return String(t).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function reveal(){
 document.body.classList.add('revealed');
 document.querySelectorAll('.opt').forEach(o=>{const x=D.rows[o.dataset.s].letters[o.dataset.l];
  o.querySelector('.rev').textContent=x.label+' | '+x.note;});
 document.querySelectorAll('section.snd').forEach(sec=>{const r=D.rows[sec.dataset.s];
  sec.querySelector('.why').innerHTML='<b>Why this row:</b> '+esc(r.why)+(r.judge?'<br><b>Judges:</b> '+esc(r.judge):'');});
 let h='<h2>Tally</h2>';
 Object.entries(D.sections).forEach(([sk,title])=>{const t={};let none=0,rated=0;
  D.order.filter(r=>D.rows[r].section===sk).forEach(r=>{const v=store[r]||{};
   Object.values(D.rows[r].letters).forEach(x=>x.arms.forEach(a=>{t[a]=t[a]||{best:0,worst:0}}));
   if(v.none){none++;rated++;return}if(v.best||v.worst)rated++;
   (arms(r,v.best)||[]).forEach(a=>t[a].best++);(arms(r,v.worst)||[]).forEach(a=>t[a].worst++);});
  h+='<h3>'+esc(title)+' ('+rated+' rows picked)</h3><table><tr><th>option</th><th>best</th><th>worst</th></tr>';
  Object.entries(t).forEach(([a,v])=>{h+='<tr><td>'+esc(D.labels[a]||a)+'</td><td class="n">'+v.best+'</td><td class="n">'+v.worst+'</td></tr>'});
  h+='<tr><td>none sound right</td><td class="n" colspan="2">'+none+'</td></tr></table>';});
 document.getElementById('tally').innerHTML=h;
 shown=true;try{localStorage.setItem(SK+'_rev','1')}catch(e){}}
document.getElementById('reveal').onclick=reveal;
document.getElementById('dl').onclick=()=>{const rows=D.order.map(r=>{const v=store[r]||{},x=D.rows[r];const L={};
  Object.entries(x.letters).forEach(([l,y])=>{L[l]=y.arms});
  return {row:r,section:x.section,title:x.title,best:v.best||null,worst:v.worst||null,none:!!v.none,
   best_arms:v.best?arms(r,v.best):null,worst_arms:v.worst?arms(r,v.worst):null,letters:L}});
 const blob=new Blob([JSON.stringify({page:'decide',seed:D.seed,saved:new Date().toISOString(),revealed:shown,
  rated:D.order.filter(r=>done(store[r])).length,rows:rows},null,1)],{type:'application/json'});
 const u=URL.createObjectURL(blob),x=document.createElement('a');x.href=u;x.download='decide_picks.json';
 document.body.appendChild(x);x.click();x.remove();setTimeout(()=>URL.revokeObjectURL(u),1000);};
document.getElementById('clear').onclick=()=>{if(!confirm('Clear every pick on this page?'))return;
 store={};shown=false;save();try{localStorage.removeItem(SK+'_rev')}catch(e){}
 document.body.classList.remove('revealed');document.getElementById('tally').innerHTML='';
 document.querySelectorAll('.rev,.why').forEach(e=>{e.textContent=''});paint();};
paint();
</script></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="one blind page from clips the pending pages already hold: loudness, world model, prompt source")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", default="decide1")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    os.makedirs(a.out, exist_ok=True)
    n, how, per_letter = build(a.out, a.seed)
    print("wrote %s (%d rows, %d clips: %d hard links, %d copies)" % (os.path.join(a.out, "index.html"), n,
                                                                    how["link"] + how["copy"], how["link"], how["copy"]))
    for arm, c in sorted(per_letter.items()):
        print("  %-12s %s" % (arm, " ".join("%s=%d" % (k, c[k]) for k in sorted(c))))
    print("key %s" % os.path.join(a.out, "decide_key.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
