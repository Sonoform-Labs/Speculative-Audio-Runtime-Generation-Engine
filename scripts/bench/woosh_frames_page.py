import html
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "data", "_gate", "woosh_frames")
COLS = [("dflow_101", "DFlow, 2 s render, 120 ms"), ("dflow_151", "DFlow, 3 s render, 136 ms"),
        ("dflow_501", "DFlow, 10 s render, 251 ms"), ("flow_101", "Flow, 2 s render, ~4 s"),
        ("flow_151", "Flow, 3 s render, ~4.2 s"), ("flow_501", "Flow, 10 s render, ~7 s")]
KEYS = ["break-window", "bullet_impact-glass", "footstep-water", "splash-water", "footstep-gravel", "footstep-stone"]


def main():
    rows = []
    for k in KEYS:
        cells = []
        for c, _ in COLS:
            files = [os.path.join(c, "%s_%02d.wav" % (k, v)) for v in range(3)]
            cells.append("<td>" + "".join("<audio controls preload='none' src='%s'></audio>" % f.replace(os.sep, "/")
                                          for f in files if os.path.isfile(os.path.join(BASE, f))) + "</td>")
        rows.append("<tr><th>%s</th>%s</tr>" % (html.escape(k.replace("-", " ").replace("_", " ")), "".join(cells)))
    head = "".join("<th>%s</th>" % html.escape(t) for _, t in COLS)
    page = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Woosh Lengths</title><style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mute:#6b6b66;--line:#ddddd8}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ececea;--mute:#9a9a95;--line:#33332f}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:24px 16px}
table{border-collapse:collapse}td,th{border-bottom:1px solid var(--line);padding:6px;vertical-align:top;text-align:left}
audio{display:block;width:180px;height:30px;margin:3px 0}p{color:var(--mute);max-width:760px}
</style></head><body><h1>Woosh render length</h1>
<p>Same prompts and seeds, rendered at 2, 3 and 10 seconds of audio, trimmed to the natural length, amplifier and stereo
on. Shorter renders are faster. Does anything get worse in the short ones?</p>
<table><tr><th>Sound</th>%s</tr>%s</table></body></html>""" % (head, "".join(rows))
    with open(os.path.join(BASE, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    print("wrote", os.path.join(BASE, "index.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
