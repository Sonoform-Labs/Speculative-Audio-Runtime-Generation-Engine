import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import glob
import html
import shutil

import vocab_eval as ve
from sarge.amp import amplify
from sarge.loudness import normalize_peak, peak_for
from sarge.stereo import widen
from sarge.types import AudioClip

ROOT = os.path.dirname(os.path.dirname(BENCH))
SRC = os.path.join(ROOT, "data", "_gate", "listen", "present")
OUT = os.path.join(ROOT, "data", "_gate", "listen", "stereo_ab")


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "a"))
    rows = []
    for f in sorted(glob.glob(os.path.join(SRC, "*.wav"))):
        stem = os.path.basename(f)[:-4]
        event = stem.split("-", 1)[0]
        x = amplify(ve.wav(f), ve.SR, event, "normal")
        mono = normalize_peak(x, peak_for("normal"))
        st = normalize_peak(widen(x, ve.SR, event), peak_for("normal"))
        AudioClip(mono, ve.SR).to_wav(os.path.join(OUT, "a", stem + "_mono.wav"))
        AudioClip(st, ve.SR).to_wav(os.path.join(OUT, "a", stem + "_stereo.wav"))
        rows.append("<tr><td>%s</td><td><audio controls preload='none' src='a/%s_mono.wav'></audio></td>"
                    "<td><audio controls preload='none' src='a/%s_stereo.wav'></audio></td></tr>"
                    % (html.escape(stem.replace("-", " ").replace("_", " #")), stem, stem))
    page = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Mono vs Stereo</title><style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mute:#6b6b66;--line:#ddddd8}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ececea;--mute:#9a9a95;--line:#33332f}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:24px 16px;max-width:820px}
table{border-collapse:collapse;width:100%%}td,th{border-bottom:1px solid var(--line);padding:8px;text-align:left}
audio{width:240px;height:32px}p{color:var(--mute)}
</style></head><body><h1>Mono vs stereo</h1>
<p>Use headphones. Both columns are the same generated sound with the amplifier on, at the same peak level. The left
column is mono, as the engine sent before. The right column is the new stereo output: the bass stays centred and the
upper range spreads across left and right, wider on gunshots and impacts, narrower on footsteps.</p>
<table><tr><th>Sound</th><th>Mono</th><th>Stereo</th></tr>%s</table></body></html>""" % "".join(rows)
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    print("wrote", os.path.join(OUT, "index.html"), len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
