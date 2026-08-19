import _bootstrap

import argparse
import csv
import io
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch ESC-50 real footstep clips as a FAD reference (CC BY-NC, eval-only).")
    ap.add_argument("--category", default="footsteps")
    ap.add_argument("--out", default=os.path.join("data", "real_footsteps"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    meta = urllib.request.urlopen(f"{BASE}/meta/esc50.csv", timeout=30).read().decode()
    rows = list(csv.DictReader(io.StringIO(meta)))
    files = [r["filename"] for r in rows if r["category"] == args.category]
    print(f"{len(files)} '{args.category}' clips (ESC-50, CC BY-NC, eval reference only)")
    for i, fn in enumerate(files):
        dest = os.path.join(args.out, fn)
        if not os.path.exists(dest):
            urllib.request.urlretrieve(f"{BASE}/audio/{fn}", dest)
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            print(f"  {i + 1}/{len(files)}", flush=True)
    print("done ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
