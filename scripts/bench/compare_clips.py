import argparse
import json
import os
import sys

import numpy as np


def clips(d):
    with open(os.path.join(d, "index.json"), encoding="utf-8") as fh:
        idx = json.load(fh)["clips"]
    out = {}
    for c in idx:
        out.setdefault((c["key"], int(c["variant"])), []).append(c)
    return out


def arr(d, c):
    return np.load(os.path.join(d, c["file"]))


def within(d, cl):
    bad = []
    for (key, v), cs in cl.items():
        if len({c["sha1"] for c in cs}) > 1:
            bad.append({"key": key, "variant": v, "files": [c["file"] for c in cs]})
    return bad


def compare(ref, other):
    a, b = clips(ref), clips(other)
    common = sorted(set(a) & set(b))
    diffs = []
    same = 0
    for k in common:
        x, y = arr(ref, a[k][0]), arr(other, b[k][0])
        if x.shape != y.shape:
            diffs.append({"key": k[0], "variant": k[1], "shape": [list(x.shape), list(y.shape)], "max_abs": None})
            continue
        m = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
        if m == 0.0 and x.dtype == y.dtype:
            same += 1
        else:
            diffs.append({"key": k[0], "variant": k[1], "shape": list(x.shape), "max_abs": m})
    return {"ref": ref, "other": other, "common": len(common), "identical": same, "different": diffs,
            "only_ref": sorted("%s v%d" % k for k in set(a) - set(b)),
            "only_other": sorted("%s v%d" % k for k in set(b) - set(a)),
            "within_ref": within(ref, a), "within_other": within(other, b),
            "variants": {"v%d" % v: sum(1 for k in common if k[1] == v) for v in sorted({k[1] for k in common})}}


def main(argv=None):
    ap = argparse.ArgumentParser(description="bit-exact check of clips saved by e2e_hybrid.py --save-dir: matches clips "
                                             "by (event|material|intensity|context, variant)")
    ap.add_argument("ref")
    ap.add_argument("others", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    out = [compare(a.ref, o) for o in a.others]
    ok = True
    for r in out:
        good = not r["different"] and not r["within_ref"] and not r["within_other"]
        ok = ok and good
        print("%s vs %s: %d common (by variant %s), %d identical, %d different, only ref %d, only other %d, "
              "inconsistent within ref %d / other %d -> %s" % (
                  os.path.basename(r["ref"].rstrip("\\/")), os.path.basename(r["other"].rstrip("\\/")), r["common"],
                  r["variants"], r["identical"], len(r["different"]), len(r["only_ref"]), len(r["only_other"]),
                  len(r["within_ref"]), len(r["within_other"]), "BIT-IDENTICAL" if good else "DIFFERENT"), flush=True)
        for d in r["different"][:20]:
            print("   ", d, flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
