import argparse
import json
import os
import sys

import numpy as np

from compare_clips import arr, clips, within


def compare(ref, other):
    a, b = clips(ref), clips(other)
    rows, missing, extra = [], [], []
    for k in sorted(a):
        if k not in b:
            missing.append({"key": k[0], "variant": k[1]})
            continue
        x, y = arr(ref, a[k][0]), arr(other, b[k][0])
        row = {"key": k[0], "variant": k[1], "shape_ref": list(x.shape), "shape": list(y.shape),
               "dtype_ref": str(x.dtype), "dtype": str(y.dtype), "max_abs": None}
        if x.shape == y.shape:
            row["max_abs"] = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
        row["identical"] = x.shape == y.shape and x.dtype == y.dtype and row["max_abs"] == 0.0
        rows.append(row)
    for k in sorted(set(b) - set(a)):
        extra.append({"key": k[0], "variant": k[1]})
    bad = [r for r in rows if not r["identical"]]
    w_ref, w_other = within(ref, a), within(other, b)
    ok = not missing and not bad and not w_ref and not w_other and len(rows) > 0
    return {"ref": ref, "other": other, "clips": len(rows), "identical": len(rows) - len(bad), "different": bad,
            "missing": missing, "extra": extra, "inconsistent_ref": w_ref, "inconsistent_other": w_other,
            "max_abs": max((r["max_abs"] for r in rows if r["max_abs"] is not None), default=None),
            "shapes_equal": all(r["shape_ref"] == r["shape"] for r in rows), "ok": ok, "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description="gate Q for startup changes: every (key, variant) clip in the baseline "
                                             "folder (e2e_hybrid.py --save-dir) must exist in each other folder with "
                                             "the same shape and dtype and a max abs diff of 0.0; any missing clip, "
                                             "difference or repeat that disagrees within a run fails")
    ap.add_argument("baseline")
    ap.add_argument("others", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--per-clip", action="store_true", help="print one line per clip")
    a = ap.parse_args(argv)
    out = [compare(a.baseline, o) for o in a.others]
    for r in out:
        print("%s vs %s: %d clips, %d identical, max abs %s, shapes equal %s, %d different, %d missing, %d extra, "
              "inconsistent within baseline %d / run %d -> %s" % (
                  os.path.basename(r["ref"].rstrip("\\/")), os.path.basename(r["other"].rstrip("\\/")), r["clips"],
                  r["identical"], r["max_abs"], r["shapes_equal"], len(r["different"]), len(r["missing"]),
                  len(r["extra"]), len(r["inconsistent_ref"]), len(r["inconsistent_other"]),
                  "PASS" if r["ok"] else "FAIL"), flush=True)
        if a.per_clip:
            for row in r["rows"]:
                print("    %-60s v%d %s %s max_abs %s" % (row["key"], row["variant"], row["shape_ref"], row["shape"],
                                                         row["max_abs"]), flush=True)
        for d in r["different"][:20]:
            print("    DIFFERENT", {k: d[k] for k in ("key", "variant", "shape_ref", "shape", "max_abs")}, flush=True)
        for m in r["missing"][:20]:
            print("    MISSING", m, flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    return 0 if all(r["ok"] for r in out) else 1


if __name__ == "__main__":
    sys.exit(main())
