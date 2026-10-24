import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, "upgrade_gate.py")
SIDES = ("guns", "rest")
BOUND_MS = 5.0
DRAWS = 4000
SEED = 20261024
MIN_KEYS = {"guns": 12, "rest": 24}


def parse_arms(items):
    out = []
    for it in items:
        name, _, flags = str(it).partition("=")
        name = name.strip()
        if not name or any(n == name for n, _f in out):
            raise SystemExit("--arm needs NAME=FLAGS with a new name, got %r" % it)
        out.append((name, flags.split()))
    if len(out) < 2:
        raise SystemExit("give the base arm first and at least one more --arm")
    return out


def schedule(names, reps):
    out = []
    for r in range(int(reps)):
        seq = list(names) if r % 2 == 0 else list(names)[::-1]
        out.extend((r, n) for n in seq)
    return out


def run(a):
    arms = parse_arms(a.arm)
    flags = dict(arms)
    os.makedirs(a.out, exist_ok=True)
    plan = {"name": a.name, "arms": [[n, f] for n, f in arms], "reps": a.reps, "base": arms[0][0],
            "schedule": [], "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for r, name in schedule([n for n, _f in arms], a.reps):
        tag = "%s_%s_r%d" % (a.name, name, r)
        cmd = [sys.executable, "-u", GATE, "--tag", tag, "--out", a.out, "--port", str(a.port), "--woosh-port",
               str(a.woosh_port), "--fill", "wide", "--fill-only", "--extra"] + flags[name] + [
            "--cache-dir", os.path.join(a.cache, tag)]
        t = time.time()
        with open(os.path.join(a.out, "run_%s.log" % tag), "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        plan["schedule"].append({"rep": r, "arm": name, "tag": tag, "rc": rc, "s": time.time() - t})
        with open(os.path.join(a.out, "plan_%s.json" % a.name), "w", encoding="utf-8") as fh:
            json.dump(plan, fh, indent=1)
        print("%s rep %d %s rc %d %.0f s" % (a.name, r, name, rc, time.time() - t), flush=True)
        time.sleep(a.pause_s)
    return 0


def fills(gate):
    out = {s: {} for s in SIDES}
    for r in gate.get("rows") or []:
        if r.get("pass") == "fill" and r.get("served") == "realtime" and r.get("side") in out:
            out[r["side"]][r["key"]] = float(r["ms"])
    return out


def table(gates, arms, reps, side):
    keys = sorted({k for g in gates.values() for k in g[side]})
    x = np.full((len(arms), reps, len(keys)), np.nan)
    for (arm, r), g in gates.items():
        for j, k in enumerate(keys):
            if k in g[side]:
                x[arms.index(arm), r, j] = g[side][k]
    return keys, x


def nanmed(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def judge_side(x, base, bound, draws=DRAWS, seed=SEED):
    n_arm, reps, n_keys = x.shape
    rng = np.random.default_rng(seed)
    pts = [nanmed(x[i]) for i in range(n_arm)]
    rb = rng.integers(0, reps, size=(draws, reps))
    kb = rng.integers(0, n_keys, size=(draws, n_keys))
    meds = np.empty((draws, n_arm))
    for d in range(draws):
        sub = x[:, rb[d]][:, :, kb[d]]
        meds[d] = [nanmed(sub[i]) for i in range(n_arm)]
    out = []
    for i in range(n_arm):
        lo, hi = np.nanpercentile(meds[:, i], [2.5, 97.5])
        rec = {"median_ms": pts[i], "ci": [float(lo), float(hi)], "n": int(np.isfinite(x[i]).sum())}
        if i != base:
            d = meds[:, i] - meds[:, base]
            dlo, dhi = np.nanpercentile(d, [2.5, 97.5])
            rec.update({"minus_base_ms": pts[i] - pts[base], "minus_base_ci": [float(dlo), float(dhi)],
                        "bound_ms": bound, "verdict": "FAIL" if dlo > bound else "PASS"})
        out.append(rec)
    return out


def load_runs(out_dir, name, plan=None):
    gates = {}
    if plan is not None:
        for s in plan["schedule"]:
            p = os.path.join(out_dir, "gate_%s.json" % s["tag"])
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as fh:
                    gates[(s["arm"], s["rep"])] = fills(json.load(fh))
        return gates
    for p in sorted(glob.glob(os.path.join(out_dir, "gate_%s_*_r*.json" % name))):
        tag = os.path.basename(p)[len("gate_%s_" % name):-len(".json")]
        arm, _, r = tag.rpartition("_r")
        with open(p, encoding="utf-8") as fh:
            gates[(arm, int(r))] = fills(json.load(fh))
    return gates


def judge(a):
    pp = os.path.join(a.out, "plan_%s.json" % a.name)
    plan = None
    if os.path.isfile(pp):
        with open(pp, encoding="utf-8") as fh:
            plan = json.load(fh)
    gates = load_runs(a.out, a.name, plan)
    if not gates:
        raise SystemExit("no gate files for %s in %s" % (a.name, a.out))
    arms = [n for n, _f in plan["arms"]] if plan else sorted({k[0] for k in gates}, key=lambda n: n != a.base)
    base = a.base or (plan["base"] if plan else arms[0])
    reps = max(r for _a, r in gates) + 1
    res = {"name": a.name, "base": base, "arms": arms, "reps": reps, "bound_ms": a.bound, "draws": DRAWS,
           "seed": SEED, "rule": "an arm fails a side only when the 2.5th percentile of its first-hit fill median "
                                 "minus the base arm's (bootstrap over runs and keys, paired) exceeds the bound",
           "runs": sorted("%s r%d" % k for k in gates), "sides": {}}
    lines = ["FILL GATE %s: first-hit fill medians over interleaved runs, base %s, bound +%.1f ms judged on the "
             "lower end of the 95%% interval" % (a.name, base, a.bound)]
    for side in SIDES:
        keys, x = table(gates, arms, reps, side)
        recs = judge_side(x, arms.index(base), a.bound)
        res["sides"][side] = {"keys": keys, "n_keys": len(keys), "enough_keys": len(keys) >= MIN_KEYS[side],
                              "arms": dict(zip(arms, recs))}
        lines.append("%s: %d keys (need %d) x %d runs per arm" % (side, len(keys), MIN_KEYS[side], reps))
        for arm, rec in zip(arms, recs):
            lines.append("  %-10s median %.2f ms [%.2f, %.2f] n %d%s" % (
                arm, rec["median_ms"], rec["ci"][0], rec["ci"][1], rec["n"],
                "" if arm == base else "; minus %s %+.2f ms [%+.2f, %+.2f] -> %s" % (
                    base, rec["minus_base_ms"], rec["minus_base_ci"][0], rec["minus_base_ci"][1], rec["verdict"])))
    res["verdict"] = {arm: ("FAIL" if any(res["sides"][s]["arms"][arm]["verdict"] == "FAIL" for s in SIDES)
                            else "PASS") for arm in arms if arm != base}
    res["enough_keys"] = all(res["sides"][s]["enough_keys"] for s in SIDES)
    lines.append("verdict: %s; key counts met: %s" % (res["verdict"], res["enough_keys"]))
    with open(os.path.join(a.out, "fill_gate_%s.json" % a.name), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(a.out, "fill_gate_%s.txt" % a.name), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="first-hit fill latency gate over interleaved upgrade_gate runs "
                                             "(--fill wide --fill-only): 12 gun and 24 Woosh keys a run, the arms "
                                             "alternate in ABBA order, medians with bootstrap intervals over runs "
                                             "and keys, and an arm fails a side only when the lower end of its "
                                             "difference from the base exceeds the bound")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--arm", action="append", default=[],
                   help="NAME=FLAGS, the base arm first; FLAGS are run_sidecar flags separated by spaces")
    r.add_argument("--reps", type=int, default=4)
    r.add_argument("--out", required=True)
    r.add_argument("--cache", required=True)
    r.add_argument("--port", type=int, default=8768)
    r.add_argument("--woosh-port", type=int, default=8773)
    r.add_argument("--pause-s", type=float, default=10.0)
    j = sub.add_parser("judge")
    j.add_argument("--name", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--base", default=None)
    j.add_argument("--bound", type=float, default=BOUND_MS)
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else judge(a)


if __name__ == "__main__":
    raise SystemExit(main())
