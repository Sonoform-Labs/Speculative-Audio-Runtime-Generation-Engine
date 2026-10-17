import argparse
import json

DRIVE_MAX_MB_S = {"C:": 2000.0, "D:": 450.0, "F:": 450.0, "G:": 2000.0}
DRIVE_MAX_IOPS = {"C:": 8000.0, "D:": 2000.0, "F:": 2000.0, "G:": 8000.0}
GAP_MIN_S = 1.0
BUSY = 0.25


def interp(xs, ys, x):
    if not xs:
        return 0.0
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    f = (x - xs[lo]) / max(1e-9, xs[hi] - xs[lo])
    return ys[lo] + f * (ys[hi] - ys[lo])


class Series:
    def __init__(self, trace):
        self.rows = [r for r in trace if "disk" in r]
        self.t = [r["t"] for r in self.rows]
        self.drives = sorted({d for r in self.rows for d in r.get("disk", {})})
        self.roles = sorted({k for r in self.rows for k in r.get("proc", {})})
        self.others = sorted({k for r in self.rows for k in r.get("others", {})})
        self.cache = {}

    def delta(self, name, fn, a, b):
        ys = self.cache.get(name)
        if ys is None:
            ys = self.cache[name] = [fn(r) for r in self.rows]
        return interp(self.t, ys, b) - interp(self.t, ys, a)

    def disk_mb(self, d, a, b):
        return self.delta("rb" + d, lambda r: (r["disk"].get(d) or [0, 0, 0, 0])[0], a, b) / 1e6

    def disk_iops(self, d, a, b):
        return self.delta("rc" + d, lambda r: (r["disk"].get(d) or [0, 0, 0, 0])[2], a, b) / max(1e-9, b - a)

    def cores(self, role, a, b):
        return self.delta("cpu" + role, lambda r: (r.get("proc", {}).get(role) or [0.0, 0, 0])[0], a, b) / max(
            1e-9, b - a)

    def proc_read_mb(self, role, a, b):
        return self.delta("io" + role, lambda r: (r.get("proc", {}).get(role) or [0.0, 0, 0])[1], a, b) / 1e6

    def other_cores(self, name, a, b):
        return self.delta("ocpu" + name, lambda r: (r.get("others", {}).get(name) or [0.0, 0, 0])[0], a, b) / max(
            1e-9, b - a)

    def other_read_mb(self, name, a, b):
        return self.delta("oio" + name, lambda r: (r.get("others", {}).get(name) or [0.0, 0, 0])[1], a, b) / 1e6

    def all_cores(self, a, b):
        return self.delta("cpu_all", lambda r: r.get("cpu_all", 0.0), a, b) / max(1e-9, b - a)

    def gpu(self, a, b):
        xs = [r["gpu"] for r in self.rows if a <= r["t"] <= b and r.get("gpu") is not None]
        if not xs:
            xs = [r["gpu"] for r in self.rows if r.get("gpu") is not None and r["t"] >= a][:1]
        return sum(xs) / len(xs) if xs else 0.0


def baseline(s):
    pre = [r for r in s.rows if r["t"] < 0]
    if len(pre) < 2:
        return {d: 0.0 for d in s.drives}, 0.0
    a, b = pre[0]["t"], pre[-1]["t"]
    return {d: s.disk_mb(d, a, b) / max(1e-9, b - a) for d in s.drives}, s.gpu(a, b)


def scores(s, proc, a, b, gpu0):
    d = max(1e-9, b - a)
    out = {}
    for drv in s.drives:
        out["disk " + drv] = max(s.disk_mb(drv, a, b) / d / DRIVE_MAX_MB_S.get(drv, 500.0),
                                 s.disk_iops(drv, a, b) / DRIVE_MAX_IOPS.get(drv, 2000.0))
    out["cpu"] = s.cores(proc, a, b)
    out["gpu"] = max(0.0, s.gpu(a, b) - gpu0) / 100.0
    return out


def dominant(sc):
    best = max(sc, key=sc.get)
    return best if sc[best] >= BUSY else "wait"


def segments(res):
    off = res.get("launch_offset_s") or 0.0
    lines = sorted(([t + off, dur, proc, label] for t, dur, proc, label in res.get("stages", [])), key=lambda x: x[0])
    out = []
    prev = 0.0
    seen = {}
    for end, dur, proc, label in lines:
        n = seen.get(label, 0) + 1
        seen[label] = n
        key = label if n == 1 else "%s #%d" % (label, n)
        if dur is not None and end - dur > prev + 0.3:
            out.append({"start": prev, "end": end - dur, "proc": proc, "label": "(unlabelled before) " + key,
                        "key": "~" + key, "stage_s": None})
            prev = end - dur
        out.append({"start": prev, "end": end, "proc": proc, "label": key, "key": key, "stage_s": dur})
        prev = max(prev, end)
    return out


def annotate(segs, s, gpu0, base):
    for g in segs:
        a, b = g["start"], g["end"]
        d = max(1e-9, b - a)
        g["seg_s"] = round(b - a, 2)
        g["mb"] = {drv: round(s.disk_mb(drv, a, b), 1) for drv in s.drives}
        g["mb_net"] = {drv: round(max(0.0, s.disk_mb(drv, a, b) - base.get(drv, 0.0) * d), 1) for drv in s.drives}
        g["iops"] = {drv: round(s.disk_iops(drv, a, b)) for drv in s.drives}
        g["cores"] = {r: round(s.cores(r, a, b), 2) for r in s.roles}
        g["others_cores"] = {o: round(s.other_cores(o, a, b), 2) for o in s.others}
        g["others_readfile_mb"] = {o: round(s.other_read_mb(o, a, b), 1) for o in s.others}
        g["cores_all"] = round(s.all_cores(a, b), 2)
        g["gpu"] = round(s.gpu(a, b), 1)
        sc = scores(s, g["proc"], a, b, gpu0)
        g["dominant"] = dominant(sc) if b - a >= 0.05 else "-"
        g["scores"] = {k: round(v, 2) for k, v in sc.items()}
    return segs


def gaps(s, gpu0, until):
    out = []
    cur = None
    for i in range(1, len(s.rows)):
        a, b = s.t[i - 1], s.t[i]
        if a < 0 or a >= until:
            continue
        busy = max(max(v for k, v in scores(s, r, a, b, gpu0).items()) for r in (s.roles or ["sidecar"]))
        if busy < BUSY:
            cur = [a, b] if cur is None else [cur[0], b]
        else:
            if cur and cur[1] - cur[0] >= GAP_MIN_S:
                out.append(cur)
            cur = None
    if cur and cur[1] - cur[0] >= GAP_MIN_S:
        out.append(cur)
    return out


def report(path):
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    s = Series(res.get("trace", []))
    base, gpu0 = baseline(s)
    segs = annotate(segments(res), s, gpu0, base)
    st = res.get("startup") or {}
    end = segs[-1]["end"] if segs else 0.0
    totals = {}
    for g in segs:
        t = totals.setdefault(g["proc"], {drv: 0.0 for drv in s.drives})
        for drv, v in g["mb_net"].items():
            t[drv] += v
    by_dom = {}
    for g in segs:
        by_dom[g["dominant"]] = round(by_dom.get(g["dominant"], 0.0) + g["seg_s"], 1)
    in_gaps = []
    for a, b in gaps(s, gpu0, end):
        lab = sorted({x["label"] for x in segs if x["start"] < b and x["end"] > a})
        in_gaps.append([round(a, 1), round(b, 1), lab])
    ev = res.get("evict") or {}
    return {"path": path, "tag": res.get("tag"), "ready_s": st.get("ready_s"), "woosh_up_s": st.get("woosh_up_s"),
            "peak_used_mib": st.get("peak_used_mib"), "used_mib_at_ready": (res.get("gpu_ready") or {}).get("used_mib"),
            "evict": {k: ev.get(k) for k in ("files", "gb", "failed", "seconds")} if ev else None,
            "baseline_mb_s": {k: round(v, 2) for k, v in base.items()}, "baseline_gpu": round(gpu0, 1),
            "segments": segs, "read_mb_by_proc": {p: {k: round(v, 1) for k, v in t.items()} for p, t in totals.items()},
            "readfile_mb_by_proc": {r: round(s.proc_read_mb(r, 0.0, end), 1) for r in s.roles},
            "cpu_s_by_proc": {r: round(s.cores(r, 0.0, end) * end, 1) for r in s.roles},
            "others_cpu_s": {o: round(s.other_cores(o, 0.0, end) * end, 1) for o in s.others},
            "others_readfile_mb": {o: round(s.other_read_mb(o, 0.0, end), 1) for o in s.others},
            "seconds_by_dominant": by_dom, "idle_gaps": in_gaps, "end_s": round(end, 1)}


def show(r):
    drives = [d for d in ("F:", "D:", "C:", "G:") if any(d in g["mb"] for g in r["segments"])]
    print("== %s  READY %.1f s  woosh up %.1f s  peak %s MiB  at READY %s MiB  evict %s" % (
        r["tag"], r["ready_s"] or -1, r["woosh_up_s"] or -1, r["peak_used_mib"], r["used_mib_at_ready"], r["evict"]))
    print("%6s %5s %5s %-6s %-46s %-8s %s %s %5s %5s %5s %4s" % (
        "end_s", "seg_s", "stage", "proc", "label", "dominant", " ".join("%6s" % ("MB" + d) for d in drives),
        " ".join("%6s" % ("io" + d) for d in drives), "c_sid", "c_woo", "c_def", "gpu"))
    for g in r["segments"]:
        print("%6.1f %5.1f %5s %-6s %-46s %-8s %s %s %5.2f %5.2f %5.2f %4.0f" % (
            g["end"], g["seg_s"], "" if g["stage_s"] is None else "%.1f" % g["stage_s"], g["proc"][:6],
            g["label"][:46], g["dominant"].replace("disk ", "disk"), " ".join(
                "%6.0f" % g["mb_net"].get(d, 0.0) for d in drives), " ".join(
                "%6.0f" % g["iops"].get(d, 0.0) for d in drives), g["cores"].get("sidecar", 0.0),
            g["cores"].get("woosh", 0.0), g["others_cores"].get("MsMpEng.exe", 0.0), g["gpu"]))
    print("seconds by dominant resource:", r["seconds_by_dominant"])
    print("disk MB read while each process was the active one (net of the pre-launch rate):", r["read_mb_by_proc"])
    print("ReadFile MB by process (io_counters: cache hits included, mmap and image faults excluded):",
          r["readfile_mb_by_proc"])
    print("cpu seconds:", r["cpu_s_by_proc"], " watched:", r["others_cpu_s"], " watched ReadFile MB:",
          r["others_readfile_mb"], " pre-launch MB/s:", r["baseline_mb_s"], " gpu idle %:", r["baseline_gpu"])
    print("idle gaps >= %.0f s (every resource under %.0f%%):" % (GAP_MIN_S, BUSY * 100))
    for a, b, lab in r["idle_gaps"]:
        print("   %6.1f-%6.1f (%4.1f s) %s" % (a, b, b - a, ", ".join(lab)[:120]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="startup breakdown from e2e_hybrid.py json: stage lines x disk, cpu, gpu")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    out = [report(p) for p in a.paths]
    for r in out:
        show(r)
        print()
    if len(out) > 1:
        keys = []
        for r in out:
            for g in r["segments"]:
                if g["key"] not in keys:
                    keys.append(g["key"])
        print("%-56s %s" % ("segment seconds", " ".join("%9s" % (r["tag"] or "?")[-9:] for r in out)))
        for k in keys:
            vals = []
            for r in out:
                g = next((x for x in r["segments"] if x["key"] == k), None)
                vals.append("%9s" % ("" if g is None else "%.1f" % g["seg_s"]))
            print("%-56s %s" % (k[:56], " ".join(vals)))
        print("%-56s %s" % ("READY", " ".join("%9.1f" % (r["ready_s"] or -1) for r in out)))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
