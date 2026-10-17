import argparse
import ctypes
import ctypes.wintypes as w
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

BENCH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BENCH))
TORCH = "2.14.1"
DEFAULT_ROOTS = [
    r"F:\AI\cache\hf\hub\models--stabilityai--stable-audio-3-medium",
    r"F:\AI\cache\woosh\checkpoints",
    r"D:\hf_cache\models--roberta-large",
    os.path.join(ROOT, ".venv214"),
    os.path.join(ROOT, ".venv_woosh"),
    os.path.join(ROOT, ".cache", "inductor-" + TORCH),
    os.path.join(ROOT, ".cache", "triton-" + TORCH),
    os.path.join(ROOT, "src"),
    os.path.join(ROOT, "scripts"),
    os.path.join(os.path.dirname(ROOT), "sarge-research", "modal", "runpinned.py"),
    os.environ.get("SARGE_WEIGHT_CACHE") or r"G:\sarge\cache",
]
IMAGES = (".dll", ".pyd", ".exe")
GENERIC_READ, FILE_WRITE_DATA = 0x80000000, 0x2
SHARE_ALL, OPEN_EXISTING = 7, 3
NO_BUFFERING, BACKUP = 0x20000000, 0x02000000
INVALID = w.HANDLE(-1).value
BUSY = ("run_sidecar", "woosh_server")
SIDECAR_PYTHON = os.path.join(ROOT, ".venv214", "Scripts", "python.exe")
PROBE_MODULES = ("torch", "triton", "numpy", "psutil", "scipy.signal", "scipy.fft", "safetensors.torch",
                 "stable_audio_3", "stable_audio_3.factory", "stable_audio_3.loading_utils",
                 "stable_audio_3.models.conditioners", "stable_audio_3.models.transformer", "transformers",
                 "transformers.models.t5gemma.modeling_t5gemma", "sarge.sidecar", "sarge.generators.stable_audio3",
                 "sarge.generators.routed", "sarge.generators.sa3_fast", "sarge.generators.sa3_hoist",
                 "sarge.generators.remote", "sarge.prefetch", "sarge.weight_cache", "sarge.startup")
PROBE = r"""
import importlib, importlib.metadata as md, json, os, site, sys
sys.path.insert(0, sys.argv[1])
for m in sys.argv[2:]:
    try:
        importlib.import_module(m)
    except Exception:
        pass
from sarge.fastmeta import packages_distributions
us = os.path.normcase(os.path.abspath(site.getusersitepackages()))
top = set()
for name, mod in list(sys.modules.items()):
    f = getattr(mod, "__file__", None)
    if f and os.path.normcase(os.path.abspath(f)).startswith(us + os.sep):
        top.add(name.split(".")[0])
pd = packages_distributions()
want = {d.lower() for t in top for d in pd.get(t, [])}
dists, files = [], set()
for dist in md.distributions(path=[us]):
    name = (dist.metadata["Name"] or "").lower()
    if name not in want or name in dists:
        continue
    dists.append(name)
    for f in dist.files or []:
        p = os.path.abspath(str(dist.locate_file(f)))
        if os.path.normcase(p).startswith(us + os.sep) and os.path.isfile(p):
            files.add(p)
print(json.dumps({"user_site": us, "dists": sorted(dists), "files": sorted(files)}))
"""

_k = ctypes.WinDLL("kernel32", use_last_error=True)
_k.CreateFileW.restype = w.HANDLE
_k.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p, w.DWORD, w.DWORD, w.HANDLE]
_k.CloseHandle.argtypes = [w.HANDLE]
_k.CloseHandle.restype = w.BOOL


def python_roots():
    return [p for p in [sys.base_prefix] if p and os.path.isdir(p)]


def default_roots(python=True):
    return DEFAULT_ROOTS + (python_roots() if python else [])


def sarge_user_files(python=SIDECAR_PYTHON, timeout=600):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8")
    try:
        r = subprocess.run([python, "-c", PROBE, os.path.join(ROOT, "src")] + list(PROBE_MODULES), cwd=ROOT,
                           env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": repr(e)[:300], "dists": [], "files": []}


def walk(root, seen):
    real = os.path.realpath(root)
    if os.path.isfile(real):
        yield root, os.path.getsize(real)
        return
    if not os.path.isdir(real) or real in seen:
        return
    seen.add(real)
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            it = list(os.scandir(d))
        except OSError:
            continue
        for e in it:
            try:
                if e.is_dir():
                    r = os.path.realpath(e.path)
                    if r not in seen:
                        seen.add(r)
                        stack.append(e.path)
                elif e.is_file():
                    yield e.path, e.stat().st_size
            except OSError:
                continue


def evict_file(path, image_flush=True):
    access = GENERIC_READ | (FILE_WRITE_DATA if image_flush and path.lower().endswith(IMAGES) else 0)
    h = _k.CreateFileW(path, access, SHARE_ALL, None, OPEN_EXISTING, NO_BUFFERING | BACKUP, None)
    if h in (None, INVALID) and access != GENERIC_READ:
        err = ctypes.get_last_error()
        h = _k.CreateFileW(path, GENERIC_READ, SHARE_ALL, None, OPEN_EXISTING, NO_BUFFERING | BACKUP, None)
        if h not in (None, INVALID):
            _k.CloseHandle(h)
            return "image_kept:%d" % err
    if h in (None, INVALID):
        return "open:%d" % ctypes.get_last_error()
    _k.CloseHandle(h)
    return None


def busy():
    try:
        import psutil
    except ImportError:
        return []
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
        except psutil.Error:
            continue
        if "python" in (p.info["name"] or "").lower() and any(b in cl for b in BUSY):
            out.append({"pid": p.pid, "cmdline": cl[:200]})
    return out


def evict(roots=None, image_flush=True, dry=False, force=False, keep_failed=40, workers=4, user_site=True):
    roots = list(roots) if roots else default_roots()
    running = busy()
    if running and not force:
        raise SystemExit("SARGE processes are running, evict only when none runs: %s" % running)
    t0 = time.perf_counter()
    probe = sarge_user_files() if user_site else {"dists": [], "files": []}
    t_probe = time.perf_counter() - t0
    seen = set()
    per_root, per_drive, failed, todo = {}, {}, [], []
    files = nbytes = nfail = 0
    label = "user site, the %d distributions the sidecar imports from it" % len(probe["dists"])
    groups = [(root, walk(root, seen)) for root in roots]
    if probe["files"]:
        groups.append((label, ((p, os.path.getsize(p)) for p in probe["files"] if os.path.isfile(p))))
    for root, found in groups:
        r = per_root.setdefault(root, {"files": 0, "gb": 0.0, "failed": 0,
                                       "exists": root == label or os.path.exists(root)})
        for path, size in found:
            todo.append((root, path))
            files += 1
            nbytes += size
            r["files"] += 1
            r["gb"] += size / 1e9
            drive = os.path.splitdrive(os.path.realpath(path))[0].upper() or "?"
            d = per_drive.setdefault(drive, {"files": 0, "gb": 0.0})
            d["files"] += 1
            d["gb"] += size / 1e9
    t_walk = time.perf_counter() - t0 - t_probe
    if not dry:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            errs = ex.map(lambda rp: evict_file(rp[1], image_flush), todo, chunksize=256)
            for (root, path), err in zip(todo, errs):
                if err:
                    nfail += 1
                    per_root[root]["failed"] += 1
                    if len(failed) < keep_failed:
                        failed.append([path, err])
    for v in list(per_root.values()) + list(per_drive.values()):
        v["gb"] = round(v["gb"], 3)
    return {"files": files, "gb": round(nbytes / 1e9, 3), "failed": nfail, "failed_sample": failed,
            "seconds": round(time.perf_counter() - t0, 1), "walk_seconds": round(t_walk, 1),
            "probe_seconds": round(t_probe, 1), "user_site": {k: v for k, v in probe.items() if k != "files"},
            "image_flush": bool(image_flush), "dry": bool(dry), "per_drive": per_drive, "per_root": per_root,
            "running_at_start": running}


def read_rate(path, chunk=16 << 20):
    t = time.perf_counter()
    n = 0
    with open(path, "rb", buffering=0) as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            n += len(b)
    dt = time.perf_counter() - t
    return {"mb": round(n / 1e6, 1), "s": round(dt, 3), "mb_s": round(n / 1e6 / dt, 1)}


def prove(paths, image_flush=True):
    out = []
    for p in paths:
        read_rate(p)
        hot = read_rate(p)
        err = evict_file(p, image_flush)
        cold = read_rate(p)
        err2 = evict_file(p, image_flush)
        out.append({"path": p, "hot": hot, "cold": cold, "evict_error": err or err2})
        print("%-80s hot %8.0f MB/s  after evict %8.0f MB/s" % (p[-80:], hot["mb_s"], cold["mb_s"]), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="drop SARGE's files from the Windows file cache (no admin): every file "
                                             "under the roots is opened unbuffered, which purges its cached pages; "
                                             ".dll/.pyd/.exe are also opened with write access (nothing is written) "
                                             "so a cached image section is flushed too")
    ap.add_argument("--roots", nargs="*", default=None, help="replace the default roots")
    ap.add_argument("--add", nargs="*", default=[], help="extra roots on top of the defaults")
    ap.add_argument("--no-python", action="store_true",
                    help="leave the base python and the user site-packages alone (by default the base python and, "
                         "from the shared user site, only the files of the distributions the sidecar imports, found "
                         "by importing the sidecar's modules in a probe process)")
    ap.add_argument("--no-image-flush", action="store_true",
                    help="unbuffered open only: measured to leave .dll/.pyd image pages cached (import torch stays hot)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry", action="store_true", help="count files and sizes only")
    ap.add_argument("--force", action="store_true", help="run even while a SARGE python process runs")
    ap.add_argument("--prove", nargs="*", default=None, help="time a full read of these files hot, evict, read again")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.prove:
        res = {"prove": prove(a.prove, not a.no_image_flush)}
    else:
        roots = (a.roots if a.roots is not None else default_roots(not a.no_python)) + list(a.add)
        res = evict(roots, image_flush=not a.no_image_flush, dry=a.dry, force=a.force, workers=a.workers,
                    user_site=not a.no_python and a.roots is None)
        for root, v in res["per_root"].items():
            print("%-100s %7d files %8.3f GB %5d failed" % (root[-100:], v["files"], v["gb"], v["failed"]), flush=True)
        print("evicted %d files, %.2f GB, %d failed, %.1f s" % (res["files"], res["gb"], res["failed"],
                                                                 res["seconds"]), flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_root", "failed_sample")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
