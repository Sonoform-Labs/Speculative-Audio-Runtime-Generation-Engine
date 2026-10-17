from __future__ import annotations

import ast
import glob
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import shutil
import struct
import threading
import time

from .startup import stage

SCHEMA = 2
DEFAULT_DIR = r"G:\sarge\cache"
KEEP = 2
PART = ".part-"
STALE_TMP_S = 3600.0
GRACE_S = 60.0
RESERVE = 2 << 30
CHUNK = 1 << 23
OFF = ("", "0", "off", "none", "no")
HERE = os.path.dirname(os.path.abspath(__file__))
LOADER = ("_RANDOM_INITS", "_skip_random_init", "StableAudio3Generator._cached_model")


def default_dir(env=None) -> str | None:
    env = env if env is not None else os.environ
    v = env.get("SARGE_WEIGHT_CACHE")
    if v is None:
        return DEFAULT_DIR if os.path.isdir(os.path.dirname(DEFAULT_DIR)) else None
    return None if str(v).strip().lower() in OFF else v


def _version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "none"


def _digest(paths) -> str:
    h = hashlib.sha1()
    for p in paths:
        with open(p, "rb") as fh:
            h.update(os.path.basename(p).encode("utf-8") + b"\0" + fh.read().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def package_sources(name: str) -> str:
    spec = importlib.util.find_spec(name)
    roots = list(spec.submodule_search_locations or []) if spec else []
    if not roots:
        return "missing"
    files = []
    for d, _dirs, fs in os.walk(roots[0]):
        files += [os.path.join(d, f) for f in fs if f.endswith(".py")]
    return _digest(sorted(files, key=lambda p: os.path.relpath(p, roots[0]).replace("\\", "/")))


def segments(path: str, names) -> str:
    with open(path, encoding="utf-8") as fh:
        src = fh.read().replace("\r\n", "\n")
    found = {}

    def visit(body, prefix):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found[prefix + n.name] = ast.get_source_segment(src, n)
                if isinstance(n, ast.ClassDef):
                    visit(n.body, prefix + n.name + ".")
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        found[prefix + t.id] = ast.get_source_segment(src, n)

    visit(ast.parse(src).body, "")
    return "\n".join("%s=%s" % (k, found.get(k)) for k in names)


def loader_code() -> str:
    try:
        seg = segments(os.path.join(HERE, "generators", "stable_audio3.py"), LOADER)
    except (OSError, SyntaxError, ValueError):
        seg = "missing"
    return "%s:%s" % (_digest([os.path.abspath(__file__), os.path.join(HERE, "fastmeta.py")]),
                      hashlib.sha1(seg.encode("utf-8")).hexdigest())


def fingerprint(model_name: str, ckpt: str, config: str) -> dict:
    ckpt = os.path.realpath(ckpt)
    st = os.stat(ckpt)
    with open(ckpt, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        head = hashlib.sha1(fh.read(n)).hexdigest()
    with open(config, "rb") as fh:
        cfg = hashlib.sha1(fh.read()).hexdigest()
    return {"schema": SCHEMA, "model": model_name, "dtype": "float16", "ckpt_bytes": st.st_size,
            "ckpt_mtime_ns": st.st_mtime_ns, "ckpt_header": head, "config": cfg, "torch": _version("torch"),
            "safetensors": _version("safetensors"), "stable_audio_3": _version("stable_audio_3"),
            "stable_audio_3_src": package_sources("stable_audio_3"), "code": loader_code()}


def key_of(fp: dict) -> str:
    return hashlib.sha1(json.dumps(fp, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def path_for(cache_dir: str, model_name: str, key: str) -> str:
    return os.path.join(cache_dir, "sa3-%s-fp16-%s.safetensors" % (model_name, key))


def locate(model_name: str, cache_dir: str | None, hub: str | None = None) -> str | None:
    if not cache_dir:
        return None
    from .prefetch import REPOS, snapshot
    repo = REPOS.get(model_name)
    snap = snapshot(repo, hub) if repo else None
    if snap is None:
        return None
    try:
        p = path_for(cache_dir, model_name, key_of(fingerprint(model_name, os.path.join(snap, "model.safetensors"),
                                                                os.path.join(snap, "model_config.json"))))
    except (OSError, ValueError, struct.error):
        return None
    return p if os.path.isfile(p) else None


def _words(t):
    import torch
    flat = t.detach().reshape(-1)
    if flat.dtype == torch.bool:
        flat = flat.view(torch.uint8)
    return flat.view(torch.int16) if flat.element_size() == 2 else flat.view(torch.uint8)


def checksums(sd: dict) -> dict:
    import torch
    names, parts = [], []
    with torch.no_grad():
        for k, v in sd.items():
            w = _words(v)
            s1 = torch.zeros((), dtype=torch.int64, device=w.device)
            s2 = torch.zeros((), dtype=torch.int64, device=w.device)
            for o in range(0, w.numel(), CHUNK):
                c = w[o:o + CHUNK].to(torch.int64)
                s1 += c.sum()
                s2 += c.mul_(c).sum()
            names.append(k)
            parts += [s1, s2]
        vals = torch.stack([p.to(parts[0].device) for p in parts]).tolist() if parts else []
    return {k: "%d:%d" % (vals[2 * i], vals[2 * i + 1]) for i, k in enumerate(names)}


def hooked(model) -> list:
    import torch
    base = torch.nn.Module._load_from_state_dict
    return [n or type(m).__name__ for n, m in model.named_modules()
            if m._load_state_dict_pre_hooks or m._load_state_dict_post_hooks or type(m)._load_from_state_dict is not base]


def restore(model, path: str, key: str, device):
    import torch
    from safetensors import safe_open
    bad = hooked(model)
    if bad:
        raise ValueError("load_state_dict hooks or overrides on %s" % (bad[:3],))
    want = model.state_dict(keep_vars=True)
    ids = {id(v) for v in want.values()}
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        if meta.get("key") != key:
            raise ValueError("cache key %s, wanted %s" % (meta.get("key"), key))
        try:
            sums = json.loads(meta["sums"])
        except (KeyError, ValueError):
            raise ValueError("cache holds no checksums") from None
        names = list(f.keys())
        if set(names) != set(want):
            raise ValueError("cache holds %d tensors, the model %d (%d differ)" % (
                len(names), len(want), len(set(names) ^ set(want))))
        for k in names:
            if list(f.get_slice(k).get_shape()) != list(want[k].shape):
                raise ValueError("shape of %s" % k)

        def move(t):
            if id(t) in ids and t.is_floating_point():
                return torch.empty_like(t, device=device, dtype=torch.float16)
            out = t.to(device)
            return out.to(torch.float16) if out.is_floating_point() else out

        model._apply(move)
        sd = model.state_dict()
        for k in names:
            v = f.get_tensor(k)
            if v.dtype != sd[k].dtype or v.shape != sd[k].shape:
                raise ValueError("dtype or shape of %s: %s %s vs %s %s" % (k, v.dtype, tuple(v.shape), sd[k].dtype,
                                                                          tuple(sd[k].shape)))
            sd[k].copy_(v)
    t = time.time()
    got = checksums(model.state_dict())
    wrong = sorted(k for k in names if got.get(k) != sums.get(k))
    if wrong:
        raise ValueError("checksum of %d tensors differs, first %s" % (len(wrong), wrong[0]))
    stage("weight cache: %d checksums match" % len(names), t)
    return model


def _alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True
        return True
    import ctypes
    from ctypes import wintypes as w
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    k.OpenProcess.restype = w.HANDLE
    k.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
    k.GetExitCodeProcess.restype = w.BOOL
    k.CloseHandle.argtypes = [w.HANDLE]
    k.CloseHandle.restype = w.BOOL
    h = k.OpenProcess(0x1000, False, pid)
    if not h:
        return ctypes.get_last_error() == 5
    try:
        code = w.DWORD()
        return not k.GetExitCodeProcess(h, ctypes.byref(code)) or code.value == 259
    finally:
        k.CloseHandle(h)


def _pid_of(name: str):
    if name.startswith(PART):
        tail = name[len(PART):]
    elif name.endswith(".tmp"):
        tail = name[:-4].rsplit(".", 1)[-1]
    else:
        return None
    return int(tail) if tail.isdigit() else None


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def leftover(name: str) -> bool:
    return name.startswith(PART) or name.startswith(".tmp") or name.endswith(".tmp")


def sweep(cache_dir: str, now: float | None = None) -> list:
    now = time.time() if now is None else now
    gone = []
    try:
        names = os.listdir(cache_dir)
    except OSError:
        return gone
    for n in names:
        if not leftover(n):
            continue
        p = os.path.join(cache_dir, n)
        age = now - _mtime(p)
        pid = _pid_of(n)
        if age <= STALE_TMP_S and (_alive(pid) if pid is not None else age <= GRACE_S):
            continue
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                os.remove(p)
            except OSError:
                pass
        if not os.path.exists(p):
            gone.append(p)
    return gone


def prune(cache_dir: str, model_name: str, keep: str, now: float | None = None, n: int = KEEP) -> list:
    gone = sweep(cache_dir, now)
    keep = os.path.normcase(os.path.abspath(keep))
    caches = []
    for p in glob.glob(os.path.join(cache_dir, "sa3-%s-fp16-*.safetensors" % model_name)):
        try:
            caches.append((os.path.normcase(os.path.abspath(p)) == keep, os.path.getmtime(p), p))
        except OSError:
            pass
    caches.sort(reverse=True)
    for _mine, _t, p in caches[max(1, int(n)):]:
        try:
            os.remove(p)
            gone.append(p)
        except OSError:
            pass
    return gone


def touch(path: str):
    try:
        os.utime(path, None)
    except OSError:
        pass


def prepare(cache_dir: str, need: int) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    sweep(cache_dir)
    free = shutil.disk_usage(cache_dir).free
    if free < need + RESERVE:
        raise OSError("not enough free space in %s for %.1f GB" % (cache_dir, need / 1e9))
    part = os.path.join(cache_dir, PART + str(os.getpid()))
    os.makedirs(part, exist_ok=True)
    return part


def snapshot_state(model) -> dict:
    return {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}


def _flush(path: str):
    fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(sd: dict, path: str, key: str, meta: dict | None = None, model_name: str | None = None,
          part: str | None = None) -> str:
    from safetensors.torch import save_file
    d = os.path.dirname(path)
    if part is None:
        part = prepare(d, sum(v.numel() * v.element_size() for v in sd.values()))
    tmp = os.path.join(part, os.path.basename(path))
    try:
        meta = dict(meta or {}, key=key)
        if "sums" not in meta:
            meta["sums"] = json.dumps(checksums(sd), sort_keys=True)
        save_file(sd, tmp, metadata={k: str(v) for k, v in meta.items()})
        _flush(tmp)
        os.replace(tmp, path)
    finally:
        shutil.rmtree(part, ignore_errors=True)
    if model_name:
        prune(d, model_name, path)
    return path


def save_async(model, path: str, key: str, meta: dict | None = None, model_name: str | None = None):
    t = time.time()
    state = model.state_dict()
    part = prepare(os.path.dirname(path), sum(v.numel() * v.element_size() for v in state.values()))
    try:
        sums = json.dumps(checksums(state), sort_keys=True)
        sd = snapshot_state(model)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    del state
    t = stage("weight cache: state copied to RAM for %s" % os.path.basename(path), t)
    meta = dict(meta or {}, sums=sums)

    def run():
        try:
            write(sd, path, key, meta, model_name, part)
            stage("weight cache: written %s (%.1f GB)" % (path, os.path.getsize(path) / 1e9), t)
        except Exception as e:
            stage("weight cache: not written, %s" % (repr(e)[:200],), t)
        finally:
            sd.clear()

    th = threading.Thread(target=run, name="weight-cache-write")
    th.start()
    return th
