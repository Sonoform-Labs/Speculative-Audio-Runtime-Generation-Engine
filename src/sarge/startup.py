from __future__ import annotations

import contextlib
import os
import sys
import time

_STATE = {"t0": None, "on": True}


def _created() -> float:
    if os.name != "nt":
        return time.time()
    try:
        import ctypes
        from ctypes import wintypes as w
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = w.HANDLE
        k.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
        k.GetProcessTimes.restype = w.BOOL
        ft = [w.FILETIME() for _ in range(4)]
        if not k.GetProcessTimes(k.GetCurrentProcess(), *[ctypes.byref(f) for f in ft]):
            return time.time()
        return ((ft[0].dwHighDateTime << 32) | ft[0].dwLowDateTime) / 1e7 - 11644473600.0
    except Exception:
        return time.time()


def _launched() -> float:
    t = _created()
    try:
        import psutil
        p = psutil.Process().parent()
        if p is not None and os.path.normcase(p.exe()) == os.path.normcase(sys.executable):
            t = min(t, p.create_time())
    except Exception:
        pass
    return t


def root() -> float:
    _STATE["t0"] = _launched()
    _STATE["on"] = True
    return _STATE["t0"]


def t0() -> float:
    if _STATE["t0"] is None:
        try:
            _STATE["t0"] = float(os.environ["SARGE_T0"])
        except (KeyError, ValueError):
            _STATE["t0"] = _launched()
    return _STATE["t0"]


def active() -> bool:
    return _STATE["on"]


def stage(msg: str, since: float | None = None, at: float | None = None) -> float:
    now = time.time() if at is None else at
    if _STATE["on"]:
        tail = "" if since is None else " (%.1f s)" % (now - since)
        print("[startup] %.1f s %s%s" % (now - t0(), msg, tail), flush=True)
    return now


@contextlib.contextmanager
def timed(msg: str):
    t = time.time()
    yield
    stage(msg, t)


def done(msg: str = "READY"):
    stage(msg)
    _STATE["on"] = False
