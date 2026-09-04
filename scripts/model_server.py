import argparse
import json
import socket
import struct
import sys
from pathlib import Path

import numpy as np


def _send(sock, obj, pcm: bytes = b""):
    body = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("<II", len(body), len(pcm)) + body + pcm)


def _recv_exactly(sock, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("client closed")
        got += k
    return bytes(buf)


def _recv(sock):
    jl, pl = struct.unpack("<II", _recv_exactly(sock, 8))
    body = json.loads(_recv_exactly(sock, jl).decode("utf-8")) if jl else {}
    pcm = _recv_exactly(sock, pl) if pl else b""
    return body, pcm


def _predictive_backend(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from sarge.generators import get_generator
    from sarge.types import CacheKey

    kw = {}
    if args.steps is not None:
        kw["steps"] = args.steps
    gen = get_generator(args.backend, **kw)
    ready = getattr(gen, "ensure_ready", None)
    if ready:
        ready()
    sr = int(getattr(gen, "sample_rate", 44_100))

    def fn(req):
        key = CacheKey(req["event_type"], req["material"],
                       req.get("intensity", "normal"), req.get("context", "default"))
        clip = gen.generate(key, variant=int(req.get("variant", 0)))
        return np.ascontiguousarray(clip.samples, dtype=np.float32), int(clip.sample_rate)

    fn({"event_type": "warmup", "material": "grass", "intensity": "normal", "variant": 0})
    return fn, sr


def make_backend(args):
    if args.backend == "meanaudio":
        from meanaudio_backend import meanaudio_backend
        return meanaudio_backend(args)
    return _predictive_backend(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    fn, sr = make_backend(args)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(8)
    port = srv.getsockname()[1]
    print(f"model-server READY backend={args.backend} sr={sr} on {args.host}:{port}", flush=True)

    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                body, _ = _recv(conn)
                t = body.get("type")
                if t == "hello":
                    _send(conn, {"ok": True, "sample_rate": sr, "backend": args.backend})
                elif t == "gen":
                    samples, s = fn(body)
                    _send(conn, {"sample_rate": s, "n": int(samples.shape[0])},
                          np.ascontiguousarray(samples, dtype="<f4").tobytes())
                else:
                    _send(conn, {"error": f"unknown type {t}"})
        except (ConnectionError, OSError):
            try:
                conn.close()
            except OSError:
                pass


if __name__ == "__main__":
    main()
