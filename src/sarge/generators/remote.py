from __future__ import annotations

import json
import socket
import struct

import numpy as np

from .base import Generator
from ..types import AudioClip, CacheKey


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
            raise ConnectionError("model server closed")
        got += k
    return bytes(buf)


def _recv(sock):
    jl, pl = struct.unpack("<II", _recv_exactly(sock, 8))
    body = json.loads(_recv_exactly(sock, jl).decode("utf-8")) if jl else {}
    pcm = _recv_exactly(sock, pl) if pl else b""
    return body, pcm


class RemoteGenerator(Generator):
    name = "remote"

    def __init__(self, host: str = "127.0.0.1", port: int = 8770,
                 connect_timeout: float = 120.0, sample_rate: int = 44_100, **_):
        self.host = host
        self.port = int(port)
        self.connect_timeout = float(connect_timeout)
        self.sample_rate = int(sample_rate)
        self.prompts: dict = {}
        self._sock = None

    def _connect(self):
        s = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = s
        body, _ = self._roundtrip({"type": "hello"})
        if body.get("sample_rate"):
            self.sample_rate = int(body["sample_rate"])

    def _ensure(self):
        if self._sock is None:
            self._connect()

    def ensure_ready(self):
        self._ensure()

    def _roundtrip(self, obj, pcm: bytes = b""):
        _send(self._sock, obj, pcm)
        return _recv(self._sock)

    def control(self, obj, reply: bool = True):
        for attempt in (0, 1):
            try:
                self._ensure()
                if not reply:
                    _send(self._sock, obj)
                    return None
                return self._roundtrip(obj)[0]
            except (ConnectionError, OSError):
                self._sock = None
                if attempt == 1:
                    raise

    def generate(self, key: CacheKey, *, variant: int = 0, prompt: str | None = None,
                 engine: str | None = None) -> AudioClip:
        req = {"type": "gen", "event_type": key.event_type, "material": key.material,
               "intensity": key.intensity, "context": key.context, "variant": int(variant)}
        if key.phys:
            req["phys"] = key.phys
        prompt = prompt or self.prompts.get(str(key))
        if prompt:
            req["prompt"] = prompt
        if engine:
            req["engine"] = engine
        for attempt in (0, 1):
            try:
                self._ensure()
                body, pcm = self._roundtrip(req)
                break
            except (ConnectionError, OSError):
                self._sock = None
                if attempt == 1:
                    raise
        if body.get("type") == "error":
            raise RuntimeError("model server: %s" % body.get("message", "error"))
        sr = int(body.get("sample_rate", self.sample_rate))
        samples = np.frombuffer(pcm, dtype="<f4").astype(np.float32)
        ch = int(body.get("channels", 1) or 1)
        if ch > 1:
            samples = samples.reshape(-1, ch)
        return AudioClip(np.ascontiguousarray(samples), sr, variant_id=variant, gen_ms=body.get("gen_ms"))

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
