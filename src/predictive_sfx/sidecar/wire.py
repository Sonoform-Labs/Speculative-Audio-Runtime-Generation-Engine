from __future__ import annotations

import json
import socket
import struct
from enum import IntEnum

import numpy as np

MAGIC = b"PSFX"
PROTOCOL_VERSION = 1
MAX_JSON = 65536
MAX_PCM = 4194304
DEFAULT_PORT = 8765
RECV_CHUNK = 65536

HEADER = struct.Struct("<4sBBBBII")
HEADER_SIZE = HEADER.size

FLAG_HAS_PCM = 1


class MsgType(IntEnum):
    HELLO = 1
    WELCOME = 2
    CONTEXT = 3
    CONTEXT_ACK = 4
    EVENT = 5
    AUDIO = 6
    PREBAKE = 7
    PREBAKE_ACK = 8
    STATS = 9
    STATS_REPLY = 10
    PING = 11
    PONG = 12
    ERROR = 255


class ClientGone(Exception):
    pass


class IdleTimeout(Exception):
    pass


class FrameError(Exception):
    def __init__(self, code: str, message: str, fatal: bool = True):
        super().__init__(message)
        self.code = code
        self.message = message
        self.fatal = fatal


def recv_exactly(sock, n: int, idle_ok: bool = False) -> bytearray:
    buf = bytearray(n)
    if n == 0:
        return buf
    view = memoryview(buf)
    got = 0
    while got < n:
        try:
            nbytes = sock.recv_into(view[got:], min(n - got, RECV_CHUNK))
        except socket.timeout:
            if idle_ok and got == 0:
                raise IdleTimeout("idle")
            raise
        if nbytes == 0:
            raise ClientGone("peer closed")
        got += nbytes
    return buf


def pack_frame(msg_type: int, body: dict, pcm: bytes = b"") -> bytes:
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    flags = FLAG_HAS_PCM if pcm else 0
    header = HEADER.pack(MAGIC, PROTOCOL_VERSION, int(msg_type), flags, 0, len(payload), len(pcm))
    if pcm:
        return b"".join((header, payload, pcm))
    return header + payload


def read_frame(sock, max_json: int = MAX_JSON, max_pcm: int = MAX_PCM, idle_ok: bool = False):
    head = recv_exactly(sock, HEADER_SIZE, idle_ok=idle_ok)
    magic, version, msg_type, flags, _reserved, json_len, pcm_len = HEADER.unpack(head)
    if magic != MAGIC:
        raise FrameError("bad_frame", "bad magic", fatal=True)
    if version != PROTOCOL_VERSION:
        raise FrameError("bad_version", f"unsupported version {version}", fatal=True)
    if json_len > max_json:
        raise FrameError("oversized", f"json_len {json_len} > {max_json}", fatal=True)
    if pcm_len > max_pcm:
        raise FrameError("oversized", f"pcm_len {pcm_len} > {max_pcm}", fatal=True)
    raw_json = recv_exactly(sock, json_len)
    pcm = bytes(recv_exactly(sock, pcm_len)) if pcm_len else b""
    try:
        body = json.loads(bytes(raw_json).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise FrameError("bad_json", "invalid json body", fatal=False)
    return int(msg_type), body, pcm


def encode_pcm(samples: np.ndarray, fmt: str = "f32le") -> bytes:
    arr = np.ascontiguousarray(samples)
    if fmt == "f32le":
        return arr.astype("<f4", copy=False).tobytes()
    if fmt == "i16le":
        clipped = np.clip(arr, -1.0, 1.0)
        return np.round(clipped * 32767.0).astype("<i2").tobytes()
    raise ValueError(f"unknown pcm format {fmt}")


def decode_pcm(buf: bytes, fmt: str = "f32le") -> np.ndarray:
    if fmt == "f32le":
        return np.frombuffer(buf, dtype="<f4").astype(np.float32)
    if fmt == "i16le":
        return np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32767.0
    raise ValueError(f"unknown pcm format {fmt}")
