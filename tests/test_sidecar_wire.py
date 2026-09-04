import socket
import struct

import numpy as np
import pytest

from sarge.sidecar import wire
from sarge.sidecar.wire import ClientGone, FrameError, MsgType


class _DripSock:
    def __init__(self, data, chunk=1):
        self.data = bytes(data)
        self.pos = 0
        self.chunk = chunk

    def recv_into(self, view, nbytes=None):
        if self.pos >= len(self.data):
            return 0
        n = min(self.chunk, len(view), len(self.data) - self.pos)
        view[:n] = self.data[self.pos:self.pos + n]
        self.pos += n
        return n


def test_header_byte_layout():
    body = {"type": "event", "event_type": "footstep", "material": "grass"}
    frame = wire.pack_frame(MsgType.EVENT, body)
    assert frame[0:4] == b"PSFX"
    assert frame[4] == wire.PROTOCOL_VERSION
    assert frame[5] == int(MsgType.EVENT)
    assert frame[6] == 0
    assert frame[7] == 0
    json_len, pcm_len = struct.unpack("<II", frame[8:16])
    assert pcm_len == 0
    assert len(frame) == wire.HEADER_SIZE + json_len


def test_frame_roundtrip_no_pcm():
    body = {"type": "ping", "t": 1.5}
    frame = wire.pack_frame(MsgType.PING, body)
    mtype, rbody, pcm = wire.read_frame(_DripSock(frame, chunk=64))
    assert mtype == MsgType.PING
    assert rbody == body
    assert pcm == b""


def test_frame_roundtrip_with_pcm():
    samples = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    pcm = wire.encode_pcm(samples, "f32le")
    frame = wire.pack_frame(MsgType.AUDIO, {"type": "audio", "n_samples": 256}, pcm)
    assert frame[6] & wire.FLAG_HAS_PCM
    mtype, body, rpcm = wire.read_frame(_DripSock(frame, chunk=7))
    assert mtype == MsgType.AUDIO
    assert rpcm == pcm
    assert np.array_equal(wire.decode_pcm(rpcm, "f32le"), samples)


def test_recv_exactly_byte_at_a_time():
    frame = wire.pack_frame(MsgType.STATS, {"type": "stats"})
    mtype, body, _ = wire.read_frame(_DripSock(frame, chunk=1))
    assert mtype == MsgType.STATS
    assert body == {"type": "stats"}


def test_read_frame_split_packets():
    frame = wire.pack_frame(MsgType.CONTEXT, {"type": "context", "surfaces": ["grass", "metal"]})
    mtype, body, _ = wire.read_frame(_DripSock(frame, chunk=3))
    assert mtype == MsgType.CONTEXT
    assert body["surfaces"] == ["grass", "metal"]


def test_oversized_json_rejected_before_alloc():
    header = wire.HEADER.pack(wire.MAGIC, 1, int(MsgType.EVENT), 0, 0, wire.MAX_JSON + 1, 0)
    drip = _DripSock(header)
    with pytest.raises(FrameError) as ei:
        wire.read_frame(drip)
    assert ei.value.code == "oversized" and ei.value.fatal
    assert drip.pos == wire.HEADER_SIZE


def test_oversized_pcm_rejected_before_alloc():
    header = wire.HEADER.pack(wire.MAGIC, 1, int(MsgType.AUDIO), wire.FLAG_HAS_PCM, 0, 2, wire.MAX_PCM + 1)
    drip = _DripSock(header)
    with pytest.raises(FrameError) as ei:
        wire.read_frame(drip)
    assert ei.value.code == "oversized" and ei.value.fatal
    assert drip.pos == wire.HEADER_SIZE


def test_truncated_tail_raises_clientgone():
    frame = wire.pack_frame(MsgType.STATS, {"type": "stats"})
    with pytest.raises(ClientGone):
        wire.read_frame(_DripSock(frame[:wire.HEADER_SIZE + 1], chunk=1))


def test_bad_magic_raises_frameerror():
    frame = bytearray(wire.pack_frame(MsgType.STATS, {"type": "stats"}))
    frame[0:4] = b"XXXX"
    with pytest.raises(FrameError) as ei:
        wire.read_frame(_DripSock(bytes(frame)))
    assert ei.value.code == "bad_frame"


def test_bad_version_raises_frameerror():
    body = b'{"type":"stats"}'
    header = wire.HEADER.pack(wire.MAGIC, 99, int(MsgType.STATS), 0, 0, len(body), 0)
    with pytest.raises(FrameError) as ei:
        wire.read_frame(_DripSock(header + body))
    assert ei.value.code == "bad_version"


def test_bad_json_is_non_fatal_and_consumes_body():
    bad = b"{not json"
    header = wire.HEADER.pack(wire.MAGIC, 1, int(MsgType.EVENT), 0, 0, len(bad), 0)
    drip = _DripSock(header + bad)
    with pytest.raises(FrameError) as ei:
        wire.read_frame(drip)
    assert ei.value.code == "bad_json" and not ei.value.fatal
    assert drip.pos == len(header) + len(bad)


def test_pcm_roundtrip_f32_bit_exact():
    samples = (np.random.default_rng(0).standard_normal(1000).astype(np.float32) * 0.5)
    out = wire.decode_pcm(wire.encode_pcm(samples, "f32le"), "f32le")
    assert np.array_equal(out, samples)


def test_pcm_roundtrip_i16_within_tolerance():
    samples = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    out = wire.decode_pcm(wire.encode_pcm(samples, "i16le"), "i16le")
    assert out.shape == samples.shape
    assert np.max(np.abs(out - samples)) <= 1.0 / 32767 + 1e-6
    assert out[0] == pytest.approx(-1.0) and out[-1] == pytest.approx(1.0)


def test_socketpair_real_roundtrip():
    a, b = socket.socketpair()
    try:
        a.sendall(wire.pack_frame(MsgType.PING, {"type": "ping", "t": 7.0}))
        mtype, body, _ = wire.read_frame(b)
        assert mtype == MsgType.PING and body["t"] == 7.0
    finally:
        a.close()
        b.close()
