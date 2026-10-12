import numpy as np

from sarge.generators.stable_audio import _extract, _natural
from sarge.stereo import widen
from sarge.types import AudioClip


def burst(sr=44100):
    t = np.arange(int(0.4 * sr))
    x = np.zeros(sr, dtype=np.float32)
    x[3000:3000 + len(t)] = (np.exp(-t / (0.05 * sr)) * np.sin(t * 0.3) * 0.5).astype(np.float32)
    return x


def test_stereo_clip_round_trips_through_wav(tmp_path):
    x = burst()
    st = np.stack([x, x * 0.5], axis=1)
    c = AudioClip(st, 44100)
    assert c.channels == 2 and c.frames == len(x) and abs(c.duration_s - len(x) / 44100) < 1e-9
    p = tmp_path / "s.wav"
    c.to_wav(p)
    back = AudioClip.from_wav(p, keep_channels=True)
    assert back.samples.shape == st.shape and np.max(np.abs(back.samples - st)) < 1e-3
    assert AudioClip.from_wav(p).samples.ndim == 1


def test_width_keeps_the_mid_and_adds_a_decorrelated_side():
    x = burst()
    y = widen(x, 44100, "shoot")
    assert y.shape == (len(x), 2)
    assert np.max(np.abs((y[:, 0] + y[:, 1]) / 2 - x)) < 1e-5
    side = (y[:, 0] - y[:, 1]) / 2
    assert np.sqrt(np.mean(side ** 2)) > 0.05 * np.sqrt(np.mean(x ** 2))
    assert abs(np.corrcoef(y[:, 0], y[:, 1])[0, 1]) < 0.995


def test_trims_keep_both_channels():
    x = burst()
    st = np.stack([x, x], axis=1)
    n = _natural(st, 44100)
    e = _extract(st, 44100, 0.6)
    assert n.ndim == 2 and n.shape[1] == 2 and n.shape[0] < len(x)
    assert e.shape == (int(0.6 * 44100), 2)
