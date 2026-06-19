import _bootstrap

import argparse
import os

import numpy as np

from predictive_sfx.generators import get_generator
from predictive_sfx.types import AudioClip, CacheKey


def _sequence(clips, sr, gait_s):
    step = int(gait_s * sr)
    n = step * (len(clips) - 1) + max(len(c) for c in clips)
    buf = np.zeros(n, dtype=np.float32)
    for i, c in enumerate(clips):
        buf[i * step:i * step + len(c)] += c
    peak = float(np.max(np.abs(buf))) or 1.0
    return (buf / peak * 0.9).astype(np.float32)


def _mean_pairwise_corr(clips):
    length = min(len(c) for c in clips)
    m = np.stack([c[:length].astype(np.float64) for c in clips])
    m = m - m.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(m, axis=1)
    vals = []
    for i in range(len(clips)):
        for j in range(i + 1, len(clips)):
            d = norms[i] * norms[j]
            vals.append(float(m[i] @ m[j] / d) if d > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build a varied-vs-repeated footstep walk demo.")
    p.add_argument("--generator", default="stableaudio")
    p.add_argument("--event-type", default="footstep", dest="event_type")
    p.add_argument("--material", default="wood")
    p.add_argument("--intensity", default="normal")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--gait-ms", type=float, default=450.0, dest="gait_ms")
    p.add_argument("--out-dir", default="data/demo", dest="out_dir")
    args = p.parse_args(argv)

    gen = get_generator(args.generator)
    key = CacheKey(args.event_type, args.material, args.intensity)
    clips = [gen.generate(key, variant=v).samples for v in range(args.count)]
    sr = gen.sample_rate
    gait_s = args.gait_ms / 1000.0

    os.makedirs(args.out_dir, exist_ok=True)
    varied = _sequence(clips, sr, gait_s)
    repeated = _sequence([clips[0]] * args.count, sr, gait_s)
    AudioClip(varied, sr).to_wav(os.path.join(args.out_dir, "varied_walk.wav"))
    AudioClip(repeated, sr).to_wav(os.path.join(args.out_dir, "repeated_walk.wav"))
    for v, c in enumerate(clips):
        AudioClip(c, sr).to_wav(os.path.join(args.out_dir, f"variant_{v:02d}.wav"))

    corr = _mean_pairwise_corr(clips)
    print(f"key: {key}")
    print(f"variants: {args.count}  gait: {args.gait_ms:.0f} ms")
    print(f"mean pairwise waveform correlation: {corr:.3f}  (lower = more distinct; ~0 = uncorrelated)")
    print(f"wrote {args.out_dir}/varied_walk.wav  (neural: every step different)")
    print(f"wrote {args.out_dir}/repeated_walk.wav  (one sample looped: the repetition problem)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
