import _bootstrap

import argparse
import glob
import os
import wave

import numpy as np
import torch
import torchaudio.functional as AF

from sage.generators import get_generator
from sage.types import CacheKey

MATS = ["wood", "stone", "grass", "metal"]


def load_wav(path):
    w = wave.open(path, "rb")
    sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
    d = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        d = d.reshape(-1, ch).mean(axis=1)
    return d, sr


def resample(c, src, dst):
    if src == dst:
        return c.astype(np.float32)
    return AF.resample(torch.from_numpy(np.ascontiguousarray(c)).float(), src, dst).numpy().astype(np.float32)


def sequence(clips, sr, gait_s=0.45):
    step = int(gait_s * sr)
    n = step * (len(clips) - 1) + max(len(c) for c in clips)
    b = np.zeros(n, dtype=np.float32)
    for i, c in enumerate(clips):
        b[i * step:i * step + len(c)] += c
    peak = float(np.max(np.abs(b))) or 1.0
    return (b / peak * 0.9).astype(np.float32)


def embed(clips_32k, dev):
    from panns_inference import AudioTagging
    at = AudioTagging(checkpoint_path=None, device=dev)
    out = []
    for c in clips_32k:
        _, e = at.inference(c[None, :].astype(np.float32))
        out.append(np.asarray(e).reshape(-1))
    del at
    torch.cuda.empty_cache()
    return np.stack(out)


def frechet(A, B, eps=1e-6):
    import scipy.linalg
    muA, muB = A.mean(0), B.mean(0)
    CA = np.cov(A, rowvar=False) + eps * np.eye(A.shape[1])
    CB = np.cov(B, rowvar=False) + eps * np.eye(B.shape[1])
    cov, _ = scipy.linalg.sqrtm(CA @ CB, disp=False)
    if np.iscomplexobj(cov):
        cov = cov.real
    d = muA - muB
    return float(d @ d + np.trace(CA + CB - 2 * cov))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FAD-style realism: generated footsteps vs real (ESC-50).")
    ap.add_argument("--real", default=os.path.join("data", "real_footsteps"))
    ap.add_argument("--models", nargs="+", default=["sa3", "stableaudio", "mock"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--pca", type=int, default=20)
    args = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    realpaths = sorted(glob.glob(os.path.join(args.real, "*.wav")))
    real = []
    for p in realpaths:
        d, sr = load_wav(p)
        real.append(resample(d, sr, 32000))
    print(f"real footsteps: {len(real)} clips")

    gen = {}
    for m in args.models:
        g = get_generator(m)
        sr = g.sample_rate
        seqs = []
        for i in range(args.n):
            mat = MATS[i % len(MATS)]
            vs = [g.generate(CacheKey("footstep", mat, "normal"), variant=i * 8 + v).samples for v in range(8)]
            seqs.append(resample(sequence(vs, sr), sr, 32000))
        del g
        torch.cuda.empty_cache()
        gen[m] = seqs
        print(f"{m}: {len(seqs)} walk sequences")

    real_e = embed(real, dev)
    gen_e = {m: embed(s, dev) for m, s in gen.items()}

    pool = np.concatenate([real_e] + list(gen_e.values()), 0)
    mu = pool.mean(0)
    _, _, Vt = np.linalg.svd(pool - mu, full_matrices=False)
    W = Vt[:args.pca].T
    proj = lambda E: (E - mu) @ W
    real_p = proj(real_e)
    gen_p = {m: proj(e) for m, e in gen_e.items()}

    rc = real_e / np.linalg.norm(real_e, axis=1, keepdims=True)
    centroid = rc.mean(0)
    centroid /= np.linalg.norm(centroid)

    def cos_to_real(E):
        En = E / np.linalg.norm(E, axis=1, keepdims=True)
        return float(np.mean(1 - En @ centroid))

    half = len(real_p) // 2
    floor = frechet(real_p[:half], real_p[half:])

    print(f"\n{'model':<14}{'FAD (PANNs, PCA' + str(args.pca) + ')':>22}{'cos-dist-to-real':>18}")
    print("-" * 54)
    print(f"{'real-vs-real':<14}{floor:>22.3f}{0.0:>18.3f}   (sanity floor)")
    for m in args.models:
        print(f"{m:<14}{frechet(gen_p[m], real_p):>22.3f}{cos_to_real(gen_e[m]):>18.3f}")
    print("\nlower = closer to real footsteps; real-vs-real is the noise floor.")
    print("(small-N FAD on ~40 real clips is indicative, not rigorous; trust the RELATIVE ordering.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
