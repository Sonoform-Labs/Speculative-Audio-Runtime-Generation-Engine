import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import glob
import json
import time

import numpy as np

import vocab_eval as ve
from sarge.generators.stable_audio import _extract
from sarge.loudness import normalize_loudness, target_for
from sarge.types import AudioClip

ROOT = os.path.dirname(os.path.dirname(BENCH))


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["same-s", "same-l"])
    ap.add_argument("--src", default=os.path.join("data", "_gate", "anchor", "raw"))
    ap.add_argument("--out", default=os.path.join("data", "_gate", "ae_roundtrip"))
    ap.add_argument("--keep-s", type=float, default=0.6)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    import torch
    from stable_audio_3.model import AutoencoderModel
    src = os.path.join(ROOT, args.src)
    files = sorted(glob.glob(os.path.join(src, "*.wav")))
    info = json.load(open(os.path.join(src, "route.json"), encoding="utf-8"))
    report = {}
    for name in args.models:
        t0 = time.perf_counter()
        ae = AutoencoderModel.from_pretrained(name, device="cuda")
        load_s = time.perf_counter() - t0
        params = sum(p.numel() for p in ae.autoencoder.parameters())
        d = os.path.join(ROOT, args.out, name.replace("-", "_"))
        os.makedirs(d, exist_ok=True)
        ms = []
        for f in files:
            x = ve.wav(f)
            sr = int(ae.sample_rate)
            audio = torch.from_numpy(np.stack([x, x])).float()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            z = ae.encode(audio, ve.SR)
            y = ae.decode(z)
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t1) * 1000.0)
            y = y.detach().float().cpu().numpy()
            while y.ndim > 2:
                y = y[0]
            mono = y.mean(axis=0) if y.ndim == 2 else y
            if sr != ve.SR:
                from validate_panel import _resample
                mono = _resample(mono, sr, ve.SR)
            mono = _extract(mono.astype(np.float32), ve.SR, args.keep_s)
            mono = normalize_loudness(mono, target_rms=target_for("normal")).astype(np.float32)
            AudioClip(mono, ve.SR).to_wav(os.path.join(d, os.path.basename(f)))
        route = dict(info)
        route["note"] = "real anchors from %s encoded and decoded through %s" % (args.src, name)
        json.dump(route, open(os.path.join(d, "route.json"), "w", encoding="utf-8"), indent=1)
        report[name] = {"params_m": params / 1e6, "load_s": load_s, "roundtrip_ms_median": float(np.median(ms)),
                        "files": len(files), "sample_rate": int(ae.sample_rate),
                        "latent_shape": list(z.shape)}
        print("%s  %.0fM params  load %.1f s  encode+decode median %.1f ms  latent %s  sr %d" % (
            name, params / 1e6, load_s, report[name]["roundtrip_ms_median"], list(z.shape), ae.sample_rate),
            flush=True)
        del ae
        torch.cuda.empty_cache()
    json.dump(report, open(os.path.join(ROOT, "scripts", "bench", "out_ae_roundtrip.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
