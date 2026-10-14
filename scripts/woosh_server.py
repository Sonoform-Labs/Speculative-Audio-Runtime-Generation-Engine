import argparse
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from model_server import _recv, _send

WOOSH_HOME = os.environ.get("WOOSH_HOME", r"F:\AI\cache\woosh")
MODELS = {"flow": ("checkpoints/Woosh-Flow", 4.5), "dflow": ("checkpoints/Woosh-DFlow", 4.5)}
SR = 48000


def load(kind):
    import torch
    os.chdir(WOOSH_HOME)
    import woosh.module.audioretrieval_module as wam
    from transformers import AutoConfig

    class ConfigOnlyRoberta(wam.RobertaModel):
        @classmethod
        def from_pretrained(cls, name, *a, **kw):
            return wam.RobertaModel(AutoConfig.from_pretrained(name, **kw))

    wam.RobertaModel = ConfigOnlyRoberta
    from woosh.components.base import LoadConfig
    ckpt, cfg = MODELS[kind]
    if kind == "flow":
        from woosh.model.ldm import LatentDiffusionModel
        ldm = LatentDiffusionModel(LoadConfig(path=ckpt))
    else:
        from woosh.model.flowmap_from_pretrained import FlowMapFromPretrained
        ldm = FlowMapFromPretrained(LoadConfig(path=ckpt))
    return torch, ldm.eval().to("cuda"), cfg


def renoise_for(steps):
    if steps == 4:
        return [0, 0.5, 0.5, 0.3]
    return [0.0] + [float(v) for v in np.linspace(0.5, 0.3, max(1, steps - 1))]


def make_gen(kind, frames, steps=4, solver="dopri5"):
    torch, ldm, cfg = load(kind)
    if kind == "flow":
        from woosh.inference.flowmatching_sampler import flowmatching_integrate
    else:
        from woosh.inference.flowmap_sampler import sample_euler

    def gen(prompt, seed):
        torch.manual_seed(int(seed))
        noise = torch.randn(1, 128, int(frames)).to("cuda")
        cond = ldm.get_cond({"audio": None, "description": [prompt]}, no_dropout=True, device="cuda")
        with torch.inference_mode():
            if kind == "flow" and solver == "dopri5":
                x = flowmatching_integrate(ldm, noise=noise, cond=cond, cfg=cfg, atol=0.001, rtol=0.001,
                                           device="cuda", dtype=torch.float64)
            elif kind == "flow":
                x = flowmatching_integrate(ldm, noise=noise, cond=cond, cfg=cfg, method=solver, device="cuda",
                                           step_size=1.0 / max(1, int(steps)))
            else:
                x = sample_euler(model=ldm, noise=noise, cond=cond, num_steps=int(steps), renoise=renoise_for(int(steps)),
                                 cfg=cfg)
            audio = ldm.autoencoder.inverse(x)
        return audio[0].detach().to(torch.float32).cpu().numpy().T

    return gen


def finish(x, event, intensity, amp, stereo):
    from sarge.generators.stable_audio import _natural
    from sarge.loudness import normalize_peak, peak_for
    y = x[:, 0] if x.ndim == 2 and x.shape[1] == 1 else x
    y = _natural(np.ascontiguousarray(y, dtype=np.float32), SR)
    if amp:
        from sarge.amp import amplify
        y = amplify(y, SR, event, intensity)
    if stereo:
        from sarge.stereo import widen
        y = widen(y, SR, event)
    return normalize_peak(y, peak_for(intensity)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="dflow")
    ap.add_argument("--frames", type=int, default=101)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--solver", default="dopri5")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--stereo", action="store_true")
    args = ap.parse_args()
    from sarge.prompts import build
    from sarge.types import CacheKey, seed_for

    gen = make_gen(args.model, args.frames, args.steps, args.solver)
    lock = threading.Lock()
    t0 = time.perf_counter()
    gen("warmup, a soft thud", 0)
    print("woosh %s ready, %d frames, warm-up %.1f s" % (args.model, args.frames, time.perf_counter() - t0), flush=True)

    def serve(conn):
        try:
            while True:
                req, _ = _recv(conn)
                if req.get("type") == "hello":
                    _send(conn, {"type": "hello", "sample_rate": SR, "model": "woosh-" + args.model})
                    continue
                key = CacheKey(req["event_type"], req["material"], req.get("intensity", "normal"),
                               req.get("context", "default"))
                prompt = build(key.event_type, key.material, key.intensity, key.context)
                t = time.perf_counter()
                with lock:
                    raw = gen(prompt, seed_for(key, int(req.get("variant", 0))))
                y = finish(raw, key.event_type, key.intensity, args.amp, args.stereo)
                ch = 1 if y.ndim == 1 else int(y.shape[1])
                _send(conn, {"type": "audio", "sample_rate": SR, "channels": ch, "frames": int(y.shape[0]),
                             "gen_ms": (time.perf_counter() - t) * 1000.0}, np.ascontiguousarray(y).tobytes())
        except (ConnectionError, OSError):
            pass
        finally:
            conn.close()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(8)
    print("woosh server listening on %s:%d" % (args.host, srv.getsockname()[1]), flush=True)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=serve, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
