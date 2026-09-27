import _bootstrap

import json
import os

import numpy as np
import torch

from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey


def snr_db(ref, x):
    n = len(ref)
    err = float(np.sum((ref[:n] - x[:n]) ** 2)) + 1e-20
    return float(10 * np.log10(float(np.sum(ref[:n] ** 2)) / err))


def main():
    g = StableAudio3Generator()
    g.ensure_ready()
    real_randn = torch.randn
    state = {"noise": None}

    def fake_randn(*args, **kw):
        shape = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else list(args)
        if state["noise"] is not None and len(shape) == 3 and shape[0] <= state["noise"].shape[0]:
            return state["noise"][: shape[0]].clone()
        return real_randn(*args, **kw)

    torch.randn = fake_randn
    g.generate(CacheKey("footstep", "wood", "normal"), variant=0)

    mats = ["wood", "stone", "grass", "metal"]
    rows = []
    for mi, mat in enumerate(mats):
        key = CacheKey("footstep", mat, "normal")
        gen = torch.Generator(device="cuda").manual_seed(1000 + mi)
        io = g._model.model.io_channels
        base = real_randn([2, io, 4096], device="cuda", generator=gen)
        state["noise"] = None
        probe = []

        def capture(*args, **kw):
            shape = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else list(args)
            if len(shape) == 3:
                probe.append(list(shape))
            return real_randn(*args, **kw)

        torch.randn = capture
        g.generate(key, variant=0)
        torch.randn = fake_randn
        L = probe[0][2]
        noise = base[:, :, :L].contiguous()

        state["noise"] = noise
        batched = g.generate_batch([(key, 0), (key, 1)])
        state["noise"] = noise[0:1]
        single0 = g.generate(key, variant=0)
        single0b = g.generate(key, variant=0)
        state["noise"] = noise[1:2]
        single1 = g.generate(key, variant=1)

        m = g._model
        p = g._prompt(key)
        kw = dict(duration=g.gen_duration_s, steps=g.steps, cfg_scale=g.cfg_scale,
                  sampler_type=g.sampler_type, disable_tqdm=True)
        raw = {}
        for tag, lat in (("latent", True), ("audio", False)):
            state["noise"] = noise
            rb = m.generate(prompt=[p, p], batch_size=2, seed=1, return_latents=lat, **kw)
            state["noise"] = noise[0:1]
            rs0 = m.generate(prompt=p, batch_size=1, seed=1, return_latents=lat, **kw)
            state["noise"] = noise[1:2]
            rs1 = m.generate(prompt=p, batch_size=1, seed=1, return_latents=lat, **kw)
            fb = rb.detach().float().cpu().numpy()
            f0 = rs0.detach().float().cpu().numpy()[0]
            f1 = rs1.detach().float().cpu().numpy()[0]
            raw[tag] = (snr_db(f0.ravel(), fb[0].ravel()), snr_db(f1.ravel(), fb[1].ravel()),
                        snr_db(fb[0].ravel(), fb[1].ravel()))
        print("        raw latent  row0 %6.1f dB  row1 %6.1f dB  different-noise %6.1f dB" % raw["latent"])
        print("        raw audio   row0 %6.1f dB  row1 %6.1f dB  different-noise %6.1f dB" % raw["audio"])

        r = {
            "material": mat,
            "latent_len": L,
            "raw_latent_snr_db": raw["latent"],
            "raw_audio_snr_db": raw["audio"],
            "rerun_snr_db": snr_db(single0.samples, single0b.samples),
            "batch_row0_snr_db": snr_db(single0.samples, batched[0].samples),
            "batch_row1_snr_db": snr_db(single1.samples, batched[1].samples),
            "row0_vs_row1_snr_db": snr_db(batched[0].samples, batched[1].samples),
            "max_abs_row0": float(np.max(np.abs(single0.samples - batched[0].samples))),
        }
        rows.append(r)
        print("%-6s L=%d  rerun %6.1f dB  batch row0 %6.1f dB  row1 %6.1f dB  different-noise %6.1f dB  max|d| %.2e"
              % (mat, L, r["rerun_snr_db"], r["batch_row0_snr_db"], r["batch_row1_snr_db"],
                 r["row0_vs_row1_snr_db"], r["max_abs_row0"]))

    torch.randn = real_randn
    json.dump(rows, open(os.path.join("scripts", "bench", "out_batch_equiv.json"), "w"), indent=1)
    print("saved -> scripts\\bench\\out_batch_equiv.json")


if __name__ == "__main__":
    main()
