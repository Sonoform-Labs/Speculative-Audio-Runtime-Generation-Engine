import json
import os
import sys

import torch

BENCH = os.path.dirname(os.path.abspath(__file__))
SHAPES = {"qkv": (1024, 3072), "attn_out": (1024, 1024), "cross_q": (1024, 1024), "ff_in": (1024, 8192),
          "ff_out": (4096, 1024)}
PER_CLIP = {"qkv": 40, "attn_out": 80, "cross_q": 40, "ff_in": 40, "ff_out": 40}


def bench(fn, n=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    s.record()
    g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / n


def main():
    torch.backends.cuda.matmul.allow_fp16_accumulation = "--fp32acc" not in sys.argv
    if "--lt" in sys.argv:
        torch.backends.cuda.preferred_blas_library("cublaslt")
    rep = {"fp16_acc": torch.backends.cuda.matmul.allow_fp16_accumulation,
           "blas": str(torch.backends.cuda.preferred_blas_library()), "us": {}}
    for name, (k, n) in SHAPES.items():
        w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
        row = {}
        for m in (86, 128, 172, 192, 256, 344):
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            us = bench(lambda: torch.nn.functional.linear(x, w))
            row[m] = {"us": round(us, 1), "tflops": round(2 * m * k * n / us / 1e6, 1)}
        rep["us"][name] = row
        print(name, {m: (v["us"], v["tflops"]) for m, v in row.items()}, flush=True)
    rep["clip_gemm_ms_at_172"] = sum(rep["us"][n][172]["us"] * PER_CLIP[n] for n in SHAPES) / 1000.0
    print("GEMM ms per clip at M=172:", round(rep["clip_gemm_ms_at_172"], 2), flush=True)
    tag = "_".join(a.strip("-") for a in sys.argv[1:]) or "default"
    json.dump(rep, open(os.path.join(BENCH, "out_gemm_probe_%s.json" % tag), "w"), indent=1)


if __name__ == "__main__":
    main()
