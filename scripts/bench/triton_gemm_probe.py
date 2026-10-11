import itertools
import json
import os

import torch
import triton
import triton.language as tl

BENCH = os.path.dirname(os.path.abspath(__file__))
SHAPES = {"qkv": (1024, 3072), "attn_out": (1024, 1024), "ff_in": (1024, 8192), "ff_out": (4096, 1024)}
PER_CLIP = {"qkv": 40, "attn_out": 120, "ff_in": 40, "ff_out": 40}


@triton.jit
def mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr, ACC16: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    width = GROUP * num_n
    gid = pid // width
    first_m = gid * GROUP
    gsize = min(num_m - first_m, GROUP)
    pm = first_m + (pid % width) % gsize
    pn = (pid % width) // gsize
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    if ACC16:
        acc = tl.zeros((BM, BN), dtype=tl.float16)
    else:
        acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        x = tl.load(a, mask=(rm[:, None] < M) & (rk[None, :] < K - k * BK), other=0.0)
        y = tl.load(b, mask=(rk[:, None] < K - k * BK) & (rn[None, :] < N), other=0.0)
        if ACC16:
            acc = tl.dot(x, y, acc, out_dtype=tl.float16)
        else:
            acc = tl.dot(x, y, acc)
        a += BK * sak
        b += BK * sbk
    c = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c, acc.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def tmm(x, wt, out, cfg, acc16=True):
    M, K = x.shape
    N = wt.shape[1]
    bm, bn, bk, warps, stages = cfg
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    mm_kernel[grid](x, wt, out, M, N, K, x.stride(0), x.stride(1), wt.stride(0), wt.stride(1),
                    out.stride(0), out.stride(1), BM=bm, BN=bn, BK=bk, GROUP=8, ACC16=acc16,
                    num_warps=warps, num_stages=stages)
    return out


def bench(fn, n=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / n


def main():
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    M = 172
    configs = [c for c in itertools.product((32, 64, 128), (64, 128, 256), (32, 64), (4, 8), (3, 4))
               if not (c[0] == 128 and c[1] == 256)]
    rep = {}
    total_cublas = total_triton = 0.0
    for name, (k, n) in SHAPES.items():
        x = torch.randn(M, k, device="cuda", dtype=torch.float16)
        w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
        wt = w.t().contiguous()
        ref32 = (x.float() @ w.float().t())
        cub = torch.nn.functional.linear(x, w)
        t_cub = bench(lambda: torch.nn.functional.linear(x, w))
        out = torch.empty(M, n, device="cuda", dtype=torch.float16)
        best = None
        for cfg in configs:
            try:
                tmm(x, wt, out, cfg)
                us = bench(lambda: tmm(x, wt, out, cfg), 100)
            except Exception:
                continue
            if best is None or us < best[0]:
                best = (us, cfg)
        tmm(x, wt, out, best[1])
        err_t = float((out.float() - ref32).abs().max())
        err_c = float((cub.float() - ref32).abs().max())
        rep[name] = {"cublas_us": round(t_cub, 1), "triton_us": round(best[0], 1), "cfg": best[1],
                     "maxerr_vs_fp32_triton": err_t, "maxerr_vs_fp32_cublas": err_c,
                     "tflops_triton": round(2 * M * k * n / best[0] / 1e6, 1)}
        total_cublas += t_cub * PER_CLIP[name]
        total_triton += min(t_cub, best[0]) * PER_CLIP[name]
        print(name, rep[name], flush=True)
    rep["clip_ms_cublas"] = total_cublas / 1000.0
    rep["clip_ms_best_of_both"] = total_triton / 1000.0
    print("per clip GEMM ms: cublas %.2f, best of both %.2f" % (rep["clip_ms_cublas"], rep["clip_ms_best_of_both"]))
    json.dump(rep, open(os.path.join(BENCH, "out_triton_gemm_probe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
