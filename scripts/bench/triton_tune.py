import itertools
import json
import os
import sys

import torch
import triton
import triton.language as tl

BENCH = os.path.dirname(os.path.abspath(__file__))
SHAPES = {"proj": (1024, 1024, False), "qkv": (1024, 3072, False), "ff_in": (1024, 8192, True),
          "ff_out": (4096, 1024, True)}
PER_CLIP = {"proj": 120, "qkv": 40, "ff_in": 40, "ff_out": 40}
M = 172


@triton.jit
def mm16(a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr, HAS_BIAS: tl.constexpr):
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
    acc = tl.zeros((BM, BN), dtype=tl.float16)
    for k in range(0, tl.cdiv(K, BK)):
        x = tl.load(a, mask=(rm[:, None] < M) & (rk[None, :] < K - k * BK), other=0.0)
        y = tl.load(b, mask=(rk[:, None] < K - k * BK) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(x, y, acc, out_dtype=tl.float16)
        a += BK * sak
        b += BK * sbk
    if HAS_BIAS:
        acc += tl.load(bias_ptr + rn, mask=rn < N, other=0.0)[None, :]
    c = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def mm16_splitk(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scs, scm, scn,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    pid = tl.program_id(0)
    sid = tl.program_id(1)
    num_n = tl.cdiv(N, BN)
    pm = pid // num_n
    pn = pid % num_n
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    kper = tl.cdiv(K, SPLIT)
    k0 = sid * kper
    rk = k0 + tl.arange(0, BK)
    a = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float16)
    for k in range(0, tl.cdiv(kper, BK)):
        x = tl.load(a, mask=(rm[:, None] < M), other=0.0)
        y = tl.load(b, mask=(rn[None, :] < N), other=0.0)
        acc = tl.dot(x, y, acc, out_dtype=tl.float16)
        a += BK * sak
        b += BK * sbk
    c = c_ptr + sid * scs + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def run_plain(x, w, bias, out, cfg):
    bm, bn, bk, warps, stages = cfg
    k = x.shape[1]
    n = w.shape[0]
    grid = (triton.cdiv(M, bm) * triton.cdiv(n, bn),)
    mm16[grid](x, w, out, bias if bias is not None else out, M, n, k, x.stride(0), x.stride(1), w.stride(1),
               w.stride(0), out.stride(0), out.stride(1), BM=bm, BN=bn, BK=bk, GROUP=8, HAS_BIAS=bias is not None,
               num_warps=warps, num_stages=stages)
    return out


def run_split(x, w, bias, part, cfg):
    bm, bn, bk, warps, stages, split = cfg
    k = x.shape[1]
    n = w.shape[0]
    grid = (triton.cdiv(M, bm) * triton.cdiv(n, bn), split)
    mm16_splitk[grid](x, w, part, M, n, k, x.stride(0), x.stride(1), w.stride(1), w.stride(0), part.stride(0),
                      part.stride(1), part.stride(2), BM=bm, BN=bn, BK=bk, SPLIT=split, num_warps=warps,
                      num_stages=stages)
    out = part[:split].sum(0, dtype=torch.float16)
    if bias is not None:
        out = out + bias
    return out


def bench(fn, n=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(3):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000.0 / n)
    return best


def main():
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    only = sys.argv[1:] or list(SHAPES)
    rep = {}
    for name in only:
        k, n, has_bias = SHAPES[name]
        x = torch.randn(M, k, device="cuda", dtype=torch.float16)
        w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
        bias = (torch.randn(n, device="cuda", dtype=torch.float16) * 0.1) if has_bias else None
        ref = torch.nn.functional.linear(x.float(), w.float(), None if bias is None else bias.float())
        t_cub = bench(lambda: torch.nn.functional.linear(x, w, bias))
        out = torch.empty(M, n, device="cuda", dtype=torch.float16)
        plain = []
        for cfg in itertools.product((32, 64, 128), (32, 64, 128, 256), (32, 64, 128), (2, 4, 8), (2, 3, 4, 5)):
            if cfg[0] * cfg[1] > 128 * 256 or cfg[1] * cfg[2] > 256 * 128 or cfg[0] * cfg[2] > 128 * 128:
                continue
            try:
                run_plain(x, w, bias, out, cfg)
                plain.append((bench(lambda: run_plain(x, w, bias, out, cfg)), cfg))
            except Exception:
                continue
        plain.sort()
        part = torch.empty(8, M, n, device="cuda", dtype=torch.float16)
        split = []
        for cfg in itertools.product((32, 64, 128), (32, 64, 128), (32, 64, 128), (4, 8), (2, 3, 4), (2, 4, 8)):
            if k % (cfg[5] * cfg[2]):
                continue
            try:
                run_split(x, w, bias, part, cfg)
                split.append((bench(lambda: run_split(x, w, bias, part, cfg)), cfg))
            except Exception:
                continue
        split.sort()
        bp = plain[0]
        err_p = float((run_plain(x, w, bias, out, bp[1]).float() - ref).abs().max())
        err_c = float((torch.nn.functional.linear(x, w, bias).float() - ref).abs().max())
        rec = {"cublas_us": round(t_cub, 2), "plain_us": round(bp[0], 2), "plain_cfg": bp[1],
               "plain_top5": [(round(t, 2), c) for t, c in plain[:5]], "err_plain": err_p, "err_cublas": err_c}
        if split:
            bs = split[0]
            err_s = float((run_split(x, w, bias, part, bs[1]).float() - ref).abs().max())
            rec.update({"split_us": round(bs[0], 2), "split_cfg": bs[1], "err_split": err_s})
        rep[name] = rec
        print(name, json.dumps(rec), flush=True)
    clip_c = sum(rep[nm]["cublas_us"] * PER_CLIP[nm] for nm in rep) / 1000.0
    clip_b = sum(min(rep[nm]["cublas_us"], rep[nm]["plain_us"], rep[nm].get("split_us", 1e9)) * PER_CLIP[nm]
                 for nm in rep) / 1000.0
    rep["clip_ms_cublas"], rep["clip_ms_best"] = clip_c, clip_b
    print("per clip: cublas %.2f ms, best %.2f ms" % (clip_c, clip_b), flush=True)
    json.dump(rep, open(os.path.join(BENCH, "out_triton_tune.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
