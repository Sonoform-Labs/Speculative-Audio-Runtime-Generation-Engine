from __future__ import annotations

from typing import Optional

import torch

CONFIGS = {(1024, 8192): (64, 256, 32, 4, 3), (1024, 3072): (64, 128, 32, 4, 3), (4096, 1024): (64, 128, 64, 4, 4),
           (1024, 1024): (64, 128, 64, 8, 4)}
DEFAULT_SHAPES = ((1024, 8192), (1024, 3072), (4096, 1024))

_OP = None


def _build():
    global _OP
    if _OP is not None:
        return _OP
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton

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

    @triton_op("sarge::linear16", mutates_args={})
    def linear16(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        m, k = x.shape
        n = w.shape[0]
        bm, bn, bk, warps, stages = CONFIGS[(k, n)]
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
        wrap_triton(mm16)[grid](x, w, out, out if bias is None else bias, m, n, k, x.stride(0), x.stride(1),
                                w.stride(1), w.stride(0), out.stride(0), out.stride(1), BM=bm, BN=bn, BK=bk, GROUP=8,
                                HAS_BIAS=bias is not None, num_warps=warps, num_stages=stages)
        return out

    _OP = linear16
    return _OP


class TritonLinear:
    def __init__(self, module, op):
        self.module = module
        self.orig = module.forward
        self.op = op

    def __call__(self, x):
        m = self.module
        if x.dtype != torch.float16 or not x.is_cuda or m.weight.dtype != torch.float16 or x.shape[-1] != m.in_features:
            return self.orig(x)
        y = self.op(x.reshape(-1, m.in_features), m.weight, m.bias)
        return y.reshape(*x.shape[:-1], m.out_features)


def install_triton_gemm(root, shapes=DEFAULT_SHAPES) -> list:
    if not torch.cuda.is_available():
        return []
    try:
        op = _build()
    except Exception:
        return []
    out = []
    for mod in root.modules():
        if not isinstance(mod, torch.nn.Linear) or (mod.in_features, mod.out_features) not in shapes:
            continue
        if isinstance(mod.forward, TritonLinear):
            continue
        t = TritonLinear(mod, op)
        mod.forward = t
        out.append(t)
    return out
