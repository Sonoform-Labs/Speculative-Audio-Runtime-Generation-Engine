import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from dec_common import SR, _win

PERIODS = (2, 3, 5, 7, 11)
MBD_WINDOWS = (2048, 1024, 512, 256, 128)
BANDS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def wn(m):
    return weight_norm(m)


class PeriodD(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        ch = [1, 32, 128, 256, 512]
        self.convs = nn.ModuleList([wn(nn.Conv2d(ch[i], ch[i + 1], (5, 1), (3, 1), padding=(2, 0))) for i in range(4)] +
                                   [wn(nn.Conv2d(512, 512, (5, 1), 1, padding=(2, 0)))])
        self.post = wn(nn.Conv2d(512, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        b, t = x.shape
        if t % self.p:
            x = F.pad(x, (0, self.p - t % self.p), mode="reflect")
        x = x.view(b, 1, -1, self.p)
        feats = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        return self.post(x).flatten(1), feats


class BandStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([wn(nn.Conv2d(2, 32, (3, 9), 1, padding=(1, 4)))] +
                                   [wn(nn.Conv2d(32, 32, (3, 9), (1, 2), padding=(1, 4))) for _ in range(3)] +
                                   [wn(nn.Conv2d(32, 32, (3, 3), 1, padding=(1, 1)))])

    def forward(self, x, feats):
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        return x


class ResD(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        nb = n // 2 + 1
        cuts = BANDS if n >= 512 else (0.0, 1.0)
        self.bands = [(int(round(a * nb)), int(round(b * nb))) for a, b in zip(cuts[:-1], cuts[1:])]
        self.stacks = nn.ModuleList([BandStack() for _ in self.bands])
        self.post = wn(nn.Conv2d(32, 1, (3, 3), 1, padding=(1, 1)))

    def spec(self, x):
        with torch.autocast("cuda", enabled=False):
            X = torch.stft(x.float(), self.n, hop_length=self.n // 4, win_length=self.n, window=_win(self.n, x.device),
                           center=True, return_complex=True)
            return torch.view_as_real(X).permute(0, 3, 2, 1).contiguous()

    def forward(self, x, bf16=True):
        s = self.spec(x)
        feats = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and x.is_cuda):
            outs = [st(s[..., a:b], feats) for st, (a, b) in zip(self.stacks, self.bands)]
            y = self.post(torch.cat(outs, dim=-1))
        return y.flatten(1), feats


class Disc(nn.Module):
    def __init__(self, kinds=("mpd", "mbd"), mbd_windows=MBD_WINDOWS, periods=PERIODS):
        super().__init__()
        self.mpd = nn.ModuleList([PeriodD(p) for p in periods]) if "mpd" in kinds else nn.ModuleList()
        self.mbd = nn.ModuleList([ResD(n) for n in mbd_windows]) if "mbd" in kinds else nn.ModuleList()

    def core(self, xs, bf16=True):
        out = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and xs.is_cuda):
            for d in self.mpd:
                lg, ft = d(xs)
                out.append((lg.float(), ft))
        for d in self.mbd:
            lg, ft = d(xs, bf16)
            out.append((lg.float(), ft))
        return out

    def prep(self, x, scale):
        return (x - x.mean(-1, keepdim=True)) * scale[:, None]

    def forward(self, x, scale, bf16=True):
        return self.core(self.prep(x, scale), bf16)


def disc_scale(x_real, floor_db=-30.0):
    pk = (x_real - x_real.mean(-1, keepdim=True)).abs().amax(-1)
    return 0.8 / torch.clamp(pk, min=10 ** (floor_db / 20.0))


def tilt(x, g_db, fc=2000.0):
    n = x.shape[-1]
    X = torch.fft.rfft(x.float(), n)
    f = torch.fft.rfftfreq(n, 1.0 / SR).to(x.device)
    g = (10 ** (g_db / 20.0)).to(x.device)[:, None]
    s = 1j * f[None, :] / fc
    H = (g * s + 1) / (s + 1)
    return torch.fft.irfft(X * H, n)


def paired_aug(x, xh, rng):
    b, n = x.shape
    shifts = rng.integers(0, 256, b)
    pol = np.where(rng.uniform(size=b) < 0.5, -1.0, 1.0)
    g_db = torch.tensor(rng.uniform(-3.0, 3.0, b), dtype=torch.float32)
    sign = torch.tensor(pol, dtype=torch.float32, device=x.device)[:, None]

    def apply(v):
        rows = []
        for i in range(b):
            s = int(shifts[i])
            rows.append(F.pad(v[i, :n - s], (s, 0)) if s else v[i])
        return tilt(torch.stack(rows) * sign, g_db)

    return apply(x), apply(xh)


def _wmean(per, w, wsum):
    return (per * w).sum() / wsum


def d_loss(out_r, out_f, w, wsum):
    tot = 0.0
    for (lr, _), (lf, _) in zip(out_r, out_f):
        tot = tot + _wmean(F.softplus(-(lr - lf)).mean(1), w, wsum)
    return tot / len(out_r)


def g_adv(out_r, out_f, w, wsum):
    tot = 0.0
    for (lr, _), (lf, _) in zip(out_r, out_f):
        tot = tot + _wmean(F.softplus(lr.detach() - lf).mean(1), w, wsum)
    return tot / len(out_r)


def fm_loss(out_r, out_f, w, wsum):
    tot, n = 0.0, 0
    for (_, fr), (_, ff) in zip(out_r, out_f):
        for a, b in zip(fr, ff):
            tot = tot + _wmean((b.float() - a.float().detach()).abs().flatten(1).mean(1), w, wsum)
            n += 1
    return tot / max(n, 1)


def pair_acc(out_r, out_f):
    acc = []
    for (lr, _), (lf, _) in zip(out_r, out_f):
        acc.append((lr.mean(1) > lf.mean(1)).float())
    return torch.stack(acc, 1).mean(1)


def d_step(disc, x, xh, w, scale, micro=2):
    b = x.shape[0]
    micro = max(1, min(int(micro), b))
    wsum = w.sum().clamp(min=1e-6)
    tot, accs = 0.0, []
    for idx in torch.arange(b, device=x.device).chunk(micro):
        out_r = disc(x[idx], scale[idx])
        out_f = disc(xh[idx].detach(), scale[idx])
        loss = d_loss(out_r, out_f, w[idx], wsum)
        loss.backward()
        tot += float(loss.detach())
        accs.append(pair_acc(out_r, out_f).detach())
    return tot, torch.cat(accs)


def gp_backward(disc, x, xh, scale, gamma, lazy=4, items=4):
    k = min(items, x.shape[0])
    tot = 0.0
    for v in (x[:k], xh[:k]):
        xs = disc.prep(v.detach(), scale[:k]).requires_grad_(True)
        out = disc.core(xs, bf16=False)
        s = sum(lg.mean(1).sum() for lg, _ in out)
        g, = torch.autograd.grad(s, xs, create_graph=True)
        pen = lazy * gamma * 0.5 * g.square().flatten(1).sum(1).mean()
        pen.backward()
        tot += float(pen.detach())
    return tot


def n_params(m):
    return sum(p.numel() for p in m.parameters())

