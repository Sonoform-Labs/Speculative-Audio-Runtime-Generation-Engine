import math
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

SOFT = 8.0
FLOOR = 0.05
NAN = float("nan")
_COMBOS = {}


def robust_stats(V):
    V = np.asarray(V, np.float64)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(V, axis=0)
            mad = np.nanmedian(np.abs(V - med), axis=0)
            sd = np.nanstd(V, axis=0)
            q99 = np.nanquantile(V, 0.99, axis=0)
    scale = np.maximum(np.maximum(1.4826 * np.nan_to_num(mad), 0.25 * np.nan_to_num(sd)), 1e-6)
    return med, scale, q99


def directional(V, st, dirs):
    med, scale, q99 = st
    z = (V - med) / scale
    two, low, high, cap = (dirs == "two"), (dirs == "low"), (dirs == "high"), (dirs == "lowcap")
    d = np.zeros_like(z)
    with np.errstate(invalid="ignore"):
        d[:, two] = np.abs(z[:, two])
        d[:, low] = np.maximum(-z[:, low], 0.0)
        d[:, high] = np.maximum(z[:, high], 0.0)
        d[:, cap] = np.maximum(-z[:, cap], 0.0) + np.maximum((V[:, cap] - (q99[cap] + 6.0)) / scale[cap], 0.0)
    zp = SOFT * np.tanh(d / SOFT)
    zp[np.isnan(V)] = SOFT
    return zp


def raw_z(V, st):
    med, scale, _q = st
    return (V - med) / scale


class Ref:
    def __init__(self, V, clusters, names, dirs, emb=None, max_nan=0.10):
        self.V = np.asarray(V, np.float64)
        self.names = list(names)
        self.cl = np.asarray(clusters, dtype=str)
        self.src, self.si = np.unique(self.cl, return_inverse=True)
        self.nanfrac = np.mean(np.isnan(self.V), axis=0) if len(self.V) else np.ones(len(names))
        self.valid = self.nanfrac < max_nan
        self.dirs = np.asarray([dirs[n] for n in names])
        self.full = robust_stats(self.V)
        self.loso = [robust_stats(self.V[self.si != k]) for k in range(len(self.src))]
        self.emb = emb
        self.col = {n: i for i, n in enumerate(self.names)}

    @property
    def k(self):
        return len(self.src)

    def src_of(self, cluster):
        hit = np.flatnonzero(self.src == str(cluster))
        return int(hit[0]) if len(hit) else -1

    def cols(self, feats):
        return [self.col[f] for f in feats if f in self.col and self.valid[self.col[f]]]


class Bank:
    def __init__(self, ref, Vq, kq):
        self.ref = ref
        self.Vq = np.asarray(Vq, np.float64).reshape(-1, len(ref.names))
        self.kq = np.asarray(kq, int)
        self.real_zp = np.empty_like(ref.V)
        self.real_z = np.empty_like(ref.V)
        for k in range(ref.k):
            m = ref.si == k
            self.real_zp[m] = directional(ref.V[m], ref.loso[k], ref.dirs)
            self.real_z[m] = raw_z(ref.V[m], ref.loso[k])
        self.q_zp = np.zeros((ref.k, len(self.Vq), len(ref.names)))
        self.q_z = np.zeros((ref.k, len(self.Vq), len(ref.names)))
        for k in range(ref.k):
            self.q_zp[k] = directional(self.Vq, ref.loso[k], ref.dirs)
            self.q_z[k] = raw_z(self.Vq, ref.loso[k])
        self._zqm = None
        self.wq = np.zeros((len(self.Vq), ref.k))
        for i, k in enumerate(self.kq):
            if k >= 0:
                self.wq[i, k] = 1.0
            else:
                self.wq[i, :] = 1.0 / ref.k

    def lat(self, feats, bounded=False):
        cols = self.ref.cols(feats)
        if not cols:
            return None
        a_r = np.sqrt(np.mean(self.real_zp[:, cols] ** 2, axis=1))
        norm = float(np.median(a_r))
        norm = max(norm, 1e-9)
        lr = np.log(np.maximum(a_r, FLOOR) / norm)
        if len(self.Vq):
            a_q = np.sqrt(np.mean(self.q_zp[:, :, cols] ** 2, axis=2))
            lq = np.sum(np.log(np.maximum(a_q, FLOOR) / norm).T * self.wq, axis=1)
        else:
            lq = np.zeros(0)
        out = {"real_raw": lr, "q_raw": lq, "norm": norm, "cols": [self.ref.names[c] for c in cols]}
        if bounded:
            lr, lq = np.maximum(lr, 0.0), np.maximum(lq, 0.0)
        out.update({"real": lr, "q": lq})
        return out

    def zq_mean(self):
        if self._zqm is None:
            self._zqm = np.einsum("kqf,qk->qf", self.q_z, self.wq)
        return self._zqm

    def zpq_mean(self):
        return np.einsum("kqf,qk->qf", self.q_zp, self.wq)


def knn_lat(E_ref, si_ref, K, Eq, kq, k=5, floor_rel=0.05):
    E_ref = np.asarray(E_ref, np.float64)
    D = 1.0 - E_ref @ E_ref.T
    a_r = np.empty(len(E_ref))
    for i in range(len(E_ref)):
        d = D[i, si_ref != si_ref[i]]
        kk = min(k, len(d))
        a_r[i] = float(np.mean(np.partition(d, kk - 1)[:kk]))
    norm = max(float(np.median(a_r)), 1e-9)
    fl = floor_rel * norm
    lr = np.log(np.maximum(a_r, fl) / norm)
    lq = np.zeros(len(Eq))
    if len(Eq):
        Dq = 1.0 - np.asarray(Eq, np.float64) @ E_ref.T
        per = np.zeros((len(Eq), K))
        for s in range(K):
            d = Dq[:, si_ref != s]
            kk = min(k, d.shape[1])
            per[:, s] = np.mean(np.partition(d, kk - 1, axis=1)[:, :kk], axis=1)
        lp = np.log(np.maximum(per, fl) / norm)
        kq = np.asarray(kq, int)
        for i, s in enumerate(kq):
            lq[i] = lp[i, s] if s >= 0 else float(np.mean(lp[i]))
    return {"real": lr, "q": lq, "norm": norm}


def kad(X, Y, perms=200, seed=0):
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    if len(X) < 2 or len(Y) < 2:
        return {"kad": NAN, "p": NAN}
    Z = np.vstack([X, Y])
    sq = np.sum(Z * Z, 1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
    bw = float(np.median(D2[np.triu_indices(len(Z), 1)])) or 1.0
    Kz = np.exp(-D2 / bw)
    n = len(X)

    def stat(idx):
        a, b = idx[:n], idx[n:]
        kxx = Kz[np.ix_(a, a)]
        kyy = Kz[np.ix_(b, b)]
        kxy = Kz[np.ix_(a, b)]
        m1, m2 = len(a), len(b)
        return (kxx.sum() - np.trace(kxx)) / (m1 * (m1 - 1)) + (kyy.sum() - np.trace(kyy)) / (m2 * (m2 - 1)) - \
            2 * kxy.mean()
    base = np.arange(len(Z))
    s0 = stat(base)
    rng = np.random.default_rng(seed)
    cnt = sum(stat(rng.permutation(len(Z))) >= s0 for _ in range(perms))
    return {"kad": float(s0), "p": float((cnt + 1) / (perms + 1))}


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return NAN
    allv = np.concatenate([pos, neg])
    from scipy.stats import rankdata
    r = rankdata(allv)
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def cliff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return NAN
    d = b[None, :] - a[:, None]
    return float((np.sum(d > 0) - np.sum(d < 0)) / d.size)


def cliff_strat(pairs):
    num, den = 0.0, 0.0
    for a, b in pairs:
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b[None, :] - a[:, None]
        num += np.sum(d > 0) - np.sum(d < 0)
        den += d.size
    return float(num / den) if den else NAN


def scale_ci(pt, lo, hi, n):
    if n < 2 or not np.isfinite(pt):
        return [lo, hi]
    s = math.sqrt(n / (n - 1.0))
    return [float(pt - (pt - lo) * s), float(pt + (hi - pt) * s)]


def pctl(v, q):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if len(v) else NAN


def median_pairwise(Z):
    if len(Z) < 2:
        return NAN
    d = np.sqrt(np.maximum(np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2), 0.0))
    iu = np.triu_indices(len(Z), 1)
    return float(np.median(d[iu]))


def distinct_draw(rng, si, n):
    srcs = np.unique(si)
    if len(srcs) >= n:
        pick = rng.choice(srcs, n, replace=False)
        return np.array([rng.choice(np.flatnonzero(si == s)) for s in pick])
    base = [rng.choice(np.flatnonzero(si == s)) for s in srcs]
    rest = rng.choice(np.setdiff1d(np.arange(len(si)), base), n - len(base), replace=False)
    return np.concatenate([np.asarray(base, int), rest])


def null_draws(si, n, draws, seed):
    rng = np.random.default_rng(seed)
    return [distinct_draw(rng, np.asarray(si), n) for _ in range(draws)]


def collapse_null(Zr, draws_idx):
    D = np.array([median_pairwise(Zr[ix]) for ix in draws_idx])
    return {"q05": pctl(D, 5), "q50": pctl(D, 50), "q95": pctl(D, 95)}


def penalty(Dg, cn):
    if not np.isfinite(Dg) or not np.isfinite(cn["q05"]):
        return 0.0
    if Dg < cn["q05"]:
        return float(math.log(cn["q50"] / max(Dg, 1e-9)))
    return 0.0


def summarize_R(L_pt, L_b, bounded, n):
    if bounded:
        L_pt = max(L_pt, 0.0)
        L_b = np.maximum(L_b, 0.0)
    R_pt = math.exp(-L_pt)
    Rb = np.exp(-L_b)
    lo, hi = pctl(Rb, 2.5), pctl(Rb, 97.5)
    ci = scale_ci(R_pt, lo, hi, n)
    return {"R": R_pt, "L": L_pt, "ci": ci, "se_L": float(np.nanstd(L_b))}


def _combos(n, m):
    k = (n, m)
    if k not in _COMBOS:
        _COMBOS[k] = np.array(list(combinations(range(n + m), n)), dtype=np.int16)
    return _COMBOS[k]


def group_D(Dm, G):
    n = G.shape[1]
    iu = np.array(list(combinations(range(n), 2)))
    if not len(iu):
        return np.full(len(G), np.nan)
    a = G[:, iu[:, 0]]
    b = G[:, iu[:, 1]]
    return np.median(Dm[a, b], axis=1)


def dist_matrix(Z, kind="z"):
    Z = np.asarray(Z, np.float64)
    if kind == "emb":
        return 1.0 - Z @ Z.T
    return np.sqrt(np.maximum(np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2), 0.0))


def perm_test(a, b, Dm=None, cn=None, exact_max=20000, mc=10000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    v = np.concatenate([a, b])
    use_pen = Dm is not None and cn is not None
    total = math.comb(na + nb, na)
    if total <= exact_max:
        GA = _combos(na, nb).astype(int)
        mask = np.ones((len(GA), na + nb), bool)
        mask[np.arange(len(GA))[:, None], GA] = False
        GB = np.nonzero(mask)[1].reshape(len(GA), nb)
        exact = True
    else:
        rng = np.random.default_rng(seed)
        P = np.argsort(rng.random((mc, na + nb)), axis=1)
        GA = np.vstack([np.arange(na)[None, :], P[:, :na]])
        GB = np.vstack([np.arange(na, na + nb)[None, :], P[:, na:]])
        exact = False
    st = v[GB].mean(1) - v[GA].mean(1)
    if use_pen:
        pa = np.array([penalty(d, cn) for d in group_D(Dm, GA)])
        pb = np.array([penalty(d, cn) for d in group_D(Dm, GB)])
        st = st + pb - pa
    obs = st[0]
    return {"stat": float(obs), "p": float(np.mean(st >= obs - 1e-12)), "p_rev": float(np.mean(st <= obs + 1e-12)),
            "exact": exact, "n": int(len(st))}


def strat_perm(strata, mc=10000, seed=0, pen=True):
    rng = np.random.default_rng(seed)
    tot = np.zeros(mc + 1)
    for s in strata:
        a, b = np.asarray(s["a"], float), np.asarray(s["b"], float)
        na, nb = len(a), len(b)
        v = np.concatenate([a, b])
        P = np.argsort(rng.random((mc, na + nb)), axis=1)
        GA = np.vstack([np.arange(na)[None, :], P[:, :na]])
        GB = np.vstack([np.arange(na, na + nb)[None, :], P[:, na:]])
        st = v[GB].mean(1) - v[GA].mean(1)
        if pen and s.get("Dm") is not None and s.get("cn") is not None:
            pa = np.array([penalty(d, s["cn"]) for d in group_D(s["Dm"], GA)])
            pb = np.array([penalty(d, s["cn"]) for d in group_D(s["Dm"], GB)])
            st = st + pb - pa
        tot += st
    obs = tot[0]
    return {"stat": float(obs), "p": float((np.sum(tot[1:] >= obs - 1e-12) + 1) / (mc + 1)),
            "p_rev": float((np.sum(tot[1:] <= obs + 1e-12) + 1) / (mc + 1))}


def mc_less(gen, real, mc=2000, seed=0):
    gen, real = np.asarray(gen, float), np.asarray(real, float)
    if not len(gen) or not len(real):
        return NAN
    v = np.concatenate([gen, real])
    ng = len(gen)
    obs = gen.mean() - real.mean()
    rng = np.random.default_rng(seed)
    P = np.argsort(rng.random((mc, len(v))), axis=1)
    d = v[P[:, :ng]].mean(1) - v[P[:, ng:]].mean(1)
    return float((np.sum(d <= obs + 1e-12) + 1) / (mc + 1))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return NAN
    return float(spearmanr(a, b).correlation)


def dose_rules(clean, doses):
    clean = np.asarray(clean, float)
    D = np.vstack([clean] + [np.asarray(d, float) for d in doses]).T
    rise = float(np.mean(D[:, -1] > D[:, 0]))
    sp = []
    x = np.arange(D.shape[1])
    for row in D:
        if np.allclose(row, row[0]):
            sp.append(0.0)
        else:
            sp.append(float(spearmanr(x, row).correlation))
    sp = np.asarray(sp)
    return {"rise_frac": rise, "spearman_pos_frac": float(np.mean(sp > 0)), "delta": cliff(D[:, 0], D[:, -1]),
            "mean_increase": float(np.mean(D[:, -1] - D[:, 0])), "n": int(len(D))}


def dose_pass(r, th):
    return bool(r["n"] > 0 and r["rise_frac"] >= th["rise"] and r["spearman_pos_frac"] >= th["spearman"] and
                r["delta"] >= th["delta"])


def pca_fit(X, k=16):
    X = np.asarray(X, np.float64)
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    k = min(k, Vt.shape[0])
    return mu, Vt[:k]


def pca_apply(p, X):
    mu, W = p
    return (np.asarray(X, np.float64) - mu) @ W.T


def lda_cv(X, y, groups):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import balanced_accuracy_score
    X, y, groups = np.asarray(X, float), np.asarray(y), np.asarray(groups)
    pred = np.empty(len(y), dtype=y.dtype)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            pred[te] = y[tr][0] if tr.any() else y[te][0]
            continue
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return float(balanced_accuracy_score(y, pred))


def lda_loo(X, y):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X, y = np.asarray(X, float), np.asarray(y)
    hit = []
    for i in range(len(y)):
        tr = np.ones(len(y), bool)
        tr[i] = False
        if len(np.unique(y[tr])) < 2:
            hit.append(False)
            continue
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X[tr], y[tr])
        hit.append(m.predict(X[i:i + 1])[0] == y[i])
    return float(np.mean(hit)) if hit else NAN


def sep_S(X, labels):
    labs = sorted(set(labels))
    labels = np.asarray(labels)
    cents = [X[labels == l].mean(0) for l in labs]
    if len(cents) < 2:
        return NAN
    between = np.mean([np.linalg.norm(cents[i] - cents[j]) for i, j in combinations(range(len(cents)), 2)])
    within = np.mean([np.mean(np.linalg.norm(X[labels == l] - c, axis=1)) for l, c in zip(labs, cents)])
    return float(between / max(within, 1e-9))


def j8d_fit(X_tr, y_tr, w_tr=None, C=0.1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_tr)
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(sc.transform(X_tr), y_tr)
    return sc, m


def j8d_prob(model, X):
    sc, m = model
    return m.predict_proba(sc.transform(np.asarray(X, float)))[:, list(m.classes_).index(1)]


def split_half_spearman(score_fn, models, seeds, splits=50, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    n = len(seeds)
    for _ in range(splits):
        p = rng.permutation(n)
        A, B = [seeds[i] for i in p[:n // 2]], [seeds[i] for i in p[n // 2: 2 * (n // 2)]]
        sa = [score_fn(m, A) for m in models]
        sb = [score_fn(m, B) for m in models]
        out.append(spearman(sa, sb))
    v = np.asarray(out, float)
    return {"median": float(np.nanmedian(v)) if np.isfinite(v).any() else NAN, "nan_frac": float(np.mean(~np.isfinite(v)))}
