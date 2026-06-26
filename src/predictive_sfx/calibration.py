from __future__ import annotations


class Calibrator:
    def __init__(self, n_bins: int = 10, prior_strength: float = 5.0, min_samples: int = 20):
        self.n_bins = int(n_bins)
        self.prior_strength = float(prior_strength)
        self.min_samples = int(min_samples)
        self.bin_hits = [0.0] * self.n_bins
        self.bin_total = [0.0] * self.n_bins
        self.hits = 0.0
        self.total = 0.0

    def _bin(self, raw: float) -> int:
        b = int(raw * self.n_bins)
        if b < 0:
            return 0
        if b >= self.n_bins:
            return self.n_bins - 1
        return b

    def observe(self, raw: float, hit: bool) -> None:
        b = self._bin(max(0.0, min(1.0, float(raw))))
        self.bin_total[b] += 1.0
        self.total += 1.0
        if hit:
            self.bin_hits[b] += 1.0
            self.hits += 1.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def calibrate(self, raw: float) -> float:
        raw = max(0.0, min(1.0, float(raw)))
        if self.total < self.min_samples:
            return raw
        g = self.hits / self.total
        b = self._bin(raw)
        return (self.bin_hits[b] + self.prior_strength * g) / (self.bin_total[b] + self.prior_strength)


class PredictorCalibration:
    def __init__(self, **kw):
        self._kw = kw
        self.cals: dict = {}

    def _cal(self, source: str) -> Calibrator:
        if source not in self.cals:
            self.cals[source] = Calibrator(**self._kw)
        return self.cals[source]

    def observe(self, source: str, raw: float, hit: bool) -> None:
        self._cal(source).observe(raw, hit)

    def calibrate(self, source: str, raw: float) -> float:
        return self._cal(source).calibrate(raw)

    def hit_rate(self, source: str) -> float:
        return self._cal(source).hit_rate
