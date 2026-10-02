from __future__ import annotations

import functools
import inspect


def decoder_align(sam) -> int:
    try:
        dec = sam.model_config["model"]["pretransform"]["config"]["decoder"]["config"]
        return max(1, int(dec.get("chunk_size", 128)) // int(dec["strides"][0]))
    except Exception:
        return 1


def prefix_len(keep_samples: int, total: int, ds: int, margin: int = 3, align: int = 1) -> int:
    need = -(-int(keep_samples) // int(ds)) + int(margin)
    need = -(-need // max(1, int(align))) * max(1, int(align))
    return max(1, min(int(total), need))


def prefix_decode(sam, z, keep_s: float, margin: int = 3, chunked=None):
    pt = sam.model.pretransform
    n = prefix_len(int(keep_s * sam.model.sample_rate), z.shape[-1], pt.downsampling_ratio, margin,
                   decoder_align(sam))
    return pt.decode(z[..., :n], chunked=chunked)


class DecodePrefix:
    def __init__(self, sam, margin: int = 3):
        self.sam = sam
        self.margin = int(margin)
        self.enabled = True
        self.installed = False
        self.reason = None
        self.keep = None
        self.depth = 0
        self.patches = []
        self.errors = []
        self.counts = {"prefix": 0, "full": 0}
        try:
            self.reason = self._install()
        except Exception as e:
            self.reason = ("install failed: %r" % (e,))[:400]
        if self.reason is not None:
            self._restore()
            self.installed = False

    def _install(self):
        cdm = getattr(self.sam, "model", None)
        pt = getattr(cdm, "pretransform", None) if cdm is not None else None
        if pt is None or not hasattr(pt, "decode") or not hasattr(pt, "downsampling_ratio"):
            return "no pretransform"
        self.pt = pt
        self.ds = int(pt.downsampling_ratio)
        self.sr = int(cdm.sample_rate)
        self.align = decoder_align(self.sam)
        self.sig = inspect.signature(type(self.sam).generate)
        for name in ("duration", "return_latents", "truncate_output_to_duration"):
            if name not in self.sig.parameters:
                return "generate has no %s" % name
        gen = self.sam.generate
        dec = pt.decode

        @functools.wraps(gen)
        def generate(*a, **k):
            if not self.installed:
                return gen(*a, **k)
            prev = self.keep
            self.keep = self._keep(a, k)
            self.depth += 1
            try:
                return gen(*a, **k)
            finally:
                self.depth -= 1
                self.keep = prev

        def decode(z, *a, **k):
            keep = self.keep
            if self.installed and self.enabled and self.depth > 0 and keep:
                n = prefix_len(keep, z.shape[-1], self.ds, self.margin, self.align)
                if n < z.shape[-1]:
                    self.counts["prefix"] += 1
                    return dec(z[..., :n], *a, **k)
            self.counts["full"] += 1
            return dec(z, *a, **k)

        self._set(self.sam, "generate", generate)
        self._set(pt, "decode", decode)
        self.installed = True
        return None

    def _keep(self, a, k):
        try:
            args = self.sig.bind_partial(self.sam, *a, **k).arguments
        except TypeError:
            return None
        p = self.sig.parameters
        rl = args.get("return_latents", p["return_latents"].default)
        tr = args.get("truncate_output_to_duration", p["truncate_output_to_duration"].default)
        if rl or not tr:
            return None
        d = args.get("duration", p["duration"].default)
        if isinstance(d, (int, float)) and not isinstance(d, bool):
            return int(d * self.sr)
        if isinstance(d, (list, tuple)) and d and all(x == d[0] for x in d):
            return int(d[0] * self.sr)
        return None

    def _set(self, obj, name, value):
        had = name in obj.__dict__
        old = obj.__dict__.get(name)
        setattr(obj, name, value)
        self.patches.append((obj, name, had, old, value))

    def _restore(self):
        for obj, name, had, old, new in reversed(self.patches):
            try:
                if obj.__dict__.get(name) is not new:
                    continue
                if had:
                    setattr(obj, name, old)
                else:
                    delattr(obj, name)
            except Exception as e:
                self.errors.append(("restore %s: %r" % (name, e))[:400])
        self.patches = []

    def stats(self) -> dict:
        return {"installed": self.installed, "enabled": self.enabled, "reason": self.reason,
                "margin": self.margin, "align": getattr(self, "align", None), **self.counts,
                "errors": self.errors[-5:]}

    def uninstall(self):
        self._restore()
        self.installed = False


def install_decode_prefix(sam, margin: int = 3) -> DecodePrefix:
    return DecodePrefix(sam, margin=margin)
