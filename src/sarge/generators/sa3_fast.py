from __future__ import annotations

from collections import OrderedDict

import torch


class GraphedForward:
    def __init__(self, orig):
        self.orig = orig
        self.pool = torch.cuda.graph_pool_handle()
        self.entries = {}
        self.failed = {}
        self.captured = 0
        self.replays = 0
        self.eager = 0

    def _key(self, x, t, kw):
        parts = [("x", tuple(x.shape), str(x.dtype)), ("t", tuple(t.shape), str(t.dtype))]
        for k in sorted(kw):
            v = kw[k]
            if torch.is_tensor(v):
                parts.append((k, tuple(v.shape), str(v.dtype)))
            elif v is None or isinstance(v, (bool, int, float, str)):
                parts.append((k, v))
            else:
                return None
        return tuple(parts)

    def _capture(self, x, t, kw):
        sx = x.clone()
        st = t.clone()
        skw = {k: v.clone() for k, v in kw.items() if torch.is_tensor(v)}
        consts = {k: v for k, v in kw.items() if not torch.is_tensor(v)}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.orig(sx, st, **skw, **consts)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
            out = self.orig(sx, st, **skw, **consts)
        return {"graph": graph, "x": sx, "t": st, "kw": skw, "out": out}

    def __call__(self, x, t, **kw):
        kw["use_checkpointing"] = False
        key = self._key(x, t, kw)
        if key is None or kw.get("return_info") or kw.get("exit_layer_ix") is not None or key in self.failed:
            self.eager += 1
            return self.orig(x, t, **kw)
        ent = self.entries.get(key)
        if ent is None:
            try:
                ent = self._capture(x, t, kw)
            except Exception as e:
                self.failed[key] = repr(e)[:400]
                torch.cuda.synchronize()
                self.eager += 1
                return self.orig(x, t, **kw)
            self.entries[key] = ent
            self.captured += 1
        ent["x"].copy_(x)
        ent["t"].copy_(t)
        for k, v in ent["kw"].items():
            v.copy_(kw[k])
        ent["graph"].replay()
        self.replays += 1
        return ent["out"].clone()


class CondCache:
    def __init__(self, cdm, device, max_prompts: int = 256):
        self.cdm = cdm
        self.device = device
        self.max_prompts = int(max_prompts)
        self.store: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _one(self, prompt, seconds):
        k = (prompt, float(seconds))
        ent = self.store.get(k)
        if ent is None:
            self.misses += 1
            with torch.inference_mode():
                out = self.cdm.conditioner([{"prompt": prompt, "seconds_total": seconds}], self.device)
            ent = {n: [x.clone() if torch.is_tensor(x) else x for x in v] for n, v in out.items()}
            self.store[k] = ent
            while len(self.store) > self.max_prompts:
                self.store.popitem(last=False)
        else:
            self.hits += 1
            self.store.move_to_end(k)
        return ent

    def get(self, prompts, seconds):
        cond = [{"prompt": p, "seconds_total": seconds} for p in prompts]
        parts = [self._one(p, seconds) for p in prompts]
        if len(parts) == 1:
            return cond, dict(parts[0])
        merged = {}
        for n in parts[0]:
            merged[n] = [torch.cat([p[n][i] for p in parts], 0) for i in range(len(parts[0][n]))]
        return cond, merged


class Accelerator:
    def __init__(self, sam, graph: bool = True, max_prompts: int = 256):
        import stable_audio_3.models.dit as dit_mod
        self.sam = sam
        self.dit = sam.dit.model
        self.cache = CondCache(sam.model, str(sam.device), max_prompts=max_prompts)
        self.lora_free = not bool(dit_mod.has_lora(self.dit))
        if self.lora_free:
            dit_mod.has_lora = lambda model: False
        self.graph = None
        if graph and torch.cuda.is_available() and str(sam.device).startswith("cuda"):
            self.graph = GraphedForward(self.dit._forward)
            self.dit._forward = self.graph

    def kwargs(self, prompts, seconds, negative_prompt=None):
        kw = {"disable_tqdm": True, "use_checkpointing": False}
        if len(prompts) > 1:
            kw["prompt"] = list(prompts)
            if negative_prompt:
                kw["negative_prompt"] = negative_prompt
            return kw
        cond, ct = self.cache.get(prompts, seconds)
        kw["conditioning"] = cond
        kw["conditioning_tensors"] = ct
        if negative_prompt:
            ncond, nct = self.cache.get([negative_prompt] * len(prompts), seconds)
            kw["negative_conditioning"] = ncond
            kw["negative_conditioning_tensors"] = nct
        return kw

    def stats(self) -> dict:
        g = self.graph
        return {"cond_hits": self.cache.hits, "cond_misses": self.cache.misses,
                "graphs": g.captured if g else 0, "replays": g.replays if g else 0,
                "eager": g.eager if g else 0, "capture_failures": len(g.failed) if g else 0}
