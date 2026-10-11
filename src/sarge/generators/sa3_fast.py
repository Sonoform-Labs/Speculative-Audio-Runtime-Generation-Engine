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


class GraphedDecode:
    def __init__(self, orig):
        self.orig = orig
        self.pool = torch.cuda.graph_pool_handle()
        self.entries = {}
        self.failed = {}
        self.captured = 0
        self.replays = 0

    def _capture(self, z, chunked):
        sz = z.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.orig(sz, chunked=chunked)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
            out = self.orig(sz, chunked=chunked)
        return {"graph": graph, "z": sz, "out": out}

    def __call__(self, z, chunked=None, **kw):
        if kw or not torch.is_tensor(z) or not z.is_cuda:
            return self.orig(z, chunked=chunked, **kw)
        key = (tuple(z.shape), str(z.dtype), chunked)
        if key in self.failed:
            return self.orig(z, chunked=chunked)
        ent = self.entries.get(key)
        if ent is None:
            try:
                ent = self._capture(z, chunked)
            except Exception as e:
                self.failed[key] = repr(e)[:400]
                torch.cuda.synchronize()
                return self.orig(z, chunked=chunked)
            self.entries[key] = ent
            self.captured += 1
        ent["z"].copy_(z)
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


class PrefixDecode:
    def __init__(self, orig, keep_frames: int, ratio: int):
        self.orig = orig
        self.keep = int(keep_frames)
        self.ratio = int(ratio)
        self.cut = 0

    def __call__(self, z, chunked=None, **kw):
        if kw or not torch.is_tensor(z) or z.dim() != 3 or z.shape[-1] <= self.keep:
            return self.orig(z, chunked=chunked, **kw)
        n = z.shape[-1]
        y = self.orig(z[..., :self.keep].contiguous(), chunked=chunked)
        self.cut += 1
        return torch.nn.functional.pad(y, (0, n * self.ratio - y.shape[-1]))


class GraphedText:
    def __init__(self, orig):
        self.orig = orig
        self.pool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self.entries = {}
        self.failed = {}
        self.captured = 0
        self.replays = 0

    def __getattr__(self, name):
        return getattr(self.orig, name)

    def _capture(self, ids, mask):
        sids = ids.clone()
        smask = mask.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.orig(input_ids=sids, attention_mask=smask)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
            out = self.orig(input_ids=sids, attention_mask=smask)["last_hidden_state"]
        return {"graph": graph, "ids": sids, "mask": smask, "out": out}

    def __call__(self, input_ids=None, attention_mask=None, **kw):
        if (kw or not torch.is_tensor(input_ids) or not torch.is_tensor(attention_mask) or not input_ids.is_cuda
                or self.pool is None):
            return self.orig(input_ids=input_ids, attention_mask=attention_mask, **kw)
        key = (tuple(input_ids.shape), str(input_ids.dtype), tuple(attention_mask.shape), str(attention_mask.dtype))
        if key in self.failed:
            return self.orig(input_ids=input_ids, attention_mask=attention_mask)
        ent = self.entries.get(key)
        if ent is None:
            if bool(attention_mask.all()):
                return self.orig(input_ids=input_ids, attention_mask=attention_mask)
            try:
                ent = self._capture(input_ids, attention_mask)
            except Exception as e:
                self.failed[key] = repr(e)[:400]
                torch.cuda.synchronize()
                return self.orig(input_ids=input_ids, attention_mask=attention_mask)
            self.entries[key] = ent
            self.captured += 1
        ent["ids"].copy_(input_ids)
        ent["mask"].copy_(attention_mask)
        ent["graph"].replay()
        self.replays += 1
        return {"last_hidden_state": ent["out"].clone()}


def install_text_graph(cdm) -> list:
    out = []
    conds = getattr(getattr(cdm, "conditioner", None), "conditioners", None) or {}
    for c in conds.values():
        inner = vars(c).get("model")
        if inner is None or isinstance(inner, GraphedText) or not hasattr(c, "tokenizer"):
            continue
        g = GraphedText(inner)
        c.__dict__["model"] = g
        out.append(g)
    return out


class Accelerator:
    def __init__(self, sam, graph: bool = True, max_prompts: int = 256, compile_mode: str | None = None,
                 compile_decode: bool = False, graph_decode: bool = False, graph_text: bool = False,
                 triton_gemm: bool = False, decode_keep_frames: int | None = None, triton_proj: bool = False):
        import stable_audio_3.models.dit as dit_mod
        self.sam = sam
        self.dit = sam.dit.model
        self.cache = CondCache(sam.model, str(sam.device), max_prompts=max_prompts)
        self.lora_free = not bool(dit_mod.has_lora(self.dit))
        if self.lora_free:
            dit_mod.has_lora = lambda model: False
        self.compile_mode = compile_mode
        self.triton_linears = []
        if triton_gemm and torch.cuda.is_available() and str(sam.device).startswith("cuda"):
            from .sa3_gemm import CONFIGS, DEFAULT_SHAPES, install_triton_gemm
            shapes = tuple(CONFIGS) if triton_proj else DEFAULT_SHAPES
            self.triton_linears = install_triton_gemm(self.dit, shapes)
        fwd = self.dit._forward
        if compile_mode:
            fwd = torch.compile(fwd, mode=compile_mode, dynamic=False)
        self.graph = None
        if graph and torch.cuda.is_available() and str(sam.device).startswith("cuda"):
            self.graph = GraphedForward(fwd)
            self.dit._forward = self.graph
        elif compile_mode:
            self.dit._forward = fwd
        self.decode_graph = None
        pt = getattr(sam.model, "pretransform", None)
        self.prefix = None
        if pt is not None and (compile_decode or graph_decode or decode_keep_frames):
            dec = pt.decode
            if compile_decode:
                dec = torch.compile(dec, mode=compile_mode or "default", dynamic=False)
            if graph_decode and torch.cuda.is_available() and str(sam.device).startswith("cuda"):
                self.decode_graph = GraphedDecode(dec)
                dec = self.decode_graph
            ratio = getattr(pt, "downsampling_ratio", None)
            if decode_keep_frames and ratio:
                self.prefix = PrefixDecode(dec, decode_keep_frames, ratio)
                dec = self.prefix
            pt.decode = dec
        self.text_graphs = []
        if graph_text and torch.cuda.is_available() and str(sam.device).startswith("cuda"):
            self.text_graphs = install_text_graph(sam.model)

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
                "eager": g.eager if g else 0, "capture_failures": len(g.failed) if g else 0,
                "decode_graphs": self.decode_graph.captured if self.decode_graph else 0,
                "decode_replays": self.decode_graph.replays if self.decode_graph else 0,
                "decode_failures": len(self.decode_graph.failed) if self.decode_graph else 0,
                "text_graphs": sum(t.captured for t in self.text_graphs),
                "text_replays": sum(t.replays for t in self.text_graphs),
                "text_failures": sum(len(t.failed) for t in self.text_graphs),
                "triton_linears": len(self.triton_linears),
                "prefix_decodes": self.prefix.cut if self.prefix else 0}
