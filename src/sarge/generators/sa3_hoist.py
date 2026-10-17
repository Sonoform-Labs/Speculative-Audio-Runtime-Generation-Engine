from __future__ import annotations

import functools
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
from einops import rearrange

from ..startup import stage

DEFAULT_PARTS = ("cond", "kv", "local")
ALL_PARTS = ("cond", "kv", "local", "rope")
_NORMS = ("none", "l2", "ln", "rms", "dyt")
_UNCACHED = ("init_audio", "inpaint_audio", "inpaint_mask", "inpaint_mask_start_seconds",
             "inpaint_mask_end_seconds", "negative_prompt")
_SNAP = ("cross_attn_cond", "local_add_cond")


def _qn(attn, q):
    if attn.qk_norm == "l2":
        return F.normalize(q, dim=-1, eps=attn.qk_norm_eps)
    if attn.qk_norm != "none":
        return attn.q_norm(q).to(q.dtype)
    return q


def _kn(attn, k):
    if attn.qk_norm == "l2":
        return F.normalize(k, dim=-1, eps=attn.qk_norm_eps)
    if attn.qk_norm != "none":
        return attn.k_norm(k).to(k.dtype)
    return k


def _kv(attn, e):
    if attn.differential:
        k, k_diff, v = attn.to_kv(e).chunk(3, dim=-1)
        k, k_diff, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=attn.kv_heads), (k, k_diff, v))
        return _kn(attn, torch.stack([k, k_diff], dim=1)), v
    k, v = attn.to_kv(e).chunk(2, dim=-1)
    k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=attn.kv_heads), (k, v))
    return _kn(attn, k), v


def _kv_layers(tr, diff: bool = False) -> list:
    out = []
    for i, layer in enumerate(tr.layers):
        a = getattr(layer, "cross_attn", None) if getattr(layer, "cross_attend", False) else None
        if (a is not None and hasattr(a, "to_q") and hasattr(a, "to_kv") and (diff or not a.differential)
                and not a.feat_scale and a.qk_norm in _NORMS):
            out.append((i, a))
    return out


def _rot(tm, t, cos, sin):
    with torch.autocast("cuda", enabled=False):
        out_dtype = t.dtype
        dtype = torch.promote_types(torch.promote_types(t.dtype, cos.dtype), torch.float32)
        t = t.to(dtype)
        rot_dim = cos.shape[-1]
        t, t_un = t[..., :rot_dim], t[..., rot_dim:]
        t = (t * cos * 1) + (tm.rotate_half(t) * sin * 1)
        t, t_un = t.to(out_dtype), t_un.to(out_dtype)
        return torch.cat((t, t_un), dim=-1)


def _clone_out(out):
    return {"E": None if out["E"] is None else out["E"].clone(),
            "kv": {i: (k.clone(), v.clone()) for i, (k, v) in out["kv"].items()},
            "local": {i: p.clone() for i, p in out["local"].items()},
            "rope": out["rope"]}


def _copy_out(dst, src):
    if dst["E"] is not None:
        dst["E"].copy_(src["E"])
    for i, (k, v) in dst["kv"].items():
        k.copy_(src["kv"][i][0])
        v.copy_(src["kv"][i][1])
    for i, p in dst["local"].items():
        p.copy_(src["local"][i])
    if dst["rope"] is not None and src["rope"] is not None and dst["rope"] is not src["rope"]:
        for n in ("freqs", "qcos", "qsin", "kcos", "ksin"):
            dst["rope"][n].copy_(src["rope"][n])


def _equal_out(a, b):
    if (a["E"] is None) != (b["E"] is None):
        return False
    if a["E"] is not None and not torch.equal(a["E"], b["E"]):
        return False
    if set(a["kv"]) != set(b["kv"]) or set(a["local"]) != set(b["local"]):
        return False
    for i in a["kv"]:
        if not (torch.equal(a["kv"][i][0], b["kv"][i][0]) and torch.equal(a["kv"][i][1], b["kv"][i][1])):
            return False
    for i in a["local"]:
        if not torch.equal(a["local"][i], b["local"][i]):
            return False
    return True


class _Slot:
    def __init__(self, key):
        self.key = key
        self.epoch = -1
        self.failed = None
        self.graph = None
        self.inp = None
        self.out = None
        self.snap = None
        self.n = 0
        self.ctx_shape = None
        self.ctx_dtype = None
        self.lt_shape = None
        self.cache = OrderedDict()


class Hoist:
    def __init__(self, sam, accelerator, parts=DEFAULT_PARTS, graph_fill: bool = True,
                 verify: bool = False, cache_prompts: int = 0, diff: bool = False):
        self.sam = sam
        self.acc = accelerator
        self.diff = bool(diff)
        self.graph_fill = bool(graph_fill)
        self.verify = bool(verify)
        self.cache_size = max(0, int(cache_prompts))
        self.enabled = True
        self.installed = False
        self.reason = None
        self.parts = set()
        self.epoch = 0
        self.depth = 0
        self.active = None
        self.ckey = None
        self.crefs = None
        self.slots = {}
        self.plain = None
        self.hoisted = None
        self.patches = []
        self.kv_layers = []
        self.local_layers = []
        self.rope_layers = []
        self.cond_fwd = None
        self.rope_fwd = None
        self.errors = []
        self.counts = {"hoisted": 0, "plain": 0, "fills": 0, "fill_graphs": 0, "eager_fills": 0,
                       "cache_hits": 0, "cache_mismatch": 0, "violations": 0}
        try:
            self.reason = self._install(tuple(parts))
        except Exception as e:
            self.reason = ("install failed: %r" % (e,))[:400]
        if self.reason is not None:
            self._restore()
            self.installed = False

    def _install(self, parts):
        import stable_audio_3.models.transformer as tm
        import stable_audio_3.models.utils as um
        self.tm = tm
        if not torch.cuda.is_available():
            return "cuda unavailable"
        acc = self.acc
        g = getattr(acc, "graph", None) if acc is not None else None
        dit = getattr(acc, "dit", None) if acc is not None else None
        if g is None or dit is None:
            return "accelerator has no cuda graph"
        if dit.__dict__.get("_forward") is not g:
            return "dit._forward is not the accelerator graph"
        if not getattr(acc, "lora_free", False):
            return "lora present"
        if getattr(um, "enable_torch_compile", False):
            return "torch.compile enabled"
        if getattr(dit, "transformer_type", None) != "continuous_transformer" or getattr(dit, "patch_size", 1) != 1:
            return "unsupported dit layout"
        tr = getattr(dit, "transformer", None)
        if tr is None or not hasattr(tr, "layers"):
            return "no transformer layers"
        if not callable(getattr(self.sam, "generate", None)):
            return "model has no generate"
        want = set(p for p in parts if p in ALL_PARTS)
        if "kv" in want:
            want.add("cond")
        if "cond" in want and getattr(dit, "to_cond_embed", None) is None:
            want -= {"cond", "kv"}
        if "kv" in want and getattr(tr, "cross_attn_rotary_pos_emb", None) is None:
            self.kv_layers = _kv_layers(tr, self.diff)
        if not self.kv_layers:
            want.discard("kv")
        if "local" in want:
            self.local_layers = [(i, layer) for i, layer in enumerate(tr.layers)
                                 if getattr(layer, "to_local_embed", None) is not None
                                 and hasattr(layer, "_apply_local_conditioning")]
        if not self.local_layers:
            want.discard("local")
        rot = getattr(tr, "rotary_pos_emb", None)
        if "rope" in want and rot is not None and getattr(rot, "scale", None) is None:
            for i, layer in enumerate(tr.layers):
                a = getattr(layer, "self_attn", None)
                if (a is not None and hasattr(a, "to_qkv") and not a.differential and not a.feat_scale
                        and a.qk_norm in _NORMS and not getattr(layer, "add_rope", False)):
                    self.rope_layers.append((i, a))
        if not self.rope_layers:
            want.discard("rope")
        if not want:
            return "nothing to hoist"
        self.parts = want
        self.dit = dit
        self.tr = tr
        self.plain = g
        self.hoisted = type(g)(g.orig)
        self._set(dit, "_forward", self)
        self._patch_generate()
        if "cond" in want:
            self.cond_fwd = dit.to_cond_embed.forward
            self._set(dit.to_cond_embed, "forward", self._make_cond(self.cond_fwd))
        if "kv" in want:
            for i, a in self.kv_layers:
                self._set(a, "forward", self._make_cross(i, a, a.forward))
        if "local" in want:
            for i, layer in self.local_layers:
                self._set(layer, "_apply_local_conditioning", self._make_local(i, layer._apply_local_conditioning))
        if "rope" in want:
            self.rope_fwd = rot.forward_from_seq_len
            self._set(rot, "forward_from_seq_len", self._make_ffsl(self.rope_fwd))
            for i, a in self.rope_layers:
                self._set(a, "forward", self._make_self(a, a.forward))
        self.installed = True
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

    def _patch_generate(self):
        orig = self.sam.generate

        @functools.wraps(orig)
        def generate(*a, **k):
            if not self.installed:
                return orig(*a, **k)
            prev = (self.ckey, self.crefs)
            try:
                self.ckey, self.crefs = self._content(a, k)
            except Exception as e:
                self.errors.append(("cache key: %r" % (e,))[:400])
                self.ckey, self.crefs = None, None
            self.epoch += 1
            self.depth += 1
            try:
                return orig(*a, **k)
            finally:
                self.depth -= 1
                self.epoch += 1
                self.ckey, self.crefs = prev

        self._set(self.sam, "generate", generate)

    def _content(self, a, k):
        if self.cache_size <= 0 or a:
            return None, None
        for name in _UNCACHED:
            if k.get(name) is not None:
                return None, None
        ct = k.get("conditioning_tensors")
        nct = k.get("negative_conditioning_tensors")
        if not isinstance(ct, dict) or (k.get("negative_conditioning") is not None and nct is None):
            return None, None
        parts, refs = [], []
        for tag, d in (("c", ct), ("n", nct)):
            if d is None:
                parts.append((tag,))
                continue
            if not isinstance(d, dict):
                return None, None
            for name in sorted(d):
                v = d[name]
                sig = []
                for t in (v if isinstance(v, (list, tuple)) else [v]):
                    if torch.is_tensor(t):
                        try:
                            ver = t._version
                        except Exception:
                            ver = -1
                        sig.append((id(t), t.data_ptr(), ver, tuple(t.shape), str(t.dtype)))
                        refs.append(t)
                    elif t is None or isinstance(t, (bool, int, float, str)):
                        sig.append(t)
                    else:
                        return None, None
                parts.append((tag, name, tuple(sig)))
        return tuple(parts), refs

    def _make_cond(self, orig):
        def forward(input):
            s = self.active
            if s is not None and s.out is not None and s.out["E"] is not None:
                if tuple(input.shape) == s.ctx_shape and input.dtype == s.ctx_dtype:
                    return s.out["E"]
            return orig(input)
        return forward

    def _make_cross(self, i, a, orig):
        def forward(x, context=None, rotary_pos_emb=None, rotary_pos_emb_k=None, causal=None,
                    flex_attention_block_mask=None, flex_attention_score_mod=None,
                    flash_attn_sliding_window=None, padding_mask=None, varlen_metadata=None):
            s = self.active
            ent = None
            if (s is not None and s.out is not None and context is not None and context is s.out["E"]
                    and rotary_pos_emb is None and rotary_pos_emb_k is None):
                ent = s.out["kv"].get(i)
            if ent is None:
                return orig(x, context=context, rotary_pos_emb=rotary_pos_emb, rotary_pos_emb_k=rotary_pos_emb_k,
                            causal=causal, flex_attention_block_mask=flex_attention_block_mask,
                            flex_attention_score_mod=flex_attention_score_mod,
                            flash_attn_sliding_window=flash_attn_sliding_window, padding_mask=padding_mask,
                            varlen_metadata=varlen_metadata)
            k, v = ent
            if a.differential:
                q, q_diff = a.to_q(x).chunk(2, dim=-1)
                q, q_diff = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=a.num_heads), (q, q_diff))
                q, q_diff = _qn(a, torch.stack([q, q_diff], dim=1)).unbind(dim=1)
                k, k_diff = k.unbind(dim=1)
            else:
                q = rearrange(a.to_q(x), "b n (h d) -> b h n d", h=a.num_heads)
                q = _qn(a, q)
            causal = a.causal if causal is None else causal
            if q.shape[-2] == 1 and causal:
                causal = False
            out = a.apply_attn(q, k, v, causal=causal, flex_attention_block_mask=flex_attention_block_mask,
                               flex_attention_score_mod=flex_attention_score_mod,
                               flash_attn_sliding_window=flash_attn_sliding_window, padding_mask=padding_mask,
                               varlen_metadata=varlen_metadata)
            if a.differential:
                out = out - a.apply_attn(q_diff, k_diff, v, causal=causal,
                                         flex_attention_block_mask=flex_attention_block_mask,
                                         flex_attention_score_mod=flex_attention_score_mod,
                                         flash_attn_sliding_window=flash_attn_sliding_window,
                                         padding_mask=padding_mask, varlen_metadata=varlen_metadata)
            return a.to_out(rearrange(out, "b h n d -> b n (h d)"))
        return forward

    def _make_local(self, i, orig):
        def apply(x, local_add_cond, modular_local_cond):
            s = self.active
            if (s is not None and s.out is not None and modular_local_cond is None and local_add_cond is not None
                    and tuple(local_add_cond.shape) == s.lt_shape):
                p = s.out["local"].get(i)
                if p is not None and p.shape == x.shape:
                    return x + p
            return orig(x, local_add_cond, modular_local_cond)
        return apply

    def _make_ffsl(self, orig):
        def forward_from_seq_len(seq_len):
            s = self.active
            rp = s.out["rope"] if s is not None and s.out is not None else None
            if rp is not None and seq_len == rp["n"]:
                return rp["pair"]
            return orig(seq_len)
        return forward_from_seq_len

    def _make_self(self, a, orig):
        tm = self.tm

        def forward(x, context=None, rotary_pos_emb=None, rotary_pos_emb_k=None, causal=None,
                    flex_attention_block_mask=None, flex_attention_score_mod=None,
                    flash_attn_sliding_window=None, padding_mask=None, varlen_metadata=None):
            s = self.active
            rp = s.out["rope"] if s is not None and s.out is not None else None
            if (rp is None or context is not None or rotary_pos_emb is None or rotary_pos_emb_k is not None
                    or rotary_pos_emb[0] is not rp["freqs"] or x.shape[-2] != rp["n"]):
                return orig(x, context=context, rotary_pos_emb=rotary_pos_emb, rotary_pos_emb_k=rotary_pos_emb_k,
                            causal=causal, flex_attention_block_mask=flex_attention_block_mask,
                            flex_attention_score_mod=flex_attention_score_mod,
                            flash_attn_sliding_window=flash_attn_sliding_window, padding_mask=padding_mask,
                            varlen_metadata=varlen_metadata)
            q, k, v = a.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=a.num_heads), (q, k, v))
            if a.qk_norm == "l2":
                q = F.normalize(q, dim=-1, eps=a.qk_norm_eps)
                k = F.normalize(k, dim=-1, eps=a.qk_norm_eps)
            elif a.qk_norm != "none":
                q, k = a.apply_qk_layernorm(q, k)
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            q = _rot(tm, q, rp["qcos"], rp["qsin"])
            k = _rot(tm, k, rp["kcos"], rp["ksin"])
            q = q.to(v.dtype)
            k = k.to(v.dtype)
            causal = a.causal if causal is None else causal
            if q.shape[-2] == 1 and causal:
                causal = False
            out = a.apply_attn(q, k, v, causal=causal, flex_attention_block_mask=flex_attention_block_mask,
                               flex_attention_score_mod=flex_attention_score_mod,
                               flash_attn_sliding_window=flash_attn_sliding_window, padding_mask=padding_mask,
                               varlen_metadata=varlen_metadata)
            return a.to_out(rearrange(out, "b h n d -> b n (h d)"))
        return forward

    def _compute(self, ctx, lc, n):
        out = {"E": None, "kv": {}, "local": {}, "rope": None}
        if ctx is not None and "cond" in self.parts:
            e = self.cond_fwd(ctx)
            out["E"] = e
            if "kv" in self.parts:
                for i, a in self.kv_layers:
                    out["kv"][i] = _kv(a, e)
        if lc is not None and "local" in self.parts:
            lt = rearrange(lc, "b c t -> b t c")
            for i, layer in self.local_layers:
                out["local"][i] = self.tm._left_pad_to_match(layer.to_local_embed(lt), n)
        if "rope" in self.parts:
            freqs, scale = self.rope_fwd(n)
            f = freqs.to(torch.float32)
            kf = (n / n) * f
            out["rope"] = {"n": n, "pair": (freqs, scale), "freqs": freqs,
                           "qcos": f[-n:, :].cos(), "qsin": f[-n:, :].sin(),
                           "kcos": kf[-n:, :].cos(), "ksin": kf[-n:, :].sin()}
        return out

    def _capture_fill(self, slot, ctx, lc):
        inp = {"ctx": None if ctx is None else ctx.clone(), "lc": None if lc is None else lc.clone()}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._compute(inp["ctx"], inp["lc"], slot.n)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            out = self._compute(inp["ctx"], inp["lc"], slot.n)
        graph.replay()
        slot.graph, slot.inp, slot.out = graph, inp, out

    def _fill(self, slot, x, kw):
        ctx = kw.get("cross_attn_cond")
        lc = kw.get("local_add_cond")
        if slot.out is None:
            slot.n = int(getattr(self.tr, "num_memory_tokens", 0)) + int(x.shape[-1])
            slot.ctx_shape = tuple(ctx.shape) if ctx is not None else None
            slot.ctx_dtype = ctx.dtype if ctx is not None else None
            slot.lt_shape = (int(lc.shape[0]), int(lc.shape[2]), int(lc.shape[1])) if lc is not None else None
            if self.graph_fill:
                try:
                    t0 = time.time()
                    self._capture_fill(slot, ctx, lc)
                    self.counts["fill_graphs"] += 1
                    stage("hoist fill graph %d" % self.counts["fill_graphs"], t0)
                except Exception as e:
                    self.errors.append(("fill capture: %r" % (e,))[:400])
                    torch.cuda.synchronize()
                    slot.graph = slot.inp = slot.out = None
            if slot.out is None:
                slot.out = self._compute(ctx, lc, slot.n)
                self.counts["eager_fills"] += 1
        elif slot.graph is not None:
            if ctx is not None:
                slot.inp["ctx"].copy_(ctx)
            if lc is not None:
                slot.inp["lc"].copy_(lc)
            slot.graph.replay()
        else:
            _copy_out(slot.out, self._compute(ctx, lc, slot.n))
            self.counts["eager_fills"] += 1
        self.counts["fills"] += 1

    def _prepare(self, slot, x, kw):
        ck = self.ckey if self.cache_size > 0 else None
        if ck is not None and slot.out is not None:
            ent = slot.cache.get(ck)
            if ent is not None:
                slot.cache.move_to_end(ck)
                if self.verify:
                    fresh = self._compute(kw.get("cross_attn_cond"), kw.get("local_add_cond"), slot.n)
                    if not _equal_out(fresh, ent["out"]):
                        self.counts["cache_mismatch"] += 1
                        slot.cache.pop(ck, None)
                        ent = None
                if ent is not None:
                    _copy_out(slot.out, ent["out"])
                    self.counts["cache_hits"] += 1
                    return
        self._fill(slot, x, kw)
        if ck is not None:
            slot.cache[ck] = {"out": _clone_out(slot.out), "refs": self.crefs}
            slot.cache.move_to_end(ck)
            while len(slot.cache) > self.cache_size:
                slot.cache.popitem(last=False)

    def _snap(self, kw):
        return {n: kw[n].clone() for n in _SNAP if torch.is_tensor(kw.get(n))}

    def _same(self, snap, kw):
        for n in _SNAP:
            v = kw.get(n)
            if torch.is_tensor(v) != (n in snap):
                return False
            if n in snap and not torch.equal(snap[n], v):
                return False
        return True

    def _plain(self, x, t, kw):
        self.counts["plain"] += 1
        return self.plain(x, t, **kw)

    def __call__(self, x, t, **kw):
        if (not self.installed or not self.enabled or self.depth <= 0 or kw.get("return_info")
                or kw.get("exit_layer_ix") is not None or kw.get("modular_local_cond") is not None
                or kw.get("prepend_cond") is not None):
            return self._plain(x, t, kw)
        key = self.hoisted._key(x, t, dict(kw, use_checkpointing=False))
        if key is None or key in self.hoisted.failed:
            return self._plain(x, t, kw)
        slot = self.slots.get(key)
        if slot is None:
            slot = self.slots[key] = _Slot(key)
        if slot.failed is not None:
            return self._plain(x, t, kw)
        try:
            if slot.epoch != self.epoch:
                self._prepare(slot, x, kw)
                slot.epoch = self.epoch
                slot.snap = self._snap(kw) if self.verify else None
            elif self.verify:
                if slot.snap is None:
                    slot.snap = self._snap(kw)
                elif not self._same(slot.snap, kw):
                    self.counts["violations"] += 1
                    self._fill(slot, x, kw)
                    slot.snap = self._snap(kw)
        except Exception as e:
            slot.failed = ("prepare: %r" % (e,))[:400]
            self.errors.append(slot.failed)
            torch.cuda.synchronize()
            return self._plain(x, t, kw)
        self.active = slot
        try:
            out = self.hoisted(x, t, **kw)
            self.counts["hoisted"] += 1
            return out
        except Exception as e:
            slot.failed = ("forward: %r" % (e,))[:400]
            self.errors.append(slot.failed)
        finally:
            self.active = None
        torch.cuda.synchronize()
        return self._plain(x, t, kw)

    def stats(self) -> dict:
        g = self.hoisted
        return {"installed": self.installed, "enabled": self.enabled, "reason": self.reason,
                "parts": sorted(self.parts), **self.counts, "slots": len(self.slots),
                "cached": sum(len(s.cache) for s in self.slots.values()),
                "slot_failures": [s.failed for s in self.slots.values() if s.failed],
                "hoist_graphs": g.captured if g else 0, "hoist_replays": g.replays if g else 0,
                "hoist_eager": g.eager if g else 0,
                "hoist_capture_failures": list(g.failed.values()) if g else [],
                "errors": self.errors[-5:]}

    def uninstall(self):
        self._restore()
        self.installed = False
        self.active = None
        self.slots = {}
        if self.hoisted is not None:
            self.hoisted.entries = {}
            self.hoisted.failed = {}


def install_hoist(sam, accelerator, parts=DEFAULT_PARTS, graph_fill: bool = True, verify: bool = False,
                  cache_prompts: int = 0, diff: bool = False) -> Hoist:
    return Hoist(sam, accelerator, parts=parts, graph_fill=graph_fill, verify=verify, cache_prompts=cache_prompts,
                 diff=diff)
