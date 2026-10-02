# Step-invariant hoist for the graphed DiT

Every sampler step runs the whole DiT `_forward` inside one CUDA graph replay. About a fifth of that replay recomputes tensors that depend only on the prompt conditioning and the latent length. The hoist computes those tensors once per `sam.generate` call, parks them in static buffers, and patches the modules that would have recomputed them so the graph reads the buffers. Output stays bit-identical because the same module calls run on the same tensor values, shapes, strides and dtypes.

Nothing here has run on the GPU yet. The files compile and a symbol-table pass found no undefined names. `bench_hoist.py` is the gate.

## What the model looks like

The facts below come from the cached config at `F:\AI\cache\hf\hub\models--stabilityai--stable-audio-3-small-sfx\snapshots\ae12755283df9d62ca39a9b050a39a0b607b8c20\model_config.json` and from the installed source. The DiT has embed_dim 1024, depth 20, 16 heads (head dim 64), cond_token_dim 768, local_add_cond_dim 257, global_cond_type adaLN, qk_norm rms, 64 memory tokens, differential off. The context is 257 tokens: 256 T5 tokens plus 1 seconds_total token (`cross_attention_cond_ids: [prompt, seconds_total]`). With `duration_padding_s 1.0` the latent is 22 frames, so the self-attention sequence is 64 + 22 = 86 tokens. `flash_attn` is absent from the venv, so `Attention.apply_attn` takes the SDPA fallback at transformer.py:667-699. In torch 2.6 `rms_norm` ships only as a composite op (`torch/include/ATen/ops/rms_norm_compositeimplicitautograd_dispatch.h`, no fused kernel), so every q or k norm is about eight small kernels.

## Which ops are invariant

All paths are relative to `D:\Python\SAGE\.venv\Lib\site-packages\stable_audio_3\`.

The sampler builds the conditioning once. `model.py:297` calls `get_conditioning_inputs`, `model.py:306-311` casts to fp16, `sampling.py:469-477` packs it into `common_kwargs` once, and `sampling.py:178` passes the same dict on every step. Inside `DiffusionTransformer.forward` the CFG branch rebuilds the batch each step from those same tensors: `batch_local_add_cond = cat([lc, lc])` at dit.py:496-499, `null_embed = zeros_like(cross_attn_cond)` at dit.py:512, `batch_cond = cat([cond, null])` at dit.py:526, then `self._forward(...)` at dit.py:554-568. None of these depend on x or t, so the values match on every step of one generate.

| Op | Where | Inputs | Hoisted |
|---|---|---|---|
| `to_cond_embed` (Linear 768 to 1024, SiLU, Linear 1024 to 1024) | dit.py:70-74, called at dit.py:197-198 | `cross_attn_cond` [2B, 257, 768] | yes, part `cond` |
| cross-attn `to_kv`, chunk, rearrange | transformer.py:553, 719, 733-734 | context = `to_cond_embed` output | yes, part `kv` |
| cross-attn `k_norm` (RMSNorm(64), fp16) | transformer.py:575-576, 589-595, 750-751 | k from `to_kv` | yes, part `kv` |
| `to_local_embed` (Linear 257 to 1024, SiLU, Linear) and left pad to 86 | transformer.py:946-951, 978-982, pad at 77-88 | `local_add_cond` rearranged at dit.py:224-225 | yes, part `local` |
| rotary table `forward_from_seq_len(86)` | transformer.py:1197-1198, 270-294 | sequence length only | opt-in, part `rope` |
| per layer `ratio * freqs`, `cos`, `sin` for q and k | transformer.py:755-781, 302-322 | freqs only | opt-in, part `rope` |
| `to_global_embed` on seconds_total | dit.py:200-202 | `global_embed` [2B, 768] | no, three tiny kernels |
| padding mask cast for V zeroing | transformer.py:675-677, extension at 1214-1232 | padding mask | no, about 20 tiny kernels per step |

Cross-attention gets no rotary embedding: `cross_attn_rotary_pos_emb` defaults to off (transformer.py:1113-1116) and the block calls `cross_attn` without `rotary_pos_emb` (transformer.py:1031-1035). Memory tokens are concatenated into x only (transformer.py:1193-1195), so the 257-token K/V never include them. The timestep path (dit.py:235-239), `preprocess_conv` (dit.py:264) and the adaLN `global_cond_embedder` all depend on t or x and stay in the graph.

## Injection point

`install_hoist(sam, accelerator)` swaps `dit._forward` (currently `Accelerator.graph`, a `GraphedForward`) for a `Hoist` object that holds two graph wrappers. The plain one is the accelerator graph, unchanged. The hoisted one is a fresh `GraphedForward` built from the same original `_forward`, so the two never share graph entries. A wrapper on `sam.generate` bumps an epoch counter and marks the generate scope. DiT calls outside that scope go to the plain graph, which keeps any direct caller safe from stale buffers.

On the first DiT call of an epoch for a given input shape key, the hoist fills a slot: it runs the original `to_cond_embed` on `kw["cross_attn_cond"]`, then for each of the 20 cross-attention layers runs `to_kv`, `chunk`, `rearrange` and `k_norm` exactly as `Attention.forward` does, then `to_local_embed` on the rearranged `local_add_cond` followed by `_left_pad_to_match(emb, 64 + T)`. The fill is itself captured as a small CUDA graph per slot (eager fallback if capture fails), so a refill is one launch with stable output addresses. The hoisted step graph then replays with `hoist.active` pointing at that slot.

Four instance-level patches read `hoist.active` and fall through to the original code whenever it is None or a shape check fails:

1. `dit.to_cond_embed.forward` returns the static E when the input shape and dtype match the slot.
2. Each cross-attention `forward` skips `to_kv` and `k_norm` when `context is slot.E`, uses the cached `(k, v)`, and runs the rest of `Attention.forward` line for line (`to_q`, `q_norm`, `apply_attn`, merge heads, `to_out`).
3. Each block `_apply_local_conditioning` returns `x + padded_local` when the shapes match and `modular_local_cond` is None.
4. With `rope`, `transformer.rotary_pos_emb.forward_from_seq_len` returns the static `(freqs, 1.)` and each self-attention `forward` applies rotary from precomputed cos and sin tables with the same expression `(t * cos * 1) + (rotate_half(t) * sin * 1)`.

Identity checks (`context is slot.E`, `rotary_pos_emb[0] is slot.freqs`) are Python-level and run only during capture, so the graph holds no value-dependent branch.

Install is skipped with a reason string when the accelerator has no graph, LoRA is present, `ENABLE_TORCH_COMPILE=1`, the DiT layout differs (patch size, transformer type), or no layer qualifies. Any exception during fill or the hoisted forward marks that slot failed and routes it to the plain graph from then on. `uninstall()` restores only patches it still owns.

## Why the output is bit-identical

The fill reads the same tensor `_forward` receives, which is the CFG-concatenated `[2B, 257, 768]` batch with cond rows first and zero rows second. `to_kv` therefore runs on the full `[2B, 257, 1024]` E with GEMM M = 2B x 257, the same shape eager uses for both the cond and null rows. The cached v stays a view into the full `to_kv` output (storage offset 1024 elements, strides (257·2048, 64, 2048, 1)) and k is the `rms_norm` output, so SDPA receives identical layouts and alignment and selects the same backend. The local term is added with the same `x + pad(emb)` operands. Reordering independent ops changes no bits, and `check_fast_equiv.py` already showed these exact module calls give identical results inside and outside a CUDA graph. The rope tables reproduce `freqs.to(float32)`, the `1.0 * freqs` ratio product and the slice before `cos` and `sin`, so the rotated q and k match too.

The invariance claim rests on reading the sampler code. `verify=True` checks it at run time: it snapshots `cross_attn_cond` and `local_add_cond` at each fill and compares every later step with `torch.equal`, counting violations. The bench runs one verify pass per config.

The generate wrapper bumps the epoch both on entry and on exit, so the first DiT call after any generate returns refills, a nested generate included. A failure while building the prompt-cache key turns the cache off for that call and logs the error. The bench also runs a prompt-alternation pass: every job once per round, two rounds, no warm call between prompts, each output compared with the pre-install reference. Verify mode compares steps inside one generate, so this pass is the check for a buffer left over from the previous prompt.

## Expected savings at the 22-frame latent

The current fast path measures 67.8 ms at 3 steps and 166.9 ms at 8 steps (`scripts\bench\out_fast_equiv.json`), so one rows-2 DiT step costs about 19.8 ms and the fixed part (decode and overhead) about 8.4 ms. The old probe at 76 frames (`scripts\bench\out_bench_graph.json`) measured, per step, 2.95 ms for the 20 cross `to_kv` GEMMs, 0.14 ms for `to_cond_embed` and 2.09 ms for `to_local_embed`. The GEMM cost does not depend on latent length. The local figure came from an eager probe that was likely launch bound, and at 22 frames the in-graph cost should be 0.1 to 0.4 ms. The cross `k_norm` was never measured; at roughly 160 small kernels per step on a [2, 16, 257, 64] tensor I estimate 0.3 to 1.0 ms. That puts the invariant work at about 3.5 to 4.5 ms per step.

The fill still pays that cost once per generate, so the saving is (steps − 1) times the per-step figure:

| Case | Today | Estimated saving |
|---|---|---|
| batch 1, 3 steps | 67.8 ms | 7 to 9 ms |
| batch 1, 8 steps (water) | 166.9 ms | 25 to 31 ms |
| fast tier, 2 steps | 47.8 ms (`out_gate_fast_tier.json`) | 3.5 to 4.5 ms |
| batch 2, 3 steps | 126.0 ms | 13 to 17 ms (the GEMMs double at 4 rows) |
| `rope` add-on | | about 0.25 ms per step, 0.6 to 0.9 ms at 3 steps |
| `cache_prompts=N` add-on, repeated prompt at batch 1 | | a further 3 to 4 ms per generate |

The prompt cache is opt-in. It keys on the identity, data pointer, version, shape and dtype of the `CondCache` tensors passed as `conditioning_tensors`, holds strong references so ids are never recycled, and copies about 50 MB per hit back into the static buffers. Each cached prompt costs about 50 MB of VRAM at batch 1. It is off for batch 2 (the prompt path) and whenever inpaint or init audio arguments appear. The sidecar replays the same few prompts with different seeds, so this is where the remaining third of the invariant work goes.

## Decode prefix

`decode_prefix.py` patches `pretransform.decode` inside `sam.generate` to decode only the first `ceil(keep / 4096) + 3` latents, rounded up to the decoder chunk alignment of 2 (`chunk_size 32 // stride 16`). For a 1 s keep that is 14 of 22 latents. Downstream code already handles a shorter decode: `sampling.py:517-518` trims the audio mask and `model.py:347-350` truncates to the duration. It is off by default, changes low bits, and needs a paired quality gate.

The roughly 8 ms figure for this idea came from the 76-latent era (16.2 ms full, 8.1 ms prefix in `out_bench_graph.json`). A straight line through those two points gives about 0.13 ms per latent plus about 6.3 ms fixed, so going from 22 to 14 latents should save only about 1 ms. With the whole fixed part of a 3-step clip near 8.4 ms, I would expect the gate to show about 1 ms of gain for a nonzero quality risk. `bench_hoist.py` times it both decode-only and end to end so the number is measured. The decoder adds `randn_like * 0.01` noise to its new tokens (autoencoders.py:146-147) in `(b n)` order. At batch 2 the second clip's draws therefore shift with the prefix; at batch 1 they may line up, but I have not checked this.

## Risks

1. Stale buffers if a future change varies conditioning between steps inside one generate, for example a per-step negative prompt, a moving inpaint mask, or a `cfg_interval` that turns CFG off mid-run. CFG on and off use different shape keys and refill separately, but same-shape value changes would go unnoticed in production. The verify pass catches this in the bench.
2. VRAM: about 71 MB of static buffers per rows-2 slot and 142 MB per rows-4 slot, a second set of step graphs (the plain graph stays alive for calls outside generate), and 50 MB per cached prompt when the cache is on.
3. First-call latency: each new shape key captures one fill graph and one hoisted step graph. The startup warm-up should call `generate` once per shape after `install_hoist`.
4. Installing `flash_attn` later would move `apply_attn` to another branch. Both paths still call the same `apply_attn`, but the bench should be rerun.
5. The patches are instance attributes on live modules. Anything else that patches the same `forward` attributes would conflict. Install the hoist right after the `Accelerator`, install the decode prefix after the hoist, and uninstall in reverse order, as the bench does. Out of order, `uninstall()` leaves a pass-through `generate` wrapper in the chain, which costs nothing but stays.
6. Thread safety matches `GraphedForward`: one generate at a time, which the generator lock already enforces.
7. The fast tier at 2 steps saves one step's worth of invariant work, half of what it spends on it today.

## Follow-ups worth measuring

Graphing the decoder at a fixed 22-latent shape probably beats the prefix, because most of the 8.4 ms fixed cost looks like launch overhead. The decoder's `randn_like` would then run under graph-safe RNG, and whether that matches eager draws is unverified. Hoisting the padding-mask cast and `to_global_embed` would save roughly another 0.05 ms per step, too little to justify more patch surface.
