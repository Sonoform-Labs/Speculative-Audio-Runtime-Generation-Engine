from types import SimpleNamespace

import pytest
import torch

from sarge.generators.stable_audio3 import StableAudio3Generator, _SpanTorch


def test_span_crop_equals_the_leading_frames_of_the_full_draw():
    p = _SpanTorch(torch, 10)
    torch.manual_seed(3)
    full = torch.randn([2, 4, 10])
    torch.manual_seed(3)
    part = p.randn([2, 4, 6])
    assert part.shape == (2, 4, 6) and part.is_contiguous() and torch.equal(part, full[..., :6])
    x = torch.zeros(1, 4, 6, dtype=torch.float16)
    torch.manual_seed(4)
    full_like = torch.randn_like(torch.zeros(1, 4, 10, dtype=torch.float16))
    torch.manual_seed(4)
    part_like = p.randn_like(x)
    assert part_like.dtype == torch.float16 and torch.equal(part_like, full_like[..., :6])


def test_span_passes_through_full_length_and_other_shapes():
    p = _SpanTorch(torch, 10)
    for size in ([1, 4, 10], [1, 4, 12], [5], [3, 6]):
        torch.manual_seed(7)
        a = p.randn(size)
        torch.manual_seed(7)
        assert torch.equal(a, torch.randn(size))
    torch.manual_seed(7)
    a = p.randn(5)
    torch.manual_seed(7)
    assert torch.equal(a, torch.randn(5))
    x = torch.zeros(1, 4, 10)
    torch.manual_seed(8)
    a = p.randn_like(x)
    torch.manual_seed(8)
    assert torch.equal(a, torch.randn_like(x))
    assert p.float16 is torch.float16 and p.manual_seed is torch.manual_seed


def test_generator_defaults_keep_noise_and_hoist_unchanged():
    g = StableAudio3Generator(model_name="medium")
    assert g.noise_span is None and g.hoist_diff is False
    with g._noise():
        pass
    g2 = StableAudio3Generator(model_name="medium", noise_span=[3, 6], hoist_diff=True)
    assert g2.noise_span == (3.0, 6.0) and g2.hoist_diff is True


def _layer(differential, feat_scale=False):
    a = SimpleNamespace(to_q=object(), to_kv=object(), differential=differential, feat_scale=feat_scale,
                        qk_norm="rms")
    return SimpleNamespace(cross_attend=True, cross_attn=a)


def test_kv_hoist_skips_differential_layers_unless_asked():
    from sarge.generators.sa3_hoist import _kv_layers
    tr = SimpleNamespace(layers=[_layer(False), _layer(True), _layer(True, feat_scale=True),
                                 SimpleNamespace(cross_attend=False)])
    assert [i for i, _a in _kv_layers(tr)] == [0]
    assert [i for i, _a in _kv_layers(tr, diff=True)] == [0, 1]


@pytest.mark.parametrize("differential", [False, True])
def test_hoisted_cross_attention_matches_the_module(differential):
    tm = pytest.importorskip("stable_audio_3.models.transformer")
    from sarge.generators.sa3_hoist import Hoist, _kv
    torch.manual_seed(11)
    a = tm.Attention(128, dim_heads=32, dim_context=128, zero_init_output=False, qk_norm="rms",
                     differential=differential).eval()
    x = torch.randn(1, 10, 128)
    e = torch.randn(1, 7, 128)
    with torch.no_grad():
        ref = a(x, context=e)
        h = Hoist.__new__(Hoist)
        h.active = SimpleNamespace(out={"E": e, "kv": {0: _kv(a, e)}})
        out = h._make_cross(0, a, a.forward)(x, context=e)
    assert torch.equal(out, ref)


def test_gemm_shapes_keep_small_sfx_and_add_medium():
    from sarge.generators.sa3_gemm import CONFIGS, DEFAULT_SHAPES, MEDIUM_SHAPES, SMALL_SHAPES
    assert SMALL_SHAPES == ((1024, 8192), (1024, 3072), (4096, 1024)) and DEFAULT_SHAPES[:3] == SMALL_SHAPES
    assert set(MEDIUM_SHAPES) <= set(DEFAULT_SHAPES) and not set(MEDIUM_SHAPES) & set(SMALL_SHAPES)
    assert all(s in CONFIGS for s in DEFAULT_SHAPES) and all(len(CONFIGS[s]) == 5 for s in SMALL_SHAPES)
    assert {c[5] for c in CONFIGS.values() if len(c) > 5} == {"f32"}


def test_hybrid_guns_build_the_gated_medium_arm(monkeypatch):
    import os
    monkeypatch.syspath_prepend(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import run_sidecar
    kw = dict(run_sidecar.TIERS["default"], fast_load=True)
    kw.update(run_sidecar.HYBRID_GUNS)
    g = StableAudio3Generator(**kw)
    assert g.model_name == "medium" and g.steps == 8 and g.sampler_type == "pingpong" and g.cfg_scale == 1.0
    assert g.noise_span == (3.0, 6.0) and g.duration_padding_s == 1.0 and g.gen_duration_s == 3.0
    assert g.hoist_diff is True and g.triton_gemm and g.fp16_accumulation
