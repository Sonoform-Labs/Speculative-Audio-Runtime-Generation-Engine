# Audio-Gen Model Survey (June 2026)

Goal: find a better generator than `stable-audio-open-small` for local SFX on an RTX 3060 (12 GB).
Method: multi-agent web survey + adversarial verification. Repos below confirmed to exist on HF
(`huggingface_hub.model_info`).

## Headline: Stable Audio 3 is real and ships an SFX-specialized model

`stabilityai/stable-audio-3-small-sfx` (Stability AI, 2026-05-20) — a **sound-effects-specialized** small
model trained on a curated Freesound subset (the small tier was split into Music and SFX variants because
mixing SFX into the music model degraded it). This directly targets our confirmed weakness (music-trained
"knock" character, ~62% material). ~0.6 B (~2 GB VRAM), same 8-step pingpong sampler, gated open-weight,
Stability AI Community License (free commercial under $1M revenue).

## Ranked shortlist (local SFX on a 3060)

| Model | Open weights | Runs on 3060 | SFX quality | Speed | License | Verdict |
|---|---|---|---|---|---|---|
| **`stabilityai/stable-audio-3-small-sfx`** | gated | yes (~2 GB) | strong (SFX-specialized) | 8-step pingpong, ~sub-1s | Community (free <$1M) | **try first** |
| `stabilityai/stable-audio-3-medium` (1.4B) | gated | yes (~5–6.5 GB) | ok (music+SFX) | 8-step, ~1–3s on 3060 | Community | fallback if small-sfx falls short |
| `stabilityai/stable-audio-open-small` (current) | gated | yes (~582 ms) | weak (~62%, "knock") | 8-step, 582 ms | Community | A/B control arm |
| Woosh-DFlow (Sony) | open | yes (~4–6 GB) | strong SFX foundation | 4-step distilled | CC-BY-**NC** | research ceiling only (non-commercial) |
| MeanAudio-L (480M) | open | yes | ok (general) | 1-step, RTF ~0.013 | **MIT** | best-licensed fast fallback |
| TangoFlux (515M) | open | yes | ok–strong (44.1k) | 25–50 step flow | non-commercial | quality reference |
| AudioGen / MAGNeT (Meta) | open | yes | ok (16 kHz) | AR slow / non-AR | CC-BY-**NC** | skip (license + 16 kHz) |
| HunyuanVideo-Foley (Tencent) | open | offload only | strong but **video→audio** | 50-step | restricted | skip (wrong modality) |
| ElevenLabs SFX V2 | cloud | no | strong | <2s (network) | per-gen | cloud fallback only |
| `facebook/audiobox-aesthetics` | open | yes | N/A (scorer) | ms | CC-BY | **already in our panel** — keep as quality filter |

## Plan: swap in SA3 Small SFX and A/B vs the current model

1. Accept the `stable-audio-3-small-sfx` gated license on HF + `huggingface-cli login`.
2. Try loading it via the existing `stable-audio-tools` backend (`get_pretrained_model(...)`, just change
   `model_name`). If SA3's config isn't supported, install the official `stable-audio-3` library and add a
   `StableAudio3Generator` (new lib, minor glue; no diffusers pipeline yet).
3. Run the 4-judge panel head-to-head: `small-sfx` vs `open-small` on the same materials. The decision is
   our own footstep/material/quality numbers, NOT the vendor FAD/CLAP.

## Honest caveats

- Vendor quality (BBC-SFX FAD 0.395 / CLAP ~0.35) is general-benchmark, not our footstep-material probe —
  confirm the win with our own A/B.
- 3060 latency for SA3 is extrapolated from H200/M4 figures; measure real RTF before trusting the
  real-time story (medium especially could be 1–3 s/clip).
- Both SA3 repos are **gated** (same friction as current); license is commercial-OK only under $1M rev.
