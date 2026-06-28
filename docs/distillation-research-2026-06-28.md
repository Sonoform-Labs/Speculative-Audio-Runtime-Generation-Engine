# SA3-small-sfx few-step distillation — research synthesis (2026-06-28)

Goal: distill `stable-audio-3-small-sfx` from 8-step (~224ms @4-step on a 3060) to **2-step (~127ms)**
at usable quality, for short transient SFX (footsteps/impacts/gunshots). Hardware for training: free
Kaggle T4/P100 (16GB). Teacher quality ~76%.

Source: 6-agent deep-research workflow + synthesis. Empirical spike confirmed `stable_audio_3` ships
the full stable-audio-tools stack (training/diffusion.py teacher-distill trainer, native LoRA incl
DoRA/LoRA-XS, rectified-flow objective `{v, rectified_flow, rf_denoiser}`, samplers
euler/rk4/dpmpp/pingpong, 459M DiT + 108M VAE = 568M fits 16GB).

## Recommended method: ARC (Adversarial Relativistic-Contrastive). Fallback: RF Reflow (FlashAudio).

Why ARC beats consistency/LCM/reflow **for this case**:
- **Weak teacher (76%).** Trajectory-matching methods (LCM, consistency, MeanFlow-from-teacher, naive
  reflow) copy the teacher → capped at 76%. ARC/DMD2 train against REAL audio (relativistic
  adversarial + contrastive) → can match or **exceed** the teacher. Decisive for a weak teacher.
- **Transients need sharpness.** MSE/consistency regress to the conditional mean → smear attacks
  (the exact footstep/gunshot failure). Adversarial restores high-freq attack energy.
- **Native SA3 recipe** = flow-pretrain → brief MSE distill warmup → adversarial post-train. The
  rf_denoiser/teacher-warmup trainer in the package = stage 1; ARC = stage 2.
- **ARC was built + validated on Stable Audio Open Small** (497M, 8→1 step) — the near-twin model.
  arXiv:2505.08175.

Why not others: LCM/TCD inherit the ceiling + velocity-param mismatch on flow models. MeanFlow =
from-scratch (3d × 4 GPU). InstaFlow = 199 A100-days. DMD2 = strong runner-up but heavier VRAM than
ARC's single lightweight relativistic discriminator.

## Loss terms (latent space)
- Relativistic adversarial loss (the sharpness/quality engine).
- Contrastive discriminator loss (correct (audio,text) > shuffled-prompt) — replaces CFG.
- CLAP loss (semantic alignment; in the Stable Audio Open ARC recipe).
- SFX add-ons (inferred, for transients): multi-resolution STFT (~7 FFT sizes 32–2048) + onset-weighted
  term (OWBCE α≈12, σ≈7 frames). These need VAE-decode during training → VRAM cost; cut first on T4.

## LoRA setup
- LoRA on the student DiT (SA3 officially supports stackable LoRA; LoRA recovers >94–98% of full).
- Targets: attention `to_q/to_k/to_v/to_out` in self- AND cross-attn (cross-attn = text cond, needed
  for contrastive). Add FF in/out if quality short.
- Rank 16 start (LCM-LoRA ≈ 5–8% params), bump to 32–64 if 1-step underfits.
- Discriminator trained fully (lightweight heads on frozen teacher features, à la ADD/LADD).

## Teacher/student + CFG
- Student = SA3-SFX DiT + LoRA, init from teacher (stabilizes the GAN).
- Teacher = frozen, used only for the short MSE warmup (~10k steps) then discarded.
- CFG: none at inference — contrastive discriminator bakes prompt adherence into the single forward pass.

## Compute (T4/P100 16GB)
- Batch 1–4 + grad-accum to eff. 16–32; fp16 mandatory (P16 ≥ T4: no bf16 on T4).
- Fits 16GB at batch 1–2 with LoRA; VAE-decode STFT/onset losses push near/over → cut on T4.
- Data = REAL SFX audio (~few-thousand–10k curated, labeled footstep/impact/gunshot clips).
- Wall-clock ~2–5 GPU-days across multiple Kaggle sessions (≤9–12h each). Anchor: RAPM = SD1.5
  distill on 1×16GB, batch 1, ~8.3h, LoRA-only.

## Honest quality recovery (transient SFX)
- **4-step:** dependable ~95–98% / meets-or-beats teacher. BUT we already run 4-step → no latency win.
- **2-step (~127ms):** realistically approaches 8-step **with ARC** (85–95%); consistency would smear.
  This is the goal, and the regime adversarial was built to rescue.
- **1-step (~83ms):** ~85–90%, transients most at risk. Stretch.

## Verdict: SERIOUS 2–4 week project (not a weekend, not a dead end)
Not a dead end (ARC proven on the twin model, SA3 ships LoRA + the recipe, 4-step ≈ teacher reliable).
Not a weekend (GAN on one 16GB card across Kaggle session limits is finicky — expect failed runs —
plus real transient-loss iteration).

Top 2 risks:
1. Adversarial instability on a single 16GB card. Mitigate: init from teacher, lightweight
   discriminator, grad-accum, keep the GAN-free RF-reflow fallback ready.
2. The 76% teacher caps prompt-adherence even if adversarial fixes sharpness. Mitigate: contrastive +
   CLAP loss, curate clean labeled SFX data — data quality is the real lever, not step count.

## Staged path
- Stage 0 (free, ~30 min): swap Euler → dpmpp/rk4 at 2–3 steps on the existing model. No training.
  Measures the free quality headroom (maybe gets to an acceptable 3-step ~172ms with zero work).
- Stage 1 (days): MSE-warmup LoRA distill (native rf_denoiser) → working 4-step student. GAN-free,
  dependable. Quality bump, no latency win — scaffold for ARC.
- Stage 2 (weeks): ARC adversarial post-train → 2-step at quality. The real ~100ms-at-quality unlock.

## Key citations
- ARC / Stable Audio Open Small 1-step — arXiv:2505.08175
- FlashAudio (RF reflow, 1-step FAD 1.49) — arXiv:2410.12266
- AudioLCM — arXiv:2406.00356 ; ConsistencyTTA — arXiv:2309.10740
- ADD — arXiv:2311.17042 ; MeanFlow / MeanAudio (1-step flow)
- SA3 LoRA + adversarial post-training pipeline — arXiv:2605.17991 (per synthesis)
