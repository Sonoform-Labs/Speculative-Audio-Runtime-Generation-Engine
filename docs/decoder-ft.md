# Decoder fine-tune (dec_data.py, dec_train.py, dec_eval.py, dec_listen.py)

Small-sfx decodes every latent through SAME-S, and a real recording passed through SAME-S alone already loses its fine transient structure: the kurtosis of the band above 4 kHz falls from 84 to 24 on gravel steps and from 534 to 99 on stone steps, while the energy in that band survives. Decoded noise also carries a comb at multiples of 172 Hz (44100 / 256, the patch size), +30.7 dB against 1.9 dB for the reference noise. The route fine-tunes the decoder alone, with the encoder and bottleneck frozen, so the generator's latents keep their meaning and decode time stays at 1.83 ms. Design, reviews and the build report sit in `sarge-research/decoder_ft/`.

## Data and losses

Training uses 4,127 FoleySet and C3GD clips (3.1 h, both CC BY 4.0), validation 598 clips, and evaluation 750 held-out clips. Every held-out uploader, gun and identical-audio group is excluded from training and validation, and the evaluation picker skips any FoleySet audio also filed under a training uploader or under several sub-categories (`dec_data.py`). ESC-50 and Sonniss audio appear nowhere. The losses are multi-resolution STFT on mid and side, a multi-period and a multi-band STFT discriminator with feature matching, and two anchors that hold the decoder near the base output on recorded DiT latents below 2 kHz and in coarse high-band envelopes.

## Runs

`ft_v1` ran with runtime-level bottleneck noise (`--bn-noise eval`) and stopped at step 5000 of 14000, when the validation score had risen for three checks in a row. `ft_v2` added `--w-gen-adv 0.05`, an unpaired critic term on decodes of DiT latents, and `--lat-compress-prob 0.3`, which shrinks encoded latents toward their clip mean by 0.4 to 0.6. That mimics the DiT latents, whose within-clip spread is 0.49 of encoded audio at the same lag-1 autocorrelation (`gen_latents_v1/latent_stats.json`). It ran 4000 main steps, resumed once with `--anchor-batch 4` after its memory guard stopped it at 9.28 GB.

## Results

| Checkpoint | held-out grain error (base 0.383) | comb on decoded noise (base 30.7 dB) | comb on generated decodes (base 21.0 dB) |
|---|---|---|---|
| ft_v1 step 3000 | 0.327 [-0.099, -0.020] | 21.5 | 13.1 |
| ft_v1 step 5000 | 0.245 [-0.205, -0.082] | 14.9 | 10.3 |
| ft_v2 step 2500 | 0.364 [-0.050, 0.006] | 26.5 | 16.5 |
| ft_v2 step 3500 | 0.319 [-0.106, -0.010] | 17.3 | 10.6 |
| ft_v2 step 6000 | 0.255 [-0.195, -0.068] | 13.2 | 8.8 |

Grain error is the median absolute log ratio of high-band kurtosis to the original over 358 held-out clips, with a 95% interval on the change; every artifact check (hiss, aliasing, flutter, pre-echo) passes for every checkpoint (`data/decoder_ft/eval/*/verdict.json`). The grain gain on real audio does not carry to generated sounds. On the six transient keys the new decoders move generated audio toward real kurtosis on some keys and away on others, stone steps most of all (-0.19 to -0.53), so the generator's latents set most of what the decoder can rebuild. The comb drop does carry: every key falls by 6 to 14 dB on generated decodes, which makes it the change most likely to be audible in the engine.

On the held-out judges (`out_vocab` tables under `data/decoder_ft/logs/`), the decoders lift rifle from 0.78 to 0.84 to 0.89, horn from 0.63 to 0.69 to 0.80, dog from 0.90 to 0.95 and door slam from 0.95 to 1.00, and every one drops stone steps (0.75 to 0.36 to 0.81, all but ft_v2 step 2500 below base). Glass, explosion, sword and grass stay at 0 under every decoder. Two checkpoints go to the ear: ft_v1 step 3000, the best balance on the judges (pistol 0.71, clicks 0.70, punch 0.16), and ft_v2 step 6000, the least comb (8.8 dB) and nearly the best held-out grain, at a cost on pistol (0.49). Blind A/B pages: `data/decoder_ft/eval/ft_s03000/listen/page/index.html` and `data/decoder_ft/eval/v2_s06000/listen/page/index.html`.

## Using a checkpoint

`scripts/run_sidecar.py --decoder-ckpt data\decoder_ft\runs\ft_v2\dec_ema_s06000.safetensors` loads it strictly before the decoder is compiled and graphed; a bad file fails once at startup and never reloads per request. The READY line names the decoder.
