# Static guard

Stable Audio 3 occasionally returns static: a block of steady broadband hiss where a sound should be. The model is random, so no setting stops it from ever producing one. The guard makes sure a static clip never reaches the player by checking every clip after it is generated and regenerating the ones that fail. Code: `src/sarge/static_guard.py`, wired into `StableAudio3Generator` (`static_guard=True` by default, `max_static_retries=2`). Shipped in `d65900f`.

## How a clip is judged

The check runs on the final clip, after `_extract` and `normalize_loudness`, and takes about 0.6 ms for a 0.6 s clip. A clip counts as static when any one of three tests fires.

1. **Hiss.** Long-term spectral flatness (Welch PSD from 50 Hz) at or above 0.12, an onset that rises 9 dB or less, and a sound that does not decay: either the median 5 ms level over the 25 ms after the peak stays at or above -8 dB, or the last 100 ms before the fade sits at or above -20 dB relative to the peak.
2. **Bed.** Flatness at or above 0.25 and the quietest 10% of 10 ms frames within 15 dB of the peak. This catches a hit sitting on a loud hiss floor.
3. **Steady tail.** A real hit followed by hiss: the stretch from 150 ms after the peak has flatness at or above 0.15, a level spread of 5 dB or less, a level of at least -32 dB, and drift of 3 dB or less.

Clips with NaN or inf count as static. Silent clips and clips under 100 ms count as clean.

Flatness alone was the obvious rule, and it fails. It flags every generated glass impact, 84% of grass footsteps and half of a set of synthetic decaying hits, because a broadband hit is spectrally as flat as hiss. Timing separates them: a hit decays, static holds. The decay test is what took generated glass impacts from 6.7% flagged to 0%.

## What happens on a fail

The failing item is regenerated alone with a new seed and the same prompt and step count, up to two times. Retry seeds come from an md5 mix of the original seed and the attempt number with bit 31 set, so they repeat from run to run and never collide with any variant seed of any key. If all three attempts fail, the generator serves the least flat one with any non-finite samples zeroed. In a batch, only the failing items are regenerated, and order and per-key steps are kept. Each model call takes the generator lock once, and the check runs outside the lock. `fast_stats()` reports `static_checks`, `static_flagged`, `static_retries`, `static_rescued` and `static_exhausted`, which together give the live static rate. The guard setting is part of the disk cache key.

## Calibration

The thresholds were set on 5,600 generated clips already on disk, 240 real ESC-50 clips cut the way the generator cuts, and 450 synthetic clips. Flag rates for the final rule:

| Set | Clips | Flagged |
|---|---|---|
| Known static clips (water at 3 steps, the 1.5 s length) | 155 | 100% |
| Synthetic hiss in 4 colours, with and without volume wobble | 310 | 100% |
| Synthetic hit on a loud hiss bed | 80 | 100% |
| Production-config clips (footsteps, impacts incl. glass, splashes, water) | 340 | 0% |
| Generated glass impacts | 30 | 0% |
| Generated footsteps on wood, stone and metal | 3,180 | 0.7% |
| Real ESC-50 water drops, fire, wind, knocks, footsteps | 800 | 0% |
| Real ESC-50 glass breaking | 160 | 18.8% |

The real glass flags are eight recordings that are one continuous crunch or sustained tinkling, which inside a 0.6 s window is steady noise with no onset. Four other measures (envelope variability, hold time, waveform spikiness, spectral change over time) overlap completely between those recordings and real static, so no general rule separates them. A generated "impact on glass" that comes out as a steady crunch has failed as an impact anyway, so a retry is the right outcome, and generated glass impacts are not flagged in practice.

An adversarial review found and fixed four defects before shipping: retry seeds that collided between variants of one key, a miss on a real hit followed by steady hiss (the reason the steady-tail test exists), exhausted all-NaN clips reaching the player, and a validation script that could fail a correct run.

## Validation on the GPU

`scripts/bench/check_static_guard.py` generated 14 keys by 50 seeds at production settings, first with the guard off and then on (`scripts/bench/out_static_guard.json`).

| Measure | Result |
|---|---|
| First-attempt static | 3 of 700 (2 grass footsteps, 1 glass impact) |
| Retried, rescued, exhausted | 3, 3, 0 |
| Static served | 0 |
| Clean clips byte-identical to guard-off output | 697 of 697 |
| Retries on clips an independent check calls clean | 0 |
| Latency, median | 53.7 ms off, 54.5 ms on, 108.6 ms when a retry happens |
| Batch pairs | 80 items, order kept, 80 of 80 identical when clean |

The three rejected first attempts and the clips served in their place are in `data/_gate/listen/static_guard_catches.wav` (built by `scripts/bench/listen_static_flags.py`).

## Limits

Static can still reach the player in two ways. The first is three static attempts in a row, which at the observed first-attempt rates is rarer than one clip in a hundred thousand. The second is noise with a steep spectrum, such as a low rumble with flatness under 0.12, which this check does not measure and which has not appeared in any generated clip so far. If `static_exhausted` ever goes above zero in a live session, raise `max_static_retries`.

The guard also objects to some real recordings, and the realism routes run straight into that. On the 0.6 s held-out references it flags 12.5 percent of FoleySet breaking glass, 17.6 percent of ESC-50 glass, 29 percent of FoleySet splashes and 11 percent of sword clips, nearly all through the hiss rule, while gunshots, footsteps, doors, clicks, punches, dogs and horns sit between 0 and 2 percent (29 of 1225 clips overall; the training-side references give 63 of 1818). Sustained shatter, running water and blade scrapes are broadband and hold their level, which is the hiss rule's definition. Generation at production settings rarely gets there (1 flag in 400 vocab clips), so today the cost is small. A route that makes glass or water more real will pay it in retries and in serving the least flat attempt: anchored generation at noise 0.7 drew 97 flags in 400 clips, and the real anchors themselves draw 7 percent. Any future fix to glass or water realism has to be judged with the guard on and off.
