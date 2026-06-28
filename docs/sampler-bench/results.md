# Sampler-swap results (objective metrics)

Reference = euler@8: CLAP_match 43%, crest 14.33, PQ 7.02.

CLAP_match = rank-1 of correct prompt among 14 (chance 7%). crest_rel = crest factor / euler@8 (transient sharpness, 1.0=as sharp as 8-step). PQ = Audiobox production quality 1-10. dead = silent/NaN clips.

```
sampler   steps   lat_ms  CLAP_match  crest_rel     PQ  dead
------------------------------------------------------------
euler         8      443         43%       1.00   7.02     0
euler         4      239         36%       1.07   7.04     0
euler         3      190         14%       1.17   7.16     0
euler         2      139         29%       0.73   6.72     0
euler         1       90         29%       0.29   6.55     0
dpmpp         4      243         36%       1.01   7.08     0
dpmpp         3      191         14%       1.17   7.16     0
dpmpp         2      144         21%       0.74   6.82     0
rk4           4       38          7%       0.43   5.68     0
rk4           3       38          7%       0.43   5.68     0
rk4           2       38          7%       0.43   5.68     0
pingpong      4      240         50%       1.00   6.88     0
pingpong      3      188         29%       1.19   7.01     0
pingpong      2      141         29%       0.61   6.83     0
```
