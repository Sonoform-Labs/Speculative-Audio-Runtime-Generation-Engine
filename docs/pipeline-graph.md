# Pipeline graph

The pipeline graph is a Unity editor tab that draws SARGE as a Blueprint-style node graph and lights the path each sound takes while the game runs. It was built for recording the ADC Bristol talk, where nothing can be demoed live: the Game view stays completely clean, and the tab beside it shows what the engine did for every footstep, shot and impact.

## Open it and dock it

Open **Window > SARGE > Pipeline Graph**. To put it next to the Game view, drag the tab by its title onto the right edge of the Game view and release when the blue drop area covers the right half. The graph is about three times as wide as it is tall, so for a 16:9 recording a full-width strip under the Game view (drop the tab on the Game view's bottom edge) shows the nodes largest. Turn off **Maximize On Play** in the Game view toolbar, because a maximised Game view covers the tab.

The graph fits itself to the tab. The scroll wheel zooms, a middle or right drag pans, and **F** or the **Frame** button fits it again. The tab's menu (the three dots at its top right) can hide the toolbar for a clean frame, and **Preview a Path** there (or the **Preview** button) plays four sample paths with no sidecar at all, so the look can be rehearsed in Edit mode.

## A clean Game view and F1

Nothing draws on the Game view by default. Hidden now: the status lines at the top left of `SargeFirstPerson` (engine, surface, gun and ammo, boots, served tallies, predictions, intent, mix, world switches) with its key help at the bottom and its "Starting sound engine" box, the reverb line of `SargeTestRange` and its floor labels (GLASS, GRAVEL, IRON BOOTS and the rest), the on-time line of `SargeOnTime`, the blue landing marker, the coloured footstep markers of `SargeMotionPredictor`, the labels of `SargeImpulseProbe`, and the HUDs of the showcase, walker and first-person samples. The demo never had a crosshair or a UI canvas.

**F1** brings every one of them back and hides them again (`SargeHud.Visible`, key in `SargeHud.Key`), and each Play starts hidden. With the startup box gone, the first minute of a hybrid session is a still frame while the sidecar loads; the player can move once the engine answers, as before.

## What lights up

Every sound lights Game Event and World Layer first and then the rest of the path it took. A lit node gets a glowing header and border, a lit wire turns bright and carries small dots toward its target, and both fade over about 1.3 s; each hop starts 75 ms after the one before, so the light runs along the path. Every node keeps a short caption of the last thing it did.

| What happened | Lit path | Example captions |
|---|---|---|
| A step on gravel that a predictor asked for, served from the cache | Game Event, World Layer, Predictors, Cache, Playback | `footstep \| gravel`, `hit · motion`, `cached 1.2 ms` |
| A repeat nobody predicted | Game Event, World Layer, Cache, Playback | `cached 0.6 ms` |
| A take Unity already holds (meta-only, or local playback under the on-time preset) | as a cache hit | `held in Unity`, `Unity copy, played at once` |
| A first-time sound | Game Event, World Layer, World Model (only when a physics or scene record went into the key), Writer (only for a written prompt), Prompt, Router, SA3 Medium or Woosh DFlow, Shaping, Playback | `template prompt`, `guns → SA3 Medium`, `rendered 148 ms` |
| A prediction | World Layer (faint), Predictors, and Cache when the server queued a render | `motion · footstep \| gravel in 0.60 s` |
| A background render (prefetch or refill) | Prompt, Router, the model, Shaping, Cache | `prefetched · footstep \| stone` |
| A written prompt lands | World Model (with a record), Writer, Prompt | `wrote · door \| wood 812 ms` |
| A written take is swapped in | Writer, Prompt, Router, the model, Shaping, Cache | `writer take swapped in` |
| Routing by meaning moves a key | Writer, Router | `by meaning: guns` |
| A Flow take renders | Cache (faint), Woosh Flow, Selection | `take 1 · footstep \| gravel 980 ms` |
| A Flow upgrade lands | Woosh Flow, Selection, Cache | `upgrade swapped 2 takes` |
| SA3 best of N (only with `--sa3-upgrade-best`) | SA3 Medium, Selection, Cache | `picked a better take` |

Once a key's takes have been upgraded, its cache hits read `cached · Flow take`. A first-time sound that a predictor had asked for, but that was still rendering when the event came, also lights Predictors faintly with `predicted, still rendering`. Times on events are the round trip Unity measured from request to reply; times on background work are the server's render times.

## Where the data comes from

`SargeService.trace` (on) sends `"trace": true` in the hello, and every audio reply then carries a small `trace` object: the path that served it (cache, render, meta-only), the route, the model, where the prompt came from (template, game or writer), whether a physics or scene record was used, whether a prediction was pending for that exact key, and whether the take was a Flow upgrade (docs/sidecar-protocol.md). The service turns each reply, each prediction acknowledgement and each local play into a plain record in `SargeTrace`, a locked ring of 512 records that the tab reads with `SargeTrace.Since`; Unity objects are touched on the main thread only. Background work arrives through the stats reply: while the tab is open the service asks for `"feed": N` on its control connection every 0.5 s (`tracePollSeconds`) and turns the items into records. With the tab closed nothing is recorded or polled.

A sidecar without the trace still lights a sensible path. The path then comes from `served`, the Predictors node lights when Unity itself sent a prediction for that material within the last 4 s, and no background records appear.

The trace changes no audio. A client that does not ask for it gets byte-identical replies (`tests/test_trace.py`, which also checks every sample against a server with `--no-trace`).

## The full-system launcher line

This line is written down here and left for you to set; the scene keeps its saved launcher arguments. To let every node light up, set `arguments` on the Sarge Launcher in `WalkArena.unity` to the line below and leave `pythonExe`, `workingDir`, `hfHome` and `port` as they are:

```
D:\Python\sarge-research\modal\runpinned.py scripts\run_sidecar.py --generator hybrid --realtime --stereo --amp --detailed-prompts --learn --cache-dir D:\Python\SAGE\.cache\sfx_bake --prompts template:caption+tt --woosh-upgrade flow --woosh-upgrade-solver midpoint:16 --woosh-upgrade-best 4 --woosh-upgrade-vram-save share,encoder --scene --world --writer llama --writer-style canon --route meaning
```

It is the B4 line of docs/push-09.md with the world and the language layer switched on. `--scene` and `--world` put the scene and physics records into the key, which lights World Model. `--writer llama --writer-style canon` starts one CPU llama-server that writes the prompts and also answers whether a sound is a firearm, which is what `--route meaning` asks; with the canon style no second routing writer is started. `--woosh-upgrade-vram-save share,encoder` is what the upgrade turns on by default, written out. `--sa3-upgrade-best 4` would also light the SA3 Medium to Selection wire, but round five found it gained nothing on new data, so the line leaves it off. The writer stores what it writes under `.cache\writer`, and written prompts and scene keys change how some sounds come out compared with B4, so this is a showcase line.

On the Unity side press **P** and **O** once the engine answers. **P** now turns on physics records for every `SargeAutoCapture` and breakable window together with `SargeService.sendScene`, and **O** turns on the impact and aim predictors; the motion predictor and the equip predictions run all the time. Without **P** the bouncing balls and props send no records and World Model stays dark.

## Checks

The Python suite passes (704 tests, 13 of them new in `tests/test_trace.py`). Every edited C# file compiles with Unity's bundled Roslyn against the 2022.3.8f1 assemblies, the editor assembly against UnityEditor.dll. A console harness that runs the real `SargeService`, `SargeClient`, `SargeTrace` and `SargeHud` on a stubbed UnityEngine against live mock sidecars passed all 29 checks, with the trace on and against a server that refuses it.

The two GPU checks wait for memory. On 2026-10-27 the machine had about 10 GB of commit free, and the first identity run's sidecar died while loading SA3 in a Windows low-virtual-memory event, before READY. `full_system_smoke.py` refuses to start below 16 GB of free commit for that reason. Once the memory is back, run these under Unity's environment, one at a time: the first e2e is the old-client identity, the second asks for the trace on both bench connections and keeps it in every row, and both must match `baseline_warm_1` clip for clip.

```
$env:HF_HOME='F:\AI\cache\hf'; $env:HF_HUB_OFFLINE='1'; Remove-Item Env:HF_HUB_CACHE,Env:TRANSFORMERS_CACHE,Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'; $env:TEMP='D:\tmp'; $env:TMP='D:\tmp'
$py='D:\Python\SAGE\.venv214\Scripts\python.exe'; $pin='D:\Python\sarge-research\modal\runpinned.py'
& $py $pin D:\Python\SAGE\scripts\bench\e2e_hybrid.py --tag graph_off --save-dir G:\sarge\graph\e2e\graph_off --extra --learn
& $py $pin D:\Python\SAGE\scripts\bench\e2e_hybrid.py --tag graph_trace --trace --save-dir G:\sarge\graph\e2e\graph_trace --extra --learn
& $py $pin D:\Python\SAGE\scripts\bench\startup_compare.py D:\Python\SAGE\data\_gate\startup\baseline_warm_1 G:\sarge\graph\e2e\graph_off G:\sarge\graph\e2e\graph_trace --json D:\Python\SAGE\data\_gate\startup\compare_graph.json
& $py $pin D:\Python\SAGE\scripts\bench\full_system_smoke.py
```

The smoke start runs the full-system line on ports 8767, 8773 and 11541 with a scratch cache and writer store under `G:\sarge\graph\smoke`, fires footsteps, shots, a physics and scene impact and a door with the trace on, watches the stats feed until a Flow swap and a written prompt arrive (150 s at most), fires the keys again, stops the whole process tree and saves `G:\sarge\graph\smoke\smoke.json`.
