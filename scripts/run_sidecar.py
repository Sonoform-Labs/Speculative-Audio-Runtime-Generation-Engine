import _bootstrap

import argparse
import atexit
import importlib.metadata
import os
import subprocess
import threading
import time

_T_SCRIPT = time.time()
from sarge import startup

startup.root()
startup.stage("python up", at=_T_SCRIPT)
_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
try:
    _TORCH = importlib.metadata.version("torch").split("+")[0]
except importlib.metadata.PackageNotFoundError:
    _TORCH = "none"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_CACHE, "inductor" if _TORCH.startswith("2.6") else "inductor-" + _TORCH))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_CACHE, "triton-" + _TORCH))

_T = startup.stage("sarge package import", _T_SCRIPT)
from sarge import prefetch, weight_cache
from sarge.prompts import TEMPLATES, check_style
from sarge.sidecar import serve

startup.stage("sarge.sidecar import", _T)

COMPILED = {"compile_mode": "default", "compile_decode": True, "graph_decode": True, "hoist_rope": True,
            "fp16_accumulation": True, "graph_text": True, "unfuse_bias": True, "triton_gemm": True,
            "decode_prefix": True}
PRESENT = {"trim": "natural", "level": "peak"}
TIERS = {
    "quality": {"sampler_type": "euler", "steps": 3, "skip_first_step": True, **COMPILED, **PRESENT},
    "default": {"sampler_type": "euler", "steps": 3, "skip_first_step": True, **COMPILED, **PRESENT},
    "fast":    {"sampler_type": "euler", "steps": 3, "skip_first_step": True, "cfg_interval": (0.9, 1.0), **COMPILED,
                **PRESENT},
}
HYBRID_GUNS = {"model_name": "medium", "sampler_type": "pingpong", "steps": 8, "detail_steps": 8, "skip_first_step": False,
               "cfg_scale": 1.0, "duration_padding_s": 1.0, "noise_span": (3.0, 6.0), "gen_duration_s": 3.0,
               "prompt_style": "detailed", "hoist_diff": True}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIER_COST_MS = {"quality": 28.0, "default": 28.0, "fast": 25.0}
WOOSH_COST_MS = {"dflow": 50.0, "flow": 1010.0}
WRITER_STYLES = ("v1", "named", "texture", "title", "caption", "canon")
POOL_SOURCES = ("canon", "ground", "named")
SCORE_MODES = ("request", "event-min")
WRITER_GROUND = ("off", "anchors", "words", "caption")
UPGRADE_SOLVERS = ("dopri5", "rk4:8", "rk4:16", "midpoint:16", "midpoint:32")


def check_first(style):
    return "same" if str(style or "") == "same" else check_style(style)


def cache_dir_of(args):
    return weight_cache.default_dir({"SARGE_WEIGHT_CACHE": args.weight_cache})


def check_weights(model_name):
    t = time.time()
    try:
        import torch
    except Exception:
        pass
    t = startup.stage("torch import", t)
    try:
        from huggingface_hub import constants, try_to_load_from_cache
        from stable_audio_3.model_configs import models
    except Exception:
        return
    t = startup.stage("stable_audio_3 import", t)
    cfg = models.get(model_name)
    if cfg is None or not constants.HF_HUB_OFFLINE:
        return
    found = all(isinstance(try_to_load_from_cache(cfg.repo_id, f), str) for f in (cfg.config_path, cfg.ckpt_path))
    startup.stage("weights check", t)
    if not found:
        raise SystemExit("%s weights (%s) are not in the offline HF cache %s; point HF_HOME or HF_HUB_CACHE at the "
                         "cache that holds them" % (model_name, cfg.repo_id, constants.HF_HUB_CACHE))


def early_woosh(env=None) -> bool:
    return (env if env is not None else os.environ).get("SARGE_WOOSH_EARLY", "1").strip().lower() not in (
        "0", "off", "no")


def woosh_styles(args):
    final = getattr(args, "prompts", None) or "tuned"
    first = getattr(args, "first_prompt", None) or "same"
    if first == "same" or getattr(args, "woosh_upgrade", "off") == "off":
        first = final
    return first, final


def start_woosh(args, hold=False):
    if not os.path.isfile(args.woosh_python):
        raise SystemExit("woosh python not found: %s" % args.woosh_python)
    first, final = woosh_styles(args)
    pin = os.path.join(os.path.dirname(REPO), "sarge-research", "modal", "runpinned.py")
    cmd = [args.woosh_python, "-u"] + ([pin] if os.path.isfile(pin) else []) + [
        os.path.join(REPO, "scripts", "woosh_server.py"), "--model", args.woosh_model,
        "--port", str(args.woosh_port), "--parent-pid", str(os.getpid())] + (
        ["--frames", str(args.woosh_frames)] if args.woosh_frames else []) + (["--amp"] if args.amp else []) + (
        [args.amp] if args.amp in ("punch", "auto") else []) + (
        ["--level", "loudness"] if getattr(args, "level", None) == "loudness" else []) + (
        ["--stereo"] if args.stereo else []) + (
        ["--prompts", first] if first != "tuned" else []) + (
        ["--upgrade", args.woosh_upgrade] if getattr(args, "woosh_upgrade", "off") != "off" else []) + (
        ["--upgrade-prompts", final] if final != first else []) + (
        ["--upgrade-solver", args.woosh_upgrade_solver]
        if getattr(args, "woosh_upgrade_solver", "dopri5") != "dopri5" else []) + (
        ["--upgrade-best", str(args.woosh_upgrade_best)] if getattr(args, "woosh_upgrade_best", 1) > 1 else []) + (
        ["--upgrade-verifier", args.woosh_upgrade_verifier]
        if getattr(args, "woosh_upgrade_best", 1) > 1 and getattr(args, "woosh_upgrade_verifier", None) else []) + (
        ["--upgrade-score", args.woosh_upgrade_score]
        if getattr(args, "woosh_upgrade_score", "request") != "request" else []) + (
        ["--upgrade-vram-free", str(args.woosh_upgrade_vram_free)]
        if getattr(args, "woosh_upgrade_vram_free", 0) else [])
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    env.pop("TRANSFORMERS_CACHE", None)
    env.update(HF_HUB_CACHE=args.woosh_hf_cache, HF_HUB_OFFLINE="1", SARGE_T0=repr(startup.t0()))
    startup.stage("woosh launch")
    proc = subprocess.Popen(cmd + (["--hold"] if hold else []), cwd=REPO, env=env,
                            stdin=subprocess.PIPE if hold else None, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    atexit.register(lambda: proc.poll() is None and proc.kill())
    if hold:
        return proc
    t0 = time.time()
    for line in proc.stdout:
        print("[woosh] " + line.rstrip(), flush=True)
        if "listening on" in line:
            threading.Thread(target=lambda: [late_line(x, args) for x in proc.stdout], daemon=True).start()
            print("woosh %s up in %.1f s" % (args.woosh_model, time.time() - t0), flush=True)
            startup.stage("woosh up", t0)
            return proc
    raise SystemExit("woosh server exited before listening (code %s)" % proc.poll())


def late_line(line, args=None):
    if getattr(args, "woosh_upgrade_best", 1) > 1 and line.startswith("woosh upgrade "):
        print("[woosh] " + line.rstrip(), flush=True)


def launch_woosh(args):
    proc = start_woosh(args, hold=True)
    up, state = threading.Event(), {"listening": False}

    def pump():
        for line in proc.stdout:
            if up.is_set():
                late_line(line, args)
                continue
            print("[woosh] " + line.rstrip(), flush=True)
            if "listening on" in line:
                state["listening"] = True
                up.set()
        up.set()

    threading.Thread(target=pump, name="woosh-out", daemon=True).start()
    return proc, up, state


def gate(proc, name):
    try:
        proc.stdin.write(name + "\n")
        proc.stdin.flush()
    except (OSError, ValueError):
        pass


def close_woosh(proc):
    try:
        proc.stdin.close()
    except (AttributeError, OSError, ValueError):
        pass
    startup.stage("woosh worker released, the guns did not get ready")


def open_woosh(args, proc, up, state):
    if proc.stdin is None or proc.stdin.closed:
        return start_woosh(args)
    t0 = time.time()
    gate(proc, "gpu")
    up.wait()
    if not state["listening"]:
        raise SystemExit("woosh server exited before listening (code %s)" % proc.poll())
    print("woosh %s up in %.1f s" % (args.woosh_model, time.time() - t0), flush=True)
    startup.stage("woosh up", t0)
    return proc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the predictive-SFX sidecar audio service.")
    ap.add_argument("--generator", default="sa3")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--pcm", default="f32le", choices=["f32le", "i16le"])
    ap.add_argument("--vram-kb", type=float, default=None,
                    help="clip cache budget in KB; 50000, or 256000 with --world (physics buckets multiply the keys)")
    ap.add_argument("--gpu-ms-budget", type=float, default=10_000_000.0)
    ap.add_argument("--gen-cost-ms", type=float, default=None)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--target-variants", type=int, default=6)
    ap.add_argument("--tier", default=None, choices=["quality", "default", "fast"],
                    help="default=euler@3, quality=euler@3 (4 to 8 steps measured worse on the clap gate with the current prompt), "
                         "fast=default with cfg on the first remaining step only (27 ms; better gunshots, clicks and stone, worse dogs). "
                         "Falls back to SARGE_GEN_TIER env var then 'default'. --steps/--sampler override it.")
    ap.add_argument("--steps", type=int, default=None, help="override the tier's step count")
    ap.add_argument("--sampler", default=None, help="override the tier's sampler (euler; rk4 DISABLED)")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--cache-dir", default=os.path.join(".cache", "sfx_bake"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-deepen", action="store_true")
    ap.add_argument("--deepen-batch", type=int, default=1,
                    help="clips per background generation; with the cuda graph a pair costs the same per clip as one "
                         "and holds the gpu twice as long, so 1 is the default")
    ap.add_argument("--learn", action="store_true",
                    help="use the runtime affordance learner + router: counts fired events for the 'learned' reply "
                         "field and, without --realtime, derives prefetch candidates from them; that path still reads "
                         "two word tables, affordance_learn._CONTINUOUS and router.REALTIME_TOLERANT (docs/generic.md)")
    ap.add_argument("--no-compile", action="store_true",
                    help="skip torch.compile and the decoder graph; starts about 30 s faster, each clip about 20 ms slower")
    ap.add_argument("--woosh-model", choices=["dflow", "flow"], default="dflow",
                    help="--generator hybrid: the Woosh model for everything except guns (dflow 4 steps, flow slower and finer)")
    ap.add_argument("--woosh-frames", default=None,
                    help="--generator hybrid: Woosh latent frames per request, 100 per second of audio (101 = 1.0 s): "
                         "one count for every sound, or family=count pairs from sarge.amp (impact, step, other for "
                         "the rest), e.g. impact=501. Each distinct count is one CUDA graph captured before READY, "
                         "about 0.5 s and 60 to 80 MB each. Default: woosh_server.FRAMES, 101 for every family, since "
                         "no count up to 501 brought any family to 5%% or fewer clips still sounding at the cut "
                         "(docs/hybrid.md)")
    ap.add_argument("--woosh-upgrade", choices=["off", "flow"], default="off",
                    help="--generator hybrid with --woosh-model dflow: after READY the Woosh worker loads Woosh Flow "
                         "beside DFlow (sharing the autoencoder and the text encoder); the first hit of a key plays "
                         "DFlow, then idle-time Flow renders of every held take (same seed, same prompt, about 1 s "
                         "each, preempted by any served render) are swapped in per key with prompt_rev + 1, kept in "
                         "memory only (docs/hybrid.md). Off by default")
    ap.add_argument("--first-prompt", type=check_first, default="same",
                    help="--woosh-upgrade flow: the prompt style of the first-hit DFlow renders (tuned, generic or "
                         "template:NAME as for --prompts), while the Flow re-renders and every gun keep --prompts; "
                         "same (default) plays --prompts on both. tuned keeps the per-sound phrasebook for first "
                         "hits as a temporary trade-off, since the caption template on DFlow measured -0.14 on the "
                         "demo keys and -0.42 on water against it; it goes once a generic first hit holds those bars "
                         "(docs/hybrid.md). A prompt the game or the writer sends wins on both renders")
    ap.add_argument("--woosh-upgrade-solver", choices=UPGRADE_SOLVERS, default="dopri5",
                    help="--woosh-upgrade flow: the ODE solver of the Flow re-renders, dopri5 (adaptive, about 86 "
                         "evaluations and 1.0 s a clip, the default) or a fixed solver with its step count, e.g. "
                         "rk4:8 at 32 evaluations (docs/hybrid.md)")
    ap.add_argument("--woosh-upgrade-best", type=int, default=1,
                    help="--woosh-upgrade flow: each upgraded take is the best of N Flow candidates as scored by a "
                         "verifier outside the judge panel (Woosh-CLAP's audio tower on the CPU against the request "
                         "words), so a key keeps distinct takes and none of them is a real recording; 1 (default) "
                         "renders one candidate as before (docs/hybrid.md)")
    ap.add_argument("--woosh-upgrade-pool", default="off",
                    help="--woosh-upgrade-best: a comma list of generic prompt sources (%s) whose prompts join the "
                         "key's own prompt as Flow candidates; the first upgraded take of a key renders N candidates "
                         "of every prompt, the verifier keeps the prompt with the best mean score for the key and "
                         "best of N picks every take after that. canon and named run the CPU writer "
                         "model, ground the caption index. Off by default" % ", ".join(POOL_SOURCES))
    ap.add_argument("--woosh-upgrade-vram-free", type=int, default=0,
                    help="--woosh-upgrade flow: the worker loads Flow only once the card reports at least this many "
                         "MiB free, so the upgrade waits for room on the GPU; 0 (default) loads as before")
    ap.add_argument("--woosh-upgrade-score", choices=SCORE_MODES, default="request",
                    help="--woosh-upgrade-best: request scores a candidate against the request words; event-min takes "
                         "the lower of that and the score against the event words alone. Default request")
    ap.add_argument("--woosh-upgrade-verifier", default=None,
                    help="--woosh-upgrade-best: the Woosh-CLAP checkpoint folder (config.yaml and "
                         "weights_audio.safetensors), default WOOSH_HOME/checkpoints/Woosh-CLAP")
    ap.add_argument("--woosh-port", type=int, default=8771)
    ap.add_argument("--woosh-hf-cache", default=r"D:\hf_cache",
                    help="--generator hybrid: the HF cache that holds roberta-large for the Woosh text encoder")
    ap.add_argument("--woosh-python", default=os.path.join(REPO, ".venv_woosh", "Scripts", "python.exe"))
    ap.add_argument("--weight-cache", default=weight_cache.default_dir() or "off",
                    help="--generator hybrid: folder for the fp16 copy of the SA3 medium weights, read at startup "
                         "in place of the 9.2 GB fp32 checkpoint (built by the first run, keyed on the checkpoint, "
                         "config, torch, stable_audio_3 and loader code, checked tensor by tensor against checksums "
                         "on every load; off disables). Default SARGE_WEIGHT_CACHE, else " + weight_cache.DEFAULT_DIR
                         + " when its parent folder exists, else off")
    ap.add_argument("--stereo", action="store_true",
                    help="send stereo clips: the model's two channels plus a decorrelated width stage by sound type "
                         "(bass stays centred); Unity builds two-channel clips from the reply metadata")
    ap.add_argument("--detailed-prompts", action="store_true",
                    help="the searched prompt grammar (sarge.prompts.build, a phrasebook tuned to the demo's event and "
                         "material names): detailed source, action and acoustic wording plus the space from the request "
                         "context (outdoor_open, indoor_small, indoor_large, tunnel; default is outdoor_open); "
                         "--generator hybrid always uses it unless --prompts generic or template:NAME")
    ap.add_argument("--prompts", type=check_style, default="tuned",
                    help="generic: every prompt on both routes is built only from the request (event and material "
                         "words split on underscores, capitals and digits, the intensity word, the context words, and "
                         "with --world the contact words), one template for every sound, SA3 gets the vendor's "
                         "'TrackType: SFX,' prefix (docs/generic.md); template:NAME: one of the universal sentence "
                         "templates in sarge.prompts.TEMPLATES (%s), filled from the same request fields, with a "
                         "+tt or -tt suffix forcing the SA3 prefix on or off; tuned (default): the prompts as before. "
                         "A prompt the game sends (prebake key or event field) wins over all of them on both routes"
                         % ", ".join(TEMPLATES))
    ap.add_argument("--route", choices=["families", "words", "class", "single"], default="families",
                    help="--generator hybrid: which sounds go to SA3 medium (guns) and which to Woosh. families "
                         "(default): substrings of sarge.amp.FAMILIES as before; words: a whole word of the event name "
                         "in routed.DISCHARGE_WORDS or routed.FIREARM_WORDS, or a firearm word in the material; class: "
                         "only a 'class' field sent by the game that names a --gun-class; single: every sound to "
                         "--single-model")
    ap.add_argument("--gun-class", default="weapon",
                    help="--route class: comma-separated class values that go to SA3 medium (default weapon)")
    ap.add_argument("--single-model", choices=["woosh", "sa3"], default="woosh",
                    help="--route single: the one model every sound goes to")
    ap.add_argument("--rapid-hz", type=float, default=1.5,
                    help="prediction: an event that fires again faster than this many times a second stays out of "
                         "the habit model and confirms only motion predictions (0 turns the filter off)")
    ap.add_argument("--amp", nargs="?", const="full", choices=["full", "punch", "auto"], default=None,
                    help="amplifier stage after generation: attack emphasis and saturation by sound type, strongest on "
                         "gunshots (about +4 to 5 dB of density at the same peak), light on footsteps; --amp punch keeps "
                         "the attack emphasis and drops the saturation; --amp auto sets the attack emphasis, the "
                         "saturation and the stereo width from the clip's attack time, crest factor, spectral "
                         "flatness and decay, with saturation held to the clip body and dropped when it would "
                         "lengthen the decay (docs/generic.md)")
    ap.add_argument("--level", choices=["peak", "loudness"], default=None,
                    help="loudness: level every clip by its 400 ms loudness (hard -14, normal -17, soft -23 LUFS) with "
                         "a 30 Hz high-pass and at most 6 dB of lookahead limiting, both routes; peak sets the peak per "
                         "intensity on both routes; left out, guns keep the tier's leveller (peak, or rms with "
                         "--short-clips) and woosh sets the peak")
    ap.add_argument("--world", action="store_true",
                    help="physics context: the server buckets the phys block Unity sends into the key and the prompt")
    ap.add_argument("--scene", action="store_true",
                    help="scene record: the server quantises the optional 'scene' object Unity sends (who made the "
                         "sound, the object, the action, the room) into the key; the seed ignores it (docs/llm-layer.md)")
    ap.add_argument("--writer", choices=["off", "llama", "ollama", "store", "ground"], default="off",
                    help="prompt writer (docs/llm-layer.md): llama spawns a CPU-only llama-server for --writer-model "
                         "after READY and writes one prompt per key in the background, saved per key in --writer-store; "
                         "ollama talks to an Ollama server already running on --writer-port; store serves a baked "
                         "prompt file (--writer-store FILE) with no model at all; ground writes each prompt from the "
                         "real recording captions nearest to the key (--writer-ground words or caption), with no "
                         "language model. Until a key has a written prompt it plays the --prompts builder, a "
                         "game-sent prompt always wins. Off by default")
    ap.add_argument("--writer-model", default="gemma4:e2b-it-qat",
                    help="--writer llama/ollama: a GGUF file or an Ollama tag found under OLLAMA_MODELS, F:\\OllamaModels "
                         "or G:\\sarge\\llm\\ollama_models (default gemma4:e2b-it-qat, Apache-2.0)")
    ap.add_argument("--writer-bin", default=None,
                    help="--writer llama: the llama-server binary (default SARGE_LLAMA_SERVER, else "
                         "F:\\Ollama\\lib\\ollama\\llama-server.exe)")
    ap.add_argument("--writer-port", type=int, default=11540, help="--writer llama/ollama: loopback port")
    ap.add_argument("--writer-threads", type=int, default=4, help="--writer: CPU threads for the model (default 4)")
    ap.add_argument("--writer-affinity", default="0xEFF",
                    help="--writer llama: CPU affinity mask set on the llama-server process (default 0xEFF, core 8 off)")
    ap.add_argument("--writer-store", default=os.path.join(REPO, ".cache", "writer"),
                    help="--writer: folder of the prompt store (prompts_<writer id>.json), or the file to serve with "
                         "--writer store")
    ap.add_argument("--writer-style", choices=list(WRITER_STYLES), default="v1",
                    help="--writer llama/ollama: how the writer is instructed and how its fields become a prompt "
                         "(docs/llm-layer.md). v1 is the shipped writer; named keeps the words of the event name "
                         "in the title; texture adds sound words on the character of the sound and new examples; "
                         "title gives Woosh the title and sound words only; caption writes the action as one plain "
                         "caption. Every style writes to a separate writer id and store. Default v1")
    ap.add_argument("--writer-ground", choices=list(WRITER_GROUND), default="off",
                    help="caption grounding (docs/llm-layer.md): retrieve real recording captions near the key from "
                         "--writer-ground-index on the CPU. anchors gives the nearest captions to the language model "
                         "as style anchors (--writer llama/ollama); words writes the request words and the sound "
                         "words the captions share, caption the request words in front of the nearest caption (both "
                         "--writer ground, no model). Every mode writes to a separate writer id and store. Off by "
                         "default")
    ap.add_argument("--writer-ground-index", default=None,
                    help="--writer-ground: the caption index folder (index.json, captions.json, emb.npy, embedder/); "
                         "default G:\\sarge\\captions\\index")
    ap.add_argument("--writer-ground-k", type=int, default=None,
                    help="--writer-ground: captions per key (default 4 anchors, 8 for words, 1 for caption)")
    ap.add_argument("--writer-when", default="seen,predict,manifest",
                    help="--writer: which keys get written: seen (fired), predict (predicted), manifest (sent ahead)")
    ap.add_argument("--writer-swap", choices=["revalidate", "next-miss"], default="revalidate",
                    help="--writer: revalidate renders take 0 with the written prompt in the background and swaps it "
                         "in when it lands; next-miss drops the template takes so the next hit renders it")
    ap.add_argument("--writer-readonly", action="store_true",
                    help="--writer llama/ollama: serve the store, never start the model or write")
    ap.add_argument("--writer-yield", action=argparse.BooleanOptionalAction, default=True,
                    help="--writer llama: suspend the llama-server process while the sidecar renders on the GPU, so "
                         "its CPU threads never compete with a render (on by default with the writer)")
    ap.add_argument("--event-log", default=None, help="write one JSON line per request (timings, served or generated)")
    ap.add_argument("--predict-fixes", action=argparse.BooleanOptionalAction, default=True,
                    help="on by default: a predicted stem maps to the fired event it names (footstep to iron boot "
                         "footstep), a prediction that sends no intensity takes the last fired one, no lead gate in "
                         "realtime mode, after expands to learned follow-up events (docs/on-time.md); "
                         "--no-predict-fixes takes every prediction literally")
    ap.add_argument("--pool-fix", action="store_true",
                    help="plan every variant of a predicted key, by deadline: the first variant of every predicted key "
                         "in ETA order, then the extra variants in ETA order")
    ap.add_argument("--concurrent-gen", action=argparse.BooleanOptionalAction, default=True,
                    help="on by default: a realtime miss generates outside the engine lock so cached keys answer "
                         "meanwhile, and no background render starts until the miss lands; served clips are the "
                         "same bits either way (docs/on-time.md); --no-concurrent-gen brings back the single-lock path")
    ap.add_argument("--async-cache-write", action=argparse.BooleanOptionalAction, default=True,
                    help="on by default: write the 16-bit disk-cache copy of each new clip on a background thread so "
                         "the reply never waits for the disk; a read of a clip still being written waits for it, so "
                         "served clips stay the same bits; --no-async-cache-write writes inline")
    ap.add_argument("--keep-warm-ms", type=float, default=0.0,
                    help="keep the GPU warm while a client is connected, one tiny job every this many ms (0 = off)")
    ap.add_argument("--sibling-on-miss", action="store_true",
                    help="on a realtime miss, play a cached sibling key at once and generate the exact key behind it")
    ap.add_argument("--short-clips", action="store_true",
                    help="the old presentation: a 0.6 s window around the loudest point at one rms per intensity; the "
                         "default keeps each sound until it dies away (up to 1.5 s) and sets its peak per intensity")
    ap.add_argument("--no-triton-gemm", action="store_true",
                    help="keep cublas with the bias fused for every DiT linear; the default swaps three shapes to an "
                         "fp16-accumulating triton gemm on pytorch 2.7+, judged equal, about 3.5 ms a clip faster")
    ap.add_argument("--no-skip-first", action="store_true",
                    help="run all 3 euler steps from t=1.0; the default drops the t=1.0 step, judged equal on held-out refs")
    ap.add_argument("--realtime", action="store_true",
                    help="generate each sound on-demand at fire time and cache it; nothing pre-generated at startup")
    ap.add_argument("--decoder-ckpt", default=None,
                    help="safetensors state_dict for the SAME-S decoder (scripts/train/dec_train.py export), loaded "
                         "strict before the decoder graph is captured; relative paths resolve against the repo root. "
                         "Falls back to SARGE_DECODER_CKPT")
    args = ap.parse_args(argv)
    if args.woosh_upgrade != "off" and (args.generator != "hybrid" or args.woosh_model == args.woosh_upgrade):
        raise SystemExit("--woosh-upgrade %s needs --generator hybrid and --woosh-model dflow" % args.woosh_upgrade)
    if args.woosh_upgrade_solver != "dopri5" and args.woosh_upgrade == "off":
        raise SystemExit("--woosh-upgrade-solver %s needs --woosh-upgrade flow" % args.woosh_upgrade_solver)
    if args.first_prompt != "same" and args.woosh_upgrade == "off":
        raise SystemExit("--first-prompt %s needs --woosh-upgrade flow" % args.first_prompt)
    if args.woosh_upgrade_best < 1:
        raise SystemExit("--woosh-upgrade-best takes 1 or more")
    if (args.woosh_upgrade_best > 1 or args.woosh_upgrade_verifier) and args.woosh_upgrade == "off":
        raise SystemExit("--woosh-upgrade-best and --woosh-upgrade-verifier need --woosh-upgrade flow")
    if args.woosh_upgrade_verifier and args.woosh_upgrade_best == 1:
        raise SystemExit("--woosh-upgrade-verifier needs --woosh-upgrade-best 2 or more")
    if (args.woosh_upgrade_vram_free and args.woosh_upgrade == "off") or args.woosh_upgrade_vram_free < 0:
        raise SystemExit("--woosh-upgrade-vram-free takes 0 or more MiB and needs --woosh-upgrade flow")
    if (args.woosh_upgrade_pool != "off" or args.woosh_upgrade_score != "request") and args.woosh_upgrade_best < 2:
        raise SystemExit("--woosh-upgrade-pool and --woosh-upgrade-score need --woosh-upgrade flow and "
                         "--woosh-upgrade-best 2 or more")
    if args.generator == "hybrid":
        prefetch.start_hybrid(HYBRID_GUNS["model_name"], args.woosh_model,
                              weights=lambda: weight_cache.locate(HYBRID_GUNS["model_name"], cache_dir_of(args)))

    tier = (args.tier or os.environ.get("SARGE_GEN_TIER") or "default").lower()
    gen_kwargs = dict(TIERS.get(tier, TIERS["default"]), fast_load=True)
    if args.sampler is not None:
        gen_kwargs["sampler_type"] = args.sampler
    if args.steps is not None:
        gen_kwargs["steps"] = args.steps
        gen_kwargs.pop("mid_noise", None)
    if args.no_compile:
        for k in COMPILED:
            gen_kwargs.pop(k, None)
    if args.stereo:
        gen_kwargs["stereo"] = True
    if args.detailed_prompts:
        gen_kwargs["prompt_style"] = "detailed"
    if args.amp:
        gen_kwargs["amp"] = True if args.amp == "full" else args.amp
    if args.short_clips:
        for k in PRESENT:
            gen_kwargs.pop(k, None)
    if args.level:
        gen_kwargs["level"] = args.level
    if args.no_triton_gemm:
        gen_kwargs.pop("unfuse_bias", None)
        gen_kwargs.pop("triton_gemm", None)
    if args.no_skip_first:
        gen_kwargs.pop("skip_first_step", None)
    if args.generator in ("sa3", "hybrid"):
        check_weights(HYBRID_GUNS["model_name"] if args.generator == "hybrid" else gen_kwargs.get("model_name",
                                                                                                    "small-sfx"))
    if args.generator == "hybrid":
        if not os.path.isfile(args.woosh_python):
            raise SystemExit("woosh python not found: %s" % args.woosh_python)
        gen_kwargs.update(HYBRID_GUNS, woosh_port=args.woosh_port, start_rest=lambda: start_woosh(args),
                          rest_cost_ms=WOOSH_COST_MS[args.woosh_model])
        if args.woosh_upgrade != "off":
            gen_kwargs["upgrade"] = args.woosh_upgrade
            first, final = woosh_styles(args)
            if first != final:
                gen_kwargs["first_prompt"] = first
        if args.route != "families":
            gen_kwargs.update(route=args.route, single_route="guns" if args.single_model == "sa3" else "rest",
                              gun_classes=tuple(c for c in args.gun_class.split(",") if c.strip()))
        if cache_dir_of(args):
            gen_kwargs["weight_cache"] = cache_dir_of(args)
        if early_woosh():
            woosh = launch_woosh(args)
            gen_kwargs.update(start_rest=lambda: open_woosh(args, *woosh), load_rest=lambda: gate(woosh[0], "load"),
                              drop_rest=lambda: close_woosh(woosh[0]))
    if args.prompts != "tuned":
        gen_kwargs["prompt_style"] = args.prompts
    ckpt = args.decoder_ckpt or os.environ.get("SARGE_DECODER_CKPT")
    if ckpt:
        ckpt = ckpt if os.path.isabs(ckpt) else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ckpt)
        if not os.path.isfile(ckpt):
            raise SystemExit("decoder checkpoint not found: %s" % ckpt)
        gen_kwargs["decoder_ckpt"] = ckpt
    sampler, steps = gen_kwargs["sampler_type"], gen_kwargs["steps"]
    gen_cost = args.gen_cost_ms if args.gen_cost_ms is not None else (
        WOOSH_COST_MS[args.woosh_model] if args.generator == "hybrid" else TIER_COST_MS.get(tier, 200.0))

    vram_kb = args.vram_kb if args.vram_kb is not None else (256_000.0 if args.world else 50_000.0)
    kwargs = dict(vram_budget_kb=vram_kb, gpu_ms_budget=args.gpu_ms_budget,
                  variants_per_key=args.variants, target_variants=args.target_variants,
                  continuous_topk=args.topk, gen_cost_ms=gen_cost)
    extra = {k: v for k, v in (("world", args.world), ("event_log", args.event_log),
                               ("predict_fixes", args.predict_fixes), ("pool_fix", args.pool_fix),
                               ("concurrent_gen", args.concurrent_gen), ("keep_warm_ms", args.keep_warm_ms),
                               ("sibling_on_miss", args.sibling_on_miss), ("cache_async", args.async_cache_write),
                               ("read_class", args.generator == "hybrid" and args.route == "class"),
                               ("scene", args.scene))
             if v}
    extra["rapid_hz"] = args.rapid_hz
    if args.woosh_upgrade != "off":
        extra["upgrade"] = args.woosh_upgrade
    writer = make_writer(args)
    if writer is not None:
        extra.update(writer=writer, writer_swap=args.writer_swap, writer_yield=args.writer_yield)
    pool = make_pool(args, writer)
    if pool is not None:
        extra.update(prompt_pool=pool, writer_yield=args.writer_yield)
    cache_dir = None if args.no_cache else args.cache_dir
    cleared = 0
    if cache_dir and os.path.isdir(cache_dir):
        for f in os.listdir(cache_dir):
            if f.endswith(".wav"):
                os.remove(os.path.join(cache_dir, f))
                cleared += 1

    if args.generator != "mock":
        print(f"loading + warming {args.generator} model (first time can take ~30s)...", flush=True)
    server = serve(generator=args.generator, host=args.host, port=args.port, pcm=args.pcm,
                   cache_dir=cache_dir, deepen=not args.no_deepen,
                   gen_kwargs=gen_kwargs,
                   learn=args.learn, realtime=args.realtime, deepen_batch=args.deepen_batch, **extra, **kwargs)
    host, port = server.server_address
    startup.done("READY")
    mode = "realtime-on-demand" if args.realtime else "predict+prewarm"
    print(f"sidecar READY on {host}:{port}  generator={args.generator}  tier={tier} "
          f"sampler={sampler} steps={steps}  gen_cost_ms={gen_cost:.0f}  mode={mode}  "
          f"learn={'on' if args.learn else 'off'}  cache={cache_dir or 'off'} (cleared {cleared})  "
          f"decoder={os.path.basename(gen_kwargs['decoder_ckpt']) if gen_kwargs.get('decoder_ckpt') else 'base'}  "
          f"concurrent-gen={'on' if args.concurrent_gen else 'off'}  "
          f"predict-fixes={'on' if args.predict_fixes else 'off'}  "
          f"prompts={args.prompts} route={args.route} amp={args.amp or 'off'}  "
          f"cache-write={'async' if args.async_cache_write else 'inline'}"
          + ("  scene=on" if args.scene else "")
          + (f"  woosh-upgrade={args.woosh_upgrade} first-prompt={woosh_styles(args)[0]} "
             f"solver={args.woosh_upgrade_solver}" if args.woosh_upgrade != "off" else "")
          + (f" best={args.woosh_upgrade_best}" if args.woosh_upgrade_best > 1 else "")
          + (f" score={args.woosh_upgrade_score}" if args.woosh_upgrade_score != "request" else "")
          + (f"  pool={pool.describe()}" if pool is not None else "")
          + (f"  writer={writer.describe()} swap={args.writer_swap}" if writer is not None else ""),
          flush=True)
    if writer is not None:
        writer.start()
    if pool is not None:
        pool.start()
    print("ctrl-c to stop", flush=True)

    stop = threading.Event()
    try:
        while not stop.wait(1.0):
            pass
    except KeyboardInterrupt:
        print("shutting down")
    finally:
        server.deepening = False
        if pool is not None:
            pool.stop()
        if writer is not None:
            writer.stop()
        server.shutdown()
        server.server_close()
    return 0


def template_of(args):
    from sarge.prompts import parse_template
    style = getattr(args, "prompts", None)
    return style if parse_template(style) else None


def make_pool(args, writer):
    if getattr(args, "woosh_upgrade_pool", "off") == "off":
        return None
    from sarge import pool as prompt_pool
    sources = [x.strip() for x in str(args.woosh_upgrade_pool).split(",") if x.strip()]
    store = args.writer_store
    if not os.path.isdir(store) and os.path.splitext(store)[1]:
        store = os.path.dirname(store)
    try:
        affinity = int(str(args.writer_affinity), 0)
        p = prompt_pool.build(sources, primary=writer, model=args.writer_model, binary=args.writer_bin,
                              port=args.writer_port, threads=args.writer_threads, affinity=affinity, store=store,
                              when=tuple(w.strip() for w in args.writer_when.split(",") if w.strip()),
                              log_path=os.path.join(store, "llama-server.log"), template=template_of(args),
                              ground_index=args.writer_ground_index)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit("--woosh-upgrade-pool %s: %s" % (args.woosh_upgrade_pool, e))
    os.makedirs(store, exist_ok=True)
    return p


def make_writer(args):
    if args.writer == "off":
        if args.writer_ground != "off":
            raise SystemExit("--writer-ground %s needs --writer ground (words, caption) or --writer llama/ollama "
                             "(anchors)" % args.writer_ground)
        return None
    from sarge import writer as prompt_writer
    when = tuple(w.strip() for w in args.writer_when.split(",") if w.strip())
    bad = [w for w in when if w not in prompt_writer.WHEN]
    if bad:
        raise SystemExit("--writer-when: unknown trigger %s (choose from %s)" % (
            ", ".join(bad), ", ".join(prompt_writer.WHEN)))
    try:
        affinity = int(str(args.writer_affinity), 0)
    except ValueError:
        raise SystemExit("--writer-affinity must be a mask such as 0xEFF")
    if affinity >> 8 & 1:
        print("warning: --writer-affinity %s includes core 8" % args.writer_affinity, flush=True)
    store = args.writer_store
    if args.writer == "store" and not os.path.isfile(store):
        raise SystemExit("--writer store needs --writer-store FILE (a prompts_<id>.json), got %s" % store)
    try:
        w = prompt_writer.build(args.writer, model=args.writer_model, binary=args.writer_bin, port=args.writer_port,
                                threads=args.writer_threads, affinity=affinity, store=store, when=when,
                                readonly=args.writer_readonly, style=args.writer_style,
                                log_path=os.path.join(store if os.path.isdir(store) or not os.path.splitext(store)[1]
                                                      else os.path.dirname(store), "llama-server.log"),
                                ground=args.writer_ground, ground_index=args.writer_ground_index,
                                ground_k=args.writer_ground_k, template=template_of(args))
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit("--writer %s: %s" % (args.writer, e))
    if args.writer != "store":
        os.makedirs(store, exist_ok=True)
    return w


if __name__ == "__main__":
    raise SystemExit(main())
