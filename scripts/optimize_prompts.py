import _bootstrap

import argparse

import numpy as np
import torch
import torchaudio.functional as AF

from predictive_sfx.generators.stable_audio import StableAudioGenerator
from predictive_sfx.types import CacheKey

CATEGORIES = ["a footstep", "a gunshot", "music", "speech", "silence", "rain", "an explosion"]
FOOT = ("footstep", "walk")

CONFIGS = [
    dict(name="baseline", prompt=None, cfg=1.0, secs=None),
    dict(name="short_1s", prompt=None, cfg=1.0, secs=1.0),
    dict(name="cfg5_1s", prompt=None, cfg=5.0, secs=1.0),
    dict(name="foley_cfg5",
         prompt="foley sound effect, a single {intensity} footstep, a leather boot stepping on a {material} surface, close mic, dry, mono, no music",
         cfg=5.0, secs=1.0),
]


def rs(c, src, dst):
    if src == dst:
        return c.astype(np.float32)
    return AF.resample(torch.from_numpy(np.ascontiguousarray(c)).float(), src, dst).numpy().astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Critic-in-the-loop prompt/param optimization scored by the panel.")
    ap.add_argument("--materials", nargs="+", default=["wood", "stone", "grass", "metal"])
    ap.add_argument("--variants", type=int, default=3)
    args = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    g = StableAudioGenerator()
    g.generate(CacheKey("footstep", "wood", "normal"), variant=0)
    full_secs = g._sample_size / g.sample_rate
    sr = g.sample_rate

    clips, cfgidx, labels = [], [], []
    for ci, cfg in enumerate(CONFIGS):
        g.prompt_template = cfg["prompt"]
        g.cfg_scale = cfg["cfg"]
        g.seconds_total = full_secs if cfg["secs"] is None else cfg["secs"]
        for mat in args.materials:
            for v in range(args.variants):
                clips.append(g.generate(CacheKey("footstep", mat, "normal"), variant=v).samples)
                cfgidx.append(ci)
                labels.append(mat)
    del g
    torch.cuda.empty_cache()

    from transformers import ClapModel, ClapProcessor
    cm = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
    cp = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    clips48 = [rs(c, sr, 48000) for c in clips]

    def clap_logits(texts):
        a = cp(audio=clips48, sampling_rate=48000, return_tensors="pt")
        t = cp(text=texts, return_tensors="pt", padding=True)
        inp = {k: v.to(dev) for k, v in {**a, **t}.items()}
        with torch.no_grad():
            return cm(**inp).logits_per_audio.float().cpu().numpy()

    cat = clap_logits(CATEGORIES)
    footidx = CATEGORIES.index("a footstep")
    matlog = clap_logits([f"the sound of a footstep on {m}" for m in args.materials])
    del cm
    torch.cuda.empty_cache()

    from transformers import ASTForAudioClassification, AutoFeatureExtractor
    an = "MIT/ast-finetuned-audioset-10-10-0.4593"
    am = ASTForAudioClassification.from_pretrained(an).to(dev).eval()
    afe = AutoFeatureExtractor.from_pretrained(an)
    aid = am.config.id2label
    ast_foot = []
    for c in clips:
        inp = afe(rs(c, sr, 16000), sampling_rate=16000, return_tensors="pt")
        inp = {k: v.to(dev) for k, v in inp.items()}
        with torch.no_grad():
            lg = am(**inp).logits[0]
        names = [aid[i].lower() for i in torch.topk(lg, 5).indices.cpu().tolist()]
        ast_foot.append(any(t in n for n in names for t in FOOT))
    del am
    torch.cuda.empty_cache()

    from panns_inference import AudioTagging
    from panns_inference.config import labels as plabels
    at = AudioTagging(checkpoint_path=None, device=dev)
    cw, _ = at.inference(np.stack([rs(c, sr, 32000) for c in clips]).astype(np.float32))
    panns_foot = []
    for row in cw:
        names = [plabels[i].lower() for i in np.argsort(row)[::-1][:5]]
        panns_foot.append(any(t in n for n in names for t in FOOT))
    del at
    torch.cuda.empty_cache()

    print(f"\n{'config':<12}{'CLAP_foot':>10}{'material':>10}{'AST_foot':>10}{'PANNs_foot':>12}")
    print("-" * 54)
    for ci, cfg in enumerate(CONFIGS):
        idxs = [i for i in range(len(clips)) if cfgidx[i] == ci]
        clap_f = np.mean([cat[i].argmax() == footidx for i in idxs])
        true = [args.materials.index(labels[i]) for i in idxs]
        mat_acc = np.mean([matlog[i].argmax() == true[k] for k, i in enumerate(idxs)])
        astf = np.mean([ast_foot[i] for i in idxs])
        pf = np.mean([panns_foot[i] for i in idxs])
        print(f"{cfg['name']:<12}{clap_f*100:9.0f}%{mat_acc*100:9.0f}%{astf*100:9.0f}%{pf*100:11.0f}%")
    print("\n(critic round 1: baseline vs shorter-gen / higher-cfg / foley-prompt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
