import _bootstrap

import argparse
import json

from predictive_sfx.sidecar import SfxClient


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bake a sound manifest into the running sidecar (prewarm + persist).")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)

    with open(args.manifest, "r", encoding="utf-8") as f:
        man = json.load(f)
    sounds = man.get("sounds")
    if not isinstance(sounds, list) or not sounds:
        print(f"invalid manifest: missing/empty 'sounds' list in {args.manifest}")
        return 2
    keys = []
    for i, s in enumerate(sounds):
        if not isinstance(s, dict) or "event_type" not in s or "material" not in s:
            print(f"invalid manifest: sound #{i} needs event_type and material")
            return 2
        keys.append({"event_type": s["event_type"], "material": s["material"],
                     "intensity": s.get("intensity", "normal"), "context": s.get("context", "default"),
                     "prompt": s.get("prompt")})

    with SfxClient(args.host, args.port) as c:
        print(f"connected -> generator={c.welcome.get('generator')}")
        ack = c.prewarm(keys)
        print(f"baked {len(keys)} sounds -> cached_keys={ack['cached_keys']}  spec_gen={ack['spec_gen']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
