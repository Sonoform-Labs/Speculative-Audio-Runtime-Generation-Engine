import _bootstrap

import argparse
import json
import urllib.request

from predictive_sfx.affordance import ITEM_MATERIAL, ITEM_SOUNDS

OLLAMA = "http://localhost:11434/api/generate"


def base_keys(surfaces, items):
    keys = []
    for s in surfaces:
        keys.append({"event_type": "footstep", "material": s, "context": "default"})
    for it in items:
        mat = ITEM_MATERIAL.get(it, it)
        for ev, _p in ITEM_SOUNDS.get(it, []):
            keys.append({"event_type": ev, "material": mat, "context": "default"})
    return keys


def heuristic_prompt(k):
    return f"a single {k['event_type']} on {k['material']}, dry isolated game sound effect"


def call_ollama(model, description, keys):
    instr = (
        "You are a game audio director. GAME DESCRIPTION:\n" + description +
        "\n\nBelow is the list of sound events this game triggers (JSON). For EACH entry add a 'prompt' "
        "field: one vivid, concise sentence describing that exact sound for a neural text-to-audio model, "
        "reflecting the game's setting, character, materials and tone (e.g. a knight in plate armor means "
        "every footstep carries a metallic clank). Keep event_type, material and context EXACTLY as given, "
        "do not add or remove entries. Return ONLY JSON: {\"sounds\": [ {event_type, material, context, prompt}, ... ]}.\n\n"
        + json.dumps({"sounds": keys}))
    body = json.dumps({"model": model, "prompt": instr, "stream": False, "format": "json",
                       "options": {"temperature": 0.4}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read().decode())
    parsed = json.loads(resp["response"])
    return parsed.get("sounds") if isinstance(parsed, dict) else parsed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scope a game's SFX palette into a prewarm manifest (LLM-authored prompts).")
    ap.add_argument("--description", required=True, help="game description text, or @path to a text file")
    ap.add_argument("--vocab", default=None, help="optional sfx_vocabulary.json exported from Unity")
    ap.add_argument("--surfaces", default="grass,stone,metal,wood")
    ap.add_argument("--items", default="gun")
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--model", default="qwen2.5-coder:14b")
    ap.add_argument("--out", default="manifest.json")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args(argv)

    desc = args.description
    if desc.startswith("@"):
        with open(desc[1:], "r", encoding="utf-8") as f:
            desc = f.read().strip()

    surfaces = [s.strip() for s in args.surfaces.split(",") if s.strip()]
    if args.vocab:
        with open(args.vocab, "r", encoding="utf-8") as f:
            surfaces = list(dict.fromkeys(json.load(f).get("surfaces", surfaces)))
    items = [s.strip() for s in args.items.split(",") if s.strip()]
    keys = base_keys(surfaces, items)

    used_llm = False
    if not args.no_llm:
        try:
            authored = call_ollama(args.model, desc, keys)
            by = {(k["event_type"], k["material"], k.get("context", "default")): k for k in keys}
            for a in authored or []:
                kk = (a.get("event_type"), a.get("material"), a.get("context", "default"))
                if kk in by and isinstance(a.get("prompt"), str) and a["prompt"].strip():
                    by[kk]["prompt"] = a["prompt"].strip()
            used_llm = True
        except Exception as e:
            print(f"LLM scoping failed ({e}); using template prompts")

    for k in keys:
        k.setdefault("prompt", heuristic_prompt(k))
        k["intensity"] = k.get("intensity", "normal")
        k["variants"] = args.variants

    manifest = {"description": desc, "llm": used_llm, "model": args.model if used_llm else None, "sounds": keys}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.out}  ({len(keys)} sounds, llm={'yes' if used_llm else 'no'})")
    for k in keys:
        print(f"  {k['event_type']}/{k['material']}: {k['prompt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
