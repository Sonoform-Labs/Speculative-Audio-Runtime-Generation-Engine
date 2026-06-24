import _bootstrap

import argparse
import json
import os
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


def scan_project(root):
    assets = os.path.join(root, "Assets")
    if not os.path.isdir(assets):
        assets = root
    cs, scenes, folders = set(), set(), set()
    for dirpath, _dirs, files in os.walk(assets):
        rel = os.path.relpath(dirpath, assets)
        if rel != "." and os.sep not in rel:
            folders.add(rel)
        for fn in files:
            if fn.endswith(".cs"):
                cs.add(fn[:-3])
            elif fn.endswith(".unity"):
                scenes.add(fn[:-6])
    digest = ("scripts: " + ", ".join(sorted(cs)[:120]) +
              "\nscenes: " + ", ".join(sorted(scenes)[:30]) +
              "\nfolders: " + ", ".join(sorted(folders)[:40]))
    return digest[:8000]


def call_ollama(model, description, digest, keys):
    parts = ["You are a game audio director."]
    if digest:
        parts.append("Digest of the game's Unity project (script names, scenes, folders):\n" + digest)
    if description:
        parts.append("Game description:\n" + description)
    parts.append(
        "From the above, INFER the game's setting, character and tone. Then for EACH sound event in the list "
        "below add a 'prompt': one vivid concise sentence describing that exact sound for a neural text-to-audio "
        "model, reflecting the inferred setting/character/materials/tone (e.g. a knight in plate armor means "
        "every footstep carries a metallic clank). Keep event_type, material and context EXACTLY as given; do "
        "not add or remove entries. Return ONLY JSON: "
        "{\"description\": \"<one-sentence inferred game summary>\", \"sounds\": [{event_type, material, context, prompt}, ...]}.")
    parts.append(json.dumps({"sounds": keys}))
    body = json.dumps({"model": model, "prompt": "\n\n".join(parts), "stream": False, "format": "json",
                       "options": {"temperature": 0.4}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.loads(r.read().decode())
    return json.loads(resp["response"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Scope a game's SFX palette into a prewarm manifest (LLM-authored prompts).")
    ap.add_argument("--description", default=None, help="game description text, or @path to a text file")
    ap.add_argument("--project", default=None, help="path to a Unity project; the LLM reads its code to self-scope")
    ap.add_argument("--vocab", default=None, help="optional sfx_vocabulary.json exported from Unity")
    ap.add_argument("--surfaces", default="grass,stone,metal,wood")
    ap.add_argument("--items", default="gun")
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--model", default="qwen2.5-coder:14b")
    ap.add_argument("--out", default="manifest.json")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args(argv)

    desc = args.description or ""
    if desc.startswith("@"):
        with open(desc[1:], "r", encoding="utf-8") as f:
            desc = f.read().strip()
    digest = scan_project(args.project) if args.project else ""

    surfaces = [s.strip() for s in args.surfaces.split(",") if s.strip()]
    if args.vocab:
        with open(args.vocab, "r", encoding="utf-8") as f:
            surfaces = list(dict.fromkeys(json.load(f).get("surfaces", surfaces)))
    items = [s.strip() for s in args.items.split(",") if s.strip()]
    keys = base_keys(surfaces, items)

    used_llm = False
    inferred = desc
    if not args.no_llm:
        try:
            parsed = call_ollama(args.model, desc, digest, keys)
            inferred = parsed.get("description") or desc
            by = {(k["event_type"], k["material"], k.get("context", "default")): k for k in keys}
            for a in parsed.get("sounds", []) or []:
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

    manifest = {"description": inferred, "llm": used_llm, "model": args.model if used_llm else None, "sounds": keys}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.out}  ({len(keys)} sounds, llm={'yes' if used_llm else 'no'})")
    if inferred:
        print(f"inferred: {inferred}")
    for k in keys:
        print(f"  {k['event_type']}/{k['material']}: {k['prompt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
