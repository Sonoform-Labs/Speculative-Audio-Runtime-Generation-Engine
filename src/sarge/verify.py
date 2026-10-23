import re

from .prompts import template

CAPTION = "template:caption"
CAND_STRIDE = 4096


def request_text(event, material="", intensity="normal", context="default", phys=""):
    t = template(CAPTION, event, material, intensity, context, phys, "woosh")
    t = re.sub(r"^A single ", "", t)
    t = re.sub(r", recorded up close", "", t)
    return t.strip().rstrip(".").strip()


def key_text(key):
    return request_text(key.event_type, key.material, key.intensity, key.context, key.phys)


def candidate_variant(variant, j):
    return int(variant) + CAND_STRIDE * int(j)


def pick(scored):
    return max(scored, key=lambda x: (x[0], -x[1]))
