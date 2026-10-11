from __future__ import annotations

import collections
import csv
import importlib.metadata as md
import inspect
import pathlib


def _names(dist) -> set:
    declared = dist.read_text("top_level.txt")
    if declared:
        return set(declared.split())
    text = dist.read_text("RECORD") or dist.read_text("installed-files.txt") or dist.read_text("SOURCES.txt") or ""
    out = set()
    for row in csv.reader(text.splitlines()):
        if not row or not row[0]:
            continue
        path = pathlib.PurePosixPath(row[0])
        name = path.parts[0] if len(path.parts) > 1 else (inspect.getmodulename(str(path)) or str(path))
        if "." not in name:
            out.add(name)
    return out


def packages_distributions() -> dict:
    out = collections.defaultdict(list)
    for dist in md.distributions():
        for name in _names(dist):
            out[name].append(dist.metadata["Name"])
    return dict(out)


def install() -> bool:
    if getattr(md.packages_distributions, "_sarge_fast", False):
        return False
    packages_distributions._sarge_fast = True
    md.packages_distributions = packages_distributions
    return True
