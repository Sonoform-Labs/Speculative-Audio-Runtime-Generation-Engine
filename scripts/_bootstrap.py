import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for _p in [_here, *_here.parents]:
    if (_p / "pyproject.toml").exists():
        _src = _p / "src"
        if _src.exists() and str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
        break
