import os
from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Expand ~ in any string paths
    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, str) and obj.startswith("~"):
            return os.path.expanduser(obj)
        return obj
    return _expand(cfg)


def resolve_path(p: str, base: str | None = None) -> Path:
    p = os.path.expanduser(p)
    path = Path(p)
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path
