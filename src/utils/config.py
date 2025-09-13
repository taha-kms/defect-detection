from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import copy
import yaml


def _deep_merge(a: dict, b: dict) -> dict:
    """Deep-merge dict b into a (non-destructive)."""
    out = copy.deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _assign_by_dots(d: dict, dotted_key: str, value: Any):
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path, *extra_paths: str | Path, overrides: Dict[str, Any] | None = None) -> dict:
    """
    Load one or more YAML files and deep-merge them left→right.
    `overrides` may set keys via dotted paths (e.g., "train.lr": 1e-3).
    """
    cfg = {}
    for p in (path, *extra_paths):
        if p is None:
            continue
        y = load_yaml(p)
        cfg = _deep_merge(cfg, y)
    if overrides:
        for k, v in overrides.items():
            _assign_by_dots(cfg, k, v)
    return cfg