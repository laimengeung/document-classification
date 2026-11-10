"""Experiment configuration: YAML files plus environment and command-line overrides.

Precedence (lowest to highest):
    1. the YAML file in ``configs/``
    2. environment variables for paths (so Colab can point at Google Drive)
    3. ``--tier`` and ``--set key.sub=value`` on the command line
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Environment variable -> config key under ``paths``.
PATH_ENV_VARS = {
    "DOCCLS_SPLITS_DIR": "splits_dir",
    "DOCCLS_DATA_DIR": "data_dir",
    "DOCCLS_OUTPUT_DIR": "output_dir",
}


def apply_override(cfg: dict, override: str) -> None:
    """Apply one ``dotted.key=value`` override to ``cfg`` in place.

    The value is parsed as YAML, so ``1e-4`` becomes a float, ``true`` a bool,
    ``[1, 2]`` a list and ``null`` None.
    """
    if "=" not in override:
        raise ValueError(f"Override must look like key.sub=value, got {override!r}")
    dotted_key, raw_value = override.split("=", 1)
    keys = dotted_key.strip().split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = yaml.safe_load(raw_value)


def resolve_path(path: str | os.PathLike) -> Path:
    """Return ``path`` as an absolute Path; relative paths are taken from the repo root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(
    path: str | os.PathLike,
    tier: str | None = None,
    overrides: Iterable[str] = (),
) -> dict[str, Any]:
    """Load a YAML config and apply environment and CLI overrides.

    Args:
        path: YAML file, absolute or relative to the repo root.
        tier: optional data tier (``dev``, ``main`` or ``large``).
        overrides: ``key.sub=value`` strings, applied last.

    Returns:
        The config dict. ``paths.*`` entries are absolute ``Path`` objects and
        ``run_name`` is always filled in.
    """
    with open(resolve_path(path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.setdefault("paths", {})
    for env_var, key in PATH_ENV_VARS.items():
        if os.environ.get(env_var):
            paths[key] = os.environ[env_var]

    if tier is not None:
        cfg["data"]["tier"] = tier
    for override in overrides:
        apply_override(cfg, override)

    for key in ("splits_dir", "data_dir", "output_dir"):
        paths[key] = resolve_path(paths[key])

    if not cfg.get("run_name"):
        cfg["run_name"] = f"{cfg['model']['name']}_{cfg['data']['tier']}"
    return cfg


def to_serializable(cfg: dict) -> dict:
    """Deep-copy ``cfg`` with Paths turned into strings (for YAML/JSON/checkpoints)."""

    def convert(value):
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        return value

    return convert(copy.deepcopy(cfg))


def save_config(cfg: dict, path: str | os.PathLike) -> None:
    """Write the fully resolved config next to a run's outputs."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(cfg), f, sort_keys=False)
