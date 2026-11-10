"""Small shared helpers: seeding, logging, safe file writes, environment info."""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs).

    With ``deterministic=False`` cuDNN autotuning stays on: faster, but two runs
    with the same seed can differ slightly in the last digits.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: give each worker its own NumPy/random seed.

    PyTorch already seeds each worker's torch RNG from the loader's generator;
    this extends that seed to the other two RNGs.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_logging(log_file: str | os.PathLike | None = None) -> logging.Logger:
    """Log to stdout and, optionally, append to ``log_file``. Returns the root logger."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    return root


def atomic_torch_save(obj: Any, path: str | os.PathLike) -> None:
    """``torch.save`` to a temp file, then rename.

    If Colab disconnects mid-write, the previous checkpoint is still intact
    instead of being replaced by a truncated file.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(obj: Any, path: str | os.PathLike) -> None:
    """Write ``obj`` as indented JSON (atomically, same reason as above)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def append_csv_row(path: str | os.PathLike, row: dict) -> None:
    """Append ``row`` to a CSV file, writing the header if the file is new."""
    path = Path(path)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def environment_info() -> dict[str, Any]:
    """Library versions and hardware, recorded with every run for reproducibility."""
    import torchvision

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import timm

        info["timm"] = timm.__version__
    except ImportError:
        info["timm"] = None
    return info
