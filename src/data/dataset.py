"""PyTorch Dataset and DataLoaders reading the working copy through the split lists."""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.labels import png_relpath, read_split_csv, tier_split_lists
from src.data.transforms import build_transforms
from src.utils import seed_worker

logger = logging.getLogger(__name__)


class DocumentImageDataset(Dataset):
    """Grayscale document pages from the working copy, with integer class labels.

    Args:
        rows: ``(source path, label)`` pairs from a split list.
        image_root: working-copy folder (PNG paths are resolved relative to it).
        transform: callable applied to the grayscale PIL image.
    """

    def __init__(self, rows: list[tuple[str, int]], image_root: str | Path, transform=None):
        self.image_root = Path(image_root)
        self.rows = list(rows)  # original RVL-CDIP paths, kept for prediction files
        self.samples = [(self.image_root / png_relpath(path), label) for path, label in rows]
        self.transform = transform

    @classmethod
    def from_split(cls, split_csv: str | Path, image_root: str | Path, transform=None,
                   available: set[str] | None = None) -> "DocumentImageDataset":
        """Build a dataset from a split list, dropping images missing from the working copy.

        ``available`` is the set of PNG paths in the manifest. Images that were
        corrupt in the archive are absent from it and are skipped with a warning.
        """
        rows = read_split_csv(split_csv)
        if available is not None:
            kept = [(p, y) for p, y in rows if png_relpath(p) in available]
            if len(kept) < len(rows):
                logger.warning("%s: %d of %d images missing from the working copy, skipped",
                               Path(split_csv).name, len(rows) - len(kept), len(rows))
            rows = kept
        if not rows:
            raise RuntimeError(f"{split_csv}: no images available in {image_root}")
        return cls(rows, image_root, transform)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                image = image.convert("L")
        except (OSError, ValueError) as error:
            # Never crash a long training run on one bad file (PRD 4.4): log, use a neighbour.
            logger.warning("unreadable image %s (%r); using the next sample", path, error)
            return self[(index + 1) % len(self)]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def labels(self) -> list[int]:
        """Class label of every sample, in order."""
        return [label for _, label in self.samples]


def load_manifest_paths(data_dir: str | Path) -> set[str] | None:
    """PNG paths listed in ``data_dir/manifest.csv``, or None if there is no manifest."""
    manifest = Path(data_dir) / "manifest.csv"
    if not manifest.exists():
        logger.warning("no manifest.csv in %s; assuming every listed image exists", data_dir)
        return None
    with open(manifest, newline="", encoding="utf-8") as f:
        return {row["path"] for row in csv.DictReader(f)}


def compute_mean_std(dataset: DocumentImageDataset, image_size: int, max_images: int = 2000,
                     seed: int = 0) -> tuple[list[float], list[float]]:
    """Single-channel pixel mean/std over a random sample of the (training) dataset.

    Used to normalize the from-scratch CNN with dataset statistics (PRD 5.1.4).
    Images are resized exactly as during training, without augmentation.
    """
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    indices = indices[:max_images]
    total, total_sq, count = 0.0, 0.0, 0
    for i in indices:
        path, _ = dataset.samples[i]
        with Image.open(path) as image:
            image = image.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float64) / 255.0
        total += pixels.sum()
        total_sq += (pixels ** 2).sum()
        count += pixels.size
    mean = float(total / count)  # plain floats: they go into YAML/JSON logs
    std = max(float(total_sq / count - mean ** 2) ** 0.5, 1e-6)
    return [round(mean, 4)], [round(std, 4)]


def build_datasets(cfg: dict, mean: list[float], std: list[float]):
    """Return ``(train_dataset, val_dataset)`` for the configured tier."""
    data_dir = cfg["paths"]["data_dir"]
    splits_dir = cfg["paths"]["splits_dir"]
    train_list, val_list = tier_split_lists(cfg["data"]["tier"])
    available = load_manifest_paths(data_dir)
    train_ds = DocumentImageDataset.from_split(
        splits_dir / f"{train_list}.csv", data_dir, build_transforms(cfg, mean, std, train=True), available)
    val_ds = DocumentImageDataset.from_split(
        splits_dir / f"{val_list}.csv", data_dir, build_transforms(cfg, mean, std, train=False), available)
    return train_ds, val_ds


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int,
                drop_last: bool = False) -> DataLoader:
    """DataLoader with seeded shuffling and workers, pinned memory on GPU machines."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
