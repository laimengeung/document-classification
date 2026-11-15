"""Generate the stratified, seeded subset lists in ``splits/`` (PRD 4.3, Step 1).

Usage:
    python -m src.data.make_splits --labels-dir data/labels --out-dir splits

Sampling: within each class the official train (and val) list is sorted and
shuffled once with a fixed seed, and every tier takes a *prefix* of that order.
That makes Dev a subset of Main and Main a subset of Large by construction, so
the archive only needs to be read once even if Large is used later.
The test list is the full official test split and is never sampled.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.config import resolve_path
from src.data.labels import (
    CLASSES,
    NUM_CLASSES,
    download_label_files,
    read_label_file,
    write_split_csv,
)
from src.utils import setup_logging, write_json

logger = logging.getLogger(__name__)

EXPECTED_SPLIT_SIZES = {"train": 320_000, "val": 40_000, "test": 40_000}
TIER_SIZES = {  # images per class
    "dev": {"train": 200, "val": 50},
    "main": {"train": 2_000, "val": 500},
    "large": {"train": 5_000},  # Large reuses main_val
}


def check_label_file(split: str, rows: list[tuple[str, int]]) -> dict[str, int]:
    """Log and sanity-check class counts for one official split. Returns per-class counts."""
    counts = Counter(label for _, label in rows)
    if len(counts) != NUM_CLASSES:
        raise ValueError(f"{split}: expected {NUM_CLASSES} classes, found {len(counts)}")
    if len(rows) != EXPECTED_SPLIT_SIZES[split]:
        logger.warning("%s: expected %d images, found %d", split, EXPECTED_SPLIT_SIZES[split], len(rows))
    paths = [p for p, _ in rows]
    if len(set(paths)) != len(paths):
        logger.warning("%s: %d duplicate paths", split, len(paths) - len(set(paths)))
    logger.info("%s: %d images, per class min %d / max %d", split, len(rows), min(counts.values()), max(counts.values()))
    return {CLASSES[k]: v for k, v in sorted(counts.items())}


def shuffled_by_class(rows: list[tuple[str, int]], seed: int) -> dict[int, list[str]]:
    """Group paths by class, sort them, and shuffle each class with its own seeded RNG."""
    by_class: dict[int, list[str]] = defaultdict(list)
    for path, label in rows:
        by_class[label].append(path)
    for label, paths in by_class.items():
        paths.sort()  # independent of the label file's line order
        random.Random(seed * 1000 + label).shuffle(paths)
    return by_class


def take_prefix(by_class: dict[int, list[str]], per_class: int) -> list[tuple[str, int]]:
    """First ``per_class`` paths of every class, as ``(path, label)`` rows."""
    rows = []
    for label in range(NUM_CLASSES):
        if len(by_class[label]) < per_class:
            raise ValueError(f"class {CLASSES[label]} has only {len(by_class[label])} images")
        rows.extend((path, label) for path in by_class[label][:per_class])
    return rows


def make_splits(labels_dir: Path, out_dir: Path, seed: int) -> dict:
    """Download label files if needed, check them, and write every split list."""
    label_paths = download_label_files(labels_dir)
    official = {split: read_label_file(path) for split, path in label_paths.items()}
    summary = {"seed": seed, "official": {}, "lists": {}}
    for split, rows in official.items():
        summary["official"][split] = {"total": len(rows), "per_class": check_label_file(split, rows)}

    train_order = shuffled_by_class(official["train"], seed)
    val_order = shuffled_by_class(official["val"], seed + 1)
    lists = {
        "dev_train": take_prefix(train_order, TIER_SIZES["dev"]["train"]),
        "dev_val": take_prefix(val_order, TIER_SIZES["dev"]["val"]),
        "main_train": take_prefix(train_order, TIER_SIZES["main"]["train"]),
        "main_val": take_prefix(val_order, TIER_SIZES["main"]["val"]),
        "large_train": take_prefix(train_order, TIER_SIZES["large"]["train"]),
        "test": sorted(official["test"]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in lists.items():
        write_split_csv(out_dir / f"{name}.csv", rows)
        summary["lists"][name] = len(rows)
        logger.info("wrote %s.csv (%d rows)", name, len(rows))

    # The tiers must nest, or the "Dev is a subset of Main" promise is broken.
    as_set = {name: set(rows) for name, rows in lists.items()}
    assert as_set["dev_train"] <= as_set["main_train"] <= as_set["large_train"]
    assert as_set["dev_val"] <= as_set["main_val"]
    write_json(summary, out_dir / "summary.json")
    return summary


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-dir", default="data/labels", help="where the official label files are cached")
    parser.add_argument("--out-dir", default="splits", help="where to write the split CSVs (committed to git)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    setup_logging()
    make_splits(resolve_path(args.labels_dir), resolve_path(args.out_dir), args.seed)


if __name__ == "__main__":
    main()
