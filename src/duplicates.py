"""Optional near-duplicate check between the training subset and the test set (PRD 4.4, 6).

RVL-CDIP is known to contain near-identical documents across its splits, which
can inflate test accuracy. This module finds, for every test page, the most
similar page in our training tier using a perceptual difference hash.

Usage:
    python -m src.duplicates find  --data-dir /content/data/working_copy --train-list main_train
    python -m src.duplicates sheet --csv <output_dir>/eval/test/duplicates.csv --data-dir ...
    python -m src.compare ... --dup-threshold 12        # accuracy with and without duplicates

Hash: the page is shrunk to 17 x 16 and each pixel is compared with its right
neighbour, giving 256 bits that describe the coarse layout. Two scans of the
same document should differ in only a few bits; unrelated pages in many more.
**Pick the threshold by looking at the ``sheet`` output** (pairs at increasing
distances) and report the value you chose. A hash only finds *visual*
near-duplicates; it is a lower bound on the true overlap.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import resolve_path  # noqa: E402
from src.data.labels import png_relpath, read_split_csv  # noqa: E402
from src.evaluate import INK, default_output_dir, pick  # noqa: E402
from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

HASH_SIDE = 16  # 16 x 16 = 256-bit hash
DISTANCE_BINS = [0, 4, 8, 12, 16, 24, 32, 48, 64]


def dhash(image: Image.Image) -> np.ndarray:
    """256-bit difference hash of a page, as 4 uint64 words."""
    small = np.asarray(image.convert("L").resize((HASH_SIDE + 1, HASH_SIDE), Image.Resampling.BOX), dtype=np.int16)
    bits = small[:, 1:] > small[:, :-1]
    return np.packbits(bits.ravel()).view(np.uint64)


def hash_files(paths: list[str], data_dir: Path) -> np.ndarray:
    """Hash every listed page (source paths); returns an ``(N, 4)`` uint64 array."""
    hashes = np.zeros((len(paths), HASH_SIDE * HASH_SIDE // 64), dtype=np.uint64)
    for i, path in enumerate(paths):
        with Image.open(data_dir / png_relpath(path)) as image:
            hashes[i] = dhash(image)
        if (i + 1) % 10000 == 0:
            logger.info("hashed %d / %d", i + 1, len(paths))
    return hashes


def _popcount(words: np.ndarray) -> np.ndarray:
    """Number of set bits per uint64 (NumPy >= 2 has a fast built-in)."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(words)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return table[words.view(np.uint8)].reshape(*words.shape, 8).sum(-1)


def nearest(train: np.ndarray, test: np.ndarray, chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """For each test hash: Hamming distance to, and index of, the closest training hash."""
    best_dist = np.empty(len(test), dtype=np.int32)
    best_idx = np.empty(len(test), dtype=np.int64)
    for start in range(0, len(test), chunk):
        block = test[start:start + chunk]
        dist = _popcount(block[:, None, :] ^ train[None, :, :]).sum(-1)
        best_idx[start:start + chunk] = dist.argmin(1)
        best_dist[start:start + chunk] = dist.min(1)
    return best_dist, best_idx


def find_duplicates(data_dir: Path, splits_dir: Path, train_list: str, test_list: str, out_csv: Path) -> pd.DataFrame:
    """Write ``out_csv`` with each test page's nearest training page and their distance."""
    train_rows = [r for r in read_split_csv(splits_dir / f"{train_list}.csv") if (data_dir / png_relpath(r[0])).exists()]
    test_rows = [r for r in read_split_csv(splits_dir / f"{test_list}.csv") if (data_dir / png_relpath(r[0])).exists()]
    logger.info("hashing %d train and %d test pages", len(train_rows), len(test_rows))
    train_hash = hash_files([p for p, _ in train_rows], data_dir)
    test_hash = hash_files([p for p, _ in test_rows], data_dir)
    dist, idx = nearest(train_hash, test_hash)
    table = pd.DataFrame({
        "path": [p for p, _ in test_rows],
        "label": [y for _, y in test_rows],
        "nearest_train_path": [train_rows[i][0] for i in idx],
        "nearest_train_label": [train_rows[i][1] for i in idx],
        "distance": dist,
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    for threshold in (0, 4, 8, 12, 16, 24):
        near = table.distance <= threshold
        same = (table.label == table.nearest_train_label)[near].mean() if near.any() else float("nan")
        logger.info("distance <= %2d: %5d test pages (%.2f%%), label agrees with the train twin in %.0f%%",
                    threshold, near.sum(), 100 * near.mean(), 100 * same)
    logger.info("-> %s", out_csv)
    return table


def pair_sheet(csv_path: Path, data_dir: Path, per_bin: int = 3, seed: int = 42) -> Path:
    """Contact sheet of test/train pairs from increasing distance bins, to choose a threshold by eye."""
    table = pd.read_csv(csv_path)
    bins = list(zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]))
    fig, axes = plt.subplots(len(bins), 2 * per_bin, figsize=(2.2 * 2 * per_bin, 2.6 * len(bins)), squeeze=False)
    for r, (low, high) in enumerate(bins):
        subset = table[(table.distance >= low) & (table.distance < high)]
        subset = subset.sample(n=min(per_bin, len(subset)), random_state=seed)
        for c in range(per_bin):
            for k, column in enumerate(("path", "nearest_train_path")):
                ax = axes[r, 2 * c + k]
                ax.axis("off")
                if c < len(subset):
                    row = subset.iloc[c]
                    with Image.open(data_dir / png_relpath(row[column])) as image:
                        ax.imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
                    ax.set_title(f"{'test' if k == 0 else 'train'}  d={row.distance}", fontsize=8, color=INK)
        axes[r, 0].text(-0.15, 0.5, f"{low}-{high - 1}", transform=axes[r, 0].transAxes, rotation=90,
                        ha="center", va="center", fontsize=10, color=INK)
    fig.suptitle("Nearest training page for test pages, by hash distance (rows); pick the largest distance that is "
                 "still clearly the same document", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out = csv_path.with_name("duplicates_sheet.png")
    fig.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point with ``find`` and ``sheet`` sub-commands."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_find = sub.add_parser("find")
    p_find.add_argument("--data-dir")
    p_find.add_argument("--splits-dir")
    p_find.add_argument("--train-list", default="main_train")
    p_find.add_argument("--test-list", default="test")
    p_find.add_argument("--out", help="default: <output_dir>/eval/test/duplicates.csv")
    p_sheet = sub.add_parser("sheet")
    p_sheet.add_argument("--csv", help="default: <output_dir>/eval/test/duplicates.csv")
    p_sheet.add_argument("--data-dir")
    args = parser.parse_args(argv)
    setup_logging()
    data_dir = Path(pick(args.data_dir, os.environ.get("DOCCLS_DATA_DIR"), resolve_path("data/working_copy")))
    default_csv = default_output_dir() / "eval" / "test" / "duplicates.csv"
    if args.command == "find":
        splits_dir = Path(pick(args.splits_dir, os.environ.get("DOCCLS_SPLITS_DIR"), resolve_path("splits")))
        find_duplicates(data_dir, splits_dir, args.train_list, args.test_list, Path(args.out) if args.out else default_csv)
    else:
        print(pair_sheet(Path(args.csv) if args.csv else default_csv, data_dir))


if __name__ == "__main__":
    main()
