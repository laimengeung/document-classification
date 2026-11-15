"""Copy the working copy from Google Drive to fast local disk, and verify it.

Usage:
    python -m src.data.local_copy unpack --src /content/drive/MyDrive/ain320_doccls/working_copy \
                                         --dst /content/data/working_copy
    python -m src.data.local_copy verify --data-dir /content/data/working_copy \
                                         --lists main_train main_val test

Reading thousands of small files straight from Drive is slow, so every Colab
session starts by unpacking the shard tars onto the VM's local disk (PRD 5.1).
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path

from src.config import resolve_path
from src.data.labels import CLASSES, NUM_CLASSES, png_relpath, read_split_csv
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def unpack_working_copy(src_dir: Path, dst_dir: Path) -> int:
    """Extract every finished shard in ``src_dir/shards`` into ``dst_dir``.

    A shard counts as finished when its CSV exists. Already-extracted shards are
    skipped (marker files in ``dst_dir/.unpacked``), so re-running is cheap.
    Returns the number of shards extracted in this call.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    markers = dst_dir / ".unpacked"
    markers.mkdir(exist_ok=True)
    extracted = 0
    shard_csvs = sorted((src_dir / "shards").glob("shard_*.csv"))
    if not shard_csvs:
        raise FileNotFoundError(f"No finished shards in {src_dir / 'shards'}; build the working copy first")
    for csv_path in shard_csvs:
        marker = markers / csv_path.stem
        if marker.exists():
            continue
        with tarfile.open(csv_path.with_suffix(".tar")) as tar:
            if hasattr(tarfile, "data_filter"):  # Python 3.12+: refuse unsafe paths
                tar.extractall(dst_dir, filter="data")
            else:
                tar.extractall(dst_dir)
        marker.touch()
        extracted += 1
        logger.info("unpacked %s", csv_path.stem)
    manifest = src_dir / "manifest.csv"
    if manifest.exists():
        shutil.copyfile(manifest, dst_dir / "manifest.csv")
    else:  # build still in progress: assemble a manifest from the shard CSVs
        rows = []
        for csv_path in shard_csvs:
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows.extend(csv.DictReader(f))
        with open(dst_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    logger.info("working copy ready in %s (%d new shards)", dst_dir, extracted)
    return extracted


def verify_working_copy(data_dir: Path, splits_dir: Path, list_names: list[str]) -> bool:
    """Check that every image in the given split lists is present on disk.

    Prints per-list and per-class counts (the "verify counts" step of the PRD
    checklist). Returns True when nothing is missing.
    """
    all_ok = True
    for name in list_names:
        rows = read_split_csv(splits_dir / f"{name}.csv")
        expected = Counter(label for _, label in rows)
        present = Counter(label for path, label in rows if (data_dir / png_relpath(path)).exists())
        missing = sum(expected.values()) - sum(present.values())
        status = "OK" if missing == 0 else f"{missing} MISSING"
        print(f"{name:12s} {sum(present.values()):6d} / {sum(expected.values()):6d}  {status}")
        if missing:
            all_ok = False
            for label in range(NUM_CLASSES):
                if present[label] != expected[label]:
                    print(f"    {CLASSES[label]:24s} {present[label]} / {expected[label]}")
    return all_ok


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point with ``unpack`` and ``verify`` sub-commands."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_unpack = sub.add_parser("unpack", help="extract shards from Drive to local disk")
    p_unpack.add_argument("--src", required=True)
    p_unpack.add_argument("--dst", required=True)
    p_verify = sub.add_parser("verify", help="check split lists against files on disk")
    p_verify.add_argument("--data-dir", required=True)
    p_verify.add_argument("--splits-dir", default="splits")
    p_verify.add_argument("--lists", nargs="+", default=["main_train", "main_val", "test"])
    args = parser.parse_args(argv)
    setup_logging()
    if args.command == "unpack":
        unpack_working_copy(resolve_path(args.src), resolve_path(args.dst))
    else:
        ok = verify_working_copy(resolve_path(args.data_dir), resolve_path(args.splits_dir), args.lists)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
