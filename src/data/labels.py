"""RVL-CDIP class names, download locations, split-list files and path helpers.

Verified 2026-10-19 against ``huggingface.co/datasets/aharley/rvl_cdip``:
    * ``data/rvl-cdip.tar.gz`` (38,762,320,458 bytes, no login needed)
    * ``data/{train,val,test}.txt``: one ``<relative path> <class index>`` per line
    * inside the archive every image is stored as ``images/<relative path>``
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path, PurePosixPath

import requests

logger = logging.getLogger(__name__)

# Index order matches the label files and the Hugging Face loader script.
CLASSES = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]
NUM_CLASSES = len(CLASSES)
INVOICE_IDX = CLASSES.index("invoice")  # the one class the client also has (PRD G4)

HF_DATA_URL = "https://huggingface.co/datasets/aharley/rvl_cdip/resolve/main/data"
ARCHIVE_URL = f"{HF_DATA_URL}/rvl-cdip.tar.gz"
ARCHIVE_BYTES = 38_762_320_458
LABEL_FILE_URLS = {split: f"{HF_DATA_URL}/{split}.txt" for split in ("train", "val", "test")}
ARCHIVE_PREFIX = "images/"

# Split lists written by ``make_splits`` (CSV: path,label).
# Dev is a subset of Main, which is a subset of Large (PRD 4.3).
SPLIT_LISTS = ("dev_train", "dev_val", "main_train", "main_val", "large_train", "test")


def official_split(list_name: str) -> str:
    """Map a split-list name (e.g. ``main_val``) to its official split (``val``)."""
    return list_name.rsplit("_", 1)[-1] if list_name != "test" else "test"


def tier_split_lists(tier: str) -> tuple[str, str]:
    """Return the (train, val) split-list names for a data tier.

    The Large tier reuses Main's validation set (PRD 4.3 table).
    """
    if tier == "dev":
        return "dev_train", "dev_val"
    if tier == "main":
        return "main_train", "main_val"
    if tier == "large":
        return "large_train", "main_val"
    raise ValueError(f"Unknown tier {tier!r}; expected dev, main or large")


def png_relpath(source_relpath: str) -> str:
    """Working-copy file name for an RVL-CDIP image: same folders, ``.png`` suffix."""
    return str(PurePosixPath(source_relpath).with_suffix(".png"))


def archive_member_to_relpath(member_name: str) -> str | None:
    """Turn an archive member name into the label-file path, or None if not an image."""
    name = member_name[2:] if member_name.startswith("./") else member_name
    if not name.startswith(ARCHIVE_PREFIX):
        return None
    return name[len(ARCHIVE_PREFIX):]


def read_label_file(path: str | os.PathLike) -> list[tuple[str, int]]:
    """Parse an official label file into ``(relative path, class index)`` pairs."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # rsplit: the label is always the last token, even if a path had spaces.
            rel_path, label = line.rsplit(" ", 1)
            label = int(label)
            if not 0 <= label < NUM_CLASSES:
                raise ValueError(f"{path}:{line_no}: label {label} out of range")
            rows.append((rel_path, label))
    return rows


def download_label_files(dest_dir: str | os.PathLike) -> dict[str, Path]:
    """Download the three official label files into ``dest_dir`` (skipped if present)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, url in LABEL_FILE_URLS.items():
        path = dest_dir / f"{split}.txt"
        if not path.exists():
            logger.info("Downloading %s", url)
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(response.content)
            os.replace(tmp, path)
        paths[split] = path
    return paths


def read_split_csv(path: str | os.PathLike) -> list[tuple[str, int]]:
    """Read a split list written by :func:`write_split_csv`."""
    with open(path, newline="", encoding="utf-8") as f:
        return [(row["path"], int(row["label"])) for row in csv.DictReader(f)]


def write_split_csv(path: str | os.PathLike, rows: list[tuple[str, int]]) -> None:
    """Write ``(path, label)`` rows as a CSV with a ``path,label`` header."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["path", "label"])
        writer.writerows(rows)
