"""Build the small working copy from the 38.8 GB RVL-CDIP archive in one streaming pass.

Usage (Colab, writing to Drive):
    python -m src.data.build_working_copy \
        --out-dir /content/drive/MyDrive/ain320_doccls/working_copy \
        --lists main_train main_val test

How it works (PRD 5.1.5):
    * The ``.tar.gz`` is read sequentially straight from Hugging Face; neither
      the archive nor the raw TIFFs are ever stored.
    * Only files in the chosen split lists are kept. Each is converted to
      grayscale, resized (see ``preprocess.py``) and encoded as PNG.
    * PNGs are packed into **shards** of a few thousand images. A finished
      shard is copied to ``--out-dir`` as ``shards/shard_NNNNN.tar`` followed
      by its ``shard_NNNNN.csv``; the CSV only exists once the tar is complete.
      A few large files copy to and from Google Drive much faster than tens of
      thousands of small PNGs.
    * **Resumable:** on restart, every image listed in a finished shard CSV (or
      in ``corrupt.csv``) is skipped. A disconnect loses at most one shard.
    * Unreadable images are logged to ``corrupt.csv`` and skipped, not fatal.
    * At the end, all shard CSVs are merged into ``manifest.csv``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import requests
from PIL import Image

from src.config import resolve_path
from src.data.labels import (
    ARCHIVE_BYTES,
    ARCHIVE_URL,
    CLASSES,
    archive_member_to_relpath,
    official_split,
    png_relpath,
    read_split_csv,
)
from src.data.preprocess import WORKING_COPY_SIZE, to_working_copy
from src.utils import setup_logging, write_json

logger = logging.getLogger(__name__)

MANIFEST_FIELDS = [
    "path", "source_path", "label", "class_name", "split",
    "orig_width", "orig_height", "png_bytes", "shard",
]
CORRUPT_FIELDS = ["source_path", "label", "split", "error"]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_wanted(splits_dir: Path, list_names: list[str]) -> dict[str, tuple[int, str]]:
    """Map every wanted source path to ``(label, official split)``.

    Lists overlap (e.g. Dev is inside Main), so the union is taken.
    """
    wanted: dict[str, tuple[int, str]] = {}
    for name in list_names:
        split = official_split(name)
        for path, label in read_split_csv(splits_dir / f"{name}.csv"):
            previous = wanted.setdefault(path, (label, split))
            if previous != (label, split):
                raise ValueError(f"{path} appears with two different labels/splits")
    return wanted


def read_done(out_dir: Path) -> tuple[set[str], int]:
    """Return (source paths already handled, index for the next shard)."""
    done: set[str] = set()
    next_index = 0
    for csv_path in sorted((out_dir / "shards").glob("shard_*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            done.update(row["source_path"] for row in csv.DictReader(f))
        next_index = max(next_index, int(csv_path.stem.split("_")[1]) + 1)
    corrupt = out_dir / "corrupt.csv"
    if corrupt.exists():
        with open(corrupt, newline="", encoding="utf-8") as f:
            done.update(row["source_path"] for row in csv.DictReader(f))
    return done, next_index


# --------------------------------------------------------------------------- #
# Archive stream
# --------------------------------------------------------------------------- #
class CountingReader:
    """File-like wrapper that counts compressed bytes read (for progress reports)."""

    def __init__(self, raw):
        self.raw = raw
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.raw.read(size)
        self.bytes_read += len(chunk)
        return chunk


def open_archive(source: str):
    """Open the archive as a raw byte stream. Returns ``(stream, closer)``.

    ``source`` is either an http(s) URL or a local file path.
    """
    if source.startswith(("http://", "https://")):
        response = requests.get(source, stream=True, timeout=(30, 300))
        response.raise_for_status()
        response.raw.decode_content = False  # we want the raw gzip bytes
        return response.raw, response.close
    handle = open(source, "rb")
    return handle, handle.close


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
class ShardWriter:
    """Packs PNGs into a local tar and publishes it to ``out_dir/shards`` when full."""

    def __init__(self, out_dir: Path, tmp_dir: Path, shard_size: int, start_index: int):
        self.shard_dir = out_dir / "shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = tmp_dir
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.index = start_index
        self.tar: tarfile.TarFile | None = None
        self.rows: list[dict] = []

    @property
    def name(self) -> str:
        return f"shard_{self.index:05d}"

    def add(self, png_path: str, png_bytes: bytes, row: dict) -> None:
        """Add one encoded PNG to the current shard (publishing it if it is full)."""
        if self.tar is None:
            self.tar = tarfile.open(self.tmp_dir / f"{self.name}.tar", "w")
        info = tarfile.TarInfo(png_path)
        info.size = len(png_bytes)
        info.mtime = int(time.time())
        self.tar.addfile(info, io.BytesIO(png_bytes))
        self.rows.append({**row, "shard": self.name})
        if len(self.rows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        """Close the current shard and copy it to the output directory (tar first, then CSV)."""
        if self.tar is None:
            return
        self.tar.close()
        local_tar = self.tmp_dir / f"{self.name}.tar"
        final_tar = self.shard_dir / f"{self.name}.tar"
        shutil.copyfile(local_tar, final_tar.with_suffix(".tar.part"))
        os.replace(final_tar.with_suffix(".tar.part"), final_tar)
        csv_part = self.shard_dir / f"{self.name}.csv.part"
        with open(csv_part, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(csv_part, self.shard_dir / f"{self.name}.csv")  # marks the shard complete
        local_tar.unlink()
        logger.info("published %s (%d images)", self.name, len(self.rows))
        self.tar, self.rows = None, []
        self.index += 1


def log_corrupt(out_dir: Path, source_path: str, label: int, split: str, error: Exception) -> None:
    """Append an unreadable image to ``corrupt.csv`` immediately (so resume skips it too)."""
    path = out_dir / "corrupt.csv"
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CORRUPT_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({"source_path": source_path, "label": label, "split": split, "error": repr(error)[:300]})
    logger.warning("corrupt image skipped: %s (%r)", source_path, error)


def merge_manifest(out_dir: Path) -> int:
    """Concatenate all finished shard CSVs into ``manifest.csv``. Returns the row count."""
    rows = []
    for csv_path in sorted((out_dir / "shards").glob("shard_*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    tmp = out_dir / "manifest.csv.part"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, out_dir / "manifest.csv")
    return len(rows)


# --------------------------------------------------------------------------- #
# Main pass
# --------------------------------------------------------------------------- #
def convert(data: bytes, size: int) -> tuple[bytes, int, int]:
    """Decode one archive image and return ``(png bytes, original width, original height)``."""
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        small = to_working_copy(image, size)
    buffer = io.BytesIO()
    small.save(buffer, format="PNG")
    return buffer.getvalue(), width, height


def stream_pass(source: str, remaining: dict[str, tuple[int, str]], writer: ShardWriter,
                out_dir: Path, size: int, limit: int | None) -> None:
    """Read the archive once from the start, converting every member still in ``remaining``.

    Converted (or corrupt) paths are removed from ``remaining`` as they are handled,
    so a retried pass only does the work that is still missing.
    """
    raw, close = open_archive(source)
    reader = CountingReader(raw)
    handled, start, last_report = 0, time.time(), time.time()
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for member in tar:
                if time.time() - last_report > 60:
                    last_report = time.time()
                    logger.info(
                        "read %.1f / %.1f GB of archive (%.0f%%), %d converted this pass, %d still wanted",
                        reader.bytes_read / 1e9, ARCHIVE_BYTES / 1e9,
                        100 * reader.bytes_read / ARCHIVE_BYTES, handled, len(remaining),
                    )
                if not member.isfile():
                    continue
                rel = archive_member_to_relpath(member.name)
                if rel is None or rel not in remaining:
                    continue  # tarfile skips the member's bytes when we move on
                label, split = remaining[rel]
                data = tar.extractfile(member).read()
                try:
                    png, width, height = convert(data, size)
                except Exception as error:  # PIL raises many types for broken files
                    log_corrupt(out_dir, rel, label, split, error)
                else:
                    writer.add(png_relpath(rel), png, {
                        "path": png_relpath(rel), "source_path": rel, "label": label,
                        "class_name": CLASSES[label], "split": split,
                        "orig_width": width, "orig_height": height, "png_bytes": len(png),
                    })
                del remaining[rel]
                handled += 1
                if not remaining or (limit is not None and handled >= limit):
                    break  # everything found: no need to read the rest of the archive
    finally:
        close()
    logger.info("pass finished: %d images in %.1f min", handled, (time.time() - start) / 60)


def build(source: str, splits_dir: Path, out_dir: Path, list_names: list[str], size: int,
          shard_size: int, tmp_dir: Path, max_retries: int, limit: int | None = None) -> dict:
    """Run (or resume) the full build. Returns the build summary that is also saved as JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session_start = time.time()
    wanted = load_wanted(splits_dir, list_names)
    done, next_index = read_done(out_dir)
    remaining = {path: info for path, info in wanted.items() if path not in done}
    logger.info("%d images wanted, %d already done, %d to go", len(wanted), len(wanted) - len(remaining), len(remaining))

    writer = ShardWriter(out_dir, tmp_dir, shard_size, next_index)
    for attempt in range(max_retries + 1):
        if not remaining:
            break
        try:
            stream_pass(source, remaining, writer, out_dir, size, limit)
            writer.flush()
            break
        except Exception:  # network drops surface as many exception types
            writer.flush()  # keep what this pass already converted
            if attempt == max_retries:
                raise
            wait = 30 * (attempt + 1)
            logger.exception("stream failed (attempt %d/%d); retrying in %d s", attempt + 1, max_retries + 1, wait)
            time.sleep(wait)
    writer.flush()

    n_manifest = merge_manifest(out_dir)
    corrupt_count = 0
    if (out_dir / "corrupt.csv").exists():
        with open(out_dir / "corrupt.csv", newline="", encoding="utf-8") as f:
            corrupt_count = sum(1 for _ in csv.DictReader(f))
    if remaining and limit is None:
        with open(out_dir / "missing.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f, lineterminator="\n").writerows([["source_path"], *[[p] for p in sorted(remaining)]])
        logger.warning("%d wanted images were not found in the archive (see missing.csv)", len(remaining))

    shard_bytes = sum(p.stat().st_size for p in (out_dir / "shards").glob("shard_*.tar"))
    log_path = out_dir / "build_log.json"
    history = []
    if log_path.exists():
        history = json.loads(log_path.read_text(encoding="utf-8")).get("sessions", [])
    history.append({"started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session_start)),
                    "minutes": round((time.time() - session_start) / 60, 1)})
    summary = {
        "source": source,
        "lists": list_names,
        "image_size": size,
        "wanted": len(wanted),
        "in_manifest": n_manifest,
        "corrupt": corrupt_count,
        "missing": len(remaining) if limit is None else None,
        "working_copy_gb": round(shard_bytes / 1e9, 3),
        "total_minutes": round(sum(s["minutes"] for s in history), 1),
        "sessions": history,
    }
    write_json(summary, log_path)
    logger.info("manifest: %d images, %d corrupt, working copy %.2f GB", n_manifest, corrupt_count, shard_bytes / 1e9)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, help="output folder (on Colab: a Google Drive folder)")
    parser.add_argument("--lists", nargs="+", default=["main_train", "main_val", "test"],
                        help="split lists to extract; add large_train for the optional Large tier")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--source", default=ARCHIVE_URL, help="archive URL or local .tar.gz path")
    parser.add_argument("--image-size", type=int, default=WORKING_COPY_SIZE)
    parser.add_argument("--shard-size", type=int, default=4000)
    parser.add_argument("--tmp-dir", default=os.path.join(tempfile.gettempdir(), "doccls_shards"),
                        help="LOCAL scratch folder for the shard being written (not Drive)")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="stop after N images (smoke tests only)")
    args = parser.parse_args(argv)

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "build.log")
    build(args.source, resolve_path(args.splits_dir), out_dir, args.lists, args.image_size,
          args.shard_size, Path(args.tmp_dir), args.max_retries, args.limit)


if __name__ == "__main__":
    main()
