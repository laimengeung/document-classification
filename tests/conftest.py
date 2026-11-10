"""Shared fixtures: a tiny synthetic RVL-CDIP look-alike and its working copy."""

from __future__ import annotations

import io
import tarfile

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.config import REPO_ROOT
from src.data.build_working_copy import build
from src.data.labels import NUM_CLASSES, write_split_csv
from src.data.local_copy import unpack_working_copy

PER_CLASS = {"main_train": 4, "main_val": 2, "test": 1}
CORRUPT = "imagesz/z/z/z/zzz00000/corrupt.tif"


def fake_page(label: int, seed: int) -> Image.Image:
    """A 'document' whose class is encoded by where a dark block sits, plus noise."""
    rng = np.random.default_rng(seed)
    page = Image.fromarray((rng.random((500, 380)) * 40 + 215).astype(np.uint8), "L")
    draw = ImageDraw.Draw(page)
    row, col = divmod(label, 4)
    draw.rectangle([20 + col * 90, 20 + row * 120, 90 + col * 90, 110 + row * 120], fill=0)
    return page.convert("1") if seed % 2 else page  # mix 1-bit "fax" and 8-bit scans


@pytest.fixture(scope="session")
def fake_dataset(tmp_path_factory):
    """Split lists plus a .tar.gz laid out like the real archive (images/<path>)."""
    root = tmp_path_factory.mktemp("rvl")
    splits_dir = root / "splits"
    splits_dir.mkdir()
    archive = root / "rvl-cdip.tar.gz"
    seed = 0
    with tarfile.open(archive, "w:gz") as tar:
        def add(rel: str, data: bytes) -> None:
            info = tarfile.TarInfo("images/" + rel)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        for list_name, per_class in PER_CLASS.items():
            rows = []
            for label in range(NUM_CLASSES):
                for _ in range(per_class):
                    seed += 1
                    rel = f"images{list_name[0]}/{label:02d}/doc{seed:05d}.tif"
                    buffer = io.BytesIO()
                    fake_page(label, seed).save(buffer, format="TIFF")
                    add(rel, buffer.getvalue())
                    rows.append((rel, label))
            if list_name == "test":
                rows.append((CORRUPT, 3))
                add(CORRUPT, b"this is not a tiff")
            write_split_csv(splits_dir / f"{list_name}.csv", rows)
        add("imagesq/unwanted.tif", b"not in any list")  # must be ignored
    return root, splits_dir, archive


@pytest.fixture(scope="session")
def working_copy(fake_dataset, tmp_path_factory):
    """Build the working copy from the fake archive, then unpack it like a Colab session."""
    root, splits_dir, archive = fake_dataset
    drive = root / "drive_working_copy"
    lists = list(PER_CLASS)
    summary = build(str(archive), splits_dir, drive, lists, size=96, shard_size=25,
                    tmp_dir=root / "tmp", max_retries=0)
    local = tmp_path_factory.mktemp("local") / "working_copy"
    unpack_working_copy(drive, local)
    return {"root": root, "splits": splits_dir, "archive": archive, "drive": drive,
            "local": local, "summary": summary, "lists": lists}


def smoke_config(working_copy, output_dir, model="custom_cnn", extra=()):
    """Config path and overrides for a tiny run on the fake working copy."""
    overrides = [
        f"paths.splits_dir={working_copy['splits'].as_posix()}",
        f"paths.data_dir={working_copy['local'].as_posix()}",
        f"paths.output_dir={output_dir.as_posix()}",
        "data.image_size=64", "data.num_workers=0", "train.batch_size=16", "train.log_every=0",
        *extra,
    ]
    return REPO_ROOT / "configs" / f"{model}.yaml", overrides


def train_argv(config_path, overrides, *flags):
    """Command-line arguments for ``src.train.main``."""
    return ["--config", str(config_path), "--tier", "main", "--device", "cpu",
            *sum((["--set", o] for o in overrides), []), *flags]
