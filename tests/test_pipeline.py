"""Smoke tests for data + training on a tiny synthetic RVL-CDIP look-alike (CPU).

Run:  python -m pytest -q

They cover the PRD checklist items that can be checked without the real data:
streaming build + resume + corrupt-file handling, unpack + verify, batch
shape/dtype/range, all three models, overfit-one-batch, training with
checkpoint/resume, and PDF -> page images.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import PER_CLASS, smoke_config, train_argv
from src import train
from src.config import REPO_ROOT, load_config
from src.data.build_working_copy import build
from src.data.dataset import build_datasets, make_loader
from src.data.labels import NUM_CLASSES
from src.data.local_copy import verify_working_copy
from src.data.pdf import load_document_pages
from src.models.factory import build_model, count_parameters, get_param_groups


def test_build_skips_corrupt_and_writes_manifest(working_copy):
    expected = sum(n * NUM_CLASSES for n in PER_CLASS.values())
    summary = working_copy["summary"]
    assert summary["in_manifest"] == expected
    assert summary["corrupt"] == 1
    assert summary["missing"] == 0
    with open(working_copy["drive"] / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    assert all(int(r["orig_width"]) == 380 and int(r["orig_height"]) == 500 for r in rows)
    assert len(list((working_copy["drive"] / "shards").glob("shard_*.tar"))) == -(-expected // 25)


def test_build_resume_does_no_extra_work(working_copy):
    w = working_copy
    summary = build(str(w["archive"]), w["splits"], w["drive"], w["lists"], size=96, shard_size=25,
                    tmp_dir=w["root"] / "tmp", max_retries=0)
    assert summary["in_manifest"] == w["summary"]["in_manifest"]
    assert len(summary["sessions"]) == 2


def test_unpacked_copy_verifies(working_copy):
    # Only the deliberately corrupt test image is missing.
    assert verify_working_copy(working_copy["local"], working_copy["splits"], ["main_train", "main_val"])
    assert not verify_working_copy(working_copy["local"], working_copy["splits"], ["test"])
    image = Image.open(next(working_copy["local"].rglob("*.png")))
    assert image.mode == "L" and image.size == (96, 96)


def test_batch_shape_dtype_range(working_copy, tmp_path):
    for model, channels in (("custom_cnn", 1), ("resnet50", 3)):
        config_path, overrides = smoke_config(working_copy, tmp_path, model)
        cfg = load_config(config_path, "main", overrides)
        mean, std = ([0.9], [0.2]) if channels == 1 else (cfg["data"]["normalize"]["mean"], cfg["data"]["normalize"]["std"])
        train_ds, val_ds = build_datasets(cfg, mean, std)
        images, labels = next(iter(make_loader(train_ds, 16, True, 0, seed=0)))
        assert images.shape == (16, channels, 64, 64) and images.dtype == torch.float32
        assert labels.dtype == torch.int64 and labels.min() >= 0 and labels.max() < NUM_CLASSES
        assert torch.isfinite(images).all() and -10 < images.min() < images.max() < 10
        assert len(train_ds) == 4 * NUM_CLASSES and len(val_ds) == 2 * NUM_CLASSES


@pytest.mark.parametrize("name", ["custom_cnn", "resnet50", "efficientnet_b0"])
def test_models_forward_and_param_groups(name):
    in_chans = 1 if name == "custom_cnn" else 3
    model = build_model({"name": name, "num_classes": NUM_CLASSES, "in_chans": in_chans, "pretrained": False})
    out = model(torch.randn(2, in_chans, 224, 224))
    assert out.shape == (2, NUM_CLASSES)
    groups = get_param_groups(model, name, lr=1e-4, head_lr=1e-3, weight_decay=0.01)
    assert sum(p.numel() for g in groups for p in g["params"]) == count_parameters(model)
    assert {g["lr"] for g in groups if g["name"].startswith("head")} == {1e-3}


def test_overfit_one_batch(working_copy, tmp_path):
    config_path, overrides = smoke_config(working_copy, tmp_path)
    result = train.main(train_argv(config_path, overrides, "--overfit-batch", "150"))
    assert result["overfit_batch_accuracy"] >= 0.95


def test_train_checkpoint_and_resume(working_copy, tmp_path):
    config_path, overrides = smoke_config(working_copy, tmp_path, extra=["train.epochs=3", "run_name=smoke"])
    argv = train_argv(config_path, overrides)
    summary = train.main(argv)
    run_dir = tmp_path / "runs" / "smoke"
    assert summary["finished"] and summary["epochs_completed"] == 3
    assert (run_dir / "best.pt").exists() and (run_dir / "last.pt").exists()
    assert len((run_dir / "history.csv").read_text().strip().splitlines()) == 4  # header + 3 epochs
    assert summary["normalize"]["mean"] != "auto"  # dataset statistics were computed

    # Simulate a Colab disconnect right after epoch 1, then rerun the same command.
    ckpt = torch.load(run_dir / "last.pt", weights_only=False)
    ckpt.update(epoch=1, history=ckpt["history"][:1], finished=False)
    ckpt["best_epoch"], ckpt["best_score"] = 1, ckpt["history"][0]["val_macro_f1"]
    torch.save(ckpt, run_dir / "last.pt")
    resumed = train.main(argv)
    assert resumed["epochs_completed"] == 3
    log = (run_dir / "train.log").read_text()
    assert "resumed from" in log and "epoch 2/3" in log
    assert (tmp_path / "runs_index.csv").exists()

    # A finished run is not retrained by accident.
    again = train.main(argv)
    assert again["finished"]


def test_pdf_to_page_images():
    pages = load_document_pages(REPO_ROOT / "tests" / "assets" / "sample.pdf")
    assert len(pages) == 2
    assert all(p.mode == "L" and p.size == (256, 256) for p in pages)
    assert np.asarray(pages[0]).min() < 100  # text and border were rendered
