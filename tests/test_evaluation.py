"""Smoke tests for evaluation, Grad-CAM, error analysis, comparison and the duplicate check (CPU)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from conftest import smoke_config, train_argv
from src import train
from src.compare import compare
from src.data.labels import CLASSES, NUM_CLASSES
from src.duplicates import dhash, find_duplicates, pair_sheet
from src.error_analysis import sample_errors, summarize
from src.evaluate import dit_class_permutation, evaluate
from src.gradcam import run_gradcam

MODELS = ("custom_cnn", "efficientnet_b0")


@pytest.fixture(scope="module")
def trained(working_copy, tmp_path_factory):
    """Two tiny trained runs (no pretrained downloads), evaluated once on the fake test set."""
    output_dir = tmp_path_factory.mktemp("outputs")
    run_dirs, metrics = {}, {}
    for model in MODELS:
        config_path, overrides = smoke_config(working_copy, output_dir, model,
                                              extra=["train.epochs=2", "model.pretrained=false"])
        train.main(train_argv(config_path, overrides))
        run_dirs[model] = output_dir / "runs" / f"{model}_main"
        metrics[model] = evaluate(run_dirs[model], num_workers=0, device="cpu", throughput_iters=2)
    return {"output_dir": output_dir, "run_dirs": run_dirs, "metrics": metrics, "wc": working_copy}


def test_evaluate_writes_all_outputs(trained):
    m = trained["metrics"]["custom_cnn"]
    out = trained["output_dir"] / "eval" / "test" / "custom_cnn_main"
    assert m["n_images"] == NUM_CLASSES and m["n_missing"] == 1  # the corrupt test image is skipped
    assert 0 <= m["accuracy"] <= 1 and 0 <= m["macro_f1"] <= 1 and m["role"] == "M1"
    assert set(m["invoice"]) == {"precision", "recall", "f1", "support"} and m["invoice"]["support"] == 1
    assert m["throughput_model_pages_per_s"] > 0 and m["parameters"] > 0
    for name in ("metrics.json", "per_class.csv", "confusion_matrix.csv", "confusion_matrix.png",
                 "predictions.csv", "probs.npy"):
        assert (out / name).exists(), name
    cm = pd.read_csv(out / "confusion_matrix.csv", index_col=0)
    assert cm.shape == (NUM_CLASSES, NUM_CLASSES) and cm.values.sum() == NUM_CLASSES
    predictions = pd.read_csv(out / "predictions.csv")
    assert len(predictions) == NUM_CLASSES and predictions.path.str.endswith(".tif").all()
    probs = np.load(out / "probs.npy")
    assert probs.shape == (NUM_CLASSES, NUM_CLASSES) and np.allclose(probs.sum(1), 1, atol=1e-2)
    assert (predictions.pred.values == probs.argmax(1)).all()


def test_test_set_is_used_once(trained):
    run_dir = trained["run_dirs"]["custom_cnn"]
    again = evaluate(run_dir, num_workers=0, device="cpu", throughput_iters=2)
    assert again.get("skipped")
    forced = evaluate(run_dir, force=True, num_workers=0, device="cpu", throughput_iters=2)
    assert not forced.get("skipped") and forced["forced_rerun"]
    assert forced["accuracy"] == trained["metrics"]["custom_cnn"]["accuracy"]  # deterministic


def test_validation_split_can_be_rerun(trained):
    m = evaluate(trained["run_dirs"]["custom_cnn"], split="val", num_workers=0, device="cpu", throughput_iters=2)
    assert m["split_list"] == "main_val" and m["n_images"] == 2 * NUM_CLASSES


@pytest.mark.parametrize("model", MODELS)
def test_gradcam_figures(trained, model):
    out = run_gradcam(trained["run_dirs"][model], classes=["invoice", "form"], per_class=2, device="cpu")
    assert (out / "gradcam_invoice.png").exists() and (out / "gradcam_form.png").exists()
    assert (out / "gradcam_overview.png").exists()
    examples = pd.read_csv(out / "examples.csv")
    assert set(examples.true_class) <= {"invoice", "form"} and len(examples) >= 2


def test_error_analysis_sample_and_summarize(trained):
    eval_dir = trained["output_dir"] / "eval" / "test" / "custom_cnn_main"
    predictions = pd.read_csv(eval_dir / "predictions.csv")
    if (predictions.label != predictions.pred).sum() == 0:
        pytest.skip("tiny model made no errors")
    out = sample_errors(eval_dir, n=5, data_dir=str(trained["wc"]["local"]))
    table = pd.read_csv(out / "errors_to_label.csv")
    assert (out / "errors_sheet_01.png").exists() and len(table) <= 5
    table["category"] = ["ambiguous", "label_noise", "model_error", "model_error", "ambiguous"][: len(table)]
    table.to_csv(out / "filled.csv", index=False)
    summary = summarize(out / "filled.csv")
    assert summary["n"] == len(table) and sum(summary["counts"].values()) == len(table)


def test_error_summary_rejects_missing_category(tmp_path):
    path = tmp_path / "errors.csv"
    pd.DataFrame({"id": [1], "true_class": ["form"], "pred_class": ["memo"], "category": [""]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        summarize(path)


def test_duplicates_and_comparison(trained):
    wc = trained["wc"]
    # Identical pages hash identically; different pages do not.
    page = Image.open(next(wc["local"].rglob("*.png")))
    assert (dhash(page) == dhash(page.copy())).all()
    dup_csv = trained["output_dir"] / "eval" / "test" / "duplicates.csv"
    table = find_duplicates(wc["local"], wc["splits"], "main_train", "test", dup_csv)
    assert len(table) == NUM_CLASSES and (table.distance >= 0).all()
    assert pair_sheet(dup_csv, wc["local"], per_bin=1).exists()

    tables = compare(trained["output_dir"] / "eval", dup_threshold=300)  # every page counts as a duplicate
    comparison = tables["comparison"]
    assert list(comparison.role) == ["M1", "M3"]
    assert (comparison["dups removed"] == NUM_CLASSES).all()
    out = trained["output_dir"] / "eval" / "test" / "comparison"
    for name in ("comparison.csv", "comparison.md", "invoice.md", "literature.md", "accuracy_vs_speed.png"):
        assert (out / name).exists(), name
    literature = tables["literature"]
    assert literature.method.str.contains("ours, best").any()
    assert literature.method.str.contains("unverified").any()


def test_dit_label_permutation():
    shuffled = list(range(NUM_CLASSES))
    np.random.default_rng(0).shuffle(shuffled)
    id2label = {dit_index: CLASSES[ours].upper().replace(" ", "_") for ours, dit_index in enumerate(shuffled)}
    permutation = dit_class_permutation(id2label)
    assert permutation == shuffled
    logits = torch.arange(NUM_CLASSES, dtype=torch.float32)[None]  # logit value == DiT index
    assert logits[:, permutation].tolist()[0] == shuffled
    with pytest.raises(ValueError):
        dit_class_permutation({0: "letter", 1: "receipt"})
