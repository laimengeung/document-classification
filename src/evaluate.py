"""Test-set evaluation of a trained model or the DiT reference (PRD 6, Checklist Steps 5-6).

Usage:
    python -m src.evaluate --run-dir <output_dir>/runs/resnet50_main            # our models (M1-M3)
    python -m src.evaluate --dit                                                # DiT reference (Ref)
    python -m src.evaluate --run-dir ... --split val                            # development only

**The test set is used once per final model** (PRD 6). If test results already
exist for a model, the command prints them and exits; ``--force`` overrides
this and the overwrite is recorded in ``metrics.json``.

Writes to ``<output_dir>/eval/<split>/<name>/``:
    metrics.json            accuracy, macro-F1, invoice P/R/F1, top confusions,
                            parameters, model size, throughput, environment
    per_class.csv           precision / recall / F1 / support per class
    confusion_matrix.csv    16 x 16 counts (rows = true class)
    confusion_matrix.png    row-normalised heatmap
    predictions.csv         one row per image: label, prediction, confidence
    probs.npy               full softmax outputs (float16), for later analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402
from torch import nn  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from src.config import resolve_path  # noqa: E402
from src.data.dataset import DocumentImageDataset, load_manifest_paths, make_loader  # noqa: E402
from src.data.labels import CLASSES, INVOICE_IDX, NUM_CLASSES, read_split_csv  # noqa: E402
from src.data.transforms import RepeatChannels, build_transforms  # noqa: E402
from src.models.factory import build_model, count_parameters  # noqa: E402
from src.train import classification_metrics  # noqa: E402
from src.utils import environment_info, setup_logging, write_json  # noqa: E402

logger = logging.getLogger(__name__)

ROLES = {"custom_cnn": "M1", "resnet50": "M2", "efficientnet_b0": "M3", "dit": "Ref"}
DIT_MODEL_ID = "microsoft/dit-base-finetuned-rvlcdip"

# Sequential blue ramp from the project's chart palette (light -> dark); 0 sits on the page colour.
SEQUENTIAL_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"


# --------------------------------------------------------------------------- #
# Loading models
# --------------------------------------------------------------------------- #
def load_trained_model(run_dir: Path) -> tuple[nn.Module, dict, dict]:
    """Load ``best.pt`` from a training run.

    Returns ``(model on CPU in eval mode, training config, checkpoint info)``.
    Pretrained weights are not downloaded again: the checkpoint has everything.
    """
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    cfg["data"]["normalize"] = ckpt["normalize"]
    model = build_model({**cfg["model"], "pretrained": False})
    model.load_state_dict(ckpt["model"])
    info = {"best_epoch": ckpt["epoch"], "val_metrics": ckpt.get("val_metrics", {}),
            "checkpoint": str(run_dir / "best.pt"),
            "checkpoint_mb": round((run_dir / "best.pt").stat().st_size / 1e6, 1)}
    return model.eval(), cfg, info


def dit_class_permutation(id2label: dict) -> list[int]:
    """For each of our class indices, the matching DiT output index (matched by name).

    The DiT checkpoint has its own label order; matching names (case and
    underscores ignored) makes the comparison independent of that order.
    """
    normalise = lambda name: str(name).lower().replace("_", " ").strip()  # noqa: E731
    by_name = {normalise(label): int(index) for index, label in id2label.items()}
    missing = [c for c in CLASSES if normalise(c) not in by_name]
    if missing or len(by_name) != NUM_CLASSES:
        raise ValueError(f"DiT labels do not match RVL-CDIP classes; missing {missing}, got {sorted(by_name)}")
    return [by_name[normalise(c)] for c in CLASSES]


class ReorderedLogits(nn.Module):
    """Wrap a Hugging Face classifier so it takes pixel tensors and returns logits in our class order."""

    def __init__(self, hf_model: nn.Module, permutation: list[int]):
        super().__init__()
        self.hf_model = hf_model
        self.register_buffer("permutation", torch.tensor(permutation), persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.hf_model(pixel_values=pixel_values).logits[:, self.permutation]


def load_dit() -> tuple[nn.Module, T.Compose, dict]:
    """Load the DiT reference with **its own** preprocessing (PRD Step 5).

    The image processor's size, interpolation and normalisation are converted
    into an equivalent torchvision pipeline, so the same fast DataLoader path is
    used for every model. Input is our 256 x 256 working copy, not the original
    TIFF; this is noted in the report as a small deviation from the DiT paper.
    """
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(DIT_MODEL_ID)
    hf_model = AutoModelForImageClassification.from_pretrained(DIT_MODEL_ID)
    size = processor.size  # DiT's config stores a bare 224; newer transformers turn it into a dict
    if isinstance(size, int):
        height = width = size
    else:
        height, width = (size["height"], size["width"]) if "height" in size else (size["shortest_edge"],) * 2
    resample = {0: T.InterpolationMode.NEAREST, 2: T.InterpolationMode.BILINEAR,
                3: T.InterpolationMode.BICUBIC}.get(int(getattr(processor, "resample", 2)), T.InterpolationMode.BILINEAR)
    steps = [T.Resize((height, width), interpolation=resample, antialias=True)]
    if getattr(processor, "do_center_crop", False):
        crop = processor.crop_size
        steps.append(T.CenterCrop((crop["height"], crop["width"])))
        height, width = crop["height"], crop["width"]
    steps += [T.ToTensor(), RepeatChannels(3), T.Normalize(processor.image_mean, processor.image_std)]
    model = ReorderedLogits(hf_model, dit_class_permutation(hf_model.config.id2label)).eval()
    info = {"model_id": DIT_MODEL_ID, "input": [3, height, width], "preprocessing": repr(T.Compose(steps))}
    return model, T.Compose(steps), info


# --------------------------------------------------------------------------- #
# Inference and speed
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device, use_amp: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """Softmax probabilities and labels for every image, plus wall-clock seconds (incl. data loading)."""
    model.eval()
    all_probs, all_labels = [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(all_probs), np.concatenate(all_labels), time.time() - start


@torch.no_grad()
def measure_throughput(model: nn.Module, input_shape: tuple[int, int, int], device: torch.device,
                       use_amp: bool, batch_size: int = 64, warmup: int = 10, iters: int = 30) -> float:
    """Model-only batch inference speed in pages per second (PRD 6: matters for bulk imports).

    Uses a random batch already on the device, so it measures the network, not
    disk or JPEG/PNG decoding. End-to-end speed is reported separately.
    """
    model.eval()
    batch = torch.randn(batch_size, *input_shape, device=device)
    for _ in range(warmup):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return batch_size * iters / (time.time() - start)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def per_class_rows(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """Precision, recall, F1 and support for each of the 16 classes."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0)
    return [{"class": CLASSES[i], "precision": round(float(precision[i]), 4), "recall": round(float(recall[i]), 4),
             "f1": round(float(f1[i]), 4), "support": int(support[i])} for i in range(NUM_CLASSES)]


def top_confusions(cm: np.ndarray, k: int = 10) -> list[dict]:
    """The ``k`` largest off-diagonal cells: which true class is most often predicted as which."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1], off.shape))[0][:k]
    row_totals = cm.sum(axis=1)
    return [{"true": CLASSES[t], "predicted": CLASSES[p], "count": int(cm[t, p]),
             "pct_of_true_class": round(100 * cm[t, p] / max(row_totals[t], 1), 2)}
            for t, p in order if cm[t, p] > 0]


def plot_confusion_matrix(cm: np.ndarray, path: Path, title: str) -> None:
    """Row-normalised confusion matrix (% of each true class), sequential blue, values in cells."""
    pct = 100 * cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
    for t in range(NUM_CLASSES):
        for p in range(NUM_CLASSES):
            if pct[t, p] >= 0.5:  # skip near-empty cells to keep the chart readable
                ax.text(p, t, f"{pct[t, p]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if pct[t, p] > 55 else INK)
    ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=45, ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(NUM_CLASSES), CLASSES, fontsize=8, color=INK)
    ax.set_xlabel("Predicted class", color=INK_SECONDARY)
    ax.set_ylabel("True class", color=INK_SECONDARY)
    ax.set_title(f"{title}\n% of each true class (rows sum to 100)", fontsize=10, color=INK, loc="left")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def save_results(out_dir: Path, metrics: dict, rows: list[tuple[str, int]],
                 probs: np.ndarray, labels: np.ndarray) -> dict:
    """Compute every PRD 6 metric from the probabilities and write all output files."""
    preds = probs.argmax(1)
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    base = classification_metrics(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=[INVOICE_IDX], zero_division=0)
    metrics.update({
        "accuracy": round(base["acc"], 5),
        "macro_f1": round(base["macro_f1"], 5),
        "invoice": {"precision": round(float(precision[0]), 5), "recall": round(float(recall[0]), 5),
                    "f1": round(float(f1[0]), 5), "support": int(support[0])},
        "top_confusions": top_confusions(cm),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_class_rows(labels, preds))
    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["true \\ predicted", *CLASSES])
        writer.writerows([CLASSES[i], *cm[i].tolist()] for i in range(NUM_CLASSES))
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["path", "label", "pred", "true_class", "pred_class", "confidence", "prob_true"])
        for (path, _), label, pred, p in zip(rows, labels, preds, probs):
            writer.writerow([path, int(label), int(pred), CLASSES[label], CLASSES[pred],
                             round(float(p[pred]), 4), round(float(p[label]), 4)])
    np.save(out_dir / "probs.npy", probs.astype(np.float16))
    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png",
                          f"{metrics['name']} ({metrics['role']}) - {metrics['split']} set, "
                          f"accuracy {100 * metrics['accuracy']:.1f}%, macro-F1 {metrics['macro_f1']:.3f}")
    write_json(metrics, out_dir / "metrics.json")
    return metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def default_output_dir() -> Path:
    """``DOCCLS_OUTPUT_DIR`` if set, else the repo's ``outputs/``."""
    return resolve_path(os.environ.get("DOCCLS_OUTPUT_DIR", "outputs"))


def pick(*candidates):
    """First candidate that is not None/empty."""
    return next((c for c in candidates if c), None)


def evaluate(run_dir: Path | None = None, dit: bool = False, split: str = "test", force: bool = False,
             data_dir: str | None = None, splits_dir: str | None = None, out_root: str | None = None,
             batch_size: int = 128, num_workers: int = 2, device: str | None = None,
             throughput_iters: int = 30) -> dict:
    """Evaluate one model on ``split`` and save every result. Returns the metrics dict."""
    if (run_dir is None) == (not dit):
        raise ValueError("give exactly one of run_dir or dit=True")
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = torch_device.type == "cuda"

    if dit:
        name, model_name, train_cfg = "dit_base", "dit", {}
        out_root_path = Path(out_root) if out_root else default_output_dir() / "eval"
    else:
        run_dir = Path(run_dir)
        name = run_dir.name
        out_root_path = Path(out_root) if out_root else run_dir.parent.parent / "eval"
    out_dir = out_root_path / split / name

    if split == "test" and (out_dir / "metrics.json").exists() and not force:
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        logger.info("%s was already evaluated on the test set (%s): accuracy %.4f, macro-F1 %.4f. "
                    "The test set is used once per model; pass --force to re-run.",
                    name, metrics.get("evaluated_at"), metrics["accuracy"], metrics["macro_f1"])
        metrics["skipped"] = True
        return metrics

    if dit:
        model, transform, model_info = load_dit()
        input_shape = tuple(model_info["input"])
        n_params = count_parameters(model.hf_model)
        meta = {"model": "dit", "tier": "full RVL-CDIP train (pretrained by Microsoft)", **model_info,
                "checkpoint_mb": None}
    else:
        model, train_cfg, ckpt_info = load_trained_model(run_dir)
        model_name = train_cfg["model"]["name"]
        mean, std = train_cfg["data"]["normalize"]["mean"], train_cfg["data"]["normalize"]["std"]
        transform = build_transforms(train_cfg, mean, std, train=False)
        size = train_cfg["data"]["image_size"]
        input_shape = (train_cfg["model"]["in_chans"], size, size)
        n_params = count_parameters(model)
        meta = {"model": model_name, "tier": train_cfg["data"]["tier"], "run_dir": str(run_dir), **ckpt_info}

    paths_cfg = train_cfg.get("paths", {})
    data_path = Path(pick(data_dir, os.environ.get("DOCCLS_DATA_DIR"), paths_cfg.get("data_dir"), resolve_path("data/working_copy")))
    splits_path = Path(pick(splits_dir, os.environ.get("DOCCLS_SPLITS_DIR"), paths_cfg.get("splits_dir"), resolve_path("splits")))
    list_name = "test" if split == "test" else "main_val"
    rows_all = read_split_csv(splits_path / f"{list_name}.csv")
    dataset = DocumentImageDataset.from_split(splits_path / f"{list_name}.csv", data_path, transform,
                                              available=load_manifest_paths(data_path))
    logger.info("%s on %s: %d images (%d missing from the working copy), device %s",
                name, split, len(dataset), len(rows_all) - len(dataset), torch_device)

    model = model.to(torch_device)
    loader = make_loader(dataset, batch_size, shuffle=False, num_workers=num_workers, seed=0)
    probs, labels, seconds = predict(model, loader, torch_device, use_amp)
    model_speed = measure_throughput(model, input_shape, torch_device, use_amp, iters=throughput_iters)

    metrics = {
        "name": name, "role": ROLES[model_name], "split": split, "split_list": list_name,
        **meta,
        "n_images": int(len(labels)), "n_missing": len(rows_all) - len(dataset),
        "parameters": n_params, "weights_mb_fp32": round(n_params * 4 / 1e6, 1),
        "throughput_model_pages_per_s": round(model_speed, 1),
        "throughput_end_to_end_pages_per_s": round(len(labels) / seconds, 1),
        "throughput_batch_size": 64, "amp": use_amp, "device": str(torch_device),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "forced_rerun": bool(force),
        "environment": environment_info(),
    }
    metrics = save_results(out_dir, metrics, dataset.rows, probs, labels)
    logger.info("%s: accuracy %.4f, macro-F1 %.4f, invoice F1 %.4f, %.0f pages/s (model only) -> %s",
                name, metrics["accuracy"], metrics["macro_f1"], metrics["invoice"]["f1"], model_speed, out_dir)
    return metrics


def main(argv: list[str] | None = None) -> dict:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-dir", help="training run folder containing best.pt")
    target.add_argument("--dit", action="store_true", help=f"evaluate the {DIT_MODEL_ID} reference")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--force", action="store_true", help="re-run even if test results exist")
    parser.add_argument("--data-dir", help="working copy (default: $DOCCLS_DATA_DIR or the run's config)")
    parser.add_argument("--splits-dir", help="split lists (default: $DOCCLS_SPLITS_DIR or splits/)")
    parser.add_argument("--out-root", help="default: <output_dir>/eval")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device")
    parser.add_argument("--throughput-iters", type=int, default=30)
    args = parser.parse_args(argv)
    setup_logging()
    return evaluate(Path(args.run_dir) if args.run_dir else None, args.dit, args.split, args.force,
                    args.data_dir, args.splits_dir, args.out_root, args.batch_size, args.num_workers,
                    args.device, args.throughput_iters)


if __name__ == "__main__":
    main()
