"""Grad-CAM heatmaps: what does each model look at? (PRD 6, goal G5, anti-goal "shortcuts").

Usage (after ``evaluate.py`` has written the model's predictions.csv):
    python -m src.gradcam --run-dir <output_dir>/runs/resnet50_main

For each chosen class (invoice always included) it picks, with a fixed seed,
some test pages the model got **right** and some it got **wrong**, computes
Grad-CAM for the *predicted* class, and saves:

    <eval_dir>/gradcam/gradcam_<class>.png    correct (top row) vs misclassified (bottom row)
    <eval_dir>/gradcam/gradcam_overview.png   one correct + one wrong per class, side by side
    <eval_dir>/gradcam/examples.csv           which pages were used

Reading the figures: a good model lights up layout (tables, headers, totals,
letterheads, form fields). Heat on page borders, scan noise or blank margins
suggests shortcut learning and must be discussed in the report.

Target layers (the last convolutional feature map of each network):
    custom_cnn       features[-1][5]  last ReLU of block 5, 14 x 14
    resnet50         layer4[-1]       7 x 7
    efficientnet_b0  bn2              final conv + BN + SiLU, 7 x 7
The coarse 7 x 7 maps of M2/M3 can only point at page regions, not single words.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import resolve_path  # noqa: E402
from src.data.labels import CLASSES, png_relpath  # noqa: E402
from src.data.transforms import build_transforms  # noqa: E402
from src.evaluate import INK, INK_SECONDARY, load_trained_model, pick  # noqa: E402
from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CLASSES = ["invoice", "form", "questionnaire", "letter", "memo",
                   "scientific report", "budget", "news article"]
HEAT_RGB = np.array([0xEB, 0x68, 0x34]) / 255.0  # single-hue overlay (palette orange), not a rainbow map
MAX_HEAT_ALPHA = 0.65


def target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """The feature map Grad-CAM explains, per architecture (see module docstring)."""
    if model_name == "custom_cnn":
        return model.features[-1][5]
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "efficientnet_b0":
        return model.bn2
    raise ValueError(f"No Grad-CAM target layer defined for {model_name!r}")


def select_examples(predictions: pd.DataFrame, classes: list[str], per_class: int, seed: int) -> pd.DataFrame:
    """Up to ``per_class`` correct and ``per_class`` misclassified pages whose *true* class is in ``classes``."""
    picked = []
    for class_name in classes:
        of_class = predictions[predictions.true_class == class_name]
        for kind, subset in (("correct", of_class[of_class.label == of_class.pred]),
                             ("misclassified", of_class[of_class.label != of_class.pred])):
            chosen = subset.sample(n=min(per_class, len(subset)), random_state=seed)
            picked.append(chosen.assign(kind=kind))
    return pd.concat(picked, ignore_index=True) if picked else predictions.iloc[0:0]


def compute_cams(model, layer, images: torch.Tensor, targets: list[int], device, batch_size: int = 16) -> np.ndarray:
    """Grad-CAM maps in [0, 1] at input resolution, one per image, for the given target classes."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model = model.to(device).eval()
    maps = []
    with GradCAM(model=model, target_layers=[layer]) as cam:
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size].to(device)
            batch_targets = [ClassifierOutputTarget(t) for t in targets[start:start + batch_size]]
            maps.append(cam(input_tensor=batch, targets=batch_targets))
    return np.concatenate(maps)


def overlay(page: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Blend a single-hue heat layer over a grayscale page: more opaque orange = more influence."""
    base = np.repeat((page.astype(np.float32) / 255.0)[..., None], 3, axis=2)
    alpha = (cam * MAX_HEAT_ALPHA)[..., None]
    return base * (1 - alpha) + HEAT_RGB * alpha


def _panel(ax, image: np.ndarray, row: pd.Series) -> None:
    """One example: overlay image plus a 'true -> predicted (confidence)' title."""
    ax.imshow(image)
    ax.set_title(f"true: {row.true_class}\npred: {row.pred_class} ({row.confidence:.2f})", fontsize=8,
                 color=INK if row.kind == "correct" else "#b03030", loc="left")
    ax.axis("off")


def plot_class(examples: pd.DataFrame, images: dict, class_name: str, model_label: str, path: Path) -> None:
    """Per-class figure: correct predictions on the top row, misclassified on the bottom row."""
    rows = [("correct", examples[(examples.true_class == class_name) & (examples.kind == "correct")]),
            ("misclassified", examples[(examples.true_class == class_name) & (examples.kind == "misclassified")])]
    cols = max(3, max(len(r) for _, r in rows))  # at least 3 columns so the title fits
    fig, axes = plt.subplots(2, cols, figsize=(2.6 * cols, 6.6), squeeze=False)
    for r, (kind, subset) in enumerate(rows):
        for c in range(cols):
            ax = axes[r, c]
            if c < len(subset):
                row = subset.iloc[c]
                _panel(ax, images[row.path], row)
            else:
                ax.axis("off")
                if c == 0:
                    ax.text(0.5, 0.5, f"no {kind} examples", ha="center", va="center", color=INK_SECONDARY)
    fig.suptitle(f"Grad-CAM, {model_label}: true class '{class_name}'\n"
                 "top: correct, bottom: misclassified; orange = regions that drove the prediction",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_overview(examples: pd.DataFrame, images: dict, classes: list[str], model_label: str, path: Path) -> None:
    """Report figure: for each class, its first correct (top) and first misclassified (bottom) page."""
    fig, axes = plt.subplots(2, len(classes), figsize=(max(2.4 * len(classes), 8.0), 6.4), squeeze=False)
    for c, class_name in enumerate(classes):
        for r, kind in enumerate(("correct", "misclassified")):
            subset = examples[(examples.true_class == class_name) & (examples.kind == kind)]
            ax = axes[r, c]
            if len(subset):
                _panel(ax, images[subset.iloc[0].path], subset.iloc[0])
            else:
                ax.axis("off")
    fig.suptitle(f"Grad-CAM, {model_label}\n"
                 "top: correct, bottom: misclassified; orange = regions that drove the prediction",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def run_gradcam(run_dir: Path, eval_dir: Path | None = None, classes: list[str] | None = None,
                per_class: int = 3, seed: int = 42, target: str = "pred", data_dir: str | None = None,
                device: str | None = None) -> Path:
    """Make all Grad-CAM figures for one trained model. Returns the output folder."""
    classes = classes or DEFAULT_CLASSES
    if "invoice" not in classes:
        classes = ["invoice", *classes]  # invoice is required by the PRD
    unknown = [c for c in classes if c not in CLASSES]
    if unknown:
        raise ValueError(f"unknown classes {unknown}")
    eval_dir = eval_dir or run_dir.parent.parent / "eval" / "test" / run_dir.name
    predictions = pd.read_csv(eval_dir / "predictions.csv")
    model, cfg, _ = load_trained_model(run_dir)
    model_name = cfg["model"]["name"]
    data_path = Path(pick(data_dir, os.environ.get("DOCCLS_DATA_DIR"), cfg["paths"]["data_dir"], resolve_path("data/working_copy")))
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    examples = select_examples(predictions, classes, per_class, seed)
    transform = build_transforms(cfg, cfg["data"]["normalize"]["mean"], cfg["data"]["normalize"]["std"], train=False)
    size = cfg["data"]["image_size"]
    pages, tensors = {}, []
    for path in examples.path:
        with Image.open(data_path / png_relpath(path)) as image:
            gray = image.convert("L")
            tensors.append(transform(gray))
            pages[path] = np.asarray(gray.resize((size, size), Image.Resampling.BILINEAR))
    targets = (examples.pred if target == "pred" else examples.label).astype(int).tolist()
    cams = compute_cams(model, target_layer(model, model_name), torch.stack(tensors), targets, torch_device)
    images = {path: overlay(pages[path], cam) for path, cam in zip(examples.path, cams)}

    out_dir = eval_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_label = f"{run_dir.name}"
    for class_name in classes:
        plot_class(examples, images, class_name, model_label, out_dir / f"gradcam_{class_name.replace(' ', '_')}.png")
    plot_overview(examples, images, classes, model_label, out_dir / "gradcam_overview.png")
    examples.assign(cam_target=target).to_csv(out_dir / "examples.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    logger.info("Grad-CAM for %d pages of %d classes -> %s", len(examples), len(classes), out_dir)
    return out_dir


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="training run folder containing best.pt")
    parser.add_argument("--eval-dir", help="default: <output_dir>/eval/test/<run name>")
    parser.add_argument("--classes", nargs="+", help=f"true classes to show (default: {DEFAULT_CLASSES})")
    parser.add_argument("--per-class", type=int, default=3, help="correct and misclassified examples per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", choices=["pred", "true"], default="pred",
                        help="explain the predicted class (default) or the true class")
    parser.add_argument("--data-dir")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    setup_logging()
    run_gradcam(Path(args.run_dir), Path(args.eval_dir) if args.eval_dir else None, args.classes,
                args.per_class, args.seed, args.target, args.data_dir, args.device)


if __name__ == "__main__":
    main()
