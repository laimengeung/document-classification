"""Comparative study tables and figure from all evaluated models (PRD 6, Checklist Step 6).

Usage:
    python -m src.compare                          # reads <output_dir>/eval/test/*/metrics.json
    python -m src.compare --dup-threshold 12       # also: accuracy without near-duplicates

Writes to ``--out-dir`` (default ``<output_dir>/eval/test/comparison``):
    comparison.csv / comparison.md     M1 vs M2 vs M3 vs DiT: accuracy, macro-F1, size, speed
    invoice.md                         invoice precision / recall / F1 per model (client-relevant, G4)
    literature.md                      our best model next to published RVL-CDIP results
    accuracy_vs_speed.png              the accuracy-vs-throughput trade-off (bulk imports)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from src.config import resolve_path  # noqa: E402
from src.evaluate import INK, INK_SECONDARY, default_output_dir  # noqa: E402
from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

ROLE_ORDER = {"M1": 0, "M2": 1, "M3": 2, "Ref": 3}
POINT_COLOR = "#2a78d6"  # palette slot 1; models are told apart by direct labels, not colour
TIER_TRAIN_IMAGES = {"dev": "3.2k", "main": "32k", "large": "80k"}


def load_metrics(eval_split_dir: Path) -> list[dict]:
    """Every ``metrics.json`` directly under ``eval/<split>/``, sorted M1, M2, M3, Ref."""
    found = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(eval_split_dir.glob("*/metrics.json"))]
    return sorted(found, key=lambda m: (ROLE_ORDER.get(m["role"], 9), m["name"]))


def accuracy_without_duplicates(eval_dir: Path, duplicates: pd.DataFrame, threshold: int) -> tuple[float, int]:
    """Accuracy on test pages whose nearest training page is farther than ``threshold``."""
    predictions = pd.read_csv(eval_dir / "predictions.csv")
    near = set(duplicates.loc[duplicates.distance <= threshold, "path"])
    kept = predictions[~predictions.path.isin(near)]
    return float((kept.label == kept.pred).mean()), int(len(predictions) - len(kept))


def comparison_table(metrics: list[dict], eval_split_dir: Path, dup_threshold: int | None) -> pd.DataFrame:
    """One row per model with every number the comparison needs."""
    duplicates = None
    if dup_threshold is not None:
        dup_csv = eval_split_dir / "duplicates.csv"
        if not dup_csv.exists():
            raise FileNotFoundError(f"{dup_csv} not found; run `python -m src.duplicates find` first")
        duplicates = pd.read_csv(dup_csv)
    rows = []
    for m in metrics:
        row = {
            "role": m["role"],
            "model": m["name"],
            "train images": TIER_TRAIN_IMAGES.get(m.get("tier"), "320k (full, by authors)"),
            "test images": m["n_images"],
            "accuracy %": round(100 * m["accuracy"], 2),
            "macro-F1": round(m["macro_f1"], 4),
            "invoice F1": round(m["invoice"]["f1"], 4),
            "params (M)": round(m["parameters"] / 1e6, 2),
            "weights (MB)": m["weights_mb_fp32"],
            "pages/s (model)": m["throughput_model_pages_per_s"],
            "pages/s (end-to-end)": m["throughput_end_to_end_pages_per_s"],
        }
        if duplicates is not None:
            accuracy, removed = accuracy_without_duplicates(eval_split_dir / m["name"], duplicates, dup_threshold)
            row[f"accuracy % w/o dups (d<={dup_threshold})"] = round(100 * accuracy, 2)
            row["dups removed"] = removed
        rows.append(row)
    return pd.DataFrame(rows)


def invoice_table(metrics: list[dict]) -> pd.DataFrame:
    """Invoice precision / recall / F1 / support per model (PRD G4: its own table)."""
    return pd.DataFrame([{"role": m["role"], "model": m["name"],
                          "precision": round(m["invoice"]["precision"], 4), "recall": round(m["invoice"]["recall"], 4),
                          "F1": round(m["invoice"]["f1"], 4), "support": m["invoice"]["support"]} for m in metrics])


def literature_table(metrics: list[dict], published_path: Path) -> pd.DataFrame:
    """Published results plus our best trained model (by test accuracy) and our DiT re-evaluation."""
    published = yaml.safe_load(published_path.read_text(encoding="utf-8"))
    rows = [{"method": p["method"] + ("" if p.get("verified") else " (unverified)"), "source": p["paper"],
             "input": p["input"], "train images": "320k", "accuracy %": p["accuracy"]} for p in published]
    ours = [m for m in metrics if m["role"] != "Ref"]
    if ours:
        best = max(ours, key=lambda m: m["accuracy"])
        rows.append({"method": f"{best['name']} (ours, best)", "source": "this work", "input": "image",
                     "train images": TIER_TRAIN_IMAGES.get(best.get("tier"), "?"),
                     "accuracy %": round(100 * best["accuracy"], 2)})
    for m in metrics:
        if m["role"] == "Ref":
            rows.append({"method": "DiT-base, re-evaluated on our test copy", "source": "this work",
                         "input": "image", "train images": "320k (by authors)", "accuracy %": round(100 * m["accuracy"], 2)})
    return pd.DataFrame(rows)


def plot_accuracy_vs_speed(table: pd.DataFrame, path: Path) -> None:
    """Scatter of test accuracy against model-only throughput; each point labelled with its model."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(table["pages/s (model)"], table["accuracy %"], s=70, color=POINT_COLOR,
               edgecolors="white", linewidths=2, zorder=3)
    for _, row in table.iterrows():
        ax.annotate(f"{row['role']} {row['model']}", (row["pages/s (model)"], row["accuracy %"]),
                    xytext=(8, 4), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xscale("log")
    plain = FuncFormatter(lambda value, _: f"{value:g}")  # 300, not 3x10^2
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(plain)
    ax.tick_params(axis="x", which="minor", labelsize=8, labelcolor=INK_SECONDARY)
    ax.set_xlabel("Throughput, pages per second (log scale, model only, batch 64)", color=INK_SECONDARY)
    ax.set_ylabel("Test accuracy (%)", color=INK_SECONDARY)
    ax.set_title("Accuracy vs. speed (up and right is better)", loc="left", fontsize=11, color=INK)
    ax.grid(alpha=0.25, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.25, y=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def compare(eval_root: Path, split: str = "test", out_dir: Path | None = None, dup_threshold: int | None = None,
            published: Path | None = None) -> dict[str, pd.DataFrame]:
    """Build and save every comparison artefact. Returns the tables."""
    eval_split_dir = eval_root / split
    metrics = load_metrics(eval_split_dir)
    if not metrics:
        raise FileNotFoundError(f"no metrics.json under {eval_split_dir}; run src.evaluate first")
    out_dir = out_dir or eval_split_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "comparison": comparison_table(metrics, eval_split_dir, dup_threshold),
        "invoice": invoice_table(metrics),
        "literature": literature_table(metrics, published or resolve_path("results/published_results.yaml")),
    }
    tables["comparison"].to_csv(out_dir / "comparison.csv", index=False)
    for name, table in tables.items():
        (out_dir / f"{name}.md").write_text(table.to_markdown(index=False) + "\n", encoding="utf-8")
    plot_accuracy_vs_speed(tables["comparison"], out_dir / "accuracy_vs_speed.png")
    tiers = {m.get("tier") for m in metrics if m["role"] != "Ref"}
    if len(tiers) > 1:
        logger.warning("models were trained on different tiers %s; the PRD compares all on Main", tiers)
    logger.info("comparison of %d models -> %s", len(metrics), out_dir)
    return tables


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-root", help="default: <output_dir>/eval")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--out-dir")
    parser.add_argument("--dup-threshold", type=int, help="hash distance counted as a duplicate (see src.duplicates)")
    parser.add_argument("--published", help="default: results/published_results.yaml")
    args = parser.parse_args(argv)
    setup_logging()
    tables = compare(Path(args.eval_root) if args.eval_root else default_output_dir() / "eval", args.split,
                     Path(args.out_dir) if args.out_dir else None, args.dup_threshold,
                     Path(args.published) if args.published else None)
    for name, table in tables.items():
        print(f"\n## {name}\n")
        print(table.to_markdown(index=False))


if __name__ == "__main__":
    main()
