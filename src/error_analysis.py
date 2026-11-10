"""Error analysis of misclassified test pages (PRD 6, Checklist Step 6).

Two steps, because sorting errors is a human judgement:

1. ``sample``: pick ~50 misclassified pages (fixed seed), write contact sheets
   to look at and a CSV with empty ``category`` and ``notes`` columns:
       python -m src.error_analysis sample --eval-dir <output_dir>/eval/test/resnet50_main

2. The team opens the sheets, fills in ``category`` for every row with one of
       ambiguous     a person could reasonably pick either class
       label_noise   the dataset label is wrong; the prediction is right or better
       model_error   the label is right and the model is wrong
   and saves the CSV (e.g. to ``results/error_analysis/resnet50_main.csv``).

3. ``summarize``: count the categories (overall and per confusion pair):
       python -m src.error_analysis summarize --csv results/error_analysis/resnet50_main.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import resolve_path  # noqa: E402
from src.data.labels import png_relpath  # noqa: E402
from src.evaluate import INK, pick  # noqa: E402
from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

CATEGORIES = ("ambiguous", "label_noise", "model_error")
PER_SHEET = 10


def sample_errors(eval_dir: Path, n: int = 50, seed: int = 42, data_dir: str | None = None,
                  out_dir: Path | None = None) -> Path:
    """Write ``errors_to_label.csv`` and contact sheets for ``n`` random misclassified pages."""
    predictions = pd.read_csv(eval_dir / "predictions.csv")
    errors = predictions[predictions.label != predictions.pred]
    chosen = errors.sample(n=min(n, len(errors)), random_state=seed).reset_index(drop=True)
    chosen.insert(0, "id", range(1, len(chosen) + 1))
    chosen["category"] = ""
    chosen["notes"] = ""
    out_dir = out_dir or eval_dir / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["id", "path", "true_class", "pred_class", "confidence", "prob_true", "category", "notes"]
    chosen[columns].to_csv(out_dir / "errors_to_label.csv", index=False)

    data_path = Path(pick(data_dir, os.environ.get("DOCCLS_DATA_DIR"), resolve_path("data/working_copy")))
    for sheet, start in enumerate(range(0, len(chosen), PER_SHEET), 1):
        part = chosen.iloc[start:start + PER_SHEET]
        fig, axes = plt.subplots(2, 5, figsize=(17, 9.4))
        for ax in axes.flat:
            ax.axis("off")
        for ax, row in zip(axes.flat, part.itertuples()):
            with Image.open(data_path / png_relpath(row.path)) as image:
                ax.imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"#{row.id}  label: {row.true_class}\npredicted: {row.pred_class} ({row.confidence:.2f})",
                         fontsize=9, color=INK, loc="left")
        fig.suptitle(f"Misclassified test pages, sheet {sheet} ({eval_dir.name})", fontsize=11, color=INK,
                     x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out_dir / f"errors_sheet_{sheet:02d}.png", dpi=110, facecolor="white")
        plt.close(fig)
    logger.info("%d of %d errors sampled -> %s (fill in the 'category' column)", len(chosen), len(errors), out_dir)
    return out_dir


def summarize(csv_path: Path) -> dict:
    """Count the categories in a filled-in error CSV; also writes ``<csv stem>_summary.json``."""
    table = pd.read_csv(csv_path, keep_default_na=False)
    table["category"] = table.category.str.strip().str.lower()
    bad = table[~table.category.isin(CATEGORIES)]
    if len(bad):
        raise ValueError(f"rows {bad.id.tolist()} have no valid category; use one of {CATEGORIES}")
    counts = table.category.value_counts().reindex(CATEGORIES, fill_value=0)
    pairs = (table.groupby(["true_class", "pred_class", "category"]).size()
             .unstack(fill_value=0).reindex(columns=list(CATEGORIES), fill_value=0))
    pairs["total"] = pairs.sum(axis=1)
    pairs = pairs.sort_values("total", ascending=False)
    summary = {
        "n": int(len(table)),
        "counts": {k: int(v) for k, v in counts.items()},
        "percent": {k: round(100 * v / len(table), 1) for k, v in counts.items()},
        "by_pair": [{"true": t, "predicted": p, **{k: int(v) for k, v in row.items()}}
                    for (t, p), row in pairs.iterrows()],
    }
    out = csv_path.with_name(csv_path.stem + "_summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{summary['n']} errors: " + ", ".join(f"{k} {summary['counts'][k]} ({summary['percent'][k]}%)" for k in CATEGORIES))
    print(pairs.head(10).to_string())
    return summary


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point with ``sample`` and ``summarize`` sub-commands."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_sample = sub.add_parser("sample", help="pick misclassified pages to label")
    p_sample.add_argument("--eval-dir", required=True, help="folder with predictions.csv")
    p_sample.add_argument("--n", type=int, default=50)
    p_sample.add_argument("--seed", type=int, default=42)
    p_sample.add_argument("--data-dir")
    p_sample.add_argument("--out-dir")
    p_sum = sub.add_parser("summarize", help="count categories in a filled-in CSV")
    p_sum.add_argument("--csv", required=True)
    args = parser.parse_args(argv)
    setup_logging()
    if args.command == "sample":
        sample_errors(Path(args.eval_dir), args.n, args.seed, args.data_dir,
                      Path(args.out_dir) if args.out_dir else None)
    else:
        summarize(Path(args.csv))


if __name__ == "__main__":
    main()
