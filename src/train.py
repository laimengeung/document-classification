"""Shared, config-driven training loop for all three models (PRD 5.4, 7.2, Steps 3-5).

Usage:
    python -m src.train --config configs/custom_cnn.yaml --tier dev
    python -m src.train --config configs/resnet50.yaml --tier main --set train.batch_size=96
    python -m src.train --config configs/custom_cnn.yaml --tier dev --overfit-batch 300

What it does:
    * fixed seeds; AdamW with differential learning rates; linear warm-up then
      cosine decay (per step); cross-entropy with label smoothing; AMP on GPU
    * validation every epoch: accuracy, **macro-F1** and invoice P/R/F1
    * early stopping on validation macro-F1
    * **checkpoint every epoch and resume automatically**: ``last.pt`` holds
      model, optimizer, scheduler, AMP scaler, epoch, best score and history,
      so a Colab disconnect costs at most one epoch. ``best.pt`` holds only the
      best model's weights (lean, PRD 7.2). Re-running the same command resumes.
    * logs: ``config.yaml``, ``train.log``, ``history.csv``, ``summary.json`` in
      the run folder, plus one row per finished run in ``runs_index.csv``

The test set is never touched here; it is used once per final model in
``evaluate.py`` (PRD 6).
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn

from src.config import load_config, save_config, to_serializable
from src.data.dataset import build_datasets, compute_mean_std, make_loader
from src.data.labels import CLASSES, INVOICE_IDX, NUM_CLASSES, tier_split_lists
from src.models.factory import build_model, count_parameters, get_param_groups
from src.utils import append_csv_row, atomic_torch_save, environment_info, set_seed, setup_logging, write_json

logger = logging.getLogger(__name__)

HISTORY_FIELDS = [
    "epoch", "lr_backbone", "lr_head", "train_loss", "train_acc", "val_loss", "val_acc",
    "val_macro_f1", "val_invoice_precision", "val_invoice_recall", "val_invoice_f1",
    "epoch_seconds", "train_images_per_second",
]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy, macro-F1 and invoice precision/recall/F1 (PRD 6, G4)."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[INVOICE_IDX], average=None, zero_division=0)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                                   average="macro", zero_division=0)),
        "invoice_precision": float(precision[0]),
        "invoice_recall": float(recall[0]),
        "invoice_f1": float(f1[0]),
    }


# --------------------------------------------------------------------------- #
# Optimisation helpers
# --------------------------------------------------------------------------- #
def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int):
    """Per-step LR multiplier: linear warm-up to 1, then cosine decay to 0."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def group_lr(optimizer: torch.optim.Optimizer, prefix: str) -> float:
    """Current learning rate of the first param group whose name starts with ``prefix``."""
    for group in optimizer.param_groups:
        if group.get("name", "").startswith(prefix):
            return group["lr"]
    return float("nan")


# --------------------------------------------------------------------------- #
# Epoch loops
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, use_amp,
                    grad_clip, log_every, epoch) -> dict[str, float]:
    """One pass over the training set. Returns loss, accuracy and throughput."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    start = time.time()
    for step, (images, labels) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)  # clip the true gradients, not the scaled ones
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch = labels.size(0)
        total_loss += loss.item() * batch
        correct += (logits.argmax(1) == labels).sum().item()
        seen += batch
        if log_every and step % log_every == 0:
            logger.info("epoch %d  step %d/%d  loss %.4f  acc %.3f  %.0f img/s", epoch, step, len(loader),
                        total_loss / seen, correct / seen, seen / (time.time() - start))
    elapsed = time.time() - start
    return {"loss": total_loss / seen, "acc": correct / seen, "images_per_second": seen / elapsed}


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp) -> dict[str, float]:
    """Loss and :func:`classification_metrics` on a held-out loader (no augmentation)."""
    model.eval()
    total_loss, seen = 0.0, 0
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits.float(), labels)
        total_loss += loss.item() * labels.size(0)
        seen += labels.size(0)
        y_true.append(labels.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
    metrics = classification_metrics(np.concatenate(y_true), np.concatenate(y_pred))
    metrics["loss"] = total_loss / seen
    return metrics


def overfit_one_batch(model, dataset, cfg, device, steps: int) -> float:
    """Sanity check (PRD Step 3): the loop must drive the loss on one fixed batch to ~0.

    Uses the *validation* transforms (no augmentation), no label smoothing and
    no regularisation, so a correct pipeline memorises the batch. Returns the
    final accuracy on that batch.
    """
    batch_size = min(32, len(dataset))
    loader = make_loader(dataset, batch_size, shuffle=True, num_workers=0, seed=cfg["seed"])
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"].get("head_lr") or cfg["train"]["lr"],
                                  weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for step in range(1, steps + 1):
        loss = criterion(model(images), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            logger.info("overfit step %d/%d  loss %.4f", step, steps, loss.item())
    model.eval()
    with torch.no_grad():
        accuracy = (model(images).argmax(1) == labels).float().mean().item()
    verdict = "OK" if accuracy >= 0.95 else "PROBLEM: the model cannot memorise one batch"
    logger.info("overfit check: batch accuracy %.3f -> %s", accuracy, verdict)
    return accuracy


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments (see module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--tier", choices=["dev", "main", "large"], help="override data.tier")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override any config value, e.g. --set train.lr=3e-4 (repeatable)")
    parser.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start over")
    parser.add_argument("--overfit-batch", type=int, default=0, metavar="STEPS",
                        help="only run the overfit-one-batch sanity check for STEPS steps")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Train one model as described by the config. Returns the run summary."""
    args = parse_args(argv)
    cfg = load_config(args.config, args.tier, args.overrides)
    run_dir = Path(cfg["paths"]["output_dir"]) / "runs" / cfg["run_name"]
    if args.overfit_batch:
        run_dir = run_dir.with_name(run_dir.name + "_overfit")
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg["seed"], cfg.get("deterministic", False))
    tcfg = cfg["train"]
    use_amp = bool(tcfg["amp"]) and device.type == "cuda"
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"

    # ---- resume state (read first: it fixes the normalization statistics) ----
    checkpoint = None
    if last_path.exists() and not args.fresh and not args.overfit_batch:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["config"]["model"]["name"] != cfg["model"]["name"]:
            raise RuntimeError(f"{last_path} belongs to another model; use --fresh or a new run_name")
        if checkpoint.get("finished"):
            logger.info("run %s already finished (best val macro-F1 %.4f at epoch %d); use --fresh to retrain",
                        cfg["run_name"], checkpoint["best_score"], checkpoint["best_epoch"])
            return checkpoint.get("summary", {})

    # ---- data ----
    norm = cfg["data"]["normalize"]
    if checkpoint is not None:
        mean, std = checkpoint["normalize"]["mean"], checkpoint["normalize"]["std"]
    elif norm["mean"] == "auto" or norm["std"] == "auto":
        from src.data.dataset import DocumentImageDataset, load_manifest_paths

        train_list, _ = tier_split_lists(cfg["data"]["tier"])
        stats_ds = DocumentImageDataset.from_split(
            cfg["paths"]["splits_dir"] / f"{train_list}.csv", cfg["paths"]["data_dir"],
            available=load_manifest_paths(cfg["paths"]["data_dir"]))
        mean, std = compute_mean_std(stats_ds, cfg["data"]["image_size"], seed=cfg["seed"])
        logger.info("dataset normalization statistics: mean %s std %s", mean, std)
    else:
        mean, std = list(norm["mean"]), list(norm["std"])
    cfg["data"]["normalize"] = {"mean": mean, "std": std}
    train_ds, val_ds = build_datasets(cfg, mean, std)
    logger.info("tier %s: %d train / %d val images", cfg["data"]["tier"], len(train_ds), len(val_ds))

    # ---- model ----
    model = build_model(cfg["model"]).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    n_params = count_parameters(model)
    logger.info("model %s: %.2fM parameters, device %s, AMP %s", cfg["model"]["name"], n_params / 1e6, device, use_amp)

    if args.overfit_batch:
        unaugmented_train = copy.copy(val_ds)  # validation transforms ...
        unaugmented_train.samples = train_ds.samples  # ... on training images
        accuracy = overfit_one_batch(model, unaugmented_train, cfg, device, args.overfit_batch)
        return {"overfit_batch_accuracy": accuracy}

    workers = cfg["data"]["num_workers"]
    train_loader = make_loader(train_ds, tcfg["batch_size"], shuffle=True, num_workers=workers,
                               seed=cfg["seed"], drop_last=len(train_ds) > tcfg["batch_size"])
    val_loader = make_loader(val_ds, tcfg["batch_size"] * 2, shuffle=False, num_workers=workers, seed=cfg["seed"])

    optimizer = torch.optim.AdamW(get_param_groups(model, cfg["model"]["name"], tcfg["lr"], tcfg.get("head_lr"),
                                                   tcfg["weight_decay"]))
    steps_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, tcfg["epochs"] * steps_per_epoch, tcfg["warmup_epochs"] * steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])

    start_epoch, best_score, best_epoch, bad_epochs, history = 1, -1.0, 0, 0, []
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_score, best_epoch = checkpoint["best_score"], checkpoint["best_epoch"]
        bad_epochs, history = checkpoint["bad_epochs"], checkpoint["history"]
        logger.info("resumed from %s: epoch %d done, best val macro-F1 %.4f", last_path, checkpoint["epoch"], best_score)
    save_config(cfg, run_dir / "config.yaml")

    def build_summary(finished: bool, stop_reason: str | None) -> dict:
        best_row = next((h for h in history if h["epoch"] == best_epoch), {})
        return {
            "run_name": cfg["run_name"],
            "model": cfg["model"]["name"],
            "tier": cfg["data"]["tier"],
            "finished": finished,
            "stop_reason": stop_reason,
            "epochs_completed": history[-1]["epoch"] if history else 0,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_score,
            "best_val_acc": best_row.get("val_acc"),
            "best_val_invoice_f1": best_row.get("val_invoice_f1"),
            "train_images": len(train_ds),
            "val_images": len(val_ds),
            "parameters": n_params,
            "best_checkpoint": str(best_path),
            "best_checkpoint_mb": round(best_path.stat().st_size / 1e6, 1) if best_path.exists() else None,
            "train_minutes": round(sum(h["epoch_seconds"] for h in history) / 60, 1),
            "normalize": {"mean": mean, "std": std},
            "environment": environment_info(),
        }

    # ---- training ----
    stop_reason = None
    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        epoch_start = time.time()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device,
                                      use_amp, tcfg.get("grad_clip"), tcfg.get("log_every", 0), epoch)
        val_stats = evaluate(model, val_loader, criterion, device, use_amp)
        row = {
            "epoch": epoch,
            "lr_backbone": group_lr(optimizer, "backbone"),
            "lr_head": group_lr(optimizer, "head"),
            "train_loss": round(train_stats["loss"], 5),
            "train_acc": round(train_stats["acc"], 5),
            "val_loss": round(val_stats["loss"], 5),
            "val_acc": round(val_stats["acc"], 5),
            "val_macro_f1": round(val_stats["macro_f1"], 5),
            "val_invoice_precision": round(val_stats["invoice_precision"], 5),
            "val_invoice_recall": round(val_stats["invoice_recall"], 5),
            "val_invoice_f1": round(val_stats["invoice_f1"], 5),
            "epoch_seconds": round(time.time() - epoch_start, 1),
            "train_images_per_second": round(train_stats["images_per_second"], 1),
        }
        history.append(row)

        improved = val_stats["macro_f1"] > best_score
        if improved:
            best_score, best_epoch, bad_epochs = val_stats["macro_f1"], epoch, 0
            atomic_torch_save({
                "model": model.state_dict(), "epoch": epoch, "val_metrics": val_stats,
                "config": to_serializable(cfg), "normalize": {"mean": mean, "std": std}, "classes": CLASSES,
            }, best_path)
        else:
            bad_epochs += 1

        if bad_epochs >= tcfg["early_stopping_patience"]:
            stop_reason = f"early stopping: no val macro-F1 gain for {bad_epochs} epochs"
        elif epoch == tcfg["epochs"]:
            stop_reason = "reached max epochs"
        finished = stop_reason is not None
        summary = build_summary(finished, stop_reason)
        atomic_torch_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
            "history": history, "normalize": {"mean": mean, "std": std}, "config": to_serializable(cfg),
            "finished": finished, "summary": summary,
        }, last_path)
        with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(history)
        write_json(summary, run_dir / "summary.json")

        logger.info(
            "epoch %d/%d  train loss %.4f acc %.4f | val loss %.4f acc %.4f macro-F1 %.4f invoice-F1 %.4f | %.0fs%s",
            epoch, tcfg["epochs"], row["train_loss"], row["train_acc"], row["val_loss"], row["val_acc"],
            row["val_macro_f1"], row["val_invoice_f1"], row["epoch_seconds"], "  * best" if improved else "")
        if finished:
            break

    summary = build_summary(True, stop_reason or "reached max epochs")
    write_json(summary, run_dir / "summary.json")
    append_csv_row(Path(cfg["paths"]["output_dir"]) / "runs_index.csv", {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": cfg["run_name"], "model": cfg["model"]["name"], "tier": cfg["data"]["tier"],
        "best_epoch": best_epoch, "best_val_macro_f1": round(best_score, 5),
        "best_val_acc": summary["best_val_acc"], "best_val_invoice_f1": summary["best_val_invoice_f1"],
        "train_minutes": summary["train_minutes"], "parameters": n_params,
        "lr": tcfg["lr"], "head_lr": tcfg.get("head_lr"), "batch_size": tcfg["batch_size"],
        "epochs": tcfg["epochs"], "seed": cfg["seed"], "checkpoint": str(best_path),
    })
    logger.info("done: %s, best val macro-F1 %.4f at epoch %d -> %s", summary["stop_reason"], best_score, best_epoch, best_path)
    return summary


if __name__ == "__main__":
    main()
