# Document Image Classification with CNNs

AIN 320 Computer Vision final project (Fall 2026, AUPP), Phase 1 proof of concept.
We classify a single document page image into one of the 16 RVL-CDIP classes and compare three CNNs:

| | Model | Init | Config |
|---|---|---|---|
| M1 | Custom CNN (5 conv blocks, documented in `src/models/custom_cnn.py`) | random | `configs/custom_cnn.yaml` |
| M2 | ResNet-50 | ImageNet, fine-tuned | `configs/resnet50.yaml` |
| M3 | EfficientNet-B0 | ImageNet, fine-tuned | `configs/efficientnet_b0.yaml` |

The full requirements and rationale are in [PRD.md](PRD.md).

## Repository layout

```
configs/      one YAML per model (all hyperparameters live here)
splits/       committed, seeded subset lists: dev/main/large train+val, full test
src/
  config.py           YAML + env + CLI config loading
  train.py            shared training loop (all models)
  evaluate.py         test metrics, confusion matrix, throughput; DiT reference (--dit)
  compare.py          comparison / invoice / literature tables, accuracy-vs-speed plot
  gradcam.py          Grad-CAM figures (correct vs misclassified, incl. invoice)
  error_analysis.py   sample ~50 errors to label, then summarise the labels
  duplicates.py       optional train/test near-duplicate check (perceptual hash)
  data/
    labels.py           classes, download URLs, split-list helpers
    make_splits.py      builds splits/ from the official label files
    build_working_copy.py  streams the 38.8 GB archive -> small PNG shards
    local_copy.py       unpacks shards to local disk, verifies counts
    preprocess.py       grayscale + resize (shared by build and inference)
    transforms.py       augmentation + normalization
    dataset.py          PyTorch Dataset / DataLoader
    pdf.py              PDF -> page images (PyMuPDF)
  models/
    custom_cnn.py       M1
    factory.py          builds M1/M2/M3, optimizer parameter groups
notebooks/    thin Colab runners (no core logic)
tests/        CPU smoke tests on synthetic data
results/      metrics and figures that go into the report (committed)
```

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q          # ~3 min on CPU; builds a fake archive and trains a tiny model
```

## Reproducing the results (Google Colab)

The team trains on free Colab. Notebooks clone this repo and call `src/`. Set `REPO_URL` in each notebook.

1. **`notebooks/00_build_working_copy.ipynb`** (once, CPU runtime, ~1–2 h): streams RVL-CDIP from
   Hugging Face and writes ~2–3 GB of 256×256 PNG shards to `MyDrive/ain320_doccls/working_copy`.
   Resumable: re-run after a disconnect.
2. **`notebooks/01_train.ipynb`** (T4 GPU): set `MODEL` and `TIER`, run all. Order: `custom_cnn` on `dev`,
   then each model on `main`. Checkpoints, logs and `runs_index.csv` go to `MyDrive/ain320_doccls/outputs`.
   Resumes automatically after a disconnect.
   **Or, to read and experiment with the training code**, use one notebook per architecture:
   `01a_train_custom_cnn.ipynb`, `01b_train_resnet50.ipynb`, `01c_train_efficientnet_b0.ipynb`. They write out
   `src/train.py` (config, augmentation, model, loop, checkpoint/resume) in editable cells, import only the data
   plumbing from `src/`, run on Colab or locally, and save checkpoints in the same format, so step 3 works on them.
   Copy any setting you keep back into `configs/` (and any custom-CNN architecture change into
   `src/models/custom_cnn.py`) so the reported runs stay reproducible from `src/`.
3. **`notebooks/02_evaluate.ipynb`** (T4 GPU): evaluates every trained model and DiT on the test set **once**,
   then builds the comparison tables, Grad-CAM figures, the error-analysis sheets and the optional duplicate
   check. Download the bundle at the end and copy it into `results/`.

The same steps from a terminal:

```bash
python -m src.data.make_splits                       # only to regenerate splits/ (already committed)
python -m src.data.build_working_copy --out-dir <drive>/working_copy --lists main_train main_val test
python -m src.data.local_copy unpack --src <drive>/working_copy --dst data/working_copy
python -m src.train --config configs/custom_cnn.yaml --tier dev --overfit-batch 200   # sanity check
python -m src.train --config configs/resnet50.yaml --tier main                        # train / resume
python -m src.evaluate --run-dir outputs/runs/resnet50_main                           # test set, once
python -m src.evaluate --dit                                                          # DiT reference
python -m src.compare                                                                 # tables + plot
python -m src.gradcam --run-dir outputs/runs/resnet50_main
python -m src.error_analysis sample --eval-dir outputs/eval/test/resnet50_main
python -m src.duplicates find && python -m src.compare --dup-threshold <d>           # optional
```

`src.evaluate` will not re-run the test set for a model that already has results (PRD 6); `--force`
overrides this and the rerun is recorded in `metrics.json`. Use `--split val` while developing.
Literature numbers in `results/published_results.yaml` are marked unverified until checked against each paper.

Paths come from the config and can be overridden with `DOCCLS_DATA_DIR`, `DOCCLS_OUTPUT_DIR` and
`DOCCLS_SPLITS_DIR`. Any config value can be overridden with `--set key.sub=value`.

## Data

* RVL-CDIP from `huggingface.co/datasets/aharley/rvl_cdip`: 16 classes, 320k / 40k / 40k official split
  (counts verified by `make_splits`; see `splits/summary.json`).
* Tiers (stratified, equal images per class, seed 42, nested Dev ⊂ Main ⊂ Large):

  | Tier | Train | Val | Use |
  |---|---|---|---|
  | Dev | 200/class (3,200) | 50/class (800) | debugging |
  | **Main** | **2,000/class (32,000)** | **500/class (8,000)** | **all reported comparisons** |
  | Large | 5,000/class (80,000) | Main's val | optional, best model only |

  Test is always the full official test split (40,000), used once per final model.

## Design decisions (for the report)

| Decision | Choice | Why |
|---|---|---|
| Page shape | Plain resize to 256×256 (stored), 224×224 (model) | Matches the RVL-CDIP literature; pad-to-square wastes ~25% of pixels on tall pages |
| Channels | Grayscale; replicated to 3 channels for ImageNet models | Pretrained weights expect RGB |
| Normalization | ImageNet stats (M2, M3); dataset stats computed from the train tier (M1) | PRD 5.1.4 |
| Augmentation | ±4° rotation, ±3% shift, ±5% scale, brightness/contrast, blur, noise; white fill; **no flips** | Simulates scan variation without destroying layout |
| Fine-tuning | Differential LR: backbone 1e-4, new head 1e-3; M2 and M3 share all hyperparameters | Fair comparison; head learns fast without wrecking features |
| Optimisation | AdamW, 1 epoch warm-up + cosine decay, label smoothing 0.1, AMP, grad clip 1.0 | Label smoothing because of RVL-CDIP label noise |
| Model selection | Early stopping on validation macro-F1 | PRD 5.4 |
| Checkpoints | `last.pt` (full state, every epoch) + `best.pt` (weights only) | Disconnect costs ≤ 1 epoch; lean Drive usage |
| Seeds | 42 everywhere; versions + GPU logged in each `summary.json` | Reproducibility |

## Status

Code done for Checklist Steps 0–6 (data, training, evaluation, DiT reference, comparison, Grad-CAM,
error analysis, duplicate check). Still to do by the team: run the notebooks on Colab, label the
error-analysis sheet, verify the published numbers, write the report.
