# PRD: Document Image Classification with CNNs

**Course:** AIN 320 — Computer Vision, Fall 2026, American University of Phnom Penh
**Instructor:** Ly Rottana
**Project grade weight:** 40% of course final score
**Status:** Phase 1 (Proof of Concept) — draft, 2026-10-19

---

## Table of Contents

1. [Overview](#1-overview)
2. [Goals, Non-Goals, Anti-Goals](#2-goals-non-goals-anti-goals)
3. [Scope and Phases](#3-scope-and-phases)
4. [Dataset](#4-dataset)
5. [Methodology](#5-methodology)
6. [Evaluation](#6-evaluation)
7. [Technical Setup](#7-technical-setup)
8. [Timeline and Milestones](#8-timeline-and-milestones)
9. [Team Roles](#9-team-roles)
10. [Risks and Mitigations](#10-risks-and-mitigations)
11. [Course Rubric Mapping](#11-course-rubric-mapping)
12. [Phase 1 Checklist](#12-phase-1-checklist)
13. [Future Phases](#13-future-phases)
- [Appendix A: Public Dataset Survey](#appendix-a-public-dataset-survey)
- [Appendix B: Literature References](#appendix-b-literature-references)

---

## 1. Overview

### 1.1 The problem

A real Cambodian business (our client) receives documents in **bulk imports**: large batches of mixed files (images and PDFs) that must be sorted by document type before anyone can process them. The documents are mostly in **Khmer and English** and include:

- electricity bills
- water bills
- rent documents
- purchase orders
- bank statements
- invoices

Today this sorting is handled by a third-party tool (Jev, which recently added multimodal support). The long-term goal is to replace that dependency with a model we build and control.

### 1.2 What we are building

A computer-vision model that takes **one document page as an image** and predicts its **document type**. PDFs are converted to page images first, so the same model handles both formats.

### 1.3 Why two audiences

This project serves two audiences at once:

1. **The AIN 320 final project.** It must select, design, train and compare computer-vision models on a dataset, and be written up in a 4–6 page technical report.
2. **The client.** It is a proof of concept showing that a self-built CNN classifier is viable, before we invest in training on their real documents.

The client's documents **cannot be used** for the course project. Phase 1 therefore uses **public data only**.

### 1.4 Why classification by image (and not by text)

Document types differ strongly in **visual layout**: tables, logos, letterheads, line items, totals, form fields. A CNN can learn these layouts without reading the text. This makes the approach language-agnostic in principle, which matters for Khmer: Khmer OCR support is weak in common open-source tools (see §13).

---

## 2. Goals, Non-Goals, Anti-Goals

### 2.1 Goals (Phase 1)

| # | Goal | Success looks like |
|---|------|--------------------|
| G1 | Build an end-to-end image classification pipeline for document pages | Raw image/PDF in, predicted class and confidence out, reproducible from the repo |
| G2 | Train and compare **three** CNN-based models on RVL-CDIP | A comparison table with accuracy, macro-F1, model size and inference speed |
| G3 | Reach results in line with published CNN baselines | Best model's test accuracy (Main tier, §4.3) is placed next to the ~90% published for ImageNet-pretrained CNNs trained on the full set, and the gap is explained. This is a reference point, not a pass/fail bar, and does not make the Full-tier run mandatory. |
| G4 | Report **invoice** performance separately | Invoice precision, recall and F1, because invoice is the one RVL-CDIP class the client also has |
| G5 | Show *what the model looks at* | Grad-CAM heatmaps and an error analysis showing whether the model uses real document structure or shortcuts |
| G6 | Deliver graded course artifacts | Well-commented source code, a 4–6 page report, a mid-progress and a final presentation |

### 2.2 Non-Goals (Phase 1)

- **No client data.** Not for training, validation or testing.
- **No Khmer-language support.** RVL-CDIP is English-only. This is a known, documented gap (§4.5).
- **No synthetic document generation.**
- **No other public datasets.** RVL-CDIP only, to avoid cross-source shortcut learning (§10).
- **No payable vs. receivable classification.** This is a business rule about who issued the invoice, not a visual difference. It belongs to Phase 3.
- **No RNNs or text/OCR branch.** Single-page image classification does not need sequence models. Text-based or multimodal approaches belong to later phases.
- **No multi-page document reasoning.** Each page is classified independently.
- **No production deployment.**

### 2.3 Anti-Goals (outcomes that count as failure even if the numbers look good)

- **A high score from shortcuts.** The model reaches high accuracy by exploiting dataset artifacts (scan noise, image borders, near-duplicate pages) rather than layout. We must check this with Grad-CAM and error analysis.
- **Inflated test scores from leakage.** Train/test duplicates in RVL-CDIP inflate the test score and we report it uncritically.
- **Overclaiming client readiness.** Results are presented as "ready for the client" when only one of the client's classes was ever tested.
- **Code that can't be rerun.** Results only exist in one person's notebook and cannot be reproduced from the repo (code preparation is 15% of the report grade).

---

## 3. Scope and Phases

| Phase | Data | Classes | Purpose |
|-------|------|---------|---------|
| **1 — Proof of Concept (this PRD)** | RVL-CDIP only | 16 standard RVL-CDIP classes | Prove the pipeline and architectures work; course deliverable |
| 2 — Client adaptation | Client's real documents (+ possibly public data from Appendix A) | Client's types: electricity bill, water bill, rent, purchase order, bank statement, invoice, other | Fine-tune the Phase 1 models on real Khmer/English documents |
| 3 — Business rules | Phase 2 model + text extraction | Adds payable vs. receivable | Compare the issuer/recipient on an invoice with the client's identity |

Only Phase 1 is planned in detail here. Phases 2–3 are outlined in §13.

---

## 4. Dataset

### 4.1 RVL-CDIP

- **Source:** Harley et al., 2015. A subset of the IIT-CDIP Test Collection (Legacy Tobacco Document Library).
- **Access:** Hugging Face `aharley/rvl_cdip` (verify the download path works; mirrors exist if not). About 38.8 GB.
- **Size:** 400,000 grayscale scanned page images (TIFF), 25,000 per class.
- **Official split:** 320,000 train / 40,000 validation / 40,000 test.
- **Image properties:** grayscale, low resolution (longest side ≤1000 px), real scans with noise, skew, fax artifacts, stamps and handwriting.
- **Labels:** one document-type label per image. No text, OCR or bounding boxes.
- **License:** listed as "other" (derived from Legacy Tobacco Document Library). Fine for academic use. Cite the original paper.

**The 16 classes:** letter, form, email, handwritten, advertisement, scientific report, scientific publication, specification, file folder, news article, budget, **invoice**, presentation, questionnaire, resume, memo.

### 4.2 Why RVL-CDIP only

- **One source, scanned the same way.** Mixing sources (e.g. synthetic bank statements plus scanned invoices) lets a CNN learn "which dataset is this from" instead of "what document is this." With one source, that shortcut is removed.
- **It is the standard benchmark,** so our results can be compared directly with published work (needed for the literature review and comparative study).
- **It is real computer vision work.** Real scans need resizing, normalization and augmentation, and transfer learning is non-trivial.

### 4.3 Subset strategy (for free Colab)

The team trains on **free Google Colab** (§7.2). The 38.8 GB raw archive is only downloaded and read once. What we keep and train on is a small **working copy** of about 1–3 GB.

**Keep all 16 classes.** Cutting classes does not shrink the download, because all classes are in one archive. Image count, not class count, drives storage and training time. Keeping 16 classes also keeps results comparable with published work.

Use **tiered subsets**, always **stratified** (equal images per class) and drawn from the **official splits**, so the test set is never touched during development:

| Tier | Train | Val | Test | Use |
|------|-------|-----|------|-----|
| Dev | 200/class (3.2k), a subset of Main | 50/class (800), a subset of Main | — | Debugging the pipeline; fast iteration |
| **Main** | **2,000/class (32k)** | **500/class (8k)** | **Full official test (40k)** | **All reported comparisons** |
| Large (optional) | 5,000/class (80k) | 500/class (8k) | Full official test (40k) | Best model only, if time and Colab quota allow |

- The full 320k training set is **out of scope** on free Colab.
- Fix the random seed for subset sampling and **save the sampled file lists** to the repo, so every model uses identical data. If the Large tier might be used, extract it in the same pass (it is a superset of Main) so the archive only needs reading once.
- **All three models must be compared on the same tier (Main).**
- Report which tier each number comes from. Published ~90% figures use the full 320k training set, so lower numbers on Main are expected and must be explained, not hidden.

### 4.4 Known data issues

- **Label noise.** Larson et al. (2023) estimate roughly 8% of RVL-CDIP labels are wrong, and a later study ("Revising RVL-CDIP") reports about 12%. Some errors will be "correct" predictions marked wrong. Mention this in error analysis.
- **Train/test overlap.** Both studies found near-duplicate documents across splits; "Revising RVL-CDIP" reports duplication as high as ~35%. This may inflate test scores. At minimum, discuss it in the report. Optionally, run a near-duplicate check (e.g. perceptual hashing) between the Main train subset and the test set and report accuracy with and without duplicates.
- **Era and domain.** 1980s–90s American tobacco-industry documents. They look nothing like modern Cambodian bills.
- **Corrupt files.** At least one corrupt image has been reported in public copies. The loader must skip unreadable files and log them, not crash.

### 4.5 Known gaps (stated explicitly in the report)

- **Khmer:** no Khmer documents in Phase 1. How well the model transfers to Khmer/English Cambodian documents is **unknown** until Phase 2. No usable public Khmer business-document dataset exists (Appendix A.3).
- **Client classes:** only *invoice* overlaps with the client's list. Electricity bill, water bill, rent, purchase order and bank statement are not tested in Phase 1.

### 4.6 Optional: Cambodia bonus idea (not in Phase 1 scope)

The course gives bonus points for data related to Cambodia or for contributing to a dataset. Phase 1 stays RVL-CDIP-only, but if time allows the team **may** propose this as an extension:

- Collect a small **Cambodian test set**: e.g. 100–200 team members' own EDC/PPWSA bills, shop receipts and invoices, with personal details redacted.
- Use it for **testing only, never training**, to measure how a model trained on RVL-CDIP handles real Khmer/English documents.

This would also be a head start on Phase 2. **It is not required, and it is not in the checklist.**

---

## 5. Methodology

### 5.1 Input pipeline

1. **Load.** TIFF/PNG/JPG images via Pillow. PDFs are rendered to one image per page with PyMuPDF (`fitz`) at ~150 DPI. RVL-CDIP has no PDFs, but the pipeline must support them for Phase 2. One small test PDF in the repo is enough to prove it works.
2. **Convert to grayscale,** then replicate to 3 channels for ImageNet-pretrained models.
3. **Resize** to the model input size (default 224×224). Document pages are tall, so record the choice made (plain resize vs. pad-to-square then resize) and keep it the same across models.
4. **Normalize** with ImageNet mean/std for pretrained models. For the from-scratch CNN, use dataset statistics.
5. **Build the working copy by streaming (one-time step).** Free Colab's temporary disk (~80–100 GB) may not hold both the 38.8 GB archive and its unpacked contents. So the extraction script must **process the archive as it streams in**:
   - read each file in the `.tar.gz` sequentially;
   - keep it only if it is in the pre-sampled file list (§4.3);
   - convert to grayscale, resize, and write it as a small PNG.

   Raw TIFFs and the archive are never stored in full.
   - The label files (small text files listing image path → class per split) are fetched first to build the sample lists.
   - Save the working copy, plus a manifest CSV (`path, label, split`), to Google Drive.
   - Expect this step to take roughly 1–2 hours. It must be **resumable**: skip files already written.
   - Store images at a fixed size ≥224 px (e.g. 256 to allow crop augmentation; builder's choice, documented).

   Every training run afterwards reads only the working copy. Copy it from Drive to the Colab local disk at session start for fast loading.

### 5.2 Augmentation (training only)

These are document-appropriate augmentations that simulate scan variation:

- small rotation (±3–5°)
- slight scale/translate
- brightness/contrast jitter
- Gaussian blur
- mild noise

**Do not use** horizontal/vertical flips or large rotations: flipped documents don't occur and would destroy layout cues.

### 5.3 Models compared

| # | Model | Role in study | Init |
|---|-------|---------------|------|
| M1 | **Custom CNN** (e.g. 4–5 conv blocks + BatchNorm + pooling + FC head) | Baseline: how far a small model trained from scratch gets | Random |
| M2 | **ResNet-50** | Standard deep CNN, transfer learning | ImageNet-pretrained, fine-tuned |
| M3 | **EfficientNet-B0** | Light, fast CNN; tests the accuracy-vs-speed trade-off, which matters for bulk imports | ImageNet-pretrained, fine-tuned |
| Ref | **DiT-base** (`microsoft/dit-base-finetuned-rvlcdip`) | Literature reference: a document-pretrained transformer, evaluated off-the-shelf on our test set | Already fine-tuned (no training) |

**Notes**

- The custom CNN architecture is left to the builder but must be documented layer by layer in the report.
- DiT is used **only for evaluation**, as "comparison with existing methods in the literature." It is not trained by us. If running it is impractical, cite its published numbers instead.

### 5.4 Training setup (defaults; the builder may tune them)

- **Loss:** cross-entropy (optionally with label smoothing ~0.1, given label noise).
- **Optimizer:** AdamW.
  - Pretrained models: learning rate ~1e-4 for the backbone and higher for the new head, or fine-tune in two stages (head only, then all layers).
  - Custom CNN: learning rate ~1e-3.
- **Schedule:** cosine decay or reduce-on-plateau. Early stopping on validation macro-F1.
- **Batch size:** as large as memory allows. Use mixed precision (AMP) if the GPU supports it.
- **Reproducibility:** fixed seeds; log every run's hyperparameters, metrics and checkpoint path.

---

## 6. Evaluation

All numbers are reported on the **held-out test set** (see §4.3). The test set is used **once per final model**, never for tuning.

**Metrics for every model**

- Overall accuracy
- **Macro-F1** (main metric; classes are balanced, but macro-F1 still shows weak classes)
- Per-class precision, recall and F1
- **Invoice precision, recall and F1** (reported separately in its own table; client-relevant)
- Confusion matrix (16×16), with discussion of the most-confused pairs (e.g. form vs. questionnaire, scientific report vs. scientific publication)
- Number of parameters and model file size
- **Inference throughput** (pages per second on the team's hardware, batch inference), which matters for bulk imports

**Comparative study**

- One table comparing M1, M2, M3 and the DiT reference
- A second table placing our best result next to published RVL-CDIP results (Appendix B)

**Interpretability and error analysis**

- **Grad-CAM** heatmaps for correctly and incorrectly classified examples from several classes, at least including invoice. Does the model look at tables, headers and totals, or at borders and noise?
- **Error analysis:** inspect ~50 misclassified test images. Categorize them as genuinely ambiguous, label noise, or real model error.
- **Duplicate check (optional):** accuracy with and without near-duplicates of training images (§4.4).

---

## 7. Technical Setup

### 7.1 Stack

- **Language:** Python 3.10+
- **Deep learning:** PyTorch + torchvision. `timm` is fine for EfficientNet. Hugging Face `transformers` is used only for the DiT reference.
- **Data:** Hugging Face `datasets` (or a direct download), Pillow, PyMuPDF
- **Metrics/plots:** scikit-learn, matplotlib. `pytorch-grad-cam` for Grad-CAM.
- **Experiment logging:** CSV/JSON logs in the repo at minimum. Weights & Biases or TensorBoard optional.

### 7.2 Compute: free Google Colab (decided)

- **GPU:** single free-tier GPU (typically T4, 16 GB). Availability is not guaranteed, and sessions disconnect after a few hours or when idle.
- **Storage:** Google Drive (free 15 GB) holds the working copy (~1–3 GB), checkpoints, and results. The raw archive never goes to Drive (§5.1).
- **Required consequences:**
  - **Checkpoint and resume.** Training saves a checkpoint to Drive every epoch and can resume from the latest one (model, optimizer, scheduler, epoch, best score). A disconnect must cost at most one epoch.
  - **Mixed precision (AMP)** on by default.
  - **Keep checkpoints lean.** Keep only the latest and best checkpoint per model; ResNet-50 checkpoints are ~100 MB each, which matters on a 15 GB Drive.
  - Rough expectation: under an hour of training per model on the Main tier (32k images). The builder should measure this on the Dev tier first.

### 7.3 Development workflow (Claude Code + Colab)

- Code is written locally in this repo (with Claude Code) and pushed to a **GitHub repository**.
- Colab notebooks are thin runners. They:
  - mount Drive;
  - `git clone`/`git pull` the repo;
  - `pip install -r requirements.txt`;
  - call scripts in `src/` with a config.

  No core logic lives only in Colab.
- Provide one runner notebook per stage: `00_build_working_copy.ipynb`, `01_train.ipynb` (model chosen by config), `02_evaluate.ipynb`.
- Paths (Drive root, data dir, checkpoint dir) come from config or environment, so the same code runs locally (CPU, Dev tier smoke tests) and on Colab.

### 7.4 Repository layout (suggested)

```
final-project/
├── PRD.md
├── README.md               # how to set up, download data, train, evaluate
├── requirements.txt
├── configs/                # one YAML per model/experiment
├── data/                   # gitignored: raw + cached data
├── splits/                 # saved subset file lists (committed)
├── src/
│   ├── data/               # download, preprocessing, dataset class, PDF→image
│   ├── models/             # custom CNN, model factory
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
├── notebooks/              # Colab runner notebooks + exploration; no core logic
├── results/                # metrics, confusion matrices, figures (committed)
└── report/                 # report draft and figures
```

### 7.5 Code quality (15% of report grade)

- Every module and function has a docstring. Non-obvious steps are commented.
- No core logic lives only in notebooks. Notebooks call `src/`.
- Results can be reproduced from the README with documented commands.
- Random seeds are fixed and documented.
- The repo is under git with a `.gitignore` for data and checkpoints.

---

## 8. Timeline and Milestones

Course dates: started 24 Aug 2026 (Week 1). Proposal was due 14 Sep. Mid-progress presentation is in **Week 10 (week of 26 Oct 2026)**. Final presentation date TBD; project deadline is end of semester.

| Week (dates) | Milestone | Checklist steps |
|---|---|---|
| Week 9 (19–25 Oct) | Setup, data downloaded and explored, preprocessing cached for the Dev tier, baseline CNN running end to end | 0, 1, 2, start 3 |
| **Week 10 (26–30 Oct)** | **Mid-progress presentation:** problem, dataset, method, baseline results, plan for the remaining models | 3, 8 |
| Weeks 11–12 | Main-tier cache. ResNet-50 and EfficientNet-B0 trained. DiT reference evaluated. | 4, 5 |
| Week 13 | Full evaluation, comparison tables, Grad-CAM, error analysis | 6 |
| Week 14 → end | Report writing, code cleanup, final presentation prep | 9 |

The Week 9→10 window is tight. If baseline results are not ready, the mid-progress presentation shows the **pipeline and Dev-tier results** instead.

---

## 9. Team Roles

Up to 4 members. Individual performance and individual Q&A are graded, so **each member owns one visible piece** and can explain it. Suggested split (names to be filled in):

| Role | Owns | Report sections |
|------|------|-----------------|
| Data lead | Download, exploration, subsets, preprocessing, augmentation, PDF→image | Dataset |
| Baseline model lead | Custom CNN design and training; training loop | Methodology (M1) |
| Transfer learning lead | ResNet-50, EfficientNet-B0, DiT reference | Methodology (M2, M3), Literature review |
| Evaluation lead | Metrics, comparison tables, Grad-CAM, error analysis | Experiments & Results, Discussion |

Everyone contributes to the introduction, abstract and presentations. If the team has fewer than 4 people, merge the Baseline and Transfer learning roles.

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Free Colab can't handle 320k images | Can't train in time | Main tier of 2,000/class (§4.3); working copy; mixed precision; EfficientNet-B0 is cheap |
| 38 GB archive vs. Colab disk / 15 GB Drive | Pipeline blocked | Stream-and-shrink extraction, resumable; only the ~1–3 GB working copy is stored (§5.1) |
| Colab disconnects or GPU unavailable | Lost training progress | Per-epoch checkpoints to Drive + resume (§7.2); run long jobs early, not the night before |
| Shortcut learning (borders, noise, duplicates) | Misleading accuracy | Grad-CAM, error analysis, optional duplicate check (§6) |
| Label noise (~8%) | Lower apparent accuracy, confusing errors | Label smoothing; discuss in error analysis |
| Week 10 mid-progress is one week away | Weak presentation | Present the pipeline and Dev-tier baseline; the plan for the rest is itself content |
| Results don't transfer to client documents | Phase 2 surprises | Stated as a known gap (§4.5); not claimed in Phase 1 |
| Work concentrated in one person's notebook | Code grade, individual grades | Repo structure and Colab workflow (§7.3–§7.4), role split (§9) |

---

## 11. Course Rubric Mapping

### 11.1 Report (4–6 pages excluding references)

| Rubric section | Weight | Source in this PRD |
|---|---|---|
| Abstract | 10% | §1, headline results from §6 |
| Introduction | 10% | §1 (problem, client motivation, why image-based) |
| Literature Review | 10% | Appendix B, §5.3 |
| Dataset | 10% | §4 (incl. known issues and gaps) |
| Methodology | 15% | §5 |
| Experiments and Results | 15% | §6 |
| Discussion and Conclusion | 10% | §6 error analysis, §4.5 gaps, §13 future phases |
| Bibliography | 5% | Appendix B |
| Code Preparation | 15% | §7.3–§7.5 |

### 11.2 Presentation

| Rubric section | Weight | Source |
|---|---|---|
| Introduction | 10% | §1 |
| Literature Review | 10% | Appendix B |
| Dataset | 15% | §4 |
| Methodology | 15% | §5 |
| Experiments and Results | 15% | §6 |
| Demo | 15% | Not covered by this PRD. The team plans it separately. |
| Individual Performance / Q&A | 10% + 10% | §9 role ownership |

**Bonus:** Cambodia-related data or dataset contribution. See the optional idea in §4.6.

---

## 12. Phase 1 Checklist

### Step 0 — Project setup
- [ ] Create a git repo with the layout from §7.4, plus `.gitignore` (data, checkpoints, caches); push to GitHub
- [ ] `requirements.txt` with pinned versions; verify install on free Colab and locally
- [ ] Colab runner notebook skeleton: mount Drive, clone/pull repo, install, check GPU (§7.3)
- [ ] Drive folder structure for working copy, checkpoints and results
- [ ] Agree on team roles (§9) and write names into this PRD

### Step 1 — Build the working copy and explore it
- [ ] Find the RVL-CDIP archive URL and label files (Hugging Face `aharley/rvl_cdip` repo files or the CMU page); confirm the label-file format
- [ ] From the label files: confirm 16 classes and 320k/40k/40k split counts
- [ ] Generate and commit stratified Main (and Dev ⊂ Main, optional Large ⊇ Main) sample lists with a fixed seed (§4.3), plus the full test list
- [ ] Streaming extraction script (§5.1): reads the archive sequentially, keeps only sampled files, writes resized grayscale PNGs + manifest CSV to Drive; resumable; logs corrupt files and skips them
- [ ] Run it on Colab; record time taken and final working-copy size
- [ ] Verify counts per class/split in the manifest match the sample lists
- [ ] Exploration notebook: sample grid per class, original size distribution (logged during extraction), class balance
- [ ] Write down observations for the report's Dataset section (noise, skew, class look-alikes)

### Step 2 — Preprocessing and loading pipeline
- [ ] Load → grayscale → 3-channel → resize → normalize (§5.1); decide resize vs. pad-to-square and document it (applied in the extraction step, so the choice is fixed early)
- [ ] PDF → page images via PyMuPDF; test on at least one sample PDF
- [ ] Training augmentations (§5.2); visual check of augmented samples (no flips)
- [ ] PyTorch `Dataset`/`DataLoader` reading the manifest; copy working copy from Drive to Colab local disk at start; benchmark loading speed
- [ ] Unit check: one batch has the right shape, dtype, value range and labels

### Step 3 — Baseline: custom CNN (M1)
- [ ] Define the architecture; document each layer
- [ ] Shared training loop (`train.py`): config-driven, seeds, logging, AMP, early stopping on val macro-F1
- [ ] Per-epoch checkpoint to Drive + resume from latest (§7.2); test it by interrupting a run
- [ ] Overfit a tiny batch to confirm the loop works
- [ ] Train on the Dev tier; then on the Main tier
- [ ] Log validation curves and the final validation score

### Step 4 — ResNet-50 fine-tuning (M2)
- [ ] Load ImageNet weights; replace the head with 16 outputs
- [ ] Choose a fine-tuning strategy (two-stage or differential learning rates); document it
- [ ] Train on the Main tier with the same subset and preprocessing as M1
- [ ] Log curves and the best checkpoint

### Step 5 — EfficientNet-B0 (M3) and DiT reference
- [ ] Fine-tune EfficientNet-B0 on the Main tier, same protocol as M2
- [ ] Run `microsoft/dit-base-finetuned-rvlcdip` inference on the test set with its own preprocessing (or record published numbers if impractical)
- [ ] Measure parameters, model size and throughput (pages/sec) for every model

### Step 6 — Evaluation and comparative study
- [ ] `evaluate.py`: test-set metrics for each final model, run once (§6)
- [ ] Comparison table: M1 vs. M2 vs. M3 vs. DiT
- [ ] Table comparing our best result with published RVL-CDIP results
- [ ] Invoice-specific precision/recall/F1 table
- [ ] Confusion matrices and discussion of top confused pairs
- [ ] Grad-CAM figures for correct and incorrect predictions (including invoice)
- [ ] Error analysis of ~50 misclassified images (ambiguous / label noise / model error)
- [ ] (Optional) near-duplicate check between train subset and test set
- [ ] Save all metrics and figures to `results/`

### Step 7 — Literature review
- [ ] Read and summarize the core papers in Appendix B
- [ ] Note each paper's method, input and RVL-CDIP accuracy (verify numbers from the papers themselves)
- [ ] Position our work: image-only CNN comparison, client motivation, Khmer gap

### Step 8 — Mid-progress presentation (Week 10)
- [ ] Slides: problem and client motivation, dataset, method, baseline results (Dev or Main tier), remaining plan
- [ ] Each member presents their own part
- [ ] Rehearse; prepare for questions on dataset choice and the Khmer gap

### Step 9 — Final report, code, presentation
- [ ] Report written in the provided template, 4–6 pages excluding references, following §11.1
- [ ] Known gaps (§4.5) and future phases (§13) in the Discussion
- [ ] Code cleanup: docstrings, comments, README with reproduction steps
- [ ] Fresh-clone test: someone other than the author reproduces one evaluation from the README
- [ ] Final presentation slides following §11.2; each member owns a section
- [ ] Submit source code and report

---

## 13. Future Phases (outline only)

### Phase 2 — Client adaptation
- **Classes:** electricity bill, water bill, rent, purchase order, bank statement, invoice, plus an "other" class for anything else.
- **Data:** the client's real documents, labeled (likely from their past sorted imports), with a strict train/test split by time or batch. Public datasets from Appendix A may supplement thin classes, mixed carefully to avoid source shortcuts.
- **Approach:** fine-tune the best Phase 1 model (RVL-CDIP-pretrained) on client data. Evaluate separately on Khmer and English documents.
- **License check before reuse:** RVL-CDIP's terms are only confirmed for academic use. Before shipping RVL-CDIP-trained weights to the client, confirm that commercial use is allowed. If it isn't, start Phase 2 from ImageNet weights and retrain on client data.
- **Bulk-import flow:** batch inference over a folder/ZIP, low-confidence pages routed to human review.
- **Text branch (optional):** Khmer OCR is limited. Tesseract `khm` exists but is weak; PaddleOCR and EasyOCR do not support Khmer; Google Document AI lists Khmer. An image-only classifier remains the primary approach.

### Phase 3 — Payable vs. receivable
- Extract the issuer/recipient from invoices (OCR or a document-understanding model) and compare them with the client's identity. This is a rule-based step on top of the classifier, not a new image class.

---

## Appendix A: Public Dataset Survey

Survey conducted 2026-10-18/19 for Phase 2 planning. **None of these are used in Phase 1.** Verify licenses before use; many are non-commercial (NC), which is fine for coursework but not for client deployment.

### A.1 Business documents (invoice, receipt, purchase order, rent)

| Dataset | Link | Contents | Classes | Real/Synthetic | License |
|---|---|---|---|---|---|
| Company-Document-Dataset-v2 | huggingface.co/datasets/AyoubChLin/Company-document-dataset-v2 | 404k pages + 120k degraded scans | 13 incl. invoice (48k), purchase_order (50k), receipt, account_statement | Synthetic (few templates) | Apache-2.0 |
| RVL-CDIP | huggingface.co/datasets/aharley/rvl_cdip | 400k scans | 16 incl. invoice | Real | Other |
| DocILE | docile.rossum.ai | 6.7k annotated + 100k synthetic + ~1M unlabeled | Invoices, POs, utility bills etc. (no type labels) | Real + synthetic | Research, request form |
| FATURA | zenodo.org/records/10371464 | 10k invoice images | Invoice | Synthetic (50 templates) | CC BY 4.0 |
| CORD v2 | huggingface.co/datasets/naver-clova-ix/cord-v2 | 1k receipt photos (Indonesia) | Receipt | Real | CC BY 4.0 |
| SROIE | HF mirrors (darentang/sroie) | ~1k scanned receipts (Malaysia) | Receipt | Real | Unstated |
| CORU | huggingface.co/datasets/Timokerr/CORU | ~20k receipts (Arabic/English) | Receipt | Real | MIT |
| RealKIE | zenodo.org/records/13327077 | FCC ad-buy invoices etc. (PDF) | Invoice/order-like | Real | CC BY-NC 4.0 |
| Leases / rent | — | **No public image dataset found** | — | — | — |

### A.2 Bank statements and utility bills

| Dataset | Link | Contents | Real/Synthetic | License |
|---|---|---|---|---|
| IDSEM | zenodo.org/records/6373179 | 75k Spanish electricity bills (PDF) | Synthetic | CC BY 4.0 |
| Multi-Document OCR Dataset | kaggle.com/datasets/miadul/multi-document-dataset-for-ocr-document-ai-and-llm | ~1k bank statements + 7 other types | Synthetic | MIT |
| Panhapich/bank-statement-detection | huggingface.co/datasets/Panhapich/bank-statement-detection | 15k bank statement pages | Synthetic | MIT |
| Personal Financial Dataset for India | kaggle.com/datasets/mehaksingal/personal-financial-dataset-for-india | 426 images incl. 91 bank statements, 99 utility bills | Likely real | ODbL |
| Roboflow BillDetection | universe.roboflow.com/abc-4rfex/billdetection | 233 images, water/electricity | Real | CC BY 4.0 |
| Roboflow Electricity Bills workspace | universe.roboflow.com/electricity-bills | Small Indian electricity/water/gas sets | Real | CC BY 4.0 |
| Water bills | — | **Under ~300 images in total publicly** | — | — |

### A.3 Khmer and Southeast Asian documents

| Dataset | Link | Contents | Status / License |
|---|---|---|---|
| KH-FUNSD | github.com/back-kh/KH-FUNSD | 158 Khmer invoices, receipts, quotations | Not yet released (email author); CC BY-NC 4.0 |
| SEA-Vision | arxiv.org/abs/2603.15409 | 15k pages incl. 1.3k Khmer (non-business) | Release planned; CC BY-NC 4.0 |
| Khmer layout dataset | arxiv.org/abs/2603.00707 | ~9k Khmer pages (books, slides, press) | Gated, in review |
| Synthetic Khmer text lines | HF: mrrtmob/khmer_english_ocr_image_line; KimkosalYon/khmer-ocr-200k-siemreap-arial | Line crops for OCR pretraining | CC BY 4.0 / MIT |
| WildKhmerST | l3i-share.univ-lr.fr/2025WildKhmerST/ | 10k scene-text images | CC BY-NC-ND 4.0 |

**Bottom line:** no public Khmer business-document image dataset is usable today. Phase 2 depends on client data.

---

## Appendix B: Literature References

Starting list for the literature review. **Verify every accuracy figure against the original paper before citing it.**

1. Harley, A. W., Ufkes, A., & Derpanis, K. G. (2015). *Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval.* ICDAR. Introduces RVL-CDIP; AlexNet-based CNN baseline (~90%).
2. Afzal, M. Z., et al. (2017). *Cutting the Error by Half: Investigation of Very Deep CNN and Advanced Training Strategies for Document Image Classification.* ICDAR. VGG-16 / ResNet-50 transfer learning (~90–91%).
3. Das, A., et al. (2018). *Document Image Classification with Intra-Domain Transfer Learning and Stacked Generalization of Deep Convolutional Neural Networks.* ICPR. Region-based CNN ensemble (~92%).
4. He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR. ResNet.
5. Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
6. Li, J., et al. (2022). *DiT: Self-supervised Pre-training for Document Image Transformer.* ACM MM. Document-pretrained transformer (~92%).
7. Huang, Y., et al. (2022). *LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking.* ACM MM. Multimodal image+text (~95%). Useful contrast: needs OCR text.
8. Larson, S., et al. (2023). *On Evaluation of Document Classification using RVL-CDIP.* EACL. Label noise and train/test overlap in RVL-CDIP.
9. *Revising RVL-CDIP: Quantifying Errors and Test-Train Overlap* (find authors/venue before citing). Reports ~12% label errors and ~35% test-train duplication.
10. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
