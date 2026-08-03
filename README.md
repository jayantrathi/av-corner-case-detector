# AV Corner-Case Detector

A perception-layer safety monitor for autonomous driving: it flags driving scenes that are out-of-distribution relative to normal driving, on the theory that the dangerous failures in autonomy are the situations a model was never trained on, not the ones it was.

## The problem

Standard AV perception (object detectors, segmentation models) can only recognize the classes they were trained on. Run one on a genuine corner case — debris in the lane, an object with no standard class, a scene the training distribution never covered — and it doesn't fail loudly. It's confident and wrong. This project targets that specific gap: a monitor that says "this situation is unusual, don't trust the stack here," independent of what any downstream detector thinks it sees.

## Approach

Two independent methods were built and evaluated against the same held-out, scene-level split of real hazard data, so the comparison is honest:

**kNN patch-embedding anomaly scorer (primary method).** A frozen ResNet backbone extracts per-patch embeddings; a patch is scored by its distance to a reference bank of "normal driving" embeddings. Evaluated at pixel resolution against [Lost & Found](https://www.6d-vision.com/lostandfounddataset)'s real, pixel-level hazard ground truth.

**RbA — Rejected by All (secondary investigation).** A frozen Mask2Former segmentation model's own per-class confidence is repurposed as an anomaly signal: a pixel is anomalous if every known class rejects it. Evaluated in two configurations — zero-shot on a vanilla Cityscapes checkpoint, and against the RbA authors' own [officially released checkpoint](https://github.com/NazirNayal8/RbA) (Swin-B, actually fine-tuned with COCO outlier-exposure supervision) to separate "wrong checkpoint" from "wrong technique."

Both methods are evaluated identically: same scene-level train/test split (no adjacent-frame leakage), same held-out hazard frames, AUROC/AUPR at pixel level, and a localization hit-rate metric (does the model's flagged region actually touch the real hazard, not just rank it highly somewhere in the frame).

## Results

| Method | Pixel AUROC | Pixel AUPR | Localization |
|---|---|---|---|
| **kNN patch-embedding (primary)** | **0.9439** | **0.0830** | top-5 hit rate: 60% (18/30) |
| RbA, zero-shot (vanilla Cityscapes ckpt) | 0.7917 | 0.0065 | region hit rate: 23% (7/30) |
| RbA, official (Swin-B, COCO outlier-exposure ckpt) | 0.8750 | 0.0113 | region hit rate: 17% (5/30) |

All numbers from the identical held-out split (30 real hazard frames, Lost & Found).

**The finding worth stating plainly: the simpler baseline wins.** RbA's global ranking quality improved with proper calibration (AUROC 0.79 → 0.875), but its actual hazard localization did not — and the same boundary/edge artifact showed up even with the authors' own properly fine-tuned checkpoint, pointing to a receptive-field property of the architecture rather than a calibration gap. Full investigation, including the root-cause diagnosis and the cross-framework (Detectron2, CPU-only) integration used to rule out the checkpoint as the cause, is written up in [RBA_FINDINGS.md](RBA_FINDINGS.md).

## Repository structure

```
av-corner-case-detector/
├── README.md
├── RBA_FINDINGS.md            # Deep-dive writeup of the RbA investigation
├── requirements.txt
├── src/
│   ├── data/                  # Scene-level splitting (leakage-proof by construction), nuScenes loading
│   ├── eval/                  # AUROC, AUPR metrics
│   └── scoring/                # kNN/Mahalanobis patch scorer, RbA scorer, embedding scorers
├── scripts/
│   ├── evaluate_patch_localization.py       # Primary method: patch-level kNN vs real ground truth
│   ├── evaluate_lost_and_found.py           # Frame-level eval on Lost & Found
│   ├── evaluate_synthetic_benchmark.py      # Same-domain synthetic-hazard honesty check
│   ├── evaluate_rba_lost_and_found.py       # RbA, zero-shot checkpoint
│   ├── evaluate_rba_official_lost_and_found.py  # RbA, official outlier-exposure checkpoint
│   ├── evaluate_roadanomaly21.py
│   ├── build_demo_frames.py / build_demo_video.py   # Alert-style demo generation
│   ├── build_nuisance_classifier.py         # Cone/bollard false-positive suppression
│   ├── build_pooled_reference.py            # Multi-source reference bank
│   ├── run_demo_coda*.py / run_demo_video_rba.py / run_demo_roadanomaly21.py
│   ├── debug_rba_*.py, inspect_hit_candidates.py, tune_knn_k.py  # Diagnostic scripts behind the findings above
│   └── extract_hazard_crops.py, download_*.sh
├── simple_demo/                # Standalone YOLO + synced camera/LiDAR visualization (no anomaly scoring)
├── external/                   # Setup script + adapter for the official RbA checkpoint (not vendored — see below)
└── notebooks/                  # Dataset exploration
```

## Setup

**Requirements**
- Python 3.10+
- PyTorch 2.0+ (MPS support for Apple Silicon)
- See `requirements.txt`

```bash
git clone https://github.com/jayantrathi/av-corner-case-detector.git
cd av-corner-case-detector
pip install -r requirements.txt
```

**Data.** Lost & Found, CODA, and nuScenes v1.0-mini are used for evaluation; each has its own registration/download process (see `scripts/download_*.sh` for Lost & Found and RoadAnomaly21).

**Official RbA checkpoint (optional).** `evaluate_rba_official_lost_and_found.py` requires the RbA authors' own Detectron2-based checkpoint, not vendored in this repo (it's ~700MB across Detectron2 + the RbA codebase + weights). Run `external/setup_rba_official.sh` to fetch and patch it for CPU-only inference — see the script's own comments for the (small) CPU compatibility patch it applies.

## Usage

```bash
# Primary method: kNN patch localization against real ground truth
python scripts/evaluate_patch_localization.py

# RbA comparison (zero-shot)
python scripts/evaluate_rba_lost_and_found.py

# RbA comparison (official checkpoint — run external/setup_rba_official.sh first)
python scripts/evaluate_rba_official_lost_and_found.py

# Demo video: alert box locking onto a flagged region across a real approach sequence
python scripts/build_demo_video.py
```

## Design decisions

**Scene-level data splits.** Adjacent video frames are near-duplicates; a frame-level split leaks the same scene across train and test and inflates every metric. Splits here are grouped by scene/drive, enforced in `src/data/splits.py` and tested in `scripts/test_splits.py`.

**Proper OOD metrics.** AUROC and AUPR, not accuracy — anomalies are rare, so a detector that flags nothing scores ~99% "accurate" while being useless. AUPR in particular is the metric that reflects whether hazard pixels actually rank above the normal-pixel majority.

**Real ground truth, not synthetic-only.** `evaluate_synthetic_benchmark.py` exists specifically because an early cross-dataset comparison (nuScenes-normal vs. CODA-anomalous) produced a suspiciously high AUROC that turned out to be partly measuring "which dataset is this" rather than "is this dangerous" — different camera, city, and compression between the two sources. The synthetic same-domain benchmark and the real Lost & Found pixel-ground-truth evaluation both exist to close that gap honestly.

**No hard exclusion zones without evidence.** The eligible-region masking used in the RbA scripts (border margin, dataset-specific exclusions) is derived from measured peak-position statistics (`debug_rba_peak_locations.py`), not guessed margins — an earlier version guessed twice and both times the hit rate stayed at exactly 0.

## Limitations and future work

- RbA's border/edge artifact persisted even with the properly-calibrated official checkpoint — worth investigating whether it's fixable with a different local-scoring transform, or is a hard limit of this architecture family for this task.
- The kNN baseline's localization hit rate (60% top-5) is solid but not production-grade; a natural next step is ensembling the kNN localization strength with RbA's improved (post-calibration) global ranking.
- Single-camera image-based detection only. `simple_demo/` includes a synced camera+LiDAR visualization pipeline, but LiDAR is currently geometry-only (no anomaly scoring on the point cloud yet).
- Not evaluated in a closed-loop driving system; this is an offline perception-layer monitor.

## References

- Lost & Found dataset: https://www.6d-vision.com/lostandfounddataset
- CODA corner-case dataset: https://github.com/adasfa1se/CODA
- nuScenes: https://www.nuscenes.org/
- RbA: Segmenting Unknown Regions Rejected by All (Nayal et al., ICCV 2023) — https://github.com/NazirNayal8/RbA
- Mahalanobis Distance for Out-of-Distribution Detection (Lee et al., 2018)

---

**Built by** Jay Rathi
**Contact** jayant12rathi@gmail.com
