# AV Corner-Case Detector

A perception-layer safety monitor that detects out-of-distribution driving scenes — the corner cases where an autonomous vehicle's perception model should not be trusted.

## The Problem

Autonomous driving stacks fail in situations they were never trained on: debris in the lane, wrong-way vehicles, unusual weather, sensor degradation. Simply running a perception model on a corner case doesn't tell you it's failing — the model is confident and wrong.

This project builds a monitor that flags when the driving scene is anomalous relative to the normal distribution the model learned from. High anomaly score = "this situation is weird; don't trust the autonomy stack here."

## Approach

**Primary method (Approach B — Embedding-based):**
- Extract frame embeddings using a frozen pretrained vision backbone (ResNet / DINO / CLIP)
- Model the feature distribution of normal driving scenes
- Score new frames by their distance from the normal feature cloud
- k-NN distance and Mahalanobis distance are the two scorers

**Baseline (Approach A — Reconstruction):**
- Train a convolutional autoencoder on normal frames
- Score frames by reconstruction error
- Weaker than Approach B but provides excellent visualization (error heatmaps show *where* the anomaly is)

Both approaches:
- Split data by scene/drive (never by frame) to prevent data leakage
- Use AUROC and AUPR for evaluation (not accuracy — anomalies are rare)
- Evaluate on held-out normal frames plus known corner cases

## Results

*In progress. Metrics coming after Phase 3.*

## Repository Structure

```
av-corner-case-detector/
├── README.md                  # This file
├── requirements.txt           # Dependencies
├── src/
│   ├── data/splits.py        # Scene-level splitting (leakage-proof)
│   ├── eval/metrics.py       # AUROC, AUPR, FPR@95 (OOD metrics)
│   ├── scoring/              # k-NN and Mahalanobis scorers
│   ├── models/               # Autoencoder architecture
│   └── viz/                  # Heatmaps and score-over-time plots
├── scripts/
│   ├── train.py             # Train the autoencoder
│   ├── evaluate.py          # Compute metrics
│   └── run_demo.py          # Generate demo video
├── notebooks/                # Exploration and data sanity checks
└── configs/                  # YAML experiment configs
```

## Setup

**Requirements:**
- Python 3.10+
- PyTorch 2.0+ with MPS support (Apple Silicon)
- See `requirements.txt`

**Install:**
```bash
git clone https://github.com/yourusername/av-corner-case-detector.git
cd av-corner-case-detector
pip install -r requirements.txt
```

**Data:**
Download BDD100K or nuScenes and place under `data/`. Both are free for research with registration.

## Usage

*Coming soon. See notebooks/ for exploration examples.*

## Demo

*90-second video TK* — shows anomaly score spiking on corner-case frames, with reconstruction error heatmaps highlighting what triggered the detection.

## Key Design Decisions

1. **Scene-level splits.** Adjacent video frames are near-duplicates. Splitting at the frame level leaks test data into training. This project groups entire scenes atomically.

2. **Frozen backbone.** The primary method uses a pretrained vision encoder and only tunes the scoring layer. This runs on CPU/MPS without GPU and lets you iterate fast.

3. **Real OOD metrics.** AUROC and AUPR, not accuracy. Anomaly detection datasets are imbalanced; accuracy is misleading (a detector that flags nothing scores ~99%).

4. **Dual approaches.** Approach B is state-of-the-art. Approach A is slower but provides the heatmap visualization that makes the demo clear.

## Limitations & Future Work

- **Current scope:** single-camera image-based detection. Multi-sensor fusion (LiDAR, radar) is future work.
- **Evaluation:** demonstrates OOD detection on a diverse corner-case set. Does not evaluate on a full closed-loop autonomous driving system.
- **Real-time performance:** not yet optimized for onboard inference; feasible with embedding caching.

## References

- nuScenes: https://www.nuscenes.org/
- BDD100K: https://bdd-data.berkeley.edu/
- CODA (corner-case dataset): https://github.com/Robin-CC/CODA
- Mahalanobis distance for OOD detection: Lee et al., 2018
- k-NN distance baseline: standard in anomaly detection literature

---

**Status:** In progress (Phase 2 as of July 2026). Metrics table and demo video coming in August.

**Built by:** Jay Rathi  
**Contact:** jayant12rathi@gmail.com
