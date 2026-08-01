# AV Corner-Case Detector

A perception-layer safety monitor that detects out-of-distribution driving scenes. These corner cases are situations where an autonomous vehicle's perception model should not be trusted.

## The Problem

Autonomous driving stacks fail in situations they were never trained on: debris in the lane, wrong-way vehicles, unusual weather, sensor degradation. Simply running a perception model on a corner case doesn't tell you it's failing. The model is confident and wrong.

This project builds a monitor that flags when the driving scene is anomalous relative to the normal distribution the model learned from. A high anomaly score indicates "this situation is unusual and the autonomy stack should not be trusted here."

## Approach

**Primary Method: Embedding-based Detection**

The core approach extracts frame embeddings using a frozen pretrained vision backbone such as ResNet, DINO, or CLIP. We model the feature distribution of normal driving scenes, then score new frames by their distance from the normal feature cloud. Two scoring methods work well: k-NN distance and Mahalanobis distance.

**Baseline: Reconstruction-based Detection**

We also train a convolutional autoencoder on normal frames and score anomalies by reconstruction error. This method is weaker than the embedding approach but provides excellent visualization through error heatmaps that show exactly where the anomaly appears.

Both approaches use the same rigorous evaluation strategy:
- Split data by scene or drive sequence, never by individual frame, to prevent data leakage
- Evaluate using AUROC and AUPR metrics designed for anomaly detection, not accuracy since anomalies are rare
- Test on held-out normal frames plus known corner cases

## Results

Work in progress. Metrics table and demo video coming after Phase 3 evaluation.

## Repository Structure

```
av-corner-case-detector/
├── README.md                  # This file
├── DEVELOPMENT.md             # Internal build guide
├── requirements.txt           # Dependencies
├── src/
│   ├── data/splits.py        # Scene-level splitting prevents leakage
│   ├── eval/metrics.py       # AUROC, AUPR, FPR at 95% recall
│   ├── scoring/              # k-NN and Mahalanobis scorers
│   ├── models/               # Autoencoder architecture
│   └── viz/                  # Heatmaps and score over time plots
├── scripts/
│   ├── train.py             # Train the autoencoder
│   ├── evaluate.py          # Compute metrics
│   └── run_demo.py          # Generate demo video
├── notebooks/                # Exploration and data sanity checks
└── configs/                  # YAML experiment configs
```

## Setup

**Requirements**
- Python 3.10 or later
- PyTorch 2.0 or later with MPS support for Apple Silicon
- See requirements.txt for full list

**Installation**

```bash
git clone https://github.com/yourusername/av-corner-case-detector.git
cd av-corner-case-detector
pip install -r requirements.txt
```

**Data**

Download BDD100K or nuScenes. Both are free for research with registration. Place the data in the data folder.

## Usage

Examples coming soon. See notebooks for exploration code.

## Demo

A 90-second video showing anomaly scores spiking on corner-case frames, with reconstruction error heatmaps highlighting what triggered each detection.

## Design Decisions

**Scene-level Data Splits**

Adjacent video frames are nearly identical. If you split at the frame level, the same scene lands on both train and test. The model effectively sees test data during training, and metrics become meaningless. This project groups entire scenes atomically into splits.

**Frozen Backbone**

The primary method uses a pretrained vision encoder with only the scoring layer tuned. This approach runs efficiently on CPU/MPS without requiring a GPU, enabling fast iteration on a laptop.

**Proper OOD Metrics**

We use AUROC and AUPR, not accuracy. Anomaly detection datasets are imbalanced; accuracy is misleading since a detector that flags nothing scores roughly 99% accurate.

**Dual Approaches**

The embedding method represents the state-of-the-art approach. The reconstruction method is slower but produces heatmap visualizations that make the demo clear and compelling.

## Limitations and Future Work

**Current Scope**

This work focuses on single-camera image-based detection. Multi-sensor fusion with LiDAR and radar is future work.

**Evaluation**

The system demonstrates OOD detection on a diverse corner-case set. It does not evaluate on a full closed-loop autonomous driving system.

**Real-time Performance**

The current implementation is not yet optimized for onboard inference, though this is feasible with embedding caching.

## References

- nuScenes: https://www.nuscenes.org/
- BDD100K: https://bdd-data.berkeley.edu/
- CODA corner-case dataset: https://github.com/Robin-CC/CODA
- Mahalanobis Distance for Out-of-Distribution Detection (Lee et al., 2018)
- k-NN Distance: standard baseline in anomaly detection literature

---

**Status** In progress (Phase 2 as of July 2026). Metrics and demo video expected in August.

**Built by** Jay Rathi  
**Contact** jayant12rathi@gmail.com
