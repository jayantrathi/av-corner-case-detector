# AV Corner-Case Detector

Out-of-distribution detector for AV perception: flags driving scenes that look unusual relative to normal driving, since standard detectors don't know what they don't know — they're confident even when wrong on something they've never seen.

## Approach

Two methods, evaluated on the same held-out split of real hazard data (Lost & Found):

- **kNN patch-embedding scorer** — frozen ResNet features, patches scored by distance to a "normal driving" reference bank.
- **RbA (Rejected by All)** — a frozen Mask2Former's own per-class confidence repurposed as an anomaly signal. Tried zero-shot and with the authors' officially released, outlier-exposure-finetuned checkpoint.

## Results

| Method | AUROC | AUPR | Localization hit rate |
|---|---|---|---|
| **kNN patch-embedding** | **0.94** | **0.083** | 60% (top-5) |
| RbA, zero-shot | 0.79 | 0.0065 | 23% |
| RbA, official checkpoint | 0.875 | 0.0113 | 17% |

The simpler kNN baseline wins. RbA's ranking improved with the properly-calibrated checkpoint but its localization didn't — same edge/boundary artifact showed up even with the authors' own weights, suggesting it's architectural rather than a calibration issue. Full writeup: [RBA_FINDINGS.md](RBA_FINDINGS.md).

## Running it

```bash
pip install -r requirements.txt
python scripts/evaluate_patch_localization.py       # primary method
python scripts/evaluate_rba_lost_and_found.py        # RbA, zero-shot
python scripts/evaluate_rba_official_lost_and_found.py  # RbA, official checkpoint (run external/setup_rba_official.sh first)
```

## Notes

- Scene-level train/test splits throughout — frame-level splits leak near-duplicate frames and inflate metrics.
- AUROC/AUPR, not accuracy, since anomalies are rare.
- `simple_demo/` has a standalone YOLO + synced camera/LiDAR visualization (no anomaly scoring, just a sanity-check pipeline).

## Limitations

Single-camera only, offline evaluation (not closed-loop). RbA's localization gap is unresolved. LiDAR side is geometry-only, no detector yet.

---
Jay Rathi — jayant12rathi@gmail.com
