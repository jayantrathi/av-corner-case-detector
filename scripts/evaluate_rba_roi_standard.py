"""THE reconciliation test: does restricting evaluation to the drivable-road
ROI reconcile our RbA numbers with the published literature?

The problem this exists to settle: the RbA paper reports AP ~0.70 on
Fishyscapes Lost & Found. Our runs reported AUPR ~0.01 on the same method
and data -- a 70x gap that almost certainly means our EVALUATION differs
from the field's, not that the method is 70x worse. The prime suspect:
we've been scoring the whole frame, including the sky / image border /
ego-hood, which is exactly where the model spuriously fires (the solid-red
bands in every example image). Standard road-anomaly benchmarks don't score
those regions -- they evaluate inside a region of interest: the drivable
road surface plus the hazard objects on it.

Lost & Found's own ground truth gives us that ROI directly:
  trainId 2 = hazard object   -> POSITIVE
  trainId 1 = free/drivable road -> NEGATIVE (in-distribution)
  trainId 0 = background/void/ego/sky -> IGNORED (outside ROI)

So the ROI evaluation is: positives = hazard pixels, negatives = road
pixels, everything else excluded. That's the standard protocol, and it's
precisely the region where the border artifact does NOT live.

This script computes AUPR / AUROC / FPR@95 two ways in one pass:
  A. FULL FRAME  (what we've been doing -- positives=hazard, negatives=all else)
  B. ROI ONLY    (the standard -- positives=hazard, negatives=road only)

If (B)'s AUPR jumps from ~0.01 toward the published ~0.70, the whole night's
"the method doesn't work" conclusion was an evaluation artifact, not the
model. If it doesn't, our integration has a real bug and every conclusion
rested on a broken baseline -- also essential to know.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/Volumes/BIggen/AV/external")
sys.path.insert(0, "/Volumes/BIggen/AV/external/RbA")

from evaluate_patch_localization import (
    RESULTS_DIR, HAZARD_TRAIN_ID, CANVAS_SIZE,
    img_path_to_label_path, load_mask_resized,
)
from evaluate_rba_lost_and_found import load_test_split
from src.eval.metrics import summarize
from rba_official_scorer import OfficialRbAScorer

ROAD_TRAIN_ID = 1  # free/drivable road -- the in-distribution negatives (confirmed:
                   # evaluate_lost_and_found.py docstring, trainId 1 = free/drivable road)


def main():
    test_normal, test_hazard = load_test_split()
    frames = test_normal + test_hazard
    print(f"Test frames: {len(test_normal)} normal + {len(test_hazard)} hazard = {len(frames)}\n")

    print("Loading OFFICIAL RbA (Swin-B, COCO outlier supervision)...")
    scorer = OfficialRbAScorer(device="cpu")
    print("Loaded.\n")

    full_scores, full_labels = [], []
    roi_scores, roi_labels = [], []

    label_hist = {}  # sanity: confirm road(1)/hazard(2) trainIds actually present

    for idx, (path, scene, is_hazard) in enumerate(frames):
        label_path = img_path_to_label_path(path)
        mask = load_mask_resized(label_path)
        if mask is None:
            continue

        img = Image.open(path).convert("RGB")
        rba_map, _ = scorer.score(img, out_size=CANVAS_SIZE)

        flat_scores = rba_map.flatten()
        flat_mask = mask.flatten()

        for v in np.unique(flat_mask):
            label_hist[int(v)] = label_hist.get(int(v), 0) + int((flat_mask == v).sum())

        # A. FULL FRAME: positive = hazard, negative = literally everything else
        full_scores.append(flat_scores)
        full_labels.append((flat_mask == HAZARD_TRAIN_ID).astype(np.int8))

        # B. ROI ONLY: keep road + hazard pixels; positive = hazard, negative = road
        roi = (flat_mask == ROAD_TRAIN_ID) | (flat_mask == HAZARD_TRAIN_ID)
        roi_scores.append(flat_scores[roi])
        roi_labels.append((flat_mask[roi] == HAZARD_TRAIN_ID).astype(np.int8))

        if (idx + 1) % 10 == 0:
            print(f"  scored {idx + 1}/{len(frames)}", flush=True)

    full_scores = np.concatenate(full_scores)
    full_labels = np.concatenate(full_labels)
    roi_scores = np.concatenate(roi_scores)
    roi_labels = np.concatenate(roi_labels)

    print("\nLabel pixel histogram (trainId -> pixel count), sanity check:")
    for k in sorted(label_hist):
        tag = {0: "background/void/ego/sky", 1: "road (negative)", 2: "hazard (positive)"}.get(k, "other")
        print(f"  trainId {k}: {label_hist[k]:>12,}  {tag}")

    full = summarize(full_scores, full_labels)
    roi = summarize(roi_scores, roi_labels)

    n_pos_full = int(full_labels.sum())
    n_pos_roi = int(roi_labels.sum())
    print(f"\nFull-frame: {len(full_labels):,} pixels, {n_pos_full:,} positive "
          f"({100*n_pos_full/len(full_labels):.4f}%)")
    print(f"ROI only:   {len(roi_labels):,} pixels, {n_pos_roi:,} positive "
          f"({100*n_pos_roi/len(roi_labels):.4f}%)  <- road+hazard only, sky/border/ego excluded")

    print("\n" + "=" * 62)
    print(f"{'metric':<10}{'FULL FRAME (old)':>22}{'ROI ONLY (standard)':>22}")
    print("=" * 62)
    for k in ["AUPR", "AUROC", "FPR@95"]:
        print(f"{k:<10}{full[k]:>22.4f}{roi[k]:>22.4f}")
    print("=" * 62)
    print("\nPublished RbA on Fishyscapes Lost & Found (for reference): AP ~0.70, FPR95 ~0.06")
    print("If ROI-ONLY AUPR is now in that neighborhood, the whole 'RbA is broken'")
    print("conclusion was an evaluation-region artifact -- the method was fine, our")
    print("full-frame metric was scoring the border/ego artifact as false positives.")

    results = {
        "full_frame": full,
        "roi_only": roi,
        "n_pixels_full": len(full_labels),
        "n_pixels_roi": len(roi_labels),
        "n_positive": n_pos_roi,
        "label_histogram": label_hist,
        "published_reference": {"AP": 0.70, "FPR95": 0.06},
    }
    out = RESULTS_DIR / "rba_roi_standard_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
