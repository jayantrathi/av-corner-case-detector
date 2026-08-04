"""Deployable test: swap the GROUND-TRUTH road ROI for the model's OWN
predicted road region -- no labels peeked at inference time.

evaluate_rba_roi_standard.py proved the model's signal is strong (AUPR 0.78)
but it defined the road region from ground-truth labels. That's standard for
BENCHMARKING, but a real system on new footage has no GT road mask. The good
news: the same Cityscapes-trained Mask2Former already predicts "road" as one
of its 19 classes (Cityscapes trainId 0), so we get the road region for free
from the same forward pass -- no new model, no new dependency.

This script uses the model's OWN predicted road as the ROI and re-measures.
The comparison that matters:
    GT-road ROI  (benchmark, evaluate_rba_roi_standard.py):  AUPR 0.7787
    predicted-road ROI (this script, deployable):            AUPR ???
If the predicted-ROI number holds up near 0.78, the system works end-to-end
on a frame with no labels. If it collapses, road-segmentation quality is the
bottleneck, and THAT becomes the next thing to fix.

Also reports HAZARD COVERAGE: the fraction of real hazard pixels that fall
inside the predicted road corridor at all. A hazard the road-finder drops
(segments as non-road, outside the corridor) can't be scored -- that's the
honest failure mode of the deployable version, and we measure it explicitly
rather than hiding it.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

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

ROAD_TRAIN_ID = 1               # GT free/drivable road
CITYSCAPES_ROAD_CLASS = 0       # model's predicted-road class (Cityscapes trainId 0 = road)
ROAD_DILATE_ITERS = 6           # dilate the predicted road slightly so a hazard sitting
                                # right at the road edge isn't dropped by a hairline gap
VIZ_DIR = RESULTS_DIR / "rba_predicted_roi_examples"
N_VIZ = 6


def predicted_road_mask(logits: np.ndarray) -> np.ndarray:
    """Model's own road prediction at CANVAS_SIZE. argmax over the 19
    Cityscapes classes == 0 gives road; resize (nearest) to the anomaly-map
    grid; fill fully-enclosed holes so a hazard surrounded by road (a hole in
    the road region) is included; dilate slightly for edge hazards."""
    pred = logits.argmax(axis=0).astype(np.uint8)          # (H,W) native
    road = (pred == CITYSCAPES_ROAD_CLASS).astype(np.uint8) * 255
    road_img = Image.fromarray(road).resize(CANVAS_SIZE, Image.Resampling.NEAREST)
    road = np.array(road_img) > 127
    road = ndimage.binary_fill_holes(road)                 # engulf hazard-shaped holes
    if ROAD_DILATE_ITERS:
        road = ndimage.binary_dilation(road, iterations=ROAD_DILATE_ITERS)
    return road


def save_viz(image_path, rba_map, road, hazard_mask, out_path):
    img = Image.open(image_path).convert("RGB").resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
    base = img.convert("RGBA")
    over = np.zeros((*rba_map.shape, 4), dtype=np.uint8)

    # blue tint = model's predicted drivable corridor
    over[road, 2] = 120
    over[road, 3] = 70

    # red heat = anomaly score, but ONLY inside the corridor (deployable view)
    lo, hi = np.percentile(rba_map[road], 50), np.percentile(rba_map[road], 99.5)
    norm = np.clip((rba_map - lo) / (hi - lo + 1e-8), 0, 1)
    red = (norm * 255).astype(np.uint8)
    over[road, 0] = red[road]
    over[road, 3] = np.maximum(over[road, 3], (norm[road] * 180).astype(np.uint8))

    comb = Image.alpha_composite(base, Image.fromarray(over, "RGBA"))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(comb)
    edge = hazard_mask ^ ndimage.binary_erosion(hazard_mask, iterations=3)
    ys, xs = np.where(edge)
    for y, x in zip(ys, xs):
        draw.point((x, y), fill=(0, 255, 0, 255))   # green = real hazard
    comb.convert("RGB").save(out_path, quality=92)


def main():
    test_normal, test_hazard = load_test_split()
    frames = test_normal + test_hazard
    print(f"Test frames: {len(test_normal)} normal + {len(test_hazard)} hazard\n")

    print("Loading OFFICIAL RbA (Swin-B)...")
    scorer = OfficialRbAScorer(device="cpu")
    print("Loaded.\n")

    scores, labels = [], []
    total_hazard_px = 0
    covered_hazard_px = 0
    road_frac_sum = 0.0
    n = 0
    viz_saved = 0
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    for idx, (path, scene, is_hazard) in enumerate(frames):
        mask = load_mask_resized(img_path_to_label_path(path))
        if mask is None:
            continue
        img = Image.open(path).convert("RGB")
        rba_map, logits = scorer.score(img, out_size=CANVAS_SIZE)
        road = predicted_road_mask(logits)
        n += 1
        road_frac_sum += road.mean()

        hazard = (mask == HAZARD_TRAIN_ID)
        total_hazard_px += int(hazard.sum())
        covered_hazard_px += int((hazard & road).sum())

        # eval within the predicted corridor: positives = hazard there, negatives = road there
        roi = road
        scores.append(rba_map[roi])
        labels.append((mask[roi] == HAZARD_TRAIN_ID).astype(np.int8))

        if is_hazard and hazard.any() and viz_saved < N_VIZ:
            save_viz(path, rba_map, road, hazard, VIZ_DIR / f"example_{viz_saved:02d}.jpg")
            viz_saved += 1

        if (idx + 1) % 10 == 0:
            print(f"  scored {idx + 1}/{len(frames)}", flush=True)

    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    coverage = covered_hazard_px / total_hazard_px if total_hazard_px else float("nan")

    print(f"\nMean predicted-road area: {100*road_frac_sum/n:.1f}% of frame")
    print(f"Hazard coverage (GT hazard px inside predicted corridor): "
          f"{covered_hazard_px:,}/{total_hazard_px:,} = {coverage:.4f}")
    print(f"  ^ hazards outside the corridor can't be scored -- the honest ceiling on recall")

    m = summarize(scores, labels)
    print("\n" + "=" * 58)
    print(f"{'metric':<10}{'GT-road ROI':>16}{'PREDICTED-road ROI':>22}")
    print("=" * 58)
    ref = {"AUPR": 0.7787, "AUROC": 0.9486, "FPR@95": 0.2933}
    for k in ["AUPR", "AUROC", "FPR@95"]:
        print(f"{k:<10}{ref[k]:>16.4f}{m[k]:>22.4f}")
    print("=" * 58)
    print("\nGT-road column = evaluate_rba_roi_standard.py (benchmark, peeks at GT road).")
    print("Predicted-road column = THIS run, model's own road prediction, no GT peek.")
    print("If they're close, the system works end-to-end on unlabeled frames.")
    print(f"Saved {viz_saved} corridor+anomaly visualizations -> {VIZ_DIR}")

    out = RESULTS_DIR / "rba_predicted_roi_results.json"
    with open(out, "w") as f:
        json.dump({"predicted_roi": m, "gt_roi_reference": ref,
                   "hazard_coverage": coverage, "mean_road_frac": road_frac_sum / n}, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
