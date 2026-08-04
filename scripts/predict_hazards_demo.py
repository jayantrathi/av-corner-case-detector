"""The actual deployable prediction pipeline -- and the demo that shows it.

This is NOT an evaluation mask. It's the model's real output pipeline, the
thing that would run on live footage, composed from pieces we've now proven
work:

  1. ROAD-FIND: the same Mask2Former predicts "road" (Cityscapes class 0) --
     the drivable corridor, no ground truth used.
  2. CLEAN the corridor so the output isn't dominated by non-hazard firing:
       - exclude the ego-hood band (bottom of frame -- the car's own body,
         which the model mislabels as road; standard practice in real AV
         stacks is to mask the ego vehicle)
       - fill small holes so a hazard sitting IN the road (a hole in the road
         region) is kept
       - erode inward a few px to kill the road/non-road boundary seam, where
         the frozen backbone spuriously fires (receptive-field edge effect)
  3. SCORE: RbA anomaly, restricted to that clean corridor.
  4. ALERT: connected-component the top-scoring pixels; the largest few
     components are the flagged hazard candidates. Draw them as boxes on the
     real frame, with the ground-truth hazard for comparison.

Outputs annotated frames (the demo) AND an AUPR within the clean corridor as
a validation number. The spatial prior ("only flag obstacles on the drivable
road, not the hood/sky/edges") is part of the PREDICTION, not a grading trick
-- it changes what the system actually outputs on screen.

Known residual, stated honestly: painted road markings and manhole covers lie
INSIDE the road and are flat (coplanar with the surface), so this mono-camera
pipeline can still flag them. Separating flat-paint from a real 3D object is
what the original Lost & Found used stereo depth for -- out of scope here,
documented as future work.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

CITYSCAPES_ROAD_CLASS = 0
HOOD_BAND_FRAC = 0.87        # rows below this fraction of height = ego-hood, excluded
# The Mercedes hood ORNAMENT (chrome star) pokes up above the hood band into the road,
# and it's a small non-road object, so RbA (correctly!) flags it every frame. A TIGHT
# mask over it isn't enough: the star throws a broad "halo" of elevated anomaly, so a
# tight mask just relocates the peak to the halo's edge (same mask-edge whack-a-mole
# seen earlier in the project). Fix: a GENEROUS ego wedge that swallows star+halo, then
# DILATE the whole ego exclusion so the road is pulled back off the seam. We own the
# ego vehicle and the road ~1-2m directly ahead is too close to act on anyway, so a
# generous ego mask is standard and costs nothing real.
EGO_ORNAMENT_ROW = (0.74, 1.00)
EGO_ORNAMENT_COL = (0.36, 0.64)
EGO_DILATE_ITERS = 18       # grow the ego exclusion so road pulls back off the halo/seam
ROAD_ERODE_ITERS = 10       # shrink the corridor inward to kill the road/non-road seam
ALERT_TOP_PERCENTILE = 99.0 # top 1% of in-corridor anomaly = candidate hazard pixels
MAX_ALERTS = 3              # draw at most this many candidate regions per frame
MIN_ALERT_AREA = 40        # ignore specks smaller than this (px)
OUT_DIR = RESULTS_DIR / "hazard_alert_demo"


def clean_road_corridor(logits: np.ndarray) -> np.ndarray:
    """Predicted drivable road, cleaned into the region the system will
    actually search: hood excluded, holes filled (keep in-road hazards),
    boundary eroded inward (kill the seam artifact)."""
    pred = logits.argmax(axis=0).astype(np.uint8)
    road = (pred == CITYSCAPES_ROAD_CLASS).astype(np.uint8) * 255
    road = np.array(Image.fromarray(road).resize(CANVAS_SIZE, Image.Resampling.NEAREST)) > 127
    road = ndimage.binary_fill_holes(road)
    h, w = road.shape
    # Build the ego-vehicle exclusion (hood band + ornament wedge) as one mask,
    # dilate it generously so it engulfs the ornament's anomaly halo, then pull the
    # road back from it -- so no alert peak can sit on the mask seam.
    ego = np.zeros_like(road)
    ego[int(h * HOOD_BAND_FRAC):, :] = True
    ego[int(h * EGO_ORNAMENT_ROW[0]):int(h * EGO_ORNAMENT_ROW[1]),
        int(w * EGO_ORNAMENT_COL[0]):int(w * EGO_ORNAMENT_COL[1])] = True
    ego = ndimage.binary_dilation(ego, iterations=EGO_DILATE_ITERS)
    road &= ~ego
    if ROAD_ERODE_ITERS:
        road = ndimage.binary_erosion(road, iterations=ROAD_ERODE_ITERS)
    return road


def alert_regions(rba_map, corridor):
    """Top-percentile anomaly pixels inside the corridor, grouped into
    connected components, returned largest-peak-first (up to MAX_ALERTS)."""
    if not corridor.any():
        return []
    thr = np.percentile(rba_map[corridor], ALERT_TOP_PERCENTILE)
    binary = (rba_map >= thr) & corridor
    labeled, n = ndimage.label(binary)
    regions = []
    for cid in range(1, n + 1):
        comp = labeled == cid
        if comp.sum() < MIN_ALERT_AREA:
            continue
        regions.append((float(rba_map[comp].max()), comp))
    regions.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in regions[:MAX_ALERTS]]


def load_font(sz):
    for p in ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_demo(image_path, regions, hazard_mask, out_path, font):
    img = Image.open(image_path).convert("RGB").resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    # ground-truth hazard in green
    edge = hazard_mask ^ ndimage.binary_erosion(hazard_mask, iterations=2)
    ys, xs = np.where(edge)
    for y, x in zip(ys, xs):
        draw.point((x, y), fill=(0, 255, 0))
    # model's alerts in red boxes
    for comp in regions:
        ys, xs = np.where(comp)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        pad = 6
        draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=(255, 40, 40), width=3)
    draw.rectangle([8, 8, 300, 34], fill=(15, 15, 15))
    draw.text((14, 12), f"{len(regions)} hazard alert(s) on road", font=font, fill=(255, 90, 90))
    img.save(out_path, quality=92)


def main():
    _, test_hazard = load_test_split()
    print(f"Held-out hazard frames: {len(test_hazard)}\n")
    print("Loading OFFICIAL RbA (Swin-B)...")
    scorer = OfficialRbAScorer(device="cpu")
    font = load_font(18)
    print("Loaded.\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scores, labels = [], []
    total_haz, covered_haz = 0, 0
    alert_hits, frames_with_alert, n = 0, 0, 0

    for i, (path, scene, _) in enumerate(test_hazard):
        mask = load_mask_resized(img_path_to_label_path(path))
        if mask is None:
            continue
        hazard = (mask == HAZARD_TRAIN_ID)
        if not hazard.any():
            continue
        n += 1

        img = Image.open(path).convert("RGB")
        rba_map, logits = scorer.score(img, out_size=CANVAS_SIZE)
        corridor = clean_road_corridor(logits)

        total_haz += int(hazard.sum())
        covered_haz += int((hazard & corridor).sum())

        # validation metric: AUPR within the clean corridor
        if corridor.any():
            scores.append(rba_map[corridor])
            labels.append((mask[corridor] == HAZARD_TRAIN_ID).astype(np.int8))

        regions = alert_regions(rba_map, corridor)
        if regions:
            frames_with_alert += 1
            if any((c & hazard).any() for c in regions):
                alert_hits += 1

        draw_demo(path, regions, hazard, OUT_DIR / f"alert_{n:02d}.jpg", font)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(test_hazard)}", flush=True)

    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    m = summarize(scores, labels)
    coverage = covered_haz / total_haz if total_haz else float("nan")
    alert_hit_rate = alert_hits / n if n else float("nan")

    print("\n" + "=" * 60)
    print("COMPOSED PREDICTION PIPELINE  (road-find -> clean -> score -> alert)")
    print("=" * 60)
    print(f"Hazard coverage (in clean corridor):     {covered_haz:,}/{total_haz:,} = {coverage:.3f}")
    print(f"Alert hit rate (a drawn box touches GT):  {alert_hits}/{n} = {alert_hit_rate:.3f}")
    print(f"AUPR within clean corridor (validation):  {m['AUPR']:.4f}")
    print(f"AUROC:                                    {m['AUROC']:.4f}")
    print(f"FPR@95:                                   {m['FPR@95']:.4f}")
    print("=" * 60)
    print("Reference: GT-road ROI benchmark AUPR 0.7787 (evaluate_rba_roi_standard.py).")
    print("Alert hit rate = the deployable localization number: on what fraction of")
    print("frames does the system draw a box that lands on the real hazard.")
    print(f"\nDemo frames -> {OUT_DIR}")

    with open(RESULTS_DIR / "hazard_alert_demo_results.json", "w") as f:
        json.dump({**m, "hazard_coverage": coverage, "alert_hit_rate": alert_hit_rate,
                   "frames": n, "alert_hits": alert_hits}, f, indent=2)


if __name__ == "__main__":
    main()
