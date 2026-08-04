"""YOLO-gated anomaly localization -- the combined system the project was
always about, and a direct fix for the observed failure mode.

THE DIAGNOSIS this script acts on: when the RbA anomaly peak misses the real
hazard, it doesn't miss randomly -- it lands on objects a closed-set
detector already knows (parked cars, pedestrians near a van, etc., visible
in the example_*.jpg misses). That's not the anomaly signal being wrong: a
weird-looking car genuinely IS locally unusual. It's us grading the anomaly
detector in isolation when the whole thesis is that a road hazard is defined
by being the thing the closed-set detector has NO box for.

THE FIX: gate the anomaly map with YOLO. Suppress anomaly wherever there's a
confident YOLO detection (already explained by closed-set perception); the
surviving peak is anomalous AND unrecognized -- which is exactly what a road
hazard is. Uses only components already in the repo (RbA official scorer +
YOLOv8, integrated earlier). This is the combined detector, not a new model.

Reports the localization hit rate under four escalating conditions on the
SAME held-out 30 Lost & Found hazard frames, so the contribution of each
piece is visible and honest:
  1. raw peak (no masking)                        -- the 0/30 starting point
  2. + border margin (edge/receptive-field mask)
  3. + local contrast (broad-gradient suppression)
  4. + YOLO gating (suppress known-object regions) -- the new lever

Also reports top-3 hit rate for the full-stack version: does the real hazard
fall among the model's top-3 surviving candidate regions? That's the metric
a safety monitor is actually judged on ("flag regions to treat with
caution"), not single-pixel argmax.
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
from evaluate_rba_lost_and_found import (
    load_test_split, border_eligible_mask, local_contrast_map,
)
from rba_official_scorer import OfficialRbAScorer

try:
    from ultralytics import YOLO
except ImportError:
    print("Missing ultralytics. Run: pip install ultralytics")
    raise SystemExit(1)

YOLO_WEIGHTS = "/Volumes/BIggen/AV/yolov8n.pt"
YOLO_CONF = 0.30       # confident-enough detections to treat as "explained"
YOLO_PAD_FRAC = 0.04   # dilate each suppressed box slightly, so the peak can't
                       # sit one pixel outside a known object and still count
TOP_K = 3


def yolo_suppress_mask(result, shape) -> np.ndarray:
    """True where a confident YOLO box says 'known object here' -- these
    regions are explained by closed-set perception and should NOT be where
    we look for an unknown hazard. Boxes are scaled from original image
    coords to the CANVAS_SIZE the anomaly map lives in, and padded slightly."""
    h, w = shape
    suppress = np.zeros(shape, dtype=bool)
    sx, sy = w / result.orig_shape[1], h / result.orig_shape[0]
    pad_x, pad_y = int(w * YOLO_PAD_FRAC), int(h * YOLO_PAD_FRAC)
    for box in result.boxes:
        if float(box.conf[0]) < YOLO_CONF:
            continue
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        x0, x1 = int(x0 * sx) - pad_x, int(x1 * sx) + pad_x
        y0, y1 = int(y0 * sy) - pad_y, int(y1 * sy) + pad_y
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        suppress[y0:y1, x0:x1] = True
    return suppress


def peak_hits(score_map, eligible, hazard_mask) -> bool:
    """Does the argmax over the eligible region land on a real hazard pixel?"""
    masked = np.where(eligible, score_map, -np.inf)
    peak = np.unravel_index(np.argmax(masked), score_map.shape)
    return bool(hazard_mask[peak])


def topk_region_hits(score_map, eligible, hazard_mask, k=TOP_K) -> bool:
    """Threshold to the top slice, connected-component label it, take the k
    largest components by peak score, and check whether ANY of them touches
    a real hazard pixel. This is the 'flag candidate regions' metric."""
    vals = score_map[eligible]
    if vals.size == 0:
        return False
    thresh = np.percentile(vals, 99.0)  # top ~1% of eligible pixels
    binary = (score_map >= thresh) & eligible
    labeled, n = ndimage.label(binary)
    if n == 0:
        return False
    # rank components by their peak score, keep top k
    comp_peak = []
    for cid in range(1, n + 1):
        comp = labeled == cid
        comp_peak.append((score_map[comp].max(), cid))
    comp_peak.sort(reverse=True)
    for _, cid in comp_peak[:k]:
        if ((labeled == cid) & hazard_mask).any():
            return True
    return False


def main():
    _, test_hazard = load_test_split()
    print(f"Held-out hazard frames: {len(test_hazard)} (same split as every other run)\n")

    print("Loading OFFICIAL RbA (Swin-B) + YOLOv8n...")
    scorer = OfficialRbAScorer(device="cpu")
    yolo = YOLO(YOLO_WEIGHTS)
    print("Loaded.\n")

    n = 0
    hits_raw = hits_border = hits_contrast = hits_gated = 0
    topk_gated = 0

    for path, scene, _ in test_hazard:
        label_path = img_path_to_label_path(path)
        mask = load_mask_resized(label_path)
        if mask is None:
            continue
        hazard_mask = (mask == HAZARD_TRAIN_ID)
        if not hazard_mask.any():
            continue
        n += 1

        img = Image.open(path).convert("RGB")
        rba_map, _ = scorer.score(img, out_size=CANVAS_SIZE)
        contrast = local_contrast_map(rba_map)

        full = np.ones(rba_map.shape, dtype=bool)
        border = border_eligible_mask(rba_map.shape)

        yolo_res = yolo.predict(str(path), conf=YOLO_CONF, verbose=False)[0]
        suppress = yolo_suppress_mask(yolo_res, rba_map.shape)
        gated_eligible = border & ~suppress

        hits_raw += peak_hits(rba_map, full, hazard_mask)
        hits_border += peak_hits(rba_map, border, hazard_mask)
        hits_contrast += peak_hits(contrast, border, hazard_mask)
        hits_gated += peak_hits(contrast, gated_eligible, hazard_mask)
        topk_gated += topk_region_hits(contrast, gated_eligible, hazard_mask)

    def pct(x):
        return f"{x}/{n} = {x / n:.4f}" if n else "n/a"

    print("=" * 66)
    print("YOLO-GATED LOCALIZATION -- escalating conditions, same 30 frames")
    print("=" * 66)
    print(f"1. raw peak, no masking:                 {pct(hits_raw)}")
    print(f"2. + border margin:                      {pct(hits_border)}")
    print(f"3. + local contrast:                     {pct(hits_contrast)}")
    print(f"4. + YOLO gating (THE new lever):        {pct(hits_gated)}")
    print(f"\nTop-{TOP_K} region hit rate (full stack): {pct(topk_gated)}")
    print("  ^ 'is the real hazard among the model's top candidate regions' --")
    print("    the metric a safety monitor is actually judged on, vs single-pixel argmax.")

    results = {
        "n_frames": n,
        "hits_raw": hits_raw,
        "hits_border": hits_border,
        "hits_contrast": hits_contrast,
        "hits_yolo_gated": hits_gated,
        f"top{TOP_K}_gated": topk_gated,
        "yolo_conf": YOLO_CONF,
    }
    out = RESULTS_DIR / "yolo_gated_localization_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
