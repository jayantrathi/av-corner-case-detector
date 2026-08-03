"""Validate the OFFICIAL RbA checkpoint (Swin-B, Detectron2, actually
fine-tuned with COCO outlier-exposure supervision -- not our vanilla
zero-shot Cityscapes checkpoint) against the SAME held-out Lost & Found
split and SAME metrics as evaluate_rba_lost_and_found.py, so tonight's
numbers are directly comparable to both the old zero-shot RbA run and the
kNN baseline.

Why this script exists as a separate file instead of adding a flag to the
original: the official model lives in a completely different framework
(Detectron2, not HuggingFace transformers) loaded from external/RbA, which
is scaffolding, not project code -- keeping it separate means the vendored
external/ dependency never becomes an import the main pipeline relies on.

Reports THREE numbers for the peak/region metrics, deliberately mirroring
the honesty discipline from the local-contrast run earlier tonight:
  1. RAW peak-pixel hit rate, no masking at all -- tests whether this
     properly-calibrated checkpoint even HAS the border/hood artifact that
     forced margin-masking on the old checkpoint. If it doesn't, that's
     itself an important, reportable finding (the artifact was a symptom of
     using an uncalibrated checkpoint, not a property of RbA-the-technique).
  2. LOCAL-CONTRAST peak hit rate, still no hard masking -- same transform
     as before, for consistency, in case some residual broad-gradient
     artifact remains even with proper calibration.
  3. REGION hit rate: local contrast + a small border margin (reused from
     evaluate_rba_lost_and_found -- NOT the Lost & Found hood box, since
     that was calibrated to the OLD checkpoint's specific artifact and has
     no verified relevance here).
"""
from __future__ import annotations
import sys
import json
import random
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
    REGION_TOP_PERCENTILE, BORDER_MARGIN_FRAC,
)
from src.eval.metrics import auroc, aupr
from rba_official_scorer import OfficialRbAScorer

VIZ_DIR = RESULTS_DIR / "rba_official_lost_and_found_examples"
N_VIZ_EXAMPLES = 8


def region_from_peak_no_hood(rba_map: np.ndarray, top_percentile: float = REGION_TOP_PERCENTILE):
    """Same connected-component-from-peak logic as largest_region_from_peak
    in evaluate_rba_lost_and_found.py, but using ONLY the generic border
    margin -- no Lost & Found hood box, since that exclusion was calibrated
    against the old checkpoint's specific artifact position, not verified
    for this model."""
    eligible = border_eligible_mask(rba_map.shape)
    contrast = local_contrast_map(rba_map)
    masked = np.where(eligible, contrast, -np.inf)
    peak_idx = np.unravel_index(np.argmax(masked), contrast.shape)
    threshold = np.percentile(contrast[eligible], 100 - top_percentile)
    binary = (contrast >= threshold) & eligible
    labeled, _ = ndimage.label(binary)
    region_id = labeled[peak_idx]
    return labeled == region_id, peak_idx


def save_viz(image_path, rba_map, hazard_mask, region_mask, peak_idx, out_path):
    img = Image.open(image_path).convert("RGB").resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
    lo, hi = np.percentile(rba_map, 1), np.percentile(rba_map, 99.5)
    norm = np.clip((rba_map - lo) / (hi - lo + 1e-8), 0, 1)
    heat = (norm * 255).astype(np.uint8)
    heat_rgba = np.zeros((*heat.shape, 4), dtype=np.uint8)
    heat_rgba[..., 0] = 255
    heat_rgba[..., 3] = (heat.astype(np.float32) * 0.55).astype(np.uint8)

    base = img.convert("RGBA")
    combined = Image.alpha_composite(base, Image.fromarray(heat_rgba, mode="RGBA"))

    from PIL import ImageDraw
    draw = ImageDraw.Draw(combined)
    gt_edge = hazard_mask ^ ndimage.binary_erosion(hazard_mask, iterations=3)
    ys, xs = np.where(gt_edge)
    for y, x in zip(ys, xs):
        draw.point((x, y), fill=(0, 255, 0, 255))
    region_edge = region_mask ^ ndimage.binary_erosion(region_mask, iterations=3)
    ys, xs = np.where(region_edge)
    for y, x in zip(ys, xs):
        draw.point((x, y), fill=(0, 255, 255, 255))
    py, px = peak_idx
    draw.ellipse([px - 6, py - 6, px + 6, py + 6], outline=(255, 255, 0, 255), width=3)

    combined.convert("RGB").save(out_path, quality=92)


def main():
    test_normal, test_hazard = load_test_split()
    print(f"Held-out test set: {len(test_normal)} normal, {len(test_hazard)} hazard "
          f"(same split as k-NN v3 and the earlier zero-shot RbA run)\n")

    print("Loading OFFICIAL RbA checkpoint (Swin-B, COCO outlier supervision, Detectron2, CPU)...")
    scorer = OfficialRbAScorer(device="cpu")
    print("Loaded.\n")

    all_scores, all_labels = [], []
    raw_hits, contrast_hits, region_hits, evaluable = 0, 0, 0, 0
    records_for_viz = []

    frames = test_normal + test_hazard
    for idx, (path, scene, is_hazard) in enumerate(frames):
        img = Image.open(path).convert("RGB")
        rba_map, _ = scorer.score(img, out_size=CANVAS_SIZE)

        label_path = img_path_to_label_path(path)
        mask = load_mask_resized(label_path)
        if mask is None:
            continue
        hazard_mask = (mask == HAZARD_TRAIN_ID)

        all_scores.append(rba_map.flatten())
        all_labels.append(hazard_mask.flatten().astype(np.int8))

        if is_hazard and hazard_mask.any():
            evaluable += 1

            raw_peak_idx = np.unravel_index(np.argmax(rba_map), rba_map.shape)
            if hazard_mask[raw_peak_idx]:
                raw_hits += 1

            contrast_map = local_contrast_map(rba_map)
            contrast_peak_idx = np.unravel_index(np.argmax(contrast_map), contrast_map.shape)
            if hazard_mask[contrast_peak_idx]:
                contrast_hits += 1

            region_mask, region_peak_idx = region_from_peak_no_hood(rba_map)
            if (region_mask & hazard_mask).any():
                region_hits += 1
            records_for_viz.append((path, rba_map, hazard_mask, region_mask, region_peak_idx))

        if (idx + 1) % 10 == 0:
            print(f"  scored {idx + 1}/{len(frames)}", flush=True)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    pixel_auroc = auroc(all_scores, all_labels)
    pixel_aupr = aupr(all_scores, all_labels)

    raw_hit_rate = raw_hits / evaluable if evaluable else float("nan")
    contrast_hit_rate = contrast_hits / evaluable if evaluable else float("nan")
    region_hit_rate = region_hits / evaluable if evaluable else float("nan")

    print(f"\n{'=' * 70}")
    print("OFFICIAL RbA (Swin-B, COCO outlier supervision) RESULTS")
    print(f"{'=' * 70}")
    print(f"PIXEL-LEVEL AUROC: {pixel_auroc:.4f}")
    print(f"PIXEL-LEVEL AUPR:  {pixel_aupr:.4f}")
    print(f"RAW peak-pixel hit rate (no masking at all):      {raw_hits}/{evaluable} = {raw_hit_rate:.4f}")
    print(f"Local-contrast peak hit rate (no masking):        {contrast_hits}/{evaluable} = {contrast_hit_rate:.4f}")
    print(f"Region hit rate (contrast + border margin only):  {region_hits}/{evaluable} = {region_hit_rate:.4f}")
    print(f"\n{'=' * 70}")
    print("COMPARISON -- all three approaches, same held-out 30 hazard frames:")
    print(f"{'=' * 70}")
    print("  kNN baseline (patch-level):        AUROC 0.9439, AUPR 0.0830, top-5 hit rate 0.60 (18/30)")
    print("  Zero-shot RbA (vanilla ckpt):       AUROC 0.7917, AUPR 0.0065, region hit rate 0.2333 (7/30)")
    print(f"  Official RbA (COCO outlier ckpt):   AUROC {pixel_auroc:.4f}, AUPR {pixel_aupr:.4f}, "
          f"region hit rate {region_hit_rate:.4f} ({region_hits}/{evaluable})")

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    sample = records_for_viz[:min(N_VIZ_EXAMPLES, len(records_for_viz))]
    for k, (path, rba_map, hazard_mask, region_mask, peak_idx) in enumerate(sample):
        out_path = VIZ_DIR / f"example_{k:02d}.jpg"
        save_viz(path, rba_map, hazard_mask, region_mask, peak_idx, out_path)
        print(f"  saved {out_path}")

    results = {
        "pixel_auroc": float(pixel_auroc),
        "pixel_aupr": float(pixel_aupr),
        "raw_peak_hit_rate": float(raw_hit_rate),
        "raw_peak_hits": raw_hits,
        "contrast_peak_hit_rate": float(contrast_hit_rate),
        "contrast_peak_hits": contrast_hits,
        "region_hit_rate": float(region_hit_rate),
        "region_hits": region_hits,
        "evaluable_frames": evaluable,
        "n_test_normal": len(test_normal),
        "n_test_hazard": len(test_hazard),
        "checkpoint": "official RbA swin_b_1dl_rba_ood_coco (COCO outlier supervision)",
    }
    out_path = RESULTS_DIR / "rba_official_lost_and_found_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
