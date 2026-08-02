"""Validate the RbA (frozen Mask2Former) anomaly scorer against Lost &
Found's real, pixel-level hazard masks -- the same gate evaluate_patch_
localization.py was held to for the k-NN approach, so the numbers are
directly comparable.

Key methodological difference from the old pipeline: there is no fitting
step. RbA is zero-shot -- the Cityscapes-pretrained Mask2Former model IS the
"what does normal driving look like" knowledge, baked in during that
model's own training, not something we fit on our reference frames. So
there's no train_normal reference bank, no row-banding, no per-band
calibration -- all of that machinery existed specifically to make raw
feature-distance comparable across a scene, and none of it is needed once
the signal itself is a properly calibrated per-class probability.

We still score the SAME held-out test split (test_normal + test_hazard,
same seed, same scene-level split) as the k-NN v3 run, purely so the
before/after numbers are apples-to-apples -- not because RbA needs a
train/test split for its own sake.

Ground truth is used directly at native pixel resolution (no patch-grid
quantization needed, since RbA produces a real per-pixel score) -- a
simplification the patch-based approach couldn't offer.
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
from evaluate_patch_localization import (
    LAF_ROOT, RESULTS_DIR, HAZARD_TRAIN_ID, SEED, CANVAS_SIZE,
    MIN_HAZARD_AREA_FRAC, TEST_NORMAL_FRAMES_CAP, TEST_HAZARD_FRAMES_CAP,
    scene_id_for, img_path_to_label_path, frame_hazard_area_frac, load_mask_resized,
)
from src.scoring.mask2former_rba import RbAScorer
from src.eval.metrics import auroc, aupr

VIZ_DIR = RESULTS_DIR / "rba_lost_and_found_examples"
N_VIZ_EXAMPLES = 8

# Region-growing threshold for turning the continuous RbA map into a
# connected "alert region" -- same spirit as connected_region_bbox in
# build_demo_frames.py, but operating on real pixels via scipy.ndimage
# instead of a coarse patch grid.
#
# NOTE: originally this was "keep pixels within X% of the frame's peak
# value" (a fraction of the distance from 0 to peak_val). That assumed
# RbA's raw score spans a wide, well-separated range for this checkpoint.
# It doesn't. debug_rba_scores.py showed real CODA frames sitting in a
# tiny band (e.g. min=-10.32, max=-9.50, std=0.045) around a large
# negative baseline -- the paper's own formula is calibrated for a
# checkpoint fine-tuned with an outlier-exposure loss (target alpha=5);
# ours is a vanilla Cityscapes checkpoint that never saw that calibration,
# and Mask2Former's 100 queries aren't mutually exclusive at inference, so
# overlapping/redundant queries inflate the aggregated per-class logit far
# past what a single calibrated detector would produce -- the whole image
# saturates near the same huge-negative baseline instead of spreading from
# strongly-inlier (~-1) to genuinely-rejected (~0).
#
# The real signal is still there, just compressed into the top slice of
# each frame's OWN distribution (in that same debug run, the top 1% cut
# was 3.29% of pixels; the top 5% cut was 99.94% -- almost all the useful
# contrast lives in roughly the top 1%). So threshold per-frame by
# percentile of that frame's own score distribution instead of a fraction
# of its peak -- this is scale-invariant and doesn't care what the
# absolute baseline is.
REGION_TOP_PERCENTILE = 1.0  # keep the top 1% of pixels by RbA score, per frame


def load_test_split():
    """Same LAF frame labeling + scene split + caps as evaluate_patch_
    localization.run_pipeline(), reproduced here so this script has no
    dependency on that file's k-NN-specific machinery -- just the loading
    logic, which is shared, honest, and worth keeping identical for a fair
    comparison."""
    img_root = LAF_ROOT / "leftImg8bit" / "leftImg8bit"
    if not img_root.exists():
        img_root = LAF_ROOT / "leftImg8bit"
    all_images = sorted(img_root.rglob("*_leftImg8bit.png"))

    frame_records = []
    for img_path in all_images:
        label_path = img_path_to_label_path(img_path)
        frac = frame_hazard_area_frac(label_path)
        if frac is None:
            continue
        if frac == 0:
            frame_records.append((img_path, scene_id_for(img_path), False))
        elif frac >= MIN_HAZARD_AREA_FRAC:
            frame_records.append((img_path, scene_id_for(img_path), True))

    scenes = sorted(set(r[1] for r in frame_records))
    random.seed(SEED)
    random.shuffle(scenes)
    n_train_scenes = int(len(scenes) * 0.7)
    test_scenes = set(scenes[n_train_scenes:])

    test_normal = [r for r in frame_records if r[1] in test_scenes and not r[2]]
    test_hazard = [r for r in frame_records if r[1] in test_scenes and r[2]]

    random.seed(SEED)
    if len(test_normal) > TEST_NORMAL_FRAMES_CAP:
        test_normal = random.sample(test_normal, TEST_NORMAL_FRAMES_CAP)
    if len(test_hazard) > TEST_HAZARD_FRAMES_CAP:
        test_hazard = random.sample(test_hazard, TEST_HAZARD_FRAMES_CAP)

    return test_normal, test_hazard


# Margins excluded from detection eligibility -- neither is a workaround
# for a fixable bug, both are checked, not assumed:
#   - bottom: Lost & Found's camera mount is fixed, so the ego vehicle's
#     own hood/hood-ornament sits in the same screen position in every
#     single frame -- a real, physical, always-present object that isn't a
#     Cityscapes class and isn't a hazard. Excluding a fixed ego-vehicle
#     region from perception logic is standard practice in real AV stacks.
#   - top: originally suspected to be a Mask2Former processor zero-padding
#     artifact smearing into the canvas on interpolation. Checked directly
#     with debug_rba_padding.py against real Lost & Found frames: this
#     checkpoint's processor resizes straight to a fixed 384x384 square
#     with ZERO padding (confirmed via pixel_mask -- 0px bottom, 0px
#     right), so that theory was wrong, not just unproven. The top-edge
#     artifact is still there with padding fully ruled out, which points
#     to a boundary/receptive-field effect in the frozen model itself --
#     pixels near an image edge have truncated context, a generic property
#     of dense prediction near borders. Not fixable on our end, so this
#     margin is the correct response to it, not a patch over a bug.
TOP_MARGIN_FRAC = 0.08
BOTTOM_MARGIN_FRAC = 0.15


def eligible_region_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.ones(shape, dtype=bool)
    mask[: int(h * TOP_MARGIN_FRAC), :] = False
    mask[int(h * (1 - BOTTOM_MARGIN_FRAC)):, :] = False
    return mask


def largest_region_from_peak(rba_map: np.ndarray, top_percentile: float = REGION_TOP_PERCENTILE):
    """Threshold the map, connected-component label it, return the
    component containing the global argmax as a boolean mask -- the
    pixel-resolution equivalent of connected_region_bbox, using
    scipy.ndimage instead of a manual flood fill (fast enough at 1600x900,
    a hand-rolled Python stack-based fill would not be).

    Threshold is the (100 - top_percentile)-th percentile of THIS frame's
    own score distribution, not a fraction of its peak -- see the
    REGION_TOP_PERCENTILE comment above for why. Both the percentile and
    the argmax are computed ONLY over eligible_region_mask -- see that
    function's docstring for why the frame edges are excluded."""
    eligible = eligible_region_mask(rba_map.shape)
    masked = np.where(eligible, rba_map, -np.inf)
    peak_idx = np.unravel_index(np.argmax(masked), rba_map.shape)
    threshold = np.percentile(rba_map[eligible], 100 - top_percentile)
    binary = (rba_map >= threshold) & eligible
    labeled, _ = ndimage.label(binary)
    region_id = labeled[peak_idx]
    return labeled == region_id, peak_idx


def save_viz(image_path, rba_map, hazard_mask, region_mask, peak_idx, out_path):
    """Side-by-side: raw RbA heatmap overlay, real ground-truth outline
    (green), and the extracted alert region (cyan) -- so a hit or miss is
    visible directly, same discipline that caught the k-NN failures.

    NOTE: normalization is PERCENTILE-clipped (p1-p99.5), not true min-max.
    This checkpoint's score distribution is a tight cluster near the max
    with a long, thin, rare-outlier tail down to the true min (same shape
    documented in debug_rba_scores.py: ~98% of pixels within ~0.36 of the
    max, min is a rare extreme). True min-max normalization drags nearly
    the whole frame to a high (fully red) value because almost every pixel
    sits close to the max relative to that rare outlier min -- washing the
    entire image red and hiding whatever real contrast exists. Percentile
    clipping shows the actual spatial structure instead."""
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
    # ground truth outline in green, thickened (3px erosion band) so it's
    # visible even against a busy heatmap, not just a 1px hairline
    gt_edge = hazard_mask ^ ndimage.binary_erosion(hazard_mask, iterations=3)
    ys, xs = np.where(gt_edge)
    for y, x in zip(ys, xs):
        draw.point((x, y), fill=(0, 255, 0, 255))
    # extracted alert region outline in CYAN, not red -- red is already the
    # heatmap's color and a red-on-red outline is invisible by construction
    region_edge = region_mask ^ ndimage.binary_erosion(region_mask, iterations=3)
    ys, xs = np.where(region_edge)
    for y, x in zip(ys, xs):
        draw.point((x, y), fill=(0, 255, 255, 255))
    py, px = peak_idx
    draw.ellipse([px - 6, py - 6, px + 6, py + 6], outline=(255, 255, 0, 255), width=3)

    combined.convert("RGB").save(out_path, quality=92)


def main():
    test_normal, test_hazard = load_test_split()
    print(f"Held-out test set: {len(test_normal)} normal, {len(test_hazard)} hazard (same split as k-NN v3 run)\n")

    device = "mps" if __import__("torch").backends.mps.is_available() else "cpu"
    print(f"Loading frozen Mask2Former (Cityscapes) on {device}...")
    scorer = RbAScorer(device=device)
    print("Loaded. No fitting step -- scoring starts immediately.\n")

    all_scores, all_labels = [], []
    hits, evaluable = 0, 0
    region_hits = 0
    records_for_viz = []

    for idx, (path, scene, is_hazard) in enumerate(test_normal + test_hazard):
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
            peak_idx = np.unravel_index(np.argmax(rba_map), rba_map.shape)
            if hazard_mask[peak_idx]:
                hits += 1
            region_mask, peak_idx = largest_region_from_peak(rba_map)
            if (region_mask & hazard_mask).any():
                region_hits += 1
            records_for_viz.append((path, rba_map, hazard_mask, region_mask, peak_idx))

        if (idx + 1) % 10 == 0:
            print(f"  scored {idx + 1}/{len(test_normal) + len(test_hazard)}", flush=True)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    pixel_auroc = auroc(all_scores, all_labels)
    pixel_aupr = aupr(all_scores, all_labels)

    hit_rate = hits / evaluable if evaluable else float("nan")
    region_hit_rate = region_hits / evaluable if evaluable else float("nan")

    print(f"\nRbA PIXEL-LEVEL AUROC: {pixel_auroc:.4f}")
    print(f"RbA PIXEL-LEVEL AUPR:  {pixel_aupr:.4f}")
    print(f"RbA PEAK-PIXEL HIT RATE (argmax lands on real hazard): {hits}/{evaluable} = {hit_rate:.4f}")
    print(f"RbA REGION HIT RATE (grown alert region touches real hazard): {region_hits}/{evaluable} = {region_hit_rate:.4f}")

    print("\nFor comparison, the k-NN v3 (patch-level, Lost & Found only) result was:")
    print("  AUROC 0.9439, AUPR 0.0830, top-1 hit rate 0.20 (6/30), top-5 hit rate 0.60 (18/30)")
    print("Not a perfectly apples-to-apples comparison (pixel-level vs patch-level ground truth),")
    print("but AUPR in particular is worth watching closely -- it's the metric that most directly")
    print("reflects whether hazard pixels rank above the vast normal-pixel majority.\n")

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    sample = random.sample(records_for_viz, min(N_VIZ_EXAMPLES, len(records_for_viz)))
    for k, (path, rba_map, hazard_mask, region_mask, peak_idx) in enumerate(sample):
        out_path = VIZ_DIR / f"example_{k:02d}.jpg"
        save_viz(path, rba_map, hazard_mask, region_mask, peak_idx, out_path)
        print(f"  saved {out_path}")

    results = {
        "pixel_auroc": float(pixel_auroc),
        "pixel_aupr": float(pixel_aupr),
        "peak_hit_rate": float(hit_rate),
        "peak_hits": hits,
        "region_hit_rate": float(region_hit_rate),
        "region_hits": region_hits,
        "evaluable_frames": evaluable,
        "n_test_normal": len(test_normal),
        "n_test_hazard": len(test_hazard),
        "checkpoint": "facebook/mask2former-swin-tiny-cityscapes-semantic",
        "region_top_percentile": REGION_TOP_PERCENTILE,
    }
    out_path = RESULTS_DIR / "rba_lost_and_found_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
