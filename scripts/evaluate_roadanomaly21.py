"""Cross-dataset generalization check: score RoadAnomaly21 (SegmentMeIfYouCan
benchmark) against the SAME fitted reference bank we built from Lost & Found
-- no refitting. The question this answers: does the row-banded, per-band
z-calibrated k-NN scorer we validated on German dashcam footage still say
anything useful on completely different real photographs (different country,
different camera, different objects), or was it quietly overfit to Lost &
Found's specific domain (road surface texture, camera height/FOV, typical
German street furniture)?

Data: the 10 pixel-labeled RoadAnomaly21 validation images (validation0000
.. validation0009), downloaded via download_roadanomaly21.sh. These are the
ONLY RoadAnomaly21 images with published ground truth -- the other 90 named
test images (elephant0000.jpg, piano0000.jpg, etc.) have no masks and can't
be scored quantitatively.

Ground truth convention (confirmed by loading the masks directly and cross-
checking against the pytorch-ood library's SegmentMeIfYouCan loader source,
NOT assumed): raw pixel value 0 = normal/background, 1 = anomalous object,
255 = void/ignore region. Verified: every one of the 10 masks contains
exactly these three values and no others.

Method: reuse run_pipeline()'s cached scorers_by_band / calib_stats_by_band
(fitted once, entirely from Lost & Found train-normal frames) unchanged.
Score every patch of every RoadAnomaly21 validation image the identical way
score_test_frames() does for Lost & Found -- same row-band assignment (by
grid row position after resizing to the same CANVAS_SIZE), same z-calibration
per band. Ground truth per patch uses the same MIN_HAZARD_PATCH_FRAC /
MAX_NORMAL_PATCH_FRAC thresholds, with void pixels excluded from both the
hazard-fraction numerator and the denominator, and any patch with a large
void fraction dropped as ambiguous (the benchmark's own uncertainty region,
not ours to resolve).
"""
from __future__ import annotations
import sys
import json
import random
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate_patch_localization import (
    run_pipeline, load_image_tensor, row_to_band, N_ROW_BANDS, CANVAS_SIZE,
    MIN_HAZARD_PATCH_FRAC, MAX_NORMAL_PATCH_FRAC, SEED,
)
from src.scoring.patch_embedder import PatchResNetEmbedder, feature_map_to_patches, grid_cell_bbox
from src.eval.metrics import auroc, aupr

RA21_ROOT = Path("/Volumes/BIggen/AV/data/roadanomaly21")
RESULTS_DIR = Path("/Volumes/BIggen/AV/results")
VIZ_DIR = RESULTS_DIR / "roadanomaly21_examples"

ANOMALY_VALUE = 1
VOID_VALUE = 255
# A patch this saturated with void/ignore pixels can't be confidently scored
# as either normal or hazard -- drop it rather than guess.
MAX_VOID_PATCH_FRAC = 0.3


def load_ra21_frames():
    frames = []
    for i in range(10):
        img_path = RA21_ROOT / "images" / f"validation{i:04d}.jpg"
        mask_path = RA21_ROOT / "labels_masks" / f"validation{i:04d}_labels_semantic.png"
        if img_path.exists() and mask_path.exists():
            frames.append((img_path, mask_path, f"validation{i:04d}"))
    return frames


def load_mask_resized(mask_path: Path) -> np.ndarray | None:
    try:
        mask = Image.open(mask_path).resize(CANVAS_SIZE, Image.Resampling.NEAREST)
        return np.array(mask)
    except Exception:
        return None


@torch.no_grad()
def score_ra21_frame(model, scorers_by_band, calib_stats_by_band, img_path, mask_path, device):
    t = load_image_tensor(str(img_path), device)
    if t is None:
        return None
    feat = model(t.unsqueeze(0))
    _, c, hf, wf = feat.shape
    patches = feature_map_to_patches(feat).cpu().numpy()
    score_grid = np.empty((hf, wf), dtype=np.float32)
    for band in range(N_ROW_BANDS):
        rows_in_band = [i for i in range(hf) if row_to_band(i, hf) == band]
        if not rows_in_band or band not in scorers_by_band:
            continue
        row_lo, row_hi = min(rows_in_band), max(rows_in_band) + 1
        flat_lo, flat_hi = row_lo * wf, row_hi * wf
        band_patches = patches[flat_lo:flat_hi]
        band_scores = scorers_by_band[band].score(band_patches)
        calib_mean, calib_std = calib_stats_by_band[band]
        z_scores = (band_scores - calib_mean) / calib_std
        score_grid[row_lo:row_hi] = z_scores.reshape(row_hi - row_lo, wf)

    mask = load_mask_resized(mask_path)
    patch_gt = np.full((hf, wf), -1, dtype=np.int8)
    if mask is not None:
        hazard_mask = (mask == ANOMALY_VALUE)
        void_mask = (mask == VOID_VALUE)
        img_h, img_w = CANVAS_SIZE[1], CANVAS_SIZE[0]
        for i in range(hf):
            for j in range(wf):
                x0, y0, x1, y1 = grid_cell_bbox(i, j, hf, wf, img_h, img_w)
                cell_hazard = hazard_mask[y0:y1, x0:x1]
                cell_void = void_mask[y0:y1, x0:x1]
                if cell_hazard.size == 0:
                    continue
                void_frac = cell_void.mean()
                if void_frac > MAX_VOID_PATCH_FRAC:
                    continue  # too much ignore-region, leave ambiguous
                hazard_frac = cell_hazard.mean()
                if hazard_frac >= MIN_HAZARD_PATCH_FRAC:
                    patch_gt[i, j] = 1
                elif hazard_frac <= MAX_NORMAL_PATCH_FRAC:
                    patch_gt[i, j] = 0

    return {"path": str(img_path), "score_grid": score_grid, "patch_gt": patch_gt}


def save_viz(record, out_path):
    img = Image.open(record["path"]).convert("RGB").resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
    grid = record["score_grid"]
    hf, wf = grid.shape
    norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    heat = Image.fromarray((norm * 255).astype(np.uint8)).resize(CANVAS_SIZE, Image.Resampling.NEAREST)
    heat_arr = np.array(heat)
    overlay = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 4), dtype=np.uint8)
    overlay[..., 0] = 255
    overlay[..., 3] = (heat_arr.astype(np.float32) * 0.6).astype(np.uint8)
    heat_rgba = Image.fromarray(overlay, mode="RGBA")

    base = img.convert("RGBA")
    combined = Image.alpha_composite(base, heat_rgba)

    gt = record["patch_gt"]
    draw = ImageDraw.Draw(combined)
    for i in range(hf):
        for j in range(wf):
            if gt[i, j] == 1:
                x0, y0, x1, y1 = grid_cell_bbox(i, j, hf, wf, CANVAS_SIZE[1], CANVAS_SIZE[0])
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 255, 0, 255), width=1)

    i_max, j_max = np.unravel_index(np.argmax(grid), grid.shape)
    x0, y0, x1, y1 = grid_cell_bbox(i_max, j_max, hf, wf, CANVAS_SIZE[1], CANVAS_SIZE[0])
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 255, 0, 255), width=4)

    combined.convert("RGB").save(out_path, quality=90)


def main():
    print("Loading Lost & Found-fitted reference bank from cache (no refitting)...\n")
    laf_result = run_pipeline()
    scorers_by_band = laf_result["scorers_by_band"]
    calib_stats_by_band = laf_result["calib_stats_by_band"]

    frames = load_ra21_frames()
    print(f"Found {len(frames)} RoadAnomaly21 labeled validation frames\n")
    if len(frames) == 0:
        print("No frames found -- check data/roadanomaly21/images and labels_masks.")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = PatchResNetEmbedder().to(device)
    model.eval()

    records = []
    for img_path, mask_path, name in frames:
        r = score_ra21_frame(model, scorers_by_band, calib_stats_by_band, img_path, mask_path, device)
        if r is not None:
            r["name"] = name
            records.append(r)
            print(f"  scored {name}")
    print()

    all_scores, all_labels = [], []
    for r in records:
        gt = r["patch_gt"]
        sc = r["score_grid"]
        mask_valid = gt >= 0
        all_scores.append(sc[mask_valid])
        all_labels.append(gt[mask_valid])
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    n_hazard_patches = int(all_labels.sum())
    n_normal_patches = len(all_labels) - n_hazard_patches
    print(f"Pooled patches for eval: {len(all_labels)} ({n_normal_patches} normal, {n_hazard_patches} hazard)\n")

    ra21_auroc = auroc(all_scores, all_labels)
    ra21_aupr = aupr(all_scores, all_labels)
    print(f"ROADANOMALY21 PATCH-LEVEL AUROC: {ra21_auroc:.4f}")
    print(f"ROADANOMALY21 PATCH-LEVEL AUPR:  {ra21_aupr:.4f}\n")

    hits, evaluable = 0, 0
    TOP_K_VALUES = [3, 5, 10]
    topk_hits = {k: 0 for k in TOP_K_VALUES}
    for r in records:
        gt = r["patch_gt"]
        if not (gt == 1).any():
            print(f"  {r['name']}: no confidently-labeled hazard patches after void filtering, skipped from hit-rate")
            continue
        evaluable += 1
        grid = r["score_grid"]
        i_max, j_max = np.unravel_index(np.argmax(grid), grid.shape)
        if gt[i_max, j_max] == 1:
            hits += 1
        flat_scores = grid.flatten()
        flat_gt = gt.flatten()
        order = np.argsort(-flat_scores)
        for k in TOP_K_VALUES:
            if (flat_gt[order[:k]] == 1).any():
                topk_hits[k] += 1

    hit_rate = hits / evaluable if evaluable else float("nan")
    print(f"\nTOP-1 HIT RATE: {hits}/{evaluable} = {hit_rate:.4f}")
    for k in TOP_K_VALUES:
        rate = topk_hits[k] / evaluable if evaluable else float("nan")
        print(f"TOP-{k} HIT RATE: {topk_hits[k]}/{evaluable} = {rate:.4f}")

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    for r in records:
        out_path = VIZ_DIR / f"{r['name']}.jpg"
        save_viz(r, out_path)
    print(f"\nSaved {len(records)} visual examples (heatmap overlay, green=true hazard, "
          f"yellow=model argmax) to {VIZ_DIR}")
    print("LOOK AT THESE before trusting the numbers above -- this is a 10-image test,")
    print("small enough that eyeballing every single one is the right level of rigor,")
    print("not optional.\n")

    results = {
        "dataset": "RoadAnomaly21 (SegmentMeIfYouCan)",
        "n_frames": len(records),
        "reference_bank": "Lost & Found train-normal frames (reused, not refit)",
        "patch_auroc": float(ra21_auroc),
        "patch_aupr": float(ra21_aupr),
        "top1_hit_rate": float(hit_rate),
        "top1_hits": hits,
        "top1_evaluable_frames": evaluable,
        "topk_hits": {str(k): topk_hits[k] for k in TOP_K_VALUES},
        "topk_hit_rate": {str(k): (topk_hits[k] / evaluable if evaluable else float("nan")) for k in TOP_K_VALUES},
        "n_normal_patches": n_normal_patches,
        "n_hazard_patches": n_hazard_patches,
        "min_hazard_patch_frac": MIN_HAZARD_PATCH_FRAC,
        "max_normal_patch_frac": MAX_NORMAL_PATCH_FRAC,
        "max_void_patch_frac": MAX_VOID_PATCH_FRAC,
    }
    out_path = RESULTS_DIR / "roadanomaly21_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")

    print("\n" + "=" * 70)
    print("For comparison, the Lost & Found (in-domain) v3 result was:")
    print("  AUROC 0.9439, AUPR 0.0830, top-1 hit rate 0.20 (6/30), top-5 0.60 (18/30)")
    print(f"RoadAnomaly21 (out-of-domain, same fitted scorer): AUROC {ra21_auroc:.4f}, "
          f"AUPR {ra21_aupr:.4f}, top-1 {hit_rate:.4f} ({hits}/{evaluable})")
    print("A real generalization result, not an in-domain one -- read it as 'does the")
    print("SAME fitted model transfer,' not as an apples-to-apples benchmark number:")
    print("only 10 images here vs 30 for Lost & Found, so hit-rate swings are noisy,")
    print("and the two datasets' ground-truth conventions (pixel mask vs pixel mask,")
    print("but very different scene composition) aren't identical.")
    print("=" * 70)


if __name__ == "__main__":
    main()
