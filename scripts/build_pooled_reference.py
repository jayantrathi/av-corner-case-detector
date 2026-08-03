"""Refit the reference bank on POOLED normal driving from all three real
sources we have -- Lost & Found (Germany), CODA (China), nuScenes mini
(Boston + Singapore) -- instead of Lost & Found alone.

Why: the cone/bollard/manhole-cover false positives on CODA (see
run_demo_coda.py results) aren't a bug, they're the direct consequence of
fitting "normal" from ~80 German dashcam frames. Cones and manholes are
common, unremarkable objects on roads everywhere -- they only look
statistically rare because Lost & Found's own reference frames happen not to
contain many of them. A k-NN anomaly scorer can only be as good as its
definition of "normal": widen that definition with genuinely diverse normal
driving from multiple countries/cities, and common-but-locally-underrepented
objects like cones stop scoring as anomalous, because now there are close
neighbors for them in the bank.

This does NOT use CODA's hazard-labeled images or their nearby patches --
only CODA images with NO labeled hazard-category box go into the "normal"
pool (same contamination discipline as excluding Lost & Found's hazard
frames from its own reference bank). nuScenes has no anomaly labels at all,
so all of it is fair game as normal-only fitting material -- it can never be
used for evaluation, only for enriching what "normal" looks like.

Evaluation after refitting reuses the SAME Lost & Found held-out test split
as before (identical scene split, same seed) so the before/after numbers are
directly comparable, plus a re-run of the CODA hazard-box visual check to see
whether the cone/bollard/manhole false triggers actually go away.
"""
from __future__ import annotations
import sys
import json
import pickle
import random
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate_patch_localization import (
    LAF_ROOT, RESULTS_DIR, HAZARD_TRAIN_ID, SEED, CANVAS_SIZE,
    MIN_HAZARD_AREA_FRAC, TRAIN_REF_FRAMES_CAP, PATCHES_PER_TRAIN_FRAME,
    TEST_NORMAL_FRAMES_CAP, TEST_HAZARD_FRAMES_CAP, N_ROW_BANDS, CALIB_FRAC,
    row_to_band, split_fit_calib, scene_id_for, img_path_to_label_path,
    frame_hazard_area_frac, load_image_tensor, score_test_frames,
    PatchResNetEmbedder, feature_map_to_patches,
)
from src.scoring.embedding_scorers import kNNScorer
from src.eval.metrics import auroc, aupr
from src.data.nuscenes_loader import load_nuscenes_frames
from run_demo_coda import load_coda_annotations, draw_gt_boxes
from run_demo_roadanomaly21 import score_image, render_alert_frame

CODA_ROOT = Path("/Volumes/BIggen/AV/data/coda/CODA/sample")
NUSCENES_ROOT = Path("/Volumes/BIggen/AV/data/nuscenes")
POOLED_CACHE_PATH = RESULTS_DIR / "patch_records_cache_pooled.pkl"
CODA_DEMO_POOLED_DIR = RESULTS_DIR / "coda_demo_pooled"

NUSCENES_FRAMES_CAP = 150


@torch.no_grad()
def extract_bank_from_frames(model, frame_paths, device, patches_per_frame, bank_by_band, seed=SEED):
    """Same sampling logic as extract_reference_bank in
    evaluate_patch_localization.py, but accumulates into an EXISTING
    bank_by_band dict so multiple sources can be pooled into one bank."""
    rng = random.Random(seed)
    for idx, path in enumerate(frame_paths):
        t = load_image_tensor(str(path), device)
        if t is None:
            continue
        feat = model(t.unsqueeze(0))
        _, c, hf, wf = feat.shape
        patches = feature_map_to_patches(feat).cpu().numpy()
        n = len(patches)
        sel = rng.sample(range(n), min(patches_per_frame, n))
        for flat_idx in sel:
            i = flat_idx // wf
            band = row_to_band(i, hf)
            bank_by_band[band].append(patches[flat_idx])
        if (idx + 1) % 20 == 0 or idx == len(frame_paths) - 1:
            print(f"    frame {idx + 1}/{len(frame_paths)}", flush=True)


def load_laf_split():
    """Reproduce the EXACT same Lost & Found frame labeling + scene split as
    run_pipeline() in evaluate_patch_localization.py, so train_normal here
    matches what the original v3 bank was built from, and test_normal/
    test_hazard here are the SAME held-out frames -- needed for an
    apples-to-apples before/after comparison."""
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
    train_scenes = set(scenes[:n_train_scenes])
    test_scenes = set(scenes[n_train_scenes:])

    train_normal = [r for r in frame_records if r[1] in train_scenes and not r[2]]
    test_normal = [r for r in frame_records if r[1] in test_scenes and not r[2]]
    test_hazard = [r for r in frame_records if r[1] in test_scenes and r[2]]

    random.seed(SEED)
    if len(train_normal) > TRAIN_REF_FRAMES_CAP:
        train_normal = random.sample(train_normal, TRAIN_REF_FRAMES_CAP)
    if len(test_normal) > TEST_NORMAL_FRAMES_CAP:
        test_normal = random.sample(test_normal, TEST_NORMAL_FRAMES_CAP)
    if len(test_hazard) > TEST_HAZARD_FRAMES_CAP:
        test_hazard = random.sample(test_hazard, TEST_HAZARD_FRAMES_CAP)

    return train_normal, test_normal, test_hazard


def load_coda_normal_paths():
    """CODA images with NO labeled hazard-category box -- same contamination
    discipline as excluding LAF hazard frames from LAF's own reference
    bank."""
    images, boxes_by_image = load_coda_annotations()
    normal_paths = []
    for image_id, meta in images.items():
        if image_id in boxes_by_image:
            continue
        p = CODA_ROOT / "images" / meta["file_name"]
        if p.exists():
            normal_paths.append(p)
    return normal_paths


def load_nuscenes_paths():
    frames = load_nuscenes_frames(str(NUSCENES_ROOT))
    if not frames:
        raise RuntimeError(
            f"load_nuscenes_frames returned 0 frames from {NUSCENES_ROOT} -- "
            "check v1.0-mini/ has scene.json/sample.json/sample_data.json and "
            "a samples/CAM_FRONT/ directory before continuing. A silent 0 here "
            "previously let a bad reference bank build 'successfully' with no "
            "nuScenes data in it at all -- this is a hard failure on purpose."
        )
    missing = [f.path for f in frames if not Path(f.path).exists()]
    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(frames)} nuScenes frame paths from the loader "
            f"don't exist on disk (e.g. {missing[0]}) -- path construction is "
            "probably wrong again, fix it rather than silently dropping frames."
        )
    paths = [Path(f.path) for f in frames]
    rng = random.Random(SEED)
    if len(paths) > NUSCENES_FRAMES_CAP:
        paths = rng.sample(paths, NUSCENES_FRAMES_CAP)
    return paths


def main():
    print("=" * 70)
    print("Building POOLED reference bank: Lost & Found + CODA + nuScenes")
    print("=" * 70 + "\n")

    train_normal_laf, test_normal, test_hazard = load_laf_split()
    print(f"Lost & Found: {len(train_normal_laf)} train-normal (reference), "
          f"{len(test_normal)} test-normal, {len(test_hazard)} test-hazard (held out, unchanged)\n")

    coda_normal_paths = load_coda_normal_paths()
    print(f"CODA: {len(coda_normal_paths)} normal (no labeled hazard box) images available for reference\n")

    nuscenes_paths = load_nuscenes_paths()
    print(f"nuScenes: {len(nuscenes_paths)} frames sampled for reference (no anomaly labels -- fit-only)\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")
    model = PatchResNetEmbedder().to(device)
    model.eval()

    bank_by_band = {b: [] for b in range(N_ROW_BANDS)}

    print("Extracting reference patches from Lost & Found train-normal frames...")
    extract_bank_from_frames(model, [p for p, _, _ in train_normal_laf], device, PATCHES_PER_TRAIN_FRAME, bank_by_band)

    print("\nExtracting reference patches from CODA normal frames...")
    extract_bank_from_frames(model, coda_normal_paths, device, PATCHES_PER_TRAIN_FRAME, bank_by_band)

    print("\nExtracting reference patches from nuScenes frames...")
    extract_bank_from_frames(model, nuscenes_paths, device, PATCHES_PER_TRAIN_FRAME, bank_by_band)

    ref_banks = {b: np.stack(v, axis=0) for b, v in bank_by_band.items() if v}
    total_ref_patches = sum(len(v) for v in ref_banks.values())
    print(f"\nPooled reference bank total: {total_ref_patches} patches across {len(ref_banks)} bands")
    for b in sorted(ref_banks):
        print(f"    band {b}: {len(ref_banks[b])} patches")
    print()

    fit_banks, calib_banks = split_fit_calib(ref_banks)
    scorers_by_band, calib_stats_by_band = {}, {}
    for b in fit_banks:
        scorer = kNNScorer(k=5)
        scorer.fit(fit_banks[b])
        scorers_by_band[b] = scorer
        calib_scores = scorer.score(calib_banks[b])
        calib_mean, calib_std = float(calib_scores.mean()), float(calib_scores.std() + 1e-6)
        calib_stats_by_band[b] = (calib_mean, calib_std)
        print(f"    band {b}: fit={len(fit_banks[b])} calib={len(calib_banks[b])} "
              f"-> normal score mean={calib_mean:.3f} std={calib_std:.3f}")
    print()

    # --- Re-evaluate on the SAME Lost & Found held-out test set as before ---
    print("Scoring the ORIGINAL (unchanged) Lost & Found held-out test set with the pooled scorer...")
    test_frames = test_normal + test_hazard
    records = score_test_frames(model, scorers_by_band, calib_stats_by_band, test_frames, device)

    all_scores, all_labels = [], []
    for r in records:
        gt = r["patch_gt"]
        sc = r["score_grid"]
        mask_valid = gt >= 0
        all_scores.append(sc[mask_valid])
        all_labels.append(gt[mask_valid])
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    pooled_auroc = auroc(all_scores, all_labels)
    pooled_aupr = aupr(all_scores, all_labels)

    hits, evaluable = 0, 0
    TOP_K_VALUES = [3, 5, 10]
    topk_hits = {k: 0 for k in TOP_K_VALUES}
    for r in records:
        if not r["is_hazard_frame"]:
            continue
        gt = r["patch_gt"]
        if not (gt == 1).any():
            continue
        evaluable += 1
        grid = r["score_grid"]
        i_max, j_max = np.unravel_index(np.argmax(grid), grid.shape)
        if gt[i_max, j_max] == 1:
            hits += 1
        order = np.argsort(-grid.flatten())
        flat_gt = gt.flatten()
        for k in TOP_K_VALUES:
            if (flat_gt[order[:k]] == 1).any():
                topk_hits[k] += 1
    hit_rate = hits / evaluable if evaluable else float("nan")

    print(f"\nPOOLED-BANK Lost & Found held-out results:")
    print(f"  AUROC {pooled_auroc:.4f}, AUPR {pooled_aupr:.4f}, top-1 hit rate {hit_rate:.4f} ({hits}/{evaluable})")
    for k in TOP_K_VALUES:
        rate = topk_hits[k] / evaluable if evaluable else float("nan")
        print(f"  top-{k} hit rate: {rate:.4f} ({topk_hits[k]}/{evaluable})")
    print("\nFor comparison, the ORIGINAL Lost & Found-only v3 bank scored:")
    print("  AUROC 0.9439, AUPR 0.0830, top-1 hit rate 0.20 (6/30), top-5 0.60 (18/30)")
    print("If pooling helped without hurting: AUROC/AUPR/hit-rate here should be flat or better,")
    print("not measurably worse -- a real regression here would mean pooling diluted the signal,")
    print("not just added false-positive resistance.\n")

    # --- Re-run the CODA hazard visual check with the pooled scorer ---
    print("Re-running the CODA hazard-box visual check with the pooled scorer...")
    images, boxes_by_image = load_coda_annotations()
    CODA_DEMO_POOLED_DIR.mkdir(parents=True, exist_ok=True)
    for image_id, boxes in boxes_by_image.items():
        meta = images[image_id]
        img_path = CODA_ROOT / "images" / meta["file_name"]
        if not img_path.exists():
            continue
        score_grid = score_image(model, scorers_by_band, calib_stats_by_band, img_path, device)
        out_path = CODA_DEMO_POOLED_DIR / f"hazard_{meta['file_name']}"
        render_alert_frame(img_path, score_grid, out_path)
        from PIL import Image
        canvas = Image.open(out_path).convert("RGB")
        draw_gt_boxes(canvas, boxes, meta["width"], meta["height"])
        canvas.save(out_path, quality=94)
        print(f"  {meta['file_name']} -> {out_path.name}")

    print(f"\nSaved to {CODA_DEMO_POOLED_DIR} -- compare directly against results/coda_demo/")
    print("(same filenames, same frames, only the fitted reference bank differs).")

    # --- Cache the pooled scorer for reuse by demo scripts ---
    result = {
        "scorers_by_band": scorers_by_band,
        "calib_stats_by_band": calib_stats_by_band,
        "total_ref_patches": total_ref_patches,
        "n_row_bands": len(ref_banks),
        "sources": {
            "lost_and_found_frames": len(train_normal_laf),
            "coda_frames": len(coda_normal_paths),
            "nuscenes_frames": len(nuscenes_paths),
        },
    }
    with open(POOLED_CACHE_PATH, "wb") as f:
        pickle.dump(result, f)
    print(f"\nCached pooled scorer to {POOLED_CACHE_PATH}")

    results_summary = {
        "pooled_auroc": float(pooled_auroc),
        "pooled_aupr": float(pooled_aupr),
        "pooled_top1_hit_rate": float(hit_rate),
        "pooled_top1_hits": hits,
        "pooled_evaluable_frames": evaluable,
        "pooled_topk_hit_rate": {str(k): (topk_hits[k] / evaluable if evaluable else float("nan")) for k in TOP_K_VALUES},
        "reference_sources": result["sources"],
        "total_ref_patches": total_ref_patches,
        "original_laf_only_auroc": 0.9439,
        "original_laf_only_aupr": 0.0830,
        "original_laf_only_top1_hit_rate": 0.20,
    }
    with open(RESULTS_DIR / "pooled_reference_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved summary to {RESULTS_DIR / 'pooled_reference_results.json'}")


if __name__ == "__main__":
    main()
