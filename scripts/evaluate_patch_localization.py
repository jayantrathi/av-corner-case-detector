"""Validate patch-level anomaly localization against Lost & Found's real
pixel-level ground truth.

This is the gate we agreed on before touching the demo again: it's not
enough for a frame-level score to separate hazard frames from normal frames
(already shown, frozen backbone, AUROC 0.6473) -- for this to be useful the
way a driving system needs it to be useful, the HOTTEST PATCH has to
actually land on the real hazard, not just anywhere in a frame that happens
to contain one.

Method (same discipline as evaluate_lost_and_found.py, extended to patches):
  - Reference bank = patch descriptors sampled from Lost & Found's own
    TRAIN-split NORMAL frames only (one-class fit, no hazard patches, no
    nuScenes -- avoids reintroducing the cross-dataset confound).
  - Test = every patch of every held-out TEST frame (normal + hazard,
    disjoint scenes from train), scored via k-NN distance to that bank.
  - Ground truth per patch = does the patch's pixel footprint overlap the
    real Cityscapes-format hazard mask (trainId 2) above a coverage
    threshold. This gives patch-level AUROC/AUPR (does score rank hazard
    patches above normal patches in general) AND a top-1 hit rate (on
    hazard frames, is the single highest-scoring patch actually one of the
    true hazard patches -- the literal "does the box land on the real
    thing" question).

Uses the frozen ImageNet backbone only -- see patch_embedder.py docstring
and the Lost & Found frame-level result for why.

v2 update, after looking at the v1 results: v1 (single global reference
bank, pooled across the whole frame) got AUROC 0.9266 but AUPR only 0.0662
and a 30% top-1 hit rate. Looking at the actual heatmaps explained why --
sky, rooflines, and trees scored anomalous almost everywhere, regardless of
whether a hazard was present, because road surface looks similar scene to
scene while sky/buildings/foliage vary a lot -- so a position-blind k-NN
bank flags "not road" more than it flags "genuinely novel object." This is
the exact confound that PaDiM-style patch anomaly detection avoids by
fitting a SEPARATE reference distribution per spatial position, rather than
one pooled bank. v2 does a coarser version of that: the reference bank is
stratified into vertical row-bands, so a patch is only ever compared against
other patches from roughly the same height in the frame (sky vs sky, road
vs road) -- full per-exact-cell modeling (true PaDiM) isn't viable with only
~80 reference frames, so row-banding is the data-constrained middle ground.

v3 update, after v2 came back essentially identical to v1 (AUROC 0.9299 vs
0.9266, AUPR 0.0657 vs 0.0662, top-1 hit rate the exact same 9/30, same
heatmaps): row-banding the REFERENCE POOL only fixes within-band ranking --
it does nothing about cross-band comparability. Raw k-NN distance is not on
a common scale across bands: a band with more inherent scene-to-scene
variety among ordinary "normal" patches (sky/rooflines, which differ a lot
frame to frame) will naturally produce larger raw distances than a
low-variety band (road surface, which looks similar everywhere), even when
BOTH bands are behaving perfectly normally. A single argmax or pooled AUROC
computed on raw, uncalibrated scores across bands will therefore always
favor whichever band has the largest natural score spread -- which is
exactly what the heatmaps showed happening, unchanged by banding alone.

Fix: calibrate. Hold out part of each band's reference patches (not used to
fit that band's k-NN bank) purely to measure what a "normal" score looks
like FOR THAT BAND -- its own mean and std. Convert every raw score to a
z-score relative to its own band's normal distribution before comparing
across bands (argmax) or pooling (AUROC/AUPR). This puts "how surprising is
this patch relative to what's normal AT THIS POSITION" on the same scale
everywhere, which is the actual apples-to-apples comparison localization
needs.
"""
from __future__ import annotations
import sys
import re
import json
import pickle
import random
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.scoring.embedding_scorers import kNNScorer
from src.scoring.patch_embedder import PatchResNetEmbedder, feature_map_to_patches, grid_cell_bbox
from src.eval.metrics import auroc, aupr

LAF_ROOT = Path("/Volumes/BIggen/AV/data/lost_and_found")
RESULTS_DIR = Path("/Volumes/BIggen/AV/results")
VIZ_DIR = RESULTS_DIR / "patch_localization_examples_v3_calibrated"
HAZARD_TRAIN_ID = 2
SEED = 0
CANVAS_SIZE = (1600, 900)  # (W, H) -- matches the rest of the pipeline

# Same frame-level hazard/normal/ambiguous discipline as evaluate_lost_and_found.py
MIN_HAZARD_AREA_FRAC = 0.0006

# Patch-level ground truth: how much of a patch's pixel footprint must be
# hazard-labeled for the patch itself to count as "hazard." Not 0% (a patch
# that's 99% road with the hazard mask's feathered edge clipping one corner
# shouldn't count) and not 50%+ (patches are already coarse -- stride 16px
# means most true-positive patches will be a mix of object + background).
# 0.15 requires a real, substantial presence of the object in that cell.
MIN_HAZARD_PATCH_FRAC = 0.15
# Below this, essentially zero overlap -- a clean "normal" patch.
MAX_NORMAL_PATCH_FRAC = 0.02
# Between the two: the hazard mask clips the patch edge -- genuinely
# ambiguous, dropped rather than forced into either bucket.

TRAIN_REF_FRAMES_CAP = 80
PATCHES_PER_TRAIN_FRAME = 150
TEST_NORMAL_FRAMES_CAP = 15
TEST_HAZARD_FRAMES_CAP = 30
N_VIZ_EXAMPLES = 6

# Row-band stratification: how many equal-height vertical bands to split the
# patch grid into. Each band gets its OWN reference bank and its own kNN
# scorer, so a patch is only ever judged against other patches from roughly
# the same height in the frame. 8 bands on a ~57-row grid gives each band
# ~7 rows -- fine enough to separate "sky band" from "road band" while still
# leaving ~80 frames * (150/8 avg) ~= 1500 reference patches per band, plenty
# for k=5 nearest-neighbor.
N_ROW_BANDS = 8


def row_to_band(i: int, grid_h: int, n_bands: int = N_ROW_BANDS) -> int:
    return min(n_bands - 1, i * n_bands // grid_h)


# Fraction of each band's reference patches held out purely for calibration
# (measuring that band's own "normal" score mean/std) rather than fed into
# the k-NN bank itself. Never score calibration patches against a bank that
# includes them -- that would trivially give near-zero self-distances and
# make every band look artificially tight.
CALIB_FRAC = 0.3


def split_fit_calib(bank_by_band: dict[int, np.ndarray], calib_frac: float = CALIB_FRAC, seed: int = SEED):
    rng = random.Random(seed)
    fit_banks, calib_banks = {}, {}
    for b, patches in bank_by_band.items():
        idx = list(range(len(patches)))
        rng.shuffle(idx)
        n_calib = max(1, int(len(idx) * calib_frac))
        calib_banks[b] = patches[idx[:n_calib]]
        fit_banks[b] = patches[idx[n_calib:]]
    return fit_banks, calib_banks

SEQ_RE = re.compile(r"^(.*)_(\d{6})_(\d{6})_leftImg8bit\.png$")


def scene_id_for(img_path: Path) -> str:
    m = SEQ_RE.match(img_path.name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return img_path.parent.name


def img_path_to_label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    parts = ["gtCoarse" if p == "leftImg8bit" else p for p in parts]
    gt_path = Path(*parts)
    gt_path = gt_path.with_name(gt_path.name.replace("_leftImg8bit.png", "_gtCoarse_labelTrainIds.png"))
    return gt_path


def frame_hazard_area_frac(label_path: Path) -> float | None:
    try:
        arr = np.array(Image.open(label_path))
        return float(np.sum(arr == HAZARD_TRAIN_ID)) / arr.size
    except Exception:
        return None


def load_image_tensor(path: str, device: str) -> torch.Tensor | None:
    try:
        img = Image.open(path).convert("RGB").resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        t[0] = (t[0] - 0.485) / 0.229
        t[1] = (t[1] - 0.456) / 0.224
        t[2] = (t[2] - 0.406) / 0.225
        return t.to(device)
    except Exception:
        return None


def load_mask_resized(label_path: Path) -> np.ndarray | None:
    """Resize the ground-truth mask to CANVAS_SIZE with NEAREST (label maps
    must never be interpolated) so it lines up pixel-for-pixel with the
    resized image the model actually saw."""
    try:
        mask = Image.open(label_path).resize(CANVAS_SIZE, Image.Resampling.NEAREST)
        return np.array(mask)
    except Exception:
        return None


@torch.no_grad()
def extract_reference_bank(model, frames, device, patches_per_frame, seed=SEED):
    """Sample `patches_per_frame` random patch descriptors from each frame's
    full spatial grid, tagged with their row-band. Returns dict {band: (N,C)
    array}. We don't need every patch from every normal frame -- just a
    large, representative sample of what "normal" patches look like AT EACH
    HEIGHT in the frame."""
    rng = random.Random(seed)
    bank_by_band: dict[int, list[np.ndarray]] = {b: [] for b in range(N_ROW_BANDS)}
    for idx, (path, _, _) in enumerate(frames):
        t = load_image_tensor(str(path), device)
        if t is None:
            continue
        feat = model(t.unsqueeze(0))  # (1, 1024, Hf, Wf)
        _, c, hf, wf = feat.shape
        patches = feature_map_to_patches(feat).cpu().numpy()  # (Hf*Wf, 1024)
        n = len(patches)
        sel = rng.sample(range(n), min(patches_per_frame, n))
        for flat_idx in sel:
            i = flat_idx // wf
            band = row_to_band(i, hf)
            bank_by_band[band].append(patches[flat_idx])
        if (idx + 1) % 10 == 0 or idx == len(frames) - 1:
            print(f"    reference frame {idx + 1}/{len(frames)}", flush=True)
    return {b: np.stack(v, axis=0) for b, v in bank_by_band.items() if v}


@torch.no_grad()
def score_test_frames(model, scorers_by_band, calib_stats_by_band, frames, device):
    """For each test frame: full patch grid -> kNN score per patch (against
    that patch's OWN row-band scorer) -> z-normalize against that band's own
    calibration mean/std -> ground truth per patch from the real mask.
    Returns per-frame records. The z-normalization is what makes scores from
    different bands comparable on a single grid (see v3 docstring note)."""
    records = []
    for idx, (path, scene, is_hazard) in enumerate(frames):
        t = load_image_tensor(str(path), device)
        if t is None:
            continue
        feat = model(t.unsqueeze(0))
        _, c, hf, wf = feat.shape
        patches = feature_map_to_patches(feat).cpu().numpy()  # (Hf*Wf, C)
        score_grid = np.empty((hf, wf), dtype=np.float32)
        # Score each row-band's rows in one batched call against that
        # band's own reference bank -- avoids per-patch Python overhead
        # while still keeping bands separate.
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

        label_path = img_path_to_label_path(path)
        mask = load_mask_resized(label_path)
        patch_gt = np.full((hf, wf), -1, dtype=np.int8)  # -1 = ambiguous/dropped
        if mask is not None:
            hazard_mask = (mask == HAZARD_TRAIN_ID)
            img_h, img_w = CANVAS_SIZE[1], CANVAS_SIZE[0]
            for i in range(hf):
                for j in range(wf):
                    x0, y0, x1, y1 = grid_cell_bbox(i, j, hf, wf, img_h, img_w)
                    cell = hazard_mask[y0:y1, x0:x1]
                    if cell.size == 0:
                        continue
                    frac = cell.mean()
                    if frac >= MIN_HAZARD_PATCH_FRAC:
                        patch_gt[i, j] = 1
                    elif frac <= MAX_NORMAL_PATCH_FRAC:
                        patch_gt[i, j] = 0

        records.append({
            "path": str(path), "scene": scene, "is_hazard_frame": is_hazard,
            "score_grid": score_grid, "patch_gt": patch_gt,
        })
        if (idx + 1) % 10 == 0 or idx == len(frames) - 1:
            print(f"    test frame {idx + 1}/{len(frames)}", flush=True)
    return records


def save_viz(record, out_path):
    """Overlay the score heatmap (red, brighter = more anomalous) on the raw
    frame, plus a green outline of the real ground-truth hazard region, so a
    human can eyeball whether the hot patch actually lands on the object."""
    img = Image.open(record["path"]).convert("RGB").resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
    grid = record["score_grid"]
    hf, wf = grid.shape
    norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    heat = Image.fromarray((norm * 255).astype(np.uint8)).resize(CANVAS_SIZE, Image.Resampling.NEAREST)
    heat_rgba = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    heat_arr = np.array(heat)
    overlay = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 4), dtype=np.uint8)
    overlay[..., 0] = 255
    overlay[..., 3] = (heat_arr.astype(np.float32) * 0.6).astype(np.uint8)
    heat_rgba = Image.fromarray(overlay, mode="RGBA")

    base = img.convert("RGBA")
    combined = Image.alpha_composite(base, heat_rgba)

    # outline the true hazard patches in green
    gt = record["patch_gt"]
    draw = ImageDraw.Draw(combined)
    for i in range(hf):
        for j in range(wf):
            if gt[i, j] == 1:
                x0, y0, x1, y1 = grid_cell_bbox(i, j, hf, wf, CANVAS_SIZE[1], CANVAS_SIZE[0])
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 255, 0, 255), width=1)

    # mark the argmax (hottest) patch with a thick yellow box
    i_max, j_max = np.unravel_index(np.argmax(grid), grid.shape)
    x0, y0, x1, y1 = grid_cell_bbox(i_max, j_max, hf, wf, CANVAS_SIZE[1], CANVAS_SIZE[0])
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 255, 0, 255), width=4)

    combined.convert("RGB").save(out_path, quality=90)


CACHE_PATH = RESULTS_DIR / "patch_records_cache_v3.pkl"
# Bump this if the cached dict's shape changes (new keys downstream scripts
# need) -- stale caches from an older version of this file get silently
# rebuilt instead of causing a KeyError somewhere else.
REQUIRED_CACHE_KEYS = {
    "records", "total_ref_patches", "n_row_bands", "n_train_normal",
    "n_test_normal", "n_test_hazard", "scorers_by_band", "calib_stats_by_band",
}


def run_pipeline(force_recompute: bool = False):
    """Full pipeline: load LAF frames, label, scene-split, build row-banded
    + calibrated patch reference, score every test frame. Returns a dict with
    `records` (per-frame score grids + ground truth) plus band metadata and
    the fitted scorers themselves.

    Cached to disk (CACHE_PATH) because this involves a real model forward
    pass over ~125 frames and is the expensive part -- anything downstream
    that just wants to RENDER from these scores (build_demo_frames.py) or
    score NEW frames with the same fitted reference (build_demo_video.py)
    should reuse the cache instead of repeating the inference."""
    if not force_recompute and CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        if REQUIRED_CACHE_KEYS.issubset(cached.keys()):
            print(f"Loading cached scored records from {CACHE_PATH}...\n")
            return cached
        print(f"Cache at {CACHE_PATH} is from an older version of this script "
              f"(missing keys) -- recomputing...\n")

    random.seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")

    img_root = LAF_ROOT / "leftImg8bit" / "leftImg8bit"
    if not img_root.exists():
        img_root = LAF_ROOT / "leftImg8bit"
    all_images = sorted(img_root.rglob("*_leftImg8bit.png"))
    print(f"Found {len(all_images)} Lost & Found frames\n")

    print("Labeling frames (same discipline as evaluate_lost_and_found.py)...")
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
        # else ambiguous, dropped

    n_hazard = sum(1 for _, _, h in frame_records if h)
    print(f"Labeled {len(frame_records)} frames: {len(frame_records) - n_hazard} normal, {n_hazard} hazard\n")

    # Identical scene split logic/seed to evaluate_lost_and_found.py so train
    # and test scenes here match that run.
    scenes = sorted(set(r[1] for r in frame_records))
    random.shuffle(scenes)
    n_train_scenes = int(len(scenes) * 0.7)
    train_scenes = set(scenes[:n_train_scenes])
    test_scenes = set(scenes[n_train_scenes:])
    print(f"Scenes: {len(scenes)} total -> {len(train_scenes)} train, {len(test_scenes)} test\n")

    train_normal = [r for r in frame_records if r[1] in train_scenes and not r[2]]
    test_normal = [r for r in frame_records if r[1] in test_scenes and not r[2]]
    test_hazard = [r for r in frame_records if r[1] in test_scenes and r[2]]

    if len(train_normal) > TRAIN_REF_FRAMES_CAP:
        train_normal = random.sample(train_normal, TRAIN_REF_FRAMES_CAP)
    if len(test_normal) > TEST_NORMAL_FRAMES_CAP:
        test_normal = random.sample(test_normal, TEST_NORMAL_FRAMES_CAP)
    if len(test_hazard) > TEST_HAZARD_FRAMES_CAP:
        test_hazard = random.sample(test_hazard, TEST_HAZARD_FRAMES_CAP)

    print(f"Reference frames (train, normal only): {len(train_normal)}")
    print(f"Test frames: {len(test_normal)} normal, {len(test_hazard)} hazard\n")

    model = PatchResNetEmbedder().to(device)
    model.eval()

    print(f"Building {N_ROW_BANDS} row-banded patch reference banks from train-normal frames...")
    ref_banks = extract_reference_bank(model, train_normal, device, PATCHES_PER_TRAIN_FRAME)
    total_ref_patches = sum(len(v) for v in ref_banks.values())
    for b in sorted(ref_banks):
        print(f"    band {b}: {len(ref_banks[b])} reference patches")
    print(f"Reference bank total: {total_ref_patches} patches across {len(ref_banks)} bands\n")

    fit_banks, calib_banks = split_fit_calib(ref_banks)

    scorers_by_band = {}
    calib_stats_by_band = {}
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

    print("Scoring test frames (full patch grid each, band-stratified + z-calibrated)...")
    test_frames = test_normal + test_hazard
    records = score_test_frames(model, scorers_by_band, calib_stats_by_band, test_frames, device)
    print(f"Scored {len(records)} test frames\n")

    result = {
        "records": records,
        "total_ref_patches": total_ref_patches,
        "n_row_bands": len(ref_banks),
        "n_train_normal": len(train_normal),
        "n_test_normal": len(test_normal),
        "n_test_hazard": len(test_hazard),
        # kept so downstream scripts (e.g. build_demo_video.py) can score
        # NEW frames -- like a full driving sequence -- without repeating
        # reference-bank construction. kNNScorer/tuples are plain
        # numpy/float data, safe to pickle.
        "scorers_by_band": scorers_by_band,
        "calib_stats_by_band": calib_stats_by_band,
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(result, f)
    print(f"Cached scored records to {CACHE_PATH} (delete this file, or pass "
          f"force_recompute=True, to rebuild from scratch)\n")
    return result


def main():
    pipeline_result = run_pipeline()
    records = pipeline_result["records"]
    total_ref_patches = pipeline_result["total_ref_patches"]
    random.seed(SEED)  # reproducible viz sampling below, whether records came from cache or fresh compute

    # Pool all patches with a definite (non-ambiguous) ground-truth label
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

    patch_auroc = auroc(all_scores, all_labels)
    patch_aupr = aupr(all_scores, all_labels)
    print(f"PATCH-LEVEL AUROC: {patch_auroc:.4f}")
    print(f"PATCH-LEVEL AUPR:  {patch_aupr:.4f}\n")

    # Top-1 hit rate: on hazard frames that have at least one confidently
    # labeled hazard patch, does the single hottest patch in the frame fall
    # on a true hazard patch? This is the literal "does the box land on the
    # real thing" question -- the one that actually matters for a demo.
    hits, evaluable = 0, 0
    for r in records:
        if not r["is_hazard_frame"]:
            continue
        gt = r["patch_gt"]
        if not (gt == 1).any():
            continue
        evaluable += 1
        i_max, j_max = np.unravel_index(np.argmax(r["score_grid"]), r["score_grid"].shape)
        if gt[i_max, j_max] == 1:
            hits += 1
    hit_rate = hits / evaluable if evaluable else float("nan")
    print(f"TOP-1 HIT RATE: {hits}/{evaluable} = {hit_rate:.4f}")
    print("(fraction of hazard frames where the single hottest patch actually")
    print(" overlaps the real, human-labeled hazard region)\n")

    # Top-K hit rate: a real driver-facing system highlights a handful of
    # candidate regions, not one pixel-perfect box -- single-argmax is a
    # stricter bar than the system actually needs. Is the true hazard
    # reliably among the K most-suspicious patches even when it's not #1
    # (e.g. because a manhole cover or drain grate outscored it)?
    TOP_K_VALUES = [3, 5, 10]
    topk_hits = {k: 0 for k in TOP_K_VALUES}
    for r in records:
        if not r["is_hazard_frame"]:
            continue
        gt = r["patch_gt"]
        if not (gt == 1).any():
            continue
        flat_scores = r["score_grid"].flatten()
        flat_gt = gt.flatten()
        order = np.argsort(-flat_scores)
        for k in TOP_K_VALUES:
            top_idx = order[:k]
            if (flat_gt[top_idx] == 1).any():
                topk_hits[k] += 1
    topk_rates = {k: (topk_hits[k] / evaluable if evaluable else float("nan")) for k in TOP_K_VALUES}
    for k in TOP_K_VALUES:
        print(f"TOP-{k} HIT RATE: {topk_hits[k]}/{evaluable} = {topk_rates[k]:.4f}")
    print("(fraction of hazard frames where at least one of the K hottest patches")
    print(" overlaps the real hazard region -- the bar a region-highlighting demo")
    print(" actually needs to clear, vs. single-argmax being exactly right)\n")

    # Save a handful of visual examples: heatmap overlay, green = true hazard
    # patches, yellow box = the model's argmax (hottest) patch.
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    hazard_records = [r for r in records if r["is_hazard_frame"] and (r["patch_gt"] == 1).any()]
    viz_sample = random.sample(hazard_records, min(N_VIZ_EXAMPLES, len(hazard_records)))
    for k, r in enumerate(viz_sample):
        out_path = VIZ_DIR / f"example_{k:02d}.jpg"
        save_viz(r, out_path)
    print(f"Saved {len(viz_sample)} visual examples to {VIZ_DIR}\n")

    results = {
        "patch_auroc": float(patch_auroc),
        "patch_aupr": float(patch_aupr),
        "top1_hit_rate": float(hit_rate),
        "top1_hits": hits,
        "top1_evaluable_frames": evaluable,
        "topk_hits": {str(k): topk_hits[k] for k in TOP_K_VALUES},
        "topk_hit_rate": {str(k): float(topk_rates[k]) for k in TOP_K_VALUES},
        "n_normal_patches": n_normal_patches,
        "n_hazard_patches": n_hazard_patches,
        "n_reference_patches": total_ref_patches,
        "n_row_bands": N_ROW_BANDS,
        "calib_frac": CALIB_FRAC,
        "calib_stats_by_band": {str(b): list(v) for b, v in calib_stats_by_band.items()},
        "n_test_frames": len(records),
        "min_hazard_patch_frac": MIN_HAZARD_PATCH_FRAC,
        "max_normal_patch_frac": MAX_NORMAL_PATCH_FRAC,
    }
    out_path = RESULTS_DIR / "patch_localization_results_v3_calibrated.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")
    print("\n" + "=" * 70)
    print("v1 (global bank, pooled across whole frame):")
    print("  AUROC 0.9266, AUPR 0.0662, top-1 hit rate 0.30")
    print("v2 (row-band-stratified, no cross-band calibration):")
    print("  AUROC 0.9299, AUPR 0.0657, top-1 hit rate 0.30 (identical hits: 9/30)")
    print("  -- banding alone changed nothing: raw score scale still differs by")
    print("     band, so a global argmax/AUROC still favors whichever band has the")
    print("     largest natural normal-to-normal variance (sky/rooflines), regardless")
    print("     of hazard content.")
    print()
    print("v3 (row-band-stratified + per-band z-score calibration, this run):")
    print(f"  AUROC {patch_auroc:.4f}, AUPR {patch_aupr:.4f}, top-1 hit rate {hit_rate:.4f} "
          f"({hits}/{evaluable})")
    for k in TOP_K_VALUES:
        print(f"  top-{k} hit rate: {topk_rates[k]:.4f} ({topk_hits[k]}/{evaluable})")
    print()
    print("Expected pattern if the top-1 regression is really 'a new false-positive")
    print("class (manhole covers / drain grates) now wins the single argmax, but the")
    print("real hazard is still nearby in the ranking': top-1 hit rate can be flat or")
    print("down while top-3/5/10 climb well above it. If top-k ALSO stays low, the")
    print("localization signal itself is weak, not just out-competed by one nuisance")
    print("class, and that changes what's worth building next.")
    print("=" * 70)


if __name__ == "__main__":
    main()
