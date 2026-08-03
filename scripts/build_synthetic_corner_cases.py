"""Build a same-domain corner-case benchmark by compositing real hazard
objects (extracted from CODA) onto real nuScenes background frames.

Why this exists: comparing whole nuScenes images against whole CODA images
measures which *dataset* a frame came from (camera, city, compression), not
whether anything dangerous is happening. Here, the background is always real
nuScenes (same domain as training data) and the only thing that changes
between a "normal" and "anomalous" frame is the presence of one real,
photographed object that has no standard AV perception class and no known
driving policy -- debris, machinery, a dustbin in the road, a sentry box,
etc. Any detector that separates these is measuring real semantic anomaly,
not a dataset fingerprint.

Uses nuScenes val+test frames as canvases only (train stays 100% clean for
one-class fitting). Multiple composites are generated per hazard crop across
different canvases/positions/scales, so results should be read with the
caveat that only ~22 unique real object appearances underlie the diversity
(disclosed in the eval report, not hidden).
"""

from __future__ import annotations
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.nuscenes_loader import load_nuscenes_frames
from src.data.splits import split_by_scene

HAZARD_DIR = Path("/Volumes/BIggen/AV/data/hazard_crops")
NUSCENES_ROOT = "/Volumes/BIggen/AV/data/nuscenes/v1.0-mini"
OUT_DIR = Path("/Volumes/BIggen/AV/data/synthetic_corner_cases")
CANVAS_SIZE = (1600, 900)  # matches the rest of the pipeline

COMPOSITES_PER_CROP = 15  # 22 crops * 15 = ~330 synthetic anomalies
SEED = 0


def color_match(crop_rgb: np.ndarray, dest_rgb: np.ndarray, alpha: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Nudge crop's per-channel mean/std toward the destination region's
    mean/std (Reinhard-style transfer), partial strength so the object stays
    recognizable but blends in tone."""
    crop_f = crop_rgb.astype(np.float32)
    dest_f = dest_rgb.astype(np.float32)
    mask = alpha > 10

    out = crop_f.copy()
    for c in range(3):
        crop_vals = crop_f[..., c][mask]
        dest_vals = dest_f[..., c]
        if crop_vals.size == 0:
            continue
        c_mean, c_std = crop_vals.mean(), crop_vals.std() + 1e-6
        d_mean, d_std = dest_vals.mean(), dest_vals.std() + 1e-6
        normalized = (crop_f[..., c] - c_mean) / c_std
        matched = normalized * d_std + d_mean
        out[..., c] = crop_f[..., c] * (1 - strength) + matched * strength

    return np.clip(out, 0, 255).astype(np.uint8)


def paste_hazard(canvas: Image.Image, crop_rgba: Image.Image, category: str) -> tuple[Image.Image, dict]:
    """Paste one hazard crop onto a canvas with position/scale heuristics,
    color matching, and a soft drop shadow. Returns composited image + bbox.

    Anchored by the object's BOTTOM edge ("ground contact point"), with a
    hard floor on how high the top edge can reach, so nothing ends up
    floating in the treeline/sky regardless of scale."""
    cw, ch = canvas.size

    # Bottom (ground-contact) anchor: lower part of the frame, where the
    # road surface actually is for a front-facing camera.
    ground_y = int(ch * random.uniform(0.58, 0.82))

    # Hard floor: top of the object may never go above 40% down the frame
    # (roughly the horizon line for this camera), independent of scale.
    horizon_y = int(ch * 0.40)
    max_allowed_h = max(20, ground_y - horizon_y)

    # Desired height from a scale heuristic (bigger when ground_y is lower
    # i.e. closer to camera), then clamp hard against the horizon floor and
    # an absolute cap so nothing dominates the frame.
    ground_frac = (ground_y - int(ch * 0.58)) / max(1, int(ch * 0.82) - int(ch * 0.58))
    scale = 0.30 + ground_frac * (0.75 - 0.30)
    scale *= random.uniform(0.9, 1.1)

    desired_h = max(16, int(crop_rgba.height * scale))
    new_h = min(desired_h, max_allowed_h, int(ch * 0.30))
    new_w = max(16, int(crop_rgba.width * (new_h / crop_rgba.height)))
    new_w = min(new_w, int(cw * 0.30))
    # re-derive height from width cap if width was the binding constraint
    if new_w == int(cw * 0.30):
        new_h = max(16, int(crop_rgba.height * (new_w / crop_rgba.width)))

    crop_resized = crop_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)

    place_x = random.randint(int(cw * 0.15), max(int(cw * 0.15) + 1, int(cw * 0.85) - new_w))
    place_x = max(0, min(place_x, cw - new_w))
    place_y = max(0, min(ground_y - new_h, ch - new_h))

    canvas_rgb = np.array(canvas.convert("RGB"))
    crop_arr = np.array(crop_resized)
    crop_rgb, crop_alpha = crop_arr[..., :3], crop_arr[..., 3]

    dest_region = canvas_rgb[place_y:place_y + new_h, place_x:place_x + new_w]
    if dest_region.shape[:2] == crop_rgb.shape[:2]:
        crop_rgb = color_match(crop_rgb, dest_region, crop_alpha, strength=0.4)

    # Soft drop shadow: dark ellipse right at the ground-contact line, blurred.
    # Anchored to ground_y (not new_h fractions) so it reads as contact
    # even when the object was clamped short by the horizon floor.
    shadow_layer = Image.new("L", canvas.size, 0)
    sdraw = ImageDraw.Draw(shadow_layer)
    shadow_box = [
        place_x + new_w * 0.05, ground_y - new_h * 0.08,
        place_x + new_w * 0.95, ground_y + new_h * 0.12,
    ]
    sdraw.ellipse(shadow_box, fill=140)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=6))

    result = canvas.convert("RGB").copy()
    # apply shadow first
    shadow_rgba = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_rgba.putalpha(shadow_layer)
    black_layer = Image.new("RGB", canvas.size, (0, 0, 0))
    result = Image.composite(black_layer, result, shadow_layer.point(lambda p: int(p * 0.5)))

    # slight blur on crop edges for softer blending, then paste with alpha
    crop_img = Image.fromarray(crop_rgb, mode="RGB")
    alpha_img = Image.fromarray(crop_alpha, mode="L").filter(ImageFilter.GaussianBlur(radius=1))
    result.paste(crop_img, (place_x, place_y), mask=alpha_img)

    bbox = [place_x, place_y, new_w, new_h]
    return result, {"category": category, "bbox": bbox}


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "images").mkdir(exist_ok=True)

    # Load hazard crops
    manifest = json.load(open(HAZARD_DIR / "manifest.json"))
    print(f"Loaded {len(manifest)} hazard crop entries\n")

    # Load nuScenes frames, same scene-level split as the rest of the project
    frames = load_nuscenes_frames(NUSCENES_ROOT)
    splits = split_by_scene(frames, val_frac=0.15, test_frac=0.15, seed=0)
    candidate_canvases = list(splits["val"]) + list(splits["test"])  # train stays untouched
    print(f"Canvas candidate pool (val+test, train excluded): {len(candidate_canvases)} frames")

    # Filter to daytime-bright canvases only. All our hazard crops were
    # photographed in daylight; pasting a daylight object onto a night frame
    # either looks like a ghostly patch (if color-matched) or a glaring
    # obviously-fake bright rectangle (if not) -- neither is a fair,
    # learnable example of "novel object," just a lighting-mismatch artifact.
    # Mean brightness alone is fooled by night scenes with bright lit signage
    # (e.g. a shopping-street scene at night can average >70 due to a few
    # glaring windows). Also require a low fraction of very-dark pixels,
    # calibrated against this dataset's actual day (mean~106, dark_px~5-15%)
    # vs night (mean~43-73, dark_px~40-66%) scenes.
    BRIGHTNESS_THRESHOLD = 90
    DARK_PIXEL_FRAC_THRESHOLD = 0.20
    canvas_frames = []
    for f in candidate_canvases:
        try:
            img = np.array(Image.open(f.path).convert("L").resize((160, 90)))
            mean_b = img.mean()
            dark_frac = (img < 50).mean()
            if mean_b >= BRIGHTNESS_THRESHOLD and dark_frac <= DARK_PIXEL_FRAC_THRESHOLD:
                canvas_frames.append(f)
        except Exception:
            continue
    print(f"Canvas pool after daytime filter: {len(canvas_frames)} frames\n")

    results = []
    img_counter = 0

    for crop_entry in manifest:
        crop_path = HAZARD_DIR / crop_entry["crop_file"]
        crop_rgba = Image.open(crop_path).convert("RGBA")
        category = crop_entry["category"]

        chosen_canvases = random.sample(canvas_frames, min(COMPOSITES_PER_CROP, len(canvas_frames)))

        for canvas_frame in chosen_canvases:
            try:
                canvas = Image.open(canvas_frame.path).convert("RGB").resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
            except Exception as e:
                continue

            composite, meta = paste_hazard(canvas, crop_rgba, category)

            out_name = f"synth_{img_counter:04d}_{category}.jpg"
            composite.save(OUT_DIR / "images" / out_name, quality=92)

            results.append({
                "file": out_name,
                "category": category,
                "bbox_xywh": meta["bbox"],
                "source_crop": crop_entry["crop_file"],
                "canvas_scene": str(canvas_frame.scene_id),
                "canvas_path": canvas_frame.path,
                "label": "anomaly",
            })
            img_counter += 1

        print(f"  {category:20s} -> {len(chosen_canvases)} composites")

    manifest_out = OUT_DIR / "synthetic_manifest.json"
    with open(manifest_out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nGenerated {len(results)} synthetic corner-case images")
    print(f"Saved to {OUT_DIR / 'images'}")
    print(f"Manifest: {manifest_out}\n")

    from collections import Counter
    scene_counts = Counter(r["canvas_scene"] for r in results)
    print(f"Composites span {len(scene_counts)} distinct canvas scenes (out of {len(set(f.scene_id for f in canvas_frames))} available)")


if __name__ == "__main__":
    main()
