"""Run the demo (box + alert banner) on CODA -- real Chinese dashcam driving
photos, not staged web imagery. This is the honest version of the RoadAnomaly21
test: same camera domain as the training data's spirit (actual road scenes
shot from a vehicle), different country, real annotated hazard objects.

Same reused, already-fitted Lost & Found reference bank as
run_demo_roadanomaly21.py -- no refitting. Also draws the REAL ground-truth
box (thin green outline, from CODA's actual COCO-style annotations) next to
the model's box (thick red) on every frame that has one, so a hit or a miss
is visible directly in the image, not buried in a metrics file -- this is
exactly the check that caught the RoadAnomaly21 problem, done up front this
time instead of after the fact.

"Hazard" here means CODA's obstruction-type categories, matching the same
operational definition used earlier in this project for the CODA YOLO
verification (Part A): debris, suitcase, dustbin, concrete_block, machinery,
chair, phone_booth, basket, plus construction_vehicle, cart, sentry_box.
Ordinary traffic (car, pedestrian, cyclist, traffic signs/lights/cones) is
NOT treated as ground-truth hazard -- the reference bank already contains
plenty of ordinary road/vehicle patches, so a car being "detected" would not
be a meaningful corner case.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate_patch_localization import run_pipeline, PatchResNetEmbedder, CANVAS_SIZE
from run_demo_roadanomaly21 import score_image, render_alert_frame

CODA_ROOT = Path("/Volumes/BIggen/AV/data/coda/CODA/sample")
OUT_DIR = Path("/Volumes/BIggen/AV/results/coda_demo")

HAZARD_CATEGORY_NAMES = {
    "debris", "suitcace", "dustbin", "concrete_block", "machinery", "chair",
    "phone_booth", "basket", "construction_vehicle", "cart", "sentry_box",
}


def load_coda_annotations():
    with open(CODA_ROOT / "corner_case.json") as f:
        d = json.load(f)
    cats = {c["id"]: c["name"] for c in d["categories"]}
    images = {im["id"]: im for im in d["images"]}
    boxes_by_image = defaultdict(list)
    for ann in d["annotations"]:
        name = cats.get(ann["category_id"])
        if name in HAZARD_CATEGORY_NAMES:
            boxes_by_image[ann["image_id"]].append((name, ann["bbox"]))  # bbox = [x, y, w, h]
    return images, boxes_by_image


def draw_gt_boxes(canvas, boxes, orig_w, orig_h):
    """boxes are in the ORIGINAL image's pixel coords -- rescale to CANVAS_SIZE."""
    sx, sy = CANVAS_SIZE[0] / orig_w, CANVAS_SIZE[1] / orig_h
    draw = ImageDraw.Draw(canvas)
    for name, (x, y, w, h) in boxes:
        x0, y0, x1, y1 = x * sx, y * sy, (x + w) * sx, (y + h) * sy
        draw.rectangle([x0, y0, x1, y1], outline=(60, 255, 60), width=3)
        draw.text((x0 + 3, max(0, y0 - 18)), f"real: {name}", fill=(60, 255, 60))


def main():
    result = run_pipeline()
    scorers_by_band = result["scorers_by_band"]
    calib_stats_by_band = result["calib_stats_by_band"]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = PatchResNetEmbedder().to(device)
    model.eval()

    images, boxes_by_image = load_coda_annotations()
    print(f"CODA: {len(images)} images total, {len(boxes_by_image)} have a real hazard-category annotation\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_hazard_images = 0
    for image_id, meta in images.items():
        img_path = CODA_ROOT / "images" / meta["file_name"]
        if not img_path.exists():
            continue
        has_hazard = image_id in boxes_by_image
        if has_hazard:
            n_hazard_images += 1

        score_grid = score_image(model, scorers_by_band, calib_stats_by_band, img_path, device)
        tag = "hazard" if has_hazard else "normal"
        out_path = OUT_DIR / f"{tag}_{meta['file_name']}"
        render_alert_frame(img_path, score_grid, out_path)

        if has_hazard:
            canvas = Image.open(out_path).convert("RGB")
            draw_gt_boxes(canvas, boxes_by_image[image_id], meta["width"], meta["height"])
            canvas.save(out_path, quality=94)

        print(f"  [{tag}] {meta['file_name']} -> {out_path.name}"
              + (f"  ({len(boxes_by_image[image_id])} real hazard box(es))" if has_hazard else ""))

    print(f"\nDone. {len(images)} frames saved to {OUT_DIR} ({n_hazard_images} have real hazard ground truth).")
    print("Red box = model's detection. Green box = real annotated hazard (only on hazard_* files).")
    print("Look at the hazard_*.jpg files first -- that's where hit/miss is checkable by eye.")


if __name__ == "__main__":
    main()
