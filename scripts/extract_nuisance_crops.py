"""Extract real cone/bollard crops from CODA -- the exact objects that keep
breaking both the k-NN and RbA anomaly detectors.

Context: extract_hazard_crops.py deliberately excluded cone/bollard/barrier
from the "novel" set, on the assumption they're "extremely common,
well-handled street furniture." That assumption turned out to be wrong in
the way that matters here -- Cityscapes (19 classes), Mapillary Vistas (65
classes), and COCO (80 classes, what our YOLOv8n detector was trained on)
ALL lack a "traffic cone" or "bollard" category. Checked directly against
each checkpoint's own id2label/class list, not assumed. So any detector
built on "is this pixel/region claimed by a known class" will structurally
flag cones and bollards as anomalous, regardless of which of these three
checkpoints backs it -- confirmed empirically across 14 real CODA frames
(run_demo_coda_rba.py) where cones/bollards were the only remaining false-
positive category once the whole-frame-box bug was fixed.

CODA's OWN taxonomy, however, already has real, hand-labeled bounding boxes
for exactly these two categories (under supercategory "traffic_facility"):
  - traffic_cone: 136 instances in the 100-image sample
  - bollard:      120 instances in the 100-image sample
No synthetic data, no scraping -- this is real, already-downloaded ground
truth for the exact failure mode. This script pulls those crops out as the
"known common object, suppress the alert" reference set for a lightweight
classifier that runs only on RbA-flagged regions (see
build_nuisance_classifier.py) -- not a full object detector across the
whole frame, just a cheap check on candidates RbA already flagged.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PIL import Image

CODA_ROOT = Path("/sessions/nifty-awesome-brown/mnt/AV/data/coda/CODA/sample")
OUT_DIR = Path("/sessions/nifty-awesome-brown/mnt/AV/data/nuisance_crops")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUISANCE_CATEGORY_NAMES = {"traffic_cone", "bollard"}
PAD_FRAC = 0.08  # same padding convention as extract_hazard_crops.py


def main():
    d = json.load(open(CODA_ROOT / "corner_case.json"))
    cat_names = {c["id"]: c["name"] for c in d["categories"]}
    images_by_id = {im["id"]: im for im in d["images"]}

    nuisance_anns = [
        a for a in d["annotations"]
        if cat_names.get(a["category_id"]) in NUISANCE_CATEGORY_NAMES
    ]
    print(f"Found {len(nuisance_anns)} cone/bollard annotations in the CODA sample\n")

    manifest = []
    n_skipped = 0
    for idx, ann in enumerate(nuisance_anns):
        img_meta = images_by_id[ann["image_id"]]
        img_path = CODA_ROOT / "images" / img_meta["file_name"]
        if not img_path.exists():
            n_skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            n_skipped += 1
            continue

        x, y, bw, bh = ann["bbox"]
        pad_x, pad_y = bw * PAD_FRAC, bh * PAD_FRAC
        x0, y0 = max(0, int(x - pad_x)), max(0, int(y - pad_y))
        x1, y1 = min(img.width, int(x + bw + pad_x)), min(img.height, int(y + bh + pad_y))

        if x1 - x0 < 8 or y1 - y0 < 8:
            n_skipped += 1
            continue

        crop = img.crop((x0, y0, x1, y1))
        category = cat_names[ann["category_id"]]
        out_name = f"{idx:03d}_{category}_{img_meta['file_name'].replace('.jpg', '')}.png"
        crop.save(OUT_DIR / out_name)

        manifest.append({
            "crop_file": out_name,
            "category": category,
            "source_image": img_meta["file_name"],
            "source_bbox_xywh": ann["bbox"],
            "crop_size": [crop.width, crop.height],
        })

    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    from collections import Counter
    cat_counts = Counter(m["category"] for m in manifest)
    print(f"Saved {len(manifest)} crops ({n_skipped} skipped -- missing image or degenerate bbox)")
    for cat, count in cat_counts.most_common():
        print(f"  {count:3d}  {cat}")
    print(f"\nManifest: {manifest_path}")

    # Size sanity check -- if most crops are tiny (<20px), a frozen-ResNet
    # embedding classifier may not have much to work with; worth knowing
    # before building on top of this rather than after.
    sizes = np.array([m["crop_size"] for m in manifest])
    if len(sizes):
        print(f"\nCrop size stats (w, h): median={np.median(sizes, axis=0)}, "
              f"min={sizes.min(axis=0)}, max={sizes.max(axis=0)}")
        n_tiny = int(((sizes[:, 0] < 20) | (sizes[:, 1] < 20)).sum())
        print(f"{n_tiny}/{len(sizes)} crops have a dimension under 20px (may be too small to embed meaningfully)")


if __name__ == "__main__":
    main()
