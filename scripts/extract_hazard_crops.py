"""Extract real hazard-object crops from CODA.

Filters CODA's ground-truth annotations down to categories that represent
genuinely novel objects for an AV perception stack -- things with no
standard trained class and no established driving policy (debris, machinery,
a dustbin in the road, a sentry box, a suitcase, a cart, a construction
vehicle). Explicitly EXCLUDES common/well-handled classes (pedestrian,
cyclist, car, cone, bollard, barrier, traffic sign/light) even though those
are statistically "rare" in a small sample -- rarity in the dataset is not
the same as being unrecognized by the car.

Each crop is saved as an RGBA PNG with a feathered alpha edge (so it composites
cleanly later without a hard rectangular seam), plus a JSON manifest with
category, source image, and bbox.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

CODA_ROOT = Path("/sessions/nifty-awesome-brown/mnt/AV/data/coda/CODA/sample")
OUT_DIR = Path("/sessions/nifty-awesome-brown/mnt/AV/data/hazard_crops")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Genuinely novel categories: no standard AV perception class, no known
# driving policy. Deliberately excludes pedestrian/cyclist/wheelchair/
# stroller/moped/bicycle/car/truck/bus/motorcycle/tricycle (known classes,
# known policy: yield/stop/avoid) and cone/bollard/barrier/sign/light
# (extremely common, well-handled street furniture).
NOVEL_CATEGORY_NAMES = {
    "debris",
    "machinery",
    "dustbin",
    "sentry_box",
    "suitcace",  # sic, typo in CODA's own taxonomy
    "cart",
    "construction_vehicle",
    "concrete_block",
    "chair",
    "phone_booth",
    "basket",
}

FEATHER_PX = 12
PAD_FRAC = 0.08  # padding around bbox as a fraction of box size


def feathered_alpha_mask(w: int, h: int, feather: int) -> np.ndarray:
    """Alpha mask: 255 in the interior, fading to 0 within `feather` px of the edge."""
    alpha = np.full((h, w), 255, dtype=np.uint8)
    if feather <= 0:
        return alpha
    ramp_x = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
    ramp_y = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    ramp = np.minimum.outer(ramp_y, ramp_x)
    fade = np.clip(ramp / feather, 0, 1)
    return (alpha.astype(np.float32) * fade).astype(np.uint8)


def main():
    d = json.load(open(CODA_ROOT / "corner_case.json"))
    cat_names = {c["id"]: c["name"] for c in d["categories"]}
    images_by_id = {im["id"]: im for im in d["images"]}

    novel_anns = [
        a for a in d["annotations"]
        if cat_names.get(a["category_id"]) in NOVEL_CATEGORY_NAMES
    ]
    print(f"Found {len(novel_anns)} annotations in novel categories\n")

    manifest = []
    for idx, ann in enumerate(novel_anns):
        img_meta = images_by_id[ann["image_id"]]
        img_path = CODA_ROOT / "images" / img_meta["file_name"]
        if not img_path.exists():
            print(f"  [skip] missing image {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [skip] failed to load {img_path}: {e}")
            continue

        x, y, bw, bh = ann["bbox"]
        pad_x = bw * PAD_FRAC
        pad_y = bh * PAD_FRAC
        x0 = max(0, int(x - pad_x))
        y0 = max(0, int(y - pad_y))
        x1 = min(img.width, int(x + bw + pad_x))
        y1 = min(img.height, int(y + bh + pad_y))

        if x1 - x0 < 8 or y1 - y0 < 8:
            print(f"  [skip] degenerate crop for ann {ann['id']}")
            continue

        crop = img.crop((x0, y0, x1, y1))
        crop_arr = np.array(crop)

        alpha = feathered_alpha_mask(crop.width, crop.height, FEATHER_PX)
        rgba = np.dstack([crop_arr, alpha])
        crop_rgba = Image.fromarray(rgba, mode="RGBA")

        category = cat_names[ann["category_id"]]
        out_name = f"{idx:03d}_{category}_{img_meta['file_name'].replace('.jpg', '')}.png"
        out_path = OUT_DIR / out_name
        crop_rgba.save(out_path)

        manifest.append({
            "crop_file": out_name,
            "category": category,
            "source_image": img_meta["file_name"],
            "source_bbox_xywh": ann["bbox"],
            "crop_size": [crop.width, crop.height],
        })

        print(f"  [{idx:2d}] {category:20s} {crop.width:4d}x{crop.height:4d}  <- {img_meta['file_name']}")

    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved {len(manifest)} hazard crops to {OUT_DIR}")
    print(f"Manifest: {manifest_path}\n")

    # Category breakdown
    from collections import Counter
    cat_counts = Counter(m["category"] for m in manifest)
    print("Category breakdown:")
    for cat, count in cat_counts.most_common():
        print(f"  {count:3d}  {cat}")


if __name__ == "__main__":
    main()
