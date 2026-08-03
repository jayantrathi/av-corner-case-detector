"""Sanity check on the confirmed top-1 hits before building more demo
content on top of them: crop in tight on exactly what the real ground-truth
mask labels as the hazard for each one, so we can see with our own eyes
whether these are genuine detections or another instance of the
manhole-cover-reads-as-anomalous confound slipping past validation.

Fast -- just loads the cached pipeline result and crops images, no model
inference.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_patch_localization import (
    run_pipeline, CANVAS_SIZE, img_path_to_label_path, load_mask_resized, HAZARD_TRAIN_ID,
)

OUT_DIR = Path("/Volumes/BIggen/AV/results/hit_candidate_crops")


def main():
    result = run_pipeline()
    records = result["records"]

    hit_candidates = []
    for r in records:
        if not r["is_hazard_frame"]:
            continue
        gt = r["patch_gt"]
        if not (gt == 1).any():
            continue
        grid = r["score_grid"]
        i_max, j_max = np.unravel_index(np.argmax(grid), grid.shape)
        if gt[i_max, j_max] == 1:
            hit_candidates.append(r)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(hit_candidates)} confirmed top-1 hits\n")

    for k, r in enumerate(hit_candidates):
        path = r["path"]
        scene = r["scene"]

        label_path = img_path_to_label_path(Path(path))
        mask = load_mask_resized(label_path)
        if mask is None:
            print(f"  hit {k}: scene={scene} -- could not load mask, skipping")
            continue
        hazard_mask = (mask == HAZARD_TRAIN_ID)
        ys, xs = np.where(hazard_mask)
        if len(xs) == 0:
            print(f"  hit {k}: scene={scene} -- no hazard pixels in mask?? (unexpected)")
            continue
        gt_box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        img = Image.open(path).convert("RGB").resize(CANVAS_SIZE, Image.Resampling.LANCZOS)

        gx0, gy0, gx1, gy1 = gt_box
        pad = 120
        cx0, cy0 = max(0, gx0 - pad), max(0, gy0 - pad)
        cx1, cy1 = min(CANVAS_SIZE[0], gx1 + pad), min(CANVAS_SIZE[1], gy1 + pad)
        crop = img.crop((cx0, cy0, cx1, cy1))
        scale = max(1, 500 // max(1, crop.width))
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)

        out_path = OUT_DIR / f"hit_{k:02d}_{scene}.jpg"
        crop.save(out_path, quality=92)
        print(f"  hit {k}: scene={scene}  file={Path(path).name}  gt_box={gt_box}  -> {out_path.name}")

    print(f"\nSaved zoomed ground-truth crops to {OUT_DIR}")
    print("Each crop is centered on exactly what the REAL Lost & Found mask labels")
    print("as hazard, zoomed in and upscaled -- this is what we validated the box")
    print("against, independent of any rendering/padding choices in the demo script.")


if __name__ == "__main__":
    main()
