"""Re-run the CODA hazard-box check with the RbA scorer instead of the k-NN
patch approach -- specifically to see whether the manhole-cover and cone/
bollard false positives (documented in results/coda_demo/) are actually
resolved, not just moved to a different nuisance object the way pooling the
reference bank partially did.

Same visual discipline as run_demo_coda.py: red box = model's detection,
green box = real annotated hazard, drawn on the same frames so a hit or
miss is visible directly. No fitting step this time -- RbA scores every
image independently and immediately.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate_patch_localization import CANVAS_SIZE
from evaluate_rba_lost_and_found import largest_region_from_peak, REGION_TOP_PERCENTILE
from src.scoring.mask2former_rba import RbAScorer
from run_demo_coda import load_coda_annotations, draw_gt_boxes, CODA_ROOT
from build_demo_frames import (
    load_font, measure_banner_size, ALERT_TITLE, ALERT_SUBTITLE,
    BOX_COLOR, BOX_WIDTH, BANNER_COLOR, TEXT_COLOR, SUBTEXT_COLOR,
)

OUT_DIR = Path("/Volumes/BIggen/AV/results/coda_demo_rba")

# The exact frames that broke the k-NN approach, called out by name so
# they're easy to find and eyeball first instead of scrolling through all 14.
KNOWN_PROBLEM_FRAMES = {
    "000002_1616005443000.jpg": "was on a traffic cone",
    "000002_1616005519499.jpg": "was on a manhole cover",
    "000006_1616008913199.jpg": "was on a traffic cone",
    "000032_1616104982899.jpg": "was on bollards",
    "000039_1616180870000.jpg": "was on a traffic cone",
}


def render_rba_alert(img_path, region_mask, out_path):
    img = Image.open(img_path).convert("RGB").resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
    canvas = img.convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    ys, xs = np.where(region_mask)
    if len(xs) == 0:
        draw.rectangle([14, 14, 230, 46], fill=(20, 20, 20, 180))
        draw.text((22, 20), "SCANNING...", fill=(120, 220, 120))
        canvas.convert("RGB").save(out_path, quality=94)
        return
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad_x, pad_y = int((x1 - x0) * 0.15) + 6, int((y1 - y0) * 0.15) + 6
    x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1, y1 = min(CANVAS_SIZE[0], x1 + pad_x), min(CANVAS_SIZE[1], y1 + pad_y)

    draw.rectangle([x0, y0, x1, y1], outline=BOX_COLOR, width=BOX_WIDTH)
    tick = 18
    for cx, cy, dx, dy in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        draw.line([(cx, cy), (cx + dx * tick, cy)], fill=BOX_COLOR, width=BOX_WIDTH + 2)
        draw.line([(cx, cy), (cx, cy + dy * tick)], fill=BOX_COLOR, width=BOX_WIDTH + 2)

    title_font, sub_font = load_font(30), load_font(20)
    banner_w, banner_h = measure_banner_size(title_font, sub_font)
    banner_y0 = y0 - banner_h - 10 if y0 - banner_h - 10 > 0 else y1 + 10
    banner_x0 = max(0, min(int(x0), CANVAS_SIZE[0] - banner_w))
    banner = Image.new("RGBA", (banner_w, banner_h), BANNER_COLOR)
    bdraw = ImageDraw.Draw(banner)
    bdraw.text((16, 12), ALERT_TITLE, font=title_font, fill=TEXT_COLOR)
    bdraw.text((16, 54), ALERT_SUBTITLE, font=sub_font, fill=SUBTEXT_COLOR)
    canvas.alpha_composite(banner, (banner_x0, int(banner_y0)))

    canvas.convert("RGB").save(out_path, quality=94)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading frozen Mask2Former (Cityscapes) on {device}...")
    scorer = RbAScorer(device=device)
    print("Loaded.\n")

    images, boxes_by_image = load_coda_annotations()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(boxes_by_image)} CODA hazard-labeled frames...\n")
    n_box_touches_hazard = 0
    for image_id, boxes in boxes_by_image.items():
        meta = images[image_id]
        img_path = CODA_ROOT / "images" / meta["file_name"]
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        rba_map, _ = scorer.score(img, out_size=CANVAS_SIZE)
        # include_hood_box=False: that exclusion is calibrated to exactly
        # where LOST & FOUND's specific camera mount puts the ego hood on
        # screen. CODA is different footage, different vehicle, different
        # camera position -- reusing that box here would silently discard
        # real CODA image content for no reason. The general border margin
        # (edge boundary/receptive-field effect) still applies, since
        # that's a property of the model, not of Lost & Found specifically.
        region_mask, peak_idx = largest_region_from_peak(rba_map, REGION_TOP_PERCENTILE, include_hood_box=False)

        out_path = OUT_DIR / f"hazard_{meta['file_name']}"
        render_rba_alert(img_path, region_mask, out_path)

        canvas = Image.open(out_path).convert("RGB")
        draw_gt_boxes(canvas, boxes, meta["width"], meta["height"])
        canvas.save(out_path, quality=94)

        # quick overlap check against the real hazard box(es) for a printed tally
        sx, sy = CANVAS_SIZE[0] / meta["width"], CANVAS_SIZE[1] / meta["height"]
        touched = False
        for _, (bx, by, bw, bh) in boxes:
            gt_mask = np.zeros(CANVAS_SIZE[::-1], dtype=bool)
            x0, y0, x1, y1 = int(bx * sx), int(by * sy), int((bx + bw) * sx), int((by + bh) * sy)
            gt_mask[y0:y1, x0:x1] = True
            if (region_mask & gt_mask).any():
                touched = True
        n_box_touches_hazard += touched

        flag = " <-- KNOWN PROBLEM FRAME (was wrong before)" if meta["file_name"] in KNOWN_PROBLEM_FRAMES else ""
        print(f"  [{'HIT' if touched else 'miss'}] {meta['file_name']}{flag} -> {out_path.name}")

    print(f"\n{n_box_touches_hazard}/{len(boxes_by_image)} frames: alert region touches the real hazard box")
    print(f"Saved to {OUT_DIR} -- compare directly against results/coda_demo/ (k-NN) and")
    print("results/coda_demo_pooled/ (pooled k-NN) for the same filenames.")
    print("\nCheck these specific frames first -- they're exactly where the old approach failed:")
    for fname, problem in KNOWN_PROBLEM_FRAMES.items():
        print(f"  {fname}: old approach {problem}")


if __name__ == "__main__":
    main()
