"""Just run the demo (box + alert banner) on the 10 RoadAnomaly21 images --
no evaluation harness, no AUROC/AUPR, no JSON report. That scoring machinery
in evaluate_roadanomaly21.py was for validating the approach, not for using
it -- this script is the "use it" path.

The fitted reference bank + per-band calibration (from Lost & Found) is
built ONCE and cached to disk by run_pipeline(). Scoring a new image is just:
forward pass through the frozen backbone -> per-patch kNN distance to the
already-fitted bank -> z-normalize with the already-fitted calib stats ->
argmax -> flood-fill a box -> draw it. No refitting, no evaluation step,
same handful of lines regardless of which image or dataset it's pointed at.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate_patch_localization import (
    run_pipeline, load_image_tensor, row_to_band, N_ROW_BANDS, CANVAS_SIZE, PatchResNetEmbedder,
)
from src.scoring.patch_embedder import feature_map_to_patches, grid_cell_bbox
from build_demo_frames import (
    connected_region_bbox, load_font, measure_banner_size,
    ALERT_TITLE, ALERT_SUBTITLE, BOX_COLOR, BOX_WIDTH, BANNER_COLOR, TEXT_COLOR, SUBTEXT_COLOR,
)

RA21_ROOT = Path("/Volumes/BIggen/AV/data/roadanomaly21")
OUT_DIR = Path("/Volumes/BIggen/AV/results/roadanomaly21_demo")


@torch.no_grad()
def score_image(model, scorers_by_band, calib_stats_by_band, img_path, device):
    """The actual inference path -- this is all that runs per new image."""
    t = load_image_tensor(str(img_path), device)
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
        band_scores = scorers_by_band[band].score(patches[flat_lo:flat_hi])
        calib_mean, calib_std = calib_stats_by_band[band]
        score_grid[row_lo:row_hi] = ((band_scores - calib_mean) / calib_std).reshape(row_hi - row_lo, wf)
    return score_grid


def render_alert_frame(img_path, score_grid, out_path):
    img = Image.open(img_path).convert("RGB").resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
    hf, wf = score_grid.shape
    i_max, j_max = np.unravel_index(np.argmax(score_grid), score_grid.shape)

    i0, j0, i1, j1 = connected_region_bbox(score_grid, i_max, j_max)
    x0, y0, _, _ = grid_cell_bbox(i0, j0, hf, wf, CANVAS_SIZE[1], CANVAS_SIZE[0])
    _, _, x1, y1 = grid_cell_bbox(i1 - 1, j1 - 1, hf, wf, CANVAS_SIZE[1], CANVAS_SIZE[0])
    pad_x, pad_y = int((x1 - x0) * 0.25) + 8, int((y1 - y0) * 0.25) + 8
    x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1, y1 = min(CANVAS_SIZE[0], x1 + pad_x), min(CANVAS_SIZE[1], y1 + pad_y)

    canvas = img.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
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
    result = run_pipeline()  # loads from cache -- already fitted, nothing recomputed here
    scorers_by_band = result["scorers_by_band"]
    calib_stats_by_band = result["calib_stats_by_band"]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = PatchResNetEmbedder().to(device)
    model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted((RA21_ROOT / "images").glob("validation*.jpg"))
    print(f"Running the demo on {len(images)} RoadAnomaly21 images...\n")
    for img_path in images:
        score_grid = score_image(model, scorers_by_band, calib_stats_by_band, img_path, device)
        out_path = OUT_DIR / img_path.name
        render_alert_frame(img_path, score_grid, out_path)
        print(f"  {img_path.name} -> {out_path}")

    print(f"\nDone. {len(images)} demo frames saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
