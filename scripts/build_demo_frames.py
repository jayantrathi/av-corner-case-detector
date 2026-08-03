"""Build the actual demo: a box locks onto the flagged region, with an
on-screen alert label -- "CORNER CASE DETECTED, no policy for this object."

This is deliberately NOT the heatmap-overlay style used in
evaluate_patch_localization.py -- that was for diagnosing the model (does
the score correctly rank hazard patches), not for showing it off. This is
what an actual alert would look like.

Uses ONLY genuine model successes on real, held-out Lost & Found frames:
  - "clean" examples: the model's own #1-ranked region (argmax patch) is
    itself inside the real, human-labeled hazard -- no cherry-picking beyond
    "did the detector's own top choice land correctly."
  - "top-3" examples: the argmax missed, but one of the top-3 patches is a
    real hit -- included as a second tier, boxed around that specific patch
    (not the argmax), clearly distinguishable in the saved filename.

This is a curated best-case reel, not a claim of average-case performance --
that's the explicit, agreed framing (top-1 hit rate on the full held-out set
is 20%; top-3 is 47%). The honest numbers live in
patch_localization_results_v3_calibrated.json; this script's job is to show
the system working when it works, clearly labeled as such.

The box itself is NOT copy-pasted from ground truth -- it's a connected
region grown from the hit patch via flood-fill over the model's own score
grid (all adjacent patches within threshold_frac of the hit patch's score,
see connected_region_bbox), so it's a genuine model output, not a look-alike.
Started at threshold_frac=0.75 -- too strict, it only grew to the single
hottest sub-patch of an object (e.g. just a toy car's steering wheel, not
the whole car) instead of the full extent. Loosened to 0.55.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate_patch_localization import run_pipeline, grid_cell_bbox, CANVAS_SIZE

RESULTS_DIR = Path("/Volumes/BIggen/AV/results")
DEMO_DIR = RESULTS_DIR / "demo_frames"

BOX_COLOR = (255, 40, 40)
BOX_WIDTH = 5
BANNER_COLOR = (20, 20, 20, 215)
TEXT_COLOR = (255, 60, 60)
SUBTEXT_COLOR = (235, 235, 235)


def connected_region_bbox(score_grid: np.ndarray, seed_i: int, seed_j: int, threshold_frac: float = 0.55):
    """Flood-fill outward from (seed_i, seed_j) to all 4-connected patches
    whose score is within threshold_frac of the seed's score. Returns the
    grid-space bbox (i0, j0, i1, j1) of the resulting blob. This is a real
    model-derived region, not a lookup of the ground-truth box."""
    hf, wf = score_grid.shape
    threshold = score_grid[seed_i, seed_j] * threshold_frac
    visited = np.zeros_like(score_grid, dtype=bool)
    stack = [(seed_i, seed_j)]
    visited[seed_i, seed_j] = True
    cells = [(seed_i, seed_j)]
    while stack:
        i, j = stack.pop()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < hf and 0 <= nj < wf and not visited[ni, nj] and score_grid[ni, nj] >= threshold:
                visited[ni, nj] = True
                stack.append((ni, nj))
                cells.append((ni, nj))
    rows = [c[0] for c in cells]
    cols = [c[1] for c in cells]
    return min(rows), min(cols), max(rows) + 1, max(cols) + 1


FONT_CANDIDATES = (
    "Arial Bold.ttf", "Helvetica.ttc", "DejaVuSans-Bold.ttf",  # bare names (Linux/fontconfig)
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",       # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
)


def load_font(size: int):
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


ALERT_TITLE = "CORNER CASE DETECTED"
ALERT_SUBTITLE = "unrecognized object -- no established driving policy"


def measure_banner_size(title_font, sub_font, pad: int = 16) -> tuple[int, int]:
    """Size the banner to fit the actual alert text -- the earlier version
    used a hardcoded 340px minimum width, which was narrower than the
    rendered title text at this font size and clipped it mid-word."""
    title_w = title_font.getlength(ALERT_TITLE)
    sub_w = sub_font.getlength(ALERT_SUBTITLE)
    width = int(max(title_w, sub_w) + pad * 2)
    height = 92
    return width, height


def render_demo_frame(record, seed_i, seed_j, out_path, tier_label):
    """Draw one clean alert-style frame: red box around the model-derived
    region, dark banner with alert text underneath."""
    img = Image.open(record["path"]).convert("RGB").resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
    grid = record["score_grid"]
    hf, wf = grid.shape

    i0, j0, i1, j1 = connected_region_bbox(grid, seed_i, seed_j)
    x0, y0, _, _ = grid_cell_bbox(i0, j0, hf, wf, CANVAS_SIZE[1], CANVAS_SIZE[0])
    _, _, x1, y1 = grid_cell_bbox(i1 - 1, j1 - 1, hf, wf, CANVAS_SIZE[1], CANVAS_SIZE[0])

    # Pad the box out a bit so it visually contains the object rather than
    # tightly hugging the patch grid.
    pad_x = int((x1 - x0) * 0.25) + 8
    pad_y = int((y1 - y0) * 0.25) + 8
    x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1, y1 = min(CANVAS_SIZE[0], x1 + pad_x), min(CANVAS_SIZE[1], y1 + pad_y)

    canvas = img.convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    # box with corner ticks (HUD style) rather than a plain rectangle
    draw.rectangle([x0, y0, x1, y1], outline=BOX_COLOR, width=BOX_WIDTH)
    tick = 18
    for cx, cy, dx, dy in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        draw.line([(cx, cy), (cx + dx * tick, cy)], fill=BOX_COLOR, width=BOX_WIDTH + 2)
        draw.line([(cx, cy), (cx, cy + dy * tick)], fill=BOX_COLOR, width=BOX_WIDTH + 2)

    # alert banner above the box (or below, if too close to the top edge)
    title_font = load_font(30)
    sub_font = load_font(20)
    banner_w, banner_h = measure_banner_size(title_font, sub_font)
    banner_y0 = y0 - banner_h - 10 if y0 - banner_h - 10 > 0 else y1 + 10
    banner_x0 = min(int(x0), CANVAS_SIZE[0] - banner_w)  # keep on-screen if box is near the right edge
    banner_x0 = max(0, banner_x0)
    banner = Image.new("RGBA", (banner_w, banner_h), BANNER_COLOR)
    bdraw = ImageDraw.Draw(banner)
    bdraw.text((16, 12), ALERT_TITLE, font=title_font, fill=TEXT_COLOR)
    bdraw.text((16, 54), ALERT_SUBTITLE, font=sub_font, fill=SUBTEXT_COLOR)
    canvas.alpha_composite(banner, (banner_x0, int(banner_y0)))

    # small tier tag, bottom-left corner, for our own honesty/bookkeeping
    tag_font = load_font(16)
    draw.text((14, CANVAS_SIZE[1] - 28), tier_label, font=tag_font, fill=(255, 255, 255))

    canvas.convert("RGB").save(out_path, quality=94)


def main():
    result = run_pipeline()
    records = result["records"]
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    top1_examples, top3_examples = [], []
    for r in records:
        if not r["is_hazard_frame"]:
            continue
        gt = r["patch_gt"]
        if not (gt == 1).any():
            continue
        grid = r["score_grid"]
        i_max, j_max = np.unravel_index(np.argmax(grid), grid.shape)
        if gt[i_max, j_max] == 1:
            top1_examples.append((r, i_max, j_max))
            continue
        # top-3: find the highest-scoring TRUE patch among the top-3 overall
        flat_scores = grid.flatten()
        order = np.argsort(-flat_scores)[:3]
        for idx in order:
            i, j = np.unravel_index(idx, grid.shape)
            if gt[i, j] == 1:
                top3_examples.append((r, i, j))
                break

    print(f"Found {len(top1_examples)} clean top-1 hits, {len(top3_examples)} top-3-only hits\n")

    saved = 0
    for k, (r, i, j) in enumerate(top1_examples):
        out_path = DEMO_DIR / f"top1_hit_{k:02d}.jpg"
        render_demo_frame(r, i, j, out_path, tier_label="model's #1-ranked region -- correct")
        print(f"  saved {out_path.name}")
        saved += 1

    for k, (r, i, j) in enumerate(top3_examples):
        out_path = DEMO_DIR / f"top3_hit_{k:02d}.jpg"
        render_demo_frame(r, i, j, out_path, tier_label="model's top-3 candidate regions -- this one correct")
        print(f"  saved {out_path.name}")
        saved += 1

    print(f"\nSaved {saved} demo frames to {DEMO_DIR}")
    print("top1_hit_*.jpg = model's single best guess was correct (rarer, ~20% of all hazard frames)")
    print("top3_hit_*.jpg = correct region was in the model's top 3 candidates, not #1 (~47% of frames)")
    print("These are curated successes for the demo reel, not average-case behavior --")
    print("the real numbers (all frames, not just hits) are in patch_localization_results_v3_calibrated.json")


if __name__ == "__main__":
    main()
