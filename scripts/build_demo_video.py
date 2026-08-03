"""Build the actual "live" demo: a full real driving sequence, played frame
by frame, with the alert box appearing once the model's own score crosses
threshold as the car approaches the hazard -- and staying locked on for the
rest of the approach.

Reuses the SAME fitted row-banded + z-calibrated reference (via
run_pipeline()'s cache) that evaluate_patch_localization.py validated --
this is not a new model, just the validated one run across every frame of
one real sequence instead of a capped, shuffled test set.

Picks one sequence automatically: among the frames where the model's own
#1-ranked region was already confirmed correct (a genuine hit, not
cherry-picked from ground truth), it looks up that frame's full scene
directory on disk (Lost & Found stores each approach as ~20-40 consecutive
frames) and scores every frame in it. The earlier frames -- where the object
is small and far away -- mostly won't trigger, which is realistic, not a
flaw: showing the alert switch on partway through the approach, once the
object is actually resolvable, is the honest story, not a fudge.

Output: a folder of numbered frames + an animated GIF (zero extra
dependencies, works everywhere) + a printed ffmpeg command for an mp4 if
ffmpeg is installed locally.
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
    run_pipeline, score_test_frames, PatchResNetEmbedder, grid_cell_bbox,
    CANVAS_SIZE, LAF_ROOT, SEQ_RE, scene_id_for,
)
from build_demo_frames import (
    connected_region_bbox, load_font, measure_banner_size, ALERT_TITLE, ALERT_SUBTITLE,
    BOX_COLOR, BOX_WIDTH, BANNER_COLOR, TEXT_COLOR, SUBTEXT_COLOR,
)

RESULTS_DIR = Path("/Volumes/BIggen/AV/results")
FRAMES_DIR = RESULTS_DIR / "demo_video_frames"
GIF_PATH = RESULTS_DIR / "demo_video.gif"

DETECTION_THRESHOLD_FRAC = 0.6  # trigger at 60% of the known-good hit frame's z-score
FRAME_MS = 140                  # ~7fps -- Lost & Found sequences are short, keep it watchable
HOLD_LAST_FRAMES = 6            # repeat the final alert frame so viewers can read it
WINDOW_RADIUS_FRAC = 0.12       # search neighborhood radius as a fraction of grid height/width
# (tightened from 0.18 -- the wider radius was still catching a manhole
# cover sitting near the real object in a busy parking-lot sequence)
MIN_CONSECUTIVE_FRAMES = 2      # require this many frames IN A ROW above threshold before
# latching the alert -- a single frame crossing threshold (a leaf, a texture
# blip) shouldn't be enough to trigger; a real approaching object stays
# above threshold consistently as the car gets closer, noise doesn't.
# (Kept at 2, not higher -- these sequences are short, ~15-40 frames total,
# and the tracked score genuinely dips frame to frame even near the real
# object; requiring too many in a row risks never latching at all.)


def windowed_argmax(score_grid: np.ndarray, center_i: int, center_j: int, radius_i: int, radius_j: int):
    """Argmax restricted to a local neighborhood around (center_i, center_j),
    NOT the whole grid. This is the fix for the manhole-cover bug: scoring
    every frame independently with a global argmax lets the box jump to
    whatever's most anomalous ANYWHERE in that frame -- a manhole cover on
    the other side of the lot, say -- even in a sequence where we already
    validated the correct region for one frame. Restricting the search to
    near where the object was last confirmed keeps the box tracking the
    SAME real object across the sequence instead of re-picking a new target
    every frame."""
    hf, wf = score_grid.shape
    i0, i1 = max(0, center_i - radius_i), min(hf, center_i + radius_i + 1)
    j0, j1 = max(0, center_j - radius_j), min(wf, center_j + radius_j + 1)
    window = score_grid[i0:i1, j0:j1]
    wi, wj = np.unravel_index(np.argmax(window), window.shape)
    return i0 + wi, j0 + wj


def find_scene_dir(img_root: Path, scene_id: str) -> Path | None:
    """Lost & Found's on-disk directories are per-LOCATION, not per-scene --
    a single location directory (e.g. .../test/15_Rechbergstr_Deckenpfronn/)
    can hold several distinct approach sequences (000006_*, 000011_*, ...).
    scene_id is "{location}_{seqnum}"; the seqnum is always the last
    underscore-separated group, so strip it to get the directory name."""
    location = scene_id.rsplit("_", 1)[0]
    matches = list(img_root.glob(f"*/{location}"))
    return matches[0] if matches else None


def frames_for_scene(scene_dir: Path, scene_id: str) -> list[Path]:
    """All frames in scene_dir that belong to THIS specific sequence (not
    other sequences sharing the same location directory)."""
    return [p for p in scene_dir.glob("*_leftImg8bit.png") if scene_id_for(p) == scene_id]


def frame_number(path: Path) -> int:
    m = SEQ_RE.match(path.name)
    return int(m.group(3)) if m else 0


def render_frame(record, out_path, alert_on: bool, seed_i: int = None, seed_j: int = None):
    img = Image.open(record["path"]).convert("RGB").resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
    canvas = img.convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    if not alert_on:
        status_font = load_font(18)
        draw.rectangle([14, 14, 230, 46], fill=(20, 20, 20, 180))
        draw.text((22, 20), "SCANNING...", font=status_font, fill=(120, 220, 120))
        canvas.convert("RGB").save(out_path, quality=94)
        return

    grid = record["score_grid"]
    hf, wf = grid.shape
    i0, j0, i1, j1 = connected_region_bbox(grid, seed_i, seed_j)
    x0, y0, _, _ = grid_cell_bbox(i0, j0, hf, wf, CANVAS_SIZE[1], CANVAS_SIZE[0])
    _, _, x1, y1 = grid_cell_bbox(i1 - 1, j1 - 1, hf, wf, CANVAS_SIZE[1], CANVAS_SIZE[0])
    pad_x = int((x1 - x0) * 0.25) + 8
    pad_y = int((y1 - y0) * 0.25) + 8
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
    banner_x0 = min(int(x0), CANVAS_SIZE[0] - banner_w)
    banner_x0 = max(0, banner_x0)
    banner = Image.new("RGBA", (banner_w, banner_h), BANNER_COLOR)
    bdraw = ImageDraw.Draw(banner)
    bdraw.text((16, 12), ALERT_TITLE, font=title_font, fill=TEXT_COLOR)
    bdraw.text((16, 54), ALERT_SUBTITLE, font=sub_font, fill=SUBTEXT_COLOR)
    canvas.alpha_composite(banner, (banner_x0, int(banner_y0)))

    canvas.convert("RGB").save(out_path, quality=94)


def main():
    result = run_pipeline()
    records = result["records"]
    scorers_by_band = result["scorers_by_band"]
    calib_stats_by_band = result["calib_stats_by_band"]

    # Find genuine top-1 hits and their scene, same logic as build_demo_frames.py
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

    if not hit_candidates:
        print("No confirmed top-1 hits in the cached records -- run evaluate_patch_localization.py first.")
        return

    print(f"Found {len(hit_candidates)} confirmed top-1 hits (same order as inspect_hit_candidates.py):")
    for k, r in enumerate(hit_candidates):
        print(f"  [{k}] scene={r['scene']}  file={Path(r['path']).name}")
    print()

    img_root = LAF_ROOT / "leftImg8bit" / "leftImg8bit"
    if not img_root.exists():
        img_root = LAF_ROOT / "leftImg8bit"

    # Optional: `python build_demo_video.py 1` targets hit_candidates[1]
    # directly (matching the index printed above / from
    # inspect_hit_candidates.py) instead of auto-picking the longest
    # sequence -- useful once you've looked at the crops and know which
    # scene is cleanest (less clutter = less risk of the tracker catching a
    # nearby nuisance object like a manhole cover).
    forced_idx = None
    if len(sys.argv) > 1:
        try:
            forced_idx = int(sys.argv[1])
        except ValueError:
            print(f"Ignoring non-integer argument {sys.argv[1]!r}\n")

    if forced_idx is not None:
        if not (0 <= forced_idx < len(hit_candidates)):
            print(f"Index {forced_idx} out of range (0-{len(hit_candidates) - 1})")
            return
        r = hit_candidates[forced_idx]
        scene_dir = find_scene_dir(img_root, r["scene"])
        if scene_dir is None:
            print(f"Could not locate on-disk scene directory for {r['scene']}")
            return
        best = (r, scene_dir, frames_for_scene(scene_dir, r["scene"]))
        print(f"Using forced selection [{forced_idx}]: scene={r['scene']}\n")
    else:
        # Default: pick the hit whose full on-disk sequence is longest.
        best = None
        for r in hit_candidates:
            scene_dir = find_scene_dir(img_root, r["scene"])
            if scene_dir is None:
                continue
            seq_frames = frames_for_scene(scene_dir, r["scene"])
            if best is None or len(seq_frames) > len(best[2]):
                best = (r, scene_dir, seq_frames)

    if best is None:
        print("Could not locate an on-disk scene directory for any hit -- check LAF_ROOT.")
        return

    hit_record, scene_dir, frame_paths = best
    frame_paths = sorted(frame_paths, key=frame_number)
    print(f"Selected scene: {hit_record['scene']}  ({len(frame_paths)} frames on disk)")
    print(f"Known-good hit frame: {Path(hit_record['path']).name}\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = PatchResNetEmbedder().to(device)
    model.eval()

    sequence_records = [(p, hit_record["scene"], True) for p in frame_paths]
    print(f"Scoring all {len(sequence_records)} frames of the sequence...")
    scored = score_test_frames(model, scorers_by_band, calib_stats_by_band, sequence_records, device)
    print(f"Scored {len(scored)} frames\n")

    # Ground the detection threshold in the known-good hit's own z-score,
    # rather than an arbitrary constant.
    hit_grid = hit_record["score_grid"]
    hf, wf = hit_grid.shape
    hit_score = float(np.max(hit_grid))
    threshold = hit_score * DETECTION_THRESHOLD_FRAC
    print(f"Known-hit frame argmax z-score: {hit_score:.2f} -> detection threshold: {threshold:.2f}")

    # Anchor: the validated region's centroid, in grid coordinates. Every
    # frame's detection is a search WITHIN a neighborhood of this anchor
    # (see windowed_argmax docstring) -- not a fresh whole-image search each
    # frame, which is what let the box jump to an unrelated manhole cover.
    anchor_i, anchor_j = np.unravel_index(np.argmax(hit_grid), hit_grid.shape)
    radius_i = max(3, int(hf * WINDOW_RADIUS_FRAC))
    radius_j = max(3, int(wf * WINDOW_RADIUS_FRAC))
    print(f"Anchor grid position: ({anchor_i}, {anchor_j})  search radius: ({radius_i}, {radius_j}) cells\n")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    gif_frames = []
    alert_latched = False
    consecutive_above = 0
    for idx, rec in enumerate(scored):
        grid = rec["score_grid"]
        track_i, track_j = windowed_argmax(grid, anchor_i, anchor_j, radius_i, radius_j)
        tracked_score = float(grid[track_i, track_j])
        if tracked_score >= threshold:
            consecutive_above += 1
        else:
            consecutive_above = 0
        if consecutive_above >= MIN_CONSECUTIVE_FRAMES:
            alert_latched = True
        out_path = FRAMES_DIR / f"frame_{idx:04d}.jpg"
        render_frame(rec, out_path, alert_on=alert_latched, seed_i=track_i, seed_j=track_j)
        gif_frames.append(out_path)
        status = "ALERT" if alert_latched else "scanning"
        print(f"  frame {idx + 1}/{len(scored)}  tracked_z={tracked_score:.2f}  [{status}]")

    if not alert_latched:
        print("\nWarning: threshold was never crossed across the whole sequence -- "
              "the known-hit frame may not have been included, or the threshold is too high.")

    # Hold the final frame a bit longer so the alert is readable.
    hold_paths = gif_frames + [gif_frames[-1]] * HOLD_LAST_FRAMES

    print(f"\nBuilding animated GIF ({len(hold_paths)} frames @ {FRAME_MS}ms)...")
    pil_frames = [Image.open(p).convert("RGB") for p in hold_paths]
    pil_frames[0].save(
        GIF_PATH, save_all=True, append_images=pil_frames[1:],
        duration=FRAME_MS, loop=0,
    )
    print(f"Saved {GIF_PATH}")

    print("\nFor a smaller, higher-quality mp4 (if ffmpeg is installed), run:")
    print(f'  ffmpeg -y -framerate {1000 // FRAME_MS} -i "{FRAMES_DIR}/frame_%04d.jpg" '
          f'-vf "tpad=stop_mode=clone:stop_duration={HOLD_LAST_FRAMES * FRAME_MS / 1000:.2f}" '
          f'-pix_fmt yuv420p "{RESULTS_DIR}/demo_video.mp4"')


if __name__ == "__main__":
    main()
