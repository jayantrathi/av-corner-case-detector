"""RbA version of build_demo_video.py -- a real Lost & Found (German dashcam)
approach sequence, scored frame by frame, with the alert locking on once a
real object is close enough to actually become the most anomalous region in
the WHOLE frame.

Design choice worth being explicit about: this does NOT restrict the search
to a small window around a fixed anchor point the way the original k-NN
build_demo_video.py's windowed_argmax did. RbA's percentile threshold
(largest_region_from_peak, top REGION_TOP_PERCENTILE% of the frame) already
runs globally every frame, same as run_demo_coda_rba.py -- consistent with
how the CODA demo was scored, not a special-cased version for this script.

Instead, the "is this a real detection or just this frame's top-1% noise"
question is answered by PROXIMITY: does the globally-detected region's
centroid land near where the real hazard actually was confirmed (from the
held-out evaluation)? Early frames, where the object is small and distant,
will have their global top-1% wander elsewhere in the frame -- that's
correctly reported as "not detected yet," not forced into a detection. Once
the object is close enough to actually be the most anomalous thing in the
frame, the global top-1% naturally lands on it and stays there. Same
honesty principle documented in the original build_demo_video.py: the box
switching on partway through the approach is the real story, not a flaw to
hide.

Requires evaluate_rba_lost_and_found.py's fixed percentile threshold (this
script imports it directly, so the fix is picked up automatically).
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
    LAF_ROOT, CANVAS_SIZE, SEQ_RE, scene_id_for,
    img_path_to_label_path, load_mask_resized, HAZARD_TRAIN_ID,
)
from evaluate_rba_lost_and_found import load_test_split, largest_region_from_peak
from build_demo_video import find_scene_dir, frames_for_scene, frame_number
from build_demo_frames import (
    load_font, measure_banner_size, ALERT_TITLE, ALERT_SUBTITLE,
    BOX_COLOR, BOX_WIDTH, BANNER_COLOR, TEXT_COLOR, SUBTEXT_COLOR,
)
from src.scoring.mask2former_rba import RbAScorer

RESULTS_DIR = Path("/Volumes/BIggen/AV/results")
FRAMES_DIR = RESULTS_DIR / "demo_video_rba_frames"
GIF_PATH = RESULTS_DIR / "demo_video_rba.gif"

FRAME_MS = 140
HOLD_LAST_FRAMES = 6
PROXIMITY_FRAC = 0.15  # detected region centroid must land within this
# fraction of the frame diagonal from the confirmed-hit centroid to count
# as "tracking the same object" rather than an unrelated top-1% region
# elsewhere in the frame
MIN_CONSECUTIVE_FRAMES = 2  # same reasoning as the k-NN version -- one
# frame crossing isn't enough, a real approach stays close consistently


def region_centroid(region_mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(region_mask)
    return float(ys.mean()), float(xs.mean())


def render_frame(img_path, region_mask, out_path, alert_on: bool):
    img = Image.open(img_path).convert("RGB").resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
    canvas = img.convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    if not alert_on:
        status_font = load_font(18)
        draw.rectangle([14, 14, 230, 46], fill=(20, 20, 20, 180))
        draw.text((22, 20), "SCANNING...", font=status_font, fill=(120, 220, 120))
        canvas.convert("RGB").save(out_path, quality=94)
        return

    ys, xs = np.where(region_mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad_x, pad_y = int((x1 - x0) * 0.25) + 8, int((y1 - y0) * 0.25) + 8
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

    _, test_hazard = load_test_split()
    print(f"Scanning {len(test_hazard)} held-out hazard frames for a confirmed region hit...")

    img_root = LAF_ROOT / "leftImg8bit" / "leftImg8bit"
    if not img_root.exists():
        img_root = LAF_ROOT / "leftImg8bit"

    hit_candidates = []
    for path, scene, _ in test_hazard:
        img = Image.open(path).convert("RGB")
        rba_map, _ = scorer.score(img, out_size=CANVAS_SIZE)
        label_path = img_path_to_label_path(path)
        mask = load_mask_resized(label_path)
        if mask is None:
            continue
        hazard_mask = (mask == HAZARD_TRAIN_ID)
        if not hazard_mask.any():
            continue
        region_mask, _ = largest_region_from_peak(rba_map)
        if (region_mask & hazard_mask).any():
            hit_candidates.append((path, scene, region_mask))
            print(f"  confirmed hit: {Path(path).name}")

    if not hit_candidates:
        print("\nNo confirmed region hits in the held-out set -- nothing to build a sequence from.")
        return

    print(f"\nFound {len(hit_candidates)} confirmed hits:")
    for k, (path, scene, _) in enumerate(hit_candidates):
        print(f"  [{k}] scene={scene}  file={Path(path).name}")
    print()

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
        path, scene, region_mask = hit_candidates[forced_idx]
        scene_dir = find_scene_dir(img_root, scene)
        if scene_dir is None:
            print(f"Could not locate on-disk scene directory for {scene}")
            return
        best = (path, scene, region_mask, scene_dir, frames_for_scene(scene_dir, scene))
        print(f"Using forced selection [{forced_idx}]: scene={scene}\n")
    else:
        best = None
        for path, scene, region_mask in hit_candidates:
            scene_dir = find_scene_dir(img_root, scene)
            if scene_dir is None:
                continue
            seq_frames = frames_for_scene(scene_dir, scene)
            if best is None or len(seq_frames) > len(best[4]):
                best = (path, scene, region_mask, scene_dir, seq_frames)

    if best is None:
        print("Could not locate an on-disk scene directory for any hit -- check LAF_ROOT.")
        return

    hit_path, hit_scene, hit_region_mask, scene_dir, frame_paths = best
    frame_paths = sorted(frame_paths, key=frame_number)
    anchor_y, anchor_x = region_centroid(hit_region_mask)
    diag = float(np.hypot(*CANVAS_SIZE))
    proximity_px = diag * PROXIMITY_FRAC
    print(f"Selected scene: {hit_scene}  ({len(frame_paths)} frames on disk)")
    print(f"Known-good hit frame: {Path(hit_path).name}")
    print(f"Anchor centroid: ({anchor_y:.0f}, {anchor_x:.0f})  proximity radius: {proximity_px:.0f}px\n")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    gif_frames = []
    alert_latched = False
    consecutive_near = 0
    for idx, fp in enumerate(frame_paths):
        img = Image.open(fp).convert("RGB")
        rba_map, _ = scorer.score(img, out_size=CANVAS_SIZE)
        region_mask, _ = largest_region_from_peak(rba_map)
        cy, cx = region_centroid(region_mask)
        dist = float(np.hypot(cy - anchor_y, cx - anchor_x))
        near = dist <= proximity_px

        consecutive_near = consecutive_near + 1 if near else 0
        if consecutive_near >= MIN_CONSECUTIVE_FRAMES:
            alert_latched = True

        out_path = FRAMES_DIR / f"frame_{idx:04d}.jpg"
        render_frame(fp, region_mask, out_path, alert_on=alert_latched)
        gif_frames.append(out_path)
        status = "ALERT" if alert_latched else ("near" if near else "scanning")
        print(f"  frame {idx + 1}/{len(frame_paths)}  centroid_dist={dist:.0f}px  [{status}]")

    if not alert_latched:
        print("\nWarning: alert never latched across the whole sequence -- the proximity radius "
              "may be too tight, or this sequence's approach doesn't bring the object close enough.")

    hold_paths = gif_frames + [gif_frames[-1]] * HOLD_LAST_FRAMES
    print(f"\nBuilding animated GIF ({len(hold_paths)} frames @ {FRAME_MS}ms)...")
    pil_frames = [Image.open(p).convert("RGB") for p in hold_paths]
    pil_frames[0].save(GIF_PATH, save_all=True, append_images=pil_frames[1:], duration=FRAME_MS, loop=0)
    print(f"Saved {GIF_PATH}")

    print("\nFor a smaller, higher-quality mp4 (if ffmpeg is installed), run:")
    print(f'  ffmpeg -y -framerate {1000 // FRAME_MS} -i "{FRAMES_DIR}/frame_%04d.jpg" '
          f'-vf "tpad=stop_mode=clone:stop_duration={HOLD_LAST_FRAMES * FRAME_MS / 1000:.2f}" '
          f'-pix_fmt yuv420p "{RESULTS_DIR}/demo_video_rba.mp4"')


if __name__ == "__main__":
    main()
