"""Camera + LiDAR side by side, synced to the same real-world moments, from
nuScenes v1.0-mini (already downloaded, no new data needed).

Left: real camera frame with YOLOv8 detections (same simple, no-custom-
logic approach as run_yolo_dashcam_demo.py, just a different dataset --
Boston/Singapore streets instead of Lost & Found's German suburbs).
Right: the SAME moment's raw LiDAR point cloud, rendered as a bird's-eye
view colored by height. No model on the LiDAR side yet -- pure
visualization, geometry only, zero fragile dependencies.

Camera and LiDAR frames are paired correctly using nuScenes' own metadata
(sample.json/sample_data.json), not guessed by filename/timestamp
proximity -- each nuScenes "sample" is a keyframe where every sensor was
captured at (as close to) the same instant.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
except ImportError:
    print("Missing dependency. Run: pip install ultralytics --break-system-packages")
    raise SystemExit(1)

NUSC_ROOT = Path("/Volumes/BIggen/AV/data/nuscenes/v1.0-mini")
META_DIR = NUSC_ROOT  # metadata JSONs live directly here, not in a nested v1.0-mini/v1.0-mini/
YOLO_WEIGHTS = Path("/Volumes/BIggen/AV/yolov8n.pt")
OUT_DIR = Path("/Volumes/BIggen/AV/simple_demo/output")
FRAMES_DIR = OUT_DIR / "camera_lidar_frames"
GIF_PATH = OUT_DIR / "camera_lidar_demo.gif"

CONF_THRESHOLD = 0.35
CAM_SIZE = (700, 394)     # downscaled camera panel
BEV_SIZE = (500, 500)     # square bird's-eye-view panel
BEV_RANGE_M = 40.0        # +/- meters shown in the BEV, forward and lateral
FRAME_MS = 200

CLASS_COLORS = {
    "car": (66, 133, 244), "truck": (234, 67, 53), "bus": (251, 188, 5),
    "person": (52, 168, 83), "bicycle": (171, 71, 188), "motorcycle": (255, 112, 67),
    "traffic light": (0, 229, 255),
}
DEFAULT_COLOR = (200, 200, 200)


def load_meta():
    scenes = json.load(open(META_DIR / "scene.json"))
    samples = json.load(open(META_DIR / "sample.json"))
    sample_data = json.load(open(META_DIR / "sample_data.json"))
    return scenes, samples, sample_data


def ordered_sample_tokens(scenes: list[dict], samples: list[dict]) -> list[str]:
    """Pick the scene with the most keyframes, walk its sample chain via
    the official prev/next links (the canonical temporal order), not a
    timestamp sort across the whole dataset -- the mini set has 10
    DIFFERENT scenes/cities, and sorting everything by timestamp together
    would cut between unrelated drives every frame."""
    best_scene = max(scenes, key=lambda s: s["nbr_samples"])
    print(f"Selected scene '{best_scene['name']}': {best_scene['description']} "
          f"({best_scene['nbr_samples']} keyframes)")

    samples_by_token = {s["token"]: s for s in samples}
    tokens = []
    tok = best_scene["first_sample_token"]
    while tok:
        tokens.append(tok)
        tok = samples_by_token[tok]["next"]
    return tokens


def build_sensor_lookup(sample_data: list[dict], channel_path_fragment: str) -> dict[str, str]:
    """sample_token -> filename, for the given sensor channel's KEYFRAME
    records only (is_key_frame=True). Matches on the path fragment with a
    trailing slash (e.g. 'samples/CAM_FRONT/') to avoid the exact substring
    bug that bit nuscenes_loader.py earlier this project -- 'CAM_FRONT' as
    a bare substring also matches 'CAM_FRONT_LEFT' and 'CAM_FRONT_RIGHT'."""
    lookup = {}
    for sd in sample_data:
        if sd["is_key_frame"] and channel_path_fragment in sd["filename"]:
            lookup[sd["sample_token"]] = sd["filename"]
    return lookup


def load_point_cloud(path: Path) -> np.ndarray:
    """nuScenes LIDAR_TOP .pcd.bin: raw float32, 5 columns per point
    (x, y, z, intensity, ring_index), in the LiDAR's own coordinate frame
    (x=forward, y=left, z=up, meters)."""
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)


def render_bev(points: np.ndarray) -> Image.Image:
    """Bird's-eye view: x (forward) mapped to image rows (top=far ahead),
    y (left) mapped to image columns, color = height (z), a simple
    blue-low to red-high gradient. No model, just geometry."""
    img = np.zeros((*BEV_SIZE, 3), dtype=np.uint8)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    in_range = (np.abs(x) < BEV_RANGE_M) & (np.abs(y) < BEV_RANGE_M)
    x, y, z = x[in_range], y[in_range], z[in_range]

    px = ((BEV_RANGE_M - x) / (2 * BEV_RANGE_M) * BEV_SIZE[0]).astype(int)
    py = ((BEV_RANGE_M - y) / (2 * BEV_RANGE_M) * BEV_SIZE[1]).astype(int)
    valid = (px >= 0) & (px < BEV_SIZE[0]) & (py >= 0) & (py < BEV_SIZE[1])
    px, py, z = px[valid], py[valid], z[valid]

    z_clip = np.clip(z, -2, 4)  # most road-scene points fall in this band
    z_norm = (z_clip + 2) / 6.0  # 0..1, low=ground, high=tall objects
    r = (z_norm * 255).astype(np.uint8)
    b = ((1 - z_norm) * 255).astype(np.uint8)
    g = (np.minimum(z_norm, 1 - z_norm) * 180).astype(np.uint8)
    img[px, py, 0] = r
    img[px, py, 1] = g
    img[px, py, 2] = b

    bev = Image.fromarray(img)
    draw = ImageDraw.Draw(bev)
    cx, cy = BEV_SIZE[0] // 2, BEV_SIZE[1] // 2
    draw.polygon([(cx - 6, cy + 8), (cx + 6, cy + 8), (cx, cy - 8)], fill=(255, 255, 0))
    for r_m in [10, 20, 30]:
        r_px = int(r_m / BEV_RANGE_M * BEV_SIZE[0] / 2)
        draw.ellipse([cx - r_px, cy - r_px, cx + r_px, cy + r_px], outline=(60, 60, 60), width=1)
    return bev


def load_font(size: int):
    for candidate in ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_camera_panel(img_path: Path, result, font) -> Image.Image:
    img = Image.open(img_path).convert("RGB").resize(CAM_SIZE, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    sx, sy = CAM_SIZE[0] / result.orig_shape[1], CAM_SIZE[1] / result.orig_shape[0]

    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue
        cls_name = result.names[int(box.cls[0])]
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        x0, y0, x1, y1 = x0 * sx, y0 * sy, x1 * sx, y1 * sy
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        label = f"{cls_name} {conf:.2f}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x0, y0 - th - 5, x0 + tw + 6, y0], fill=color)
        draw.text((x0 + 3, y0 - th - 3), label, font=font, fill=(0, 0, 0))
    return img


def main():
    scenes, samples, sample_data = load_meta()
    tokens = ordered_sample_tokens(scenes, samples)

    cam_lookup = build_sensor_lookup(sample_data, "samples/CAM_FRONT/")
    lidar_lookup = build_sensor_lookup(sample_data, "samples/LIDAR_TOP/")

    paired = [(t, cam_lookup[t], lidar_lookup[t]) for t in tokens if t in cam_lookup and t in lidar_lookup]
    print(f"{len(paired)}/{len(tokens)} keyframes have both CAM_FRONT and LIDAR_TOP")

    print("Loading YOLOv8n...")
    model = YOLO(str(YOLO_WEIGHTS))
    font = load_font(15)

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for i, (tok, cam_file, lidar_file) in enumerate(paired):
        cam_path = NUSC_ROOT / cam_file
        lidar_path = NUSC_ROOT / lidar_file

        result = model.predict(str(cam_path), conf=CONF_THRESHOLD, verbose=False)[0]
        cam_panel = render_camera_panel(cam_path, result, font)

        points = load_point_cloud(lidar_path)
        bev_panel = render_bev(points)

        combined = Image.new("RGB", (CAM_SIZE[0] + BEV_SIZE[0], max(CAM_SIZE[1], BEV_SIZE[1])), (10, 10, 10))
        combined.paste(cam_panel, (0, 0))
        combined.paste(bev_panel, (CAM_SIZE[0], 0))
        draw = ImageDraw.Draw(combined)
        draw.text((8, CAM_SIZE[1] - 20), "camera + YOLO", font=font, fill=(200, 200, 200))
        draw.text((CAM_SIZE[0] + 8, BEV_SIZE[1] - 20), f"LiDAR BEV -- {len(points)} points", font=font, fill=(200, 200, 200))

        out_path = FRAMES_DIR / f"frame_{i:04d}.jpg"
        combined.save(out_path, quality=92)
        out_paths.append(out_path)
        print(f"  [{i + 1}/{len(paired)}]  {cam_path.name}")

    print("\nBuilding GIF...")
    pil_frames = [Image.open(p).convert("RGB") for p in out_paths]
    pil_frames[0].save(GIF_PATH, save_all=True, append_images=pil_frames[1:], duration=FRAME_MS, loop=0)
    print(f"Saved {GIF_PATH}")


if __name__ == "__main__":
    main()
