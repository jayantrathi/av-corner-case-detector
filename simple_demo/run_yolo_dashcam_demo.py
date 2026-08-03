"""Simple, standalone demo: run YOLOv8 (COCO-pretrained, no custom logic,
no thresholds to tune) on a real dashcam video sequence and render clean
bounding boxes.

Deliberately self-contained -- does not import anything from scripts/ or
src/. No anomaly scoring, no margin masks, no percentile thresholds. Just
a mature, well-tested object detector doing what it's actually good at:
finding cars, people, cyclists, trucks, traffic lights in real footage.

Usage: python run_yolo_dashcam_demo.py
"""
from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
except ImportError:
    print("Missing dependency. Run: pip install ultralytics --break-system-packages")
    raise SystemExit(1)

LAF_ROOT = Path("/Volumes/BIggen/AV/data/lost_and_found")
YOLO_WEIGHTS = Path("/Volumes/BIggen/AV/yolov8n.pt")
OUT_DIR = Path("/Volumes/BIggen/AV/simple_demo/output")
FRAMES_DIR = OUT_DIR / "frames"
GIF_PATH = OUT_DIR / "dashcam_demo.gif"

CONF_THRESHOLD = 0.35
CANVAS_SIZE = (1200, 675)  # downscaled from native ~2048x1024 for a faster, still-clean demo
FRAME_MS = 120
MAX_FRAMES = 40  # cap so the GIF stays a reasonable size/runtime

SEQ_RE = re.compile(r"^(.*)_(\d{6})_(\d{6})_leftImg8bit\.png$")

# Distinct, readable colors per COCO class we actually expect to see on a
# dashcam -- anything else falls back to a neutral gray.
CLASS_COLORS = {
    "car": (66, 133, 244), "truck": (234, 67, 53), "bus": (251, 188, 5),
    "person": (52, 168, 83), "bicycle": (171, 71, 188), "motorcycle": (255, 112, 67),
    "traffic light": (0, 229, 255),
}
DEFAULT_COLOR = (200, 200, 200)


def find_longest_sequence(img_root: Path) -> list[Path]:
    """Group frames by (location, sequence number) and return the longest
    run of consecutive frames -- the smoothest-looking clip available."""
    all_images = sorted(img_root.rglob("*_leftImg8bit.png"))
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in all_images:
        m = SEQ_RE.match(p.name)
        if not m:
            continue
        location, seq, _frame = m.groups()
        groups[f"{location}_{seq}"].append(p)

    best_key = max(groups, key=lambda k: len(groups[k]))
    frames = sorted(groups[best_key], key=lambda p: int(SEQ_RE.match(p.name).group(3)))
    print(f"Selected sequence '{best_key}' with {len(frames)} frames")
    return frames


def load_font(size: int):
    for candidate in ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_frame(img_path: Path, result, out_path: Path, font):
    img = Image.open(img_path).convert("RGB").resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    sx, sy = CANVAS_SIZE[0] / result.orig_shape[1], CANVAS_SIZE[1] / result.orig_shape[0]

    n_detections = 0
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue
        cls_name = result.names[int(box.cls[0])]
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        x0, y0, x1, y1 = x0 * sx, y0 * sy, x1 * sx, y1 * sy
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{cls_name} {conf:.2f}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.rectangle([x0, y0 - th - 6, x0 + tw + 8, y0], fill=color)
        draw.text((x0 + 4, y0 - th - 4), label, font=font, fill=(0, 0, 0))
        n_detections += 1

    draw.rectangle([10, 10, 220, 40], fill=(20, 20, 20))
    draw.text((18, 16), f"{n_detections} objects tracked", font=font, fill=(120, 220, 120))
    img.save(out_path, quality=92)


def main():
    img_root = LAF_ROOT / "leftImg8bit" / "leftImg8bit"
    if not img_root.exists():
        img_root = LAF_ROOT / "leftImg8bit"

    frames = find_longest_sequence(img_root)
    if len(frames) > MAX_FRAMES:
        frames = frames[-MAX_FRAMES:]  # keep the tail, closest to the vehicle's approach
        print(f"Capped to last {MAX_FRAMES} frames")

    print("Loading YOLOv8n...")
    model = YOLO(str(YOLO_WEIGHTS))
    font = load_font(16)

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for i, frame_path in enumerate(frames):
        result = model.predict(str(frame_path), conf=CONF_THRESHOLD, verbose=False)[0]
        out_path = FRAMES_DIR / f"frame_{i:04d}.jpg"
        render_frame(frame_path, result, out_path, font)
        out_paths.append(out_path)
        print(f"  [{i + 1}/{len(frames)}] {frame_path.name} -> {out_path.name}")

    print("\nBuilding GIF...")
    pil_frames = [Image.open(p).convert("RGB") for p in out_paths]
    pil_frames[0].save(GIF_PATH, save_all=True, append_images=pil_frames[1:], duration=FRAME_MS, loop=0)
    print(f"Saved {GIF_PATH}")
    print(f"\nFor an mp4 (if ffmpeg is installed):")
    print(f'  ffmpeg -y -framerate {1000 // FRAME_MS} -i "{FRAMES_DIR}/frame_%04d.jpg" '
          f'-pix_fmt yuv420p "{OUT_DIR}/dashcam_demo.mp4"')


if __name__ == "__main__":
    main()
