"""Verify the YOLO object-detection layer before trusting it as a signal.

The proposed hybrid design: run YOLOv8n (COCO-pretrained, no training needed)
alongside the patch-level embedding scorer. A patch gets its anomaly flag
SUPPRESSED if it overlaps a confident known-class detection (car/person/
bike/bus/etc -- "the model has a policy for this, it's not a corner case
just because it's rare"). A patch gets flagged "unknown" if it has high
embedding distance AND no confident detection at all -- a second, independent
vote that this is something the perception stack has no class for.

That design only works if YOLO's behavior on our actual hazard objects is
what we assume it is. This script checks that assumption in three parts
instead of taking it on faith:

  A. The 22 real hazard crops (debris/machinery/dustbin/sentry_box/cart/
     suitcase/construction_vehicle, extracted from CODA). Ideally YOLO finds
     nothing confident here -- if it does, e.g. mislabeling a "cart" as a
     COCO "handbag"-adjacent class, that's a real failure mode to know about
     before relying on "no detection = unknown" as a signal.
  B. A sanity check on ordinary nuScenes frames -- confirms the detector
     pipeline itself works (finds cars/people at reasonable confidence)
     rather than silently being broken and always returning nothing.
  C. Real Lost & Found hazard frames -- crop the true hazard region from the
     real ground-truth mask, check whether any YOLO detection overlaps it
     and with what class/confidence. This is the case that matters most
     since it's real photographed data, not a synthetic composite.

Requires: pip install ultralytics --break-system-packages (if in a sandbox)
or just pip install ultralytics locally. Auto-downloads yolov8n.pt (~6MB,
COCO-pretrained) on first run.
"""
from __future__ import annotations
import sys
import re
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Run: pip install ultralytics --break-system-packages")
    sys.exit(1)

HAZARD_CROPS_DIR = Path("/Volumes/BIggen/AV/data/hazard_crops")
NUSCENES_ROOT = "/Volumes/BIggen/AV/data/nuscenes/v1.0-mini"
LAF_ROOT = Path("/Volumes/BIggen/AV/data/lost_and_found")
RESULTS_DIR = Path("/Volumes/BIggen/AV/results")
HAZARD_TRAIN_ID = 2
CONF_THRESHOLD = 0.25
SEED = 0
N_LAF_SAMPLE = 15

SEQ_RE = re.compile(r"^(.*)_(\d{6})_(\d{6})_leftImg8bit\.png$")


def img_path_to_label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    parts = ["gtCoarse" if p == "leftImg8bit" else p for p in parts]
    gt_path = Path(*parts)
    gt_path = gt_path.with_name(gt_path.name.replace("_leftImg8bit.png", "_gtCoarse_labelTrainIds.png"))
    return gt_path


def hazard_bbox_from_mask(label_path: Path) -> tuple[int, int, int, int] | None:
    """Pixel bbox (x0,y0,x1,y1) of the hazard region in a Lost & Found mask,
    or None if there's no meaningful hazard area."""
    try:
        arr = np.array(Image.open(label_path))
    except Exception:
        return None
    ys, xs = np.where(arr == HAZARD_TRAIN_ID)
    if len(xs) < 30:  # too small to be a meaningful region
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def iou(box_a, box_b) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / (area_a + area_b - inter)


def overlap_frac_of_gt(box_a, gt_box) -> float:
    """What fraction of the ground-truth box does this detection cover --
    more informative than IoU when the detection box and hazard box are
    very different sizes (e.g. YOLO draws a big box loosely around a small
    real object)."""
    ax0, ay0, ax1, ay1 = box_a
    gx0, gy0, gx1, gy1 = gt_box
    ix0, iy0 = max(ax0, gx0), max(ay0, gy0)
    ix1, iy1 = min(ax1, gx1), min(ay1, gy1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    gt_area = (gx1 - gx0) * (gy1 - gy0)
    return inter / gt_area if gt_area > 0 else 0.0


def part_a_hazard_crops(model):
    print("=" * 70)
    print("PART A: real hazard crops (CODA) -- what does YOLO see?")
    print("=" * 70)
    manifest = json.load(open(HAZARD_CROPS_DIR / "manifest.json"))
    results = []
    for m in manifest:
        crop_path = HAZARD_CROPS_DIR / m["crop_file"]
        img = Image.open(crop_path).convert("RGB")
        pred = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
        dets = []
        for box in pred.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            dets.append({"class": cls_name, "conf": conf})
        dets.sort(key=lambda d: -d["conf"])
        results.append({"category": m["category"], "file": m["crop_file"], "detections": dets})
        top = dets[0] if dets else None
        flag = f"  <-- FALSE POSITIVE: {top['class']} @ {top['conf']:.2f}" if top else "  (nothing detected, as hoped)"
        print(f"  {m['category']:22s} {m['crop_file'][:40]:40s}{flag}")

    n_clean = sum(1 for r in results if not r["detections"])
    print(f"\n{n_clean}/{len(results)} hazard crops -> no confident detection")
    print(f"{len(results) - n_clean}/{len(results)} hazard crops -> YOLO found SOMETHING (see flags above)\n")
    return results


def part_b_sanity_check(model):
    print("=" * 70)
    print("PART B: sanity check on ordinary nuScenes frames")
    print("=" * 70)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.nuscenes_loader import load_nuscenes_frames
    from src.data.splits import split_by_scene

    frames = load_nuscenes_frames(NUSCENES_ROOT)
    splits = split_by_scene(frames, val_frac=0.15, test_frac=0.15, seed=0)
    sample = random.sample(list(splits["val"]), min(10, len(splits["val"])))

    n_with_detection = 0
    class_counts = {}
    for f in sample:
        img = Image.open(f.path).convert("RGB")
        pred = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
        if len(pred.boxes) > 0:
            n_with_detection += 1
        for box in pred.boxes:
            cls_name = model.names[int(box.cls[0])]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    print(f"{n_with_detection}/{len(sample)} ordinary frames had at least one confident detection")
    print(f"Classes seen: {class_counts}")
    if n_with_detection < len(sample) * 0.5:
        print("WARNING: fewer than half of ordinary driving frames got any detection --")
        print("this suggests the pipeline itself may be misconfigured, not that the")
        print("scenes are genuinely empty of cars/people.")
    print()
    return {"n_with_detection": n_with_detection, "n_sample": len(sample), "class_counts": class_counts}


def part_c_lost_and_found(model):
    print("=" * 70)
    print("PART C: real Lost & Found hazard regions -- the case that matters most")
    print("=" * 70)
    img_root = LAF_ROOT / "leftImg8bit" / "leftImg8bit"
    if not img_root.exists():
        img_root = LAF_ROOT / "leftImg8bit"
    all_images = sorted(img_root.rglob("*_leftImg8bit.png"))

    candidates = []
    for img_path in all_images:
        label_path = img_path_to_label_path(img_path)
        bbox = hazard_bbox_from_mask(label_path)
        if bbox is not None:
            candidates.append((img_path, bbox))

    sample = random.sample(candidates, min(N_LAF_SAMPLE, len(candidates)))
    print(f"Sampled {len(sample)} real hazard frames (of {len(candidates)} candidates)\n")

    results = []
    for img_path, gt_box in sample:
        img = Image.open(img_path).convert("RGB")
        pred = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
        overlaps = []
        for box in pred.boxes:
            xyxy = tuple(box.xyxy[0].tolist())
            frac = overlap_frac_of_gt(xyxy, gt_box)
            if frac > 0.1:
                overlaps.append({
                    "class": model.names[int(box.cls[0])],
                    "conf": float(box.conf[0]),
                    "overlap_frac_of_hazard": frac,
                })
        overlaps.sort(key=lambda d: -d["overlap_frac_of_hazard"])
        results.append({"file": img_path.name, "gt_box": gt_box, "overlaps": overlaps})
        if overlaps:
            top = overlaps[0]
            print(f"  {img_path.name[:50]:50s} <-- {top['class']} @ {top['conf']:.2f} "
                  f"(covers {top['overlap_frac_of_hazard']*100:.0f}% of real hazard region)")
        else:
            print(f"  {img_path.name[:50]:50s}    no detection over the hazard region (as hoped)")

    n_clean = sum(1 for r in results if not r["overlaps"])
    print(f"\n{n_clean}/{len(results)} real hazard regions -> no YOLO detection covering them")
    print(f"{len(results) - n_clean}/{len(results)} real hazard regions -> YOLO detected something there\n")
    return results


def main():
    random.seed(SEED)
    print("Loading YOLOv8n (COCO-pretrained)...\n")
    model = YOLO("yolov8n.pt")

    a_results = part_a_hazard_crops(model)
    b_results = part_b_sanity_check(model)
    c_results = part_c_lost_and_found(model)

    out = {"part_a_hazard_crops": a_results, "part_b_sanity_check": b_results, "part_c_lost_and_found": c_results}
    out_path = RESULTS_DIR / "yolo_verification_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_a_clean = sum(1 for r in a_results if not r["detections"])
    n_c_clean = sum(1 for r in c_results if not r["overlaps"])
    print(f"CODA hazard crops with no false-positive detection: {n_a_clean}/{len(a_results)}")
    print(f"Sanity check (ordinary frames get detections):      {b_results['n_with_detection']}/{b_results['n_sample']}")
    print(f"Real Lost & Found hazards with no covering detection: {n_c_clean}/{len(c_results)}")
    print()
    print("If A and C are mostly clean (no detection) and B mostly fires (detections")
    print("on ordinary cars/people), the 'no known-class detection' signal is trustworthy")
    print("and the suppression design in the plan holds. Any misses above are specific,")
    print("nameable failure modes -- worth listing before wiring this into the demo,")
    print("not something to wave away.")
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
