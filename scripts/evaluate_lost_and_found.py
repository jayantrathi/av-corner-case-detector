"""Evaluate on Lost & Found: the cleanest test we have.

Real photographed road hazards (lost cargo, debris), real pixel-level
ground truth (trainId 2 = hazard object, trainId 1 = free/drivable road,
trainId 0 = background), same camera/domain for normal and hazard frames
within each sequence.

Critically: we do NOT score Lost & Found against the nuScenes-trained
reference -- that would reintroduce the exact cross-dataset confound that
made the original CODA number meaningless (different camera/city/compression
looking like "anomaly" when it's really just "different dataset"). Instead
we fit the k-NN reference on Lost & Found's OWN normal frames (scene-split,
so no near-duplicate leakage) and test on its OWN held-out normal + hazard
frames. Any separation here is a real semantic-novelty signal, not a
domain-fingerprint shortcut.

Runs both frozen ImageNet ResNet50 and the nuScenes-fine-tuned backbone, to
see whether the fine-tuning transfers to a completely different real domain
or was overfit to the nuScenes/CODA gap specifically.
"""

from __future__ import annotations
import sys
import re
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.models as models

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.scoring.embedding_scorers import kNNScorer
from src.eval.metrics import auroc, aupr

LAF_ROOT = Path("/Volumes/BIggen/AV/data/lost_and_found")
RESULTS_DIR = Path("/Volumes/BIggen/AV/results")
HAZARD_TRAIN_ID = 2
SEED = 0


class ResNetEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        feat = self.backbone(x)
        return feat.view(feat.size(0), -1)


def load_finetuned_model(device: str) -> ResNetEmbedder:
    model = ResNetEmbedder().to(device)
    state_dict = torch.load(RESULTS_DIR / "resnet50_backbone_finetuned.pt", map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        backbone_state = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
        model.backbone.load_state_dict(backbone_state)
    return model


def load_image_as_tensor(path: str, device: str) -> torch.Tensor | None:
    try:
        img = Image.open(path).convert("RGB").resize((1600, 900), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        t[0] = (t[0] - 0.485) / 0.229
        t[1] = (t[1] - 0.456) / 0.224
        t[2] = (t[2] - 0.406) / 0.225
        return t.to(device)
    except Exception:
        return None


@torch.no_grad()
def embed_batch(model, tensors: list[torch.Tensor], device: str, batch_size: int = 16) -> np.ndarray:
    import time
    model.eval()
    out = []
    n_batches = (len(tensors) + batch_size - 1) // batch_size
    t0 = time.time()
    for bi, i in enumerate(range(0, len(tensors), batch_size)):
        batch = torch.stack(tensors[i:i + batch_size]).to(device)
        out.append(model(batch).cpu().numpy())
        elapsed = time.time() - t0
        print(f"    batch {bi + 1}/{n_batches}  ({i + len(batch)}/{len(tensors)} images, {elapsed:.1f}s elapsed)", flush=True)
    return np.concatenate(out, axis=0)


def img_path_to_label_path(img_path: Path) -> Path:
    """Map a leftImg8bit path to its gtCoarse labelTrainIds path."""
    parts = list(img_path.parts)
    parts = ["gtCoarse" if p == "leftImg8bit" else p for p in parts]
    gt_path = Path(*parts)
    gt_path = gt_path.with_name(gt_path.name.replace("_leftImg8bit.png", "_gtCoarse_labelTrainIds.png"))
    return gt_path


MIN_HAZARD_AREA_FRAC = 0.0006  # ~1250px on a 2048x1024 frame -- a genuinely
# visible object, not a speck. Lost & Found sequences track the SAME object
# continuously as the car approaches it over 20-40 frames, so "any pixel at
# all" would label nearly every frame in a sequence "hazard" even when the
# object is a distant dot -- that's not the distinction we want to test.


def frame_hazard_status(label_path: Path) -> str | None:
    """'normal' (zero hazard pixels), 'hazard' (hazard area above threshold),
    or 'ambiguous' (present but too small to count either way -- excluded).
    None if unreadable."""
    try:
        arr = np.array(Image.open(label_path))
        hazard_px = int(np.sum(arr == HAZARD_TRAIN_ID))
        if hazard_px == 0:
            return "normal"
        frac = hazard_px / arr.size
        if frac >= MIN_HAZARD_AREA_FRAC:
            return "hazard"
        return "ambiguous"
    except Exception:
        return None


SEQ_RE = re.compile(r"^(.*)_(\d{6})_(\d{6})_leftImg8bit\.png$")


def scene_id_for(img_path: Path) -> str:
    """{location}_{sequence_number} -- the atomic unit; individual frames
    within it are near-duplicates of the same drive-through."""
    m = SEQ_RE.match(img_path.name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return img_path.parent.name


def main():
    random.seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")

    img_root = LAF_ROOT / "leftImg8bit" / "leftImg8bit"
    if not img_root.exists():
        img_root = LAF_ROOT / "leftImg8bit"
    all_images = sorted(img_root.rglob("*_leftImg8bit.png"))
    print(f"Found {len(all_images)} Lost & Found frames\n")

    print("Reading ground-truth masks to label each frame (hazard / normal / ambiguous)...")
    frame_records = []  # (img_path, scene_id, status)
    n_ambiguous = 0
    for img_path in all_images:
        label_path = img_path_to_label_path(img_path)
        status = frame_hazard_status(label_path)
        if status is None:
            continue
        if status == "ambiguous":
            n_ambiguous += 1
            continue
        frame_records.append((img_path, scene_id_for(img_path), status == "hazard"))

    n_hazard = sum(1 for _, _, h in frame_records if h)
    n_normal = len(frame_records) - n_hazard
    print(f"Labeled {len(frame_records)} frames: {n_normal} normal, {n_hazard} hazard "
          f"(dropped {n_ambiguous} ambiguous -- hazard present but too small/distant to count)\n")

    # Scene-level split: entire physical sequences go to train or test, never
    # both -- same discipline as nuScenes, for the same reason (adjacent
    # frames in a sequence are near-duplicates of each other). Skewed toward
    # train since we need a robust normal reference and normal frames are
    # the scarcer class here.
    scenes = sorted(set(r[1] for r in frame_records))
    random.shuffle(scenes)
    n_train_scenes = int(len(scenes) * 0.7)
    train_scenes = set(scenes[:n_train_scenes])
    test_scenes = set(scenes[n_train_scenes:])
    print(f"Scenes: {len(scenes)} total -> {len(train_scenes)} train, {len(test_scenes)} test\n")

    # Train reference = ONLY normal frames from train scenes (one-class fit,
    # never contaminate the "normal" reference with a hazard frame).
    train_normal = [r for r in frame_records if r[1] in train_scenes and not r[2]]
    # Test set = all frames (normal + hazard) from held-out scenes.
    test_normal = [r for r in frame_records if r[1] in test_scenes and not r[2]]
    test_hazard = [r for r in frame_records if r[1] in test_scenes and r[2]]

    print(f"Train reference (normal only): {len(train_normal)} frames")
    print(f"Test normal: {len(test_normal)} frames")
    print(f"Test hazard: {len(test_hazard)} frames\n")

    # Cap for tractability -- this is scoring on CPU-adjacent MPS, keep it
    # reasonable while still being a real sample.
    MAX_PER_SPLIT = 400
    if len(train_normal) > MAX_PER_SPLIT:
        train_normal = random.sample(train_normal, MAX_PER_SPLIT)
    if len(test_normal) > MAX_PER_SPLIT:
        test_normal = random.sample(test_normal, MAX_PER_SPLIT)
    if len(test_hazard) > MAX_PER_SPLIT:
        test_hazard = random.sample(test_hazard, MAX_PER_SPLIT)

    print(f"After capping: train_normal={len(train_normal)} test_normal={len(test_normal)} test_hazard={len(test_hazard)}\n")

    print("Loading images...")
    def load_all(records):
        tensors = []
        for path, _, _ in records:
            t = load_image_as_tensor(str(path), device)
            if t is not None:
                tensors.append(t)
        return tensors

    train_normal_tensors = load_all(train_normal)
    test_normal_tensors = load_all(test_normal)
    test_hazard_tensors = load_all(test_hazard)
    print(f"Loaded: train_normal={len(train_normal_tensors)} test_normal={len(test_normal_tensors)} test_hazard={len(test_hazard_tensors)}\n")

    labels = np.concatenate([
        np.zeros(len(test_normal_tensors)),
        np.ones(len(test_hazard_tensors)),
    ])
    test_tensors = test_normal_tensors + test_hazard_tensors

    results = {}

    for model_name, load_fn in [
        ("frozen_imagenet", lambda: ResNetEmbedder().to(device)),
        ("nuscenes_finetuned", lambda: load_finetuned_model(device)),
    ]:
        print("=" * 70)
        print(f"MODEL: {model_name}  (reference fit ONLY on Lost & Found's own normal frames)")
        print("=" * 70)

        model = load_fn()
        train_emb = embed_batch(model, train_normal_tensors, device)
        test_emb = embed_batch(model, test_tensors, device)

        scorer = kNNScorer(k=5)
        scorer.fit(train_emb)
        scores = scorer.score(test_emb)

        model_auroc = auroc(scores, labels)
        model_aupr = aupr(scores, labels)
        print(f"AUROC: {model_auroc:.4f}")
        print(f"AUPR:  {model_aupr:.4f}")

        normal_scores = scores[:len(test_normal_tensors)]
        hazard_scores = scores[len(test_normal_tensors):]
        print(f"Normal scores: mean={normal_scores.mean():.4f} std={normal_scores.std():.4f}")
        print(f"Hazard scores: mean={hazard_scores.mean():.4f} std={hazard_scores.std():.4f}\n")

        results[model_name] = {
            "auroc": float(model_auroc),
            "aupr": float(model_aupr),
            "normal_mean": float(normal_scores.mean()),
            "hazard_mean": float(hazard_scores.mean()),
            "n_train_normal": len(train_normal_tensors),
            "n_test_normal": len(test_normal_tensors),
            "n_test_hazard": len(test_hazard_tensors),
        }

    print("=" * 70)
    print("LOST & FOUND -- WITHIN-DOMAIN RESULT (the honest number)")
    print("=" * 70)
    print(f"Frozen ImageNet:     AUROC={results['frozen_imagenet']['auroc']:.4f}")
    print(f"nuScenes fine-tuned: AUROC={results['nuscenes_finetuned']['auroc']:.4f}")
    print("This is real photographed hazards vs. real normal road, same camera,")
    print("same sequences family, zero compositing, zero cross-dataset confound.")
    print("Whatever this number is, it's the truth.")
    print("=" * 70 + "\n")

    out_path = RESULTS_DIR / "lost_and_found_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
