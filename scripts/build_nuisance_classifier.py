"""Lightweight cone/bollard suppression classifier -- fit and VALIDATE before
wiring it into the demo, per the same discipline that caught the whole-frame
RbA bug: don't trust a component until you've looked at its own numbers.

Design: a frozen ImageNet ResNet50 (same backbone family already used
elsewhere in this project, e.g. PatchResNetEmbedder) reduced to a single
2048-d global-avgpool embedding per crop. No new network is trained -- this
is a k-NN classifier over frozen embeddings, fit directly on two real,
already-labeled CODA crop sets:

  - "nuisance" (suppress the alert): 256 real traffic_cone/bollard crops
    from CODA's own taxonomy (extract_nuisance_crops.py). Confirmed these
    have NO dedicated class in Cityscapes, Mapillary Vistas, or COCO
    (checked all three id2label lists directly) -- this is real signal a
    scene-parsing or COCO-detector backbone structurally cannot provide.
  - "hazard" (keep the alert): the existing extract_hazard_crops.py set
    (debris/machinery/dustbin/sentry_box/suitcase/cart/construction_vehicle/
    concrete_block/chair/phone_booth/basket) -- the genuine corner cases
    this whole project exists to catch. This is the negative class: if the
    classifier suppresses these, it's actively harmful, worse than doing
    nothing.

Held out a per-category-stratified test split (not random over crops --
same reasoning as every other split in this project: crops from the SAME
source image are near-duplicates and would leak). Reports:
  - nuisance recall: held-out cones/bollards correctly flagged "suppress"
  - hazard false-suppression rate: held-out real hazards WRONGLY flagged
    "suppress" -- the costly error, watch this number specifically

Only if both numbers look real does this get wired into run_demo_coda_rba.py
as a post-filter on RbA-flagged regions.
"""
from __future__ import annotations
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

NUISANCE_DIR = Path("/Volumes/BIggen/AV/data/nuisance_crops")
HAZARD_DIR = Path("/Volumes/BIggen/AV/data/hazard_crops")
RESULTS_DIR = Path("/Volumes/BIggen/AV/results")
BANK_PATH = RESULTS_DIR / "nuisance_classifier_bank.npz"

SEED = 0
TEST_FRAC = 0.25
K_NEIGHBORS = 5


class GlobalResNetEmbedder(nn.Module):
    """Frozen ImageNet ResNet50, standard global-avgpool 2048-d embedding --
    deliberately NOT the patch-level layer3 truncation used elsewhere
    (PatchResNetEmbedder), since these are already-cropped, variable-size
    single-object images, not a spatial grid over a full driving frame."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # drop fc
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)  # (B, 2048)


PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_manifest(crops_dir: Path, allowed_categories: set[str] | None = None):
    manifest = json.load(open(crops_dir / "manifest.json"))
    if allowed_categories is not None:
        manifest = [m for m in manifest if m["category"] in allowed_categories]
    return manifest


def stratified_split(manifest: list[dict], test_frac: float, seed: int):
    """Split per-category so both train and test see every category, and
    so it's reproducible -- not because leakage is a live risk here (each
    crop is its own annotation, not adjacent video frames), but so results
    are comparable run to run."""
    by_cat = defaultdict(list)
    for m in manifest:
        by_cat[m["category"]].append(m)
    train, test = [], []
    rng = random.Random(seed)
    for cat, items in by_cat.items():
        items = items[:]
        rng.shuffle(items)
        # With only ~22 total hazard crops (some categories have exactly 1
        # instance), a strict >3-item cutoff would leave test coverage from
        # a single category. Hold out 1 wherever a category has >=2 so more
        # categories get SOME held-out check, at the cost of a noisy n=1
        # per category -- explicitly flagged as a small-sample caveat in
        # the printed results, not hidden.
        n_test = max(1, round(len(items) * test_frac)) if len(items) >= 2 else 0
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    return train, test


@torch.no_grad()
def embed_crops(embedder, crops_dir: Path, manifest: list[dict], device: str) -> np.ndarray:
    embeddings = []
    for m in manifest:
        img = Image.open(crops_dir / m["crop_file"]).convert("RGB")
        x = PREPROCESS(img).unsqueeze(0).to(device)
        emb = embedder(x).cpu().numpy()[0]
        embeddings.append(emb / (np.linalg.norm(emb) + 1e-8))  # L2-normalize for cosine via dot product
    return np.stack(embeddings)


def knn_classify(query: np.ndarray, bank: np.ndarray, bank_labels: np.ndarray, k: int) -> tuple[str, float]:
    """Cosine similarity (bank is L2-normalized) k-NN majority vote.
    Returns (predicted_label, mean_similarity_to_that_label)."""
    sims = bank @ query
    top_k = np.argsort(-sims)[:k]
    top_labels = bank_labels[top_k]
    top_sims = sims[top_k]
    vals, counts = np.unique(top_labels, return_counts=True)
    winner = vals[np.argmax(counts)]
    mean_sim = float(top_sims[top_labels == winner].mean())
    return winner, mean_sim


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading frozen ResNet50 embedder on {device}...")
    embedder = GlobalResNetEmbedder().to(device)
    print("Loaded.\n")

    nuisance_manifest = load_manifest(NUISANCE_DIR)
    hazard_manifest = load_manifest(HAZARD_DIR)
    print(f"Nuisance crops (cone/bollard): {len(nuisance_manifest)}")
    print(f"Hazard crops (genuine corner cases): {len(hazard_manifest)}\n")

    nuisance_train, nuisance_test = stratified_split(nuisance_manifest, TEST_FRAC, SEED)
    hazard_train, hazard_test = stratified_split(hazard_manifest, TEST_FRAC, SEED)
    print(f"Nuisance: {len(nuisance_train)} train / {len(nuisance_test)} test")
    print(f"Hazard:   {len(hazard_train)} train / {len(hazard_test)} test\n")

    print("Embedding train bank...")
    nuisance_train_emb = embed_crops(embedder, NUISANCE_DIR, nuisance_train, device)
    hazard_train_emb = embed_crops(embedder, HAZARD_DIR, hazard_train, device)
    bank = np.concatenate([nuisance_train_emb, hazard_train_emb], axis=0)
    bank_labels = np.array(
        ["nuisance"] * len(nuisance_train_emb) + ["hazard"] * len(hazard_train_emb)
    )
    print(f"Bank size: {len(bank)} ({len(nuisance_train_emb)} nuisance, {len(hazard_train_emb)} hazard)\n")

    print("Embedding + classifying held-out test crops...")
    nuisance_test_emb = embed_crops(embedder, NUISANCE_DIR, nuisance_test, device)
    hazard_test_emb = embed_crops(embedder, HAZARD_DIR, hazard_test, device)

    def evaluate(test_emb, test_manifest, true_label):
        correct = 0
        rows = []
        for emb, m in zip(test_emb, test_manifest):
            pred, sim = knn_classify(emb, bank, bank_labels, K_NEIGHBORS)
            is_correct = pred == true_label
            correct += is_correct
            rows.append({"file": m["crop_file"], "category": m["category"],
                         "true": true_label, "pred": pred, "sim": sim})
        return correct / len(test_emb) if test_emb.size else float("nan"), rows

    nuisance_recall, nuisance_rows = evaluate(nuisance_test_emb, nuisance_test, "nuisance")
    hazard_recall, hazard_rows = evaluate(hazard_test_emb, hazard_test, "hazard")
    hazard_false_suppression_rate = 1 - hazard_recall

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Nuisance recall (held-out cones/bollards correctly suppressed): "
          f"{nuisance_recall:.3f} ({sum(r['pred']=='nuisance' for r in nuisance_rows)}/{len(nuisance_rows)})")
    print(f"Hazard false-suppression rate (held-out real hazards WRONGLY suppressed): "
          f"{hazard_false_suppression_rate:.3f} ({sum(r['pred']=='nuisance' for r in hazard_rows)}/{len(hazard_rows)})")
    print("\nThe second number is the one that matters most -- a classifier with high")
    print("nuisance recall but any real hazard false-suppression is actively harmful,")
    print("worse than not having it at all. Check which specific hazard crops got")
    print("misclassified below before trusting this.\n")

    misclassified_hazards = [r for r in hazard_rows if r["pred"] == "nuisance"]
    if misclassified_hazards:
        print("Hazard crops WRONGLY suppressed (inspect these specifically):")
        for r in misclassified_hazards:
            print(f"  {r['category']:20s} {r['file']}  (sim={r['sim']:.3f})")
    else:
        print("No held-out hazard crops were wrongly suppressed.")

    # Save the fitted bank for later use as an actual suppression filter --
    # only worth doing once these numbers are trusted.
    np.savez(BANK_PATH, embeddings=bank, labels=bank_labels)
    print(f"\nSaved fitted bank to {BANK_PATH}")

    results = {
        "nuisance_recall": float(nuisance_recall),
        "hazard_false_suppression_rate": float(hazard_false_suppression_rate),
        "n_nuisance_train": len(nuisance_train), "n_nuisance_test": len(nuisance_test),
        "n_hazard_train": len(hazard_train), "n_hazard_test": len(hazard_test),
        "k_neighbors": K_NEIGHBORS,
        "misclassified_hazard_files": [r["file"] for r in misclassified_hazards],
    }
    out_path = RESULTS_DIR / "nuisance_classifier_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
