"""Evaluate frozen vs fine-tuned ResNet50+kNN on the same-domain synthetic
corner-case benchmark.

This is the honesty check: the earlier nuScenes-vs-CODA AUROC of 97.58%
could have been measuring "which dataset is this" rather than "is something
dangerous happening" (different camera, city, compression, color grading).
Here, the ONLY difference between a normal and anomalous frame is the
presence of one real, photographed object with no standard AV perception
class -- everything else (background, camera, lighting) is identical.

Runs BOTH the original frozen ImageNet ResNet50 and the fine-tuned backbone,
so we can see whether fine-tuning actually helped detect real novel objects,
or whether it was just exploiting the nuScenes/CODA domain gap.
"""

from __future__ import annotations
import sys
import json
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

RESULTS_DIR = Path("/Volumes/BIggen/AV/results")
SYNTH_DIR = Path("/Volumes/BIggen/AV/data/synthetic_corner_cases")


class ResNetEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        feat = self.backbone(x)
        return feat.view(feat.size(0), -1)


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
    model.eval()
    out = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i + batch_size]).to(device)
        feats = model(batch)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0)


def load_finetuned_model(device: str) -> ResNetEmbedder:
    model = ResNetEmbedder().to(device)
    state_dict = torch.load(RESULTS_DIR / "resnet50_backbone_finetuned.pt", map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        backbone_state = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
        model.backbone.load_state_dict(backbone_state)
    return model


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")

    manifest = json.load(open(SYNTH_DIR / "synthetic_manifest.json"))
    print(f"Synthetic anomaly composites: {len(manifest)}")

    # Normal set = the unique untouched canvas frames used to build composites.
    # Same background, same camera, same lighting as the anomalies -- the
    # only variable that differs is "was a hazard object pasted here."
    unique_canvases = sorted(set(m["canvas_path"] for m in manifest))
    print(f"Unique normal (untouched) canvases: {len(unique_canvases)}\n")

    # Load and tensor-ify all images once
    print("Loading synthetic anomaly images...")
    anomaly_tensors, anomaly_categories = [], []
    for m in manifest:
        t = load_image_as_tensor(str(SYNTH_DIR / "images" / m["file"]), device)
        if t is not None:
            anomaly_tensors.append(t)
            anomaly_categories.append(m["category"])

    print(f"Loaded {len(anomaly_tensors)} anomaly images")

    print("Loading normal canvas images...")
    normal_tensors = []
    for p in unique_canvases:
        t = load_image_as_tensor(p, device)
        if t is not None:
            normal_tensors.append(t)
    print(f"Loaded {len(normal_tensors)} normal images\n")

    labels = np.concatenate([np.zeros(len(normal_tensors)), np.ones(len(anomaly_tensors))])
    all_tensors = normal_tensors + anomaly_tensors

    results = {}

    for model_name, train_emb_file, load_fn in [
        ("frozen_imagenet", "phase2_embeddings_train.npy", lambda: ResNetEmbedder().to(device)),
        ("finetuned", "phase2_embeddings_train_v2.npy", lambda: load_finetuned_model(device)),
    ]:
        print("=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        model = load_fn()
        embeddings = embed_batch(model, all_tensors, device)

        train_emb = np.load(RESULTS_DIR / train_emb_file)
        scorer = kNNScorer(k=5)
        scorer.fit(train_emb)
        scores = scorer.score(embeddings)

        overall_auroc = auroc(scores, labels)
        overall_aupr = aupr(scores, labels)
        print(f"Overall AUROC: {overall_auroc:.4f}")
        print(f"Overall AUPR:  {overall_aupr:.4f}\n")

        # Per-category breakdown (compare each category's anomaly scores
        # against the same normal pool)
        normal_scores = scores[:len(normal_tensors)]
        anomaly_scores = scores[len(normal_tensors):]

        cat_results = {}
        by_cat = defaultdict(list)
        for score, cat in zip(anomaly_scores, anomaly_categories):
            by_cat[cat].append(score)

        print(f"{'Category':22s} {'N':>4s} {'AUROC':>8s} {'mean score':>12s}")
        for cat, cat_scores in sorted(by_cat.items()):
            cat_scores = np.array(cat_scores)
            cat_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(cat_scores))])
            cat_all_scores = np.concatenate([normal_scores, cat_scores])
            try:
                cat_auroc = auroc(cat_all_scores, cat_labels)
            except ValueError:
                cat_auroc = float("nan")
            cat_results[cat] = {"n": len(cat_scores), "auroc": float(cat_auroc), "mean_score": float(cat_scores.mean())}
            print(f"{cat:22s} {len(cat_scores):4d} {cat_auroc:8.4f} {cat_scores.mean():12.4f}")

        print(f"\nNormal score distribution: mean={normal_scores.mean():.4f} std={normal_scores.std():.4f}\n")

        results[model_name] = {
            "overall_auroc": float(overall_auroc),
            "overall_aupr": float(overall_aupr),
            "normal_mean": float(normal_scores.mean()),
            "normal_std": float(normal_scores.std()),
            "by_category": cat_results,
        }

    print("=" * 70)
    print("FROZEN vs FINE-TUNED -- DID FINE-TUNING ACTUALLY HELP?")
    print("=" * 70)
    frozen_auroc = results["frozen_imagenet"]["overall_auroc"]
    finetuned_auroc = results["finetuned"]["overall_auroc"]
    print(f"Frozen ImageNet AUROC:  {frozen_auroc:.4f}")
    print(f"Fine-tuned AUROC:       {finetuned_auroc:.4f}")
    if finetuned_auroc > frozen_auroc + 0.03:
        print("-> Fine-tuning genuinely helped on same-domain novel objects.")
    elif finetuned_auroc < frozen_auroc - 0.03:
        print("-> Fine-tuning made same-domain detection WORSE. It likely overfit")
        print("   to the nuScenes/CODA domain gap rather than learning semantics.")
    else:
        print("-> Roughly the same. Fine-tuning didn't add real semantic value here")
        print("   -- the earlier 97.58% was probably mostly domain-shift detection.")
    print("=" * 70 + "\n")

    out_path = RESULTS_DIR / "synthetic_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()
