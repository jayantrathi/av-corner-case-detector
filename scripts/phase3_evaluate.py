"""Phase 3: Rigorous evaluation on corner-case dataset.

Load CODA corner cases + nuScenes normal frames.
Evaluate both Phase 1 and Phase 2 detectors.
Compute AUROC/AUPR and generate comparison plots.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.autoencoder import ConvAutoencoder
from src.scoring.embedding_scorers import kNNScorer
from src.eval.metrics import auroc, aupr
from src.data.nuscenes_loader import load_nuscenes_frames
from src.data.splits import split_by_scene


def load_image_as_tensor(path: str, device: str) -> torch.Tensor:
    """Load and normalize image to tensor."""
    img = Image.open(path).convert('RGB')
    img = img.resize((1600, 900), Image.Resampling.LANCZOS)
    img = np.array(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img.to(device)


def load_coda_frames(coda_path: str) -> list[tuple[str, int]]:
    """Load CODA corner-case images. Return list of (path, is_anomaly=1)."""
    coda_dir = Path(coda_path) / 'CODA' / 'sample'
    images_dir = coda_dir / 'images'

    frames = []
    for img_file in sorted(images_dir.glob('*.jpg')):
        frames.append((str(img_file), 1))  # 1 = anomaly

    return frames


@torch.no_grad()
def score_phase1(model, images_tensor: list[torch.Tensor], device: str) -> np.ndarray:
    """Compute reconstruction error scores (Phase 1)."""
    model.eval()
    scores = []

    for img in images_tensor:
        error = model.anomaly_score(img.unsqueeze(0).to(device))
        scores.append(error.cpu().item())

    return np.array(scores)


def score_phase2(embeddings: np.ndarray, train_emb: np.ndarray, k: int = 5) -> np.ndarray:
    """Compute k-NN scores (Phase 2)."""
    scorer = kNNScorer(k=k)
    scorer.fit(train_emb)
    return scorer.score(embeddings)


def get_embeddings_batch(model, images_tensor: list[torch.Tensor], device: str) -> np.ndarray:
    """Extract ResNet50 embeddings for a batch of images."""
    model.eval()
    embeddings = []

    for img in images_tensor:
        with torch.no_grad():
            feat = model(img.unsqueeze(0).to(device))
        embeddings.append(feat.cpu().numpy().squeeze())

    return np.array(embeddings)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")

    results_dir = Path("/Volumes/BIggen/AV/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    coda_path = "/Volumes/BIggen/AV/data/coda"

    # Load normal frames from nuScenes
    print("Loading nuScenes normal frames...")
    frames = load_nuscenes_frames("/Volumes/BIggen/AV/data/nuscenes/v1.0-mini")
    splits = split_by_scene(frames, val_frac=0.15, test_frac=0.15, seed=0)
    test_normal_frames = splits["test"]
    print(f"Normal test frames: {len(test_normal_frames)}\n")

    # Load corner cases from CODA
    print("Loading CODA corner-case frames...")
    coda_frames = load_coda_frames(coda_path)
    print(f"Corner-case frames: {len(coda_frames)}\n")

    # Create evaluation set: normal + corner cases
    eval_frames = [
        (f.path, 0) for f in test_normal_frames  # 0 = normal
    ] + coda_frames  # 1 = anomaly

    print(f"Total evaluation frames: {len(eval_frames)}")
    print(f"  Normal: {len(test_normal_frames)}")
    print(f"  Anomalies: {len(coda_frames)}\n")

    # Load and prepare images
    print("Loading images...")
    images_tensor = []
    labels = []

    for path, label in eval_frames:
        try:
            img = load_image_as_tensor(path, device)
            images_tensor.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {path}: {e}")

    labels = np.array(labels)
    print(f"Successfully loaded {len(images_tensor)} images\n")

    # Phase 1: Autoencoder scores
    print("=" * 60)
    print("PHASE 1: AUTOENCODER RECONSTRUCTION ERROR")
    print("=" * 60)

    print("Loading Phase 1 model...")
    model_phase1 = ConvAutoencoder(latent_dim=128).to(device)
    model_phase1.load_state_dict(
        torch.load(results_dir / "autoencoder_phase1.pt", map_location=device)
    )

    print("Computing reconstruction error scores...")
    phase1_scores = score_phase1(model_phase1, images_tensor, device)

    phase1_auroc = auroc(phase1_scores, labels)
    phase1_aupr = aupr(phase1_scores, labels)

    print(f"AUROC: {phase1_auroc:.4f}")
    print(f"AUPR:  {phase1_aupr:.4f}\n")

    # Phase 2: k-NN scores
    print("=" * 60)
    print("PHASE 2: k-NN ON FROZEN RESNET50 EMBEDDINGS")
    print("=" * 60)

    print("Loading ResNet50...")
    import torchvision.models as models
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model_phase2 = nn.Sequential(*list(resnet.children())[:-1]).to(device)

    print("Extracting embeddings...")
    embeddings = get_embeddings_batch(model_phase2, images_tensor, device)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)  # Flatten to (N, 2048)

    # Save evaluation embeddings for tuning
    np.save(results_dir / "phase3_embeddings_eval.npy", embeddings)

    # Load training embeddings for k-NN fitting
    train_emb = np.load(results_dir / "phase2_embeddings_train.npy")

    print("Computing k-NN scores...")
    phase2_scores = score_phase2(embeddings, train_emb, k=5)

    phase2_auroc = auroc(phase2_scores, labels)
    phase2_aupr = aupr(phase2_scores, labels)

    print(f"AUROC: {phase2_auroc:.4f}")
    print(f"AUPR:  {phase2_aupr:.4f}\n")

    # Save results
    print("=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"\nPhase 1 (Autoencoder):")
    print(f"  AUROC: {phase1_auroc:.4f}")
    print(f"  AUPR:  {phase1_aupr:.4f}")

    print(f"\nPhase 2 (k-NN + ResNet50):")
    print(f"  AUROC: {phase2_auroc:.4f}")
    print(f"  AUPR:  {phase2_aupr:.4f}")

    if phase2_auroc > phase1_auroc:
        improvement = (phase2_auroc - phase1_auroc) / phase1_auroc * 100
        print(f"\nPhase 2 wins: {improvement:.1f}% better AUROC")
    else:
        improvement = (phase1_auroc - phase2_auroc) / phase2_auroc * 100
        print(f"\nPhase 1 wins: {improvement:.1f}% better AUROC")
    print("=" * 60 + "\n")

    # Save metrics
    results = {
        "evaluation_set": {
            "normal_frames": len(test_normal_frames),
            "anomaly_frames": len(coda_frames),
            "total_frames": len(eval_frames),
        },
        "phase1_autoencoder": {
            "auroc": float(phase1_auroc),
            "aupr": float(phase1_aupr),
            "scores": phase1_scores.tolist(),
        },
        "phase2_knn_resnet50": {
            "auroc": float(phase2_auroc),
            "aupr": float(phase2_aupr),
            "scores": phase2_scores.tolist(),
        },
    }

    results_path = results_dir / "phase3_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}\n")

    # Plot ROC curves
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Phase 1 ROC
    fpr1, tpr1, _ = roc_curve(labels, phase1_scores)
    axes[0].plot(fpr1, tpr1, linewidth=2, label=f"Phase 1 (AUROC={phase1_auroc:.3f})")
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Phase 1: Autoencoder ROC Curve")
    axes[0].legend()
    axes[0].grid()

    # Phase 2 ROC
    fpr2, tpr2, _ = roc_curve(labels, phase2_scores)
    axes[1].plot(fpr2, tpr2, linewidth=2, label=f"Phase 2 (AUROC={phase2_auroc:.3f})")
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Phase 2: k-NN + ResNet50 ROC Curve")
    axes[1].legend()
    axes[1].grid()

    fig.tight_layout()
    fig.savefig(results_dir / "phase3_roc_curves.png", dpi=100, bbox_inches='tight')
    print("ROC curves saved\n")

    print("Phase 3 evaluation complete!")
    print("Next: Phase 4 — demo video on held-out driving clip\n")


if __name__ == "__main__":
    main()
