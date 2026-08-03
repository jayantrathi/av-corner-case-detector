"""Phase 3 v2: Rigorous evaluation with improved models.

After running the two improvement scripts:
- scripts/train_autoencoder_v2.py (50 epochs)
- scripts/finetune_resnet_backbone.py (fine-tuned backbone)
- scripts/extract_embeddings_v2.py (embeddings from fine-tuned model)

This script evaluates both improved detectors on the CODA corner-case dataset.
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
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    """Extract embeddings for a batch of images."""
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

    # For Phase 2, we use pre-extracted embeddings which have all 281 samples
    # Create labels for all 281 (matching embeddings shape)
    n_normal = len(test_normal_frames)
    n_anomaly = len(coda_frames)
    labels_full = np.concatenate([
        np.zeros(n_normal, dtype=int),    # 0 = normal
        np.ones(n_anomaly, dtype=int)     # 1 = anomaly
    ])

    # Load images for Phase 1 (autoencoder needs raw images)
    # Only load successfully; track which indices succeeded
    print("Loading images for Phase 1 evaluation...")
    images_tensor = []
    valid_indices_phase1 = []

    for idx, (path, label) in enumerate(eval_frames):
        try:
            img = load_image_as_tensor(path, device)
            images_tensor.append(img)
            valid_indices_phase1.append(idx)
        except Exception as e:
            pass  # Skip failed loads silently

    # For Phase 1, use labels corresponding to successfully loaded images
    labels_phase1 = labels_full[valid_indices_phase1]

    print(f"Successfully loaded {len(images_tensor)} images for Phase 1")
    print(f"Using {len(labels_full)} pre-extracted embeddings for Phase 2\n")

    # ============================================================================
    # PHASE 1 v2: Improved Autoencoder (50 epochs)
    # ============================================================================
    print("=" * 60)
    print("PHASE 1 v2: IMPROVED AUTOENCODER (50 epochs)")
    print("=" * 60)

    model_v2_path = results_dir / "autoencoder_phase1_v2.pt"
    if model_v2_path.exists():
        print("Loading improved Phase 1 model...")
        model_phase1_v2 = ConvAutoencoder(latent_dim=128).to(device)
        model_phase1_v2.load_state_dict(torch.load(model_v2_path, map_location=device))

        print("Computing reconstruction error scores...")
        phase1_v2_scores = score_phase1(model_phase1_v2, images_tensor, device)

        phase1_v2_auroc = auroc(phase1_v2_scores, labels_phase1)
        phase1_v2_aupr = aupr(phase1_v2_scores, labels_phase1)

        print(f"AUROC: {phase1_v2_auroc:.4f}")
        print(f"AUPR:  {phase1_v2_aupr:.4f}\n")
    else:
        print(f"Warning: {model_v2_path} not found. Run train_autoencoder_v2.py first.\n")
        phase1_v2_auroc = None
        phase1_v2_aupr = None
        phase1_v2_scores = None

    # ============================================================================
    # PHASE 2 v2: Improved k-NN with Fine-tuned ResNet50
    # ============================================================================
    print("=" * 60)
    print("PHASE 2 v2: k-NN WITH FINE-TUNED RESNET50")
    print("=" * 60)

    embeddings_v2_path = results_dir / "phase3_embeddings_eval_v2.npy"
    train_emb_v2_path = results_dir / "phase2_embeddings_train_v2.npy"

    if embeddings_v2_path.exists() and train_emb_v2_path.exists():
        print("Loading pre-extracted embeddings from fine-tuned ResNet50...")
        embeddings_v2 = np.load(embeddings_v2_path)  # Already combined: test + CODA
        train_emb_v2 = np.load(train_emb_v2_path)

        print("Computing k-NN scores (k=5)...")
        phase2_v2_scores = score_phase2(embeddings_v2, train_emb_v2, k=5)

        phase2_v2_auroc = auroc(phase2_v2_scores, labels_full)
        phase2_v2_aupr = aupr(phase2_v2_scores, labels_full)

        print(f"AUROC: {phase2_v2_auroc:.4f}")
        print(f"AUPR:  {phase2_v2_aupr:.4f}\n")
    else:
        missing_files = []
        if not embeddings_v2_path.exists():
            missing_files.append(f"  - {embeddings_v2_path}")
        if not train_emb_v2_path.exists():
            missing_files.append(f"  - {train_emb_v2_path}")
        print(f"Warning: Phase 2 v2 embeddings missing:\n" + "\n".join(missing_files))
        print(f"Run extract_embeddings_v2.py first.\n")
        phase2_v2_auroc = None
        phase2_v2_aupr = None
        phase2_v2_scores = None

    # ============================================================================
    # COMPARISON: Original vs Improved
    # ============================================================================
    print("=" * 60)
    print("COMPARISON: ORIGINAL vs IMPROVED MODELS")
    print("=" * 60)

    # Load original Phase 3 results for comparison
    original_results_path = results_dir / "phase3_results.json"
    if original_results_path.exists():
        with open(original_results_path) as f:
            original_results = json.load(f)

        phase1_orig_auroc = original_results["phase1_autoencoder"]["auroc"]
        phase1_orig_aupr = original_results["phase1_autoencoder"]["aupr"]
        phase2_orig_auroc = original_results["phase2_knn_resnet50"]["auroc"]
        phase2_orig_aupr = original_results["phase2_knn_resnet50"]["aupr"]

        print(f"\nPHASE 1 (Autoencoder):")
        print(f"  Original:  AUROC={phase1_orig_auroc:.4f}, AUPR={phase1_orig_aupr:.4f}")
        if phase1_v2_auroc is not None:
            print(f"  Improved:  AUROC={phase1_v2_auroc:.4f}, AUPR={phase1_v2_aupr:.4f}")
            improvement = (phase1_v2_auroc - phase1_orig_auroc) / phase1_orig_auroc * 100
            print(f"  Change:    {improvement:+.1f}%")

        print(f"\nPHASE 2 (k-NN + ResNet50):")
        print(f"  Original:  AUROC={phase2_orig_auroc:.4f}, AUPR={phase2_orig_aupr:.4f}")
        if phase2_v2_auroc is not None:
            print(f"  Improved:  AUROC={phase2_v2_auroc:.4f}, AUPR={phase2_v2_aupr:.4f}")
            improvement = (phase2_v2_auroc - phase2_orig_auroc) / phase2_orig_auroc * 100
            print(f"  Change:    {improvement:+.1f}%")

    print("=" * 60 + "\n")

    # Save results
    results = {
        "evaluation_set": {
            "normal_frames": len(test_normal_frames),
            "anomaly_frames": len(coda_frames),
            "total_frames": len(eval_frames),
        },
    }

    if phase1_v2_auroc is not None:
        results["phase1_autoencoder_v2"] = {
            "auroc": float(phase1_v2_auroc),
            "aupr": float(phase1_v2_aupr),
            "scores": phase1_v2_scores.tolist() if phase1_v2_scores is not None else [],
        }

    if phase2_v2_auroc is not None:
        results["phase2_knn_resnet50_v2"] = {
            "auroc": float(phase2_v2_auroc),
            "aupr": float(phase2_v2_aupr),
            "scores": phase2_v2_scores.tolist() if phase2_v2_scores is not None else [],
        }

    results_path = results_dir / "phase3_results_v2.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}\n")

    # Plot comparison ROC curves if both models worked
    if phase1_v2_scores is not None and phase2_v2_scores is not None:
        from sklearn.metrics import roc_curve

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Phase 1 ROC (uses only successfully loaded images)
        fpr1, tpr1, _ = roc_curve(labels_phase1, phase1_v2_scores)
        axes[0].plot(fpr1, tpr1, linewidth=2, label=f"Phase 1 v2 (AUROC={phase1_v2_auroc:.3f})")
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("Phase 1 v2: Improved Autoencoder ROC Curve")
        axes[0].legend()
        axes[0].grid()

        # Phase 2 ROC (uses all 281 pre-extracted embeddings)
        fpr2, tpr2, _ = roc_curve(labels_full, phase2_v2_scores)
        axes[1].plot(fpr2, tpr2, linewidth=2, label=f"Phase 2 v2 (AUROC={phase2_v2_auroc:.3f})")
        axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("Phase 2 v2: k-NN + Fine-tuned ResNet50 ROC Curve")
        axes[1].legend()
        axes[1].grid()

        fig.tight_layout()
        fig.savefig(results_dir / "phase3_roc_curves_v2.png", dpi=100, bbox_inches='tight')
        print("ROC curves saved\n")

    print("Phase 3 v2 evaluation complete!")


if __name__ == "__main__":
    main()
