"""Phase 2: Embedding-based anomaly detection with k-NN scoring.

Load precomputed embeddings, fit k-NN on normal training embeddings,
score test/val, compute metrics, and compare to Phase 1 baseline.
"""

from __future__ import annotations
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scoring.embedding_scorers import kNNScorer
from src.eval.metrics import auroc, aupr


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Embedding-based evaluation")
    parser.add_argument(
        "--embedding-dir",
        default="/Volumes/BIggen/AV/results",
        help="Directory with precomputed embeddings",
    )
    parser.add_argument(
        "--results-dir",
        default="/Volumes/BIggen/AV/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="k for k-NN"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading precomputed embeddings...")
    train_emb = np.load(Path(args.embedding_dir) / "phase2_embeddings_train.npy")
    val_emb = np.load(Path(args.embedding_dir) / "phase2_embeddings_val.npy")
    test_emb = np.load(Path(args.embedding_dir) / "phase2_embeddings_test.npy")

    print(f"Train embeddings: {train_emb.shape}")
    print(f"Val embeddings: {val_emb.shape}")
    print(f"Test embeddings: {test_emb.shape}\n")

    # Fit k-NN scorer on normal training embeddings
    print(f"Fitting k-NN scorer (k={args.k}) on training embeddings...")
    scorer = kNNScorer(k=args.k)
    scorer.fit(train_emb)

    # Score val and test
    print("Computing anomaly scores...")
    val_scores = scorer.score(val_emb)
    test_scores = scorer.score(test_emb)

    print(f"Val scores: mean={val_scores.mean():.6f}, std={val_scores.std():.6f}")
    print(f"Test scores: mean={test_scores.mean():.6f}, std={test_scores.std():.6f}\n")

    # Since all frames are normal (no corner cases in mini dataset yet),
    # we can't compute real AUROC/AUPR. Instead, show score distributions.
    # In Phase 3, we'll test on actual corner-case datasets.

    print("Saving results...")
    phase2_results = {
        "method": "k-NN on frozen ResNet50 embeddings",
        "k": args.k,
        "embedding_dim": int(train_emb.shape[1]),
        "val_scores": val_scores.tolist(),
        "test_scores": test_scores.tolist(),
        "val_mean": float(val_scores.mean()),
        "val_std": float(val_scores.std()),
        "test_mean": float(test_scores.mean()),
        "test_std": float(test_scores.std()),
    }

    results_path = results_dir / "phase2_results.json"
    with open(results_path, "w") as f:
        json.dump(phase2_results, f, indent=2)
    print(f"Results saved to {results_path}\n")

    # Load Phase 1 results for comparison
    try:
        with open(results_dir / "phase1_results.json") as f:
            phase1_results = json.load(f)

        print("=" * 60)
        print("PHASE 1 vs PHASE 2 COMPARISON")
        print("=" * 60)
        print(f"\nPhase 1 (Autoencoder):")
        phase1_val = np.array(phase1_results.get('val_scores', []))
        phase1_test = np.array(phase1_results.get('test_scores', []))
        if len(phase1_val) > 0:
            print(f"  Val scores: mean={np.mean(phase1_val):.6f}, std={np.std(phase1_val):.6f}")
        if len(phase1_test) > 0:
            print(f"  Test scores: mean={np.mean(phase1_test):.6f}, std={np.std(phase1_test):.6f}")

        print(f"\nPhase 2 (k-NN + ResNet50):")
        print(f"  Val scores: mean={val_scores.mean():.6f}, std={val_scores.std():.6f}")
        print(f"  Test scores: mean={test_scores.mean():.6f}, std={test_scores.std():.6f}")
        print("=" * 60)
        print("\nNote: Scores are on normal frames only (mini dataset).")
        print("Real AUROC/AUPR requires corner-case test set (Phase 3).\n")

    except FileNotFoundError:
        print("Phase 1 results not found; skipping comparison.\n")

    # Plot score distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Val
    axes[0].hist(val_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel("k-NN Anomaly Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Phase 2: Validation Scores (k-NN, ResNet50)")
    axes[0].grid()

    # Test
    axes[1].hist(test_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel("k-NN Anomaly Score")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Phase 2: Test Scores (k-NN, ResNet50)")
    axes[1].grid()

    fig.tight_layout()
    fig.savefig(results_dir / "phase2_score_distributions.png", dpi=100, bbox_inches='tight')
    print(f"Score distribution plot saved\n")

    print("Phase 2 evaluation complete!")
    print("Next: Phase 3 — proper evaluation on corner-case dataset\n")


if __name__ == "__main__":
    main()
