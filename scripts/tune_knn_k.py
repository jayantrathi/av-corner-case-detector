"""Find optimal k for k-NN scorer by trying different k values."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to find src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring.embedding_scorers import kNNScorer
from src.eval.metrics import auroc, aupr

results_dir = Path(__file__).parent.parent / "results"

# Load Phase 3 evaluation data
phase3_results = json.load(open(results_dir / "phase3_results.json"))

# Extract scores and labels from Phase 3
# Phase 3 already computed scores, so use those to determine labels
eval_embeddings = np.load(results_dir / "phase3_embeddings_eval.npy")
train_embeddings = np.load(results_dir / "phase2_embeddings_train.npy")

# Reconstruct labels from Phase 3 metadata
n_normal = phase3_results["evaluation_set"]["normal_frames"]
n_anomaly = phase3_results["evaluation_set"]["anomaly_frames"]

# Ensure we have the right length
n_total = len(eval_embeddings)
labels = np.zeros(n_total)
labels[n_normal:min(n_normal + n_anomaly, n_total)] = 1

print("Finding optimal k for k-NN...\n")
print("k\tAUROC\tAUPR")
print("-" * 40)

best_k = 1
best_auroc = 0

for k in [1, 3, 5, 7, 10, 15, 20, 30]:
    if k >= len(train_embeddings):
        print(f"{k}\tskipped (k > training set size)")
        continue

    scorer = kNNScorer(k=k)
    scorer.fit(train_embeddings)
    scores = scorer.score(eval_embeddings)

    roc = auroc(scores, labels)
    pr = aupr(scores, labels)

    print(f"{k}\t{roc:.4f}\t{pr:.4f}")

    if roc > best_auroc:
        best_auroc = roc
        best_k = k

print("-" * 40)
print(f"\nBest k: {best_k} (AUROC: {best_auroc:.4f})")
