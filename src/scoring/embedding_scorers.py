"""Embedding-based anomaly scoring (Approach B — the recommended primary).

Idea: a frozen pretrained vision backbone maps each frame to a feature
vector. Fit the distribution of NORMAL feature vectors. At test time, a
frame whose features sit far from the normal cloud is anomalous.

Two scorers here, both standard and strong:
  - kNNScorer:        distance to k nearest normal neighbors. Simple, robust,
                      hard to beat. This is your baseline AND often your best.
  - MahalanobisScorer: distance under the normal feature covariance. Cheap at
                      test time, strong when features are roughly Gaussian.

Both fit ONLY on normal training features. Never fit on anomalies, never fit
on the test set.
"""
from __future__ import annotations
import numpy as np


class kNNScorer:
    """Anomaly score = mean distance to the k nearest normal training features."""

    def __init__(self, k: int = 5):
        self.k = k
        self._train: np.ndarray | None = None

    def fit(self, normal_feats: np.ndarray) -> "kNNScorer":
        """normal_feats: (N, D) features from NORMAL frames only."""
        self._train = np.asarray(normal_feats, dtype=np.float32)
        if self._train.ndim != 2:
            raise ValueError("expected (N, D) feature matrix")
        return self

    def score(self, feats: np.ndarray, batch: int = 512) -> np.ndarray:
        """Return one anomaly score per row of feats (higher = more anomalous)."""
        if self._train is None:
            raise RuntimeError("call fit() first")
        feats = np.asarray(feats, dtype=np.float32)
        out = np.empty(len(feats), dtype=np.float32)
        tr = self._train
        tr_sq = (tr ** 2).sum(1)  # precompute
        for i in range(0, len(feats), batch):
            chunk = feats[i:i + batch]
            # squared euclidean distances (chunk x train)
            d = (chunk ** 2).sum(1)[:, None] - 2 * chunk @ tr.T + tr_sq[None, :]
            d = np.maximum(d, 0)
            # k smallest per row
            kth = np.partition(d, self.k, axis=1)[:, :self.k]
            out[i:i + batch] = np.sqrt(kth).mean(1)
        return out


class MahalanobisScorer:
    """Anomaly score = Mahalanobis distance to the mean of normal features."""

    def __init__(self, shrinkage: float = 1e-3):
        self.shrinkage = shrinkage
        self._mean = None
        self._inv_cov = None

    def fit(self, normal_feats: np.ndarray) -> "MahalanobisScorer":
        X = np.asarray(normal_feats, dtype=np.float64)
        self._mean = X.mean(0)
        cov = np.cov(X, rowvar=False)
        # shrink toward diagonal for numerical stability / small-sample safety
        cov += self.shrinkage * np.eye(cov.shape[0])
        self._inv_cov = np.linalg.inv(cov)
        return self

    def score(self, feats: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("call fit() first")
        X = np.asarray(feats, dtype=np.float64)
        diff = X - self._mean
        # sqrt( (x-mu)^T Sigma^-1 (x-mu) ) per row
        m = np.einsum("ij,jk,ik->i", diff, self._inv_cov, diff)
        return np.sqrt(np.maximum(m, 0)).astype(np.float32)


if __name__ == "__main__":
    # Self-check: normal features clustered, anomalies offset. Both scorers
    # should rank anomalies higher.
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.eval.metrics import summarize

    rng = np.random.default_rng(0)
    D = 64
    normal_train = rng.normal(0, 1, (2000, D))
    normal_test = rng.normal(0, 1, (500, D))
    anom_test = rng.normal(0.6, 1, (100, D))  # subtle shift, realistic difficulty

    test_feats = np.vstack([normal_test, anom_test])
    labels = np.concatenate([np.zeros(500), np.ones(100)])

    for name, scorer in [("kNN", kNNScorer(k=5)), ("Mahalanobis", MahalanobisScorer())]:
        scorer.fit(normal_train)
        s = scorer.score(test_feats)
        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in summarize(s, labels).items()))
