"""Geometric ground-plane reasoning for anomaly detection.

The idea (this is the Stage-3 novel contribution):
  Appearance-only anomaly detectors false-fire on flat road markings, manhole
  covers, and painted arrows -- they LOOK unusual but they lie IN the road
  plane. A real obstacle (tire, box, debris, lost cargo) physically STICKS UP
  out of that plane. So if we estimate the road's 3D plane and measure each
  pixel's height above it, we can suppress the "anomaly" on things that are
  coplanar with the road while keeping it on things that protrude.

Pipeline:
  1. Monocular depth from an off-the-shelf model (Depth Anything V2). This is a
     MODULE, not our contribution -- the contribution is the fusion below.
  2. Back-project pixels to 3D camera coordinates using an assumed pinhole
     (focal length from a nominal horizontal FOV; principal point at center).
  3. Fit the road plane by RANSAC over the PREDICTED-road pixels only (no label
     peek -- the road mask comes from the segmenter's own argmax).
  4. height(u,v) = |signed distance to that plane|, normalized by the median
     road depth so it's robust to monocular depth's unknown global scale.

Coplanar paint/manholes -> height ~ 0. Protruding obstacles -> height large.

Note on honesty: monocular depth has no true metric scale, so we normalize by
road depth and work with a *relative* height. That's enough to separate
"in the plane" from "sticking out", which is all the gate needs.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class GroundPlaneHeight:
    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str | None = None,
        fov_deg: float = 64.0,
        ransac_iters: int = 200,
        ransac_tol: float = 0.03,
        seed: int = 0,
    ):
        from transformers import pipeline

        self.device = device or _pick_device()
        # transformers depth pipeline: returns {"predicted_depth", "depth"}.
        # Depth Anything outputs an inverse-depth / disparity-like map
        # (larger value = closer). We convert to pseudo-depth z = 1/disp below.
        self.pipe = pipeline("depth-estimation", model=model_name,
                             device=0 if self.device == "cuda" else -1)
        self.fov_deg = fov_deg
        self.ransac_iters = ransac_iters
        self.ransac_tol = ransac_tol
        self.rng = np.random.default_rng(seed)

    def _disparity(self, img: Image.Image) -> np.ndarray:
        out = self.pipe(img)
        d = out["predicted_depth"]
        if isinstance(d, torch.Tensor):
            d = d.squeeze().float().cpu().numpy()
        else:
            d = np.asarray(d, dtype=np.float32)
        return d  # H,W, larger = closer (disparity-like)

    @staticmethod
    def _intrinsics(w: int, h: int, fov_deg: float):
        fx = 0.5 * w / np.tan(np.radians(fov_deg) / 2.0)
        fy = fx
        cx, cy = w / 2.0, h / 2.0
        return fx, fy, cx, cy

    def _fit_plane_ransac(self, pts: np.ndarray):
        """pts: (N,3). Returns (unit_normal(3,), offset) for n·P + offset = 0,
        fit robustly to the dominant (road) plane."""
        N = len(pts)
        best_inl, best = -1, None
        tol = self.ransac_tol * (np.median(np.abs(pts[:, 2])) + 1e-6)
        for _ in range(self.ransac_iters):
            i = self.rng.choice(N, 3, replace=False)
            p0, p1, p2 = pts[i]
            n = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(n)
            if nn < 1e-8:
                continue
            n = n / nn
            off = -n @ p0
            dist = np.abs(pts @ n + off)
            inl = int((dist < tol).sum())
            if inl > best_inl:
                best_inl, best = inl, (n, off, dist < tol)
        if best is None:  # degenerate fallback: least-squares plane
            n, off = self._lsq_plane(pts)
            return n, off
        n, off, mask = best
        # refit least-squares on inliers for precision
        n, off = self._lsq_plane(pts[mask])
        return n, off

    @staticmethod
    def _lsq_plane(pts: np.ndarray):
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c)
        n = vt[-1]
        n = n / (np.linalg.norm(n) + 1e-12)
        off = -n @ c
        return n, off

    def height_map(self, img: Image.Image, road_mask: np.ndarray) -> np.ndarray:
        """img: PIL RGB. road_mask: bool HxW at the image's (resized) resolution
        marking predicted-road pixels. Returns relative height-above-plane map
        (HxW float, >=0), same resolution as road_mask."""
        H, W = road_mask.shape
        disp = self._disparity(img)
        if disp.shape != (H, W):
            disp = np.array(Image.fromarray(disp).resize((W, H), Image.Resampling.BILINEAR))
        z = 1.0 / (np.clip(disp, 1e-3, None))  # pseudo-depth, up to global scale

        fx, fy, cx, cy = self._intrinsics(W, H, self.fov_deg)
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        X = (us - cx) * z / fx
        Y = (vs - cy) * z / fy
        P = np.stack([X, Y, z], axis=-1)  # H,W,3

        road = road_mask & np.isfinite(z) & (z < np.percentile(z[road_mask], 99) if road_mask.any() else True)
        road_pts = P[road_mask]
        if len(road_pts) < 50:  # not enough road to fit -> no gating signal
            return np.zeros((H, W), dtype=np.float32)
        # subsample for speed
        if len(road_pts) > 20000:
            sel = self.rng.choice(len(road_pts), 20000, replace=False)
            road_pts = road_pts[sel]

        n, off = self._fit_plane_ransac(road_pts)
        dist = np.abs(P.reshape(-1, 3) @ n + off).reshape(H, W)
        scale = np.median(np.abs(z[road_mask])) + 1e-6
        height_rel = (dist / scale).astype(np.float32)
        return height_rel


if __name__ == "__main__":
    # Smoke test: needs a road mask; approximate with "bottom-center trapezoid"
    # just to verify depth + plane fit run and produce a sane height map.
    import sys

    from pathlib import Path

    laf = Path("/Volumes/BIggen/AV/data/lost_and_found")
    cands = list(laf.rglob("*_leftImg8bit.png"))
    if not cands:
        print("No Lost & Found image; pass a path.")
        sys.exit(1)
    img = Image.open(cands[0]).convert("RGB").resize((1200, 675))
    W, H = img.size
    # crude road prior for the smoke test only
    road = np.zeros((H, W), dtype=bool)
    for r in range(int(H * 0.55), H):
        half = int((r - H * 0.55) / (H * 0.45) * W * 0.45)
        road[r, W // 2 - half : W // 2 + half] = True

    gp = GroundPlaneHeight()
    hm = gp.height_map(img, road)
    print(f"height map {hm.shape} min={hm.min():.3f} max={hm.max():.3f} "
          f"median_road={np.median(hm[road]):.3f} (should be ~0 on road)")
    out = Path("/Volumes/BIggen/AV/results/depth_height_sample.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    vis = np.clip(hm / (np.percentile(hm, 99) + 1e-6) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(vis).save(out)
    print(f"saved {out} -- road should be dark (~0 height), protruding things bright")
