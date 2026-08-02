"""Quick diagnostic: print the actual distribution of RbA scores for one
frame, before we guess at what's wrong. The CODA demo run produced a
suspicious whole-image "detection" on every single frame -- either the
score map itself has collapsed (near-constant, no real contrast between
confidently-normal pixels and genuinely uncertain ones), or the region-
growing threshold logic in largest_region_from_peak has a bug. This tells
us which."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate_patch_localization import CANVAS_SIZE
from src.scoring.mask2former_rba import RbAScorer

CODA_ROOT = Path("/Volumes/BIggen/AV/data/coda/CODA/sample")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    scorer = RbAScorer(device=device)

    img_path = CODA_ROOT / "images" / "000002_1616005519499.jpg"
    img = Image.open(img_path).convert("RGB")
    rba_map, semantic_map = scorer.score(img, out_size=CANVAS_SIZE)

    print(f"Image: {img_path.name}")
    print(f"rba_map shape: {rba_map.shape}, dtype: {rba_map.dtype}")
    print(f"  min={rba_map.min():.6f}  max={rba_map.max():.6f}  mean={rba_map.mean():.6f}  std={rba_map.std():.6f}")
    print(f"  percentiles: 1%={np.percentile(rba_map,1):.6f}  25%={np.percentile(rba_map,25):.6f}  "
          f"50%={np.percentile(rba_map,50):.6f}  75%={np.percentile(rba_map,75):.6f}  "
          f"99%={np.percentile(rba_map,99):.6f}  99.9%={np.percentile(rba_map,99.9):.6f}")

    peak_val = rba_map.max()
    print(f"\npeak_val = {peak_val:.6f}")
    for frac in [0.99, 0.95, 0.9, 0.7, 0.5, 0.3]:
        threshold = peak_val * frac if peak_val > 0 else peak_val / frac
        n_above = int((rba_map >= threshold).sum())
        pct = 100 * n_above / rba_map.size
        print(f"  threshold_frac={frac}: threshold={threshold:.6f}  pixels_above={n_above} ({pct:.2f}% of image)")

    print(f"\nsemantic_map unique classes predicted: {sorted(set(semantic_map.flatten().tolist()))}")
    print(f"id2label sample: {dict(list(scorer.id2label.items())[:5])}")

    # save the raw heatmap, unnormalized-relative-to-itself, so we can SEE
    # whether there's real spatial structure
    from PIL import Image as PILImage
    norm = (rba_map - rba_map.min()) / (rba_map.max() - rba_map.min() + 1e-8)
    heat = PILImage.fromarray((norm * 255).astype(np.uint8))
    out_path = Path("/Volumes/BIggen/AV/results/rba_debug_heatmap.png")
    heat.save(out_path)
    print(f"\nSaved raw heatmap (no threshold, no box, just the normalized score) to {out_path}")


if __name__ == "__main__":
    main()
