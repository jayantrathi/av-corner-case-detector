"""We've guessed at margin sizes twice now (top 8%, bottom 15%) and both
times the hit rate stayed at exactly 0/30 -- meaning the guesses weren't
big enough, or missed a dimension (left/right) entirely. Instead of a
third guess, measure directly: score all 30 held-out hazard frames, find
where the (margin-restricted) peak actually lands as a fraction of frame
height/width, and print the distribution. If there's a real cluster (e.g.
consistently ~78-84% down the frame for the hood, or consistently near
x=0 for a left-edge artifact), that tells us exactly what margin would
actually cover it -- instead of another round of screenshot-guessing.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate_patch_localization import CANVAS_SIZE
from evaluate_rba_lost_and_found import load_test_split, eligible_region_mask, REGION_TOP_PERCENTILE
from src.scoring.mask2former_rba import RbAScorer


def main():
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    scorer = RbAScorer(device=device)

    _, test_hazard = load_test_split()
    print(f"Scoring {len(test_hazard)} held-out hazard frames, logging raw peak position "
          f"(no eligibility restriction) and margin-restricted peak position for each.\n")

    eligible = None
    raw_positions, restricted_positions = [], []
    for path, scene, _ in test_hazard:
        img = Image.open(path).convert("RGB")
        rba_map, _ = scorer.score(img, out_size=CANVAS_SIZE)
        h, w = rba_map.shape
        if eligible is None:
            eligible = eligible_region_mask((h, w))

        raw_peak = np.unravel_index(np.argmax(rba_map), rba_map.shape)
        masked = np.where(eligible, rba_map, -np.inf)
        restricted_peak = np.unravel_index(np.argmax(masked), rba_map.shape)

        raw_frac = (raw_peak[0] / h, raw_peak[1] / w)
        restricted_frac = (restricted_peak[0] / h, restricted_peak[1] / w)
        raw_positions.append(raw_frac)
        restricted_positions.append(restricted_frac)
        print(f"  {Path(path).name[:45]:45s}  raw=({raw_frac[0]:.2f},{raw_frac[1]:.2f})  "
              f"restricted=({restricted_frac[0]:.2f},{restricted_frac[1]:.2f})")

    raw_arr = np.array(raw_positions)
    restricted_arr = np.array(restricted_positions)
    print(f"\nRAW peak position (row_frac, col_frac) -- median: {np.median(raw_arr, axis=0)}  "
          f"p10-p90 row: [{np.percentile(raw_arr[:,0],10):.2f}, {np.percentile(raw_arr[:,0],90):.2f}]  "
          f"p10-p90 col: [{np.percentile(raw_arr[:,1],10):.2f}, {np.percentile(raw_arr[:,1],90):.2f}]")
    print(f"RESTRICTED peak position -- median: {np.median(restricted_arr, axis=0)}  "
          f"p10-p90 row: [{np.percentile(restricted_arr[:,0],10):.2f}, {np.percentile(restricted_arr[:,0],90):.2f}]  "
          f"p10-p90 col: [{np.percentile(restricted_arr[:,1],10):.2f}, {np.percentile(restricted_arr[:,1],90):.2f}]")
    print("\nIf RESTRICTED positions still cluster tightly (e.g. row_frac around 0.75-0.85, "
          "or col_frac near 0 or 1), that's the real, evidence-based margin to set -- not another guess.")


if __name__ == "__main__":
    main()
