"""Verify, don't assume: check whether pixel_mask is actually present and
whether the padding-crop fix in mask2former_rba.py is doing anything
measurable, before trusting (or re-guessing about) the top-edge artifact.

Directly inspects the processor's own output (pixel_mask shape/values) and
compares the top strip of rba_map against the middle of the frame, with and
without the crop applied -- numbers, not just "does the image look red."
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate_patch_localization import CANVAS_SIZE
from src.scoring.mask2former_rba import RbAScorer

CODA_ROOT = Path("/Volumes/BIggen/AV/data/coda/CODA/sample")
LAF_ROOT = Path("/Volumes/BIggen/AV/data/lost_and_found")


@torch.no_grad()
def score_with_optional_crop(scorer, image, out_size, apply_crop: bool):
    """Reimplements RbAScorer.score()'s body so we can toggle the crop on
    and off for an apples-to-apples comparison on the SAME forward pass."""
    inputs = scorer.processor(images=image, return_tensors="pt").to(scorer.device)
    outputs = scorer.model(**inputs)

    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits
    target_h, target_w = out_size[1], out_size[0]

    pixel_mask = inputs.get("pixel_mask")
    info = {"pixel_mask_present": pixel_mask is not None}
    if pixel_mask is not None:
        valid_h = int(pixel_mask[0, :, 0].sum().item())
        valid_w = int(pixel_mask[0, 0, :].sum().item())
        pix_h, pix_w = pixel_mask.shape[-2:]
        mask_h, mask_w = masks_queries_logits.shape[-2:]
        info.update(pix_h=pix_h, pix_w=pix_w, valid_h=valid_h, valid_w=valid_w,
                    mask_h=mask_h, mask_w=mask_w,
                    pad_h_px=pix_h - valid_h, pad_w_px=pix_w - valid_w)
        if apply_crop:
            valid_mask_h = max(1, round(valid_h / pix_h * mask_h))
            valid_mask_w = max(1, round(valid_w / pix_w * mask_w))
            info.update(valid_mask_h=valid_mask_h, valid_mask_w=valid_mask_w)
            masks_queries_logits = masks_queries_logits[:, :, :valid_mask_h, :valid_mask_w]

    mask_probs = F.interpolate(
        masks_queries_logits, size=(target_h, target_w), mode="bilinear", align_corners=False
    ).sigmoid()
    class_probs = class_queries_logits.softmax(dim=-1)[..., :-1]
    L = torch.einsum("bnk,bnhw->bkhw", class_probs, mask_probs)
    p_inlier = L.sigmoid()
    rba_map = -p_inlier.sum(dim=1).squeeze(0)
    return rba_map.cpu().numpy(), info


def report(name, rba_map):
    h, w = rba_map.shape
    top_strip = rba_map[: int(h * 0.08), :]
    mid_band = rba_map[int(h * 0.35): int(h * 0.65), :]
    print(f"  [{name}] top-8%-strip: mean={top_strip.mean():.4f} max={top_strip.max():.4f}  |  "
          f"middle-30%-band: mean={mid_band.mean():.4f} max={mid_band.max():.4f}")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    scorer = RbAScorer(device=device)

    img_root = LAF_ROOT / "leftImg8bit" / "leftImg8bit"
    if not img_root.exists():
        img_root = LAF_ROOT / "leftImg8bit"
    laf_images = sorted(img_root.rglob("*_leftImg8bit.png"))[:3]

    for img_path in laf_images:
        print(f"\n{'='*70}\n{img_path.name}\n{'='*70}")
        img = Image.open(img_path).convert("RGB")

        rba_no_crop, info = score_with_optional_crop(scorer, img, CANVAS_SIZE, apply_crop=False)
        rba_crop, _ = score_with_optional_crop(scorer, img, CANVAS_SIZE, apply_crop=True)

        print(f"  pixel_mask_present={info['pixel_mask_present']}")
        if info["pixel_mask_present"]:
            print(f"  pixel_values (processor space): {info['pix_h']}x{info['pix_w']}  "
                  f"valid: {info['valid_h']}x{info['valid_w']}  "
                  f"pad: {info['pad_h_px']}px bottom, {info['pad_w_px']}px right")
            print(f"  masks_queries_logits: {info['mask_h']}x{info['mask_w']}  "
                  f"-> cropped to {info.get('valid_mask_h')}x{info.get('valid_mask_w')}")
        report("no crop ", rba_no_crop)
        report("w/ crop ", rba_crop)
        top_change = (rba_crop[:int(rba_crop.shape[0] * 0.08)].mean()
                      - rba_no_crop[:int(rba_no_crop.shape[0] * 0.08)].mean())
        print(f"  top-strip mean CHANGE from crop: {top_change:.4f}")


if __name__ == "__main__":
    main()
