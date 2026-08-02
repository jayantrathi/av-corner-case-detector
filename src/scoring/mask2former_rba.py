"""RbA ("Rejected by All") anomaly scoring on top of a frozen, pretrained
Mask2Former model fine-tuned on Cityscapes.

This replaces the ResNet50 + k-NN patch-distance approach used earlier in
this project. That approach measured "how visually/texturally unusual is
this patch compared to a reference bank" -- which can't distinguish
"statistically rare but harmless" (manhole covers, traffic cones, bollards)
from "genuinely novel and dangerous," because it never reasons about what
the object actually IS, only how its raw pixels compare to a stored sample.

RbA instead asks a semantically grounded question: "is there ANY known road
class (road, car, person, sidewalk, vegetation, sign, ...) that confidently
claims this pixel?" A manhole cover looks like road to a Cityscapes-trained
segmentation model (it IS road, structurally) -- high confidence, not
anomalous. A cone is common enough in real driving footage that models
trained on it usually handle it too. Something with no resemblance to any
of Cityscapes' ~19 classes -- a fallen cargo box, a stray animal, an
overturned vehicle -- gets low confidence from every single class query
simultaneously. That gap is what RbA measures.

Formula, exactly as defined in the paper (Nayal et al., "RbA: Segmenting
Unknown Regions Rejected by All", ICCV 2023, arXiv:2211.14293, eq. 1-5):

  M_n(x)   = sigmoid(mask_logit_n(x))          -- how much query n claims pixel x
  P_n(k)   = softmax(class_logits_n)[k]         -- query n's confidence in class k
  L_k(x)   = sum_n  P_n(k) * M_n(x)             -- aggregated per-class logit at x
  p(y=k|x) = sigmoid(L_k(x))                    -- reinterpreted as an independent
                                                    one-vs-all binary probability
  RbA(x)   = - sum_k  p(y=k|x)                  -- outlier score (HIGHER = more anomalous)

A pixel confidently claimed by exactly one class has p(y=k|x) close to 1 for
that class and near 0 for the rest, so RbA(x) is close to -1 (strongly
inlier). A pixel that every class rejects has all p(y=k|x) near 0, so
RbA(x) is close to 0 -- the least negative, i.e. the most anomalous, value
the score can take.

No fitting/reference-bank step, unlike the k-NN approach -- this is a frozen,
zero-shot forward pass. The model's Cityscapes training IS the "what does
normal driving look like" step; there's nothing left to fit on our end.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CHECKPOINT = "facebook/mask2former-swin-tiny-cityscapes-semantic"


class RbAScorer:
    def __init__(self, device: str = "cpu", checkpoint: str = CHECKPOINT):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint).to(device)
        self.model.eval()
        # Cityscapes id2label from the model config -- used for the debug
        # semantic-map render, not for scoring itself.
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def score(self, image: Image.Image, out_size: tuple[int, int] | None = None):
        """Returns (rba_map, semantic_map) as numpy arrays.

        rba_map: (H, W) float32, higher = more anomalous (see module docstring
            for exact formula). Resized to out_size (W, H) if given, else the
            processor's native output resolution.
        semantic_map: (H, W) int, the model's own predicted class id per
            pixel (standard Mask2Former argmax semantic segmentation) --
            purely for sanity-check visualization, not used in scoring.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        class_queries_logits = outputs.class_queries_logits  # (1, N, K+1)
        masks_queries_logits = outputs.masks_queries_logits  # (1, N, h, w)

        target_h, target_w = masks_queries_logits.shape[-2:]
        if out_size is not None:
            target_w, target_h = out_size

        # NOTE: this was originally written to fix a suspected zero-padding
        # artifact (HF's general docs say the processor resizes preserving
        # aspect ratio then pads to a multiple of size_divisor=32). Checked
        # directly with debug_rba_padding.py against real Lost & Found
        # images: pixel_mask reports 0px padding on bottom and right --
        # this checkpoint's config resizes straight to a fixed 384x384
        # square, no padding at all. So this crop is a no-op for this
        # checkpoint (confirmed: identical rba_map with/without it) and
        # was NOT the cause of the top-of-frame artifact seen in the Lost &
        # Found examples. Left in place (harmless, and would matter for a
        # checkpoint/config that DOES pad), but the real explanation for
        # the edge artifact is a boundary/receptive-field effect in the
        # frozen model itself -- pixels near the image edge have truncated
        # context, a generic property of dense prediction near borders,
        # not a bug in this pipeline. Handled via eligibility margins in
        # evaluate_rba_lost_and_found.py instead of "fixed" here, because
        # there's nothing here to fix.
        pixel_mask = inputs.get("pixel_mask")
        if pixel_mask is not None:
            valid_h = int(pixel_mask[0, :, 0].sum().item())
            valid_w = int(pixel_mask[0, 0, :].sum().item())
            pix_h, pix_w = pixel_mask.shape[-2:]
            mask_h, mask_w = masks_queries_logits.shape[-2:]
            # masks_queries_logits is a downsampled feature map of
            # pixel_values at whatever stride the pixel decoder uses --
            # rather than assume the exact stride, scale the valid extent
            # by the same fraction it occupies in pixel_values space.
            valid_mask_h = max(1, round(valid_h / pix_h * mask_h))
            valid_mask_w = max(1, round(valid_w / pix_w * mask_w))
            masks_queries_logits = masks_queries_logits[:, :, :valid_mask_h, :valid_mask_w]

        mask_probs = F.interpolate(
            masks_queries_logits, size=(target_h, target_w), mode="bilinear", align_corners=False
        ).sigmoid()  # (1, N, H, W) = M_n(x)

        # drop the last "no object" class before softmax-ing over real classes
        class_probs = class_queries_logits.softmax(dim=-1)[..., :-1]  # (1, N, K) = P_n(k)

        # L_k(x) = sum_n P_n(k) * M_n(x)
        L = torch.einsum("bnk,bnhw->bkhw", class_probs, mask_probs)  # (1, K, H, W)

        p_inlier = L.sigmoid()  # p(y=k|x)
        rba_map = -p_inlier.sum(dim=1).squeeze(0)  # (H, W), higher = more anomalous

        semantic_map = p_inlier.argmax(dim=1).squeeze(0)  # standard argmax-over-classes prediction

        return rba_map.cpu().numpy(), semantic_map.cpu().numpy()
