"""Spatial (patch-level) feature extraction from a frozen ResNet50 backbone.

Unlike the whole-frame embedder (global avgpool -> one 2048-d vector per
image), this keeps the spatial feature map so each grid cell becomes its own
"patch descriptor" from a SINGLE forward pass -- not 50-100 separate
crop-and-resize passes through the network per image, which would be both
much slower and exactly the kind of naive approach that got us into timing
trouble before.

Layer choice: layer3, not layer4. By layer4, ResNet50's effective receptive
field is large enough that a single grid cell already "sees" most of a
900x1600 frame -- a hazard in one corner would still shift the score of a
patch on the opposite side of the image, which defeats the point of
patch-level localization. layer3 has a smaller receptive field and finer
spatial resolution (stride 16 vs stride 32) while still being deep enough to
carry real semantic content. This is the same intermediate-depth choice used
by standard patch-based anomaly detection methods (PaDiM, SPADE) for the
same reason.

Always the frozen ImageNet backbone -- the Lost & Found control proved the
nuScenes-fine-tuned backbone doesn't generalize (AUROC 0.49, below random),
so there is no reason to build localization on top of it.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as models


class PatchResNetEmbedder(nn.Module):
    """Frozen ImageNet ResNet50, truncated after layer3. A forward pass on a
    (B, 3, H, W) batch returns (B, 1024, Hf, Wf) spatial features, roughly
    one 1024-d descriptor per 16x16 input region (true receptive field is
    larger than 16x16, but the stride -- and therefore localization
    granularity -- is 16px)."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3,
        )
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


def feature_map_to_patches(feat: torch.Tensor) -> torch.Tensor:
    """(B, C, Hf, Wf) -> (B*Hf*Wf, C), row-major over (row, col) per image,
    images concatenated in batch order. Inverse indexing: patch index
    b*Hf*Wf + i*Wf + j corresponds to image b, grid row i, grid col j."""
    b, c, h, w = feat.shape
    return feat.permute(0, 2, 3, 1).reshape(b * h * w, c)


def grid_cell_bbox(i: int, j: int, grid_h: int, grid_w: int, img_h: int, img_w: int) -> tuple[int, int, int, int]:
    """Pixel-space bbox (x0, y0, x1, y1) covered by grid cell (row i, col j),
    in the same pixel coordinates as the image the feature map was computed
    from (i.e. AFTER any resize applied before the model, e.g. 1600x900)."""
    y0 = int(round(i * img_h / grid_h))
    y1 = int(round((i + 1) * img_h / grid_h))
    x0 = int(round(j * img_w / grid_w))
    x1 = int(round((j + 1) * img_w / grid_w))
    return x0, y0, x1, y1
