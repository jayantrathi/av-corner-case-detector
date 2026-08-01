"""Extract embeddings from a frozen pretrained ResNet50.

Phase 2: embedding-based anomaly detection. Extract features once, then
iterate on scoring without re-running the backbone.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torchvision.models as models

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.nuscenes_loader import load_nuscenes_frames
from src.data.splits import split_by_scene


class FrameDataset:
    """Simple dataset wrapper for frames."""

    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        img = Image.open(frame.path).convert('RGB')
        img = img.resize((1600, 900), Image.Resampling.LANCZOS)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        # Normalize to ImageNet stats
        img[0] = (img[0] - 0.485) / 0.229
        img[1] = (img[1] - 0.456) / 0.224
        img[2] = (img[2] - 0.406) / 0.225

        return img


@torch.no_grad()
def extract_embeddings(model, loader, device, num_batches=None):
    """Extract embeddings for all frames in loader."""
    model.eval()
    embeddings = []

    for batch_idx, images in enumerate(loader):
        if num_batches and batch_idx >= num_batches:
            break

        images = images.to(device)
        feats = model(images)
        embeddings.extend(feats.cpu().numpy())

        if (batch_idx + 1) % 5 == 0:
            print(f"Processed {(batch_idx + 1) * loader.batch_size} frames...")

    return np.array(embeddings)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from frozen ResNet50")
    parser.add_argument(
        "--data-root",
        default="/Volumes/BIggen/AV/data/nuscenes/v1.0-mini",
        help="Path to nuScenes dataset",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size"
    )
    parser.add_argument(
        "--output-dir",
        default="/Volumes/BIggen/AV/results",
        help="Output directory for embeddings",
    )
    args = parser.parse_args()

    # Setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading nuScenes frames...")
    frames = load_nuscenes_frames(args.data_root)
    splits = split_by_scene(frames, val_frac=0.15, test_frac=0.15, seed=0)
    train_frames = splits["train"]
    val_frames = splits["val"]
    test_frames = splits["test"]
    print(
        f"Train: {len(train_frames)}, Val: {len(val_frames)}, Test: {len(test_frames)}\n"
    )

    # Load pretrained ResNet50
    print("Loading pretrained ResNet50...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Remove final classification layer, keep up to avgpool
    model = nn.Sequential(*list(resnet.children())[:-1])  # Remove fc
    model = model.to(device)
    print(f"Embedding dimension: 2048\n")

    # Extract embeddings for each split
    for split_name, split_frames in [
        ("train", train_frames),
        ("val", val_frames),
        ("test", test_frames),
    ]:
        print(f"Extracting embeddings for {split_name}...")
        dataset = FrameDataset(split_frames)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        embeddings = extract_embeddings(model, loader, device)
        embeddings = embeddings.squeeze()  # Remove batch dimension from avgpool

        # Save
        save_path = output_dir / f"phase2_embeddings_{split_name}.npy"
        np.save(save_path, embeddings)
        print(f"Saved {embeddings.shape} embeddings to {save_path}\n")

    print("Embedding extraction complete!")


if __name__ == "__main__":
    main()
