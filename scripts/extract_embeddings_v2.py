"""Extract embeddings from the fine-tuned ResNet50 backbone.

After running finetune_resnet_backbone.py, use this to extract embeddings
on the training and evaluation sets with the improved model.
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
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.nuscenes_loader import load_nuscenes_frames
from src.data.splits import split_by_scene


def load_coda_frames(coda_path: str) -> list[str]:
    """Load CODA corner-case image paths."""
    coda_dir = Path(coda_path) / 'CODA' / 'sample' / 'images'
    return sorted([str(f) for f in coda_dir.glob('*.jpg')])


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


class ImagePathDataset:
    """Dataset wrapper for raw image file paths."""

    def __init__(self, image_paths):
        self.paths = image_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((1600, 900), Image.Resampling.LANCZOS)
            img = np.array(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)

            # Normalize to ImageNet stats
            img[0] = (img[0] - 0.485) / 0.229
            img[1] = (img[1] - 0.456) / 0.224
            img[2] = (img[2] - 0.406) / 0.225

            return img
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return zeros if image fails to load
            return torch.zeros(3, 900, 1600)


class ResNetEmbedder(nn.Module):
    """ResNet50 for embedding extraction."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        """Extract 2048-dimensional embeddings."""
        feat = self.backbone(x)
        return feat.squeeze(-1).squeeze(-1)


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

        # feats should be (B, 2048) after the model forward pass
        # Make sure it's flattened to 2D
        feats = feats.view(feats.size(0), -1)  # Flatten any extra dims

        embeddings.extend(feats.cpu().numpy())

        if (batch_idx + 1) % 5 == 0:
            print(f"Processed {(batch_idx + 1) * loader.batch_size} frames...")

    embeddings_arr = np.array(embeddings)
    print(f"  Extracted shape: {embeddings_arr.shape}")
    return embeddings_arr


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from fine-tuned ResNet50")
    parser.add_argument(
        "--data-root",
        default="/Volumes/BIggen/AV/data/nuscenes/v1.0-mini",
        help="Path to nuScenes dataset",
    )
    parser.add_argument(
        "--model-path",
        default="/Volumes/BIggen/AV/results/resnet50_backbone_finetuned.pt",
        help="Path to fine-tuned model checkpoint",
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

    # Load fine-tuned model
    print("Loading fine-tuned ResNet50...")
    model = ResNetEmbedder().to(device)

    # Load checkpoint
    if Path(args.model_path).exists():
        state_dict = torch.load(args.model_path, map_location=device)
        # The checkpoint might only have backbone keys, so load carefully
        try:
            model.load_state_dict(state_dict)
            print(f"Loaded fine-tuned model from {args.model_path}\n")
        except RuntimeError:
            # Try loading just the backbone part
            backbone_state_dict = {
                k.replace("backbone.", ""): v
                for k, v in state_dict.items()
                if k.startswith("backbone.")
            }
            model.backbone.load_state_dict(backbone_state_dict)
            print(f"Loaded fine-tuned backbone from {args.model_path}\n")
    else:
        print(f"Warning: Model not found at {args.model_path}, using pretrained only\n")

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

        # Save with v2 suffix to distinguish from original frozen embeddings
        save_path = output_dir / f"phase2_embeddings_{split_name}_v2.npy"
        np.save(save_path, embeddings)
        print(f"Saved {embeddings.shape} embeddings to {save_path}\n")

    # Also extract embeddings for evaluation set: test + CODA
    print("=" * 60)
    print("Extracting embeddings for evaluation set (test + CODA)...")
    print("=" * 60)

    coda_path = "/Volumes/BIggen/AV/data/coda"
    coda_paths = load_coda_frames(coda_path)
    print(f"Found {len(coda_paths)} CODA corner-case images\n")

    # Combine test frames + CODA frames
    test_emb = np.load(output_dir / "phase2_embeddings_test_v2.npy")
    print(f"Test embeddings: {test_emb.shape}")

    # Extract CODA embeddings
    print("Extracting CODA embeddings...")
    coda_dataset = ImagePathDataset(coda_paths)
    coda_loader = DataLoader(coda_dataset, batch_size=args.batch_size, shuffle=False)
    coda_emb = extract_embeddings(model, coda_loader, device)
    print(f"CODA embeddings: {coda_emb.shape}\n")

    # Combine and save as evaluation set
    eval_emb = np.vstack([test_emb, coda_emb])
    eval_save_path = output_dir / "phase3_embeddings_eval_v2.npy"
    np.save(eval_save_path, eval_emb)
    print(f"Saved combined evaluation set {eval_emb.shape} to {eval_save_path}\n")

    print("=" * 60)
    print("Embedding extraction complete!")


if __name__ == "__main__":
    main()
