"""ResNet50 backbone fine-tuning for embedding-based detection.

Phase 2 used a frozen ResNet50 from ImageNet. Here we unfreeze the last few
layers of the backbone and train on nuScenes normal frames to adapt the
embedding space to the driving domain.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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


class ResNetEmbedder(nn.Module):
    """ResNet50 for embedding extraction with fine-tunable backbone."""

    def __init__(self):
        super().__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Extract layers up to avgpool (2048 embedding dim)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        """Extract 2048-dimensional embeddings."""
        feat = self.backbone(x)
        return feat.squeeze(-1).squeeze(-1)  # Remove spatial dims


def contrastive_loss(embeddings, labels, margin=1.0):
    """Compute simple contrastive loss: push normal embeddings closer, anomalies farther.

    For this training phase, all samples are normal, so we use an intra-class
    contrastive loss that pulls embeddings together.
    """
    # Compute pairwise distances
    dists = torch.cdist(embeddings, embeddings)

    # Pull same-class pairs closer (but all are class 0 during training)
    # Use a simple center-loss variant: minimize variance of embeddings
    center = embeddings.mean(dim=0, keepdim=True)
    intra_loss = ((embeddings - center) ** 2).sum(dim=1).mean()

    return intra_loss


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 backbone on nuScenes")
    parser.add_argument(
        "--data-root",
        default="/Volumes/BIggen/AV/data/nuscenes/v1.0-mini",
        help="Path to nuScenes dataset",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (lower for fine-tuning)"
    )
    parser.add_argument(
        "--freeze-until", type=int, default=7,
        help="Freeze layers 0-N, fine-tune layer N onwards (ResNet50 has 8 layers)"
    )
    parser.add_argument(
        "--output-dir",
        default="/Volumes/BIggen/AV/results",
        help="Output directory for model",
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

    print(f"Train frames: {len(train_frames)}")
    print(f"Val frames: {len(val_frames)}\n")

    # Create datasets and loaders
    train_dataset = FrameDataset(train_frames)
    val_dataset = FrameDataset(val_frames)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    print("Loading ResNet50 with pretrained ImageNet weights...")
    model = ResNetEmbedder().to(device)

    # Freeze early layers, fine-tune late layers
    print(f"Freezing layers 0-{args.freeze_until}, fine-tuning layer {args.freeze_until} onwards...\n")
    for idx, layer in enumerate(model.backbone):
        if idx < args.freeze_until:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True

    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,}\n")

    # Loss and optimizer (only update trainable params)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # Training loop
    train_losses = []
    val_losses = []

    print(f"Training for {args.epochs} epochs with lr={args.lr}...\n")

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)

            # Forward pass
            embeddings = model(images)

            # Contrastive loss (pull normal embeddings closer)
            loss = contrastive_loss(embeddings, labels=None, margin=1.0)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                embeddings = model(images)
                loss = contrastive_loss(embeddings, labels=None, margin=1.0)
                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= n_val_batches
        val_losses.append(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{args.epochs}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    print("\nTraining complete!")

    # Save model
    model_path = output_dir / "resnet50_backbone_finetuned.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}\n")

    # Plot training curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label="Train Loss", linewidth=2)
    ax.plot(val_losses, label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive Loss")
    ax.set_title(f"ResNet50 Fine-tuning (30 epochs, lr={args.lr})")
    ax.legend()
    ax.grid()

    plot_path = output_dir / "resnet50_training_curves.png"
    fig.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"Training curves saved to {plot_path}\n")

    # Final stats
    print("=" * 60)
    print("FINAL STATS")
    print("=" * 60)
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss:   {val_losses[-1]:.6f}")
    print(f"Best val loss:    {min(val_losses):.6f} (epoch {np.argmin(val_losses) + 1})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
