"""Autoencoder training v2: 50+ epochs for better convergence.

Phase 1 took only 10 epochs. This script trains longer to let the model
fully capture normal frame patterns on nuScenes.
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
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.autoencoder import ConvAutoencoder
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
        return img


def main():
    parser = argparse.ArgumentParser(description="Train autoencoder for 50+ epochs")
    parser.add_argument(
        "--data-root",
        default="/Volumes/BIggen/AV/data/nuscenes/v1.0-mini",
        help="Path to nuScenes dataset",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
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
    print("Initializing ConvAutoencoder...")
    model = ConvAutoencoder(latent_dim=128).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    train_losses = []
    val_losses = []

    print(f"\nTraining for {args.epochs} epochs with lr={args.lr}...\n")

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)

            # Forward pass
            recon, _ = model(images)
            loss = criterion(recon, images)

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
                recon, _ = model(images)
                loss = criterion(recon, images)
                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= n_val_batches
        val_losses.append(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{args.epochs}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    print("\nTraining complete!")

    # Save model
    model_path = output_dir / "autoencoder_phase1_v2.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}\n")

    # Plot training curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label="Train Loss", linewidth=2)
    ax.plot(val_losses, label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Autoencoder Training (50 epochs, lr={args.lr})")
    ax.legend()
    ax.grid()

    plot_path = output_dir / "autoencoder_training_curves_v2.png"
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
