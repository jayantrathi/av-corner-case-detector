"""Train the autoencoder baseline on normal driving frames.

Phase 1: reconstruction-based anomaly detection.
Train on normal frames, compute error on both normal and anomalous frames.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.nuscenes_loader import load_nuscenes_frames
from src.data.splits import split_by_scene
from src.models.autoencoder import ConvAutoencoder
from src.eval.metrics import auroc, aupr


class FrameDataset(Dataset):
    """Simple dataset: load frames, normalize to [0, 1]."""

    def __init__(self, frames, transform=None):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        img = Image.open(frame.path).convert('RGB')

        # Resize to (900, 1600) for the autoencoder
        img = img.resize((1600, 900), Image.Resampling.LANCZOS)

        # To tensor: (H, W, 3) -> (3, H, W), normalize to [0, 1]
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        if self.transform:
            img = self.transform(img)

        return img


def train_epoch(model, loader, optimizer, device, epoch, num_epochs):
    """Train one epoch."""
    model.train()
    total_loss = 0

    for batch_idx, images in enumerate(loader):
        images = images.to(device)

        # Forward: reconstruction
        recon, _ = model(images)
        loss = nn.MSELoss()(recon, images)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 5 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(loader)}] "
                f"Loss: {loss.item():.6f}"
            )

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Avg Loss: {avg_loss:.6f}\n")
    return avg_loss


@torch.no_grad()
def compute_anomaly_scores(model, loader, device):
    """Compute anomaly scores for all frames in loader."""
    model.eval()
    scores = []

    for images in loader:
        images = images.to(device)
        score = model.anomaly_score(images)
        scores.extend(score.cpu().numpy())

    return np.array(scores)


def main():
    parser = argparse.ArgumentParser(description="Train autoencoder baseline")
    parser.add_argument(
        "--data-root",
        default="/Volumes/BIggen/AV/data/nuscenes",
        help="Path to nuScenes dataset",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--latent-dim", type=int, default=128, help="Latent dimension"
    )
    parser.add_argument(
        "--output-dir",
        default="/Volumes/BIggen/AV/results",
        help="Output directory for model and results",
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
    print(f"Train: {len(train_frames)} frames, Val: {len(val_frames)} frames\n")

    # Dataset and loader
    train_dataset = FrameDataset(train_frames)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    val_dataset = FrameDataset(val_frames)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    train_losses = []
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        train_losses.append(loss)

    # Save model
    model_path = output_dir / "autoencoder_phase1.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}\n")

    # Evaluate on train and val
    print("Computing anomaly scores on train and val sets...")
    train_scores = compute_anomaly_scores(model, train_loader, device)
    val_scores = compute_anomaly_scores(model, val_loader, device)

    print(f"Train scores: mean={train_scores.mean():.6f}, std={train_scores.std():.6f}")
    print(f"Val scores:   mean={val_scores.mean():.6f}, std={val_scores.std():.6f}\n")

    # Save results
    results = {
        "train_losses": train_losses,
        "train_scores": train_scores.tolist(),
        "val_scores": val_scores.tolist(),
    }

    import json
    results_path = output_dir / "phase1_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}\n")

    # Plot training loss
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training Loss")
    ax.legend()
    ax.grid()
    fig.savefig(output_dir / "phase1_training_loss.png", dpi=100, bbox_inches="tight")
    print(f"Training loss plot saved\n")

    # Plot anomaly score distributions
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(train_scores, bins=20, alpha=0.6, label="Train (Normal)")
    ax.hist(val_scores, bins=20, alpha=0.6, label="Val (Normal)")
    ax.set_xlabel("Anomaly Score (Mean Reconstruction Error)")
    ax.set_ylabel("Count")
    ax.set_title("Phase 1: Anomaly Score Distribution (All Normal Frames)")
    ax.legend()
    ax.grid()
    fig.savefig(output_dir / "phase1_score_distribution.png", dpi=100, bbox_inches="tight")
    print(f"Score distribution plot saved\n")

    print("Phase 1 training complete!")
    print(f"Next: Phase 2 — embedding-based detector (primary method)\n")


if __name__ == "__main__":
    main()
