"""Convolutional autoencoder for reconstruction-based anomaly detection.

Simple, interpretable baseline: train on normal frames, high reconstruction
error on anomalies. The error heatmap shows where the anomaly is.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Lightweight conv autoencoder for 1600x900 RGB images."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: compress 1600x900x3 -> latent_dim
        self.encoder = nn.Sequential(
            # Input: (B, 3, 900, 1600)
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 450, 800)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 225, 400)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 112, 200)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (B, 256, 56, 100)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 256, 1, 1)
            nn.Flatten(),  # (B, 256)
        )

        # Bottleneck projection
        self.bottleneck = nn.Linear(256, latent_dim)

        # Decoder: expand latent_dim back to 1600x900x3
        self.decoder_fc = nn.Linear(latent_dim, 256 * 56 * 100)

        self.decoder = nn.Sequential(
            # Input: (B, 256, 56, 100)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 112, 200)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 224, 400)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 448, 800)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # (B, 3, 896, 1600)
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent vector."""
        return self.bottleneck(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to image."""
        x = self.decoder_fc(z)
        x = x.view(x.size(0), 256, 56, 100)
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode then decode. Returns reconstruction and latent."""
        z = self.encode(x)
        recon = self.decode(z)

        # Pad reconstruction to match input size if needed
        # Input is (B, 3, 900, 1600), decoder output might be (B, 3, 896, 1600)
        if recon.shape[2] != x.shape[2] or recon.shape[3] != x.shape[3]:
            pad_h = x.shape[2] - recon.shape[2]
            pad_w = x.shape[3] - recon.shape[3]
            recon = torch.nn.functional.pad(recon, (0, pad_w, 0, pad_h), mode='constant', value=0)

        return recon, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-pixel L2 reconstruction error.

        Args:
            x: Input image (B, 3, H, W) normalized to [0, 1]

        Returns:
            Error map (B, 1, H, W) with per-pixel L2 distance
        """
        recon, _ = self(x)
        # Per-pixel L2 error, averaged across channels
        error = torch.sqrt(((x - recon) ** 2).mean(dim=1, keepdim=True) + 1e-6)
        return error

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute scalar anomaly score per image.

        Args:
            x: Input image (B, 3, H, W)

        Returns:
            Scalar score (B,) — mean reconstruction error
        """
        error = self.reconstruction_error(x)
        return error.view(error.size(0), -1).mean(dim=1)
