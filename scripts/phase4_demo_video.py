"""Phase 4: Demo video showing real-time anomaly detection.

Visualize both Phase 1 (autoencoder) and Phase 2 (k-NN + fine-tuned ResNet50)
detectors running on a sequence of frames. Show:
- Original frame
- Autoencoder reconstruction
- Anomaly scores from both methods
- Ground truth (normal vs anomaly)
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torchvision.models as models

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.autoencoder import ConvAutoencoder
from src.scoring.embedding_scorers import kNNScorer
from src.data.nuscenes_loader import load_nuscenes_frames
from src.data.splits import split_by_scene


class ResNetEmbedder(nn.Module):
    """ResNet50 for embedding extraction (fine-tuned version)."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        """Extract 2048-dimensional embeddings."""
        feat = self.backbone(x)
        return feat.squeeze(-1).squeeze(-1)


def load_image_pil(path: str) -> Image.Image:
    """Load image as PIL Image."""
    return Image.open(path).convert('RGB')


def load_image_as_tensor(path: str, device: str) -> torch.Tensor:
    """Load and normalize image to tensor."""
    img = Image.open(path).convert('RGB')
    img = img.resize((1600, 900), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    return img_tensor.to(device)


@torch.no_grad()
def get_autoencoder_reconstruction(model, img_tensor: torch.Tensor, device: str) -> Image.Image:
    """Get reconstruction and error map from autoencoder."""
    model.eval()
    recon, _ = model(img_tensor.unsqueeze(0).to(device))
    recon = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
    recon = np.clip(recon * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(recon)


@torch.no_grad()
def get_resnet_embedding(model, img_tensor: torch.Tensor, device: str) -> np.ndarray:
    """Extract embedding from ResNet50."""
    model.eval()
    emb = model(img_tensor.unsqueeze(0).to(device))
    return emb.cpu().numpy().squeeze()


def create_demo_frame(
    original_pil: Image.Image,
    reconstruction_pil: Image.Image,
    autoencoder_score: float,
    knn_score: float,
    label: int,
    frame_num: int,
) -> Image.Image:
    """Create a demo frame with visualizations."""
    # Resize images for display (smaller than original 1600x900)
    display_h, display_w = 450, 800
    original = original_pil.resize((display_w, display_h), Image.Resampling.LANCZOS)
    recon = reconstruction_pil.resize((display_w, display_h), Image.Resampling.LANCZOS)

    # Create canvas: 2 images side by side + info panel below
    canvas_w = display_w * 2 + 40  # 2 images + spacing
    canvas_h = display_h + 200  # images + info panel
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))

    # Paste images
    canvas.paste(original, (20, 20))
    canvas.paste(recon, (display_w + 20, 20))

    # Add text annotations
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()
        font_small = font

    # Labels
    draw.text((30, display_h + 30), "Original Frame", fill=(255, 255, 255), font=font)
    draw.text((display_w + 30, display_h + 30), "Autoencoder Reconstruction", fill=(255, 255, 255), font=font)

    # Scores
    y_pos = display_h + 65
    draw.text((30, y_pos), f"Frame {frame_num:04d}", fill=(200, 200, 200), font=font_small)

    # Autoencoder score
    ae_color = (255, 100, 100) if autoencoder_score > 0.5 else (100, 200, 100)
    draw.text((30, y_pos + 25), f"Autoencoder Score: {autoencoder_score:.4f}", fill=ae_color, font=font)

    # k-NN score
    knn_color = (255, 100, 100) if knn_score > 0.5 else (100, 200, 100)
    draw.text((30, y_pos + 50), f"k-NN Score: {knn_score:.4f}", fill=knn_color, font=font)

    # Ground truth
    label_text = "Ground Truth: ANOMALY" if label == 1 else "Ground Truth: NORMAL"
    label_color = (255, 100, 100) if label == 1 else (100, 200, 100)
    draw.text((30, y_pos + 75), label_text, fill=label_color, font=font)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Demo video with anomaly detection")
    parser.add_argument(
        "--data-root",
        default="/Volumes/BIggen/AV/data/nuscenes/v1.0-mini",
        help="Path to nuScenes dataset",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=30, help="Number of frames in demo"
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second for output video"
    )
    parser.add_argument(
        "--output-dir",
        default="/Volumes/BIggen/AV/results",
        help="Output directory for video",
    )
    args = parser.parse_args()

    # Setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading models...")
    model_phase1 = ConvAutoencoder(latent_dim=128).to(device)
    model_phase1.load_state_dict(
        torch.load(output_dir / "autoencoder_phase1_v2.pt", map_location=device)
    )

    model_phase2 = ResNetEmbedder().to(device)
    state_dict = torch.load(output_dir / "resnet50_backbone_finetuned.pt", map_location=device)
    try:
        model_phase2.load_state_dict(state_dict)
    except RuntimeError:
        backbone_state = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
        model_phase2.backbone.load_state_dict(backbone_state)

    # Load k-NN scorer
    train_emb_v2 = np.load(output_dir / "phase2_embeddings_train_v2.npy")
    knn_scorer = kNNScorer(k=5)
    knn_scorer.fit(train_emb_v2)

    print(f"Models loaded. Using {args.sequence_length} frames.\n")

    # Load data
    print("Loading nuScenes frames...")
    frames = load_nuscenes_frames(args.data_root)
    splits = split_by_scene(frames, val_frac=0.15, test_frac=0.15, seed=0)

    # Use test frames (all normal)
    demo_frames = splits["test"][:args.sequence_length]
    print(f"Using {len(demo_frames)} normal frames from test set\n")

    # Process frames
    print(f"Processing {len(demo_frames)} frames...")
    demo_images = []

    for frame_idx, frame in enumerate(demo_frames):
        # Load image
        img_tensor = load_image_as_tensor(frame.path, device)
        img_pil = load_image_pil(frame.path)

        # Phase 1: Autoencoder
        recon_pil = get_autoencoder_reconstruction(model_phase1, img_tensor, device)
        with torch.no_grad():
            ae_score = model_phase1.anomaly_score(img_tensor.unsqueeze(0).to(device)).item()

        # Phase 2: k-NN
        emb = get_resnet_embedding(model_phase2, img_tensor, device)
        knn_score = knn_scorer.score(emb.reshape(1, -1))[0]

        # Normalize scores to [0, 1]
        ae_score_norm = 1.0 / (1.0 + np.exp(-5 * (ae_score - 0.5)))  # Sigmoid normalization
        knn_score_norm = np.clip(knn_score / 5.0, 0, 1)  # Clip to [0, 1]

        # Create frame
        frame_img = create_demo_frame(
            img_pil,
            recon_pil,
            ae_score_norm,
            knn_score_norm,
            label=0,  # All test frames are normal
            frame_num=frame_idx,
        )

        demo_images.append(frame_img)

        if (frame_idx + 1) % 10 == 0:
            print(f"  Processed {frame_idx + 1}/{len(demo_frames)} frames")

    print(f"Processed {len(demo_frames)} frames\n")

    # Save video
    print("Saving video...")
    video_path = output_dir / "phase4_demo.mp4"

    # Convert PIL images to numpy arrays
    frames_np = [np.array(img) for img in demo_images]

    # Write video using cv2
    frame_h, frame_w = frames_np[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (frame_w, frame_h))

    for frame_np in frames_np:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"Video saved to {video_path}\n")

    print("=" * 60)
    print("PHASE 4 DEMO COMPLETE!")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Frames: {len(demo_frames)}")
    print(f"FPS: {args.fps}")
    print(f"Resolution: {frame_w}x{frame_h}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
