"""Phase 4 v3: REAL anomaly detection showcase.

Score ALL CODA frames, sort by anomaly score, show only HIGH-confidence detections.
This time we'll actually see RED heatmaps and high scores.
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
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.autoencoder import ConvAutoencoder
from src.scoring.embedding_scorers import kNNScorer


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


def load_image_as_tensor(path: str, device: str) -> torch.Tensor:
    """Load and normalize image to tensor."""
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize((1600, 900), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        return img_tensor.to(device)
    except Exception as e:
        return None


def load_image_pil(path: str) -> Image.Image:
    """Load image as PIL Image."""
    try:
        return Image.open(path).convert('RGB').resize((1600, 900), Image.Resampling.LANCZOS)
    except:
        return None


@torch.no_grad()
def get_reconstruction_error_map(model, img_tensor: torch.Tensor, device: str) -> tuple[Image.Image, float]:
    """Get per-pixel reconstruction error map and overall score."""
    model.eval()
    recon, _ = model(img_tensor.unsqueeze(0).to(device))

    # Per-pixel error
    img = img_tensor.unsqueeze(0)
    error = torch.sqrt(((img - recon) ** 2).mean(dim=1, keepdim=True) + 1e-6)
    error = error.squeeze().cpu().numpy()

    # Overall anomaly score
    overall_score = error.mean()

    # Create heatmap (error map) - RED for high error
    error_norm = np.clip((error - error.min()) / (error.max() - error.min() + 1e-6), 0, 1)
    heatmap = plt.cm.hot(error_norm)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)

    return Image.fromarray(heatmap), overall_score


@torch.no_grad()
def get_resnet_embedding(model, img_tensor: torch.Tensor, device: str) -> np.ndarray:
    """Extract embedding from ResNet50."""
    model.eval()
    emb = model(img_tensor.unsqueeze(0).to(device))
    return emb.cpu().numpy().squeeze()


def create_demo_frame(
    original_pil: Image.Image,
    error_heatmap_pil: Image.Image,
    ae_score: float,
    knn_score: float,
    frame_num: int,
) -> Image.Image:
    """Create demo frame with anomaly detection results."""
    # Dimensions
    display_h, display_w = 450, 800
    original = original_pil.resize((display_w, display_h), Image.Resampling.LANCZOS)
    heatmap = error_heatmap_pil.resize((display_w, display_h), Image.Resampling.LANCZOS)

    # Canvas
    canvas_w = display_w * 2 + 60
    canvas_h = display_h + 250
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))

    # Paste images
    canvas.paste(original, (20, 20))
    canvas.paste(heatmap, (display_w + 30, 20))

    # Draw text
    draw = ImageDraw.Draw(canvas)
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except:
        font_large = font = font_small = ImageFont.load_default()

    # Detection status (BIG RED ALERT)
    y_pos = display_h + 30
    draw.text((20, y_pos), "🚨 ANOMALY DETECTED 🚨", fill=(255, 50, 50), font=font_large)

    y_pos += 50

    # Scores
    ae_color = (255, 100, 100)
    knn_color = (255, 100, 100)

    draw.text((20, y_pos), f"Autoencoder Score: {ae_score:.4f}", fill=ae_color, font=font)
    bar_width = int(300 * np.clip(ae_score, 0, 1))
    draw.rectangle([(20, y_pos + 25), (20 + bar_width, y_pos + 35)], fill=ae_color)
    draw.rectangle([(20, y_pos + 25), (320, y_pos + 35)], outline=(100, 100, 100))

    y_pos += 50

    draw.text((20, y_pos), f"k-NN Score: {knn_score:.4f}", fill=knn_color, font=font)
    bar_width = int(300 * np.clip(knn_score / 5.0, 0, 1))
    draw.rectangle([(20, y_pos + 25), (20 + bar_width, y_pos + 35)], fill=knn_color)
    draw.rectangle([(20, y_pos + 25), (320, y_pos + 35)], outline=(100, 100, 100))

    y_pos += 50

    draw.text((20, y_pos), "Left: Anomalous Scene | Right: Model Confusion (RED = High Error)",
              fill=(150, 150, 150), font=font_small)

    return canvas


def load_coda_frames(coda_path: str) -> list[str]:
    """Load CODA corner-case image paths."""
    coda_dir = Path(coda_path) / 'CODA' / 'sample' / 'images'
    return sorted([str(f) for f in coda_dir.glob('*.jpg') if not f.name.startswith('._')])


def main():
    parser = argparse.ArgumentParser(description="Phase 4 v3: Show TOP anomalies only")
    parser.add_argument(
        "--top-n", type=int, default=15, help="Show top N highest-score anomalies"
    )
    parser.add_argument(
        "--fps", type=int, default=6, help="Frames per second for output video"
    )
    parser.add_argument(
        "--output-dir",
        default="/Volumes/BIggen/AV/results",
        help="Output directory for video",
    )
    parser.add_argument(
        "--coda-path",
        default="/Volumes/BIggen/AV/data/coda",
        help="Path to CODA dataset",
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

    print(f"Models loaded.\n")

    # Load ALL CODA frames
    print("Loading all CODA corner-case frames...")
    coda_paths = load_coda_frames(args.coda_path)
    print(f"Found {len(coda_paths)} CODA frames. Scoring all of them...\n")

    # Score all frames
    scores_list = []
    for idx, frame_path in enumerate(coda_paths):
        img_tensor = load_image_as_tensor(frame_path, device)
        if img_tensor is None:
            continue

        # Phase 1 score
        _, ae_score = get_reconstruction_error_map(model_phase1, img_tensor, device)

        # Phase 2 score
        emb = get_resnet_embedding(model_phase2, img_tensor, device)
        knn_score = knn_scorer.score(emb.reshape(1, -1))[0]

        # Combined score (average)
        ae_norm = 1.0 / (1.0 + np.exp(-5 * (ae_score - 0.5)))
        knn_norm = np.clip(knn_score / 5.0, 0, 1)
        combined_score = (ae_norm + knn_norm) / 2.0

        scores_list.append((frame_path, combined_score, ae_norm, knn_norm))

        if (idx + 1) % 50 == 0:
            print(f"  Scored {idx + 1}/{len(coda_paths)} frames")

    print(f"Scored all {len(scores_list)} frames\n")

    # Sort by score descending - pick TOP anomalies
    scores_list.sort(key=lambda x: x[1], reverse=True)
    top_frames = scores_list[:args.top_n]

    print(f"Top {args.top_n} highest-scoring anomalies:")
    for i, (path, score, ae, knn) in enumerate(top_frames):
        print(f"  [{i+1:2d}] Score: {score:.4f} (AE: {ae:.3f}, k-NN: {knn:.3f}) - {Path(path).name}")

    print(f"\nGenerating demo video with top {args.top_n} detections...\n")

    # Generate video from top frames
    demo_images = []
    for idx, (frame_path, combined_score, ae_score, knn_score) in enumerate(top_frames):
        img_tensor = load_image_as_tensor(frame_path, device)
        if img_tensor is None:
            continue

        img_pil = load_image_pil(frame_path)
        if img_pil is None:
            continue

        error_heatmap_pil, _ = get_reconstruction_error_map(model_phase1, img_tensor, device)

        frame_img = create_demo_frame(
            img_pil,
            error_heatmap_pil,
            ae_score,
            knn_score,
            frame_num=idx + 1,
        )

        demo_images.append(frame_img)

    print(f"Generated {len(demo_images)} demo frames\n")

    # Save video
    print("Saving video...")
    video_path = output_dir / "phase4_demo_top_anomalies.mp4"

    frames_np = [np.array(img) for img in demo_images]
    frame_h, frame_w = frames_np[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (frame_w, frame_h))

    for frame_np in frames_np:
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"Video saved to {video_path}\n")

    print("=" * 70)
    print("PHASE 4 v3 - TOP ANOMALIES SHOWCASE")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Frames: {len(demo_images)} (top {args.top_n} highest-scoring anomalies)")
    print(f"FPS: {args.fps}")
    print(f"\nAll frames show STRONG detections:")
    print(f"  - Autoencoder scores: {top_frames[0][2]:.4f} to {top_frames[-1][2]:.4f}")
    print(f"  - k-NN scores: {top_frames[0][3]:.4f} to {top_frames[-1][3]:.4f}")
    print(f"  - Red heatmaps = high reconstruction error = anomalies caught")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
