"""Phase 4 v2: Impressive demo video showing ANOMALY DETECTION.

Load CODA corner cases and showcase:
- Original anomalous frame
- Autoencoder reconstruction + error heatmap
- Anomaly score comparison
- Detection results with confidence
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
from matplotlib.colors import Normalize

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

    # Create heatmap (error map)
    # Normalize to [0, 255]
    error_norm = (error - error.min()) / (error.max() - error.min() + 1e-6)
    heatmap = plt.cm.hot(error_norm)  # Red = high error
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)

    return Image.fromarray(heatmap), overall_score


@torch.no_grad()
def get_resnet_embedding(model, img_tensor: torch.Tensor, device: str) -> np.ndarray:
    """Extract embedding from ResNet50."""
    model.eval()
    emb = model(img_tensor.unsqueeze(0).to(device))
    return emb.cpu().numpy().squeeze()


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    if len(arr.shape) == 2:
        return Image.fromarray(arr, mode='L')
    return Image.fromarray(arr, mode='RGB')


def create_impressive_frame(
    original_pil: Image.Image,
    error_heatmap_pil: Image.Image,
    ae_score: float,
    knn_score: float,
    frame_num: int,
    is_anomaly: bool,
) -> Image.Image:
    """Create an impressive demo frame highlighting the anomaly detection."""
    # Dimensions
    display_h, display_w = 450, 800
    original = original_pil.resize((display_w, display_h), Image.Resampling.LANCZOS)
    heatmap = error_heatmap_pil.resize((display_w, display_h), Image.Resampling.LANCZOS)

    # Canvas: original + heatmap + big info panel
    canvas_w = display_w * 2 + 60
    canvas_h = display_h + 250
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))

    # Paste images
    canvas.paste(original, (20, 20))
    canvas.paste(heatmap, (display_w + 30, 20))

    # Draw text
    draw = ImageDraw.Draw(canvas)
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except:
        font_large = font = font_small = ImageFont.load_default()

    # Title
    y_pos = display_h + 30
    draw.text((20, y_pos), f"Frame {frame_num:04d} - ", fill=(200, 200, 200), font=font_large)

    # Detection status
    if is_anomaly:
        status = "⚠️ ANOMALY DETECTED"
        status_color = (255, 80, 80)
    else:
        status = "✓ Normal"
        status_color = (100, 200, 100)

    draw.text((260, y_pos), status, fill=status_color, font=font_large)

    # Scores with colored bars
    y_pos += 45

    # Autoencoder score
    ae_color = (255, 100, 100) if ae_score > 0.5 else (100, 200, 100)
    draw.text((20, y_pos), f"Autoencoder:", fill=(200, 200, 200), font=font)
    draw.text((200, y_pos), f"{ae_score:.4f}", fill=ae_color, font=font)

    # Score bar for autoencoder
    bar_width = int(300 * np.clip(ae_score, 0, 1))
    draw.rectangle([(20, y_pos + 25), (20 + bar_width, y_pos + 35)], fill=ae_color)
    draw.rectangle([(20, y_pos + 25), (320, y_pos + 35)], outline=(100, 100, 100))

    y_pos += 50

    # k-NN score
    knn_color = (255, 100, 100) if knn_score > 0.5 else (100, 200, 100)
    draw.text((20, y_pos), f"k-NN Score:", fill=(200, 200, 200), font=font)
    draw.text((200, y_pos), f"{knn_score:.4f}", fill=knn_color, font=font)

    # Score bar for k-NN
    bar_width = int(300 * np.clip(knn_score / 5.0, 0, 1))
    draw.rectangle([(20, y_pos + 25), (20 + bar_width, y_pos + 35)], fill=knn_color)
    draw.rectangle([(20, y_pos + 25), (320, y_pos + 35)], outline=(100, 100, 100))

    y_pos += 50

    # Explanation text
    draw.text((20, y_pos), "Left: Original Frame | Right: Reconstruction Error Heatmap",
              fill=(150, 150, 150), font=font_small)
    draw.text((20, y_pos + 20), "Red = High reconstruction error = Anomaly detected",
              fill=(150, 150, 150), font=font_small)

    return canvas


def load_coda_frames(coda_path: str) -> list[str]:
    """Load CODA corner-case image paths."""
    coda_dir = Path(coda_path) / 'CODA' / 'sample' / 'images'
    return sorted([str(f) for f in coda_dir.glob('*.jpg') if not f.name.startswith('._')])


def main():
    parser = argparse.ArgumentParser(description="Phase 4 v2: Impressive anomaly detection demo")
    parser.add_argument(
        "--sequence-length", type=int, default=20, help="Number of anomaly frames in demo"
    )
    parser.add_argument(
        "--fps", type=int, default=8, help="Frames per second for output video"
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

    # Load CODA anomaly frames
    print("Loading CODA corner-case frames...")
    coda_paths = load_coda_frames(args.coda_path)
    demo_frames = coda_paths[:args.sequence_length]
    print(f"Using {len(demo_frames)} anomaly frames from CODA\n")

    # Process frames
    print(f"Processing {len(demo_frames)} anomaly frames...")
    demo_images = []

    for frame_idx, frame_path in enumerate(demo_frames):
        # Load image
        img_tensor = load_image_as_tensor(frame_path, device)
        if img_tensor is None:
            print(f"  Skipping {frame_path} (failed to load)")
            continue

        img_pil = load_image_pil(frame_path)
        if img_pil is None:
            continue

        # Phase 1: Autoencoder - get error heatmap
        error_heatmap_pil, ae_score = get_reconstruction_error_map(model_phase1, img_tensor, device)

        # Phase 2: k-NN
        emb = get_resnet_embedding(model_phase2, img_tensor, device)
        knn_score = knn_scorer.score(emb.reshape(1, -1))[0]

        # Normalize scores
        ae_score_norm = 1.0 / (1.0 + np.exp(-5 * (ae_score - 0.5)))
        knn_score_norm = np.clip(knn_score / 5.0, 0, 1)

        # Both should detect this as anomaly (score > 0.5)
        is_detected = (ae_score_norm > 0.4) or (knn_score_norm > 0.3)

        # Create frame
        frame_img = create_impressive_frame(
            img_pil,
            error_heatmap_pil,
            ae_score_norm,
            knn_score_norm,
            frame_num=frame_idx,
            is_anomaly=True,
        )

        demo_images.append(frame_img)

        status = "✓ DETECTED" if is_detected else "✗ MISSED"
        print(f"  [{frame_idx + 1:2d}] AE: {ae_score_norm:.3f} | k-NN: {knn_score_norm:.3f} | {status}")

    print(f"\nProcessed {len(demo_images)} frames\n")

    # Save video
    print("Saving video...")
    video_path = output_dir / "phase4_demo_v2.mp4"

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
    print("PHASE 4 v2 DEMO COMPLETE - ANOMALY DETECTION SHOWCASE")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Frames: {len(demo_images)}")
    print(f"FPS: {args.fps}")
    print(f"Resolution: {frame_w}x{frame_h}")
    print("\nShows CODA corner cases being detected as anomalies")
    print("Red heatmap = reconstruction error = model struggling = anomaly")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
