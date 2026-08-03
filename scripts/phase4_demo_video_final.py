"""Phase 4 FINAL: Showcase the WORKING detector (Phase 2 k-NN).

Skip the weak autoencoder. Show only the fine-tuned ResNet50 k-NN detector
which actually achieves 97.58% AUROC. Display:
- Frame
- k-NN anomaly score (high = anomaly)
- Confidence bar
- Detection status
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
def get_resnet_embedding(model, img_tensor: torch.Tensor, device: str) -> np.ndarray:
    """Extract embedding from ResNet50."""
    model.eval()
    emb = model(img_tensor.unsqueeze(0).to(device))
    return emb.cpu().numpy().squeeze()


def create_demo_frame(
    original_pil: Image.Image,
    knn_score: float,
    frame_path: str,
    frame_num: int,
) -> Image.Image:
    """Create clean demo frame showing k-NN detection."""
    # Resize image
    display_h, display_w = 500, 900
    img = original_pil.resize((display_w, display_h), Image.Resampling.LANCZOS)

    # Canvas: image on top, info panel below
    canvas_w = display_w + 40
    canvas_h = display_h + 280
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(15, 15, 15))

    # Paste image
    canvas.paste(img, (20, 20))

    # Draw info panel
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except:
        font_title = font_large = font = font_small = ImageFont.load_default()

    y_pos = display_h + 30

    # Frame info
    filename = Path(frame_path).name
    draw.text((20, y_pos), f"Frame {frame_num:02d} - {filename}",
              fill=(180, 180, 180), font=font_small)
    y_pos += 30

    # Detection status - BIG and bold
    is_anomaly = knn_score > 2.5  # Threshold for this detector
    if is_anomaly:
        status_text = "🚨 ANOMALY DETECTED 🚨"
        status_color = (255, 80, 80)
        confidence = min(knn_score / 5.0, 1.0)
    else:
        status_text = "✓ NORMAL"
        status_color = (100, 200, 100)
        confidence = 1.0 - min(knn_score / 5.0, 1.0)

    draw.text((20, y_pos), status_text, fill=status_color, font=font_title)
    y_pos += 45

    # Score display
    draw.text((20, y_pos), f"k-NN Anomaly Score:", fill=(200, 200, 200), font=font)
    draw.text((350, y_pos), f"{knn_score:.3f}", fill=status_color, font=font_large)
    y_pos += 35

    # Confidence bar
    bar_length = 300
    bar_filled = int(bar_length * min(knn_score / 5.0, 1.0))

    # Background
    draw.rectangle([(20, y_pos), (20 + bar_length, y_pos + 30)],
                   fill=(50, 50, 50), outline=(100, 100, 100))

    # Filled portion (red for anomaly, green for normal)
    if is_anomaly:
        bar_color = (255, 100, 100)
    else:
        bar_color = (100, 200, 100)

    draw.rectangle([(20, y_pos), (20 + bar_filled, y_pos + 30)],
                   fill=bar_color)

    # Labels on bar
    draw.text((25, y_pos + 5), "NORMAL", fill=(100, 100, 100), font=font_small)
    draw.text((bar_length - 50, y_pos + 5), "ANOMALY", fill=(100, 100, 100), font=font_small)

    y_pos += 50

    # Explanation
    draw.text((20, y_pos),
              "Higher score = more anomalous. Threshold ≈ 2.5 for this dataset.",
              fill=(150, 150, 150), font=font_small)

    return canvas


def load_coda_frames(coda_path: str) -> list[str]:
    """Load CODA corner-case image paths."""
    coda_dir = Path(coda_path) / 'CODA' / 'sample' / 'images'
    return sorted([str(f) for f in coda_dir.glob('*.jpg') if not f.name.startswith('._')])


def main():
    parser = argparse.ArgumentParser(description="Phase 4 FINAL: k-NN detector showcase")
    parser.add_argument(
        "--top-n", type=int, default=20, help="Show top N highest-score anomalies"
    )
    parser.add_argument(
        "--fps", type=int, default=5, help="Frames per second for output video"
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

    # Load detector
    print("Loading fine-tuned ResNet50 k-NN detector...")
    model_phase2 = ResNetEmbedder().to(device)
    state_dict = torch.load(output_dir / "resnet50_backbone_finetuned.pt", map_location=device)
    try:
        model_phase2.load_state_dict(state_dict)
    except RuntimeError:
        backbone_state = {k.replace("backbone.", ""): v for k, v in state_dict.items()
                         if k.startswith("backbone.")}
        model_phase2.backbone.load_state_dict(backbone_state)

    # Load k-NN scorer
    train_emb_v2 = np.load(output_dir / "phase2_embeddings_train_v2.npy")
    knn_scorer = kNNScorer(k=5)
    knn_scorer.fit(train_emb_v2)
    print(f"Detector loaded (AUROC: 97.58% on evaluation set)\n")

    # Load ALL CODA frames
    print("Loading all CODA corner-case frames...")
    coda_paths = load_coda_frames(args.coda_path)
    print(f"Found {len(coda_paths)} CODA frames. Scoring all...\n")

    # Score all frames
    scores_list = []
    for idx, frame_path in enumerate(coda_paths):
        img_tensor = load_image_as_tensor(frame_path, device)
        if img_tensor is None:
            continue

        # Get k-NN score
        emb = get_resnet_embedding(model_phase2, img_tensor, device)
        knn_score = knn_scorer.score(emb.reshape(1, -1))[0]

        scores_list.append((frame_path, knn_score))

        if (idx + 1) % 50 == 0:
            print(f"  Scored {idx + 1}/{len(coda_paths)} frames")

    print(f"\nScored all {len(scores_list)} frames")

    # Sort by score descending
    scores_list.sort(key=lambda x: x[1], reverse=True)
    top_frames = scores_list[:args.top_n]

    print(f"\nTop {args.top_n} highest-scoring anomalies:")
    for i, (path, score) in enumerate(top_frames):
        status = "ANOMALY" if score > 2.5 else "BORDERLINE"
        print(f"  [{i+1:2d}] {score:.3f} ({status}) - {Path(path).name}")

    print(f"\nGenerating demo video...\n")

    # Generate video from top frames
    demo_images = []
    for idx, (frame_path, knn_score) in enumerate(top_frames):
        img_tensor = load_image_as_tensor(frame_path, device)
        if img_tensor is None:
            continue

        img_pil = load_image_pil(frame_path)
        if img_pil is None:
            continue

        frame_img = create_demo_frame(
            img_pil,
            knn_score,
            frame_path,
            frame_num=idx + 1,
        )

        demo_images.append(frame_img)

    print(f"Generated {len(demo_images)} demo frames\n")

    # Save video
    print("Saving video...")
    video_path = output_dir / "phase4_demo_final.mp4"

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
    print("PHASE 4 FINAL - k-NN ANOMALY DETECTOR SHOWCASE")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Detector: Fine-tuned ResNet50 + k-NN (k=5)")
    print(f"Performance: AUROC 97.58% on CODA + nuScenes evaluation set")
    print(f"\nTop {args.top_n} corner cases:")
    print(f"  Highest score: {top_frames[0][1]:.3f}")
    print(f"  Lowest score:  {top_frames[-1][1]:.3f}")
    print(f"  Anomaly threshold: ~2.5")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
