"""Phase 4 CLEAN: Use only REAL CODA frames (filter junk).

Use the 100 good CODA frames (non-junk), sort by score, showcase real detections.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring.embedding_scorers import kNNScorer


class ResNetEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        feat = self.backbone(x)
        return feat.squeeze(-1).squeeze(-1)


def load_image_as_tensor(path: str, device: str) -> torch.Tensor:
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize((1600, 900), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        return img_tensor.to(device)
    except:
        return None


def load_image_pil(path: str) -> Image.Image:
    try:
        return Image.open(path).convert('RGB').resize((1600, 900), Image.Resampling.LANCZOS)
    except:
        return None


@torch.no_grad()
def get_resnet_embedding(model, img_tensor: torch.Tensor, device: str) -> np.ndarray:
    model.eval()
    emb = model(img_tensor.unsqueeze(0).to(device))
    return emb.cpu().numpy().squeeze()


def create_demo_frame(
    original_pil: Image.Image,
    knn_score: float,
    frame_path: str,
    frame_num: int,
    rank: int,
) -> Image.Image:
    """Create clean demo frame showing k-NN detection."""
    display_h, display_w = 500, 900
    img = original_pil.resize((display_w, display_h), Image.Resampling.LANCZOS)

    canvas_w = display_w + 40
    canvas_h = display_h + 280
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(15, 15, 15))

    canvas.paste(img, (20, 20))

    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except:
        font_title = font_large = font = font_small = ImageFont.load_default()

    y_pos = display_h + 30

    filename = Path(frame_path).name
    draw.text((20, y_pos), f"#{rank} - {filename}",
              fill=(180, 180, 180), font=font_small)
    y_pos += 30

    # Detection status
    is_high_anomaly = knn_score > 1.0
    if is_high_anomaly:
        status_text = "🚨 CORNER CASE DETECTED"
        status_color = (255, 100, 100)
    else:
        status_text = "⚠️ UNUSUAL SCENE"
        status_color = (255, 180, 80)

    draw.text((20, y_pos), status_text, fill=status_color, font=font_title)
    y_pos += 45

    # Score
    draw.text((20, y_pos), f"Anomaly Score:", fill=(200, 200, 200), font=font)
    draw.text((280, y_pos), f"{knn_score:.3f}", fill=status_color, font=font_large)
    y_pos += 35

    # Confidence bar
    bar_length = 300
    bar_filled = int(bar_length * min(knn_score / 2.5, 1.0))

    draw.rectangle([(20, y_pos), (20 + bar_length, y_pos + 30)],
                   fill=(50, 50, 50), outline=(100, 100, 100))

    if is_high_anomaly:
        bar_color = (255, 100, 100)
    else:
        bar_color = (255, 180, 80)

    draw.rectangle([(20, y_pos), (20 + bar_filled, y_pos + 30)],
                   fill=bar_color)

    draw.text((25, y_pos + 5), "NORMAL", fill=(100, 100, 100), font=font_small)
    draw.text((bar_length - 50, y_pos + 5), "ANOMALY", fill=(100, 100, 100), font=font_small)

    y_pos += 50

    draw.text((20, y_pos),
              "Higher score = more unusual/anomalous corner case detected",
              fill=(150, 150, 150), font=font_small)

    return canvas


def load_coda_frames(coda_path: str) -> list[str]:
    coda_dir = Path(coda_path) / 'CODA' / 'sample' / 'images'
    return sorted([str(f) for f in coda_dir.glob('*.jpg') if not f.name.startswith('._')])


def main():
    parser = argparse.ArgumentParser(description="Phase 4 CLEAN: Real CODA anomaly showcase")
    parser.add_argument("--top-n", type=int, default=15, help="Show top N anomalies")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--output-dir", default="/Volumes/BIggen/AV/results")
    parser.add_argument("--coda-path", default="/Volumes/BIggen/AV/data/coda")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")

    output_dir = Path(args.output_dir)

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

    train_emb_v2 = np.load(output_dir / "phase2_embeddings_train_v2.npy")
    knn_scorer = kNNScorer(k=5)
    knn_scorer.fit(train_emb_v2)
    print(f"Detector loaded (AUROC: 97.58% on nuScenes vs CODA)\n")

    # Load CODA frames
    print("Loading CODA corner-case frames...")
    coda_paths = load_coda_frames(args.coda_path)
    print(f"Found {len(coda_paths)} CODA frames. Scoring...\n")

    # Score all frames
    scores_list = []
    for idx, frame_path in enumerate(coda_paths):
        img_tensor = load_image_as_tensor(frame_path, device)
        if img_tensor is None:
            continue

        emb = get_resnet_embedding(model_phase2, img_tensor, device)
        knn_score = knn_scorer.score(emb.reshape(1, -1))[0]

        # Filter: skip junk frames (score == 1.952268)
        if abs(knn_score - 1.952268) < 0.0001:
            continue

        scores_list.append((frame_path, knn_score))

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(coda_paths)} frames")

    print(f"\nGood CODA frames (non-junk): {len(scores_list)}\n")

    # Sort by score descending
    scores_list.sort(key=lambda x: x[1], reverse=True)
    top_frames = scores_list[:args.top_n]

    print(f"Top {args.top_n} corner cases:")
    for i, (path, score) in enumerate(top_frames):
        print(f"  [{i+1:2d}] {score:.3f} - {Path(path).name}")

    print(f"\nGenerating demo video...\n")

    # Generate video
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
            rank=idx + 1,
        )

        demo_images.append(frame_img)

    print(f"Generated {len(demo_images)} demo frames\n")

    # Save video
    print("Saving video...")
    video_path = output_dir / "phase4_demo_clean.mp4"

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
    print("PHASE 4 FINAL - REAL ANOMALY DETECTION SHOWCASE")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Detector: Fine-tuned ResNet50 + k-NN (k=5)")
    print(f"Evaluation: 97.58% AUROC (normal nuScenes vs CODA corner cases)")
    print(f"\nData Quality:")
    print(f"  Good CODA frames shown: {len(demo_images)}")
    print(f"  (Filtered {len(scores_list) - len(top_frames)} lower-scoring anomalies)")
    print(f"\nTop {args.top_n} corner cases:")
    print(f"  Highest score: {top_frames[0][1]:.3f}")
    print(f"  Lowest score:  {top_frames[-1][1]:.3f}")
    print(f"  Mean score:    {np.mean([s for _, s in top_frames]):.3f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
