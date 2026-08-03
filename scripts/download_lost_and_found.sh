#!/bin/bash
# Download the Lost & Found dataset (real photographed road hazards).
# Source: https://huggingface.co/datasets/kumuji/lost_and_found
# License: Daimler AG, non-commercial academic/research use, must cite.
#
# Only pulls what we need: left camera images (5.8GB) + coarse ground-truth
# masks (37.8MB) + label definitions. Skips right-camera, disparity, and
# odometry data -- not needed for this project. Total: ~5.85GB.

set -e

DEST="/Volumes/BIggen/AV/data/lost_and_found"
mkdir -p "$DEST"
cd "$DEST"

echo "Downloading gtCoarse.zip (37.8MB, ground-truth masks)..."
curl -L -o gtCoarse.zip "https://huggingface.co/datasets/kumuji/lost_and_found/resolve/main/gtCoarse.zip?download=true"

echo "Downloading labels.py (class ID definitions)..."
curl -L -o labels.py "https://huggingface.co/datasets/kumuji/lost_and_found/resolve/main/labels.py?download=true"

echo "Downloading leftImg8bit.zip (5.8GB, camera images) -- this will take a while..."
curl -L -o leftImg8bit.zip "https://huggingface.co/datasets/kumuji/lost_and_found/resolve/main/leftImg8bit.zip?download=true"

echo "Extracting..."
unzip -q gtCoarse.zip -d gtCoarse
unzip -q leftImg8bit.zip -d leftImg8bit

echo ""
echo "Done. Directory structure:"
find "$DEST" -maxdepth 2 -type d
echo ""
echo "Sample image files:"
find "$DEST/leftImg8bit" -name "*.png" | head -5
echo ""
echo "Sample ground-truth files:"
find "$DEST/gtCoarse" -name "*.png" | head -5
