#!/bin/bash
# Download the 10 pixel-labeled validation images from RoadAnomaly21
# (SegmentMeIfYouCan benchmark), mirrored on Hugging Face.
# Source: https://huggingface.co/datasets/kumuji/roadanomaly21_roadobstacle21
# License: see RoadAnomaly21/LICENSE.txt in the repo -- non-commercial
# research use, must cite the SMIYC paper (Chan et al., NeurIPS 2021).
#
# RoadAnomaly21 has 100 test images total, but only 10 (validation0000..0009)
# have published pixel-level ground-truth masks -- the rest (airplane0000,
# bear0000, camel0000, etc.) are unlabeled test-server-only images. We only
# need the 10 labeled ones for quantitative cross-dataset evaluation.
#
# Confirmed via direct HF tree API query: labels_masks/ contains exactly
# validation0000_labels_semantic.png .. validation0009_labels_semantic.png
# (plus _color.png visualization variants, which we skip), and these pair
# by numeric stem with images/validation0000.jpg .. validation0009.jpg.
#
# These are genuinely different-source images from Lost & Found -- compiled
# from diverse web imagery across many countries/settings, not one German
# city's dashcam footage. Good test of whether the fitted reference bank /
# calibration generalizes, or was overfit to the L&F domain.

set -e

DEST="/Volumes/BIggen/AV/data/roadanomaly21"
mkdir -p "$DEST/images" "$DEST/labels_masks"
cd "$DEST"

BASE="https://huggingface.co/datasets/kumuji/roadanomaly21_roadobstacle21/resolve/main/RoadAnomaly21"

for i in 0 1 2 3 4 5 6 7 8 9; do
  n=$(printf "%04d" "$i")
  echo "Downloading validation${n}..."
  curl -sL -o "images/validation${n}.jpg" "${BASE}/images/validation${n}.jpg?download=true"
  curl -sL -o "labels_masks/validation${n}_labels_semantic.png" "${BASE}/labels_masks/validation${n}_labels_semantic.png?download=true"
done

echo ""
echo "Done. Files:"
ls -la "$DEST/images"
ls -la "$DEST/labels_masks"
