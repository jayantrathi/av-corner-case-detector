#!/usr/bin/env bash
# Sets up the OFFICIAL RbA repo + an outlier-exposure-finetuned checkpoint,
# CPU-only, no CUDA required -- run this on your Mac (not in any sandbox).
#
# Why this is possible at all: the official install docs (INSTALL.md) ask you
# to compile a CUDA kernel for MultiScaleDeformableAttention (MSDeformAttn),
# which won't build without an NVIDIA GPU. BUT the actual forward() call in
# ms_deform_attn.py already has a try/except that falls back to a pure-
# PyTorch implementation (ms_deform_attn_core_pytorch, grid_sample-based) if
# the compiled op fails -- that part needs zero patching. The only real
# blocker is that the import of the (uncompiled) CUDA module raises at
# IMPORT time, not call time, which crashes the whole package before that
# try/except ever runs. The one-line patch below fixes exactly that: let the
# import fail quietly (MSDA = None) instead of raising, so the module loads,
# and the existing fallback in forward() takes over correctly from there.
# Verified directly against the real source on GitHub, not guessed.
#
# What this gets us: the Swin-B checkpoint finetuned with actual COCO
# outlier-exposure supervision, reporting AP 70.8 / FPR95 6.3 on Fishyscapes
# Lost & Found in the paper's own numbers -- vs. our current zero-shot
# vanilla Cityscapes checkpoint, which was never calibrated for this task.
set -euo pipefail

EXT_DIR="/Volumes/BIggen/AV/external"
mkdir -p "$EXT_DIR"
cd "$EXT_DIR"

echo "== [1/6] Cloning Detectron2 and installing CPU-only =="
if [ ! -d detectron2 ]; then
  git clone https://github.com/facebookresearch/detectron2.git
fi
python3 -c "import torch" || { echo "ERROR: torch not importable in the active environment -- activate the same conda env (av-detector) your other scripts use, then re-run."; exit 1; }
# --no-build-isolation is required here: pip's default PEP 517 isolated
# build env does NOT include torch, even though your active environment
# does -- detectron2's setup.py imports torch at build time to decide which
# extensions to build, so the isolated build fails with "No module named
# 'torch'" without this flag. This is a known detectron2 install issue, not
# specific to this project. Detectron2's setup.py itself checks
# torch.cuda.is_available() and skips CUDA extensions automatically when
# it's False -- no other flags needed for CPU-only.
pip install --no-build-isolation -e ./detectron2

echo "== [2/6] Cloning the official RbA repo (it vendors its own mask2former/ copy) =="
if [ ! -d RbA ]; then
  git clone https://github.com/NazirNayal8/RbA.git
fi
cd RbA
# NOTE: requirements.txt lists "zmq" verbatim -- that's not a real installable
# PyPI package (the real one is pyzmq, which is imported AS zmq). That line
# makes the whole `pip install -r requirements.txt` batch fail silently
# (hence the `|| true` masking it), so NONE of these were ever actually
# installed -- explains discovering timm/fairscale/zmq one at a time via
# import errors. Installing the corrected, complete list in one shot instead.
pip install cython scipy shapely timm h5py submitit scikit-image easydict \
  albumentations fairscale pyzmq webp ood-metrics opencv-python
pip install git+https://github.com/cocodataset/panopticapi.git   # not on PyPI under this name, needs git install
# mask2former/modeling/__init__.py and mask2former/__init__.py eagerly import
# ALL backbones (vit, mvit, swin, wideresnet38, mix_transformer) and ALL
# dataset mappers to register them with detectron2, even though we only need
# Swin for inference -- so we pay the import-time dependency cost of code
# paths we'll never call. If one more shows up anyway, same move: pip install
# whatever the traceback names.

echo "== [3/6] Patching the CPU import blocker (1 line, see header comment above) =="
DEFORM_FUNC="mask2former/modeling/pixel_decoder/ops/functions/ms_deform_attn_func.py"
python3 - "$DEFORM_FUNC" <<'PYEOF'
import sys, re
path = sys.argv[1]
src = open(path).read()
needle = 'except ModuleNotFoundError as e:'
if 'MSDA = None' in src:
    print("Already patched, skipping.")
else:
    old = '''try:
    import MultiScaleDeformableAttention as MSDA
except ModuleNotFoundError as e:
    info_string = (
        "\\n\\nPlease compile MultiScaleDeformableAttention CUDA op with the following commands:\\n"
        "\\t`cd mask2former/modeling/pixel_decoder/ops`\\n"
        "\\t`sh make.sh`\\n"
    )
    raise ModuleNotFoundError(info_string)'''
    new = '''try:
    import MultiScaleDeformableAttention as MSDA
except ModuleNotFoundError as e:
    # PATCHED for CPU-only use: let this import fail quietly instead of
    # crashing the package. ms_deform_attn.py's forward() already has a
    # try/except that falls back to the pure-PyTorch ms_deform_attn_core_pytorch
    # implementation whenever calling into MSDA raises -- None.ms_deform_attn_forward(...)
    # will raise AttributeError there, which that except clause already catches.
    MSDA = None'''
    if old not in src:
        print("WARNING: expected text not found verbatim -- inspect this file by hand:", path)
        sys.exit(1)
    src = src.replace(old, new)
    open(path, 'w').write(src)
    print("Patched:", path)
PYEOF

echo "== [4/6] Downloading config + checkpoint (swin_b_1dl_rba_ood_coco) =="
mkdir -p ckpts/swin_b_1dl_rba_ood_coco
if [ ! -f ckpts/swin_b_1dl_rba_ood_coco/config.yaml ]; then
  curl -L -o ckpts/swin_b_1dl_rba_ood_coco/config.yaml \
    https://raw.githubusercontent.com/NazirNayal8/RbA/main/ckpts/swin_b_1dl_rba_ood_coco/config.yaml
fi
if [ ! -f ckpts/swin_b_1dl_rba_ood_coco/model_final.pth ]; then
  curl -L -o /tmp/swin_b_1dl_rba_ood_coco.zip \
    https://github.com/NazirNayal8/RbA/releases/download/model-weights/swin_b_1dl_rba_ood_coco.zip
  # The zip's own internal top-level folder is ALSO named swin_b_1dl_rba_ood_coco/,
  # so unzip into ckpts/ (not ckpts/swin_b_1dl_rba_ood_coco/) to avoid doubling
  # the path -- confirmed via `ls` that the first run produced exactly that
  # doubled nesting.
  unzip -o /tmp/swin_b_1dl_rba_ood_coco.zip -d ckpts
else
  echo "Checkpoint already present, skipping download."
fi

echo "== [5/6] Sanity import check (no dataset registration, just: does it import) =="
python3 -c "
import torch
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
import detectron2
print('detectron2 OK:', detectron2.__version__)
from mask2former.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import MSDA
print('MSDA is None (expected on CPU):', MSDA is None)
from mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
print('Mask2Former pixel decoder imports OK')
"

echo "== [6/6] Done. Next: run rba_official_scorer.py to load the model and sanity-score one image. =="
