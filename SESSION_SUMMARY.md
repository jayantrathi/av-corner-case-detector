# Tonight's session — what actually happened

## Where it started

RbA (rejection-based anomaly scoring on frozen Mask2Former) was stuck at 0/30 hit rate on real Lost & Found hazard frames, even after several rounds of real fixes (border margin, ego-hood exclusion, dataset scoping). A new run showed a fresh artifact — the flagged region hugging the edge of every exclusion box — and that was the point it stopped feeling worth it.

## The pivot

Decision: stop patching RbA, build something simple and reliable instead.
- `simple_demo/run_yolo_dashcam_demo.py` — plain YOLOv8 on real Lost & Found footage. Worked immediately.
- `simple_demo/run_camera_lidar_demo.py` — YOLO + a synced LiDAR bird's-eye-view panel on nuScenes, correctly paired via nuScenes' own sample-token metadata (not timestamp guessing). Worked, looked good.

Then the honest gut-check: is YOLO-on-a-dataset actually worth anything for the project? No — it's plumbing, not a contribution. The real project is still the corner-case detector.

## Back to RbA: the actual fix

Root cause of the boundary-hugging bug: RbA's scores are smooth spatial gradients, not sharp spikes, so hard-masking a region just relocates the "winner" to the mask's edge. Fix: local-contrast scoring — subtract a heavily-blurred version of the score map from itself, so broad artifacts (border falloff, the hood) get flattened while genuine small local anomalies survive.

Result: raw peak hit rate 0/30 → local-contrast-only (no masking) 5/30 → contrast + border margin 7/30. Real improvement, not dramatic, but real and explained.

## "Is this actually possible to make good?"

Checked the RbA paper's own released checkpoints. Their Swin-B model, fine-tuned with actual COCO outlier-exposure supervision, reports AP ~71 on Fishyscapes Lost & Found — much stronger than our zero-shot vanilla Cityscapes checkpoint. Decided to integrate it for real instead of guessing.

## The integration saga

Their code is Detectron2-based (not HuggingFace `transformers`), officially requires a compiled CUDA kernel with no documented CPU path. Went and read the actual source instead of assuming it was a dead end:
- Found the deformable-attention module already had a pure-PyTorch fallback, just not wired up correctly on CPU — one-line patch.
- Hit a real chain of missing/mis-specified dependencies one at a time (`timm`, `fairscale`, `zmq` — their `requirements.txt` literally lists the wrong package name, which silently broke the whole batch install).
- `pip install -e detectron2` failed on build isolation not seeing torch — `--no-build-isolation` fixed it.
- Config defaulted to `MODEL.DEVICE: cuda`, crashed on a CPU-only torch build — forced to `cpu` in the launch config.
- Checkpoint zip extracted into a doubled nested folder (its own internal folder name matched the target dir) — model file was one level deeper than expected.

Every one of these was a real, one-at-a-time blocker, not guesswork — each fix was verified against actual error output or actual source code before being applied.

## The result, honestly

| | AUROC | AUPR | Localization |
|---|---|---|---|
| kNN baseline | 0.9439 | 0.0830 | 60% top-5 |
| RbA zero-shot | 0.7917 | 0.0065 | 23% region |
| RbA official checkpoint | 0.8750 | 0.0113 | 17% region |

Global ranking improved with the real checkpoint. Localization did not — and the same border artifact was still visible in every example frame, even with the properly-calibrated model. That points to something structural in the architecture's receptive field near image edges, not a checkpoint problem. Likely secondary factor: COCO's outlier-exposure objects are much larger than Lost & Found's actual hazards (small lost cargo), so the model learned to reject "weird COCO-sized things," not small out-of-place ones.

Conclusion: the simpler kNN baseline is still the better-performing method. That's a legitimate, defensible finding, not a failure — full writeup in `RBA_FINDINGS.md`.

## Resume / repo cleanup

- Reviewed the actual resume — recommended leading any new bullet with the kNN numbers (clean, quantifiable) rather than the RbA investigation (a negative result reads badly compressed into one bullet line, even though the underlying work is solid).
- Cleaned the repo before its first real push: cut an entire obsolete earlier pipeline (autoencoder baseline, fine-tuned-ResNet branch, a CODA cross-dataset eval later found to have a confound, five duplicate iterations of the same demo script), removed a few literal junk files from a broken early setup script, pulled internal AI-build-scaffolding docs (`DEVELOPMENT.md`, `PHASE0-START.md`) out of what's public, and rewrote `README.md` to actually reflect the current state instead of "results coming soon."
