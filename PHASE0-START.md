# Phase 0 — Start Here

This scaffold already has the leakage-proof foundation built and tested. Your Phase 0 job is to plug a real dataset into it. Do these in order.

## What's already here and working
- `src/data/splits.py` — scene-level splitting. **Leakage is impossible by construction.** Tested.
- `src/eval/metrics.py` — AUROC, AUPR, FPR@95. The correct OOD metrics. Tested.
- `src/scoring/embedding_scorers.py` — kNN and Mahalanobis scorers (your primary method + baseline). Tested.
- `scripts/test_splits.py` — run it: `python scripts/test_splits.py`. Proves the split works and shows why naive splitting leaks 100% of scenes.

Run all three self-tests first to confirm your environment:
```bash
python scripts/test_splits.py
python src/eval/metrics.py
python src/scoring/embedding_scorers.py
```

## Your Phase 0 tasks (in order)
1. **Pick and download one dataset** for the "normal" distribution. Default recommendation: BDD100K (diverse) or nuScenes (multi-sensor prestige). Put it under `data/` (gitignored).
2. **Write a dataset adapter** in `src/data/` that walks the dataset and produces a list of `Frame(path, scene_id, is_anomaly)` objects. The ONLY hard requirement: `scene_id` must be the real scene/drive/sequence id from the dataset, so the split stays honest. For nuScenes that's the sample's scene token; for BDD it's the video/clip id.
3. **Split and sanity-check:**
   ```python
   from src.data.splits import split_by_scene, assert_no_scene_leakage, split_summary
   splits = split_by_scene(all_frames, seed=0)
   assert_no_scene_leakage(splits)      # must pass
   print(split_summary(splits))
   ```
4. **Look at your data.** Load 10 frames from train and display them in a notebook. Confirm they look like normal driving. This 10-minute step catches half of all "model bugs."
5. **Commit.** First green square. You now have a tested foundation and a loading dataset.

## The rule that matters most
Every time you create a split, call `assert_no_scene_leakage`. It's cheap and it makes the project's #1 failure mode impossible. Don't remove it "once things work" — keep it in the pipeline permanently.

## Then move to Phase 1
Once a dataset loads and splits cleanly, start the autoencoder baseline (Phase 1 in README.md). But honestly, because the embedding scorers are already built, you may find it faster to jump to Approach B first: extract features from a frozen backbone (torchvision ResNet or a DINO model), fit `kNNScorer` on normal train features, score val, and read the AUROC. That gives you a real number on day one or two.
