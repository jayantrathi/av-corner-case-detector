# Corner-Case Detection: RbA Investigation — Findings Summary

## One-line version (for a resume)

Built and rigorously evaluated an out-of-distribution hazard detector for autonomous-driving perception, benchmarking a rejection-based anomaly-scoring method (RbA) on frozen and outlier-exposure-finetuned segmentation backbones against a kNN embedding baseline on real-world hazard datasets with pixel/box-level ground truth (Lost & Found, CODA); diagnosed and confirmed an architecture-level localization limitation via controlled ablation against the method's own published checkpoint.

## Longer version (for an interview, or a project README)

Standard closed-set object detectors (YOLO, etc.) can only recognize the classes they were trained on — they have no mechanism to flag something genuinely unexpected on the road. This project investigates whether a frozen semantic segmentation model's own per-class confidence can be repurposed as an anomaly signal: RbA ("Rejected by All") scores a pixel as anomalous if every known class rejects it, requiring no synthetic anomaly data or retraining to use.

I implemented RbA on top of a frozen Mask2Former (Cityscapes-pretrained) and evaluated it against a kNN patch-embedding baseline I'd built earlier, using real pixel-level hazard masks from the Lost & Found dataset (German dashcam footage with genuine small lost-cargo hazards) and real box-level annotations from CODA. Both methods were evaluated on the identical held-out, scene-level train/test split to keep the comparison honest.

The zero-shot RbA result was weak (AUROC 0.79, AUPR 0.0065, 23% region hit rate) — well below the kNN baseline (AUROC 0.94, AUPR 0.083, 60% top-5 hit rate). Rather than accept that at face value, I diagnosed it: measured raw peak-score positions across the full held-out set and found the model's highest-confidence "anomaly" was almost always sitting on the image border or the ego-vehicle's hood, not on the real hazard — a receptive-field boundary effect. Hard-masking those regions out just relocated the false peak to the mask's edge (RbA's scores are smooth spatial gradients, not sharp spikes), so I replaced peak-selection with a local-contrast transform (subtracting a heavily-blurred version of the score map from itself) to suppress broad, low-frequency artifacts while preserving genuinely local anomalies.

To rule out "wrong checkpoint" as the root cause, I integrated the RbA authors' own officially released checkpoint — Swin-B, actually fine-tuned with COCO outlier-exposure supervision, reporting AP ~71 on Fishyscapes Lost & Found in their paper. This required a non-trivial cross-framework integration (the official code is Detectron2-based, not HuggingFace `transformers`; I patched a CPU fallback for a CUDA-only attention kernel to run it without a GPU). The properly-calibrated checkpoint improved global ranking quality (AUROC 0.79 → 0.875, AUPR 0.0065 → 0.0113) but did **not** fix localization (region hit rate actually dropped slightly, 23% → 17%) — and the same border artifact was still visibly present in every example frame. That's the real finding: the boundary effect is a structural property of the architecture's receptive field near image edges, not a symptom of an undercalibrated checkpoint. The remaining gap versus the paper's own published numbers is most plausibly a size-domain mismatch — COCO's outlier-exposure objects are typically much larger and more salient than Lost & Found's actual hazards (small lost cargo items), so the model learned to reject "weird COCO-sized things" rather than anything small and out of place.

## Results table

| Method | Pixel AUROC | Pixel AUPR | Localization metric |
|---|---|---|---|
| kNN patch-embedding baseline | 0.9439 | 0.0830 | top-5 hit rate: 60% (18/30) |
| RbA, zero-shot (vanilla Cityscapes ckpt) | 0.7917 | 0.0065 | region hit rate: 23% (7/30) |
| RbA, official (Swin-B, COCO outlier-exposure ckpt) | 0.8750 | 0.0113 | region hit rate: 17% (5/30) |

All numbers from the identical held-out scene-level split (30 hazard frames, Lost & Found).

## Key findings, stated plainly

1. **The simpler baseline wins.** The kNN embedding approach beats both RbA variants on every metric measured, including the properly-calibrated official checkpoint. Sophistication didn't pay off here — worth stating directly rather than hiding.
2. **The border artifact is architectural, not a checkpoint problem.** It persisted even with the authors' own outlier-exposure-trained weights, which rules out "our checkpoint just wasn't calibrated" as the explanation.
3. **Hard-masking known artifact regions is the wrong fix in general** when the underlying signal is a smooth gradient rather than a localized spike — local-contrast filtering is the more principled approach, and measurably outperformed raw masking even though it didn't fully close the gap.
4. **Object-size domain mismatch is a plausible, testable explanation** for why the official checkpoint's strong published numbers didn't transfer to this specific benchmark — Lost & Found's hazards are unusually small relative to COCO's typical outlier-exposure objects.

## What's not resolved

RbA-on-frozen-segmentation, even properly calibrated, underperforms the baseline for small-object hazard localization on this dataset. Untried next steps: fine-tuning on hazard-sized outlier exposure specifically (rather than relying on COCO's object-scale distribution), or combining RbA's pixel-level signal with the kNN detector's stronger localization as an ensemble.
