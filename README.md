# AV Corner-Case Detector — Build README

A start-to-finish guide for building an out-of-distribution / anomaly detector for autonomous-driving scenes. This is Project 1 of three portfolio builds. Drop this file into the project so every build session starts with full context.

---

## What you're building (and what you're deliberately NOT building)

**Building:** a system that looks at driving scenes and flags the ones that are *unusual* — out-of-distribution relative to normal driving. Debris in the lane, a wrong-way vehicle, an animal on the highway, unusual weather, a sensor going bad. The output is an anomaly score per frame (or per scene) plus a visualization of what triggered it.

**NOT building:** a self-driving stack. No planning, no control, no "my car drives itself." You rejected that for the right reason — it's a worse copy of what AV companies already have. This project's whole pitch is that it targets the part they *haven't* solved: knowing when the world has gone off-distribution and the autonomy stack should not be trusted.

**The one-sentence pitch for interviews:** "A perception-layer safety monitor that flags out-of-distribution driving scenes, because the dangerous failures in autonomy are the situations the model was never trained on."

---

## Why this is a strong portfolio piece

- **It's a real open problem.** OOD detection for autonomy is genuinely unsolved and actively researched. You're not competing with their product, you're working on their pain.
- **It's safety-framed.** Safety-critical thinking is exactly what perception teams screen for.
- **It's finishable.** Pure software, public datasets, runs on your own MacBook (Apple Silicon). No hardware to fail, no cluster to depend on.
- **It demos in 90 seconds.** Feed it a drive, watch the anomaly score spike on the weird frames, show the heatmap of what triggered it. That's a legible, visual demo.
- **It's honest.** You can state exactly what it does and doesn't do, which survives a technical grilling.

---

## The core technical idea

Anomaly detection has a precise meaning here: you learn what "normal" driving looks like from a large dataset of ordinary driving, then score how far a new frame departs from that learned normal. High departure = corner case.

There are three honest ways to frame it. Pick ONE as your primary; the others are extensions.

### Approach A — Reconstruction-based (autoencoder)
Train an autoencoder to compress and rebuild normal driving frames. It gets good at reconstructing normal scenes and bad at reconstructing things it never saw. High reconstruction error = anomaly.
- **Pros:** conceptually clean, easy to explain, easy to visualize (the error heatmap literally shows *where* the weird thing is).
- **Cons:** can be fooled (sometimes reconstructs anomalies too well); raw-pixel reconstruction error is noisy.
- **Good for:** your first working version and the clearest demo visual.

### Approach B — Feature-density / embedding-based (recommended primary)
Take a pretrained vision backbone (e.g. a frozen ResNet/DINO/CLIP image encoder), extract feature embeddings for lots of normal frames, and model that feature distribution (k-NN distance, Mahalanobis distance, or a simple density model). A new frame far from the normal feature cloud is anomalous.
- **Pros:** state-of-the-art-ish for OOD, robust, doesn't need you to train a big model from scratch, fast to iterate.
- **Cons:** less of a "look what I trained" story, more of a "look what I engineered" story (which is fine and arguably more senior).
- **Good for:** your strongest, most defensible primary approach.

### Approach C — Supervised on labeled rare events
If your dataset labels rare/unusual objects or scenarios, train a classifier or detector to spot them directly.
- **Pros:** concrete, high accuracy on the specific things you labeled.
- **Cons:** only catches anomalies you have labels for, which defeats half the point (real corner cases are the ones nobody labeled). Also label-hungry.
- **Good for:** a supporting evaluation, not the core.

**Recommendation:** primary = Approach B (embedding-based). Build Approach A first as a warm-up because its visual is the best for demos, then move to B for the real result, and report B as your headline with A's heatmap as the visualization. Mention C as future work.

---

## Compute: this runs on your MacBook (Apple Silicon)

You're building this on your own MacBook Pro with Apple Silicon, and it can handle the full, deep version of this project. The laptop is **not** a reason to reduce scope, and nothing in this README shrinks the project to fit the hardware. Build it as ambitious and rigorous as you want. Here's the honest picture of what the hardware does and doesn't constrain:

- **The one real constraint is training a large model from scratch via backprop** — that's the only operation where a laptop is meaningfully slower than a GPU cluster. And this project barely touches it. Your primary method (Approach B) *freezes* the backbone and only runs forward passes to extract embeddings. So even a deep, full-scale, months-long version of this is mostly forward passes plus CPU scoring, which Apple Silicon runs comfortably. Depth and "runs on my Mac" are not in tension here.
- **Use the MPS backend** for the forward passes: `device = "mps" if torch.backends.mps.is_available() else "cpu"`. Real acceleration on Apple Silicon. A few ops occasionally fall back to CPU with a warning — normal, ignore it.
- **Approach A (autoencoder) is the only piece that trains a model.** This is the one spot the from-scratch constraint applies. It's a warm-up and a demo visual, so it doesn't need to be huge — but if you *want* it deep, you can, just expect training to take longer and lean on smaller resolution / MPS to keep it tractable. Your call, not a limit imposed by the hardware.
- **Data scale is your choice, not the laptop's.** You can run the full dataset if you want the rigor; you can also work from a large representative subset if you'd rather iterate faster. Both are legitimate. If you subset, keep scenes diverse and the scene-level split intact. This is a speed/rigor tradeoff *you* control, not a ceiling — full-scale is on the table.
- **Cache your embeddings to disk** (`.npy`). Extract features once, then iterate on scoring and evaluation instantly without re-running the backbone. This is the single habit that makes even a large project feel fast to work on. It's about iteration speed, not scope.

Net: go deep. The MacBook comfortably runs the serious version of this because the heavy model is frozen. The only thing to be deliberate about is from-scratch training (Approach A), and even that is fine if you keep an eye on resolution and use MPS.

---

## Dataset choice

You need lots of *normal* driving plus some genuinely weird scenes to test against. Options:

| Dataset | Why | Watch out for |
|---|---|---|
| **nuScenes** | Rich (camera + LiDAR + radar), well-documented, widely recognized by AV people, has diverse scenes. Strong resume signal. | Large download; devkit has a learning curve. |
| **KITTI** | Classic, smaller, easy to start, everyone knows it. | Older, less diverse, mostly "normal" so you supply your own anomalies. |
| **Berkeley DeepDrive (BDD100K)** | Huge, very diverse (weather, time of day, scene types), image-focused, great for "normal distribution." | Big; annotation format quirks. |
| **CODA / corner-case datasets** | Purpose-built collections of driving corner cases — ideal as your *test* set of known-weird scenes. | Smaller; use as evaluation, not training. |

**Recommended combo:** learn "normal" from **BDD100K** (diversity makes the normal model robust) or **nuScenes** (if you want the multi-sensor prestige), and evaluate on a **corner-case set (like CODA)** plus held-out normal frames. That split — normal-for-training, corner-cases-for-testing — is exactly the right experimental design and shows you understand the problem.

Confirm licensing/access for each; most are free for research with registration.

---

## Milestone plan (finishable by December)

Rough five-phase arc. Each phase ends with something that works, so you're never far from a demo.

### Phase 0 — Setup (a few days)
- Environment built on your MacBook (Apple Silicon): PyTorch with MPS support, the dataset tools. No CUDA — you'll use `device="mps"`.
- One dataset downloaded and loading. You can display an image and its metadata.
- Repo created, README (this file) committed. First green square.
- **Done when:** you can load and visualize frames in a notebook.

### Phase 1 — Baseline reconstruction detector (Approach A) (~1 week)
- Train a convolutional autoencoder on normal frames.
- Compute reconstruction error; visualize the error heatmap.
- Eyeball it on a few obvious anomalies.
- **Done when:** weird frames score higher than normal ones, and you have a heatmap image.

### Phase 2 — Embedding-based detector (Approach B) (~1–2 weeks)
- Extract embeddings from a frozen pretrained backbone for many normal frames.
- Fit a density/distance model (start with k-NN distance or Mahalanobis).
- Score test frames; this becomes your primary method.
- **Done when:** B outperforms A on your test set and you can show the numbers.

### Phase 3 — Proper evaluation (~1 week)
- Build the evaluation set: held-out normal + known corner cases.
- Compute real metrics: AUROC, AUPR, and a threshold analysis (false-positive rate at a chosen operating point). These are the standard OOD-detection metrics and using them signals you know the field.
- Compare A vs B honestly in a table.
- **Done when:** you have a metrics table and ROC curves.

### Phase 3.5 — Downstream-impact evaluation (OPTIONAL, the depth extension) (~1 week)
This is the answer to "isn't just anomaly detection a bit thin?" — the way to add real depth **without** building a self-driving stack and without inviting the "you made a worse Waymo" comparison. The claim you're proving here is: *my detector catches the scenes where a downstream driving model would fail.* That's a senior-level results story.

**The key idea:** you do NOT build autonomy. You take some *existing, off-the-shelf* driving model as a measuring stick, and you show that its failures line up with your anomaly scores. Your detector stays the star; the driving model is just the thing you measure against.

Three ways to do this, cheapest first — pick based on time:

- **Option 1 (cheapest, recommended): proxy via a pretrained perception model's confidence/error.** Run an off-the-shelf detector/segmenter (e.g. a pretrained object detector or a pretrained lane/segmentation model) on your test frames. On corner-case frames it will fail more — miss objects, produce low-confidence or nonsensical output. Show that *its failure/error correlates with your anomaly score*. Headline result: "on the 10% of frames my detector flags as most anomalous, the downstream perception model's error is Nx higher." No autonomy built, pure measurement.
- **Option 2 (medium): a trivial driving policy in simulation as the stick.** If you want a "policy fails" story, use a minimal existing policy (even a lane-follower) in a sim like CARLA purely as a failure sensor — measure how often it messes up, and show those moments are the ones your detector flagged. The policy is deliberately dumb and off-the-shelf; it is NOT your contribution and you say so explicitly.
- **Option 3 (only if a labeled set exists): correlate with logged disengagements/failures.** Some datasets tag hard/failure frames. Show your score predicts them.

**Framing discipline (critical):** always describe this as "I quantified my detector's value to a downstream stack," never "I built a self-driving system." The driving model is a ruler, not a deliverable. If asked in an interview, be crisp: "I didn't build the autonomy — I used a pretrained model as a stand-in to measure whether my monitor catches the situations that break perception. It does, by [factor]."

- **Done when:** you have a result of the form "downstream failure rate is X× higher on frames my detector flags," plotted, on a clean scene-level split.
- **Skip it if:** you're tight on time. Phases 0–4 alone are a complete, strong project. This is the depth *option*, not a requirement — and it must never delay shipping the core.

### Phase 4 — Demo + polish (~1 week)
- Build the demo: run the detector across a driving clip, plot the anomaly score over time, and flag/visualize the frames that spike. If you did Phase 3.5, the demo's punchline becomes "...and these flagged frames are exactly where the downstream model fails," which is much stronger.
- Record the 90-second video. Write the README results section. Clean the repo.
- **Done when:** the demo video exists and the repo is presentable.

Total: roughly 5–6 weeks for the core (Phases 0–4), plus ~1 week if you take the Phase 3.5 depth extension. Fits before December with margin for debugging. Ship the core first; add 3.5 only once 0–4 are done.

---

## Repository structure

```
av-corner-case-detector/
├── README.md                 # project overview + results (this becomes public-facing)
├── requirements.txt
├── configs/                  # yaml configs for experiments
├── data/                     # (gitignored) dataset lives here
│   └── .gitkeep
├── src/
│   ├── data/                 # dataset loading, splits, transforms
│   ├── models/               # autoencoder, embedding extractor
│   ├── scoring/              # anomaly scoring (recon error, kNN, Mahalanobis)
│   ├── eval/                 # metrics (AUROC/AUPR), curves
│   └── viz/                  # heatmaps, score-over-time plots
├── notebooks/                # exploration, sanity checks
├── scripts/                  # train.py, evaluate.py, run_demo.py
├── results/                  # metrics tables, figures, saved outputs
└── demo/                     # the demo clip + output video
```

Keep `data/` out of git (it's huge). Everything else is versioned.

---

## The senior-engineer discipline (this is where rookie mistakes hide)

These are the things that separate a portfolio project that impresses from one that quietly embarrasses. The companion skill enforces these during the build, but know them going in:

1. **Split your data by scene/drive, never by random frame.** Adjacent frames in a driving clip are nearly identical. A random train/test split leaks almost-copies across the boundary and gives you a beautiful, meaningless score. Split so that whole drives/scenes are entirely in train OR entirely in test. This is the single most common way people fool themselves in this exact domain.

2. **Establish a baseline before anything fancy.** The dumbest reasonable method (e.g. k-NN distance on raw pretrained features) is your baseline. If your fancy model can't beat it, the fancy model is worthless. Always have a number to beat.

3. **Look at your data before you model it.** Plot frames. Look at the anomalies you're testing against. Half the "bugs" in ML are actually dataset misunderstandings you'd catch in ten minutes of looking.

4. **Normal must actually be normal.** If your "normal" training set secretly contains corner cases, your detector learns to treat them as normal. Sanity-check what's in your training distribution.

5. **Use the field's real metrics.** AUROC and AUPR, not accuracy. Anomalies are rare, so accuracy is meaningless (a detector that flags nothing is 99% "accurate"). Using the right metrics signals you know the literature.

6. **Fix your random seeds and log your configs.** Every run reproducible. When a result surprises you, you need to be able to trust it and rerun it.

7. **Don't tune on your test set.** Keep a validation split for choosing thresholds and hyperparameters. Touch the test set once, at the end. Otherwise your reported numbers are inflated and you won't even know it.

8. **Version the demo footage separately.** The clip you demo on should be one the model never trained on. Obvious, easy to mess up under deadline pressure.

---

## What "done" looks like

- A repo with clean code and a README that a stranger understands in two minutes.
- A metrics table: Approach A vs B, AUROC/AUPR, on a proper scene-split evaluation.
- A 90-second demo video: anomaly score over a drive, spiking on the weird frames, with heatmaps.
- The honest one-paragraph description of what it does, its limitations, and what you'd do next.

That package is genuinely interview-strong. It says: *this person understands autonomy's real failure modes, knows OOD detection, does rigorous evaluation, and ships.*

---

## First session kickoff (what to tell Claude when you start building)

Paste this into the first build session in the new project:

> I'm starting Project 1, the AV corner-case detector. I'm building on my own MacBook Pro (Apple Silicon, so PyTorch MPS), I'm comfortable in PyTorch, and this needs to be finishable by December and demo well. I want to build the deep, rigorous version — the laptop can handle it since the backbone is frozen. Let's start at Phase 0. Help me decide between nuScenes and BDD100K for the normal distribution, get my environment set up with MPS, and get a dataset loading and visualizing in a notebook. Hold me to the senior-engineer discipline in the README — especially the scene-level data split.

Then work the phases in order. Don't skip the baseline. Don't skip looking at the data.
