"""Sanity tests for scene-level splitting. Run: python scripts/test_splits.py

These prove the leakage guarantee holds and demonstrate why a naive
random split would fail. No external dependencies.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.splits import Frame, split_by_scene, assert_no_scene_leakage, split_summary
import random


def make_fake_frames(n_scenes=100, frames_per_scene=40):
    """Simulate a dataset: many scenes, each with many near-identical frames."""
    frames = []
    for s in range(n_scenes):
        for i in range(frames_per_scene):
            frames.append(Frame(path=f"scene{s:03d}/frame{i:03d}.jpg", scene_id=s))
    return frames


def test_no_leakage():
    frames = make_fake_frames()
    splits = split_by_scene(frames, val_frac=0.15, test_frac=0.15, seed=0)
    assert_no_scene_leakage(splits)          # raises if any scene crosses a boundary
    print("PASS: scene-level split has zero leakage")
    print(split_summary(splits))


def test_naive_split_would_leak():
    """Demonstrate the WRONG way, to make the failure concrete."""
    frames = make_fake_frames()
    rng = random.Random(0)
    shuffled = frames[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * 0.8)
    train = set(f.scene_id for f in shuffled[:cut])
    test = set(f.scene_id for f in shuffled[cut:])
    overlap = train & test
    # With random frame splitting, nearly every scene ends up in BOTH.
    print(f"\nNaive random frame split: {len(overlap)} of 100 scenes leak across train/test")
    print("(That is why we never split by frame. Each leaked scene = memorized test data.)")


def test_determinism():
    frames = make_fake_frames()
    a = split_by_scene(frames, seed=42)
    b = split_by_scene(frames, seed=42)
    assert [f.path for f in a["test"]] == [f.path for f in b["test"]]
    print("\nPASS: same seed gives identical splits (reproducible)")


if __name__ == "__main__":
    test_no_leakage()
    test_naive_split_would_leak()
    test_determinism()
    print("\nAll split checks passed.")
