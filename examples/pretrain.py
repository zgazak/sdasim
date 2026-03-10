#!/usr/bin/env python3
"""Generate random pretraining scenes with custom distributions.

This script shows how to use SceneDistribution and random_scene()
for more control than the CLI `sdasim generate` subcommand offers.

Usage:
    python examples/pretrain.py
    python examples/pretrain.py --num-scenes 500 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sdasim
from sdasim.sampler import SceneDistribution, random_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pretraining scenes")
    parser.add_argument("-n", "--num-scenes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", default="output/pretrain")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Custom distribution: favor rate_sidereal, dim targets, small images
    dist = SceneDistribution(
        modes=["sidereal", "rate_track", "rate_sidereal"],
        mode_weights=[0.3, 0.3, 0.4],  # 40% rate_sidereal
        height=256,
        width=256,
        exposure_range=(1.0, 3.0),
        num_frames=8,
        psf_sigma_range=(0.4, 1.5),
        tracking_rate_range=(2.0, 10.0),
        n_on_rate_range=(1, 3),
        n_off_rate_range=(0, 3),
        target_mv_range=(12.0, 16.0),
        off_rate_speed_range=(2.0, 15.0),
        enable_shot_noise=True,
        enable_read_noise=True,
    )

    out = Path(args.output)
    mode_counts: dict[str, int] = {}

    for i in range(args.num_scenes):
        cfg = random_scene(dist=dist, seed=args.seed + i, device=args.device)
        scene = sdasim.Scene(cfg)

        # Track mode distribution
        mode_label = cfg.mode or (
            "sidereal" if cfg.star_motion.translation == [0.0, 0.0] else "rate_track"
        )
        mode_counts[mode_label] = mode_counts.get(mode_label, 0) + 1

        images, annotations = scene.render_sequence()
        scene_dir = out / f"scene_{i:04d}"
        sdasim.io.write_sequence(images, annotations, scene_dir, fmt="npy")

        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{args.num_scenes}] {mode_label}")

    print(f"\nDone: {args.num_scenes} scenes -> {out}/")
    print(f"Mode distribution: {mode_counts}")


if __name__ == "__main__":
    main()
