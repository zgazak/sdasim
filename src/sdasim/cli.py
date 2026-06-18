"""Command-line interface for sdasim.

Usage:
    sdasim render examples/rate_track.yaml -o output/rate_track/
    sdasim render examples/sidereal.yaml --fmt npy
    sdasim bench examples/speed_test.yaml --frames 500
    sdasim bench  # uses built-in defaults
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def cmd_render(args: argparse.Namespace) -> None:
    """Render a sequence from a config file and write to disk."""
    import sdasim

    print(f"Loading config: {args.config}")
    scene = sdasim.Scene.from_yaml(args.config)
    sensor = scene.sensor

    n = args.frames if args.frames is not None else sensor.num_frames
    print(
        f"Rendering {n} frames at {sensor.height}x{sensor.width} "
        f"({scene.star_positions.shape[0]} stars, "
        f"{len(scene.config.targets)} targets, "
        f"device={scene.device})"
    )

    t0 = time.perf_counter()
    images, annotations = scene.render_batch(list(range(n)))
    elapsed = time.perf_counter() - t0

    print(f"Rendered in {elapsed:.2f}s ({n / elapsed:.1f} fps)")

    out = Path(args.output)
    paths = sdasim.io.write_sequence(images, annotations, out, fmt=args.fmt)
    print(f"Wrote {len(paths)} files to {out}/")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate random pretraining scenes."""
    import sdasim
    from sdasim.sampler import SceneDistribution, random_scene

    dist = SceneDistribution(
        height=args.height,
        width=args.width,
        num_frames=args.frames_per_scene,
        enable_shot_noise=not args.no_noise,
        enable_read_noise=not args.no_noise,
    )

    out = Path(args.output)
    total_frames = 0

    for i in range(args.num_scenes):
        scene_seed = args.seed + i if args.seed is not None else None
        cfg = random_scene(dist=dist, seed=scene_seed, device=args.device)
        scene = sdasim.Scene(cfg)

        scene_dir = out / f"scene_{i:04d}"
        n = cfg.sensor.num_frames
        mode_label = cfg.mode or (
            "sidereal" if cfg.star_motion.translation == [0.0, 0.0] else "rate_track"
        )

        if i == 0 or (i + 1) % 10 == 0 or i == args.num_scenes - 1:
            print(
                f"[{i + 1}/{args.num_scenes}] mode={mode_label}, "
                f"{n} frames, {scene.star_positions.shape[0]} stars, "
                f"{len(cfg.targets)} targets"
            )

        images, annotations = scene.render_sequence()
        sdasim.io.write_sequence(images, annotations, scene_dir, fmt=args.fmt)
        total_frames += n

    print(f"Done: {args.num_scenes} scenes, {total_frames} total frames -> {out}/")


def cmd_bench(args: argparse.Namespace) -> None:
    """Benchmark rendering throughput (no disk I/O)."""
    import torch

    import sdasim

    if args.config:
        print(f"Loading config: {args.config}")
        scene = sdasim.Scene.from_yaml(args.config)
    else:
        # Built-in default: 256x256, moderate star field
        from sdasim.config import (
            SceneConfig,
            SensorConfig,
            StarFieldConfig,
            StarMotionConfig,
            TargetConfig,
        )

        cfg = SceneConfig(
            sensor=SensorConfig(height=256, width=256, y_fov=0.25, x_fov=0.25, exposure=1.0),
            stars=StarFieldConfig(
                mv_bins=[10, 11, 12, 13, 14, 15, 16, 17, 18],
                density=[5.0, 10.27, 24.33, 35.19, 60.02, 110.1, 180.3, 285.5],
            ),
            star_motion=StarMotionConfig(translation=[2.0, -3.0], temporal_osf=50),
            targets=[TargetConfig(origin=[0.5, 0.5], velocity=[5.0, 8.0], mv=13.0)],
            seed=7,
        )
        scene = sdasim.Scene(cfg)

    sensor = scene.sensor
    n = args.frames if args.frames is not None else sensor.num_frames
    n = max(n, 10)  # at least 10 frames for meaningful timing

    print(
        f"Benchmarking: {n} frames at {sensor.height}x{sensor.width}, "
        f"{scene.star_positions.shape[0]} stars, "
        f"t_osf={scene.config.star_motion.temporal_osf}, "
        f"device={scene.device}"
    )

    # Warmup
    scene.render(0)

    if scene.device.type == "cuda":
        import torch

        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for i in range(n):
        img, _ = scene.render(i)
    if scene.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    fps = n / elapsed
    ms = elapsed / n * 1000
    print(f"  {n} frames in {elapsed:.2f}s")
    print(f"  {fps:.1f} fps  ({ms:.1f} ms/frame)")

    if args.grad:
        sigma = torch.tensor(sensor.psf_sigma, requires_grad=True, device=scene.device)
        scene.render(0)  # warmup

        if scene.device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for i in range(n):
            img, _ = scene.render(i, psf_sigma=sigma)
            img.sum().backward()
        if scene.device.type == "cuda":
            torch.cuda.synchronize()
        grad_elapsed = time.perf_counter() - t0

        grad_fps = n / grad_elapsed
        grad_ms = grad_elapsed / n * 1000
        print(f"  with backward: {grad_fps:.1f} fps  ({grad_ms:.1f} ms/frame)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="sdasim",
        description="Speed-optimized differentiable satellite scene simulator",
    )
    sub = parser.add_subparsers(dest="command")

    # render
    p_render = sub.add_parser("render", help="Render a sequence and write to disk")
    p_render.add_argument("config", help="Path to YAML config file")
    p_render.add_argument(
        "-o", "--output", default="output", help="Output directory (default: output)"
    )
    p_render.add_argument(
        "-n", "--frames", type=int, default=None, help="Number of frames (default: from config)"
    )
    p_render.add_argument(
        "--fmt", default="npy", choices=["npy", "fits"], help="Output format (default: npy)"
    )

    # generate
    p_gen = sub.add_parser("generate", help="Generate random pretraining scenes")
    p_gen.add_argument(
        "-n", "--num-scenes", type=int, default=100, help="Number of scenes (default: 100)"
    )
    p_gen.add_argument("--seed", type=int, default=None, help="Base random seed")
    p_gen.add_argument("-o", "--output", default="output/pretrain", help="Output directory")
    p_gen.add_argument(
        "--fmt", default="npy", choices=["npy", "fits"], help="Output format (default: npy)"
    )
    p_gen.add_argument("--height", type=int, default=256, help="Image height (default: 256)")
    p_gen.add_argument("--width", type=int, default=256, help="Image width (default: 256)")
    p_gen.add_argument(
        "--frames-per-scene", type=int, default=8, help="Frames per scene (default: 8)"
    )
    p_gen.add_argument("--device", default="auto", help="Device (default: auto)")
    p_gen.add_argument("--no-noise", action="store_true", help="Disable shot and read noise")

    # bench
    p_bench = sub.add_parser("bench", help="Benchmark rendering throughput (no disk I/O)")
    p_bench.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to YAML config (optional, uses built-in default)",
    )
    p_bench.add_argument(
        "-n",
        "--frames",
        type=int,
        default=None,
        help="Number of frames (default: from config or 100)",
    )
    p_bench.add_argument("--grad", action="store_true", help="Also benchmark with backward pass")

    args = parser.parse_args(argv)

    if args.command == "render":
        cmd_render(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "bench":
        cmd_bench(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
