"""Random scene generation for pretraining workflows.

Generates randomized SceneConfigs with physically plausible parameters.
Supports sidereal, rate_track, and rate_sidereal modes with configurable
distributions for all scene parameters.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from sdasim.config import (
    SceneConfig,
    SensorConfig,
    StarFieldConfig,
    StarMotionConfig,
    TargetConfig,
)


@dataclass
class SceneDistribution:
    """Distribution parameters for random scene generation.

    All range fields are (min, max) inclusive. Lists are sampled uniformly.
    """

    modes: list[str] = field(default_factory=lambda: ["sidereal", "rate_track", "rate_sidereal"])
    mode_weights: list[float] | None = None  # uniform if None

    height: int = 256
    width: int = 256
    y_fov: float = 0.5
    x_fov: float = 0.5

    exposure_range: tuple[float, float] = (0.5, 4.0)
    gap_range: tuple[float, float] = (0.1, 1.0)
    num_frames: int = 8
    zeropoint: float = 23.5
    psf_sigma_range: tuple[float, float] = (0.4, 2.0)

    dark_current_range: tuple[float, float] = (1.0, 20.0)
    read_noise_range: tuple[float, float] = (1.0, 15.0)
    electronic_noise_range: tuple[float, float] = (1.0, 10.0)
    background_mv_range: tuple[float, float] = (19.0, 22.0)
    bias: float = 50.0
    gain: float = 8.0
    fwc: float = 100000.0
    a2d_bias: float = 500.0
    a2d_dtype: str = "uint16"

    # Star field
    star_mv_bins: list[float] = field(
        default_factory=lambda: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    )
    star_density: list[float] = field(
        default_factory=lambda: [
            0.04,
            0.10,
            0.67,
            2.48,
            5.0,
            10.27,
            24.33,
            35.19,
            60.02,
            110.1,
            180.3,
            285.5,
        ]
    )

    # Motion
    tracking_rate_range: tuple[float, float] = (1.0, 15.0)
    temporal_osf: int = 100

    # Targets
    n_on_rate_range: tuple[int, int] = (0, 5)
    n_off_rate_range: tuple[int, int] = (0, 5)
    target_mv_range: tuple[float, float] = (10.0, 17.0)
    off_rate_speed_range: tuple[float, float] = (1.0, 20.0)
    on_rate_residual_range: tuple[float, float] = (0.0, 0.5)

    # Noise
    enable_shot_noise: bool = True
    enable_read_noise: bool = True

    # Empirical (calibration-driven) rendering — see SceneDistribution.from_calibration()
    psf_model: str = "gaussian"
    noise_model: str = "basic"
    empirical_psf_path: str | None = None
    empirical_noise_path: str | None = None
    empirical_streak_path: str | None = None
    psf_param_scale: float = 1.0
    streak_param_sample: bool = False
    streak_param_extend: float = 1.0
    apply_rowcol_median: bool = True

    @classmethod
    def from_calibration(cls, calib_dir: str, extend: float = 1.3, **overrides):
        """Build a distribution that generates data matching (and ~`extend`× past) a calibration
        produced by ``sdasim.calibrate``.

        Enables empirical PSF/noise/streak rendering + per-streak amplitude sampling, sets the
        gain and the streak-rate range from the measured distributions. Streak ORIENTATION stays
        uniform over the full rotation (the sampler draws velocity direction over 0..2π).
        """
        import json
        import os

        noise = json.load(open(os.path.join(calib_dir, "noise_model.json")))
        streak = json.load(open(os.path.join(calib_dir, "streak_model.json")))
        rate_m, rate_s = streak.get("dist_rate_px_s", [10.0, 5.0])
        lo, hi = max(0.5, rate_m - extend * rate_s), rate_m + extend * rate_s
        kw = dict(
            psf_model="empirical",
            noise_model="empirical",
            empirical_psf_path=os.path.join(calib_dir, "psf_basis.npz"),
            empirical_noise_path=os.path.join(calib_dir, "noise_model.json"),
            empirical_streak_path=os.path.join(calib_dir, "streak_model.json"),
            psf_param_scale=extend,
            streak_param_sample=True,
            streak_param_extend=extend,
            gain=float(noise.get("gain_e_per_adu", 1.0)),
            tracking_rate_range=(lo, hi),  # star-trail rate (rate_track mode)
            off_rate_speed_range=(
                lo,
                hi,
            ),  # target-streak rate (sidereal mode); orientation uniform
        )
        kw.update(overrides)
        return cls(**kw)


def random_scene(
    dist: SceneDistribution | None = None,
    seed: int | None = None,
    device: str = "auto",
) -> SceneConfig:
    """Generate a random SceneConfig for pretraining.

    Args:
        dist: Distribution parameters. Uses defaults if None.
        seed: Random seed for reproducibility. None for non-deterministic.
        device: Device string for the scene.

    Returns:
        A fully specified SceneConfig ready to construct a Scene.
    """
    if dist is None:
        dist = SceneDistribution()

    rng = random.Random(seed)

    # Pick mode
    if dist.mode_weights is not None:
        mode_choice = rng.choices(dist.modes, weights=dist.mode_weights, k=1)[0]
    else:
        mode_choice = rng.choice(dist.modes)

    # Sensor parameters
    exposure = rng.uniform(*dist.exposure_range)
    gap = rng.uniform(*dist.gap_range)
    psf_sigma = rng.uniform(*dist.psf_sigma_range)
    dark_current = rng.uniform(*dist.dark_current_range)
    read_noise = rng.uniform(*dist.read_noise_range)
    electronic_noise = rng.uniform(*dist.electronic_noise_range)
    background_mv = rng.uniform(*dist.background_mv_range)

    sensor = SensorConfig(
        height=dist.height,
        width=dist.width,
        y_fov=dist.y_fov,
        x_fov=dist.x_fov,
        exposure=exposure,
        gap=gap,
        num_frames=dist.num_frames,
        zeropoint=dist.zeropoint,
        psf_sigma=psf_sigma,
        dark_current=dark_current,
        read_noise=read_noise,
        electronic_noise=electronic_noise,
        background_mv=background_mv,
        bias=dist.bias,
        gain=dist.gain,
        fwc=dist.fwc,
        a2d_bias=dist.a2d_bias,
        a2d_dtype=dist.a2d_dtype,
        psf_model=dist.psf_model,
        noise_model=dist.noise_model,
        empirical_psf_path=dist.empirical_psf_path,
        empirical_noise_path=dist.empirical_noise_path,
        empirical_streak_path=dist.empirical_streak_path,
        psf_param_scale=dist.psf_param_scale,
        streak_param_sample=dist.streak_param_sample,
        streak_param_extend=dist.streak_param_extend,
        apply_rowcol_median=dist.apply_rowcol_median,
    )

    stars = StarFieldConfig(
        mode="bins",
        mv_bins=list(dist.star_mv_bins),
        density=list(dist.star_density),
    )

    # Random tracking direction + magnitude
    track_speed = rng.uniform(*dist.tracking_rate_range)
    track_angle = rng.uniform(0, 2 * math.pi)
    translation = [
        track_speed * math.cos(track_angle),
        track_speed * math.sin(track_angle),
    ]

    star_motion = StarMotionConfig(
        rotation=0.0,
        translation=translation,
        temporal_osf=dist.temporal_osf,
    )

    # Generate targets
    n_on = rng.randint(*dist.n_on_rate_range)
    n_off = rng.randint(*dist.n_off_rate_range)

    targets: list[TargetConfig] = []

    for _ in range(n_on):
        # On-rate targets: inertial velocity ≈ -translation (moves with tracked object)
        residual_r = rng.uniform(-dist.on_rate_residual_range[1], dist.on_rate_residual_range[1])
        residual_c = rng.uniform(-dist.on_rate_residual_range[1], dist.on_rate_residual_range[1])
        v_inertial = [-translation[0] + residual_r, -translation[1] + residual_c]

        targets.append(
            TargetConfig(
                mode="line",
                origin=[rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)],
                velocity=v_inertial,
                mv=rng.uniform(*dist.target_mv_range),
            )
        )

    for _ in range(n_off):
        # Off-rate targets: random inertial velocity (streaks in both modes)
        off_speed = rng.uniform(*dist.off_rate_speed_range)
        off_angle = rng.uniform(0, 2 * math.pi)
        v_inertial = [
            off_speed * math.cos(off_angle),
            off_speed * math.sin(off_angle),
        ]

        targets.append(
            TargetConfig(
                mode="line",
                origin=[rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)],
                velocity=v_inertial,
                mv=rng.uniform(*dist.target_mv_range),
            )
        )

    # Build config based on mode
    # For mode=None (sidereal/rate_track), convert inertial → sensor frame at config time.
    # For mode="rate_sidereal", store inertial velocities — scene.render() converts per-frame.
    if mode_choice == "sidereal":
        # Sidereal: stars fixed, targets streak at inertial velocity.
        # Set star_motion to zero, target velocity = inertial velocity (already set).
        star_motion = StarMotionConfig(
            rotation=0.0,
            translation=[0.0, 0.0],
            temporal_osf=1,
        )
        scene_mode = None
        sidereal_start = None

    elif mode_choice == "rate_track":
        # Rate-track: stars streak at translation, targets in sensor frame.
        # Convert target velocities: V_sensor = V_inertial + translation
        for tgt in targets:
            tgt.velocity = [
                tgt.velocity[0] + translation[0],
                tgt.velocity[1] + translation[1],
            ]
        scene_mode = None
        sidereal_start = None

    elif mode_choice == "rate_sidereal":
        # rate_sidereal: first N-K rate-track, last K sidereal.
        # Target velocities stay as inertial — scene.render() converts per-frame.
        sidereal_start = rng.randint(
            max(1, dist.num_frames // 2),
            max(1, dist.num_frames - 1),
        )
        scene_mode = "rate_sidereal"

    else:
        raise ValueError(f"Unknown mode: {mode_choice}")

    # Derive a torch seed from the config-level seed
    torch_seed = rng.randint(0, 2**31 - 1) if seed is not None else None

    return SceneConfig(
        sensor=sensor,
        stars=stars,
        star_motion=star_motion,
        targets=targets,
        seed=torch_seed,
        device=device,
        enable_shot_noise=dist.enable_shot_noise,
        enable_read_noise=dist.enable_read_noise,
        mode=scene_mode,
        sidereal_start=sidereal_start,
    )
