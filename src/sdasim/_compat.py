"""satsim configuration converter.

Maps satsim config dicts to sdasim SceneConfig. If the satsim config uses
$sample/$ref/$generator, calls satsim.config.loading.realize() first
(requires satsim installed). For flat configs, no satsim dependency needed.
"""

from __future__ import annotations

from typing import Any

from sdasim.config import (
    SceneConfig,
    SensorConfig,
    StarFieldConfig,
    StarMotionConfig,
    TargetConfig,
)
from sdasim.fpa import eod_to_sigma


def _has_dynamic_keys(d: Any) -> bool:
    """Check if a dict tree contains satsim dynamic keys ($sample, $ref, etc.)."""
    if isinstance(d, dict):
        for k, v in d.items():
            if k.startswith("$"):
                return True
            if _has_dynamic_keys(v):
                return True
    elif isinstance(d, list):
        return any(_has_dynamic_keys(item) for item in d)
    return False


def _get(d: dict, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dict."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
        if d is default:
            return default
    return d


def from_satsim_config(satsim_dict: dict, seed: int | None = None) -> SceneConfig:
    """Convert a satsim configuration dict to a sdasim SceneConfig.

    If the config contains dynamic keys ($sample, $ref, $generator), this
    function calls satsim.config.loading.realize() first.

    Args:
        satsim_dict: satsim configuration dictionary.
        seed: Random seed for config resolution.

    Returns:
        SceneConfig ready for Scene construction.
    """
    cfg = satsim_dict

    # Resolve dynamic keys if present
    if _has_dynamic_keys(cfg):
        try:
            from satsim.config.loading import realize
        except ImportError:
            raise ImportError(
                "satsim config contains dynamic keys ($sample/$ref/$generator) "
                "but satsim is not installed. Install it or use a flat config."
            )
        cfg = realize(cfg, seed=seed)

    # Extract FPA / sensor config
    fpa = _get(cfg, "fpa", default={})
    height = _get(fpa, "height", default=512)
    width = _get(fpa, "width", default=512)
    y_fov = _get(fpa, "y_fov", default=0.5)
    x_fov = _get(fpa, "x_fov", default=0.5)

    time_cfg = _get(fpa, "time", default={})
    exposure = _get(time_cfg, "exposure", default=2.0)
    gap = _get(time_cfg, "gap", default=0.5)

    num_frames = _get(fpa, "num_frames", default=1)
    zeropoint = _get(fpa, "zeropoint", default=23.5)

    # PSF: convert from EOD if Gaussian PSF specified
    psf_cfg = _get(fpa, "psf", default={})
    osf = _get(fpa, "s_osf", default=1)
    if "eod" in psf_cfg:
        psf_sigma = eod_to_sigma(psf_cfg["eod"], osf=1.0)  # native resolution
    elif "sigma" in psf_cfg:
        psf_sigma = psf_cfg["sigma"] / osf  # convert from oversampled to native
    else:
        psf_sigma = 1.5

    # Noise
    noise_cfg = _get(fpa, "noise", default={})
    read_noise = _get(noise_cfg, "read", default=10.0)
    electronic_noise = _get(noise_cfg, "electronic", default=5.0)

    # A2D
    a2d_cfg = _get(fpa, "a2d", default={})
    gain = _get(a2d_cfg, "gain", default=8.0)
    fwc = _get(a2d_cfg, "fwc", default=100000.0)
    a2d_bias = _get(a2d_cfg, "bias", default=500.0)
    a2d_dtype = _get(a2d_cfg, "dtype", default="uint16")

    # Dark current
    dark_current = _get(fpa, "dark_current", default=10.0)

    # Background
    bg_cfg = _get(cfg, "background", default={})
    background_mv = _get(bg_cfg, "galactic", "mv", default=21.0)
    if background_mv is None:
        background_mv = 21.0

    # Bias
    bias = _get(fpa, "bias", default=50.0)

    sensor = SensorConfig(
        height=height,
        width=width,
        y_fov=y_fov,
        x_fov=x_fov,
        exposure=exposure,
        gap=gap,
        num_frames=num_frames,
        zeropoint=zeropoint,
        psf_sigma=psf_sigma,
        dark_current=dark_current,
        read_noise=read_noise,
        electronic_noise=electronic_noise,
        background_mv=background_mv,
        bias=bias,
        gain=gain,
        fwc=fwc,
        a2d_bias=a2d_bias,
        a2d_dtype=a2d_dtype,
    )

    # Stars
    geom = _get(cfg, "geometry", default={})
    stars_cfg = _get(geom, "stars", default={})

    star_mode = _get(stars_cfg, "mode", default="bins")
    if star_mode == "random" or "mv" in stars_cfg:
        # Random bins mode
        mv_bins = _get(stars_cfg, "mv", "bins", default=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        density = _get(stars_cfg, "mv", "density", default=[1.0] * (len(mv_bins) - 1))
        stars = StarFieldConfig(mode="bins", mv_bins=mv_bins, density=density)
    elif star_mode == "sstrc7":
        stars = StarFieldConfig(
            mode="sstrc7",
            catalog_path=_get(stars_cfg, "path", default=None),
        )
    else:
        stars = StarFieldConfig(mode="bins")

    # Star motion
    star_motion_cfg = _get(geom, "star_motion", default={})

    rotation = _get(star_motion_cfg, "rotation", default=0.0)
    translation = _get(star_motion_cfg, "translation", default=[0.0, 0.0])
    t_osf = _get(cfg, "t_osf", default=_get(fpa, "t_osf", default=100))

    star_motion = StarMotionConfig(
        rotation=rotation,
        translation=translation,
        temporal_osf=t_osf,
    )

    # Targets (observation objects)
    obs_list = _get(geom, "obs", default=[])
    if not isinstance(obs_list, list):
        obs_list = [obs_list]

    targets = []
    for obs in obs_list:
        origin = _get(obs, "origin", default=[0.5, 0.5])
        velocity = _get(obs, "velocity", default=[0.0, 0.0])
        mv = _get(obs, "mv", default=12.0)
        mode = _get(obs, "mode", default="line")
        targets.append(TargetConfig(mode=mode, origin=origin, velocity=velocity, mv=mv))

    return SceneConfig(
        sensor=sensor,
        stars=stars,
        star_motion=star_motion,
        targets=targets,
        seed=seed,
        device="auto",
        enable_shot_noise=_get(noise_cfg, "photon", default=True),
        enable_read_noise=read_noise > 0 or electronic_noise > 0,
    )
