"""Streamlined configuration dataclasses and YAML loader.

All values are concrete — no $sample/$ref/$generator resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SensorConfig:
    height: int = 512
    width: int = 512
    y_fov: float = 0.5
    x_fov: float = 0.5
    exposure: float = 2.0
    gap: float = 0.5
    num_frames: int = 1
    zeropoint: float = 23.5
    psf_sigma: float = 1.5
    dark_current: float = 10.0
    read_noise: float = 10.0
    electronic_noise: float = 5.0
    is_cmos: bool = False
    background_mv: float = 21.0
    bias: float = 50.0
    gain: float = 8.0
    fwc: float = 100000.0
    a2d_bias: float = 500.0
    a2d_dtype: str = "uint16"
    # --- empirical (opt-in) rendering; defaults preserve the Gaussian/basic path ---
    psf_model: str = "gaussian"  # {"gaussian", "empirical"}
    noise_model: str = "basic"  # {"basic", "empirical"}
    empirical_psf_path: str | None = None  # path to psf_basis.npz
    empirical_noise_path: str | None = None  # path to noise_model.json
    empirical_streak_path: str | None = None  # path to streak_model.json (along-track texture)
    psf_param_scale: float = (
        1.0  # sample size/ellipticity from real dist; >1 extrapolates; 0 = mean PSF
    )
    psf_speckle_amp: float = 0.0  # atmospheric-boiling streak STRUCTURE amplitude (0 = off)
    speckle_corr_px: float = 4.0  # along-track correlation length of the speckle boil
    speckle_modes: int = 6  # number of eigen-PSF modes used for speckle
    apply_rowcol_median: bool = True  # row/col-median subtract (senpai stage). A FULL-FRAME op;
    #                                      set False when rendering a small patch/cutout (else an
    #                                      axis-aligned streak digs a trench in the patch median).
    streak_param_sample: bool = False  # draw streak intensity/wobble amplitudes per render from
    #                                      the measured distributions (streak_model.json dist_*)
    streak_param_extend: float = (
        1.0  # std multiplier for that sampling (covers + extends past measured)
    )


@dataclass
class StarFieldConfig:
    mode: str = "bins"
    mv_bins: list[float] = field(
        default_factory=lambda: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    )
    density: list[float] = field(
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
    # For catalog modes (sstrc7/gaia)
    catalog_path: str | None = None
    ra: float = 0.0
    dec: float = 0.0
    rot: float = 0.0
    pad_mult: float = 1.0
    mv_max: float | None = None


@dataclass
class StarMotionConfig:
    rotation: float = 0.0
    translation: list[float] = field(default_factory=lambda: [0.0, 0.0])
    temporal_osf: int = 100


@dataclass
class TargetConfig:
    mode: str = "line"
    origin: list[float] = field(default_factory=lambda: [0.5, 0.5])
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0])
    mv: float = 12.0


@dataclass
class SiteConfig:
    latitude: float = 0.0   # degrees
    longitude: float = 0.0  # degrees
    altitude: float = 0.0   # meters


@dataclass
class ObjectConfig:
    norad_id: str = ""
    ra: float | None = None        # degrees
    dec: float | None = None       # degrees
    ra_rate: float | None = None   # deg/s
    dec_rate: float | None = None  # deg/s
    mv: float = 12.0
    dither_arcsec: float = 0.0


@dataclass
class SceneConfig:
    sensor: SensorConfig = field(default_factory=SensorConfig)
    stars: StarFieldConfig = field(default_factory=StarFieldConfig)
    star_motion: StarMotionConfig = field(default_factory=StarMotionConfig)
    targets: list[TargetConfig] = field(default_factory=list)
    seed: int | None = None
    device: str = "auto"
    enable_shot_noise: bool = True
    enable_read_noise: bool = True
    mode: str | None = None
    sidereal_start: int | None = None
    obs_time: str | None = None  # ISO start time, e.g. "2024-01-01T00:00:00"
    # --- orbital scene mode (additions) ---
    site: SiteConfig | None = None
    objects: list[ObjectConfig] = field(default_factory=list)
    tracking: str | None = None


_NESTED_TYPES: dict[str, type] = {}


def _init_nested_types() -> None:
    """Populate the nested type lookup (called after all dataclasses are defined)."""
    if not _NESTED_TYPES:
        _NESTED_TYPES.update(
            {
                "SensorConfig": SensorConfig,
                "StarFieldConfig": StarFieldConfig,
                "StarMotionConfig": StarMotionConfig,
                "TargetConfig": TargetConfig,
                "SiteConfig": SiteConfig,
                "ObjectConfig": ObjectConfig,
            }
        )


# Map field names to their nested dataclass types
_FIELD_TYPE_MAP: dict[str, type] = {
    "sensor": SensorConfig,
    "stars": StarFieldConfig,
    "star_motion": StarMotionConfig,
    "site": SiteConfig,
}


def _dataclass_from_dict(cls, data: dict) -> Any:
    """Recursively build a dataclass from a dict."""
    if not isinstance(data, dict):
        return data
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, val in data.items():
        if key in fields:
            if isinstance(val, dict) and key in _FIELD_TYPE_MAP:
                kwargs[key] = _dataclass_from_dict(_FIELD_TYPE_MAP[key], val)
            elif isinstance(val, list) and key == "targets":
                kwargs[key] = [_dataclass_from_dict(TargetConfig, t) for t in val]
            elif isinstance(val, list) and key == "objects":
                kwargs[key] = [_dataclass_from_dict(ObjectConfig, o) for o in val]
            else:
                kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str | Path) -> SceneConfig:
    """Load a SceneConfig from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return _dataclass_from_dict(SceneConfig, data)
