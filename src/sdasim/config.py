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
    background_mv: float = 21.0  # sky surface brightness, mag/arcsec^2
    bias: float = 50.0
    gain: float = 8.0
    fwc: float = 100000.0
    flat_radiance: float = 0.0
    a2d_bias: float = 500.0
    a2d_dtype: str = "uint16"
    # Frame-to-frame pointing wander, per-axis RMS arcsec. Each
    # frame gets one random rigid offset of the whole field (stars + targets),
    # the short-exposure-stack jitter signature. WITHIN-frame smear is not
    # simulated here — fold it into psf_sigma in quadrature as v1 did.
    jitter_rms_arcsec: float = 0.0
    # --- empirical (opt-in) rendering; defaults preserve the Gaussian/basic path ---
    psf_model: str = "gaussian"  # {"gaussian", "empirical", "pupil"}
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
    # Motion-blur sub-sampling for STARS: how many stationary sub-sources a
    # moving star becomes. None (default) = auto — sdasim derives it per frame
    # from the streak length so sub-source spacing keeps sampling artifacts
    # below the frame's own noise floor (see render.motion_sampling_spacing).
    # An explicit int pins the value verbatim (byte-reproducible renders,
    # sigma-differentiable training).
    temporal_osf: int | None = None


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
class CatalogConfig:
    """Auto-discovery of field satellites streaking through the FOV.

    When enabled, the whole Space-Track catalog is propagated to each frame's
    epoch, satellites whose streak touches the (sunlit) FOV are found, and they
    are rendered as line targets. Requires ``site`` to be set.
    """

    enabled: bool = False
    source: str = "spacetrack"        # "spacetrack" | path to a local 2-/3-line file
    env_path: str | None = None       # .env holding SPACETRACK_USERNAME / SPACETRACK_PASSWORD
    default_diameter: float = 1.0     # m, fallback size when RCS is unknown
    albedo: float = 0.2               # geometric albedo for the Lambertian-sphere mv
    psf_pad_px: float | None = None   # FOV entry/exit pad; default 3*psf_sigma


@dataclass
class SegmentConfig:
    """One circular segment of a segmented aperture (positions/diameter in meters)."""

    x: float = 0.0  # m, center offset from array center (+x = +column direction)
    y: float = 0.0  # m
    d: float = 0.5  # m, segment diameter


@dataclass
class ApertureConfig:
    """Aperture geometry. Drives collecting area, zeropoint
    resolution (with ``mirror`` + ``band``), and the optional diffraction PSF
    (``sensor.psf_model='pupil'``).
    """

    enabled: bool = False
    shape: str = "circle"  # circle | ellipse | rect | segmented
    diameter: float = 1.0  # m: circle diameter; ellipse major axis; rect width
    diameter_minor: float | None = None  # m: ellipse minor / rect height (None -> diameter)
    obscuration: float = 0.0  # linear ratio (circle central hole); ellipse/rect: unused
    segments: list[SegmentConfig] = field(default_factory=list)  # shape='segmented'
    phased: bool = True  # segmented: co-phased (full-array core) vs incoherent co-add
    piston_rms_nm: float = 0.0  # phased segmented: residual piston OPD, nm RMS
    tiptilt_rms_arcsec: float = 0.0  # segmented: per-axis segment tip/tilt, image-space arcsec
    gap_fraction: float = 0.0  # segmented: gap/pitch ratio -> Strehl (1-w)^4 bookkeeping


@dataclass
class MirrorConfig:
    """Mirror surface quality + coating.

    Specular throughput per reflection is R_coating * exp(-(4*pi*sigma/lambda)^2)
    (Bennett-Porteus); the scattered TIS fraction refills a sky-following
    background term (near-specular roughness scatter), replacing the v1
    hand-computed background_mv/zeropoint debits.
    """

    coating: str = "bare_al"  # bare_al | protected_al | protected_ag | gold
    roughness_nm: float = 9.0  # RMS micro-roughness (v1: porosity-pitted bare SPDT Al)
    n_reflections: int = 1  # powered/fold mirror count in the train


@dataclass
class BandConfig:
    """Observing band. When enabled, sensor.zeropoint is COMPUTED
    from aperture area x mirror throughput x qe x band photon flux (bands.py) and
    overwrites any YAML value; star magnitudes come from the sstrc7 SED
    interpolated to the band center; catalog satellite photometry uses the
    band's solar magnitude.
    """

    enabled: bool = False
    name: str = "V"  # V | R | I | J | H | KS | SWIR_YJH (bands.BANDS keys)
    qe: float = 0.68  # detector QE x non-mirror optics (corrector, window, filter)
    airmass: float = 1.3  # for the default atmospheric transmission term


@dataclass
class SkyConfig:
    """Sky background model.

    'static' keeps the v1 behavior (sensor.background_mv, frozen at construction,
    scatter-brightened when band+mirror are enabled). 'auto' derives the sky
    per frame from sun position (site + obs_time + pointing) via sky.py:
    night / twilight (Patat 2006) / daytime (elongation-dependent MODTRAN anchors).
    """

    model: str = "static"  # static | auto
    aerosol_delta_mag: float = 0.0  # positive = brighter sky, all regimes
    elongation_deg: float | None = None  # None -> computed from pointing vs sun


@dataclass
class AberrationsConfig:
    """Static wavefront aberrations in waves RMS (Noll-normalized Zernikes).

    These are the numbers an interferometer or a Zemax "Zernike Standard"
    analysis reports (RMS-normalized circular-pupil coefficients — NOT Fringe
    Zernikes). Applied once per Scene; commanded per-frame defocus rides on
    top via the ``defocus_um`` render override.
    """

    defocus: float = 0.0  # Z4: residual best-focus error
    astig_0: float = 0.0  # Z6: elongates along rows/cols; axis flips across focus
    astig_45: float = 0.0  # Z5: same, along the diagonals
    coma_x: float = 0.0  # Z8: comatic flare along columns
    coma_y: float = 0.0  # Z7: comatic flare along rows
    spherical: float = 0.0  # Z11: makes intra/extra donut profiles differ
    # Extra defocus (waves RMS) at the field CORNER, growing quadratically with
    # boresight-relative field radius — field curvature / focal-plane bow.
    field_curvature: float = 0.0
    # Field-linear coma: waves RMS at the field CORNER, oriented radially and
    # growing linearly with field radius — the classic "comatic corners"
    # pattern (clean core on-axis, outward-pointing tails at the edges).
    # Positive values point the tails away from the boresight. Distinct from
    # coma_x/coma_y, which are field-CONSTANT (decenter/tilt miscollimation:
    # every star gets the same tail).
    coma_radial: float = 0.0
    # Field-quadratic astigmatism: waves RMS at the field CORNER, growing as
    # field radius squared with the doubled-angle orientation rule (the astig
    # axis tracks the field azimuth: corner stars elongate radially on one side
    # of focus, tangentially on the other). This is the intrinsic off-axis
    # astigmatism every telescope has when aligned — the dominant field
    # aberration of RC-style designs. Combined with the field-CONSTANT
    # astig_0/astig_45 (miscollimation), it produces binodal astigmatism: two
    # off-center points of zero astigmatism.
    astig_radial: float = 0.0


@dataclass
class OpticsConfig:
    """Pupil geometry for the geometric defocus/aberration model (optics.py).

    Disabled by default; when disabled the render path is untouched. Enabling
    with zero aberrations and ``defocus_um=0`` is also free — the model
    collapses to the plain Gaussian PSF until something actually blurs.
    """

    enabled: bool = False
    f_number: float = 8.0  # working f-number of the optical system
    obscuration: float = 0.0  # secondary diameter / aperture diameter (0 = clear)
    wavelength_nm: float = 550.0  # converts waves RMS <-> microns of wavefront
    pixel_pitch_um: float = 9.0  # detector pixel pitch, microns
    # --- cost ceiling for the defocus expansion --------------------------------
    # None (default) inherits the scene-level max_render_seconds; set here only
    # to give the optics stage its own ceiling. The budget is converted to a
    # sub-source cap via a measured splat rate; when it binds, pupil sampling is
    # coarsened (donuts get grainy; flux and moments stay exact) and metadata
    # flags pupil_sampling_clamped.
    max_render_seconds: float | None = None
    # Explicit sub-source cap. Overrides max_render_seconds when set — use it
    # when a run must render identically across machines, since a calibrated
    # budget is by definition machine-dependent.
    max_sub_sources: int | None = None
    aberrations: AberrationsConfig = field(default_factory=AberrationsConfig)


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
    # Wall-clock ceiling for sub-source work (motion expansion + defocus pupil
    # expansion combined). Converted to a sub-source cap via a splat-throughput
    # measurement on this machine at Scene construction. Motion sampling is
    # served first (beading is more visible than donut graininess); the defocus
    # stage then divides the same pool by the already-expanded source count.
    # Degradations are flagged in render metadata.
    max_render_seconds: float = 0.5
    # Sub-source spacing pin, in units of psf_sigma. None (default) = auto:
    # derived per frame from the brightest source vs the frame's noise floor so
    # sampling ripple stays statistically invisible. Set a float (e.g. 1.5) to
    # pin the spacing for reproducibility across scenes/frames.
    sampling_quality: float | None = None
    mode: str | None = None
    sidereal_start: int | None = None
    obs_time: str | None = None  # ISO start time, e.g. "2024-01-01T00:00:00"
    # --- orbital scene mode (additions) ---
    site: SiteConfig | None = None
    objects: list[ObjectConfig] = field(default_factory=list)
    tracking: str | None = None
    # --- catalog discovery + explicit mount rate (additions) ---
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    mount_ra_rate: float = 0.0   # deg/s, inertial; 0.0 == sidereal track
    mount_dec_rate: float = 0.0  # deg/s
    # --- geometric defocus/aberration model (additions) ---
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    # --- aperture/mirror/band/sky (additions) ---
    aperture: ApertureConfig = field(default_factory=ApertureConfig)
    mirror: MirrorConfig = field(default_factory=MirrorConfig)
    band: BandConfig = field(default_factory=BandConfig)
    sky: SkyConfig = field(default_factory=SkyConfig)


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
                "CatalogConfig": CatalogConfig,
                "OpticsConfig": OpticsConfig,
                "AberrationsConfig": AberrationsConfig,
            }
        )


# Map field names to their nested dataclass types
_FIELD_TYPE_MAP: dict[str, type] = {
    "sensor": SensorConfig,
    "stars": StarFieldConfig,
    "star_motion": StarMotionConfig,
    "site": SiteConfig,
    "catalog": CatalogConfig,
    "optics": OpticsConfig,
    "aberrations": AberrationsConfig,
    "aperture": ApertureConfig,
    "mirror": MirrorConfig,
    "band": BandConfig,
    "sky": SkyConfig,
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
            elif isinstance(val, list) and key == "segments":
                kwargs[key] = [_dataclass_from_dict(SegmentConfig, s) for s in val]
            else:
                kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str | Path) -> SceneConfig:
    """Load a SceneConfig from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return _dataclass_from_dict(SceneConfig, data)
