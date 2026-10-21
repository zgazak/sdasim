"""Integration tests for the aperture/mirror/band/sky extensions.

Covers the wiring of aperture/mirror/band/sky config sections into Scene:
zeropoint resolution (v1 anchors), scatter background brightening, sun-driven
per-frame sky, per-frame exposure override (short-exposure stacks), frame-to-
frame jitter, the pupil-kernel PSF path, the high-flux Poisson switchover, and
daytime saturation behavior. Module-level physics is tested in test_bands.py /
test_sky.py / test_pupil.py; these tests exercise the Scene plumbing.
"""

from __future__ import annotations

import math

import pytest
import torch

from sdasim.config import (
    ApertureConfig,
    BandConfig,
    MirrorConfig,
    SceneConfig,
    SegmentConfig,
    SensorConfig,
    SiteConfig,
    SkyConfig,
    StarFieldConfig,
    StarMotionConfig,
    TargetConfig,
    load_config,
)
from sdasim.noise import poisson_noise
from sdasim.scene import Scene

MAUI = SiteConfig(latitude=20.82028, longitude=-156.27944, altitude=1000.0)


def _base_config(**kw) -> SceneConfig:
    """Small, fast scene: random star bins, one static target, no noise."""
    cfg = SceneConfig(
        sensor=SensorConfig(
            height=128,
            width=128,
            y_fov=0.05,
            x_fov=0.05,
            exposure=1.0,
            gap=0.5,
            num_frames=2,
            zeropoint=25.0,
            psf_sigma=1.5,
            dark_current=0.01,
            read_noise=1.6,
            electronic_noise=1.0,
            background_mv=21.0,
            gain=0.31,
            fwc=20000.0,
        ),
        stars=StarFieldConfig(mode="bins", pad_mult=0.0),
        star_motion=StarMotionConfig(),
        targets=[TargetConfig(origin=[0.5, 0.5], velocity=[0.0, 0.0], mv=12.0)],
        seed=7,
        enable_shot_noise=False,
        enable_read_noise=False,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------- config parsing
def test_yaml_sections_roundtrip(tmp_path):
    p = tmp_path / "v2.yaml"
    p.write_text(
        """
sensor: {height: 64, width: 64, jitter_rms_arcsec: 2.5}
aperture:
  enabled: true
  shape: segmented
  phased: false
  segments:
    - {x: -0.6, y: 0.35, d: 1.0}
    - {x: 0.6, y: 0.35, d: 1.0}
    - {x: 0.0, y: -0.7, d: 1.0}
mirror: {coating: gold, roughness_nm: 9.0, n_reflections: 2}
band: {enabled: true, name: SWIR_YJH, qe: 0.8}
sky: {model: static, aerosol_delta_mag: 0.5}
"""
    )
    cfg = load_config(p)
    assert cfg.aperture.enabled and cfg.aperture.shape == "segmented"
    assert not cfg.aperture.phased
    assert len(cfg.aperture.segments) == 3
    assert isinstance(cfg.aperture.segments[0], SegmentConfig)
    assert cfg.aperture.segments[2].y == -0.7
    assert cfg.mirror.coating == "gold" and cfg.mirror.n_reflections == 2
    assert cfg.band.name == "SWIR_YJH" and cfg.band.qe == 0.8
    assert cfg.sky.aerosol_delta_mag == 0.5
    assert cfg.sensor.jitter_rms_arcsec == 2.5


# ------------------------------------------------------- zeropoint resolution
def test_zeropoint_resolution_matches_v1_anchor():
    """0.7 m f/2 bare-Al v1 recipe -> ZP 25.14 (the am_f2_070.yaml value)."""
    cfg = _base_config(
        aperture=ApertureConfig(enabled=True, shape="circle", diameter=0.7, obscuration=0.18),
        mirror=MirrorConfig(coating="bare_al", roughness_nm=9.0, n_reflections=1),
        band=BandConfig(enabled=True, name="V", qe=0.68),
    )
    cfg.sensor.zeropoint = 99.0  # must be overwritten
    scene = Scene(cfg)
    assert scene.sensor.zeropoint == pytest.approx(25.14, abs=0.05)


def test_static_scatter_brightening_matches_v1():
    """sigma=9 nm bare Al in V: sky 20.0 -> 19.955 (v1 hand-computed debit)."""
    cfg = _base_config(
        aperture=ApertureConfig(enabled=True, shape="circle", diameter=0.7, obscuration=0.18),
        mirror=MirrorConfig(coating="bare_al", roughness_nm=9.0, n_reflections=1),
        band=BandConfig(enabled=True, name="V", qe=0.68),
    )
    cfg.sensor.background_mv = 20.0
    scene = Scene(cfg)
    assert scene.sensor.background_mv == pytest.approx(19.955, abs=0.01)


def test_segmented_unphased_area_and_zeropoint():
    """3x1.0 m unphased segments ~ 3x the 1.0 m area -> ZP +2.5log10(3)."""
    seg = [SegmentConfig(-0.6, 0.35, 1.0), SegmentConfig(0.6, 0.35, 1.0),
           SegmentConfig(0.0, -0.7, 1.0)]
    mono = _base_config(
        aperture=ApertureConfig(enabled=True, shape="circle", diameter=1.0),
        band=BandConfig(enabled=True, name="H", qe=0.7),
    )
    tri = _base_config(
        aperture=ApertureConfig(enabled=True, shape="segmented", segments=seg, phased=False),
        band=BandConfig(enabled=True, name="H", qe=0.7),
    )
    zp_mono = Scene(mono).sensor.zeropoint
    zp_tri = Scene(tri).sensor.zeropoint
    assert zp_tri - zp_mono == pytest.approx(2.5 * math.log10(3.0), abs=0.02)


def test_band_requires_aperture():
    cfg = _base_config(band=BandConfig(enabled=True, name="V"))
    with pytest.raises(ValueError, match="aperture"):
        Scene(cfg)


# ------------------------------------------------------------- sun-driven sky
@pytest.mark.parametrize(
    "obs_time,is_day", [("2026-09-01T22:24:00Z", True), ("2026-09-01T10:00:00Z", False)]
)
def test_auto_sky_day_night(obs_time, is_day):
    pytest.importorskip("satkit")
    cfg = _base_config(
        aperture=ApertureConfig(enabled=True, shape="circle", diameter=0.5),
        mirror=MirrorConfig(coating="gold", roughness_nm=9.0),
        band=BandConfig(enabled=True, name="SWIR_YJH", qe=0.8),
        sky=SkyConfig(model="auto", elongation_deg=90.0),
        site=MAUI,
        obs_time=obs_time,
    )
    scene = Scene(cfg)
    _, _, _, meta = scene.render_signals(0)
    assert meta["sky_model"] == "auto"
    if is_day:
        assert meta["sun_alt_deg"] > 60
        assert meta["sky_mag_arcsec2"] < 8.0  # daytime SWIR sky
    else:
        assert meta["sun_alt_deg"] < -30
        assert meta["sky_mag_arcsec2"] > 13.0  # night SWIR sky
    # construction seeded the static bookkeeping value from frame 0
    assert scene.sensor.background_mv == pytest.approx(meta["sky_mag_arcsec2"], abs=0.01)


def test_auto_sky_requires_site_and_obs_time():
    cfg = _base_config(
        aperture=ApertureConfig(enabled=True, diameter=0.5),
        band=BandConfig(enabled=True, name="V"),
        sky=SkyConfig(model="auto"),
    )
    with pytest.raises(ValueError, match="site and obs_time"):
        Scene(cfg)


# -------------------------------------------------------- per-frame exposure
def test_exposure_override_scales_sources_and_metadata():
    cfg = _base_config()
    scene = Scene(cfg)
    _, _, tgt_full, meta_full = scene.render_signals(0)
    _, _, tgt_short, meta_short = scene.render_signals(0, exposure_s=0.01, frame_start_s=0.0)
    assert meta_full["exposure"] == 1.0
    assert meta_short["exposure"] == 0.01
    ratio = float(tgt_short.sum() / tgt_full.sum())
    assert ratio == pytest.approx(0.01, rel=1e-4)


def test_frame_start_override():
    cfg = _base_config()
    scene = Scene(cfg)
    _, _, _, meta = scene.render_signals(5, exposure_s=0.01, frame_start_s=0.05)
    assert meta["frame_time"] == pytest.approx(0.05)


# ------------------------------------------------------------------- jitter
def test_jitter_offsets_field_deterministically():
    cfg = _base_config()
    cfg.sensor.jitter_rms_arcsec = 3.0
    s1, s2 = Scene(cfg), Scene(_base_config())
    s2.sensor.jitter_rms_arcsec = 0.0
    _, _, _, m1 = s1.render_signals(0)
    _, _, _, m1b = s1.render_signals(1)
    _, _, _, m0 = s2.render_signals(0)
    assert m0["jitter_offset_px"] is None
    assert m1["jitter_offset_px"] is not None
    assert m1["jitter_offset_px"] != m1b["jitter_offset_px"]  # fresh draw per frame
    # scale sanity: 3 arcsec at 0.05deg/128px = 1.406 arcsec/px -> ~2.1 px RMS
    ps = 0.05 * 3600.0 / 128
    draws = [abs(v) for v in m1["jitter_offset_px"] + m1b["jitter_offset_px"]]
    assert max(draws) < 6 * 3.0 / ps  # within 6 sigma
    # same seed -> same offsets
    s4 = _base_config()
    s4.sensor.jitter_rms_arcsec = 3.0
    _, _, _, m4 = Scene(s4).render_signals(0)
    assert m4["jitter_offset_px"] == pytest.approx(m1["jitter_offset_px"])


# ------------------------------------------------------------ pupil PSF path
@pytest.mark.parametrize("shape,kw", [
    ("rect", dict(diameter=0.4, diameter_minor=0.3)),
    ("ellipse", dict(diameter=0.5, diameter_minor=0.3)),
    ("segmented", dict(segments=[SegmentConfig(-0.3, 0.0, 0.4), SegmentConfig(0.3, 0.0, 0.4)],
                       phased=True, piston_rms_nm=60.0)),
])
def test_pupil_psf_model_renders_energy_conserving(shape, kw):
    cfg = _base_config(
        aperture=ApertureConfig(enabled=True, shape=shape, **kw),
        band=BandConfig(enabled=True, name="H", qe=0.7),
    )
    cfg.sensor.psf_model = "pupil"
    cfg.optics.f_number = 8.0
    cfg.optics.pixel_pitch_um = 10.0
    cfg.stars = StarFieldConfig(mode="bins", density=[0.0] * 12, pad_mult=0.0)
    scene = Scene(cfg)
    assert scene._pupil_kernel is not None
    assert float(scene._pupil_kernel.sum()) == pytest.approx(1.0, abs=1e-3)
    _, _, target_signal, meta = scene.render_signals(0)
    # single mid-frame target: deposited energy within the frame ~ its intensity
    expect = float(scene.render_signals(0)[3]["target_intensities"][0])
    assert float(target_signal.sum()) == pytest.approx(expect, rel=0.05)


def test_pupil_excludes_optics():
    cfg = _base_config(
        aperture=ApertureConfig(enabled=True, shape="circle", diameter=0.5),
        band=BandConfig(enabled=True, name="V", qe=0.68),
    )
    cfg.sensor.psf_model = "pupil"
    cfg.optics.enabled = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        Scene(cfg)


# ------------------------------------------------- Poisson high-flux switchover
def test_poisson_gaussian_switchover_statistics():
    torch.manual_seed(0)
    lam = 5.0e4
    x = torch.full((512, 512), lam)
    y = poisson_noise(x)
    assert float(y.mean()) == pytest.approx(lam, rel=1e-3)
    assert float(y.std()) == pytest.approx(math.sqrt(lam), rel=0.05)
    assert float(y.min()) >= 0.0
    # small-lambda path still exact Poisson
    z = poisson_noise(torch.full((512, 512), 5.0))
    assert float(z.mean()) == pytest.approx(5.0, rel=0.05)
    assert float(z.var()) == pytest.approx(5.0, rel=0.05)
    # mixed tensor: both regimes in one call
    m = torch.tensor([5.0, 2.0e4])
    out = poisson_noise(m.expand(100000, 2).contiguous())
    assert float(out[:, 0].mean()) == pytest.approx(5.0, rel=0.05)
    assert float(out[:, 1].mean()) == pytest.approx(2.0e4, rel=1e-3)


# ------------------------------------------------------- daytime saturation
def test_daytime_visible_saturates():
    """VIS + full daytime sky + 1 s: the frame must slam into full well."""
    cfg = _base_config()
    cfg.sensor.background_mv = 4.0  # daytime visible sky
    cfg.enable_shot_noise = True
    cfg.enable_read_noise = True
    scene = Scene(cfg)
    digital, _, _, _ = scene.render_signals(0)
    max_dn = math.floor((cfg.sensor.fwc) / cfg.sensor.gain)
    assert float((digital >= max_dn - 1).float().mean()) > 0.99


# ----------------------------------------------------- sstrc7 SWIR magnitudes
def test_sstrc7_filter_center_changes_star_flux():
    import os

    catalog = os.environ.get("SSTRC7_CATALOG", "")
    if not os.path.isdir(catalog):
        pytest.skip("set SSTRC7_CATALOG to run")
    pytest.importorskip("sstrc7")
    base = dict(
        aperture=ApertureConfig(enabled=True, shape="circle", diameter=0.5),
        mirror=MirrorConfig(coating="protected_ag", roughness_nm=2.0),
    )
    vis = _base_config(band=BandConfig(enabled=True, name="V", qe=0.68), **base)
    swir = _base_config(band=BandConfig(enabled=True, name="H", qe=0.7), **base)
    for cfg in (vis, swir):
        cfg.stars = StarFieldConfig(
            mode="sstrc7", catalog_path=catalog, ra=45.0, dec=10.0, pad_mult=0.0, mv_max=14.0
        )
        cfg.sensor.y_fov = cfg.sensor.x_fov = 0.5
        cfg.sensor.height = cfg.sensor.width = 256
    s_vis, s_swir = Scene(vis), Scene(swir)
    assert s_vis.star_positions.shape[0] > 0
    # Same sky field, different band: magnitudes (hence pe) must differ per star
    if s_vis.star_intensities.shape[0] == s_swir.star_intensities.shape[0]:
        ratio = (s_swir.star_intensities / s_vis.star_intensities.clamp(min=1e-9)).cpu()
        assert float(ratio.std() / ratio.mean()) > 0.05  # star colors, not a scalar offset
