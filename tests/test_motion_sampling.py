"""Tests for auto-resolved star motion sampling (noise-floor-derived t_osf).

The spacing rule inverts the exact Gaussian-comb ripple law (Poisson
summation): semi-amplitude a = 2*exp(-2*pi^2*sigma^2/d^2). Auto mode chooses
d so that a stays below ~half the per-pixel noise on the brightest streak.
"""

from __future__ import annotations

import math

import pytest
import torch

from sdasim.config import SceneConfig, SensorConfig, StarFieldConfig, StarMotionConfig
from sdasim.render import motion_sampling_spacing, resolve_temporal_osf
from sdasim.scene import Scene

R_MAX = 0.5 * math.hypot(256, 256)


class TestSpacingRule:
    def test_bright_source_demands_finer_sampling(self):
        d_bright = motion_sampling_spacing(1.5, peak_pe=50_000.0, noise_pe=40.0)
        d_faint = motion_sampling_spacing(1.5, peak_pe=200.0, noise_pe=40.0)
        assert d_bright < d_faint

    def test_quality_pin_overrides_photometry(self):
        d = motion_sampling_spacing(1.5, peak_pe=1e9, noise_pe=1e-3, quality=2.0)
        assert d == pytest.approx(3.0)

    def test_clamps_and_pixel_floor(self):
        # Absurdly bright: clamps at 1.3*sigma, never demands sub-1.3 spacing.
        assert motion_sampling_spacing(1.5, 1e12, 1.0) == pytest.approx(1.3 * 1.5)
        # Sky-swamped: relaxes but never beyond 2.2*sigma.
        assert motion_sampling_spacing(1.5, 10.0, 1e6) == pytest.approx(2.2 * 1.5)
        # Tiny sigma: the 1 px pixel-lattice floor wins.
        assert motion_sampling_spacing(0.3, 1e12, 1.0) == 1.0

    def test_ripple_below_noise_end_to_end(self):
        """Render a streak at the derived spacing; measured ripple < noise floor."""
        from sdasim.render import expand_motion
        from sdasim.splat import splat_gaussians

        sigma, i_tot = 1.5, 3.0e5
        streak = 120.0
        peak = i_tot / (math.sqrt(2 * math.pi) * sigma * streak)
        noise = math.sqrt(peak + 900.0)  # source shot + sky 900 pe/px
        d = motion_sampling_spacing(sigma, peak, noise)
        t_osf = math.ceil(streak / d)
        pos = torch.tensor([[64.0, 20.0]])
        p, i = expand_motion(
            pos, torch.tensor([i_tot]), [0.0, streak / 2.0], 0.0, 0.0, 2.0, t_osf,
            (64.0, 160.0),
        )
        img = splat_gaussians(128, 320, p, i, sigma)
        # Trail spans cols 20..140 (start 20 + 120 px); sample well inside it.
        band = img[60:69, 40:120].sum(dim=0)
        ripple_pe = (band.max() - band.mean()).item() * peak / band.mean().item()
        assert ripple_pe < noise


class TestResolveTemporalOsf:
    def test_explicit_pin_wins(self):
        assert resolve_temporal_osf(37, [100.0, 0.0], 0.0, 5.0, 3.0, R_MAX, 10, 100) == (37, False)

    def test_static_field_is_one(self):
        assert resolve_temporal_osf(None, [0.0, 0.0], 0.0, 5.0, 3.0, R_MAX, 10, None) == (1, False)

    def test_scales_with_rate_and_exposure(self):
        osf_slow, _ = resolve_temporal_osf(None, [4.0, 0.0], 0.0, 2.0, 3.0, R_MAX, 10, None)
        osf_fast, _ = resolve_temporal_osf(None, [40.0, 0.0], 0.0, 2.0, 3.0, R_MAX, 10, None)
        osf_long, _ = resolve_temporal_osf(None, [4.0, 0.0], 0.0, 20.0, 3.0, R_MAX, 10, None)
        assert osf_fast > osf_slow and osf_long > osf_slow
        assert osf_fast == math.ceil(80.0 / 3.0)

    def test_moving_field_floors_at_two(self):
        """Centroid-cliff guard: any motion -> at least 2 sub-sources (midpoint
        placement), never the start-of-exposure single sample."""
        osf, _ = resolve_temporal_osf(None, [0.05, 0.0], 0.0, 2.0, 3.0, R_MAX, 10, None)
        assert osf == 2

    def test_rotation_only_field_expands(self):
        """Rotation-collapse guard: pure rotation must still expand."""
        osf, _ = resolve_temporal_osf(None, [0.0, 0.0], 0.02, 2.0, 3.0, R_MAX, 10, None)
        arc = 0.02 * R_MAX * 2.0
        assert osf == math.ceil(arc / 3.0)

    def test_budget_caps_with_flag(self):
        osf, clamped = resolve_temporal_osf(None, [500.0, 0.0], 0.0, 2.0, 3.0, R_MAX, 1000, 50_000)
        assert clamped and osf == 50  # 50_000 // 1000
        osf2, cl2 = resolve_temporal_osf(None, [500.0, 0.0], 0.0, 2.0, 3.0, R_MAX, 10, 50_000)
        assert not cl2 and osf2 == math.ceil(1000.0 / 3.0)


class TestSceneIntegration:
    def _scene(self, **kw):
        cfg = SceneConfig(
            sensor=SensorConfig(height=128, width=128, y_fov=0.5, x_fov=0.5, exposure=2.0),
            stars=StarFieldConfig(mode="bins"),
            seed=11,
            **kw,
        )
        return Scene(cfg)

    def test_auto_default_resolves_and_reports(self):
        scene = self._scene(star_motion=StarMotionConfig(translation=[0.0, 6.0]))
        _, _, _, meta = scene.render_signals(0)
        assert meta["star_temporal_osf"] >= 2
        assert meta["sampling_spacing_px"] is not None
        assert meta["star_osf_clamped"] is False
        # spacing is sigma-relative and clamped to [1.3, 2.2] sigma (or 1 px floor)
        sigma = scene.sensor.psf_sigma
        assert 1.0 <= meta["sampling_spacing_px"] <= 2.2 * sigma + 1e-6

    def test_pinned_config_still_wins(self):
        scene = self._scene(star_motion=StarMotionConfig(translation=[0.0, 6.0], temporal_osf=7))
        _, _, _, meta = scene.render_signals(0)
        assert meta["star_temporal_osf"] == 7
        assert meta["sampling_spacing_px"] is None

    def test_noise_free_scene_has_finite_spacing(self):
        """Shot/read noise off: A/D quantization still provides a floor, so the
        auto rule cannot demand infinite sampling."""
        scene = self._scene(
            star_motion=StarMotionConfig(translation=[0.0, 6.0]),
            enable_shot_noise=False,
            enable_read_noise=False,
        )
        _, _, _, meta = scene.render_signals(0)
        assert meta["sampling_spacing_px"] >= 1.0
        assert meta["star_temporal_osf"] < 100

    def test_budget_cap_flags_metadata(self):
        scene = self._scene(
            star_motion=StarMotionConfig(translation=[0.0, 500.0]),  # LEO-ish rate
            max_render_seconds=0.02,
        )
        _, _, _, meta = scene.render_signals(0)
        assert meta["star_osf_clamped"] is True
        n = scene.star_positions.shape[0]
        assert n * meta["star_temporal_osf"] <= scene._subsource_budget() + n

    def test_unlimited_budget_when_disabled(self):
        scene = self._scene(
            star_motion=StarMotionConfig(translation=[0.0, 500.0]),
            max_render_seconds=0.0,
        )
        assert scene._subsource_budget() is None
        _, _, _, meta = scene.render_signals(0)
        assert meta["star_osf_clamped"] is False
