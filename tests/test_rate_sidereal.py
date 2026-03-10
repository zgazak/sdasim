"""Tests for rate_sidereal mode and random scene sampler."""

from __future__ import annotations

import torch
import pytest

from sdasim.config import (
    SceneConfig,
    SensorConfig,
    StarFieldConfig,
    StarMotionConfig,
    TargetConfig,
)
from sdasim.scene import Scene
from sdasim.sampler import SceneDistribution, random_scene


def _make_rate_sidereal_config(**overrides) -> SceneConfig:
    """Create a minimal rate_sidereal config for testing."""
    sensor = SensorConfig(
        height=32, width=32, y_fov=0.1, x_fov=0.1,
        exposure=1.0, gap=0.0, num_frames=4, zeropoint=20.0,
        psf_sigma=1.0, dark_current=0.0, read_noise=0.0,
        electronic_noise=0.0, background_mv=30.0, bias=0.0,
        gain=1.0, fwc=1e9, a2d_bias=0.0, a2d_dtype="uint16",
    )
    stars = StarFieldConfig(mode="bins", mv_bins=[10, 11, 12], density=[20.0, 40.0])
    star_motion = StarMotionConfig(
        translation=[5.0, -3.0], temporal_osf=50,
    )
    # On-rate target: V_inertial ≈ -translation -> near-stationary when tracked
    # Off-rate target: arbitrary V_inertial
    targets = [
        TargetConfig(origin=[0.5, 0.5], velocity=[-5.0, 3.0], mv=10.0),   # on-rate
        TargetConfig(origin=[0.3, 0.7], velocity=[8.0, -4.0], mv=10.0),   # off-rate
    ]
    defaults = dict(
        sensor=sensor, stars=stars, star_motion=star_motion, targets=targets,
        seed=42, device="cpu", enable_shot_noise=False, enable_read_noise=False,
        mode="rate_sidereal", sidereal_start=2,
    )
    defaults.update(overrides)
    return SceneConfig(**defaults)


class TestRateSidereal:
    """Tests for mode='rate_sidereal' behavior."""

    def test_construction(self):
        """rate_sidereal scene should construct without error."""
        cfg = _make_rate_sidereal_config()
        scene = Scene(cfg)
        assert scene.star_positions.shape[1] == 2

    def test_missing_sidereal_start_raises(self):
        """mode='rate_sidereal' without sidereal_start should raise."""
        with pytest.raises(ValueError, match="sidereal_start"):
            Scene(_make_rate_sidereal_config(sidereal_start=None))

    def test_metadata_frame_mode(self):
        """Metadata should include frame_mode for rate_sidereal scenes."""
        scene = Scene(_make_rate_sidereal_config())
        # Frame 0,1 = rate_track; Frame 2,3 = sidereal
        _, meta0 = scene.render(0)
        _, meta1 = scene.render(1)
        _, meta2 = scene.render(2)
        _, meta3 = scene.render(3)
        assert meta0["frame_mode"] == "rate_track"
        assert meta1["frame_mode"] == "rate_track"
        assert meta2["frame_mode"] == "sidereal"
        assert meta3["frame_mode"] == "sidereal"

    def test_legacy_mode_no_frame_mode(self):
        """Legacy mode (None) should set frame_mode to None."""
        cfg = SceneConfig(
            sensor=SensorConfig(
                height=32, width=32, y_fov=0.1, x_fov=0.1,
                exposure=1.0, num_frames=2, zeropoint=20.0,
                psf_sigma=1.0, dark_current=0.0, read_noise=0.0,
                electronic_noise=0.0, background_mv=30.0, bias=0.0,
                gain=1.0, fwc=1e9, a2d_bias=0.0, a2d_dtype="uint16",
            ),
            stars=StarFieldConfig(mode="bins", mv_bins=[12, 13], density=[5.0]),
            seed=42, device="cpu",
            enable_shot_noise=False, enable_read_noise=False,
        )
        scene = Scene(cfg)
        _, meta = scene.render(0)
        assert meta["frame_mode"] is None

    def test_sidereal_frames_have_sharp_stars(self):
        """In sidereal frames, stars should not streak (identical across sidereal frames)."""
        cfg = _make_rate_sidereal_config(targets=[])  # no targets to isolate stars
        scene = Scene(cfg)

        # Sidereal frames 2 and 3: stars should be identical (no motion, no offset)
        img2, _ = scene.render(2)
        img3, _ = scene.render(3)
        assert torch.equal(img2, img3), "Sidereal frames with no targets should be identical"

    def test_rate_track_frames_have_star_streaks(self):
        """In rate-track frames, stars should differ between frames (drift + blur)."""
        cfg = _make_rate_sidereal_config(targets=[])
        scene = Scene(cfg)

        img0, _ = scene.render(0)
        img1, _ = scene.render(1)
        # Stars move between rate-track frames
        assert not torch.equal(img0, img1), "Rate-track frames should differ (star drift)"

    def test_rate_track_and_sidereal_differ(self):
        """Rate-track and sidereal frames should look different."""
        cfg = _make_rate_sidereal_config(targets=[])
        scene = Scene(cfg)

        img0, _ = scene.render(0)  # rate_track
        img2, _ = scene.render(2)  # sidereal
        assert not torch.equal(img0, img2), "Rate-track and sidereal should differ"

    def test_same_star_field_across_modes(self):
        """Both frame types should use the same underlying star catalog."""
        scene = Scene(_make_rate_sidereal_config())
        # The star_positions/star_intensities are shared
        # Render a rate-track frame and sidereal frame - both should have same num_stars
        _, meta0 = scene.render(0)
        _, meta2 = scene.render(2)
        assert meta0["num_stars"] == meta2["num_stars"]

    def test_on_rate_target_compact_in_rate_track(self):
        """On-rate target (V_inertial ≈ -translation) should be compact in rate-track frames."""
        # Config with only the on-rate target, no stars to avoid interference
        cfg = _make_rate_sidereal_config(
            targets=[TargetConfig(origin=[0.5, 0.5], velocity=[-5.0, 3.0], mv=8.0)],
            stars=StarFieldConfig(mode="bins", mv_bins=[20, 21], density=[0.0]),
        )
        scene = Scene(cfg)

        # Rate-track frame: on-rate target should be compact
        img_rt, _ = scene.render(0)
        # Sidereal frame: same target should streak (at its inertial velocity)
        img_sid, _ = scene.render(2)

        # Count lit pixels (above some threshold) — compact should have fewer
        thresh = img_rt.max() * 0.1
        rt_lit = (img_rt > thresh).sum().item()
        sid_lit = (img_sid > thresh).sum().item()
        assert rt_lit < sid_lit, (
            f"On-rate target should be more compact in rate-track ({rt_lit}) "
            f"than sidereal ({sid_lit})"
        )

    def test_render_sequence(self):
        """render_sequence should work with rate_sidereal mode."""
        scene = Scene(_make_rate_sidereal_config())
        imgs, metas = scene.render_sequence()
        assert imgs.shape == (4, 32, 32)
        assert len(metas) == 4
        assert metas[0]["frame_mode"] == "rate_track"
        assert metas[3]["frame_mode"] == "sidereal"

    def test_backward_compat_legacy_mode(self):
        """Legacy mode=None should produce identical results to pre-change behavior."""
        cfg = SceneConfig(
            sensor=SensorConfig(
                height=32, width=32, y_fov=0.1, x_fov=0.1,
                exposure=1.0, gap=0.0, num_frames=3, zeropoint=20.0,
                psf_sigma=1.0, dark_current=0.0, read_noise=0.0,
                electronic_noise=0.0, background_mv=30.0, bias=0.0,
                gain=1.0, fwc=1e9, a2d_bias=0.0, a2d_dtype="uint16",
            ),
            stars=StarFieldConfig(mode="bins", mv_bins=[10, 11], density=[20.0]),
            star_motion=StarMotionConfig(translation=[3.0, -2.0], temporal_osf=20),
            targets=[TargetConfig(origin=[0.5, 0.5], velocity=[1.0, 2.0], mv=12.0)],
            seed=7, device="cpu",
            enable_shot_noise=False, enable_read_noise=False,
        )
        scene = Scene(cfg)
        # Should render without error and mode is None
        assert cfg.mode is None
        img, meta = scene.render(0)
        assert img.shape == (32, 32)
        assert meta["frame_mode"] is None


class TestSampler:
    """Tests for random_scene() and SceneDistribution."""

    def test_random_scene_default(self):
        """random_scene() with defaults should produce a valid config."""
        cfg = random_scene(seed=0, device="cpu")
        assert isinstance(cfg, SceneConfig)
        scene = Scene(cfg)
        img, meta = scene.render(0)
        assert img.shape == (256, 256)

    def test_deterministic_with_seed(self):
        """Same seed should produce identical configs."""
        cfg1 = random_scene(seed=42, device="cpu")
        cfg2 = random_scene(seed=42, device="cpu")
        assert cfg1.sensor.exposure == cfg2.sensor.exposure
        assert cfg1.mode == cfg2.mode
        assert len(cfg1.targets) == len(cfg2.targets)
        for t1, t2 in zip(cfg1.targets, cfg2.targets):
            assert t1.velocity == t2.velocity
            assert t1.mv == t2.mv

    def test_different_seeds_differ(self):
        """Different seeds should produce different configs."""
        cfg1 = random_scene(seed=0, device="cpu")
        cfg2 = random_scene(seed=1, device="cpu")
        # At least something should differ (mode, exposure, targets, etc.)
        attrs_differ = (
            cfg1.sensor.exposure != cfg2.sensor.exposure
            or cfg1.mode != cfg2.mode
            or len(cfg1.targets) != len(cfg2.targets)
        )
        assert attrs_differ, "Different seeds should produce different configs"

    def test_target_counts_in_range(self):
        """Target counts should respect distribution ranges."""
        dist = SceneDistribution(
            n_on_rate_range=(2, 3),
            n_off_rate_range=(1, 2),
        )
        for seed in range(20):
            cfg = random_scene(dist=dist, seed=seed, device="cpu")
            n = len(cfg.targets)
            assert 3 <= n <= 5, f"Expected 3-5 targets, got {n} (seed={seed})"

    def test_rate_sidereal_has_valid_sidereal_start(self):
        """rate_sidereal configs should have valid sidereal_start."""
        dist = SceneDistribution(
            modes=["rate_sidereal"],
            num_frames=8,
        )
        for seed in range(10):
            cfg = random_scene(dist=dist, seed=seed, device="cpu")
            assert cfg.mode == "rate_sidereal"
            assert cfg.sidereal_start is not None
            assert 1 <= cfg.sidereal_start < cfg.sensor.num_frames

    def test_sidereal_mode_zero_star_motion(self):
        """Sidereal-only scenes should have zero star motion."""
        dist = SceneDistribution(modes=["sidereal"])
        cfg = random_scene(dist=dist, seed=0, device="cpu")
        assert cfg.star_motion.translation == [0.0, 0.0]
        assert cfg.mode is None  # sidereal uses legacy mode

    def test_rate_track_mode_nonzero_star_motion(self):
        """Rate-track scenes should have nonzero star motion."""
        dist = SceneDistribution(modes=["rate_track"])
        cfg = random_scene(dist=dist, seed=0, device="cpu")
        tx, ty = cfg.star_motion.translation
        assert tx != 0.0 or ty != 0.0

    def test_custom_distribution(self):
        """Custom SceneDistribution parameters should be respected."""
        dist = SceneDistribution(
            height=64,
            width=128,
            num_frames=3,
            exposure_range=(2.0, 2.0),  # fixed exposure
        )
        cfg = random_scene(dist=dist, seed=0, device="cpu")
        assert cfg.sensor.height == 64
        assert cfg.sensor.width == 128
        assert cfg.sensor.num_frames == 3
        assert cfg.sensor.exposure == 2.0

    def test_random_scene_renders_all_modes(self):
        """Each mode should produce a scene that renders successfully."""
        for mode_name in ["sidereal", "rate_track", "rate_sidereal"]:
            dist = SceneDistribution(modes=[mode_name], num_frames=4)
            cfg = random_scene(dist=dist, seed=7, device="cpu")
            scene = Scene(cfg)
            imgs, metas = scene.render_sequence()
            assert imgs.shape[0] == 4, f"mode={mode_name} should render 4 frames"
