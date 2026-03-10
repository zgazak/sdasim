"""Tests for the Scene class."""

from __future__ import annotations

import tempfile

import torch
import yaml
import pytest

from sdasim.config import SceneConfig, SensorConfig, StarFieldConfig, StarMotionConfig, TargetConfig
from sdasim.scene import Scene


class TestScene:
    """Tests for Scene construction and rendering."""

    def _make_config(self, **overrides) -> SceneConfig:
        """Create a minimal config for testing."""
        sensor = SensorConfig(
            height=32, width=32, y_fov=0.1, x_fov=0.1,
            exposure=1.0, num_frames=4, zeropoint=20.0,
            psf_sigma=1.0, dark_current=5.0, read_noise=8.0,
            electronic_noise=3.0, background_mv=20.0, bias=100.0,
            gain=4.0, fwc=50000.0, a2d_bias=200.0, a2d_dtype="uint16",
        )
        stars = StarFieldConfig(
            mode="bins", mv_bins=[12, 13, 14], density=[5.0, 10.0],
        )
        targets = [TargetConfig(origin=[0.5, 0.5], velocity=[2.0, 3.0], mv=12.0)]
        cfg = SceneConfig(
            sensor=sensor, stars=stars, targets=targets,
            seed=42, device="cpu", enable_shot_noise=False, enable_read_noise=False,
        )
        return cfg

    def test_construction(self):
        """Scene should construct without error."""
        cfg = self._make_config()
        scene = Scene(cfg)
        assert scene.star_positions.shape[1] == 2
        assert scene.star_intensities.shape[0] == scene.star_positions.shape[0]

    def test_render_single_frame(self):
        """Single frame render should produce valid image."""
        scene = Scene(self._make_config())
        img, meta = scene.render(frame_idx=0)
        assert img.shape == (32, 32)
        assert meta["frame_idx"] == 0
        assert meta["num_targets"] == 1

    def test_metadata_velocity_fields(self):
        """Metadata should include velocity and exposure fields."""
        cfg = self._make_config()
        scene = Scene(cfg)
        _, meta = scene.render(frame_idx=0)

        # Star velocity matches config (legacy mode uses star_motion.translation)
        assert "star_velocity" in meta
        assert meta["star_velocity"] == cfg.star_motion.translation
        assert "star_rotation" in meta
        assert meta["star_rotation"] == cfg.star_motion.rotation

        # Target velocities present
        assert "target_velocities" in meta
        assert len(meta["target_velocities"]) == 1
        assert meta["target_velocities"][0] == [2.0, 3.0]

        # Exposure
        assert "exposure" in meta
        assert meta["exposure"] == cfg.sensor.exposure

    def test_metadata_no_targets_empty_velocities(self):
        """With no targets, target_velocities should be empty list."""
        cfg = SceneConfig(
            sensor=SensorConfig(
                height=32, width=32, y_fov=0.1, x_fov=0.1,
                exposure=1.0, num_frames=1, zeropoint=20.0,
                psf_sigma=1.0, dark_current=5.0, read_noise=8.0,
                electronic_noise=3.0, background_mv=20.0, bias=100.0,
                gain=4.0, fwc=50000.0, a2d_bias=200.0, a2d_dtype="uint16",
            ),
            stars=StarFieldConfig(mode="bins", mv_bins=[12, 13], density=[5.0]),
            targets=[],
            seed=42, device="cpu",
            enable_shot_noise=False, enable_read_noise=False,
        )
        scene = Scene(cfg)
        _, meta = scene.render(0)
        assert meta["target_velocities"] == []

    def test_render_batch(self):
        """Batch render should produce stacked images."""
        scene = Scene(self._make_config())
        imgs, metas = scene.render_batch([0, 1, 2])
        assert imgs.shape == (3, 32, 32)
        assert len(metas) == 3

    def test_render_sequence(self):
        """Sequence render should produce all frames."""
        scene = Scene(self._make_config())
        imgs, metas = scene.render_sequence()
        assert imgs.shape == (4, 32, 32)
        assert len(metas) == 4

    def test_from_yaml(self, tmp_path):
        """Scene.from_yaml should load and render."""
        config_data = {
            "sensor": {
                "height": 32, "width": 32, "y_fov": 0.1, "x_fov": 0.1,
                "exposure": 1.0, "num_frames": 2, "zeropoint": 20.0,
                "psf_sigma": 1.0, "dark_current": 5.0, "read_noise": 8.0,
                "electronic_noise": 3.0, "background_mv": 20.0, "bias": 100.0,
                "gain": 4.0, "fwc": 50000.0, "a2d_bias": 200.0, "a2d_dtype": "uint16",
            },
            "stars": {
                "mode": "bins",
                "mv_bins": [12, 13, 14],
                "density": [5.0, 10.0],
            },
            "seed": 42,
            "device": "cpu",
            "enable_shot_noise": False,
            "enable_read_noise": False,
        }
        yaml_path = tmp_path / "scene.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_data, f)

        scene = Scene.from_yaml(yaml_path)
        img, meta = scene.render()
        assert img.shape == (32, 32)

    def test_override_psf_sigma(self):
        """Overriding psf_sigma should change output."""
        scene = Scene(self._make_config())
        img1, _ = scene.render(psf_sigma=0.5)
        img2, _ = scene.render(psf_sigma=3.0)
        # Different sigma should produce different images
        assert not torch.equal(img1, img2)

    def test_deterministic_with_no_noise(self):
        """Without noise, same scene should produce same image."""
        cfg = self._make_config()
        scene1 = Scene(cfg)
        scene2 = Scene(cfg)
        img1, _ = scene1.render()
        img2, _ = scene2.render()
        assert torch.equal(img1, img2)

    def test_star_motion_between_frames(self):
        """Stars should shift between frames when star_motion.translation is set."""
        cfg = SceneConfig(
            sensor=SensorConfig(
                height=64, width=64, y_fov=0.1, x_fov=0.1,
                exposure=1.0, gap=0.0, num_frames=3, zeropoint=20.0,
                psf_sigma=1.0, dark_current=0.0, read_noise=0.0,
                electronic_noise=0.0, background_mv=30.0, bias=0.0,
                gain=1.0, fwc=1e9, a2d_bias=0.0, a2d_dtype="uint16",
            ),
            stars=StarFieldConfig(
                mv_bins=[10, 11], density=[50.0],
            ),
            star_motion=StarMotionConfig(
                translation=[5.0, 0.0],  # 5 px/sec drift in row
                temporal_osf=1,  # no within-frame blur, just inter-frame shift
            ),
            seed=99, device="cpu",
            enable_shot_noise=False, enable_read_noise=False,
        )
        scene = Scene(cfg)
        img0, _ = scene.render(0)
        img1, _ = scene.render(1)
        img2, _ = scene.render(2)
        # Frames should differ because stars shift between frames
        assert not torch.equal(img0, img1), "Frame 0 and 1 should differ (star drift)"
        assert not torch.equal(img1, img2), "Frame 1 and 2 should differ (star drift)"

    def test_target_visible_and_moves(self):
        """Target should be visible and move between frames."""
        cfg = SceneConfig(
            sensor=SensorConfig(
                height=64, width=64, y_fov=0.1, x_fov=0.1,
                exposure=1.0, gap=0.0, num_frames=4, zeropoint=20.0,
                psf_sigma=1.0, dark_current=0.0, read_noise=0.0,
                electronic_noise=0.0, background_mv=30.0, bias=0.0,
                gain=1.0, fwc=1e9, a2d_bias=0.0, a2d_dtype="uint16",
            ),
            stars=StarFieldConfig(
                mv_bins=[18, 19], density=[0.0],  # no stars
            ),
            targets=[TargetConfig(origin=[0.5, 0.5], velocity=[10.0, 0.0], mv=10.0)],
            seed=42, device="cpu",
            enable_shot_noise=False, enable_read_noise=False,
        )
        scene = Scene(cfg)
        img0, meta0 = scene.render(0)
        img1, meta1 = scene.render(1)

        # Target should produce signal (no stars, no background)
        assert img0.max() > 0, "Target should produce non-zero signal"

        # Target should be at different position in frame 1
        assert not torch.equal(img0, img1), "Target should move between frames"

        # Peak should be near center for frame 0 (origin=[0.5, 0.5])
        peak0 = (img0 == img0.max()).nonzero()[0]
        assert abs(peak0[0].item() - 32) < 5, f"Target peak row should be near 32, got {peak0[0]}"

        # Peak should shift +10 rows in frame 1 (velocity=[10,0], exposure=1s, gap=0s)
        peak1 = (img1 == img1.max()).nonzero()[0]
        assert peak1[0].item() > peak0[0].item(), "Target should move in row direction"
