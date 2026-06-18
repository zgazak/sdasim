"""Tests for configuration loading."""

from __future__ import annotations

import yaml

from sdasim.config import SceneConfig, _dataclass_from_dict, load_config


class TestSceneConfig:
    """Tests for config dataclasses."""

    def test_defaults(self):
        """Default config should be valid."""
        cfg = SceneConfig()
        assert cfg.sensor.height == 512
        assert cfg.sensor.width == 512
        assert cfg.enable_shot_noise is True
        assert cfg.targets == []

    def test_from_dict(self):
        """Building from nested dict should work."""
        data = {
            "sensor": {"height": 256, "width": 256, "psf_sigma": 2.0},
            "targets": [{"mode": "line", "origin": [0.5, 0.5], "velocity": [1.0, 2.0], "mv": 10.0}],
            "seed": 123,
        }
        cfg = _dataclass_from_dict(SceneConfig, data)
        assert cfg.sensor.height == 256
        assert cfg.sensor.psf_sigma == 2.0
        assert len(cfg.targets) == 1
        assert cfg.targets[0].mv == 10.0
        assert cfg.seed == 123

    def test_yaml_roundtrip(self, tmp_path):
        """Load from YAML file."""
        config_data = {
            "sensor": {
                "height": 128,
                "width": 128,
                "y_fov": 0.25,
                "x_fov": 0.25,
                "exposure": 1.0,
                "num_frames": 4,
                "zeropoint": 20.0,
                "psf_sigma": 1.0,
                "dark_current": 5.0,
                "read_noise": 8.0,
                "electronic_noise": 3.0,
                "background_mv": 20.0,
                "bias": 100.0,
                "gain": 4.0,
                "fwc": 50000.0,
                "a2d_bias": 200.0,
                "a2d_dtype": "uint16",
            },
            "stars": {
                "mode": "bins",
                "mv_bins": [10, 11, 12],
                "density": [5.0, 10.0],
            },
            "seed": 42,
        }
        yaml_path = tmp_path / "test.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_config(yaml_path)
        assert cfg.sensor.height == 128
        assert cfg.sensor.gain == 4.0
        assert cfg.stars.mode == "bins"
        assert len(cfg.stars.density) == 2
