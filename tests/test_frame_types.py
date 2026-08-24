"""Calibration frame types: dark/bias/flat suppress sources and keep only the
components the shutter/illumination admits. Bias is an electronic readout
offset, so it must carry no shot noise."""

import pytest
import torch

import sdasim
from sdasim.config import SceneConfig, SensorConfig, StarFieldConfig, TargetConfig
from sdasim.fpa import analog_to_digital


def _scene(enable_shot=False, enable_read=False, **sensor_kw):
    cfg = SceneConfig(
        sensor=SensorConfig(height=64, width=64, exposure=2.0, **sensor_kw),
        stars=StarFieldConfig(mode="bins"),
        targets=[TargetConfig(mode="line", origin=[0.5, 0.5], velocity=[3.0, 0.0], mv=10.0)],
        seed=0,
        enable_shot_noise=enable_shot,
        enable_read_noise=enable_read,
    )
    return sdasim.Scene(cfg)


def test_light_is_default_and_has_sources():
    digital, star, target, meta = _scene().render_signals(0)
    assert meta["frame_type"] == "light"
    assert target.sum() > 0  # mv=10 target always lands flux


def test_dark_has_no_sources_or_sky():
    scene = _scene()
    digital, star, target, meta = scene.render_signals(0, frame_type="dark")
    assert meta["frame_type"] == "dark"
    assert meta["num_stars"] == 0 and meta["num_targets"] == 0
    assert star.sum() == 0 and target.sum() == 0
    # Noise off: every pixel is exactly a2d(dark_current + bias)
    s = scene.sensor
    expected = analog_to_digital(
        torch.tensor(scene.dark_current_pe + scene.bias_pe),
        s.gain, s.fwc, s.a2d_bias, s.a2d_dtype,
    )
    assert torch.all(digital == expected)


def test_bias_is_zero_second_dark():
    scene = _scene()
    digital, _, _, meta = scene.render_signals(0, frame_type="bias")
    assert meta["frame_type"] == "bias"
    s = scene.sensor
    expected = analog_to_digital(
        torch.tensor(float(scene.bias_pe)), s.gain, s.fwc, s.a2d_bias, s.a2d_dtype
    )
    assert torch.all(digital == expected)  # no dark current despite exposure=2s


def test_bias_carries_no_shot_noise():
    # Shot noise ON, read noise off: a bias frame must still be constant.
    digital, _, _, _ = _scene(enable_shot=True).render_signals(0, frame_type="bias")
    assert digital.min() == digital.max()


def test_flat_level_is_radiance_times_exposure():
    scene = _scene(flat_radiance=1000.0)
    digital, star, target, _ = scene.render_signals(0, frame_type="flat")
    assert star.sum() == 0 and target.sum() == 0
    s = scene.sensor
    expected = analog_to_digital(
        torch.tensor(1000.0 * s.exposure + scene.dark_current_pe + scene.bias_pe),
        s.gain, s.fwc, s.a2d_bias, s.a2d_dtype,
    )
    assert torch.all(digital == expected)
    # Brighter lamp -> brighter flat
    brighter, _, _, _ = _scene(flat_radiance=2000.0).render_signals(0, frame_type="flat")
    assert brighter.float().mean() > digital.float().mean()


def test_flat_requires_radiance():
    with pytest.raises(ValueError, match="flat_radiance"):
        _scene().render_signals(0, frame_type="flat")


def test_invalid_frame_type_raises():
    with pytest.raises(ValueError, match="frame_type"):
        _scene().render_signals(0, frame_type="darkk")


def test_frame_types_export():
    assert sdasim.FRAME_TYPES == ("light", "dark", "bias", "flat")
