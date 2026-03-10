"""Tests for target trajectory computation."""

from __future__ import annotations

import torch
import pytest

from sdasim.config import TargetConfig, SensorConfig
from sdasim.targets import compute_target_positions


class TestComputeTargetPositions:
    """Tests for target position computation."""

    def test_single_static_target(self):
        """Static target at center."""
        targets = [TargetConfig(origin=[0.5, 0.5], velocity=[0.0, 0.0], mv=12.0)]
        sensor = SensorConfig(height=64, width=64, zeropoint=23.5, exposure=2.0)
        pos, ints, vel = compute_target_positions(targets, sensor, frame_idx=0, device="cpu")
        assert pos.shape == (1, 2)
        assert abs(pos[0, 0].item() - 32.0) < 0.01
        assert abs(pos[0, 1].item() - 32.0) < 0.01
        assert ints.shape == (1,)
        assert ints[0].item() > 0
        assert vel.shape == (1, 2)

    def test_moving_target(self):
        """Target should move between frames."""
        targets = [TargetConfig(origin=[0.5, 0.5], velocity=[10.0, 5.0], mv=12.0)]
        sensor = SensorConfig(height=64, width=64, exposure=1.0, gap=0.0, zeropoint=23.5)
        pos0, _, vel0 = compute_target_positions(targets, sensor, frame_idx=0, device="cpu")
        pos1, _, _ = compute_target_positions(targets, sensor, frame_idx=1, device="cpu")
        # At frame 1: position = origin + velocity * 1.0
        assert pos1[0, 0].item() > pos0[0, 0].item()
        assert vel0[0, 0].item() == 10.0

    def test_no_targets(self):
        """Empty target list should return empty tensors."""
        sensor = SensorConfig()
        pos, ints, vel = compute_target_positions([], sensor, device="cpu")
        assert pos.shape == (0, 2)
        assert ints.shape == (0,)
        assert vel.shape == (0, 2)

    def test_multiple_targets(self):
        """Multiple targets should be returned."""
        targets = [
            TargetConfig(origin=[0.2, 0.3], mv=10.0),
            TargetConfig(origin=[0.8, 0.7], mv=14.0),
        ]
        sensor = SensorConfig(height=64, width=64, zeropoint=23.5, exposure=1.0)
        pos, ints, vel = compute_target_positions(targets, sensor, device="cpu")
        assert pos.shape == (2, 2)
        # Brighter target (lower mv) should have more PE
        assert ints[0].item() > ints[1].item()
