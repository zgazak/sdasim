"""Tests for star catalog generation."""

from __future__ import annotations

import torch
import pytest

from sdasim.stars import generate_random_stars


class TestGenerateRandomStars:
    """Tests for random star generation."""

    def test_basic_generation(self):
        """Should generate stars with correct shapes."""
        pos, ints = generate_random_stars(
            height=64, width=64,
            y_fov=0.5, x_fov=0.5,
            mv_bins=[10, 11, 12, 13],
            density=[5.0, 10.0, 20.0],
            zeropoint=23.5, exposure=2.0,
            seed=42, device="cpu",
        )
        assert pos.shape[1] == 2
        assert pos.shape[0] == ints.shape[0]
        assert pos.shape[0] > 0

    def test_deterministic_with_seed(self):
        """Same seed should produce same stars."""
        kwargs = dict(
            height=64, width=64, y_fov=0.5, x_fov=0.5,
            mv_bins=[10, 11, 12], density=[5.0, 10.0],
            zeropoint=23.5, exposure=2.0, seed=123, device="cpu",
        )
        pos1, int1 = generate_random_stars(**kwargs)
        pos2, int2 = generate_random_stars(**kwargs)
        assert torch.equal(pos1, pos2)
        assert torch.equal(int1, int2)

    def test_different_seeds(self):
        """Different seeds should produce different stars."""
        kwargs = dict(
            height=64, width=64, y_fov=0.5, x_fov=0.5,
            mv_bins=[10, 11, 12, 13], density=[5.0, 10.0, 20.0],
            zeropoint=23.5, exposure=2.0, device="cpu",
        )
        pos1, _ = generate_random_stars(seed=1, **kwargs)
        pos2, _ = generate_random_stars(seed=2, **kwargs)
        assert not torch.equal(pos1, pos2)

    def test_intensities_positive(self):
        """All star intensities should be positive."""
        _, ints = generate_random_stars(
            height=64, width=64,
            y_fov=0.5, x_fov=0.5,
            mv_bins=[10, 11, 12], density=[10.0, 20.0],
            zeropoint=23.5, exposure=2.0,
            seed=42, device="cpu",
        )
        assert (ints > 0).all()

    def test_density_scaling(self):
        """Higher density should produce more stars."""
        kwargs = dict(
            height=64, width=64, y_fov=1.0, x_fov=1.0,
            mv_bins=[10, 11], zeropoint=23.5, exposure=2.0,
            seed=42, device="cpu",
        )
        pos_low, _ = generate_random_stars(density=[1.0], **kwargs)
        pos_high, _ = generate_random_stars(density=[100.0], **kwargs)
        assert pos_high.shape[0] > pos_low.shape[0]

    def test_empty_density(self):
        """Zero density should produce no stars."""
        pos, ints = generate_random_stars(
            height=64, width=64,
            y_fov=0.5, x_fov=0.5,
            mv_bins=[10, 11], density=[0.0],
            zeropoint=23.5, exposure=2.0,
            seed=42, device="cpu",
        )
        assert pos.shape[0] == 0
        assert ints.shape[0] == 0
