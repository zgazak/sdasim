"""Tests for the Gaussian splatting kernel."""

from __future__ import annotations

import torch

from sdasim.splat import splat_gaussians


class TestSplatGaussians:
    """Tests for splat_gaussians."""

    def test_single_source_centered(self):
        """A single centered source should conserve total energy."""
        pos = torch.tensor([[16.0, 16.0]])
        intensity = torch.tensor([1000.0])
        img = splat_gaussians(32, 32, pos, intensity, sigma=1.5)
        assert img.shape == (32, 32)
        # Total energy should equal input intensity (within footprint)
        assert abs(img.sum().item() - 1000.0) < 1.0

    def test_energy_conservation_multiple(self):
        """Total energy from multiple sources should be conserved."""
        pos = torch.tensor([[10.0, 10.0], [20.0, 20.0], [15.0, 15.0]])
        intensities = torch.tensor([500.0, 300.0, 200.0])
        img = splat_gaussians(32, 32, pos, intensities, sigma=1.5)
        total = intensities.sum().item()
        assert abs(img.sum().item() - total) < 1.0

    def test_subpixel_positioning(self):
        """Sub-pixel shifts should redistribute energy smoothly."""
        # Centered on pixel
        pos_on = torch.tensor([[16.0, 16.0]])
        img_on = splat_gaussians(32, 32, pos_on, torch.tensor([1000.0]), sigma=1.5)

        # Shifted by 0.5
        pos_off = torch.tensor([[16.5, 16.5]])
        img_off = splat_gaussians(32, 32, pos_off, torch.tensor([1000.0]), sigma=1.5)

        # Both should conserve energy
        assert abs(img_on.sum().item() - 1000.0) < 1.0
        assert abs(img_off.sum().item() - 1000.0) < 1.0

        # Peak should be lower for off-center (energy spread)
        assert img_off.max() < img_on.max()

    def test_boundary_source(self):
        """Sources near edges should clip gracefully without error."""
        pos = torch.tensor([[0.0, 0.0]])
        img = splat_gaussians(32, 32, pos, torch.tensor([1000.0]), sigma=1.5)
        assert img.shape == (32, 32)
        # Some energy lost to clipping
        assert img.sum().item() < 1000.0
        assert img.sum().item() > 0.0

    def test_out_of_bounds_source(self):
        """Fully out-of-bounds sources should produce zero image."""
        pos = torch.tensor([[-100.0, -100.0]])
        img = splat_gaussians(32, 32, pos, torch.tensor([1000.0]), sigma=1.5)
        assert img.sum().item() == 0.0

    def test_empty_sources(self):
        """No sources should produce zero image."""
        pos = torch.zeros(0, 2)
        intensity = torch.zeros(0)
        img = splat_gaussians(32, 32, pos, intensity, sigma=1.5)
        assert img.shape == (32, 32)
        assert img.sum().item() == 0.0

    def test_per_source_sigma(self):
        """Per-source sigma should produce different PSF widths."""
        pos = torch.tensor([[16.0, 8.0], [16.0, 24.0]])
        intensities = torch.tensor([1000.0, 1000.0])
        sigma = torch.tensor([0.5, 3.0])
        img = splat_gaussians(32, 32, pos, intensities, sigma=sigma)
        # Narrow source should have higher peak
        narrow_peak = img[16, 8]
        wide_peak = img[16, 24]
        assert narrow_peak > wide_peak

    def test_differentiable_positions(self):
        """Gradients should flow through positions."""
        pos = torch.tensor([[16.0, 16.0]], requires_grad=True)
        intensity = torch.tensor([1000.0])
        img = splat_gaussians(32, 32, pos, intensity, sigma=1.5)
        loss = img.sum()
        loss.backward()
        # Gradient exists (may be zero for centered source, but should not error)
        assert pos.grad is not None

    def test_differentiable_intensities(self):
        """Gradients should flow through intensities."""
        pos = torch.tensor([[16.0, 16.0]])
        intensity = torch.tensor([1000.0], requires_grad=True)
        img = splat_gaussians(32, 32, pos, intensity, sigma=1.5)
        loss = img.sum()
        loss.backward()
        assert intensity.grad is not None
        # d(sum)/d(intensity) should be ~1.0 (energy conservation)
        assert abs(intensity.grad.item() - 1.0) < 0.01

    def test_differentiable_sigma(self):
        """Gradients should flow through sigma."""
        pos = torch.tensor([[16.0, 16.0]])
        intensity = torch.tensor([1000.0])
        sigma = torch.tensor(1.5, requires_grad=True)
        img = splat_gaussians(32, 32, pos, intensity, sigma=sigma)
        # Peak value depends on sigma
        loss = img.max()
        loss.backward()
        assert sigma.grad is not None

    def test_custom_radius(self):
        """Custom radius should limit the footprint."""
        pos = torch.tensor([[16.0, 16.0]])
        intensity = torch.tensor([1000.0])
        # Small radius — some energy lost
        img_small = splat_gaussians(32, 32, pos, intensity, sigma=3.0, radius=2)
        img_large = splat_gaussians(32, 32, pos, intensity, sigma=3.0, radius=10)
        assert img_small.sum().item() < img_large.sum().item()

    def test_non_negative_output(self):
        """Output should be non-negative everywhere."""
        torch.manual_seed(0)
        pos = torch.rand(50, 2) * 30 + 1
        intensities = torch.rand(50) * 1000 + 1
        img = splat_gaussians(32, 32, pos, intensities, sigma=1.5)
        assert (img >= 0).all()
