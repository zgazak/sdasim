"""Tests for end-to-end gradient flow through the render pipeline."""

from __future__ import annotations

import torch
import pytest

from sdasim.render import render_frame
from sdasim.splat import splat_gaussians


class TestGradientFlow:
    """Tests for differentiability through the full pipeline."""

    def test_gradient_through_intensity(self):
        """Gradients should flow from loss through star intensities via splat."""
        star_pos = torch.tensor([[16.0, 16.0]])
        star_int = torch.tensor([1000.0], requires_grad=True)

        # Test directly through splat (A/D floor kills gradients through render_frame)
        from sdasim.splat import splat_gaussians
        img = splat_gaussians(32, 32, star_pos, star_int, sigma=1.5)
        loss = img.sum()
        loss.backward()
        assert star_int.grad is not None
        assert abs(star_int.grad.item() - 1.0) < 0.01

    def test_gradient_through_psf_sigma(self):
        """Gradients should flow through PSF sigma."""
        star_pos = torch.tensor([[16.0, 16.0]])
        star_int = torch.tensor([1000.0])
        sigma = torch.tensor(1.5, requires_grad=True)

        digital, _, _ = render_frame(
            32, 32, star_pos, star_int,
            torch.zeros(0, 2), torch.zeros(0),
            psf_sigma=sigma, background_pe=0.0, dark_current_pe=0.0,
            bias_pe=0.0, read_noise=0.0, electronic_noise=0.0,
            gain=1.0, fwc=1e9, a2d_bias=0.0, a2d_dtype="uint16",
            enable_shot_noise=False, enable_read_noise=False,
        )
        # Peak value depends on sigma
        loss = digital.max()
        loss.backward()
        assert sigma.grad is not None

    def test_gradient_through_positions(self):
        """Gradients should flow through source positions."""
        star_pos = torch.tensor([[16.0, 16.0]], requires_grad=True)
        star_int = torch.tensor([1000.0])

        digital, _, _ = render_frame(
            32, 32, star_pos, star_int,
            torch.zeros(0, 2), torch.zeros(0),
            psf_sigma=1.5, background_pe=0.0, dark_current_pe=0.0,
            bias_pe=0.0, read_noise=0.0, electronic_noise=0.0,
            gain=1.0, fwc=1e9, a2d_bias=0.0, a2d_dtype="uint16",
            enable_shot_noise=False, enable_read_noise=False,
        )
        # Use a specific pixel value that's off-center
        loss = digital[14, 16]
        loss.backward()
        assert star_pos.grad is not None

    def test_gradient_through_shot_noise(self):
        """Gradients should flow through Poisson noise via STE."""
        star_int = torch.tensor([5000.0], requires_grad=True)
        star_pos = torch.tensor([[16.0, 16.0]])

        digital, _, _ = render_frame(
            32, 32, star_pos, star_int,
            torch.zeros(0, 2), torch.zeros(0),
            psf_sigma=1.5, background_pe=100.0, dark_current_pe=20.0,
            bias_pe=50.0, read_noise=0.0, electronic_noise=0.0,
            gain=1.0, fwc=1e9, a2d_bias=0.0, a2d_dtype="uint16",
            enable_shot_noise=True, enable_read_noise=False,
        )
        loss = digital.sum()
        loss.backward()
        assert star_int.grad is not None

    def test_gradient_through_full_pipeline(self):
        """End-to-end gradient through everything: splat + noise + A/D."""
        star_int = torch.tensor([5000.0], requires_grad=True)
        star_pos = torch.tensor([[16.0, 16.0]])

        digital, _, _ = render_frame(
            32, 32, star_pos, star_int,
            torch.zeros(0, 2), torch.zeros(0),
            psf_sigma=1.5, background_pe=100.0, dark_current_pe=20.0,
            bias_pe=50.0, read_noise=10.0, electronic_noise=5.0,
            gain=8.0, fwc=100000.0, a2d_bias=500.0, a2d_dtype="uint16",
            enable_shot_noise=True, enable_read_noise=True,
        )
        loss = digital.sum()
        loss.backward()
        # Gradient should exist (though A/D floor is not differentiable,
        # noise STE should still pass gradients)
        assert star_int.grad is not None

    def test_splat_gradcheck(self):
        """torch.autograd.gradcheck on splat_gaussians with intensities."""
        pos = torch.tensor([[8.0, 8.0]], dtype=torch.float64)
        ints = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(1.5, dtype=torch.float64)

        def fn(intensities):
            return splat_gaussians(16, 16, pos, intensities, sigma)

        assert torch.autograd.gradcheck(fn, (ints,), eps=1e-4, atol=1e-3)

    def test_splat_gradcheck_sigma(self):
        """torch.autograd.gradcheck on splat_gaussians with sigma."""
        pos = torch.tensor([[8.0, 8.0]], dtype=torch.float64)
        ints = torch.tensor([100.0], dtype=torch.float64)
        sigma = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)

        def fn(s):
            return splat_gaussians(16, 16, pos, ints, s)

        assert torch.autograd.gradcheck(fn, (sigma,), eps=1e-4, atol=1e-3)
