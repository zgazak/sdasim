"""Tests for differentiable noise models."""

from __future__ import annotations

import torch

from sdasim.noise import gaussian_noise, poisson_noise


class TestPoissonNoise:
    """Tests for Poisson noise with STE gradient."""

    def test_mean_matches_signal(self):
        """Poisson mean should match input signal over many samples."""
        signal = torch.full((1000,), 100.0)
        samples = torch.stack([poisson_noise(signal) for _ in range(200)])
        mean = samples.mean(dim=0)
        # Mean should be close to 100
        assert abs(mean.mean().item() - 100.0) < 2.0

    def test_variance_matches_signal(self):
        """Poisson variance should equal the mean (signal)."""
        signal = torch.full((1000,), 100.0)
        samples = torch.stack([poisson_noise(signal) for _ in range(500)])
        var = samples.var(dim=0)
        # Variance should be close to 100
        assert abs(var.mean().item() - 100.0) < 10.0

    def test_non_negative_output(self):
        """Poisson output should be non-negative."""
        signal = torch.full((100,), 10.0)
        result = poisson_noise(signal)
        assert (result >= 0).all()

    def test_zero_signal(self):
        """Zero signal should produce zero output."""
        signal = torch.zeros(100)
        result = poisson_noise(signal)
        assert (result == 0).all()

    def test_negative_signal_clamped(self):
        """Negative signal should be clamped to zero before sampling."""
        signal = torch.tensor([-10.0, -5.0, 0.0, 5.0])
        result = poisson_noise(signal)
        assert (result >= 0).all()

    def test_ste_gradient(self):
        """Gradient should pass through via STE."""
        signal = torch.tensor([100.0], requires_grad=True)
        result = poisson_noise(signal)
        result.backward()
        # STE: gradient should be 1.0
        assert signal.grad is not None
        assert signal.grad.item() == 1.0


class TestGaussianNoise:
    """Tests for Gaussian noise with reparameterization."""

    def test_mean_preserved(self):
        """Mean of noisy signal should match original signal."""
        signal = torch.full((10000,), 50.0)
        samples = torch.stack([gaussian_noise(signal, sigma=10.0) for _ in range(100)])
        assert abs(samples.mean().item() - 50.0) < 1.0

    def test_variance_matches_sigma(self):
        """Variance of noise should match sigma^2."""
        signal = torch.zeros(10000)
        sigma = 10.0
        samples = torch.stack([gaussian_noise(signal, sigma) for _ in range(100)])
        var = samples.var().item()
        assert abs(var - sigma**2) < 10.0

    def test_zero_sigma(self):
        """Zero sigma should return signal unchanged."""
        signal = torch.tensor([1.0, 2.0, 3.0])
        result = gaussian_noise(signal, sigma=0.0)
        assert torch.equal(result, signal)

    def test_gradient_flows(self):
        """Gradient should flow through reparameterization."""
        signal = torch.tensor([50.0], requires_grad=True)
        result = gaussian_noise(signal, sigma=10.0)
        result.backward()
        assert signal.grad is not None
        assert signal.grad.item() == 1.0
