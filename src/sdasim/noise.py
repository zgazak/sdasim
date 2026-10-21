"""Differentiable noise models.

Poisson noise uses straight-through estimator (STE) for gradients.
Gaussian noise uses the reparameterization trick.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Above this rate, Poisson is sampled as round(N(lambda, sqrt(lambda))): the
# normal approximation's skew is ~1/sqrt(lambda) (<1% here) while torch.poisson
# gets expensive — the daytime-sky regime (sky-filled pixels at 1e4-1e7 e-)
# is exactly where this bites.
POISSON_GAUSSIAN_THRESHOLD = 1.0e4


class _PoissonSTE(torch.autograd.Function):
    """Poisson sampling with straight-through estimator gradient."""

    @staticmethod
    def forward(ctx, signal: Tensor) -> Tensor:
        lam = signal.clamp(min=0.0)
        big = lam > POISSON_GAUSSIAN_THRESHOLD
        if bool(big.any()):
            gauss = torch.round(lam + torch.sqrt(lam) * torch.randn_like(lam)).clamp(min=0.0)
            small = torch.poisson(torch.where(big, torch.zeros_like(lam), lam))
            return torch.where(big, gauss, small)
        return torch.poisson(lam)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        # STE: pass gradient through unchanged
        return grad_output


def poisson_noise(signal: Tensor) -> Tensor:
    """Apply Poisson noise with differentiable STE gradient.

    Args:
        signal: Input signal in photoelectrons (non-negative).

    Returns:
        Poisson-sampled signal, same shape.
    """
    return _PoissonSTE.apply(signal)


def gaussian_noise(signal: Tensor, sigma: float | Tensor) -> Tensor:
    """Apply additive Gaussian noise via reparameterization trick.

    Args:
        signal: Input signal.
        sigma: Noise standard deviation (scalar or tensor).

    Returns:
        Signal + Gaussian noise.
    """
    if isinstance(sigma, (int, float)) and sigma == 0.0:
        return signal
    return signal + sigma * torch.randn_like(signal)
