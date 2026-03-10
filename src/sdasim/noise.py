"""Differentiable noise models.

Poisson noise uses straight-through estimator (STE) for gradients.
Gaussian noise uses the reparameterization trick.
"""

from __future__ import annotations

import torch
from torch import Tensor


class _PoissonSTE(torch.autograd.Function):
    """Poisson sampling with straight-through estimator gradient."""

    @staticmethod
    def forward(ctx, signal: Tensor) -> Tensor:
        return torch.poisson(signal.clamp(min=0.0))

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
