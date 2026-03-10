"""Gaussian splatting kernel — the core rendering primitive.

Replaces satsim's oversample -> FFT convolve -> downsample pipeline with
direct analytical Gaussian splatting at native resolution.

For torch.compile() compatibility, pass radius explicitly to avoid graph breaks.
"""

from __future__ import annotations

import torch
from torch import Tensor


def splat_gaussians(
    height: int,
    width: int,
    positions: Tensor,
    intensities: Tensor,
    sigma: Tensor | float,
    radius: int | None = None,
) -> Tensor:
    """Splat Gaussian PSFs onto an image at native resolution.

    For each source, computes a (2R+1)x(2R+1) Gaussian footprint centered on
    its sub-pixel position and accumulates onto the output image via scatter_add.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        positions: (N, 2) sub-pixel (row, col) positions.
        intensities: (N,) photoelectron counts per source.
        sigma: PSF sigma in pixels. Scalar or (N,) per-source.
        radius: Footprint radius in pixels. Defaults to ceil(3 * max(sigma)).
            For torch.compile(), pass this explicitly to avoid graph breaks.

    Returns:
        (H, W) accumulated image tensor.
    """
    device = positions.device
    dtype = positions.dtype
    N = positions.shape[0]

    if N == 0:
        return torch.zeros(height, width, dtype=dtype, device=device)

    # Normalize sigma to (N,) tensor on the same device as positions
    if not isinstance(sigma, Tensor):
        sigma = torch.tensor(sigma, dtype=dtype, device=device)
    else:
        sigma = sigma.to(dtype=dtype, device=device)
    if sigma.dim() == 0:
        sigma = sigma.expand(N)

    if radius is None:
        # Graph break here for torch.compile — pass radius explicitly to avoid
        radius = int(torch.ceil(3.0 * sigma.max()).item())

    # Build local pixel grid offsets: (2R+1,)
    offsets = torch.arange(-radius, radius + 1, dtype=dtype, device=device)  # (K,)
    K = offsets.shape[0]

    # Source positions
    row_center = positions[:, 0].unsqueeze(1)  # (N, 1)
    col_center = positions[:, 1].unsqueeze(1)  # (N, 1)

    # Integer pixel positions in the footprint
    row_int = row_center.round().long() + offsets.long().unsqueeze(0)  # (N, K)
    col_int = col_center.round().long() + offsets.long().unsqueeze(0)  # (N, K)

    # Distance from sub-pixel center to integer pixel centers
    dr = row_int.float() - row_center  # (N, K)
    dc = col_int.float() - col_center  # (N, K)

    # Separable 1D Gaussian components: (N, K)
    sigma_col = sigma.unsqueeze(1)  # (N, 1)
    gauss_r = torch.exp(-0.5 * (dr / sigma_col) ** 2)  # (N, K)
    gauss_c = torch.exp(-0.5 * (dc / sigma_col) ** 2)  # (N, K)

    # 2D Gaussian via outer product: (N, K, K) = (N, K, 1) * (N, 1, K)
    gauss_2d = gauss_r.unsqueeze(2) * gauss_c.unsqueeze(1)  # (N, K, K)

    # Normalize each source's footprint to sum to 1
    gauss_sum = gauss_2d.sum(dim=(1, 2), keepdim=True).clamp(min=1e-12)
    gauss_2d = gauss_2d / gauss_sum

    # Scale by intensity: (N, K, K)
    gauss_2d = gauss_2d * intensities.unsqueeze(1).unsqueeze(2)

    # Compute absolute pixel indices: (N, K, K)
    row_idx = row_int.unsqueeze(2).expand(-1, -1, K)  # rows vary along dim 1
    col_idx = col_int.unsqueeze(1).expand(-1, K, -1)  # cols vary along dim 2

    # Flatten for scatter_add
    row_flat = row_idx.reshape(-1)
    col_flat = col_idx.reshape(-1)
    val_flat = gauss_2d.reshape(-1)

    # Bounds mask: zero out-of-bounds values instead of boolean indexing
    # (more compile-friendly than dynamic indexing)
    valid = (row_flat >= 0) & (row_flat < height) & (col_flat >= 0) & (col_flat < width)
    # Clamp indices to valid range (clamped values get zeroed by mask)
    row_safe = row_flat.clamp(0, height - 1)
    col_safe = col_flat.clamp(0, width - 1)
    val_safe = val_flat * valid.float()

    # Linear index and scatter
    linear_idx = row_safe * width + col_safe
    image = torch.zeros(height * width, dtype=dtype, device=device)
    image.scatter_add_(0, linear_idx, val_safe)

    return image.reshape(height, width)
