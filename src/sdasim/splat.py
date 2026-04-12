"""Gaussian splatting kernel — the core rendering primitive.

Replaces satsim's oversample -> FFT convolve -> downsample pipeline with
direct analytical Gaussian splatting at native resolution.

For torch.compile() compatibility, pass radius explicitly to avoid graph breaks.

Two entry points:
  - splat_gaussians(): single-frame (H, W) output.
  - splat_gaussians_batched(): multi-frame (B, H, W) output in a single
    scatter_add call. The batched version is what enables amortizing kernel
    launch overhead across an entire training batch of heterogeneous scenes.
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


def splat_gaussians_batched(
    batch_size: int,
    height: int,
    width: int,
    positions: Tensor,
    intensities: Tensor,
    frame_ids: Tensor,
    sigma: Tensor | float,
    radius: int | None = None,
) -> Tensor:
    """Splat Gaussians for multiple frames into a single (B, H, W) tensor.

    All sources across every frame in the batch are passed in as one flat
    list with a per-source `frame_ids` tag. Rendering happens via one
    scatter_add into a flat (B*H*W,) buffer, so the entire batch costs one
    kernel launch instead of B. For small frames with moderate source counts
    this is a large speedup because GPU kernel launch overhead dominates.

    Args:
        batch_size: Number of output frames.
        height: Image height in pixels.
        width: Image width in pixels.
        positions: (N_total, 2) sub-pixel (row, col) positions across all frames.
        intensities: (N_total,) photoelectron counts per source.
        frame_ids: (N_total,) int64, which frame each source belongs to.
        sigma: PSF sigma in pixels. May be:
            - scalar (applied to every source),
            - (N_total,) per-source,
            - (batch_size,) per-frame (gathered by frame_ids).
        radius: Footprint radius in pixels. Defaults to ceil(3 * max(sigma)).

    Returns:
        (batch_size, height, width) accumulated image tensor.
    """
    device = positions.device
    dtype = positions.dtype
    N = positions.shape[0]

    if N == 0:
        return torch.zeros(batch_size, height, width, dtype=dtype, device=device)

    # Normalize sigma to (N,)
    if not isinstance(sigma, Tensor):
        sigma = torch.tensor(sigma, dtype=dtype, device=device)
    else:
        sigma = sigma.to(dtype=dtype, device=device)
    if sigma.dim() == 0:
        sigma = sigma.expand(N)
    elif sigma.dim() == 1 and sigma.shape[0] == batch_size and batch_size != N:
        sigma = sigma[frame_ids]
    # else: assume already (N,)

    frame_ids = frame_ids.to(device=device, dtype=torch.long)

    if radius is None:
        radius = int(torch.ceil(3.0 * sigma.max()).item())

    offsets = torch.arange(-radius, radius + 1, dtype=dtype, device=device)  # (K,)
    K = offsets.shape[0]

    row_center = positions[:, 0].unsqueeze(1)  # (N, 1)
    col_center = positions[:, 1].unsqueeze(1)  # (N, 1)

    row_int = row_center.round().long() + offsets.long().unsqueeze(0)  # (N, K)
    col_int = col_center.round().long() + offsets.long().unsqueeze(0)  # (N, K)

    dr = row_int.to(dtype) - row_center  # (N, K)
    dc = col_int.to(dtype) - col_center  # (N, K)

    sigma_col = sigma.unsqueeze(1)  # (N, 1)
    gauss_r = torch.exp(-0.5 * (dr / sigma_col) ** 2)  # (N, K)
    gauss_c = torch.exp(-0.5 * (dc / sigma_col) ** 2)  # (N, K)

    gauss_2d = gauss_r.unsqueeze(2) * gauss_c.unsqueeze(1)  # (N, K, K)
    gauss_sum = gauss_2d.sum(dim=(1, 2), keepdim=True).clamp(min=1e-12)
    gauss_2d = gauss_2d / gauss_sum
    gauss_2d = gauss_2d * intensities.unsqueeze(1).unsqueeze(2)

    row_idx = row_int.unsqueeze(2).expand(-1, -1, K)  # (N, K, K)
    col_idx = col_int.unsqueeze(1).expand(-1, K, -1)
    frame_exp = frame_ids.view(-1, 1, 1).expand(N, K, K)

    row_flat = row_idx.reshape(-1)
    col_flat = col_idx.reshape(-1)
    frame_flat = frame_exp.reshape(-1)
    val_flat = gauss_2d.reshape(-1)

    valid = (
        (row_flat >= 0) & (row_flat < height)
        & (col_flat >= 0) & (col_flat < width)
    )
    row_safe = row_flat.clamp(0, height - 1)
    col_safe = col_flat.clamp(0, width - 1)
    val_safe = val_flat * valid.to(dtype)

    linear_idx = (
        frame_flat * (height * width) + row_safe * width + col_safe
    )
    image = torch.zeros(batch_size * height * width, dtype=dtype, device=device)
    image.scatter_add_(0, linear_idx, val_safe)

    return image.reshape(batch_size, height, width)
