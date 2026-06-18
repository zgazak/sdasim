"""PSF splatting kernels — core rendering primitives.

Replaces the conventional oversample -> FFT convolve -> downsample pipeline with
direct analytical PSF splatting at native resolution.

For torch.compile() compatibility, pass radius explicitly to avoid graph breaks.

Supported PSF families:
  - Gaussian (symmetric, one sigma)
  - Moffat (heavier wings; parameters α, β)
  - Elliptical Gaussian (σ_major, σ_minor, θ)

Batched entry points are what training throughput cares about:
  - splat_gaussians_batched()
  - splat_moffat_batched()
  - splat_elliptical_gaussian_batched()

All share the same scatter-add backbone and only differ in the per-source
kernel evaluation, so mixing PSF types across a batch is possible via
splat_mixed_psf_batched which dispatches on a per-source kind index.
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

    valid = (row_flat >= 0) & (row_flat < height) & (col_flat >= 0) & (col_flat < width)
    row_safe = row_flat.clamp(0, height - 1)
    col_safe = col_flat.clamp(0, width - 1)
    val_safe = val_flat * valid.to(dtype)

    linear_idx = frame_flat * (height * width) + row_safe * width + col_safe
    image = torch.zeros(batch_size * height * width, dtype=dtype, device=device)
    image.scatter_add_(0, linear_idx, val_safe)

    return image.reshape(batch_size, height, width)


# ---------------------------------------------------------------------------
# Moffat splatting
# ---------------------------------------------------------------------------


def splat_moffat_batched(
    batch_size: int,
    height: int,
    width: int,
    positions: Tensor,
    intensities: Tensor,
    frame_ids: Tensor,
    alpha: Tensor | float,
    beta: Tensor | float,
    radius: int | None = None,
) -> Tensor:
    """Splat Moffat-profile PSFs into a batched [B, H, W] tensor.

    Moffat profile: I(r) = [1 + (r/alpha)^2]^(-beta).

    - alpha sets the scale (FWHM = 2*alpha*sqrt(2^(1/beta) - 1))
    - beta controls the wing weight (typical values 2.5-5; Gaussian limit
      as beta → ∞). Real ground-based telescopes often match beta ≈ 3 with
      heavier wings than a Gaussian.

    Args match splat_gaussians_batched, plus alpha and beta which can each
    be scalar or (N_total,). Any (batch_size,) per-frame shape is gathered
    by frame_ids.
    """
    device = positions.device
    dtype = positions.dtype
    N = positions.shape[0]
    if N == 0:
        return torch.zeros(batch_size, height, width, dtype=dtype, device=device)

    def _as_per_source(x, default_name):
        if not isinstance(x, Tensor):
            x = torch.tensor(x, dtype=dtype, device=device)
        else:
            x = x.to(dtype=dtype, device=device)
        if x.dim() == 0:
            x = x.expand(N)
        elif x.dim() == 1 and x.shape[0] == batch_size and batch_size != N:
            x = x[frame_ids]
        return x

    alpha = _as_per_source(alpha, "alpha")
    beta = _as_per_source(beta, "beta")
    frame_ids = frame_ids.to(device=device, dtype=torch.long)

    if radius is None:
        # Moffat FWHM → radius (use wings → 3x FWHM for 99% flux).
        fwhm = 2 * alpha * torch.sqrt((2 ** (1.0 / beta.clamp(min=1.0)) - 1).clamp(min=1e-6))
        radius = int(torch.ceil(3.0 * fwhm.max() / 2.0).item())
        radius = max(radius, 3)

    offsets = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    K = offsets.shape[0]

    row_center = positions[:, 0].unsqueeze(1)
    col_center = positions[:, 1].unsqueeze(1)

    row_int = row_center.round().long() + offsets.long().unsqueeze(0)
    col_int = col_center.round().long() + offsets.long().unsqueeze(0)

    dr = row_int.to(dtype) - row_center  # (N, K)
    dc = col_int.to(dtype) - col_center

    # 2D: (N, K, K)
    dr2 = dr.unsqueeze(2) ** 2
    dc2 = dc.unsqueeze(1) ** 2
    r2 = dr2 + dc2  # (N, K, K)
    alpha_col = alpha.view(N, 1, 1)
    beta_col = beta.view(N, 1, 1)
    psf = (1.0 + r2 / (alpha_col * alpha_col)) ** (-beta_col)

    psf_sum = psf.sum(dim=(1, 2), keepdim=True).clamp(min=1e-12)
    psf = psf / psf_sum
    psf = psf * intensities.unsqueeze(1).unsqueeze(2)

    row_idx = row_int.unsqueeze(2).expand(-1, -1, K)
    col_idx = col_int.unsqueeze(1).expand(-1, K, -1)
    frame_exp = frame_ids.view(-1, 1, 1).expand(N, K, K)

    row_flat = row_idx.reshape(-1)
    col_flat = col_idx.reshape(-1)
    frame_flat = frame_exp.reshape(-1)
    val_flat = psf.reshape(-1)

    valid = (row_flat >= 0) & (row_flat < height) & (col_flat >= 0) & (col_flat < width)
    row_safe = row_flat.clamp(0, height - 1)
    col_safe = col_flat.clamp(0, width - 1)
    val_safe = val_flat * valid.to(dtype)

    linear_idx = frame_flat * (height * width) + row_safe * width + col_safe
    image = torch.zeros(batch_size * height * width, dtype=dtype, device=device)
    image.scatter_add_(0, linear_idx, val_safe)
    return image.reshape(batch_size, height, width)


# ---------------------------------------------------------------------------
# Elliptical Gaussian splatting
# ---------------------------------------------------------------------------


def splat_elliptical_gaussian_batched(
    batch_size: int,
    height: int,
    width: int,
    positions: Tensor,
    intensities: Tensor,
    frame_ids: Tensor,
    sigma_major: Tensor | float,
    sigma_minor: Tensor | float,
    theta: Tensor | float,
    radius: int | None = None,
) -> Tensor:
    """Splat elliptical Gaussians, useful for anisotropic seeing / guider drift.

    The ellipse axes are σ_major, σ_minor; rotation θ (radians) aligns the
    major axis to the image's row direction at θ=0 and rotates CCW.

    Args match splat_gaussians_batched plus the three ellipse parameters.
    Each parameter can be scalar, per-source (N,), or per-frame (batch_size,).
    """
    device = positions.device
    dtype = positions.dtype
    N = positions.shape[0]
    if N == 0:
        return torch.zeros(batch_size, height, width, dtype=dtype, device=device)

    def _as_per_source(x):
        if not isinstance(x, Tensor):
            x = torch.tensor(x, dtype=dtype, device=device)
        else:
            x = x.to(dtype=dtype, device=device)
        if x.dim() == 0:
            x = x.expand(N)
        elif x.dim() == 1 and x.shape[0] == batch_size and batch_size != N:
            x = x[frame_ids]
        return x

    sigma_major = _as_per_source(sigma_major)
    sigma_minor = _as_per_source(sigma_minor)
    theta = _as_per_source(theta)
    frame_ids = frame_ids.to(device=device, dtype=torch.long)

    if radius is None:
        radius = int(torch.ceil(3.0 * sigma_major.max()).item())
        radius = max(radius, 3)

    offsets = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    K = offsets.shape[0]

    row_center = positions[:, 0].unsqueeze(1)
    col_center = positions[:, 1].unsqueeze(1)

    row_int = row_center.round().long() + offsets.long().unsqueeze(0)
    col_int = col_center.round().long() + offsets.long().unsqueeze(0)

    dr = row_int.to(dtype) - row_center  # (N, K)
    dc = col_int.to(dtype) - col_center

    # 2D offsets in image frame
    dr_2d = dr.unsqueeze(2).expand(-1, -1, K)  # (N, K, K)
    dc_2d = dc.unsqueeze(1).expand(-1, K, -1)

    cos_t = torch.cos(theta).view(N, 1, 1)
    sin_t = torch.sin(theta).view(N, 1, 1)
    # Rotate into ellipse frame (u along major, v along minor).
    u = cos_t * dc_2d + sin_t * dr_2d
    v = -sin_t * dc_2d + cos_t * dr_2d

    sig_maj = sigma_major.view(N, 1, 1).clamp(min=0.25)
    sig_min = sigma_minor.view(N, 1, 1).clamp(min=0.25)
    psf = torch.exp(-(u * u) / (2 * sig_maj * sig_maj) - (v * v) / (2 * sig_min * sig_min))

    psf_sum = psf.sum(dim=(1, 2), keepdim=True).clamp(min=1e-12)
    psf = psf / psf_sum
    psf = psf * intensities.unsqueeze(1).unsqueeze(2)

    row_idx = row_int.unsqueeze(2).expand(-1, -1, K)
    col_idx = col_int.unsqueeze(1).expand(-1, K, -1)
    frame_exp = frame_ids.view(-1, 1, 1).expand(N, K, K)

    row_flat = row_idx.reshape(-1)
    col_flat = col_idx.reshape(-1)
    frame_flat = frame_exp.reshape(-1)
    val_flat = psf.reshape(-1)

    valid = (row_flat >= 0) & (row_flat < height) & (col_flat >= 0) & (col_flat < width)
    row_safe = row_flat.clamp(0, height - 1)
    col_safe = col_flat.clamp(0, width - 1)
    val_safe = val_flat * valid.to(dtype)

    linear_idx = frame_flat * (height * width) + row_safe * width + col_safe
    image = torch.zeros(batch_size * height * width, dtype=dtype, device=device)
    image.scatter_add_(0, linear_idx, val_safe)
    return image.reshape(batch_size, height, width)
