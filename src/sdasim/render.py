"""Hot-path rendering pipeline.

Pipeline: splat stars -> splat targets -> add background/DC/bias ->
          Poisson noise (STE) -> read noise (reparam) -> A/D.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from sdasim.splat import splat_gaussians
from sdasim.noise import poisson_noise, gaussian_noise
from sdasim.fpa import analog_to_digital


def expand_motion(
    positions: Tensor,
    intensities: Tensor,
    velocity: list[float] | Tensor | None,
    rotation: float,
    t_start: float,
    t_end: float,
    t_osf: int,
    center: tuple[float, float],
) -> tuple[Tensor, Tensor]:
    """Expand sources into sub-sources along motion path.

    Vectorized temporal integration: N sources -> N*t_osf sub-sources.

    Args:
        positions: (N, 2) starting positions (row, col).
        intensities: (N,) total PE per source for this exposure.
        velocity: Motion in pixels/sec. Either:
            - [row_rate, col_rate] single velocity for all sources, or
            - (N, 2) tensor of per-source velocities.
        rotation: Rotation rate in rad/sec.
        t_start: Start time of exposure in seconds.
        t_end: End time of exposure in seconds.
        t_osf: Temporal oversampling factor (number of sub-steps).
        center: (row, col) center of rotation.

    Returns:
        (expanded_positions, expanded_intensities):
            (N*t_osf, 2) and (N*t_osf,) tensors.
    """
    if t_osf <= 1:
        return positions, intensities

    device = positions.device
    dtype = positions.dtype
    N = positions.shape[0]

    if N == 0:
        return positions, intensities

    # Time samples: midpoints of uniform sub-intervals
    t_steps = torch.linspace(t_start, t_end, t_osf + 1, device=device, dtype=dtype)
    t_mid = (t_steps[:-1] + t_steps[1:]) / 2.0  # (t_osf,)

    # Broadcast: positions (N, 1, 2), t_mid (1, t_osf)
    pos = positions.unsqueeze(1).expand(-1, t_osf, -1).clone()  # (N, t_osf, 2)

    # Apply translation
    if velocity is not None:
        if not isinstance(velocity, Tensor):
            velocity = torch.tensor(velocity, dtype=dtype, device=device)
        else:
            velocity = velocity.to(dtype=dtype, device=device)
        dt = t_mid.unsqueeze(0) - t_start  # (1, t_osf)
        if velocity.dim() == 1:
            # Single velocity for all sources: (2,)
            pos[..., 0] = pos[..., 0] + velocity[0] * dt
            pos[..., 1] = pos[..., 1] + velocity[1] * dt
        else:
            # Per-source velocity: (N, 2)
            pos[..., 0] = pos[..., 0] + velocity[:, 0:1] * dt  # (N, 1) * (1, t_osf)
            pos[..., 1] = pos[..., 1] + velocity[:, 1:2] * dt

    # Apply rotation about center
    if rotation != 0.0:
        dt = t_mid.unsqueeze(0) - t_start  # (1, t_osf)
        angle = rotation * dt  # (1, t_osf)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        cr, cc = center
        r_rel = pos[..., 0] - cr  # (N, t_osf)
        c_rel = pos[..., 1] - cc

        pos[..., 0] = cos_a * r_rel - sin_a * c_rel + cr
        pos[..., 1] = sin_a * r_rel + cos_a * c_rel + cc

    # Flatten: (N, t_osf, 2) -> (N*t_osf, 2)
    expanded_pos = pos.reshape(N * t_osf, 2)
    # Each sub-source gets intensity / t_osf
    expanded_int = (intensities / t_osf).unsqueeze(1).expand(-1, t_osf).reshape(N * t_osf)

    return expanded_pos, expanded_int


def render_frame(
    height: int,
    width: int,
    star_positions: Tensor,
    star_intensities: Tensor,
    target_positions: Tensor,
    target_intensities: Tensor,
    psf_sigma: float | Tensor,
    background_pe: float,
    dark_current_pe: float,
    bias_pe: float,
    read_noise: float,
    electronic_noise: float,
    gain: float,
    fwc: float,
    a2d_bias: float,
    a2d_dtype: str,
    enable_shot_noise: bool = True,
    enable_read_noise: bool = True,
    star_velocity: list[float] | Tensor | None = None,
    star_rotation: float = 0.0,
    target_velocities: Tensor | None = None,
    t_osf: int = 1,
    target_t_osf: int | None = None,
    exposure: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render a single frame through the full pipeline.

    Args:
        height, width: Image dimensions.
        star_positions: (S, 2) star positions at frame start.
        star_intensities: (S,) star PE counts.
        target_positions: (T, 2) target positions at frame start.
        target_intensities: (T,) target PE counts.
        psf_sigma: PSF sigma in pixels.
        background_pe: Background PE per pixel.
        dark_current_pe: Dark current PE per pixel.
        bias_pe: Bias PE per pixel.
        read_noise: Read noise RMS in electrons.
        electronic_noise: Electronic noise RMS in electrons.
        gain: Electrons per DN.
        fwc: Full-well capacity.
        a2d_bias: A/D bias offset in electrons.
        a2d_dtype: Output dtype string.
        enable_shot_noise: Apply Poisson noise.
        enable_read_noise: Apply read/electronic noise.
        star_velocity: Star motion [row, col] pixels/sec (uniform for all stars).
        star_rotation: Star rotation rate rad/sec.
        target_velocities: (T, 2) per-target velocities in pixels/sec, or None.
        t_osf: Temporal oversampling factor for motion blur.
        exposure: Exposure time in seconds.

    Returns:
        (digital, star_signal, target_signal): Digital image, star-only PE, target-only PE.
    """
    device = star_positions.device
    center = (height / 2.0, width / 2.0)

    # --- Stars: within-frame motion blur ---
    has_star_motion = (
        star_rotation != 0.0
        or (star_velocity is not None and any(v != 0.0 for v in (
            star_velocity if isinstance(star_velocity, (list, tuple))
            else star_velocity.tolist()
        )))
    )
    if has_star_motion and t_osf > 1 and star_positions.shape[0] > 0:
        exp_pos, exp_int = expand_motion(
            star_positions, star_intensities,
            star_velocity, star_rotation,
            0.0, exposure, t_osf, center,
        )
    else:
        exp_pos, exp_int = star_positions, star_intensities

    star_signal = splat_gaussians(height, width, exp_pos, exp_int, psf_sigma)

    # --- Targets: per-target within-frame motion blur ---
    # Use target_t_osf if provided, otherwise auto-compute from velocity.
    tgt_osf = target_t_osf
    if tgt_osf is None and target_velocities is not None and target_positions.shape[0] > 0:
        # Auto: ~1 sub-step per pixel of max target motion during exposure
        max_speed = target_velocities.abs().max().item()
        streak_px = max_speed * exposure
        tgt_osf = max(1, int(streak_px * 2))  # 2 sub-steps per pixel for quality

    if (target_velocities is not None and tgt_osf is not None and tgt_osf > 1
            and target_positions.shape[0] > 0):
        # Per-target velocities — expand_motion handles (N, 2) velocity
        tgt_exp_pos, tgt_exp_int = expand_motion(
            target_positions, target_intensities,
            target_velocities, 0.0,
            0.0, exposure, tgt_osf, center,
        )
        target_signal = splat_gaussians(height, width, tgt_exp_pos, tgt_exp_int, psf_sigma)
    else:
        target_signal = splat_gaussians(
            height, width, target_positions, target_intensities, psf_sigma,
        )

    # Combine signal
    signal = star_signal + target_signal

    # Add background, dark current, bias (per pixel, uniform)
    signal = signal + background_pe + dark_current_pe + bias_pe

    # Shot noise (Poisson)
    if enable_shot_noise:
        signal = poisson_noise(signal)

    # Read + electronic noise (Gaussian)
    if enable_read_noise:
        rn_sigma = math.sqrt(read_noise**2 + electronic_noise**2)
        if rn_sigma > 0:
            signal = gaussian_noise(signal, rn_sigma)

    # A/D conversion
    digital = analog_to_digital(signal, gain, fwc, a2d_bias, a2d_dtype)

    return digital, star_signal, target_signal
