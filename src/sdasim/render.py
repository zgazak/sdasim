"""Hot-path rendering pipeline.

Pipeline: splat stars -> splat targets -> add background/DC/bias ->
          Poisson noise (STE) -> read noise (reparam) -> A/D.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from sdasim.fpa import analog_to_digital
from sdasim.noise import gaussian_noise, poisson_noise
from sdasim.splat import splat_gaussians

if TYPE_CHECKING:
    from sdasim.optics import OpticsModel


def motion_sampling_spacing(
    sigma: float,
    peak_pe: float,
    noise_pe: float,
    quality: float | None = None,
) -> float:
    """Largest sub-source spacing (px) whose sampling ripple hides in the noise.

    A chain of Gaussians (sigma s) spaced d apart carries a periodic brightness
    ripple of semi-amplitude a = 2*exp(-2*pi^2*s^2/d^2) (exact, by Poisson
    summation). Rather than fix d/s at a universal constant, derive the ripple
    each frame can actually afford: on the brightest streak, surface brightness
    peak_pe rides on per-pixel noise noise_pe, so ripple below ~half the noise
    is statistically invisible. Inverting the ripple law for that amplitude
    gives the spacing. Bright star on dark sky -> tighter sampling; faint
    streaks on bright sky -> coarser, for free.

    quality pins d = quality*sigma instead (reproducibility across scenes).
    The result is clamped to [1.3, 2.2]*sigma — the exponential makes margin
    cheap below and unacceptable above — and floored at 1 px, below which the
    pixel lattice itself dominates and finer sampling buys nothing.
    """
    if quality is None:
        a = min(0.5 * noise_pe / max(peak_pe, 1e-9), 0.1)
        q = math.pi * math.sqrt(2.0 / math.log(2.0 / a))
        quality = min(max(q, 1.3), 2.2)
    return max(quality * sigma, 1.0)


def resolve_temporal_osf(
    configured: int | None,
    velocity,
    rotation: float,
    exposure: float,
    spacing_px: float,
    r_max_px: float,
    n_sources: int,
    budget_sub_sources: int | None,
) -> tuple[int, bool]:
    """Resolve the star motion-blur sub-sampling for this frame.

    Returns (t_osf, clamped). An explicitly configured int wins verbatim.
    Auto: enough sub-sources that spacing along the longest streak stays at
    spacing_px. The streak bound folds in rotation via the farthest source
    (r_max_px from the rotation center): |v| + |omega|*r_max. Floored at 2
    whenever there is any motion — expand_motion places sub-sources at
    sub-interval midpoints, so t_osf=1 would shift a moving star's centroid
    from mid-exposure to exposure start (a half-streak astrometric jump).
    Capped by the sub-source budget (beading is preferred to an unbounded
    LEO-rate allocation), with clamped=True when the cap bites.
    """
    if configured is not None:
        return max(1, int(configured)), False
    v0, v1 = (float(velocity[0]), float(velocity[1])) if velocity is not None else (0.0, 0.0)
    speed = math.hypot(v0, v1) + abs(rotation) * r_max_px
    streak_px = speed * exposure
    if streak_px <= 1e-9:
        return 1, False
    want = max(2, math.ceil(streak_px / spacing_px))
    if budget_sub_sources is None or n_sources <= 0:
        return want, False
    cap = max(2, budget_sub_sources // n_sources)
    return min(want, cap), want > cap


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
    defer_read_noise: bool = False,
    star_velocity: list[float] | Tensor | None = None,
    star_rotation: float = 0.0,
    target_velocities: Tensor | None = None,
    t_osf: int = 1,
    target_t_osf: int | None = None,
    exposure: float = 1.0,
    optics: "OpticsModel | None" = None,
    defocus_um: float | Tensor = 0.0,
    psf_kernel: Tensor | None = None,
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
        optics: Optional OpticsModel; expands stars and targets into defocused
            sub-sources (composes with motion expansion).
        defocus_um: Commanded focal shift from best focus in microns (may be a
            scalar tensor to keep gradients to the focus command).
        psf_kernel: Optional (K, K) normalized PSF kernel (e.g. a diffraction
            kernel from sdasim.pupil for non-circular/segmented apertures).
            When given, sources are bilinear-deposited and FFT-convolved with
            the kernel instead of Gaussian-splatted (psf_sigma unused for the
            PSF; motion expansion still applies). Mutually exclusive with
            ``optics`` — the kernel already encodes diffraction + static WFE.

    Returns:
        (digital, star_signal, target_signal): Digital image, star-only PE, target-only PE.
    """
    if psf_kernel is not None and optics is not None:
        raise ValueError("psf_kernel and optics are mutually exclusive PSF models")
    center = (height / 2.0, width / 2.0)

    # --- Stars: within-frame motion blur ---
    has_star_motion = star_rotation != 0.0 or (
        star_velocity is not None
        and any(
            v != 0.0
            for v in (
                star_velocity
                if isinstance(star_velocity, (list, tuple))
                else star_velocity.tolist()
            )
        )
    )
    if has_star_motion and t_osf > 1 and star_positions.shape[0] > 0:
        exp_pos, exp_int = expand_motion(
            star_positions,
            star_intensities,
            star_velocity,
            star_rotation,
            0.0,
            exposure,
            t_osf,
            center,
        )
    else:
        exp_pos, exp_int = star_positions, star_intensities

    # Defocus/aberration expansion composes with motion expansion (pupil x time).
    star_optics_stats = None
    if optics is not None:
        exp_pos, exp_int = optics.expand(exp_pos, exp_int, defocus_um, center, psf_sigma=psf_sigma)
        if star_positions.shape[0] > 0:
            # Stars are the dominant population; snapshot their expansion stats
            # so the (often empty) target expansion below can't mask them in
            # the frame metadata (e.g. report clamped=False after the stars
            # were budget-clamped).
            star_optics_stats = (optics.last_k, optics.last_blur_px, optics.last_clamped)

    if psf_kernel is not None:
        from sdasim.empirical import render_empirical_signal

        star_signal = render_empirical_signal(height, width, exp_pos, exp_int, psf_kernel)
    else:
        star_signal = splat_gaussians(height, width, exp_pos, exp_int, psf_sigma)

    # --- Targets: per-target within-frame motion blur ---
    # Use target_t_osf if provided (Scene resolves it against the noise floor
    # and budget), otherwise fall back to the same spacing rule with a plain
    # sigma-relative spacing. Speed is the max VECTOR-NORM velocity (the old
    # per-axis abs().max() under-read diagonal streaks by up to sqrt(2)).
    tgt_osf = target_t_osf
    if tgt_osf is None and target_velocities is not None and target_positions.shape[0] > 0:
        max_speed = float(target_velocities.norm(dim=1).max().item())
        sigma_val = (
            float(psf_sigma.detach().reshape(()).max().item())
            if isinstance(psf_sigma, Tensor)
            else float(psf_sigma)
        )
        tgt_osf, _ = resolve_temporal_osf(
            None, [max_speed, 0.0], 0.0, exposure,
            max(1.5 * sigma_val, 1.0), 0.0, target_positions.shape[0], None,
        )

    if (
        target_velocities is not None
        and tgt_osf is not None
        and tgt_osf > 1
        and target_positions.shape[0] > 0
    ):
        # Per-target velocities — expand_motion handles (N, 2) velocity
        tgt_exp_pos, tgt_exp_int = expand_motion(
            target_positions,
            target_intensities,
            target_velocities,
            0.0,
            0.0,
            exposure,
            tgt_osf,
            center,
        )
    else:
        tgt_exp_pos, tgt_exp_int = target_positions, target_intensities

    if optics is not None:
        tgt_exp_pos, tgt_exp_int = optics.expand(
            tgt_exp_pos, tgt_exp_int, defocus_um, center, psf_sigma=psf_sigma
        )
        if star_optics_stats is not None:
            k, blur, clamped = star_optics_stats
            optics.last_k, optics.last_blur_px = k, blur
            optics.last_clamped = clamped or optics.last_clamped

    if psf_kernel is not None:
        from sdasim.empirical import render_empirical_signal

        target_signal = render_empirical_signal(height, width, tgt_exp_pos, tgt_exp_int, psf_kernel)
    else:
        target_signal = splat_gaussians(height, width, tgt_exp_pos, tgt_exp_int, psf_sigma)

    # Combine signal
    signal = star_signal + target_signal

    # Add background, dark current, bias (per pixel, uniform)
    signal = signal + background_pe + dark_current_pe

    # Shot noise (Poisson)
    if enable_shot_noise:
        signal = poisson_noise(signal)

    signal = signal + bias_pe

    # When defer_read_noise is True, return analog signal before read noise
    # and A/D so the caller can apply them after binning (CCD mode).
    if defer_read_noise:
        return signal, star_signal, target_signal

    # Read + electronic noise (Gaussian)
    if enable_read_noise:
        rn_sigma = math.sqrt(read_noise**2 + electronic_noise**2)
        if rn_sigma > 0:
            signal = gaussian_noise(signal, rn_sigma)

    # A/D conversion
    digital = analog_to_digital(signal, gain, fwc, a2d_bias, a2d_dtype)

    return digital, star_signal, target_signal
