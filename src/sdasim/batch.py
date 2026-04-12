"""Batched multi-scene rendering.

Fuses N heterogeneous Scenes into a single kernel launch. Each scene can have
its own star catalog, its own PSF sigma, its own noise/gain/background/etc.

Restrictions in this first pass (matches what zerosda's pretraining loop
needs):
  - All scenes must share (height, width) and a2d_dtype.
  - Only supports frame_idx=0 and mode=None. Rate_sidereal mode-dispatch
    is left to the non-batched Scene.render() path.
  - Star motion (rate tracking) is supported per-scene via expand_motion.
  - Targets are rendered with their own velocities, also via expand_motion.

Key win: all stars and all targets across all B scenes are splatted with two
kernel launches total (one for stars, one for targets), instead of 2*B.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from sdasim.fpa import MAX_PIXEL_VALUE
from sdasim.noise import gaussian_noise, poisson_noise
from sdasim.render import expand_motion
from sdasim.scene import Scene
from sdasim.splat import splat_gaussians_batched
from sdasim.targets import compute_target_positions


_CACHE_ATTR = "_batch_source_cache"


def _ensure_source_cache(scene: Scene) -> dict:
    """Lazily cache per-scene source tensors for batched rendering.

    This precomputes the motion-expanded star and target source lists plus the
    per-scene scalar params used downstream. All of this is deterministic given
    the scene's config and doesn't change across reuses, so computing it once
    at pool-fill time moves a lot of per-batch Python work out of the hot path.
    """
    cache = getattr(scene, _CACHE_ATTR, None)
    if cache is not None:
        return cache

    sensor = scene.sensor
    sm = scene.config.star_motion
    device = scene.device
    center = (sensor.height / 2.0, sensor.width / 2.0)

    # --- Stars with within-frame motion blur ---
    sp = scene.star_positions
    si = scene.star_intensities
    has_motion = (
        sm.rotation != 0.0
        or sm.translation[0] != 0.0
        or sm.translation[1] != 0.0
    )
    if has_motion and sm.temporal_osf > 1 and sp.shape[0] > 0:
        star_pos, star_int = expand_motion(
            sp, si, sm.translation, sm.rotation,
            0.0, sensor.exposure, sm.temporal_osf, center,
        )
    else:
        star_pos, star_int = sp, si

    # --- Targets with per-target motion blur ---
    tp, ti, tv = compute_target_positions(scene.config.targets, sensor, 0, device)
    if tp.shape[0] > 0:
        max_speed = float(tv.abs().max().item()) if tv.numel() else 0.0
        streak_px = max_speed * sensor.exposure
        tgt_osf = max(1, int(streak_px * 2))
        if tgt_osf > 1:
            tgt_pos, tgt_int = expand_motion(
                tp, ti, tv, 0.0, 0.0, sensor.exposure, tgt_osf, center,
            )
        else:
            tgt_pos, tgt_int = tp, ti
    else:
        tgt_pos = torch.zeros(0, 2, device=device, dtype=torch.float32)
        tgt_int = torch.zeros(0, device=device, dtype=torch.float32)

    cache = {
        "star_pos": star_pos,
        "star_int": star_int,
        "star_count": int(star_pos.shape[0]),
        "tgt_pos": tgt_pos,
        "tgt_int": tgt_int,
        "tgt_count": int(tgt_pos.shape[0]),
        "psf_sigma": float(sensor.psf_sigma),
        "background_pe": float(scene.background_pe),
        "dark_current_pe": float(scene.dark_current_pe),
        "bias_pe": float(scene.bias_pe),
        "read_noise": float(sensor.read_noise),
        "electronic_noise": float(sensor.electronic_noise),
        "gain": float(sensor.gain),
        "fwc": float(sensor.fwc),
        "a2d_bias": float(sensor.a2d_bias),
    }
    setattr(scene, _CACHE_ATTR, cache)
    return cache


@dataclass
class BatchRenderResult:
    """Output of render_scene_batch.

    Attributes:
        digital: [B, H, W] integer-ADU digital image.
        star_signal: [B, H, W] pre-noise star-only PE.
        target_signal: [B, H, W] pre-noise target-only PE.
        star_positions_per_frame: list of [N_b, 2] per-frame pre-expansion star
            positions (same as each scene's Scene.star_positions). Useful for
            generating training heatmaps without running the splat again.
        num_stars: [B] int counts of sources in each frame (pre-motion-expand).
    """

    digital: Tensor
    star_signal: Tensor
    target_signal: Tensor
    star_positions_per_frame: list[Tensor]
    num_stars: Tensor


def _collect_cached(
    scenes: Sequence[Scene],
    device: torch.device,
    kind: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Gather pre-computed (via _ensure_source_cache) star or target sources.

    This collects already-cached per-scene tensors and builds (positions,
    intensities, frame_ids, per_source_sigma) via a single torch.cat +
    repeat_interleave pair. Extremely cheap because the motion expansion
    work was done once at cache-fill time.
    """
    pos_key = f"{kind}_pos"
    int_key = f"{kind}_int"
    cnt_key = f"{kind}_count"

    pos_chunks: list[Tensor] = []
    int_chunks: list[Tensor] = []
    counts: list[int] = []
    sigmas: list[float] = []
    fids: list[int] = []

    for i, scene in enumerate(scenes):
        c = _ensure_source_cache(scene)
        n = c[cnt_key]
        if n == 0:
            continue
        pos_chunks.append(c[pos_key])
        int_chunks.append(c[int_key])
        counts.append(n)
        sigmas.append(c["psf_sigma"])
        fids.append(i)

    if not pos_chunks:
        z2 = torch.zeros(0, 2, dtype=torch.float32, device=device)
        z1 = torch.zeros(0, dtype=torch.float32, device=device)
        zi = torch.zeros(0, dtype=torch.long, device=device)
        return z2, z1, zi, z1

    positions = torch.cat(pos_chunks, dim=0)
    intensities = torch.cat(int_chunks, dim=0)
    counts_t = torch.tensor(counts, dtype=torch.long, device=device)
    frame_ids = torch.repeat_interleave(
        torch.tensor(fids, dtype=torch.long, device=device), counts_t
    )
    per_source_sigma = torch.repeat_interleave(
        torch.tensor(sigmas, dtype=torch.float32, device=device), counts_t
    )
    return positions, intensities, frame_ids, per_source_sigma


def render_scene_batch(scenes: Sequence[Scene]) -> BatchRenderResult:
    """Render N scenes in a single fused pass.

    Each scene is treated as its own telescope/exposure with its own PSF,
    background, read noise, gain, etc. All scenes contribute to one
    (B, H, W) output via per-source frame_id tagging on the splats.

    Args:
        scenes: Sequence of Scene objects. All must share (height, width,
            a2d_dtype, device). Scene.config.mode must be None for each scene.

    Returns:
        BatchRenderResult with digital, star_signal, target_signal, and the
        per-frame pre-expansion star positions (handy for building heatmaps).
    """
    if len(scenes) == 0:
        raise ValueError("render_scene_batch requires at least one scene")

    ref = scenes[0]
    device = ref.device
    H = ref.sensor.height
    W = ref.sensor.width
    a2d_dtype = ref.sensor.a2d_dtype
    B = len(scenes)

    for s in scenes:
        if s.sensor.height != H or s.sensor.width != W:
            raise ValueError(
                "render_scene_batch requires all scenes to share (height, width); "
                f"got {(s.sensor.height, s.sensor.width)} vs {(H, W)}"
            )
        if s.sensor.a2d_dtype != a2d_dtype:
            raise ValueError(
                "render_scene_batch requires all scenes to share a2d_dtype"
            )
        if s.config.mode is not None:
            raise ValueError(
                "render_scene_batch only supports mode=None (sidereal/rate_track); "
                f"got mode={s.config.mode}"
            )

    # Ensure caches exist (lazy, idempotent on pool scenes).
    for s in scenes:
        _ensure_source_cache(s)

    per_frame_positions = [s.star_positions for s in scenes]
    star_pos, star_int, star_fid, star_sig = _collect_cached(scenes, device, "star")
    tgt_pos, tgt_int, tgt_fid, tgt_sig = _collect_cached(scenes, device, "tgt")

    # Single fused splat per kind. splat_gaussians_batched handles the empty
    # source case by returning zeros of the right shape.
    star_signal = splat_gaussians_batched(
        B, H, W, star_pos, star_int, star_fid, star_sig,
    )
    target_signal = splat_gaussians_batched(
        B, H, W, tgt_pos, tgt_int, tgt_fid, tgt_sig,
    )

    signal = star_signal + target_signal

    # Per-frame scalar params from cache. Build a single tensor per param.
    caches = [getattr(s, _CACHE_ATTR) for s in scenes]
    bg = torch.tensor(
        [c["background_pe"] for c in caches], dtype=torch.float32, device=device,
    ).view(B, 1, 1)
    dc = torch.tensor(
        [c["dark_current_pe"] for c in caches], dtype=torch.float32, device=device,
    ).view(B, 1, 1)
    bias = torch.tensor(
        [c["bias_pe"] for c in caches], dtype=torch.float32, device=device,
    ).view(B, 1, 1)

    signal = signal + bg + dc + bias

    enable_shot = all(s.config.enable_shot_noise for s in scenes)
    enable_read = all(s.config.enable_read_noise for s in scenes)
    if enable_shot:
        signal = poisson_noise(signal)
    if enable_read:
        rn_sigma = torch.tensor(
            [
                math.sqrt(c["read_noise"] ** 2 + c["electronic_noise"] ** 2)
                for c in caches
            ],
            dtype=torch.float32,
            device=device,
        ).view(B, 1, 1)
        signal = signal + rn_sigma * torch.randn_like(signal)

    a2d_bias_t = torch.tensor(
        [c["a2d_bias"] for c in caches], dtype=torch.float32, device=device,
    ).view(B, 1, 1)
    fwc_t = torch.tensor(
        [c["fwc"] for c in caches], dtype=torch.float32, device=device,
    ).view(B, 1, 1)
    gain_t = torch.tensor(
        [c["gain"] for c in caches], dtype=torch.float32, device=device,
    ).view(B, 1, 1)

    biased = (signal + a2d_bias_t).clamp(min=0.0)
    biased = torch.minimum(biased, fwc_t)
    dn = torch.floor(biased / gain_t)
    max_val = MAX_PIXEL_VALUE.get(a2d_dtype, 65535.0)
    dn = dn.clamp(min=0.0, max=max_val)

    num_stars = torch.tensor(
        [p.shape[0] for p in per_frame_positions],
        dtype=torch.long,
        device=device,
    )

    return BatchRenderResult(
        digital=dn,
        star_signal=star_signal,
        target_signal=target_signal,
        star_positions_per_frame=per_frame_positions,
        num_stars=num_stars,
    )
