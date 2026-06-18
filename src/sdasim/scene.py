"""Scene class: pre-compiled, reusable scene for fast rendering.

Separates slow setup (catalog loading, tensor allocation) from fast rendering
(pure tensor ops). Scene.render() is the hot path.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from sdasim.config import SceneConfig, SensorConfig, load_config
from sdasim.device import resolve_device
from sdasim.fpa import mv_to_pe
from sdasim.render import render_frame
from sdasim.stars import generate_random_stars
from sdasim.targets import compute_target_positions


def _apply_star_offset(
    positions: Tensor,
    translation: list[float],
    rotation: float,
    elapsed: float,
    center: tuple[float, float],
) -> Tensor:
    """Offset star positions by accumulated motion since t=0.

    Args:
        positions: (N, 2) base star positions.
        translation: [row_rate, col_rate] in px/sec.
        rotation: Rotation rate in rad/sec.
        elapsed: Time elapsed since start of sequence (seconds).
        center: (row, col) center of rotation.

    Returns:
        (N, 2) offset positions.
    """
    if elapsed == 0.0:
        return positions

    pos = positions.clone()

    # Translation
    if translation[0] != 0.0 or translation[1] != 0.0:
        pos[:, 0] = pos[:, 0] + translation[0] * elapsed
        pos[:, 1] = pos[:, 1] + translation[1] * elapsed

    # Rotation about center
    if rotation != 0.0:
        angle = rotation * elapsed
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        cr, cc = center
        r_rel = pos[:, 0] - cr
        c_rel = pos[:, 1] - cc
        pos[:, 0] = cos_a * r_rel - sin_a * c_rel + cr
        pos[:, 1] = sin_a * r_rel + cos_a * c_rel + cc

    return pos


class Scene:
    """Pre-compiled scene for fast rendering.

    Construction is slow (loads catalogs, allocates tensors).
    render() is fast (pure tensor ops, differentiable).
    """

    def __init__(self, config: SceneConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.sensor = config.sensor

        # Validate mode
        if config.mode == "rate_sidereal" and config.sidereal_start is None:
            raise ValueError(
                "mode='rate_sidereal' requires sidereal_start to be set"
            )

        # Seed
        if config.seed is not None:
            torch.manual_seed(config.seed)

        # Load stars (slow, one-time)
        self._load_stars(config)

        # Pre-compute background PE per pixel
        bg_pe_per_sec = mv_to_pe(self.sensor.zeropoint, self.sensor.background_mv)
        self.background_pe = float(bg_pe_per_sec * self.sensor.exposure)

        # Dark current: rate * exposure
        self.dark_current_pe = self.sensor.dark_current * self.sensor.exposure

        # Bias
        self.bias_pe = self.sensor.bias

    def _load_stars(self, config: SceneConfig) -> None:
        """Load star catalog based on config."""
        sc = config.stars
        sensor = config.sensor

        if sc.mode == "bins":
            self.star_positions, self.star_intensities = generate_random_stars(
                height=sensor.height,
                width=sensor.width,
                y_fov=sensor.y_fov,
                x_fov=sensor.x_fov,
                mv_bins=sc.mv_bins,
                density=sc.density,
                zeropoint=sensor.zeropoint,
                exposure=sensor.exposure,
                pad_mult=sc.pad_mult,
                seed=config.seed,
                device=self.device,
            )
        elif sc.mode == "sstr7":
            from sdasim.stars import load_sstr7

            self.star_positions, self.star_intensities = load_sstr7(
                height=sensor.height,
                width=sensor.width,
                y_fov=sensor.y_fov,
                x_fov=sensor.x_fov,
                ra=sc.ra,
                dec=sc.dec,
                rot=sc.rot,
                zeropoint=sensor.zeropoint,
                exposure=sensor.exposure,
                catalog_path=sc.catalog_path,
                pad_mult=sc.pad_mult,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown star mode: {sc.mode}")

    def render(
        self,
        frame_idx: int = 0,
        **overrides: Any,
    ) -> tuple[Tensor, dict]:
        """Render a single frame.

        Args:
            frame_idx: Frame index (affects target positions and star motion).
            **overrides: Override any render parameter (psf_sigma, target positions, etc.)

        Returns:
            (digital_image, metadata_dict)
        """
        sensor = self.sensor
        sm = self.config.star_motion

        # Frame timing
        frame_start = frame_idx * (sensor.exposure + sensor.gap)

        # Determine frame mode for rate_sidereal
        is_sidereal = (
            self.config.mode == "rate_sidereal"
            and self.config.sidereal_start is not None
            and frame_idx >= self.config.sidereal_start
        )

        # Compute target positions/velocities for this frame
        tgt_pos, tgt_int, tgt_vel = compute_target_positions(
            self.config.targets, sensor, frame_idx, self.device,
        )

        # Allow overrides
        psf_sigma = overrides.get("psf_sigma", sensor.psf_sigma)
        star_positions = overrides.get("star_positions", self.star_positions)
        star_intensities = overrides.get("star_intensities", self.star_intensities)
        target_positions = overrides.get("target_positions", tgt_pos)
        target_intensities = overrides.get("target_intensities", tgt_int)
        target_velocities = overrides.get("target_velocities", tgt_vel)

        center = (sensor.height / 2.0, sensor.width / 2.0)

        # --- Mode dispatch ---
        if self.config.mode == "rate_sidereal":
            # Target velocities in config are inertial (star-fixed frame).
            # In rate-track frames, convert to sensor frame:
            #   apparent_vel = V_inertial + translation
            translation_t = torch.tensor(
                sm.translation, dtype=torch.float32, device=self.device,
            )

            if is_sidereal:
                # Sidereal frame: stars are sharp, targets streak at inertial velocity
                star_vel = [0.0, 0.0]
                star_rot = 0.0
                star_osf = 1
                # No inter-frame star offset (stars are fixed on sky)
            else:
                # Rate-track frame: stars streak, targets appear at sensor-frame velocity
                target_velocities = target_velocities + translation_t
                # Correct target positions for accumulated sensor slew
                target_positions = target_positions + translation_t * frame_start
                star_vel = sm.translation
                star_rot = sm.rotation
                star_osf = sm.temporal_osf
                # Apply accumulated inter-frame star drift
                star_positions = _apply_star_offset(
                    star_positions, sm.translation, sm.rotation, frame_start, center,
                )
        else:
            # Legacy mode (None): star_motion applies uniformly,
            # target velocities are already in sensor frame
            star_vel = sm.translation
            star_rot = sm.rotation
            star_osf = sm.temporal_osf
            star_positions = _apply_star_offset(
                star_positions, sm.translation, sm.rotation, frame_start, center,
            )

        # Target velocities: pass None if no targets
        if target_velocities.shape[0] == 0:
            target_velocities = None

        digital, star_signal, target_signal = render_frame(
            height=sensor.height,
            width=sensor.width,
            star_positions=star_positions,
            star_intensities=star_intensities,
            target_positions=target_positions,
            target_intensities=target_intensities,
            psf_sigma=psf_sigma,
            background_pe=self.background_pe,
            dark_current_pe=self.dark_current_pe,
            bias_pe=self.bias_pe,
            read_noise=sensor.read_noise,
            electronic_noise=sensor.electronic_noise,
            gain=sensor.gain,
            fwc=sensor.fwc,
            a2d_bias=sensor.a2d_bias,
            a2d_dtype=sensor.a2d_dtype,
            enable_shot_noise=self.config.enable_shot_noise,
            enable_read_noise=self.config.enable_read_noise,
            star_velocity=star_vel,
            star_rotation=star_rot,
            target_velocities=target_velocities,
            t_osf=star_osf,
            exposure=sensor.exposure,
        )

        # Frame mode label
        if self.config.mode == "rate_sidereal":
            frame_mode = "sidereal" if is_sidereal else "rate_track"
        else:
            frame_mode = None

        metadata = {
            "frame_idx": frame_idx,
            "frame_time": frame_start,
            "frame_mode": frame_mode,
            "num_stars": star_positions.shape[0],
            "num_targets": target_positions.shape[0] if target_positions is not None else 0,
            "target_positions": target_positions.detach().cpu().tolist(),
            "target_intensities": target_intensities.detach().cpu().tolist(),
            "star_velocity": list(star_vel),  # [row_rate, col_rate] px/sec
            "star_rotation": star_rot,  # rad/sec
            "target_velocities": (
                target_velocities.detach().cpu().tolist()
                if target_velocities is not None else []
            ),  # [[row_rate, col_rate], ...] px/sec
            "exposure": sensor.exposure,  # seconds
            "obs_time": self.config.obs_time,
            "gain": sensor.gain,
            "read_noise": sensor.read_noise,
            "dark_current": sensor.dark_current,
            "y_fov": sensor.y_fov,
            "x_fov": sensor.x_fov,
            "gap": sensor.gap,
            "psf_sigma": float(psf_sigma) if not isinstance(psf_sigma, float) else psf_sigma,
            "zeropoint": sensor.zeropoint,
            "a2d_dtype": sensor.a2d_dtype,
            "_height": sensor.height,
            "_width": sensor.width,
        }

        return digital, metadata

    def render_batch(
        self,
        frame_indices: list[int],
        **overrides: Any,
    ) -> tuple[Tensor, list[dict]]:
        """Render multiple frames.

        Args:
            frame_indices: List of frame indices.
            **overrides: Passed to each render() call.

        Returns:
            (images, metadata_list): (N, H, W) tensor, list of metadata dicts.
        """
        images = []
        metadata = []
        for idx in frame_indices:
            img, meta = self.render(idx, **overrides)
            images.append(img)
            metadata.append(meta)

        return torch.stack(images), metadata

    def render_sequence(self, **overrides: Any) -> tuple[Tensor, list[dict]]:
        """Render all frames in the sequence.

        Returns:
            (images, metadata_list): (num_frames, H, W) tensor, list of metadata dicts.
        """
        indices = list(range(self.sensor.num_frames))
        return self.render_batch(indices, **overrides)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Scene:
        """Create a Scene from a YAML config file."""
        config = load_config(path)
        return cls(config)

    @classmethod
    def from_satsim(cls, satsim_config: str | dict, seed: int | None = None) -> Scene:
        """Create a Scene from a satsim config (requires _compat module).

        Args:
            satsim_config: Path to satsim JSON/YAML or a dict.
            seed: Random seed for config resolution.

        Returns:
            Scene instance.
        """
        from sdasim._compat import from_satsim_config

        if isinstance(satsim_config, (str, Path)):
            import json
            import yaml

            path = Path(satsim_config)
            with open(path) as f:
                if path.suffix in (".yaml", ".yml"):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
        else:
            data = satsim_config

        config = from_satsim_config(data, seed=seed)
        return cls(config)
