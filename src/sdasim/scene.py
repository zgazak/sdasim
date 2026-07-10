"""Scene class: pre-compiled, reusable scene for fast rendering.

Separates slow setup (catalog loading, tensor allocation) from fast rendering
(pure tensor ops). Scene.render() is the hot path.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from sdasim.config import SceneConfig, TargetConfig, load_config
from sdasim.device import resolve_device
from sdasim.empirical import (
    BasicWhiteNoise,
    EmpiricalNoise,
    EmpiricalPSF,
    gaussian_kernel,
    render_frame_empirical,
)
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
        self._object_states = None
        # Star-field bookkeeping. `_star_mode` records the loaded catalog mode so
        # render() knows whether the field is sky-backed (sstrc7) and can re-project
        # it to a per-frame commanded pointing. `_reproj_cache` memoizes the most
        # recent (ra, dec) -> (positions, intensities) so repeated renders at one
        # pointing don't re-run the WCS projection.
        self._star_mode: str | None = None
        self._reproj_cache: tuple[tuple[float, float], tuple[Tensor, Tensor]] | None = None

        # Validate mode
        if config.mode == "rate_sidereal" and config.sidereal_start is None:
            raise ValueError("mode='rate_sidereal' requires sidereal_start to be set")

        # Seed
        if config.seed is not None:
            torch.manual_seed(config.seed)

        # Orbital mechanics mode: objects present → derive everything from TLEs
        if config.objects:
            self._init_from_objects(config)
        else:
            # Load stars (slow, one-time)
            self._load_stars(config)

        # Catalog discovery (opt-in): auto-stream field satellites through the FOV.
        # The TLE catalog is parsed once here (slow); per-frame propagation is fast.
        self._catalog = None
        if config.catalog.enabled:
            if config.site is None:
                raise ValueError("catalog.enabled=True requires site to be set")
            from sdasim.orbits import load_catalog

            self._catalog = load_catalog(config.catalog)

        # Pre-compute background PE per pixel. background_mv is a SURFACE
        # BRIGHTNESS (mag/arcsec^2), so scale the per-arcsec^2 flux by the pixel
        # solid angle to get per-pixel electrons. (A point-source star needs no
        # such factor — all its flux lands in its PSF regardless of pixel scale.)
        pixel_area_arcsec2 = (self.sensor.y_fov * 3600.0 / self.sensor.height) * (
            self.sensor.x_fov * 3600.0 / self.sensor.width
        )
        bg_pe_per_sec = (
            mv_to_pe(self.sensor.zeropoint, self.sensor.background_mv) * pixel_area_arcsec2
        )
        self.background_pe = float(bg_pe_per_sec * self.sensor.exposure)

        # Dark current: rate * exposure
        self.dark_current_pe = self.sensor.dark_current * self.sensor.exposure

        # Bias
        self.bias_pe = self.sensor.bias

        # Empirical (opt-in) bases — loaded once; default path leaves these None
        self._np_rng = np.random.default_rng(config.seed)
        # per-scene torch generator so empirical noise/texture/wobble/speckle are reproducible
        # per scene regardless of global RNG order (important for batched rendering)
        if config.seed is not None:
            self._gen = torch.Generator(device=self.device)
            self._gen.manual_seed(int(config.seed))
        else:
            self._gen = None
        self.emp_psf = None
        self.emp_noise = None
        if self.sensor.psf_model == "empirical":
            if not self.sensor.empirical_psf_path:
                raise ValueError("psf_model='empirical' requires sensor.empirical_psf_path")
            self.emp_psf = EmpiricalPSF(self.sensor.empirical_psf_path)
        if self.sensor.noise_model == "empirical":
            if not self.sensor.empirical_noise_path:
                raise ValueError("noise_model='empirical' requires sensor.empirical_noise_path")
            self.emp_noise = EmpiricalNoise(self.sensor.empirical_noise_path)
        # Streak texture: 4 calibrated atmospheric components, opt-in via amplitude in
        # streak_model.json: scintillation (intensity), tip/tilt (position), seeing-breathing
        # (PSF size), and a post-PSF high-frequency plateau (the measured 3-9px texture band).
        self.streak_tex = dict(
            scint_rms=0.0,
            tiptilt_rms=0.0,
            seeing_rms=0.0,
            hf_rms=0.0,
            beta=2.0,
            decon_px=1.3,
            hf_klo=0.08,
            hf_khi=0.30,
            hf_slope=1.2,
            floor_frac=0.45,
        )
        self.streak_dist = None  # measured (mean,std) of amplitudes, for per-render sampling
        # seeing-breathing kernel (symmetric PSF size derivative), built once
        self.size_mode = self.emp_psf.size_mode(device=self.device) if self.emp_psf else None
        if self.sensor.empirical_streak_path:
            import json as _json

            sj = _json.load(open(self.sensor.empirical_streak_path))
            for k in self.streak_tex:
                if ("tex_" + k) in sj:
                    self.streak_tex[k] = float(sj["tex_" + k])
            self.streak_dist = {
                "scint": sj.get("dist_tex_scint_rms"),
                "tiptilt": sj.get("dist_tex_tiptilt_rms"),
                "seeing": sj.get("dist_tex_seeing_rms"),
            }

    def _load_stars(self, config: SceneConfig) -> None:
        """Load star catalog based on config."""
        sc = config.stars
        sensor = config.sensor
        self._star_mode = sc.mode

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
        elif sc.mode == "sstrc7":
            self.star_positions, self.star_intensities = self._load_sstrc7_stars(sc.ra, sc.dec)
        else:
            raise ValueError(f"Unknown star mode: {sc.mode}")

    def _load_sstrc7_stars(self, ra: float, dec: float) -> tuple[Tensor, Tensor]:
        """Project the SSTRC7 catalog onto the focal plane for a pointing center.

        Split out from `_load_stars` so `render()` can re-project the (sky-backed)
        star field to a per-frame commanded pointing without a full Scene rebuild.
        Everything but the pointing (FOV, rotation, exposure, catalog path, mag cut)
        is fixed at construction and read from config here.
        """
        from sdasim.stars import load_sstrc7

        sc = self.config.stars
        sensor = self.sensor
        return load_sstrc7(
            height=sensor.height,
            width=sensor.width,
            y_fov=sensor.y_fov,
            x_fov=sensor.x_fov,
            ra=ra,
            dec=dec,
            rot=sc.rot,
            zeropoint=sensor.zeropoint,
            exposure=sensor.exposure,
            catalog_path=sc.catalog_path,
            pad_mult=sc.pad_mult,
            mv_max=sc.mv_max,
            device=self.device,
        )

    def _stars_for_pointing(self, ra: float, dec: float) -> tuple[Tensor, Tensor]:
        """Return the SSTRC7 star field for a pointing center, memoizing the last one.

        The parsed catalog zones are cached in `sstrc7`, so a re-projection is just a
        WCS transform over the in-FOV stars (~sub-millisecond), not a disk read; the
        one-entry cache here additionally skips recompute when consecutive renders
        share a pointing.
        """
        key = (round(float(ra), 9), round(float(dec), 9))
        if self._reproj_cache is not None and self._reproj_cache[0] == key:
            return self._reproj_cache[1]
        stars = self._load_sstrc7_stars(ra, dec)
        self._reproj_cache = (key, stars)
        return stars

    def _init_from_objects(self, config: SceneConfig) -> None:
        """Initialize scene from orbital objects (TLE-based physics mode).

        Fetches TLEs, propagates orbits, derives pointing/star field,
        and populates star motion + target configs from orbital mechanics.
        """
        from datetime import datetime, timezone

        from sdasim.orbits import compute_object_states

        # Parse observation time
        if config.obs_time:
            obs_time = datetime.fromisoformat(config.obs_time.replace("Z", "+00:00"))
        else:
            obs_time = datetime.now(timezone.utc)

        # Compute orbital states
        states = compute_object_states(
            config.objects, config.site, config.sensor, obs_time,
        )
        self._object_states = states

        primary = states[0]
        tracking = config.tracking or "rate_track"

        # Set star field pointing from primary target.
        # In rate_track mode, shift the star field center backward by half an
        # exposure so that streak centroids (at mid-exposure) align with the
        # satellite at image center (which is at the start-of-exposure position).
        if tracking == "rate_track":
            import math
            half_exp = config.sensor.exposure / 2.0
            config.stars.ra = primary.ra - math.degrees(primary.ra_rate) * half_exp
            config.stars.dec = primary.dec - math.degrees(primary.dec_rate) * half_exp
        else:
            config.stars.ra = primary.ra
            config.stars.dec = primary.dec

        # Load star field for derived pointing
        self._load_stars(config)

        # Configure star motion based on tracking mode
        if tracking == "rate_track":
            # Stars appear to move opposite to primary target's motion
            config.star_motion.translation = [-primary.row_rate, -primary.col_rate]
        elif tracking == "sidereal":
            # Stars are stationary, targets streak
            config.star_motion.translation = [0.0, 0.0]
            config.star_motion.temporal_osf = 1
        else:
            raise ValueError(f"Unknown tracking mode: {tracking!r}")

        # Convert object states to target configs
        targets = []
        for i, st in enumerate(states):
            if tracking == "rate_track":
                if i == 0:
                    # Primary is tracked → zero velocity
                    vel = [0.0, 0.0]
                else:
                    # Secondary velocity relative to primary
                    vel = [
                        st.row_rate - primary.row_rate,
                        st.col_rate - primary.col_rate,
                    ]
            else:
                # Sidereal: full pixel rate
                vel = [st.row_rate, st.col_rate]

            origin = [
                st.pixel_row / config.sensor.height,
                st.pixel_col / config.sensor.width,
            ]
            targets.append(TargetConfig(
                mode="line",
                origin=origin,
                velocity=vel,
                mv=st.mv,
            ))

        config.targets = targets

    def _resolve_mount_rate(self, **overrides: Any) -> tuple[float, float]:
        """Resolve inertial mount RA/Dec rate (rad/s) for the current frame.

        Precedence: explicit override / config (deg/s) > primary tracked object
        (already rad/s) > sidereal (0).
        """
        cfg = self.config
        has_explicit = (
            "mount_ra_rate" in overrides
            or "mount_dec_rate" in overrides
            or cfg.mount_ra_rate
            or cfg.mount_dec_rate
        )
        if has_explicit:
            mr = overrides.get("mount_ra_rate", cfg.mount_ra_rate)
            md = overrides.get("mount_dec_rate", cfg.mount_dec_rate)
            return math.radians(mr), math.radians(md)
        if self._object_states:
            primary = self._object_states[0]
            return primary.ra_rate, primary.dec_rate
        return 0.0, 0.0

    def _discover_for_frame(
        self, frame_start: float, **overrides: Any
    ) -> tuple[list, list[float]]:
        """Run catalog discovery for this frame.

        Returns ``(streaks, star_translation_px)`` where ``star_translation_px``
        is the per-frame star drift [row_rate, col_rate] induced by the mount
        (stars stream opposite the mount; sidereal → [0, 0]).
        """
        from datetime import datetime, timedelta, timezone

        from sdasim.orbits import angular_to_pixel_rates, discover_streaks

        cfg = self.config
        sensor = self.sensor

        # Pointing defaults to the loaded star-field center; overridable per frame
        # (note: overriding it moves the satellites, not the pre-loaded star field).
        point_ra = overrides.get("point_ra", cfg.stars.ra)
        point_dec = overrides.get("point_dec", cfg.stars.dec)

        base_iso = overrides.get("obs_time", cfg.obs_time)
        if base_iso:
            base = datetime.fromisoformat(base_iso.replace("Z", "+00:00"))
        else:
            base = datetime.now(timezone.utc)
        t_frame = base + timedelta(seconds=frame_start)

        mr_rad, md_rad = self._resolve_mount_rate(**overrides)

        cat = cfg.catalog
        pad = cat.psf_pad_px if cat.psf_pad_px is not None else 3.0 * sensor.psf_sigma

        streaks = discover_streaks(
            self._catalog, t_frame, cfg.site, sensor,
            point_ra, point_dec, mr_rad, md_rad,
            cat.albedo, pad,
        )

        # Stars stream opposite the mount; sidereal (mount=0) → static.
        row_rate, col_rate = angular_to_pixel_rates(
            mr_rad, md_rad, point_dec,
            sensor.y_fov, sensor.x_fov, sensor.height, sensor.width,
        )
        return streaks, [-row_rate, -col_rate]

    def render(self, frame_idx: int = 0, **overrides: Any) -> tuple[Tensor, dict]:
        """Render a single frame -> (digital_image, metadata_dict)."""
        digital, _, _, meta = self.render_signals(frame_idx, **overrides)
        return digital, meta

    def render_signals(
        self,
        frame_idx: int = 0,
        **overrides: Any,
    ) -> tuple[Tensor, Tensor, Tensor, dict]:
        """Render one frame -> (digital, star_signal, target_signal, metadata).

        Args:
            frame_idx: Frame index (affects target positions and star motion).
            **overrides: Override any render parameter (psf_sigma, target positions, etc.)
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
            self.config.targets,
            sensor,
            frame_idx,
            self.device,
        )

        # Allow overrides
        psf_sigma = overrides.get("psf_sigma", sensor.psf_sigma)
        star_positions = overrides.get("star_positions", self.star_positions)
        star_intensities = overrides.get("star_intensities", self.star_intensities)
        target_positions = overrides.get("target_positions", tgt_pos)
        target_intensities = overrides.get("target_intensities", tgt_int)
        target_velocities = overrides.get("target_velocities", tgt_vel)

        # Sky-backed star fields (sstrc7) track the commanded pointing per frame.
        # When point_ra/point_dec are supplied (and the caller hasn't overridden the
        # star field directly), re-project the catalog to that center so a sequence
        # with moving pointing renders the correct stars for each frame instead of
        # the field baked in at construction. Cheap: parsed zones are cached, so this
        # is a WCS transform, not a disk read. Other modes (random `bins`) aren't
        # sky-backed, so pointing overrides leave their field unchanged.
        if (
            self._star_mode == "sstrc7"
            and "star_positions" not in overrides
            and "star_intensities" not in overrides
            and ("point_ra" in overrides or "point_dec" in overrides)
        ):
            star_positions, star_intensities = self._stars_for_pointing(
                overrides.get("point_ra", self.config.stars.ra),
                overrides.get("point_dec", self.config.stars.dec),
            )

        center = (sensor.height / 2.0, sensor.width / 2.0)

        # --- Catalog discovery: append auto-found field satellites as line targets ---
        cat_star_translation = None
        if self._catalog is not None:
            streaks, cat_star_translation = self._discover_for_frame(frame_start, **overrides)
            if streaks:
                extra_pos = torch.tensor(
                    [[s.pixel_row, s.pixel_col] for s in streaks],
                    dtype=torch.float32, device=self.device,
                )
                extra_int = torch.tensor(
                    [mv_to_pe(sensor.zeropoint, s.mv) * sensor.exposure for s in streaks],
                    dtype=torch.float32, device=self.device,
                )
                extra_vel = torch.tensor(
                    [[s.row_rate, s.col_rate] for s in streaks],
                    dtype=torch.float32, device=self.device,
                )
                target_positions = torch.cat([target_positions, extra_pos], dim=0)
                target_intensities = torch.cat([target_intensities, extra_int], dim=0)
                target_velocities = torch.cat([target_velocities, extra_vel], dim=0)

        # --- Mode dispatch ---
        if self._catalog is not None:
            # Catalog mode: mount rate drives star streaking; targets (configured +
            # discovered) already carry sensor-frame apparent velocities, used as-is.
            star_vel = cat_star_translation
            star_rot = 0.0
            moving = star_vel[0] != 0.0 or star_vel[1] != 0.0
            star_osf = sm.temporal_osf if moving else 1
            star_positions = _apply_star_offset(
                star_positions, star_vel, 0.0, frame_start, center,
            )
        elif self.config.mode == "rate_sidereal":
            # Target velocities in config are inertial (star-fixed frame).
            # In rate-track frames, convert to sensor frame:
            #   apparent_vel = V_inertial + translation
            translation_t = torch.tensor(
                sm.translation,
                dtype=torch.float32,
                device=self.device,
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
                    star_positions,
                    sm.translation,
                    sm.rotation,
                    frame_start,
                    center,
                )
        else:
            # Legacy mode (None): star_motion applies uniformly,
            # target velocities are already in sensor frame
            star_vel = sm.translation
            star_rot = sm.rotation
            star_osf = sm.temporal_osf
            star_positions = _apply_star_offset(
                star_positions,
                sm.translation,
                sm.rotation,
                frame_start,
                center,
            )

        # Target velocities: pass None if no targets
        if target_velocities.shape[0] == 0:
            target_velocities = None

        defer_read_noise = overrides.get("defer_read_noise", False)

        use_empirical = sensor.psf_model == "empirical" or sensor.noise_model == "empirical"
        psf_params = None
        tex_used = None
        if use_empirical:
            # Per-frame PSF kernel (empirical mean PSF warped by sampled size/ellipticity,
            # or a Gaussian kernel when only the empirical noise is requested).
            if sensor.psf_model == "empirical":
                if sensor.psf_param_scale <= 0:
                    psf_params = self.emp_psf.param_mean
                else:
                    psf_params = self.emp_psf.sample_params(self._np_rng, sensor.psf_param_scale)
                kernel = self.emp_psf.kernel(params=psf_params, device=self.device)
            else:
                K = self.emp_psf.K if self.emp_psf is not None else 31
                kernel = gaussian_kernel(float(psf_sigma), K, device=self.device)
            noise = (
                self.emp_noise
                if sensor.noise_model == "empirical"
                else BasicWhiteNoise(sensor.read_noise)
            )
            # streak texture amplitudes: per-streak diversity via streak_param_sample
            tex = dict(self.streak_tex)
            if sensor.streak_param_sample:
                ext = sensor.streak_param_extend
                f = max(0.2, float(self._np_rng.normal(1.0, 0.3 * ext)))
                for key in ("scint_rms", "tiptilt_rms", "seeing_rms", "hf_rms"):
                    tex[key] = tex[key] * f
            tex_used = tex
            digital, star_signal, target_signal = render_frame_empirical(
                height=sensor.height,
                width=sensor.width,
                star_positions=star_positions,
                star_intensities=star_intensities,
                target_positions=target_positions,
                target_intensities=target_intensities,
                kernel=kernel,
                noise=noise,
                star_velocity=star_vel,
                star_rotation=star_rot,
                target_velocities=target_velocities,
                t_osf=star_osf,
                exposure=sensor.exposure,
                tex_scint_rms=tex["scint_rms"],
                tex_tiptilt_rms=tex["tiptilt_rms"],
                tex_seeing_rms=tex["seeing_rms"],
                tex_hf_rms=tex["hf_rms"],
                tex_beta=tex["beta"],
                tex_decon_px=tex["decon_px"],
                tex_hf_klo=tex["hf_klo"],
                tex_hf_khi=tex["hf_khi"],
                tex_hf_slope=tex["hf_slope"],
                tex_floor_frac=tex["floor_frac"],
                size_mode=self.size_mode,
                match_stage=sensor.apply_rowcol_median,
                generator=self._gen,
            )
        else:
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
                defer_read_noise=defer_read_noise,
                star_velocity=star_vel,
                star_rotation=star_rot,
                target_velocities=target_velocities,
                t_osf=star_osf,
                exposure=sensor.exposure,
            )

        # Frame mode label
        if self._catalog is not None:
            frame_mode = "catalog"
        elif self.config.mode == "rate_sidereal":
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
                target_velocities.detach().cpu().tolist() if target_velocities is not None else []
            ),  # [[row_rate, col_rate], ...] px/sec
            "exposure": sensor.exposure,  # seconds
            "obs_time": self.config.obs_time,
            "point_ra": overrides.get("point_ra", self.config.stars.ra),  # deg (star-field center)
            "point_dec": overrides.get("point_dec", self.config.stars.dec),  # deg
            "gain": sensor.gain,
            "read_noise": sensor.read_noise,
            "dark_current": sensor.dark_current,
            "y_fov": sensor.y_fov,
            "x_fov": sensor.x_fov,
            "gap": sensor.gap,
            "psf_sigma": float(psf_sigma) if not isinstance(psf_sigma, float) else psf_sigma,
            "zeropoint": sensor.zeropoint,
            "a2d_dtype": sensor.a2d_dtype,
            "psf_model": sensor.psf_model,
            "noise_model": sensor.noise_model,
            "psf_params": ([float(p) for p in psf_params] if psf_params is not None else None),
            "streak_texture": (tex_used if tex_used is not None else None),
            "_height": sensor.height,
            "_width": sensor.width,
        }

        return digital, star_signal, target_signal, metadata

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
