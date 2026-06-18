"""Optional I/O writers for FITS, annotations, and sequences.

These are convenience utilities for batch workflow, never in the hot path.
Requires astropy for FITS writing (lazy import).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
from torch import Tensor

def _build_fits_header(meta: dict, frame_index: int) -> dict:
    """Build FITS header dict from render metadata.

    Args:
        meta: Metadata dict from Scene.render().
        frame_index: Sequence index of this frame.

    Returns:
        Dict of FITS header keyword/value pairs.
    """
    header: dict = {}

    header["FRAMEIDX"] = frame_index

    # Exposure time (required by senpai)
    exposure = meta.get("exposure")
    if exposure is not None:
        header["EXPTIME"] = (float(exposure), "[s] exposure time")

    # Observation date/time (required by senpai)
    obs_time_str = meta.get("obs_time")
    if obs_time_str:
        try:
            base_time = datetime.fromisoformat(obs_time_str).replace(tzinfo=timezone.utc)
        except ValueError:
            base_time = datetime.now(timezone.utc)
    else:
        base_time = datetime.now(timezone.utc)
    frame_time = meta.get("frame_time", 0.0)
    obs_dt = base_time + timedelta(seconds=frame_time)
    header["DATE-OBS"] = (obs_dt.strftime("%Y-%m-%dT%H:%M:%S.%f"), "UTC observation start")

    # Sensor / detector parameters
    if "gain" in meta:
        header["GAIN"] = (float(meta["gain"]), "[e-/DN] detector gain")
    if "read_noise" in meta:
        header["RDNOISE"] = (float(meta["read_noise"]), "[e-] read noise RMS")
    if "dark_current" in meta:
        header["DARKCUR"] = (float(meta["dark_current"]), "[e-/pix/s] dark current")
    if "psf_sigma" in meta:
        header["PSFSIGMA"] = (float(meta["psf_sigma"]), "[pix] PSF Gaussian sigma")
    if "zeropoint" in meta:
        header["ZPOINT"] = (float(meta["zeropoint"]), "[mag] photometric zeropoint")

    # Field of view
    if "y_fov" in meta:
        header["YFOV"] = (float(meta["y_fov"]), "[deg] vertical field of view")
    if "x_fov" in meta:
        header["XFOV"] = (float(meta["x_fov"]), "[deg] horizontal field of view")

    # Frame timing
    if "gap" in meta:
        header["FRAMEGAP"] = (float(meta["gap"]), "[s] inter-frame gap")

    # Tracking mode and rates (used by senpai for frame classification)
    frame_mode = meta.get("frame_mode")
    if frame_mode is not None:
        header["TRKMODE"] = (frame_mode, "tracking mode")
    star_vel = meta.get("star_velocity", [0.0, 0.0])
    height = meta.get("_height")
    width = meta.get("_width")
    y_fov = meta.get("y_fov")
    x_fov = meta.get("x_fov")
    if y_fov and x_fov and height and width:
        ps_ra = x_fov * 3600.0 / width
        ps_dec = y_fov * 3600.0 / height
        header["RARATE"] = (star_vel[1] * ps_ra, "[arcsec/s] RA tracking rate")
        header["DECRATE"] = (star_vel[0] * ps_dec, "[arcsec/s] Dec tracking rate")

    # Source counts
    if "num_stars" in meta:
        header["NSTARS"] = (int(meta["num_stars"]), "number of star sources")
    if "num_targets" in meta:
        header["NTARGETS"] = (int(meta["num_targets"]), "number of target sources")

    header["SIMULATE"] = (True, "synthetic image from sdasim")

    return header


def write_fits(image: Tensor, path: str | Path, header: dict | None = None) -> None:
    """Write a 2D image tensor to a FITS file.

    Args:
        image: (H, W) image tensor.
        path: Output file path.
        header: Optional FITS header keywords. Values can be plain values
                or (value, comment) tuples.
    """
    try:
        from astropy.io import fits
    except ImportError:
        raise ImportError("FITS writing requires astropy. Install with: pip install astropy")

    import numpy as np

    data = image.detach().cpu().numpy().astype(np.float32)
    hdu = fits.PrimaryHDU(data)
    if header:
        for key, val in header.items():
            hdu.header[key] = val
    hdu.writeto(str(path), overwrite=True)


def write_annotations(metadata: dict | list[dict], path: str | Path) -> None:
    """Write annotation metadata to a JSON file.

    Args:
        metadata: Single frame metadata dict or list of dicts.
        path: Output file path.
    """
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def write_sequence(
    images: Tensor,
    metadata: list[dict],
    output_dir: str | Path,
    prefix: str = "frame",
    fmt: str = "fits",
) -> list[Path]:
    """Write a sequence of rendered frames and annotations.

    Args:
        images: (N, H, W) tensor of rendered frames.
        metadata: List of metadata dicts, one per frame.
        output_dir: Output directory.
        prefix: Filename prefix.
        fmt: Output format ('fits' or 'npy').

    Returns:
        List of written file paths.
    """
    import numpy as np

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(images.shape[0]):
        frame = images[i]
        if fmt == "fits":
            fpath = out / f"{prefix}_{i:04d}.fits"
            header = _build_fits_header(metadata[i], i)
            write_fits(frame, fpath, header=header)
        elif fmt == "npy":
            fpath = out / f"{prefix}_{i:04d}.npy"
            np.save(str(fpath), frame.detach().cpu().numpy())
        else:
            raise ValueError(f"Unknown format: {fmt}")
        paths.append(fpath)

    # Write annotations
    ann_path = out / f"{prefix}_annotations.json"
    write_annotations(metadata, ann_path)
    paths.append(ann_path)

    return paths
