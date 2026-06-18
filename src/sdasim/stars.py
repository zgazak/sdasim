"""Star catalog generation.

Supports random bins (built-in), SSTR7, and Gaia (require astropy).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from sdasim.device import resolve_device
from sdasim.fpa import mv_to_pe


def generate_random_stars(
    height: int,
    width: int,
    y_fov: float,
    x_fov: float,
    mv_bins: list[float],
    density: list[float],
    zeropoint: float,
    exposure: float,
    pad_mult: float = 1.0,
    seed: int | None = None,
    device: str | torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate random stars using magnitude bins and star density.

    Ports satsim's geometry/random.py:gen_random_points.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        y_fov: Vertical field of view in degrees.
        x_fov: Horizontal field of view in degrees.
        mv_bins: N+1 magnitude bin boundaries.
        density: N star densities per square degree for each bin.
        zeropoint: Sensor zeropoint magnitude.
        exposure: Exposure time in seconds.
        pad_mult: Padding multiplier (generates stars over expanded area).
        seed: Random seed.
        device: Target device.

    Returns:
        (positions, intensities): (M, 2) row/col positions, (M,) PE counts.
    """
    rng = np.random.default_rng(seed)
    dev = resolve_device(device)

    # Total sky area in square degrees
    sky_area = ((2 * pad_mult + 1) * x_fov) * ((2 * pad_mult + 1) * y_fov)

    all_rows = []
    all_cols = []
    all_pe = []

    for i in range(len(density)):
        # Expected number of stars in this bin
        n_expected = density[i] * sky_area
        n_stars = rng.poisson(n_expected)
        if n_stars == 0:
            continue

        # Random magnitudes within this bin
        mv_low, mv_high = mv_bins[i], mv_bins[i + 1]
        mvs = rng.uniform(mv_low, mv_high, size=n_stars)

        # Convert to PE
        pe_per_sec = np.array([mv_to_pe(zeropoint, float(m)) for m in mvs])
        pe = pe_per_sec * exposure

        # Random positions in expanded FOV
        row_min = -height * pad_mult
        row_max = height * (pad_mult + 1)
        col_min = -width * pad_mult
        col_max = width * (pad_mult + 1)

        rows = rng.uniform(row_min, row_max, size=n_stars)
        cols = rng.uniform(col_min, col_max, size=n_stars)

        all_rows.append(rows)
        all_cols.append(cols)
        all_pe.append(pe)

    if not all_rows:
        return (
            torch.zeros(0, 2, dtype=torch.float32, device=dev),
            torch.zeros(0, dtype=torch.float32, device=dev),
        )

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    pe = np.concatenate(all_pe)

    positions = torch.tensor(np.stack([rows, cols], axis=1), dtype=torch.float32, device=dev)
    intensities = torch.tensor(pe, dtype=torch.float32, device=dev)

    return positions, intensities


def load_sstr7(
    height: int,
    width: int,
    y_fov: float,
    x_fov: float,
    ra: float,
    dec: float,
    rot: float,
    zeropoint: float,
    exposure: float,
    catalog_path: str | None = None,
    pad_mult: float = 1.0,
    device: str | torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Load stars from SSTRC7 catalog. Requires astropy.

    Args:
        height, width: Image dimensions.
        y_fov, x_fov: Field of view in degrees.
        ra, dec: Pointing direction in degrees.
        rot: Rotation angle in degrees.
        zeropoint: Sensor zeropoint.
        exposure: Exposure time in seconds.
        catalog_path: Path to SSTRC7 catalog directory.
        pad_mult: Padding multiplier.
        device: Target device.

    Returns:
        (positions, intensities): (M, 2) row/col, (M,) PE counts.
    """
    from sdasim.sstr7 import query_by_los

    dev = resolve_device(device)

    kwargs = {"height": height, "width": width, "y_fov": y_fov, "x_fov": x_fov,
              "ra": ra, "dec": dec, "rot": rot, "pad_mult": pad_mult}
    if catalog_path is not None:
        kwargs["rootPath"] = catalog_path

    rr, cc, mv = query_by_los(**kwargs)

    pe = np.array([mv_to_pe(zeropoint, float(m)) * exposure for m in mv])
    positions = torch.tensor(
        np.stack([np.array(rr), np.array(cc)], axis=1),
        dtype=torch.float32, device=dev,
    )
    intensities = torch.tensor(pe, dtype=torch.float32, device=dev)
    return positions, intensities
