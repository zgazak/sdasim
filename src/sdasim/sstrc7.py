"""SSTRC7 star catalog access, backed by the ``sstrc7`` package.

This module used to carry its own copy of the SSTRC7 reader. That reader
decoded one 60-byte record at a time with ``struct.unpack`` and cached the
parsed results in a byte-bounded LRU, which is why it needed a memory budget
at all. The ``sstrc7`` package reads the same files through numpy memory maps
and a structured dtype, so records are never decoded individually and the OS
page cache handles eviction -- no budget to tune, and queries run roughly an
order of magnitude faster.

Install it with the ``catalogs`` extra::

    pip install "sdasim[catalogs]"

and fetch the data once with ``sstrc7 get`` (see ``python -m sstrc7.cli info``).

The two entry points below keep the signatures and return types the previous
implementation had, so callers do not change. One behavioural difference is
worth knowing: the old RA clip worked by truncating a zone at the first record
that crossed the bound, which let stars up to a six-degree RA zone past the
requested box leak into the result, and could drop a star exactly on the
boundary. Results are now exactly the requested region -- verified
star-for-star against a brute-force scan, including across the RA = 0 wrap and
over both poles. Widen ``pad_mult`` if you were relying on the extra margin.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Kept for backwards compatibility. Prefer leaving ``rootPath`` unset, which
# lets the sstrc7 package resolve $SDASIM_SSTRC7_PATH, then $SSTRC7_PATH, then
# ~/.sstrc7.
DEFAULT_SSTRC7_PATH = os.environ.get("SDASIM_SSTRC7_PATH", "sstrc7")

RECORD_LEN_BYTES = 60

#: Compact per-star representation returned by :func:`query_by_min_max`.
#: ``ra`` and ``dec`` are radians; ``mv`` is a magnitude, 32.0 when unmeasured.
_STAR_DTYPE = np.dtype([("ra", "f8"), ("dec", "f8"), ("mv", "f8")])

_MISSING_MAGNITUDE = 32.0


def _resolve(root_path: str | os.PathLike[str] | None):
    """Open the catalog, preserving the old relative-directory default.

    Passing ``None`` defers to the sstrc7 package's own resolution order. If
    nothing is configured there but a ``sstrc7/`` directory sits in the working
    directory -- the historical default of this module -- use that.
    """
    from sstrc7 import open_catalog
    from sstrc7.paths import PATH_ENV_VARS

    if root_path is None and not any(os.environ.get(var) for var in PATH_ENV_VARS):
        legacy = Path(DEFAULT_SSTRC7_PATH)
        if legacy.is_dir():
            root_path = legacy

    return open_catalog(root_path)


def _get_wcs(height, width, y_ifov, x_ifov, ra, dec, rot=0.0):
    """Build an astropy WCS for focal-plane <-> sky projection.

    ``height`` and ``width`` are accepted (and ignored) so the signature keeps
    matching the callers in :mod:`sdasim.orbits`; the projection is fully
    determined by the instantaneous field of view, pointing, and rotation.
    """
    from sstrc7.query import _build_wcs

    return _build_wcs(y_ifov, x_ifov, ra, dec, rot)


def _magnitudes(field, filter_center: float | None) -> np.ndarray:
    """One magnitude per star, with the catalog's sentinel for unmeasured stars."""
    mag = field.at_wavelength(filter_center) if filter_center is not None else field.visual
    return np.where(np.isnan(mag), _MISSING_MAGNITUDE, mag).astype(np.float64)


def query_by_min_max(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    rootPath: str | None = None,
    clip_min_max: bool = True,
    filter_center: float | None = None,
) -> np.ndarray:
    """Query the catalog by RA/Dec bounds in radians.

    Args:
        ra_min, ra_max: right ascension bounds. ``ra_min > ra_max`` selects the
            interval wrapping through zero.
        dec_min, dec_max: declination bounds.
        rootPath: catalog directory, or None to resolve from the environment.
        clip_min_max: accepted for signature compatibility. Results are always
            clipped exactly to the requested bounds.
        filter_center: interpolate magnitudes to this wavelength in nm instead
            of taking the best available broadband magnitude.

    Returns:
        A structured array of :data:`_STAR_DTYPE`.
    """
    del clip_min_max  # results are always exact now

    field = _resolve(rootPath).query_box(ra_min, ra_max, dec_min, dec_max, radians=True)

    out = np.empty(len(field), dtype=_STAR_DTYPE)
    out["ra"] = field.ra_rad
    out["dec"] = field.dec_rad
    out["mv"] = _magnitudes(field, filter_center)
    return out


def query_by_los(
    height: int,
    width: int,
    y_fov: float,
    x_fov: float,
    ra: float,
    dec: float,
    rot: float = 0.0,
    rootPath: str | None = None,
    pad_mult: float = 0.0,
    origin: str = "center",
    filter_ob: bool = True,
    filter_center: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query the catalog and project stars onto the focal plane.

    Args:
        height, width: image dimensions in pixels.
        y_fov, x_fov: field of view in degrees.
        ra, dec: pointing direction in degrees.
        rot: focal-plane rotation in degrees.
        rootPath: catalog directory, or None to resolve from the environment.
        pad_mult: padding multiplier for the star field.
        origin: 'center' or 'corner'.
        filter_ob: if True, remove stars outside the padded FOV.
        filter_center: optional wavelength (nm) for magnitude interpolation.

    Returns:
        (rows, cols, magnitudes) — pixel positions and visual magnitudes.
    """
    return _resolve(rootPath).query_by_los(
        height,
        width,
        y_fov,
        x_fov,
        ra,
        dec,
        rot,
        pad_mult=pad_mult,
        origin=origin,
        filter_ob=filter_ob,
        filter_center=filter_center,
    )
