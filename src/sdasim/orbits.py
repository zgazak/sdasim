"""Orbital mechanics: TLE fetch, SGP4 propagation, angular/pixel rate computation.

No torch dependency — produces numpy/Python outputs consumed by Scene setup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sdasim.config import ObjectConfig, SensorConfig, SiteConfig


@dataclass
class ObjectState:
    """Propagated state of a single object on the focal plane."""

    norad_id: str
    ra: float  # degrees
    dec: float  # degrees
    ra_rate: float  # rad/s
    dec_rate: float  # rad/s
    row_rate: float  # px/s
    col_rate: float  # px/s
    mv: float
    pixel_row: float  # position on focal plane
    pixel_col: float  # position on focal plane


# ---------------------------------------------------------------------------
# TLE fetching
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _fetch_tles_cached(ids_frozen: frozenset[str], timeout: float) -> dict[str, tuple[str, str]]:
    """Cached inner fetch. Returns {norad_id: (line1, line2)}."""
    import httpx

    url = "https://spacebook.com/api/entity/tle"
    resp = httpx.get(url, timeout=timeout)
    resp.raise_for_status()

    tles: dict[str, tuple[str, str]] = {}
    lines = resp.text.strip().splitlines()
    i = 0
    while i < len(lines) - 1:
        line = lines[i].strip()
        # TLE line 1 starts with "1 "
        if line.startswith("1 "):
            l1 = line
            l2 = lines[i + 1].strip()
            # Extract NORAD ID from line 1 columns 2-6
            norad = l1[2:7].strip()
            if norad in ids_frozen:
                tles[norad] = (l1, l2)
            i += 2
        else:
            # Name line — skip
            i += 1

    missing = ids_frozen - set(tles.keys())
    if missing:
        raise ValueError(f"TLEs not found for NORAD IDs: {missing}")

    return tles


def fetch_tles_sync(
    norad_ids: list[str], timeout: float = 30.0
) -> dict[str, tuple[str, str]]:
    """Fetch TLEs for the given NORAD IDs.

    Returns:
        {norad_id: (tle_line1, tle_line2)}
    """
    return _fetch_tles_cached(frozenset(norad_ids), timeout)


# ---------------------------------------------------------------------------
# SGP4 propagation via satkit
# ---------------------------------------------------------------------------


def propagate(
    tle_line1: str,
    tle_line2: str,
    obs_time: datetime,
    site: SiteConfig,
) -> tuple[float, float, float, float, float]:
    """Propagate a TLE to obs_time and compute topocentric RA/Dec + rates.

    Args:
        tle_line1, tle_line2: Two-line element set.
        obs_time: Observation time (UTC).
        site: Observer site.

    Returns:
        (ra_deg, dec_deg, ra_rate_rad_s, dec_rate_rad_s, range_m)
    """
    import satkit

    # Create TLE and propagate
    tle = satkit.TLE.from_lines([tle_line1, tle_line2])
    t = satkit.time(obs_time.year, obs_time.month, obs_time.day,
                    obs_time.hour, obs_time.minute, obs_time.second +
                    obs_time.microsecond / 1e6)

    # Propagate to TEME frame
    pos_teme, vel_teme = satkit.sgp4(tle, t)

    # Convert TEME -> GCRF (inertial)
    q_teme2gcrf = satkit.frametransform.qteme2gcrf(t)
    pos_gcrf = q_teme2gcrf * pos_teme
    vel_gcrf = q_teme2gcrf * vel_teme

    # Observer position in GCRF
    obs_itrf = satkit.itrfcoord(
        latitude_deg=site.latitude,
        longitude_deg=site.longitude,
        altitude=site.altitude,
    )
    obs_gcrf = satkit.frametransform.qitrf2gcrf(t) * obs_itrf.vector

    # Topocentric vector
    topo = pos_gcrf - obs_gcrf
    topo_vel = vel_gcrf  # observer velocity negligible for angular rates

    # Convert to RA/Dec
    range_m = float(np.linalg.norm(np.array(topo)))
    x, y, z = float(topo[0]), float(topo[1]), float(topo[2])

    ra_rad = math.atan2(y, x) % (2 * math.pi)
    dec_rad = math.atan2(z, math.sqrt(x * x + y * y))

    ra_deg = math.degrees(ra_rad)
    dec_deg = math.degrees(dec_rad)

    # Angular rates from topocentric position and velocity
    vx, vy, vz = float(topo_vel[0]), float(topo_vel[1]), float(topo_vel[2])
    rxy2 = x * x + y * y
    rxy = math.sqrt(rxy2)

    # d(RA)/dt = (x*vy - y*vx) / (x^2 + y^2)
    ra_rate = (x * vy - y * vx) / rxy2 if rxy2 > 0 else 0.0

    # d(Dec)/dt = (vz * rxy^2 - z * (x*vx + y*vy)) / (r^2 * rxy)
    r2 = range_m * range_m
    dec_rate = (vz * rxy2 - z * (x * vx + y * vy)) / (r2 * rxy) if rxy > 0 else 0.0

    # Rates are in rad/s (satkit velocities are m/s, positions are m)
    # Actually satkit uses km and km/s, so rates are already rad/s
    return ra_deg, dec_deg, ra_rate, dec_rate, range_m


# ---------------------------------------------------------------------------
# TLE dithering
# ---------------------------------------------------------------------------


def dither_tle(
    tle_line1: str,
    tle_line2: str,
    dither_arcsec: float,
    site: SiteConfig,
    obs_time: datetime,
) -> tuple[str, str]:
    """Perturb TLE orbital elements to shift apparent position.

    Perturbs inclination and RAAN scaled by range / semi-major-axis to achieve
    approximately `dither_arcsec` apparent offset.

    Returns:
        (perturbed_line1, perturbed_line2)
    """
    if dither_arcsec == 0.0:
        return tle_line1, tle_line2

    import satkit

    tle = satkit.TLE.from_lines([tle_line1, tle_line2])

    # Get range
    _, _, _, _, range_m = propagate(tle_line1, tle_line2, obs_time, site)

    # Semi-major axis from mean motion (rev/day)
    mu = 398600.4418  # km^3/s^2
    n_rev_day = tle.mean_motion  # rev/day
    n_rad_s = n_rev_day * 2 * math.pi / 86400.0
    a_km = (mu / (n_rad_s ** 2)) ** (1.0 / 3.0)

    # Scale factor: how much orbital angle change maps to apparent angle change
    range_km = range_m / 1000.0
    scale = range_km / a_km if a_km > 0 else 1.0

    # Convert dither to radians and apply scale
    dither_rad = math.radians(dither_arcsec / 3600.0)
    orbital_dither = dither_rad * scale

    # Random direction
    rng = np.random.default_rng()
    angle = rng.uniform(0, 2 * math.pi)

    # Perturb inclination and RAAN
    inc_pert = orbital_dither * math.cos(angle)
    raan_pert = orbital_dither * math.sin(angle)

    # Modify TLE lines - inclination is cols 8-16 of line 2, RAAN is cols 17-25
    inc_orig = float(tle_line2[8:16])
    raan_orig = float(tle_line2[17:25])

    inc_new = inc_orig + math.degrees(inc_pert)
    raan_new = (raan_orig + math.degrees(raan_pert)) % 360.0

    # Rebuild line 2 with perturbed elements
    new_line2 = (
        tle_line2[:8]
        + f"{inc_new:8.4f}"
        + tle_line2[16:17]
        + f"{raan_new:8.4f}"
        + tle_line2[25:]
    )

    # Fix checksum
    new_line2 = _fix_tle_checksum(new_line2)

    return tle_line1, new_line2


def _fix_tle_checksum(line: str) -> str:
    """Recompute TLE line checksum (last character)."""
    s = 0
    for c in line[:68]:
        if c.isdigit():
            s += int(c)
        elif c == "-":
            s += 1
    return line[:68] + str(s % 10)


# ---------------------------------------------------------------------------
# Angular → pixel rate conversion
# ---------------------------------------------------------------------------


def angular_to_pixel_rates(
    ra_rate: float,
    dec_rate: float,
    dec_deg: float,
    y_fov: float,
    x_fov: float,
    height: int,
    width: int,
) -> tuple[float, float]:
    """Convert angular rates (rad/s) to pixel rates (px/s).

    Args:
        ra_rate: RA angular rate in rad/s.
        dec_rate: Dec angular rate in rad/s.
        dec_deg: Declination in degrees (for cos(dec) correction).
        y_fov: Vertical FOV in degrees.
        x_fov: Horizontal FOV in degrees.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        (row_rate_px_s, col_rate_px_s)
    """
    y_ifov_rad = math.radians(y_fov / height)  # rad per pixel
    x_ifov_rad = math.radians(x_fov / width)

    cos_dec = math.cos(math.radians(dec_deg))

    row_rate = dec_rate / y_ifov_rad
    col_rate = ra_rate * cos_dec / x_ifov_rad

    return row_rate, col_rate


# ---------------------------------------------------------------------------
# WCS projection for focal-plane placement
# ---------------------------------------------------------------------------


def _radec_to_pixel(
    ra_deg: float,
    dec_deg: float,
    ref_ra: float,
    ref_dec: float,
    y_fov: float,
    x_fov: float,
    height: int,
    width: int,
) -> tuple[float, float]:
    """Project RA/Dec onto focal plane relative to reference pointing.

    Uses TAN (gnomonic) projection. Returns (row, col) in pixels.
    """
    from sdasim.sstrc7 import _get_wcs

    y_ifov = y_fov / height
    x_ifov = x_fov / width
    wcs = _get_wcs(height, width, y_ifov, x_ifov, ref_ra, ref_dec, 0.0)

    # world2pix returns (x, y) = (col, row)
    col, row = wcs.wcs_world2pix(
        np.array([ra_deg]), np.array([dec_deg]), 0
    )

    # Shift to image-center origin
    return float(row[0]) + height / 2.0, float(col[0]) + width / 2.0


# ---------------------------------------------------------------------------
# Top-level: compute all object states
# ---------------------------------------------------------------------------


def _is_raw_radec(obj: ObjectConfig) -> bool:
    """True if the object supplies raw RA/Dec + rates (no TLE needed)."""
    return obj.ra is not None and obj.dec is not None


def compute_object_states(
    objects: list[ObjectConfig],
    site: SiteConfig | None,
    sensor: SensorConfig,
    obs_time: datetime,
) -> list[ObjectState]:
    """Compute focal-plane states for all objects.

    Objects with ``norad_id`` are resolved via TLE fetch + SGP4.
    Objects with ``ra``/``dec``/``ra_rate``/``dec_rate`` are used directly.
    The first object is the primary (defines pointing direction).

    Args:
        objects: List of object configurations.
        site: Observer site (required for TLE-based objects).
        sensor: Sensor configuration.
        obs_time: Observation time (UTC).

    Returns:
        List of ObjectState, one per object (same order as input).
    """
    # Fetch TLEs only for objects that need them
    tle_ids = [obj.norad_id for obj in objects if obj.norad_id and not _is_raw_radec(obj)]
    tles = fetch_tles_sync(tle_ids) if tle_ids else {}

    states = []
    primary_ra = None
    primary_dec = None

    for obj in objects:
        if _is_raw_radec(obj):
            # Raw RA/Dec object — rates arrive in deg/s, convert to rad/s
            ra = obj.ra
            dec = obj.dec
            ra_rate_rad = math.radians(obj.ra_rate) if obj.ra_rate else 0.0
            dec_rate_rad = math.radians(obj.dec_rate) if obj.dec_rate else 0.0
        else:
            # TLE-based object
            if site is None:
                raise ValueError(
                    f"site is required for TLE-based object {obj.norad_id!r}"
                )
            l1, l2 = tles[obj.norad_id]
            if obj.dither_arcsec > 0:
                l1, l2 = dither_tle(l1, l2, obj.dither_arcsec, site, obs_time)
            ra, dec, ra_rate_rad, dec_rate_rad, _ = propagate(l1, l2, obs_time, site)

        row_rate, col_rate = angular_to_pixel_rates(
            ra_rate_rad, dec_rate_rad, dec,
            sensor.y_fov, sensor.x_fov,
            sensor.height, sensor.width,
        )

        # Primary target defines pointing
        if primary_ra is None:
            primary_ra = ra
            primary_dec = dec
            pixel_row = sensor.height / 2.0
            pixel_col = sensor.width / 2.0
        else:
            pixel_row, pixel_col = _radec_to_pixel(
                ra, dec, primary_ra, primary_dec,
                sensor.y_fov, sensor.x_fov,
                sensor.height, sensor.width,
            )

        states.append(ObjectState(
            norad_id=obj.norad_id,
            ra=ra,
            dec=dec,
            ra_rate=ra_rate_rad,
            dec_rate=dec_rate_rad,
            row_rate=row_rate,
            col_rate=col_rate,
            mv=obj.mv,
            pixel_row=pixel_row,
            pixel_col=pixel_col,
        ))

    return states
