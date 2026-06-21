"""Orbital mechanics: TLE fetch, SGP4 propagation, angular/pixel rate computation.

No torch dependency — produces numpy/Python outputs consumed by Scene setup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sdasim.config import CatalogConfig, ObjectConfig, SensorConfig, SiteConfig


# WGS84 equatorial radius (cylindrical-umbra shadow test) and the Sun's apparent
# visual magnitude (Lambertian-sphere photometry, matches simple-snr-calc).
R_EARTH = 6378137.0
SUN_MV = -26.74
# Earth's sidereal rotation rate [rad/s] (IAU/WGS84). The observer's velocity
# (Omega x r_obs) is subtracted when forming apparent angular rates: it is not
# negligible for high orbits -- ~1.7"/s (~11% of the rate) for GEO.
OMEGA_EARTH = 7.2921159e-5
# Refetch the full Space-Track catalog at most once per day.
_MAX_AGE_HOURS = 24.0
# Representative optical diameter [m] per Space-Track RCS_SIZE bucket. RCS is a
# radar quantity, so these are deliberately coarse population proxies, not truth.
_RCS_DIAMETER = {"SMALL": 0.3, "MEDIUM": 0.8, "LARGE": 2.5}


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
    obs_gcrf = np.asarray(
        satkit.frametransform.qitrf2gcrf(t) * obs_itrf.vector, dtype=float
    ).reshape(3)

    # Topocentric position and velocity, relative to the inertial star field.
    # The observer's own velocity (Earth rotation, Omega x r_obs) is NOT
    # negligible for the apparent angular rate of high orbits -- for GEO it is
    # ~1.7"/s (~11% of the rate) -- so subtract it from the satellite velocity.
    obs_vel = np.cross([0.0, 0.0, OMEGA_EARTH], obs_gcrf)
    topo = np.asarray(pos_gcrf, dtype=float).reshape(3) - obs_gcrf
    topo_vel = np.asarray(vel_gcrf, dtype=float).reshape(3) - obs_vel

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


# ---------------------------------------------------------------------------
# Catalog discovery: auto-stream field satellites through the FOV
# ---------------------------------------------------------------------------


@dataclass
class LoadedCatalog:
    """A parsed TLE catalog plus a per-object optical diameter [m]."""

    tles: list  # list[satkit.TLE]
    diameters: np.ndarray  # (N,) meters, aligned to tles


@dataclass
class StreakState:
    """A discovered satellite streak on the focal plane (sensor frame)."""

    pixel_row: float
    pixel_col: float
    row_rate: float  # px/s, apparent (sensor frame)
    col_rate: float  # px/s
    mv: float
    range_m: float


# In-process cache keyed by (path, mtime, default_diameter) so per-exposure Scene
# rebuilds don't re-parse the 30k-object catalog.
_CAT_CACHE: dict = {}


def _cache_dir() -> Path:
    """OS-agnostic temp dir for Space-Track caches (stable name → cache hits)."""
    import tempfile

    d = Path(tempfile.gettempdir()) / "sdasim"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_creds(env_path: str | None) -> tuple[str, str]:
    """Resolve Space-Track credentials from the environment or a .env file.

    Real environment variables take precedence; a .env file (``env_path`` or a
    ``.env`` in the cwd) is used as a fallback.
    """
    import os

    env: dict[str, str] = {}
    for cand in ([env_path] if env_path else [".env"]):
        if cand and Path(cand).expanduser().exists():
            for line in Path(cand).expanduser().read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")

    ident = os.environ.get("SPACETRACK_USERNAME") or env.get("SPACETRACK_USERNAME")
    pw = os.environ.get("SPACETRACK_PASSWORD") or env.get("SPACETRACK_PASSWORD")
    if not ident or not pw:
        raise ValueError(
            "Space-Track credentials not found; set SPACETRACK_USERNAME and "
            "SPACETRACK_PASSWORD in the environment or a .env file"
        )
    return ident, pw


def _spacetrack_fetch(env_path: str | None) -> Path:
    """Ensure a fresh Space-Track GP cache exists; return its path.

    Re-downloads only when the cache is older than ``_MAX_AGE_HOURS``. The SATCAT
    (for RCS_SIZE → diameter) is fetched best-effort alongside it.
    """
    import time

    import httpx

    gp_path = _cache_dir() / "spacetrack_gp.3le"
    if gp_path.exists() and (time.time() - gp_path.stat().st_mtime) < _MAX_AGE_HOURS * 3600:
        return gp_path

    ident, pw = _load_creds(env_path)
    base = "https://www.space-track.org"
    with httpx.Client(timeout=120.0) as c:
        login = c.post(f"{base}/ajaxauth/login", data={"identity": ident, "password": pw})
        login.raise_for_status()
        gp = c.get(
            f"{base}/basicspacedata/query/class/gp/decay_date/null-val/"
            "epoch/%3Enow-30/orderby/norad_cat_id/format/3le"
        )
        gp.raise_for_status()
        gp_path.write_text(gp.text)
        try:
            sc = c.get(
                f"{base}/basicspacedata/query/class/satcat/DECAY/null-val/CURRENT/Y/"
                "orderby/NORAD_CAT_ID/format/csv"
            )
            sc.raise_for_status()
            (_cache_dir() / "spacetrack_satcat.csv").write_text(sc.text)
        except Exception:
            pass  # diameters fall back to the configured default

    return gp_path


def _diameters_for(tles: list, default_diameter: float) -> np.ndarray:
    """Per-object optical diameter [m] from the cached SATCAT RCS_SIZE buckets."""
    import csv

    diam = np.full(len(tles), float(default_diameter))
    sc_path = _cache_dir() / "spacetrack_satcat.csv"
    if not sc_path.exists():
        return diam

    sizes: dict[int, float] = {}
    with open(sc_path, newline="") as f:
        for row in csv.DictReader(f):
            bucket = (row.get("RCS_SIZE") or "").strip().upper()
            if bucket in _RCS_DIAMETER:
                try:
                    sizes[int(row["NORAD_CAT_ID"])] = _RCS_DIAMETER[bucket]
                except (KeyError, ValueError):
                    continue

    for i, tle in enumerate(tles):
        d = sizes.get(int(tle.satnum))
        if d:
            diam[i] = d
    return diam


def load_catalog(cfg: CatalogConfig) -> LoadedCatalog:
    """Load (and cache) the TLE catalog described by a CatalogConfig.

    ``cfg.source == "spacetrack"`` fetches the full GP catalog (daily cache);
    any other value is treated as a path to a local 2-/3-line file.
    """
    import satkit

    if cfg.source == "spacetrack":
        path = _spacetrack_fetch(cfg.env_path)
    else:
        path = Path(cfg.source).expanduser()

    key = (str(path), path.stat().st_mtime, float(cfg.default_diameter))
    if key not in _CAT_CACHE:
        tles = satkit.TLE.from_file(str(path))
        if not isinstance(tles, list):
            tles = [tles]
        _CAT_CACHE[key] = LoadedCatalog(tles, _diameters_for(tles, cfg.default_diameter))
    return _CAT_CACHE[key]


def _phase_angle_factor(phi_rad: np.ndarray) -> np.ndarray:
    """Lambertian-sphere phase factor (Tousey/Sussman); matches simple-snr-calc."""
    return (2.0 / (3.0 * np.pi**2)) * (np.sin(phi_rad) + (np.pi - phi_rad) * np.cos(phi_rad))


def discover_streaks(
    catalog: LoadedCatalog,
    obs_time: datetime,
    site: SiteConfig,
    sensor: SensorConfig,
    point_ra: float,
    point_dec: float,
    mount_ra_rate: float,  # rad/s, inertial (0 == sidereal)
    mount_dec_rate: float,  # rad/s
    albedo: float,
    psf_pad_px: float,
) -> list[StreakState]:
    """Find sunlit satellites whose streak touches the FOV at ``obs_time``.

    Propagates the whole catalog at one epoch (vectorized), keeps objects whose
    streak segment intersects the PSF-padded frame, are above the horizon and
    are not eclipsed, then returns them as sensor-frame line targets.
    """
    import satkit

    from sdasim.sstrc7 import _get_wcs

    tm = satkit.time(
        obs_time.year, obs_time.month, obs_time.day,
        obs_time.hour, obs_time.minute,
        obs_time.second + obs_time.microsecond / 1e6,
    )

    # Observer + Sun in GCRF (meters)
    obs_itrf = satkit.itrfcoord(
        latitude_deg=site.latitude, longitude_deg=site.longitude, altitude=site.altitude,
    )
    obs_g = np.asarray(satkit.frametransform.qitrf2gcrf(tm) * obs_itrf.vector, float).reshape(3)
    sun_g = np.asarray(satkit.sun.pos_gcrf(tm), float).reshape(3)

    # Propagate the whole catalog at this epoch (vectorized); velocity comes free.
    # Rotate TEME->GCRF with the quaternion's rotation MATRIX and matmul: `q * arr`
    # does NOT apply the rotation row-wise for an (N, 3) array (it only rotates a
    # single 3-vector correctly). satkit squeezes the single-TLE case to 1-D, so
    # force (N, 3) first.
    p_teme, v_teme = satkit.sgp4(catalog.tles, tm)
    rot = np.asarray(satkit.frametransform.qteme2gcrf(tm).as_rotation_matrix(), float)
    p_g = np.asarray(p_teme, float).reshape(-1, 3) @ rot.T
    v_g = np.asarray(v_teme, float).reshape(-1, 3) @ rot.T

    # Subtract the observer's velocity (Earth rotation) so rates are relative to
    # the inertial star field; non-negligible for high orbits (see propagate()).
    obs_vel = np.cross([0.0, 0.0, OMEGA_EARTH], obs_g)
    topo = p_g - obs_g[None, :]
    v_rel = v_g - obs_vel[None, :]
    x, y, z = topo[:, 0], topo[:, 1], topo[:, 2]
    vx, vy, vz = v_rel[:, 0], v_rel[:, 1], v_rel[:, 2]

    rng = np.sqrt(x * x + y * y + z * z)
    rxy2 = x * x + y * y
    rxy = np.sqrt(rxy2)
    ra = np.degrees(np.arctan2(y, x)) % 360.0
    dec = np.degrees(np.arctan2(z, rxy))

    with np.errstate(divide="ignore", invalid="ignore"):
        ra_rate = np.where(rxy2 > 0, (x * vy - y * vx) / rxy2, 0.0)
        dec_rate = np.where(
            rxy > 0, (vz * rxy2 - z * (x * vx + y * vy)) / (rng * rng * rxy), 0.0
        )

    # Apparent (sensor-frame) rates = inertial - mount. Sidereal is just mount=0.
    app_ra_rate = ra_rate - mount_ra_rate
    app_dec_rate = dec_rate - mount_dec_rate
    cos_dec = np.cos(np.radians(dec))
    omega = np.sqrt((app_ra_rate * cos_dec) ** 2 + app_dec_rate**2)  # on-sky rad/s

    # Coarse great-circle cull around the pointing (per-object streak margin).
    pr, pd = math.radians(point_ra), math.radians(point_dec)
    dr, dd = np.radians(ra), np.radians(dec)
    cos_sep = math.sin(pd) * np.sin(dd) + math.cos(pd) * np.cos(dd) * np.cos(dr - pr)
    sep = np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))

    fov_r = 0.5 * math.hypot(sensor.x_fov, sensor.y_fov)
    ifov = 0.5 * (sensor.x_fov / sensor.width + sensor.y_fov / sensor.height)  # deg/px
    streak_deg = np.degrees(omega * sensor.exposure)
    keep = sep < (fov_r + streak_deg + psf_pad_px * ifov)

    # Above the local horizon.
    up = obs_g / np.linalg.norm(obs_g)
    keep &= (topo @ up) > 0

    # Sunlit only (cylindrical umbra): eclipsed objects are never rendered.
    sh = sun_g / np.linalg.norm(sun_g)
    along = p_g @ sh
    perp = np.linalg.norm(p_g - along[:, None] * sh[None, :], axis=1)
    keep &= ~((along < 0) & (perp < R_EARTH))

    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        return []

    # Apparent magnitude (Lambertian sphere; inverse of radius_from_mv).
    to_sun = sun_g[None, :] - p_g[idx]
    to_obs = -topo[idx]
    cphi = np.sum(to_sun * to_obs, axis=1) / (
        np.linalg.norm(to_sun, axis=1) * np.linalg.norm(to_obs, axis=1)
    )
    paf = _phase_angle_factor(np.arccos(np.clip(cphi, -1.0, 1.0)))
    r = catalog.diameters[idx] / 2.0
    mv = SUN_MV - 2.5 * np.log10(
        np.clip(albedo * paf * np.pi * r * r / (rng[idx] ** 2), 1e-30, None)
    )

    # Project to pixels (center origin → array index), matching _radec_to_pixel.
    H, W = sensor.height, sensor.width
    y_ifov, x_ifov = sensor.y_fov / H, sensor.x_fov / W
    wcs = _get_wcs(H, W, y_ifov, x_ifov, point_ra, point_dec, 0.0)
    col, row = wcs.wcs_world2pix(ra[idx], dec[idx], 0)
    row = np.asarray(row, float) + H / 2.0
    col = np.asarray(col, float) + W / 2.0

    vrow = app_dec_rate[idx] / math.radians(y_ifov)
    vcol = app_ra_rate[idx] * cos_dec[idx] / math.radians(x_ifov)

    # Entry/exit: streak-segment bbox vs PSF-padded frame (over-inclusive = safe).
    r1r = row + vrow * sensor.exposure
    r1c = col + vcol * sensor.exposure
    on = (
        (np.maximum(row, r1r) >= -psf_pad_px)
        & (np.minimum(row, r1r) <= H + psf_pad_px)
        & (np.maximum(col, r1c) >= -psf_pad_px)
        & (np.minimum(col, r1c) <= W + psf_pad_px)
    )

    return [
        StreakState(
            pixel_row=float(row[j]), pixel_col=float(col[j]),
            row_rate=float(vrow[j]), col_rate=float(vcol[j]),
            mv=float(mv[j]), range_m=float(rng[idx][j]),
        )
        for j in np.nonzero(on)[0]
    ]
