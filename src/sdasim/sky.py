"""Sun-driven sky surface brightness model: night / twilight / day / moon.

Returns sky surface brightness in mag/arcsec^2 (Vega) per band as a function
of sun altitude, sun-target elongation, and (optionally) moon geometry. Sky
surface brightness is intensive — independent of the optics — so this module
stays in mag/arcsec^2 and the Scene converts to photoelectrons per pixel via
the sensor zeropoint and plate scale.

Physics and sources (per-number citations sit next to each constant below):

  * NIGHT: dark-site zenith values. Optical V/R/I from Paranal at sunspot
    maximum (Patat 2003, A&A 400, 1183); near-IR broadband J/H/Ks from
    Paranal/ISAAC, OH-airglow dominated (Cuby, Lidman & Moutou 2000, ESO
    Messenger 101, 2).
  * TWILIGHT (V/R/I): the Patat, Ugolnikov & Postylyakov 2006 (A&A 455, 385)
    zenith polynomial ``m = a0 + a1*(zeta-95) + a2*(zeta-95)^2`` in sun zenith
    distance zeta (deg), valid zeta = 95-105; the sky reaches the night level
    at zeta ~ 105.5-106 — i.e. *before* formal astronomical dark.
  * TWILIGHT (SWIR): the SWIR sky reaches full night darkness much earlier
    than the visible — measured twilight-to-night boundaries at sun altitude
    -9.3 deg (J) and -7.4 deg (H) at Dome A (AIRBT; Li et al. 2026,
    arXiv:2607.01923). Modeled here as a smooth monotone (smoothstep)
    interpolation from the sunset (sun_alt = 0) daytime value down to the
    night value at the band's boundary altitude.
  * DAY: solar-elongation-dependent tables anchored to MODTRAN5 Haleakala
    GEO-belt curves (Jim, Gibson & Pier 2012, AMOS) for J/K, reconciled with
    the IRTF measurement of 6-6.5 mag/arcsec^2 in J, H and K at 45 deg from
    the Sun (J. Rayner, reported in Jim et al. 2012); visible from MODTRAN
    daytime modeling, ~2 mag/arcsec^2 near the Sun to ~5 in deep blue sky
    (Zimmer, McGraw & Ackermann 2020, AMOS). H is interpolated between J and
    K. A small sun-altitude term (~0.3 mag brighter at sun_alt 60 vs 15)
    captures the midday-vs-morning airmass-path difference (Jim et al. 2012
    diurnal curves).
  * MOON: full Krisciunas & Schaefer 1991 (PASP 103, 1033) analytic scattered
    moonlight model for V (accuracy 8-23%), applied as a flux increment so it
    fades out naturally in twilight/day; the same magnitude delta is applied
    to R/I. Y/J backgrounds rise ~1 mag (x2.5) only for a gibbous-to-full
    moon within 30 deg of the line of sight; H/Ks are unaffected (Sullivan &
    Simcoe 2012, PASP).
  * SWIR_YJH (1.0-1.7 um InGaAs longpass) is synthesized at every regime as
    the flux-weighted combination of the J and H model values (weights =
    Vega zeropoint x bandwidth), giving a night level of ~15.3 mag/arcsec^2.

Sun ephemerides use satkit (already an sdasim dependency; same patterns as
``orbits.py``): ``satkit.sun.pos_gcrf`` for the geocentric sun vector and the
ITRF/ENU site transforms for altitude-azimuth. satkit is imported lazily so
the radiometric model works without it.

Continuity: ``sky_brightness`` is continuous in sun altitude across all
regime boundaries for every band (piecewise models share their endpoint
values by construction).

No torch dependency — plain floats, consumed by Scene setup.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Mapping

import numpy as np

__all__ = [
    "sun_radec",
    "sun_altaz",
    "elongation_deg",
    "sky_brightness",
    "sky_summary",
]


# ---------------------------------------------------------------------------
# Band tables (local to sky.py for now; integration reconciles with bands.py)
# ---------------------------------------------------------------------------

# Dark-site zenith night sky [mag/arcsec^2, Vega].
# V/R/I: Paranal at sunspot maximum (Patat 2003, A&A 400, 1183).
# J/H/KS: Paranal/ISAAC broadband, OH-dominated (Cuby et al. 2000, ESO
# Messenger 101). SWIR_YJH is derived from J+H (see _yjh_from_jh), ~15.3.
_NIGHT_MAG: dict[str, float] = {
    "V": 21.6,
    "R": 20.9,
    "I": 19.7,
    "J": 16.5,
    "H": 14.4,
    "KS": 13.0,
}

# Sun altitude [deg] at (or below) which each band is at its night level.
# V/R/I: night level reached at sun zenith distance ~105.5-106 deg (Patat et
# al. 2006); we bridge the polynomial to the night table over 105.5-106.5.
# J/H: measured twilight-to-night boundaries at Dome A: sun -9.3 deg (J),
# -7.4 deg (H) (Li et al. 2026, arXiv:2607.01923, AIRBT). KS: no published
# boundary; assumed equal to H (night KS is brighter than H, so it can only
# be reached earlier — -7.4 is conservative).
_NIGHT_SUN_ALT: dict[str, float] = {
    "V": -16.5,
    "R": -16.5,
    "I": -16.5,
    "J": -9.3,
    "H": -7.4,
    "KS": -7.4,
}

# Twilight polynomial m(zeta) = a0 + a1*(zeta-95) + a2*(zeta-95)^2, zeta =
# sun zenith distance [deg], valid 95-105 (Patat, Ugolnikov & Postylyakov
# 2006, A&A 455, 385, Table 1; RMS 0.12-0.40 mag).
_PATAT_COEFFS: dict[str, tuple[float, float, float]] = {
    "V": (11.84, 1.518, -0.057),
    "R": (11.40, 1.567, -0.064),
    "I": (10.93, 1.470, -0.062),
}
_PATAT_ZETA_MAX = 105.5  # end of polynomial validity (clamped)
_PATAT_ZETA_MIN = 95.0
_PATAT_NIGHT_ZETA = 106.5  # night table reached here (1-deg bridge)

# Daytime sky vs solar elongation [ (elongation_deg, mag/arcsec^2), ... ].
# V: MODTRAN daytime visible, ~2 near the Sun to ~5 in deep blue sky far from
# it (Zimmer, McGraw & Ackermann 2020, AMOS; Caddy 2024 consistent) — 2.7 at
# the 12-deg solar-exclusion edge rising to ~4.5 at 150 deg.
_DAY_TABLE_V: list[tuple[float, float]] = [
    (12.0, 2.7), (30.0, 3.0), (60.0, 3.3), (90.0, 3.7),
    (120.0, 4.2), (150.0, 4.5), (165.0, 4.6),
]
# J/K: MODTRAN5 Haleakala GEO-belt ranges (Jim, Gibson & Pier 2012, AMOS):
# J 3.5-5 at 20-30 deg, 5-6.3 at 60, 5.7-7.4 at 90, flat 5.8-6.8 beyond 120;
# K 5.4-6.6 at 20-30, 6.5-7.8 at 60, 7-8.8 at 90-160. The J knots below sit
# at the dark (high-look-elevation) edge of the Jim ranges so that the curve
# reproduces the IRTF *measurement* of 6-6.5 mag/arcsec^2 in J/H/K at 45 deg
# elongation (Rayner, in Jim et al. 2012) — the mid-range Jim values would
# give J(45) ~ 4.9, ~1 mag brighter than measured.
_DAY_TABLE_J: list[tuple[float, float]] = [
    (12.0, 4.4), (30.0, 5.0), (60.0, 6.2), (90.0, 6.5),
    (120.0, 6.5), (165.0, 6.6),
]
_DAY_TABLE_K: list[tuple[float, float]] = [
    (12.0, 5.2), (30.0, 6.0), (60.0, 7.5), (90.0, 8.3),
    (120.0, 8.3), (165.0, 8.4),
]
# Daytime R and I ride the V curve, redder = darker (Rayleigh ~lambda^-4 sky
# is blue; offsets bracket the V->I daytime color per Zimmer et al. 2020).
_DAY_R_OFFSET = 0.3
_DAY_I_OFFSET = 0.8

_ELONG_MIN = 12.0   # solar-exclusion edge (daytime systems: 15-20 deg typ.)
_ELONG_MAX = 165.0

# Vega zeropoints F0 [W/m^2/um] (Bessell 1998, via sky radiometry survey) and
# 2MASS bandwidths [um] (Skrutskie et al. 2006; Cohen et al. 2003) used to
# flux-weight J and H into the SWIR_YJH longpass composite.
_F0_J_W_M2_UM = 3.13e-9
_F0_H_W_M2_UM = 1.13e-9
_WIDTH_J_UM = 0.162
_WIDTH_H_UM = 0.251
_W_J = _F0_J_W_M2_UM * _WIDTH_J_UM
_W_H = _F0_H_W_M2_UM * _WIDTH_H_UM

_BAND_NAMES = ("V", "R", "I", "J", "H", "KS", "SWIR_YJH")


# ---------------------------------------------------------------------------
# Sun ephemerides (satkit)
# ---------------------------------------------------------------------------


def _parse_time(obs_time: str | datetime) -> datetime:
    """Coerce an ISO-8601 string or datetime to a UTC datetime.

    Naive datetimes are assumed to already be UTC (same convention as
    ``orbits.propagate``).
    """
    if isinstance(obs_time, str):
        obs_time = datetime.fromisoformat(obs_time.replace("Z", "+00:00"))
    if obs_time.tzinfo is not None:
        obs_time = obs_time.astimezone(timezone.utc).replace(tzinfo=None)
    return obs_time


def _satkit_time(obs_time: str | datetime):
    """Build a satkit.time from an ISO string / datetime (UTC)."""
    import satkit

    t = _parse_time(obs_time)
    return satkit.time(
        t.year, t.month, t.day,
        t.hour, t.minute, t.second + t.microsecond / 1e6,
    )


def sun_radec(obs_time: str | datetime) -> tuple[float, float]:
    """Geocentric (GCRF) right ascension / declination of the Sun.

    Solar parallax (< 9 arcsec) is ignored — negligible against the tens of
    degrees over which the sky model varies.

    Args:
        obs_time: Observation time (UTC): ISO-8601 string (``Z`` suffix ok)
            or datetime (naive = UTC).

    Returns:
        (ra_deg, dec_deg) with ra in [0, 360).
    """
    import satkit

    tm = _satkit_time(obs_time)
    sun_g = np.asarray(satkit.sun.pos_gcrf(tm), dtype=float).reshape(3)
    ra = math.degrees(math.atan2(sun_g[1], sun_g[0])) % 360.0
    dec = math.degrees(math.atan2(sun_g[2], math.hypot(sun_g[0], sun_g[1])))
    return ra, dec


def sun_altaz(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    obs_time: str | datetime,
) -> tuple[float, float]:
    """Topocentric altitude / azimuth of the Sun (geometric, no refraction).

    Same satkit site pattern as ``orbits.propagate``: the site is an ITRF
    coordinate; the sun GCRF vector (``satkit.sun.pos_gcrf``) is rotated into
    ITRF and expressed in the site's local East-North-Up frame.

    Args:
        lat_deg: Geodetic latitude [deg, +N].
        lon_deg: Longitude [deg, +E].
        alt_m: Site altitude above the WGS84 ellipsoid [m].
        obs_time: Observation time (UTC): ISO string or datetime.

    Returns:
        (alt_deg, az_deg): altitude above the horizon and azimuth measured
        east of true north, both in degrees.
    """
    import satkit

    tm = _satkit_time(obs_time)
    sun_g = np.asarray(satkit.sun.pos_gcrf(tm), dtype=float).reshape(3)

    site = satkit.itrfcoord(
        latitude_deg=lat_deg, longitude_deg=lon_deg, altitude=alt_m,
    )
    sun_itrf = np.asarray(
        satkit.frametransform.qgcrf2itrf(tm) * sun_g, dtype=float
    ).reshape(3)
    topo = sun_itrf - np.asarray(site.vector, dtype=float).reshape(3)
    east, north, up = np.asarray(site.qenu2itrf.conj * topo, dtype=float).reshape(3)

    az = math.degrees(math.atan2(east, north)) % 360.0
    alt = math.degrees(math.atan2(up, math.hypot(east, north)))
    return alt, az


def elongation_deg(
    sun_radec: tuple[float, float],
    target_radec: tuple[float, float],
) -> float:
    """Great-circle angle between the Sun and a target [deg].

    Uses the numerically stable atan2 (Vincenty) form — exact at 0 and 180.

    Args:
        sun_radec: (ra_deg, dec_deg) of the Sun.
        target_radec: (ra_deg, dec_deg) of the target.

    Returns:
        Solar elongation in degrees, in [0, 180].
    """
    ra1, dec1 = math.radians(sun_radec[0]), math.radians(sun_radec[1])
    ra2, dec2 = math.radians(target_radec[0]), math.radians(target_radec[1])
    dra = ra2 - ra1
    num = math.hypot(
        math.cos(dec2) * math.sin(dra),
        math.cos(dec1) * math.sin(dec2)
        - math.sin(dec1) * math.cos(dec2) * math.cos(dra),
    )
    den = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(dra)
    )
    return math.degrees(math.atan2(num, den))


# ---------------------------------------------------------------------------
# Radiometric helpers
# ---------------------------------------------------------------------------


def _yjh_from_jh(mag_j: float, mag_h: float) -> float:
    """Flux-weighted J+H composite for the 1.0-1.7 um InGaAs longpass band.

    Weights are Vega zeropoint x bandwidth per band, so equal-magnitude J and
    H skies combine to the same magnitude. Night J=16.5 / H=14.4 gives ~15.3.
    """
    flux = _W_J * 10.0 ** (-0.4 * mag_j) + _W_H * 10.0 ** (-0.4 * mag_h)
    return -2.5 * math.log10(flux / (_W_J + _W_H))


def _day_base_mag(band: str, elong: float) -> float:
    """Daytime sky vs solar elongation (tables above), before sun-alt term."""
    elong = min(max(elong, _ELONG_MIN), _ELONG_MAX)
    if band in ("V", "R", "I"):
        knots = _DAY_TABLE_V
        offset = {"V": 0.0, "R": _DAY_R_OFFSET, "I": _DAY_I_OFFSET}[band]
    elif band == "J":
        knots, offset = _DAY_TABLE_J, 0.0
    elif band == "KS":
        knots, offset = _DAY_TABLE_K, 0.0
    elif band == "H":
        # H sits between J and K; the mean of the two curves gives H(45 deg)
        # ~ 6.2, matching the IRTF 6-6.5 measurement (Jim et al. 2012).
        return 0.5 * (_day_base_mag("J", elong) + _day_base_mag("KS", elong))
    else:  # pragma: no cover - guarded by _normalize_band
        raise ValueError(f"no day table for band {band!r}")
    xs = [k[0] for k in knots]
    ys = [k[1] for k in knots]
    return float(np.interp(elong, xs, ys)) + offset


def _sun_alt_day_term(sun_alt: float) -> float:
    """Mag delta for sun altitude in daytime: ~0.3 mag brighter at 60 vs 15.

    Higher sun = shorter scattering path to the line of sight = brighter sky
    (Jim et al. 2012 diurnal curves show a few-tenths-mag midday brightening).
    Linear in sun altitude, referenced to 0 at sun_alt = 15, clamped flat
    outside 0-60 deg. Negative return = brighter (magnitudes).
    """
    return -0.3 * (min(max(sun_alt, 0.0), 60.0) - 15.0) / 45.0


def _day_mag(band: str, sun_alt: float, elong: float) -> float:
    """Full daytime model: elongation table + sun-altitude term."""
    return _day_base_mag(band, elong) + _sun_alt_day_term(sun_alt)


def _patat_mag(band: str, zeta: float) -> float:
    """Patat et al. 2006 twilight polynomial at sun zenith distance zeta."""
    a0, a1, a2 = _PATAT_COEFFS[band]
    x = zeta - 95.0
    return a0 + a1 * x + a2 * x * x


def _smoothstep(t: float) -> float:
    """C1 monotone ramp 0->1 on t in [0, 1] (zero slope at both ends)."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _moonless_mag(band: str, sun_alt: float, elong: float) -> float:
    """Sun-only sky brightness for one physical band (not SWIR_YJH).

    Piecewise in sun altitude, continuous at every boundary by construction:
    the twilight segment interpolates between the day value at sun_alt = 0
    and the night value at the band's night boundary altitude.
    """
    sun_alt = min(max(sun_alt, -90.0), 90.0)
    night = _NIGHT_MAG[band]
    night_alt = _NIGHT_SUN_ALT[band]

    if sun_alt >= 0.0:
        return _day_mag(band, sun_alt, elong)
    if sun_alt <= night_alt:
        return night

    if band in _PATAT_COEFFS:
        # V/R/I: Patat polynomial for zeta in [95, 105.5]; linear bridges
        # (a) sunset -> zeta 95 (poly not fitted there) and (b) zeta 105.5 ->
        # night table at 106.5 (poly ends 0.1-0.2 mag short of the night
        # values; Patat: night level reached at zeta ~ 105.5-106).
        zeta = 90.0 - sun_alt  # sun zenith distance
        if zeta <= _PATAT_ZETA_MIN:
            day0 = _day_mag(band, 0.0, elong)
            m95 = _patat_mag(band, _PATAT_ZETA_MIN)
            return day0 + (m95 - day0) * (zeta - 90.0) / (_PATAT_ZETA_MIN - 90.0)
        if zeta <= _PATAT_ZETA_MAX:
            return _patat_mag(band, zeta)
        m_end = _patat_mag(band, _PATAT_ZETA_MAX)
        frac = (zeta - _PATAT_ZETA_MAX) / (_PATAT_NIGHT_ZETA - _PATAT_ZETA_MAX)
        return m_end + (night - m_end) * frac

    # SWIR (J/H/KS): smooth monotone ramp from the sunset day value down to
    # the night level at the measured boundary altitude (Dome A, Li et al.
    # 2026: J night at sun -9.3 deg, H at -7.4 deg).
    day0 = _day_mag(band, 0.0, elong)
    t = sun_alt / night_alt  # 0 at sunset -> 1 at the night boundary
    return day0 + (night - day0) * _smoothstep(t)


# ---------------------------------------------------------------------------
# Moonlight (Krisciunas & Schaefer 1991)
# ---------------------------------------------------------------------------


def _mag_to_nl(v_mag: float) -> float:
    """V mag/arcsec^2 -> nanoLamberts (Garstang 1989, via K&S 1991 eq. 1)."""
    return 34.08 * math.exp(20.7233 - 0.92104 * v_mag)


def _nl_to_mag(b_nl: float) -> float:
    """nanoLamberts -> V mag/arcsec^2 (inverse of _mag_to_nl)."""
    return (20.7233 - math.log(b_nl / 34.08)) / 0.92104


def _ks_airmass(zd_deg: float) -> float:
    """K&S 1991 eq. 3 optical path X(Z) = (1 - 0.96 sin^2 Z)^-1/2."""
    s = math.sin(math.radians(zd_deg))
    return (1.0 - 0.96 * s * s) ** -0.5


def _moon_excess_nl(
    phase_angle_deg: float,
    separation_deg: float,
    moon_zd_deg: float,
    target_zd_deg: float,
    k_v: float,
) -> float:
    """Scattered-moonlight V-band sky brightness [nanoLamberts].

    Krisciunas & Schaefer 1991 (PASP 103, 1033), eqs. 8-15:
      B_moon = f(rho) * I* * 10^(-0.4 k X(Zm)) * [1 - 10^(-0.4 k X(Z))]
    with I* the lunar illuminance vs phase angle alpha (eq. 20, including
    the alpha^4 opposition term), f(rho) the Rayleigh + Mie scattering
    function of moon-target separation rho (eq. 21), and X the optical path
    (eq. 3). Model accuracy 8-23%; reproduces their Table 2 deltas.
    """
    if moon_zd_deg >= 90.0:  # moon below the horizon: no scattered light
        return 0.0
    alpha = abs(phase_angle_deg)
    # Lunar illuminance vs phase (K&S eq. 20).
    i_star = 10.0 ** (-0.4 * (3.84 + 0.026 * alpha + 4e-9 * alpha**4))
    # Scattering function: Rayleigh (1.06 + cos^2 rho) + Mie 10^(6.15-rho/40).
    cos_rho = math.cos(math.radians(separation_deg))
    f_rho = 10.0**5.36 * (1.06 + cos_rho * cos_rho) + 10.0 ** (
        6.15 - separation_deg / 40.0
    )
    return (
        f_rho
        * i_star
        * 10.0 ** (-0.4 * k_v * _ks_airmass(moon_zd_deg))
        * (1.0 - 10.0 ** (-0.4 * k_v * _ks_airmass(target_zd_deg)))
    )


def _moon_j_mag(mag_j: float, moon: Mapping[str, float]) -> float:
    """Lunar contribution to the J sky: +1.0 mag only if gibbous and close.

    Y/J backgrounds rise x2.5 (~1 mag) only for a gibbous-to-full moon
    (phase angle < 90 deg) within 30 deg of the line of sight; no effect in
    H/Ks (Sullivan & Simcoe 2012, PASP). Implemented as a fixed flux
    increment of 1.5x the J *night* sky so it is exactly +1.0 mag at night
    and fades to negligible against the bright twilight/day sky.
    """
    if float(moon["phase_angle_deg"]) < 90.0 and float(moon["separation_deg"]) < 30.0:
        f = 10.0 ** (-0.4 * mag_j) + (10.0**0.4 - 1.0) * 10.0 ** (
            -0.4 * _NIGHT_MAG["J"]
        )
        return -2.5 * math.log10(f)
    return mag_j


def _normalize_band(band_name: str) -> str:
    """Uppercase and validate a band name."""
    band = band_name.strip().upper()
    if band == "YJH":
        band = "SWIR_YJH"
    if band not in _BAND_NAMES:
        raise ValueError(
            f"unknown band {band_name!r}; expected one of {_BAND_NAMES}"
        )
    return band


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------


def sky_brightness(
    band_name: str,
    sun_alt_deg: float,
    elongation_deg: float = 90.0,
    moon: Mapping[str, float] | None = None,
    aerosol_delta_mag: float = 0.0,
) -> float:
    """Sky surface brightness [mag/arcsec^2, Vega] for one band.

    Continuous in ``sun_alt_deg`` across the night / twilight / day regime
    boundaries for every band (see module docstring for the per-regime
    models and sources).

    Args:
        band_name: One of V, R, I, J, H, KS, SWIR_YJH (case-insensitive).
        sun_alt_deg: Sun altitude above the horizon [deg]; negative below.
        elongation_deg: Sun-target great-circle angle [deg]; only affects
            the daytime (and, via its sunset endpoint, twilight) sky.
            Clamped to [12, 165] (solar exclusion / antisolar limit).
        moon: Optional mapping with keys ``phase_angle_deg`` (0 = full moon),
            ``separation_deg`` (moon-target), ``moon_zd_deg``,
            ``target_zd_deg`` (zenith distances), and optional ``k_v``
            (V extinction, mag/airmass; default 0.172, K&S 1991). None =
            no moon contribution.
        aerosol_delta_mag: Day-to-day aerosol-loading variability knob
            [mag]; positive = brighter sky (heavier aerosol scattering,
            +-1 mag typical; Thomas & Cobb 2019). Applied in all regimes.

    Returns:
        Sky surface brightness in mag/arcsec^2 (Vega system).
    """
    band = _normalize_band(band_name)

    if band == "SWIR_YJH":
        m_j = sky_brightness("J", sun_alt_deg, elongation_deg, moon,
                             aerosol_delta_mag)
        m_h = sky_brightness("H", sun_alt_deg, elongation_deg, moon,
                             aerosol_delta_mag)
        return _yjh_from_jh(m_j, m_h)

    mag = _moonless_mag(band, sun_alt_deg, elongation_deg) - aerosol_delta_mag

    if moon is None or band in ("H", "KS"):
        return mag

    if band == "J":
        return _moon_j_mag(mag, moon)

    # V/R/I: full K&S model, added in flux (nanoLamberts) so it fades
    # naturally as the sun-driven sky brightens; the brightening delta is
    # computed in the V system and applied unchanged to R/I (K&S is a V-band
    # model; scattered moonlight is near-solar in color).
    v_moonless = (
        mag if band == "V"
        else _moonless_mag("V", sun_alt_deg, elongation_deg) - aerosol_delta_mag
    )
    excess_nl = _moon_excess_nl(
        float(moon["phase_angle_deg"]),
        float(moon["separation_deg"]),
        float(moon["moon_zd_deg"]),
        float(moon["target_zd_deg"]),
        float(moon.get("k_v", 0.172)),
    )
    if excess_nl <= 0.0:
        return mag
    v_total = _nl_to_mag(_mag_to_nl(v_moonless) + excess_nl)
    delta_v = v_moonless - v_total  # > 0: moon brightens the sky
    return mag - delta_v


def sky_summary(
    band_name: str,
    sun_alt_deg: float,
    elongation_deg: float = 90.0,
    moon: Mapping[str, float] | None = None,
    aerosol_delta_mag: float = 0.0,
) -> dict:
    """Sky brightness plus regime label and inputs, for frame metadata.

    Args:
        Same as :func:`sky_brightness`.

    Returns:
        Dict with keys: ``band``, ``regime`` ("day" / "twilight" / "night"),
        ``mag_arcsec2``, ``sun_alt_deg``, ``elongation_deg`` (clamped value
        actually used), ``moon`` (input mapping as a plain dict, or None),
        ``aerosol_delta_mag``, ``night_mag_arcsec2`` and
        ``night_sun_alt_deg`` (the band's night level and boundary).
    """
    band = _normalize_band(band_name)
    # SWIR_YJH inherits J's (later) night boundary; its night level is the
    # J+H flux-weighted composite.
    if band == "SWIR_YJH":
        night_alt = _NIGHT_SUN_ALT["J"]
        night_mag = _yjh_from_jh(_NIGHT_MAG["J"], _NIGHT_MAG["H"])
    else:
        night_alt = _NIGHT_SUN_ALT[band]
        night_mag = _NIGHT_MAG[band]

    if sun_alt_deg >= 0.0:
        regime = "day"
    elif sun_alt_deg <= night_alt:
        regime = "night"
    else:
        regime = "twilight"

    return {
        "band": band,
        "regime": regime,
        "mag_arcsec2": sky_brightness(
            band, sun_alt_deg, elongation_deg, moon, aerosol_delta_mag
        ),
        "sun_alt_deg": float(sun_alt_deg),
        "elongation_deg": float(
            min(max(elongation_deg, _ELONG_MIN), _ELONG_MAX)
        ),
        "moon": dict(moon) if moon is not None else None,
        "aerosol_delta_mag": float(aerosol_delta_mag),
        "night_mag_arcsec2": night_mag,
        "night_sun_alt_deg": night_alt,
    }
