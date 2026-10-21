"""Tests for the sun-driven sky surface brightness model (``sdasim.sky``).

Validation anchors come from the literature the model is built on:
Patat 2003 / Patat et al. 2006 (night + twilight V/R/I), Cuby et al. 2000
(night J/H/Ks), Li et al. 2026 Dome A (SWIR twilight boundaries), Jim et al.
2012 + IRTF (daytime J/H/K vs elongation), Zimmer et al. 2020 (daytime
visible), and Krisciunas & Schaefer 1991 Table 2 (moonlight deltas).
"""

from __future__ import annotations

import numpy as np
import pytest

from sdasim.sky import (
    elongation_deg,
    sky_brightness,
    sky_summary,
    sun_altaz,
    sun_radec,
)

ALL_BANDS = ("V", "R", "I", "J", "H", "KS", "SWIR_YJH")

# Deep-night sun altitude (below every band's night boundary).
NIGHT_ALT = -40.0
# Daytime reference sun altitude used in the day-level tests (the model's
# sun-altitude term is ~0 near 15-20 deg, so table anchors read directly).
DAY_ALT = 20.0


# ---------------------------------------------------------------------------
# Night levels
# ---------------------------------------------------------------------------


def test_night_levels():
    """Dark-site night sky: Patat 2003 (V/R/I) + Cuby 2000 (J/H/Ks)."""
    assert sky_brightness("V", NIGHT_ALT) == pytest.approx(21.6, abs=0.1)
    assert sky_brightness("R", NIGHT_ALT) == pytest.approx(20.9, abs=0.1)
    assert sky_brightness("I", NIGHT_ALT) == pytest.approx(19.7, abs=0.1)
    assert sky_brightness("J", NIGHT_ALT) == pytest.approx(16.5, abs=0.1)
    assert sky_brightness("H", NIGHT_ALT) == pytest.approx(14.4, abs=0.1)
    assert sky_brightness("KS", NIGHT_ALT) == pytest.approx(13.0, abs=0.1)
    # Flux-weighted J+H composite for the 1.0-1.7 um longpass band (~15.3).
    assert sky_brightness("SWIR_YJH", NIGHT_ALT) == pytest.approx(15.3, abs=0.15)


def test_night_independent_of_elongation():
    """At night the model must not depend on solar elongation."""
    for elong in (12.0, 45.0, 90.0, 165.0):
        assert sky_brightness("V", NIGHT_ALT, elong) == pytest.approx(21.6, abs=0.01)
        assert sky_brightness("J", NIGHT_ALT, elong) == pytest.approx(16.5, abs=0.01)


# ---------------------------------------------------------------------------
# Twilight (Patat et al. 2006 polynomial; Dome A SWIR boundaries)
# ---------------------------------------------------------------------------


def test_patat_twilight_anchors():
    """VALIDATION ANCHOR 1: V at sun -6 = 13.3 +/- 0.4; at -12 = 19.7 +/- 0.4."""
    assert sky_brightness("V", -6.0) == pytest.approx(13.3, abs=0.4)
    assert sky_brightness("V", -12.0) == pytest.approx(19.7, abs=0.4)
    # Additional published waypoints from the same fit (Patat et al. 2006):
    # sun -9: V = 17.0; civil end: R = 12.9, I = 12.3.
    assert sky_brightness("V", -9.0) == pytest.approx(17.0, abs=0.4)
    assert sky_brightness("R", -6.0) == pytest.approx(12.9, abs=0.4)
    assert sky_brightness("I", -6.0) == pytest.approx(12.3, abs=0.4)


def test_swir_darkens_earlier_than_visible():
    """SWIR reaches night level early: J by sun -9.3, H by -7.4 (Dome A)."""
    assert sky_brightness("J", -9.3) == pytest.approx(16.5, abs=0.05)
    assert sky_brightness("H", -7.4) == pytest.approx(14.4, abs=0.05)
    # ... while V is still ~4-5 mag brighter than its night level there.
    assert sky_brightness("V", -9.3) < 21.6 - 3.0
    # And in mid-SWIR-twilight the sky is genuinely between day and night.
    m = sky_brightness("J", -5.0)
    assert sky_brightness("J", 0.0) < m < 16.5


# ---------------------------------------------------------------------------
# Daytime (elongation-dependent)
# ---------------------------------------------------------------------------


def test_day_levels():
    """VALIDATION ANCHOR 2: IRTF 6-6.5 at 45 deg; visible 3-5 far from sun."""
    assert 5.5 <= sky_brightness("J", DAY_ALT, 45.0) <= 6.6
    assert 5.8 <= sky_brightness("H", DAY_ALT, 45.0) <= 6.6
    assert 3.0 <= sky_brightness("V", DAY_ALT, 90.0) <= 5.0
    # K (Jim et al. 2012 anchor at 90 deg elongation: ~8.3).
    assert sky_brightness("KS", DAY_ALT, 90.0) == pytest.approx(8.3, abs=0.4)
    # Sky darkens away from the sun; blue sky: V < R < I in brightness
    # (i.e. increasing mag with wavelength in the visible).
    v, r, i = (sky_brightness(b, DAY_ALT, 90.0) for b in ("V", "R", "I"))
    assert v < r < i
    # H sits between J and K.
    j, h, k = (sky_brightness(b, DAY_ALT, 90.0) for b in ("J", "H", "KS"))
    assert j < h < k


def test_day_elongation_monotone_and_clamped():
    """Sky brightens toward the sun; elongation clamps to [12, 165]."""
    mags = [sky_brightness("J", DAY_ALT, e) for e in (12, 30, 60, 90, 120)]
    assert all(m1 <= m2 + 1e-12 for m1, m2 in zip(mags, mags[1:]))
    assert sky_brightness("J", DAY_ALT, 5.0) == pytest.approx(
        sky_brightness("J", DAY_ALT, 12.0), abs=1e-9
    )
    assert sky_brightness("J", DAY_ALT, 179.0) == pytest.approx(
        sky_brightness("J", DAY_ALT, 165.0), abs=1e-9
    )


def test_day_sun_altitude_term():
    """~0.3 mag brighter (smaller mag) at sun_alt 60 than at sun_alt 15."""
    d = sky_brightness("J", 15.0, 90.0) - sky_brightness("J", 60.0, 90.0)
    assert d == pytest.approx(0.3, abs=0.05)


def test_night_to_day_swing():
    """VALIDATION ANCHOR 3: J swing ~10-12 mag; V swing ~17-18 mag."""
    j_swing = sky_brightness("J", NIGHT_ALT, 90.0) - sky_brightness("J", 30.0, 90.0)
    v_swing = sky_brightness("V", NIGHT_ALT, 90.0) - sky_brightness("V", 30.0, 90.0)
    assert 9.5 <= j_swing <= 12.5
    assert 16.5 <= v_swing <= 18.5
    # Worst case (brightest daytime sky at the solar-exclusion edge).
    j_max = sky_brightness("J", NIGHT_ALT, 12.0) - sky_brightness("J", 60.0, 12.0)
    assert j_max <= 13.0


# ---------------------------------------------------------------------------
# Continuity (VALIDATION ANCHOR 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("band", ALL_BANDS)
def test_continuity_in_sun_alt(band):
    """No jumps > 0.5 mag anywhere on a fine sun-altitude sweep."""
    alts = np.arange(-30.0, 65.0 + 1e-9, 0.1)
    mags = np.array([sky_brightness(band, a, 90.0) for a in alts])
    assert np.all(np.isfinite(mags))
    steps = np.abs(np.diff(mags))
    assert steps.max() < 0.5, (
        f"{band}: {steps.max():.3f} mag jump at "
        f"sun_alt ~ {alts[int(steps.argmax())]:.1f}"
    )
    # The sweep spans night -> day, so it must brighten overall.
    assert mags[0] > mags[-1]


@pytest.mark.parametrize("band", ALL_BANDS)
def test_continuity_in_elongation(band):
    """No jumps in the elongation direction either (daytime)."""
    elongs = np.arange(12.0, 165.0 + 1e-9, 0.5)
    mags = np.array([sky_brightness(band, DAY_ALT, e) for e in elongs])
    assert np.all(np.isfinite(mags))
    assert np.abs(np.diff(mags)).max() < 0.5


# ---------------------------------------------------------------------------
# Moon (Krisciunas & Schaefer 1991)
# ---------------------------------------------------------------------------


def _moon(alpha, rho, zm=60.0, z=0.0, k_v=0.172):
    return {
        "phase_angle_deg": alpha,
        "separation_deg": rho,
        "moon_zd_deg": zm,
        "target_zd_deg": z,
        "k_v": k_v,
    }


def test_moon_none_is_noop():
    assert sky_brightness("V", NIGHT_ALT, moon=None) == pytest.approx(21.6, abs=0.01)


def test_moon_ks1991_table2():
    """Reproduce K&S 1991 Table 2 deltas (k=0.172, moon ZD 60, dark V 21.59).

    Table 2: quarter moon (alpha 90), rho 90 -> DeltaV = -0.71; gibbous
    (alpha 30), rho 30 -> DeltaV = -2.87.
    """
    dark = sky_brightness("V", NIGHT_ALT)
    dv_quarter = dark - sky_brightness("V", NIGHT_ALT, moon=_moon(90.0, 90.0))
    dv_gibbous = dark - sky_brightness("V", NIGHT_ALT, moon=_moon(30.0, 30.0))
    assert dv_quarter == pytest.approx(0.71, abs=0.2)
    assert dv_gibbous == pytest.approx(2.87, abs=0.3)


def test_moon_same_delta_in_r_and_i():
    """R/I take the same magnitude brightening as V."""
    m = _moon(30.0, 30.0)
    dv = sky_brightness("V", NIGHT_ALT) - sky_brightness("V", NIGHT_ALT, moon=m)
    dr = sky_brightness("R", NIGHT_ALT) - sky_brightness("R", NIGHT_ALT, moon=m)
    di = sky_brightness("I", NIGHT_ALT) - sky_brightness("I", NIGHT_ALT, moon=m)
    assert dr == pytest.approx(dv, abs=0.01)
    assert di == pytest.approx(dv, abs=0.01)


def test_moon_nir_rules():
    """J: +1 mag only if gibbous AND within 30 deg; H/KS never affected."""
    gibbous_close = _moon(30.0, 20.0)
    assert sky_brightness("J", NIGHT_ALT, moon=gibbous_close) == pytest.approx(
        15.5, abs=0.1
    )
    # Gibbous but far, and close but crescent: no J effect.
    assert sky_brightness("J", NIGHT_ALT, moon=_moon(30.0, 60.0)) == pytest.approx(
        16.5, abs=0.01
    )
    assert sky_brightness("J", NIGHT_ALT, moon=_moon(120.0, 20.0)) == pytest.approx(
        16.5, abs=0.01
    )
    # H and KS unchanged even for a full moon nearby.
    for band, night in (("H", 14.4), ("KS", 13.0)):
        assert sky_brightness(band, NIGHT_ALT, moon=_moon(5.0, 20.0)) == pytest.approx(
            night, abs=0.01
        )
    # SWIR_YJH brightens (through its J component) but by less than 1 mag.
    yjh_dark = sky_brightness("SWIR_YJH", NIGHT_ALT)
    yjh_moon = sky_brightness("SWIR_YJH", NIGHT_ALT, moon=gibbous_close)
    assert 0.0 < yjh_dark - yjh_moon < 1.0


def test_moon_below_horizon_is_noop():
    m = _moon(30.0, 30.0, zm=95.0)
    assert sky_brightness("V", NIGHT_ALT, moon=m) == pytest.approx(21.6, abs=0.01)


def test_moon_negligible_in_daytime():
    """Flux-additive moon term must vanish against the daytime sky."""
    m = _moon(30.0, 30.0)
    day = sky_brightness("V", 30.0, 90.0)
    assert day - sky_brightness("V", 30.0, 90.0, moon=m) < 0.02


# ---------------------------------------------------------------------------
# Aerosol knob / band handling / summary
# ---------------------------------------------------------------------------


def test_aerosol_delta():
    """Positive aerosol delta brightens the sky by that many magnitudes."""
    base = sky_brightness("J", DAY_ALT, 90.0)
    assert base - sky_brightness("J", DAY_ALT, 90.0, aerosol_delta_mag=1.0) == (
        pytest.approx(1.0, abs=1e-9)
    )


def test_band_name_normalization():
    assert sky_brightness("v", NIGHT_ALT) == sky_brightness("V", NIGHT_ALT)
    assert sky_brightness("Ks", NIGHT_ALT) == sky_brightness("KS", NIGHT_ALT)
    assert sky_brightness("yjh", NIGHT_ALT) == sky_brightness("SWIR_YJH", NIGHT_ALT)
    with pytest.raises(ValueError, match="unknown band"):
        sky_brightness("Q", NIGHT_ALT)


def test_sky_summary():
    s = sky_summary("V", NIGHT_ALT, 90.0)
    assert s["regime"] == "night"
    assert s["band"] == "V"
    assert s["mag_arcsec2"] == pytest.approx(21.6, abs=0.1)
    assert s["night_sun_alt_deg"] < 0.0
    assert sky_summary("V", -6.0)["regime"] == "twilight"
    assert sky_summary("V", 30.0)["regime"] == "day"
    # SWIR bands hit "night" at their earlier boundaries.
    assert sky_summary("H", -8.0)["regime"] == "night"
    assert sky_summary("J", -8.0)["regime"] == "twilight"
    # Summary magnitude always matches sky_brightness.
    m = _moon(30.0, 20.0)
    s = sky_summary("SWIR_YJH", -3.0, 45.0, moon=m, aerosol_delta_mag=0.5)
    assert s["mag_arcsec2"] == pytest.approx(
        sky_brightness("SWIR_YJH", -3.0, 45.0, moon=m, aerosol_delta_mag=0.5)
    )
    assert s["moon"] == m
    assert s["elongation_deg"] == 45.0


# ---------------------------------------------------------------------------
# Sun ephemerides (satkit) — VALIDATION ANCHOR 5
# ---------------------------------------------------------------------------


def test_sun_altaz_haleakala():
    """Haleakala 2026-09-01: high sun near local solar noon, deep night 10 UTC."""
    pytest.importorskip("satkit")
    # ~local solar noon (22:24 UTC at lon -156.28); sun nearly overhead
    # (early September, latitude 20.8 N ~ solar declination + a few deg).
    alt, az = sun_altaz(20.82, -156.28, 3055.0, "2026-09-01T22:24:00Z")
    assert alt > 60.0
    assert 0.0 <= az < 360.0
    # Middle of the local night: sun far below the horizon.
    alt_night, _ = sun_altaz(20.82, -156.28, 3055.0, "2026-09-01T10:00:00Z")
    assert alt_night < -30.0


def test_sun_radec_september():
    """Sun RA/Dec on 2026-09-01: RA ~ 160.7 deg, Dec ~ +8.1 deg."""
    pytest.importorskip("satkit")
    from datetime import datetime

    ra, dec = sun_radec("2026-09-01T22:24:00Z")
    assert ra == pytest.approx(160.7, abs=2.0)
    assert dec == pytest.approx(8.1, abs=1.5)
    # datetime input (naive = UTC) agrees with the ISO-string path.
    ra2, dec2 = sun_radec(datetime(2026, 9, 1, 22, 24, 0))
    assert ra2 == pytest.approx(ra, abs=1e-9)
    assert dec2 == pytest.approx(dec, abs=1e-9)


def test_elongation_deg_basics():
    assert elongation_deg((0.0, 0.0), (90.0, 0.0)) == pytest.approx(90.0)
    assert elongation_deg((10.0, 20.0), (10.0, 20.0)) == pytest.approx(0.0, abs=1e-9)
    assert elongation_deg((0.0, 0.0), (180.0, 0.0)) == pytest.approx(180.0)
    # Pole-to-equator quarter circle, RA-independent at the pole.
    assert elongation_deg((123.0, 90.0), (0.0, 0.0)) == pytest.approx(90.0)
    # Small-angle stability.
    assert elongation_deg((0.0, 0.0), (0.001, 0.0)) == pytest.approx(0.001, rel=1e-6)


def test_elongation_with_real_sun():
    """Antisolar point sits at elongation 180; 90-deg RA offset near 90."""
    pytest.importorskip("satkit")
    s = sun_radec("2026-09-01T22:24:00Z")
    anti = ((s[0] + 180.0) % 360.0, -s[1])
    assert elongation_deg(s, anti) == pytest.approx(180.0, abs=1e-6)
    target = ((s[0] + 90.0) % 360.0, 0.0)
    assert elongation_deg(s, target) == pytest.approx(90.0, abs=10.0)
