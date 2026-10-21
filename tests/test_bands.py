"""Tests for band & coating radiometry (sdasim.bands).

Validation anchors (sources in the module docstring):

1. v1 zeropoint anchor: the v1 YAML zeropoints (am_f2_035/070/100:
   23.57 / 25.14 / 25.92, mag giving 1 e-/s) must be reproduced by the
   compute_zeropoint chain for the v1 recipe - f/2 prime focus, bare
   LPBF+SPDT aluminum sigma = 9 nm, one reflection, band V - and their
   differences are pure area scaling (v1 built them that way).
2. TIS: sigma = 10 nm scatters 5.22% at 550 nm and 0.58% at 1650 nm
   (Bennett & Porteus 1961 smooth-surface TIS; 9x SWIR advantage).
3. Coatings: gold ~0.79 in V but >=0.97 in H (SWIR-only coating);
   protected silver >=0.97 from V through Ks (ARIEL/Gemini class).
4. Solar magnitudes: V = -26.74 exactly (v1 SUN_MV convention); J/H from
   measured solar colors (V-J) = 1.198, (V-H) = 1.484 (Casagrande,
   Ramirez, Melendez & Asplund 2012, ApJ 761, 16).
"""

from __future__ import annotations

import math

import pytest

from sdasim.bands import (
    BANDS,
    COATINGS,
    DEFAULT_AIRMASS,
    SUN_MV,
    Band,
    atm_transmission,
    compute_zeropoint,
    describe,
    specular_throughput,
    tis_fraction,
)

# ---------------------------------------------------------------------------
# v1 anchor recipe (documented choice).
#
# The v1 sensors are UNFILTERED broadband silicon CMOS calibrated to V
# magnitudes, so band "V" carries the effective 400-1000 nm collection
# width (600 nm). The chosen effective QE is
#
#   V1_QE = 0.68  (band-averaged BSI silicon QE ~0.72 x prime-focus
#                  corrector/window transmission ~0.95)
#
# and the atmosphere is the band default, T_atm = 10^(-0.4*0.14*1.3) =
# 0.846 (k_V = 0.14 mag/airmass, airmass 1.3), so qe * T_atm = 0.575 -
# consistent with the v1 YAML's "atm 0.7"-times-QE bookkeeping. With the
# bare-Al sigma = 9 nm specular throughput (0.877) this lands the 0.7 m
# zeropoint at 25.135 vs the v1 YAML 25.14.
# ---------------------------------------------------------------------------
V1_QE = 0.68

# v1 per-aperture central obscurations (fixed ~100/125/150 mm camera
# package at prime focus: am_f2_035/070/100.yaml).
V1_APERTURES = {
    "035": (0.35, 0.29, 23.57),
    "070": (0.70, 0.18, 25.14),
    "100": (1.00, 0.15, 25.92),
}


def _v1_area(diameter_m: float, obscuration: float) -> float:
    return math.pi / 4.0 * diameter_m**2 * (1.0 - obscuration**2)


class TestBandRegistry:
    """The BANDS registry: contents and internal consistency."""

    def test_required_bands_present(self):
        for name in ("V", "R", "I", "J", "H", "KS", "SWIR_YJH"):
            assert name in BANDS
            assert isinstance(BANDS[name], Band)
            assert BANDS[name].name == name

    def test_photon_zp_v_classic_value(self):
        """V photon ZP is the classic ~1000 ph/s/cm^2/A = 1.005e11 /m^2/um."""
        assert BANDS["V"].photon_zp == pytest.approx(1.005e11, rel=0.01)

    def test_photon_zp_consistent_with_f0(self):
        """photon_zp = f0 * lambda_c / (h c) for every band."""
        hc = 6.62607015e-34 * 2.99792458e8
        for band in BANDS.values():
            expected = band.f0_w_m2_um * band.center_nm * 1e-9 / hc
            assert band.photon_zp == pytest.approx(expected, rel=1e-12)

    def test_vega_f0_values(self):
        """Vega zero points match Bessell 1998 / Cohen 2003 (2MASS)."""
        assert BANDS["V"].f0_w_m2_um == pytest.approx(3.63e-8, rel=1e-6)
        assert BANDS["J"].f0_w_m2_um == pytest.approx(3.13e-9, rel=0.01)
        assert BANDS["H"].f0_w_m2_um == pytest.approx(1.13e-9, rel=0.01)
        assert BANDS["KS"].f0_w_m2_um == pytest.approx(4.28e-10, rel=0.01)
        # SWIR_YJH is band-averaged Vega flux over 1.0-1.7 um: between the
        # H and J zero points.
        f0_yjh = BANDS["SWIR_YJH"].f0_w_m2_um
        assert BANDS["H"].f0_w_m2_um < f0_yjh < BANDS["J"].f0_w_m2_um

    def test_band_edges(self):
        swir = BANDS["SWIR_YJH"]
        assert swir.lambda_min_nm == pytest.approx(1000.0)
        assert swir.lambda_max_nm == pytest.approx(1700.0)

    # --- validation anchor 4: solar magnitudes -----------------------------

    def test_solar_mag_v_exact(self):
        """V = -26.74 exactly (v1 SUN_MV convention)."""
        assert BANDS["V"].solar_mag == -26.74
        assert SUN_MV == -26.74

    def test_solar_mag_j_h_in_range(self):
        """J/H solar mags from cited colors (Casagrande et al. 2012:
        (V-J) = 1.198, (V-H) = 1.484) land in [-28.5, -27.0]."""
        assert -28.5 <= BANDS["J"].solar_mag <= -27.0
        assert -28.5 <= BANDS["H"].solar_mag <= -27.0
        assert BANDS["J"].solar_mag == pytest.approx(-26.74 - 1.198, abs=1e-9)
        assert BANDS["H"].solar_mag == pytest.approx(-26.74 - 1.484, abs=1e-9)

    def test_solar_mag_monotonic_redward(self):
        """The Sun is redder than Vega: solar mags brighten V -> Ks."""
        seq = [BANDS[n].solar_mag for n in ("V", "R", "I", "J", "H", "KS")]
        assert all(a > b for a, b in zip(seq, seq[1:]))

    def test_solar_mag_swir_yjh_flux_weighted(self):
        """SWIR_YJH solar mag lies between the J and H values (~-27.95)."""
        yjh = BANDS["SWIR_YJH"].solar_mag
        assert BANDS["H"].solar_mag < yjh < BANDS["J"].solar_mag
        assert yjh == pytest.approx(-27.95, abs=0.1)

    # --- sky fallbacks -----------------------------------------------------

    def test_sky_night_key_numbers(self):
        """Night zenith sky: Patat 2003 (V/R/I), Cuby 2000 (J/H/Ks)."""
        assert BANDS["V"].sky_night == pytest.approx(21.6)
        assert BANDS["R"].sky_night == pytest.approx(20.9)
        assert BANDS["I"].sky_night == pytest.approx(19.7)
        assert BANDS["J"].sky_night == pytest.approx(16.5)
        assert BANDS["H"].sky_night == pytest.approx(14.4)
        assert BANDS["KS"].sky_night == pytest.approx(13.0)

    def test_sky_day_key_numbers(self):
        """Day (60-90 deg elongation): Zimmer 2020 (V), Jim 2012 (J/K)."""
        assert BANDS["V"].sky_day_ref == pytest.approx(4.0)
        assert BANDS["J"].sky_day_ref == pytest.approx(5.5)
        assert BANDS["KS"].sky_day_ref == pytest.approx(7.5)
        # H tracks between J and K (IRTF: 6-6.5 at 45 deg elongation).
        assert BANDS["J"].sky_day_ref < BANDS["H"].sky_day_ref < BANDS["KS"].sky_day_ref

    def test_sky_day_brighter_than_night(self):
        for band in BANDS.values():
            assert band.sky_day_ref < band.sky_night  # smaller mag = brighter

    def test_extinction_positive_and_small(self):
        for band in BANDS.values():
            assert 0.0 < band.extinction_mag_airmass < 0.5


class TestTisFraction:
    """Validation anchor 2: the (4 pi sigma / lambda)^2 scatter law."""

    def test_sigma10_at_550(self):
        """sigma = 10 nm -> 5.22% at 550 nm (+/-2% rel)."""
        assert tis_fraction(10.0, 550.0) == pytest.approx(0.0522, rel=0.02)

    def test_sigma10_at_1650(self):
        """sigma = 10 nm -> 0.58% at 1650 nm (+/-2% rel)."""
        assert tis_fraction(10.0, 1650.0) == pytest.approx(0.0058, rel=0.02)

    def test_v_to_h_ratio_is_nine(self):
        """(1650/550)^2 = 9.0: exactly 9x less scatter in H than V."""
        ratio = tis_fraction(10.0, 550.0) / tis_fraction(10.0, 1650.0)
        assert ratio == pytest.approx(9.0, rel=1e-9)

    def test_scales_as_sigma_squared(self):
        assert tis_fraction(20.0, 550.0) == pytest.approx(
            4.0 * tis_fraction(10.0, 550.0), rel=1e-12
        )

    def test_band_evaluates_at_center(self):
        assert tis_fraction(10.0, "V") == pytest.approx(
            tis_fraction(10.0, BANDS["V"].center_nm), rel=1e-12
        )
        assert tis_fraction(10.0, BANDS["H"]) == pytest.approx(
            tis_fraction(10.0, 1662.0), rel=1e-12
        )

    def test_swir_yjh_band_average(self):
        """SWIR_YJH averages 1/lambda^2 over 1.0-1.7 um:
        <1/lam^2> = 1/(lam_min*lam_max), the TIS at the geometric-mean
        wavelength sqrt(1000*1700) = 1304 nm."""
        tis = tis_fraction(10.0, "SWIR_YJH")
        expected = (4.0 * math.pi * 10.0) ** 2 / (1000.0 * 1700.0)
        assert tis == pytest.approx(expected, rel=1e-12)
        assert tis == pytest.approx(0.00929, rel=0.02)
        # Average sits strictly between the band-edge values.
        assert tis_fraction(10.0, 1700.0) < tis < tis_fraction(10.0, 1000.0)

    def test_unknown_band_raises(self):
        with pytest.raises(KeyError, match="unknown band"):
            tis_fraction(10.0, "Q")


class TestCoatings:
    """Validation anchor 3: coating reflectivity table."""

    def test_all_coatings_cover_all_bands(self):
        assert set(COATINGS) == {"bare_al", "protected_al", "protected_ag", "gold"}
        for refl in COATINGS.values():
            assert set(refl) == set(BANDS)
            for value in refl.values():
                assert 0.0 < value <= 1.0

    def test_gold_v_and_h(self):
        """Gold: ~0.79 in V (poor) but >=0.97 in H (SWIR-only coating)."""
        assert COATINGS["gold"]["V"] == pytest.approx(0.79, abs=0.02)
        assert COATINGS["gold"]["H"] >= 0.97

    def test_protected_ag_broadband(self):
        """Protected silver >= 0.97 from V through Ks (ARIEL/Gemini)."""
        for name in ("V", "R", "I", "J", "H", "KS"):
            assert COATINGS["protected_ag"][name] >= 0.97

    def test_bare_al_key_values(self):
        """Bare Al: V 0.915, I dip ~0.86, H 0.974 (recon key numbers)."""
        assert COATINGS["bare_al"]["V"] == pytest.approx(0.915, abs=0.01)
        assert COATINGS["bare_al"]["I"] == pytest.approx(0.86, abs=0.01)
        assert COATINGS["bare_al"]["H"] == pytest.approx(0.974, abs=0.005)

    def test_swir_yjh_interpolation(self):
        """SWIR_YJH reflectivity lies between each coating's J and H."""
        for refl in COATINGS.values():
            lo, hi = sorted((refl["J"], refl["H"]))
            assert lo <= refl["SWIR_YJH"] <= hi


class TestSpecularThroughput:
    def test_formula(self):
        """R_spec = R_coat * exp(-(4 pi sigma / lambda)^2)."""
        expected = 0.915 * math.exp(-((4.0 * math.pi * 9.0 / 550.0) ** 2))
        assert specular_throughput("bare_al", 9.0, "V", 1) == pytest.approx(
            expected, rel=1e-12
        )

    def test_n_reflections_compounds(self):
        one = specular_throughput("protected_ag", 2.0, "V", 1)
        three = specular_throughput("protected_ag", 2.0, "V", 3)
        assert three == pytest.approx(one**3, rel=1e-12)

    def test_zero_roughness_is_pure_reflectivity(self):
        assert specular_throughput("gold", 0.0, "H", 1) == pytest.approx(
            COATINGS["gold"]["H"], rel=1e-12
        )

    def test_bare_al_swir_advantage(self):
        """A rough bare-Al mirror is a better SWIR optic than visible:
        higher reflectivity AND 9x less scatter."""
        v = specular_throughput("bare_al", 10.0, "V", 1)
        swir = specular_throughput("bare_al", 10.0, "SWIR_YJH", 1)
        assert swir > v
        assert swir > 0.95  # near-premium in SWIR
        assert v < 0.88  # visibly lossy in V

    def test_unknown_coating_raises(self):
        with pytest.raises(KeyError, match="unknown coating"):
            specular_throughput("unobtainium", 5.0, "V", 1)


class TestZeropointAnchor:
    """Validation anchor 1: reproduce the v1 YAML zeropoints.

    v1 recipe: f/2 prime focus, bare LPBF+SPDT aluminum sigma = 9 nm, one
    reflection (corrector losses folded into qe = V1_QE), band V, default
    atmosphere (k_V = 0.14, airmass 1.3). Areas use each aperture's v1
    obscuration (0.29 / 0.18 / 0.15 - fixed camera package). The v1 YAMLs
    scaled the 0.7 m anchor by pure area, so the zeropoint DIFFERENCES
    must match to +/-0.03 mag.
    """

    def _zeropoint(self, key: str) -> float:
        diameter, obscuration, _ = V1_APERTURES[key]
        throughput = specular_throughput("bare_al", 9.0, "V", 1)
        return compute_zeropoint(
            _v1_area(diameter, obscuration), throughput, V1_QE, "V"
        )

    def test_070_anchor(self):
        """0.7 m f/2, eps 0.18 -> ZP 25.14 +/- 0.15 (v1 am_f2_070.yaml)."""
        assert self._zeropoint("070") == pytest.approx(25.14, abs=0.15)

    def test_035_scaling(self):
        """0.35 m same recipe (eps 0.29) -> 23.57 +/- 0.15."""
        assert self._zeropoint("035") == pytest.approx(23.57, abs=0.15)

    def test_100_scaling(self):
        """1.0 m same recipe (eps 0.15) -> 25.92 +/- 0.15."""
        assert self._zeropoint("100") == pytest.approx(25.92, abs=0.15)

    def test_ratios_are_pure_area_scaling(self):
        """ZP differences vs the v1 YAML values match to +/-0.03 mag."""
        zp = {k: self._zeropoint(k) for k in V1_APERTURES}
        v1 = {k: V1_APERTURES[k][2] for k in V1_APERTURES}
        assert (zp["070"] - zp["035"]) == pytest.approx(v1["070"] - v1["035"], abs=0.03)
        assert (zp["100"] - zp["070"]) == pytest.approx(v1["100"] - v1["070"], abs=0.03)

    def test_differences_exactly_area(self):
        """With everything else fixed, dZP = 2.5 log10(area ratio)."""
        zp070, zp035 = self._zeropoint("070"), self._zeropoint("035")
        a070 = _v1_area(*V1_APERTURES["070"][:2])
        a035 = _v1_area(*V1_APERTURES["035"][:2])
        assert zp070 - zp035 == pytest.approx(2.5 * math.log10(a070 / a035), abs=1e-9)

    def test_default_atmosphere(self):
        """Default T_atm = band extinction at airmass 1.3."""
        expected = 10.0 ** (-0.4 * BANDS["V"].extinction_mag_airmass * 1.3)
        assert atm_transmission("V") == pytest.approx(expected, rel=1e-12)
        assert atm_transmission("V", DEFAULT_AIRMASS) == pytest.approx(
            expected, rel=1e-12
        )
        # Explicit atm_transmission overrides the default: T = 1 (space)
        # brightens the zeropoint by exactly -2.5 log10(T_default).
        throughput = specular_throughput("bare_al", 9.0, "V", 1)
        area = _v1_area(*V1_APERTURES["070"][:2])
        zp_atm = compute_zeropoint(area, throughput, V1_QE, "V")
        zp_space = compute_zeropoint(area, throughput, V1_QE, "V", atm_transmission=1.0)
        assert zp_space - zp_atm == pytest.approx(
            -2.5 * math.log10(expected), abs=1e-9
        )

    def test_band_accepts_instance_and_lowercase(self):
        area = _v1_area(*V1_APERTURES["070"][:2])
        throughput = specular_throughput("bare_al", 9.0, "V", 1)
        zp_str = compute_zeropoint(area, throughput, V1_QE, "V")
        zp_obj = compute_zeropoint(area, throughput, V1_QE, BANDS["V"])
        zp_lower = compute_zeropoint(area, throughput, V1_QE, "v")
        assert zp_str == zp_obj == zp_lower


class TestDescribe:
    def test_full_chain_matches_compute_zeropoint(self):
        area = _v1_area(0.70, 0.18)
        info = describe(
            "V", coating="bare_al", sigma_nm=9.0, n_reflections=1,
            area_m2=area, qe=V1_QE,
        )
        throughput = specular_throughput("bare_al", 9.0, "V", 1)
        assert info["band"] == "V"
        assert info["reflectivity"] == pytest.approx(COATINGS["bare_al"]["V"])
        assert info["tis_fraction"] == pytest.approx(tis_fraction(9.0, "V"), rel=1e-12)
        assert info["throughput"] == pytest.approx(throughput, rel=1e-12)
        assert info["zeropoint"] == pytest.approx(
            compute_zeropoint(area, throughput, V1_QE, "V"), abs=1e-12
        )
        assert info["solar_mag"] == -26.74
        assert info["sky_night"] == pytest.approx(21.6)

    def test_partial_chain_leaves_none(self):
        info = describe("SWIR_YJH")
        assert info["band"] == "SWIR_YJH"
        assert info["photon_zp"] > 0.0
        assert info["throughput"] is None
        assert info["zeropoint"] is None
        assert info["atm_transmission"] == pytest.approx(
            atm_transmission("SWIR_YJH"), rel=1e-12
        )

    def test_explicit_atm_transmission_respected(self):
        info = describe("H", atm_transmission=0.5)
        assert info["atm_transmission"] == 0.5
