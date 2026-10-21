"""Band and coating radiometry for cheap-aperture SDA sensors.

Self-contained photometric backbone for the aperture/band/sky trade axes: it owns
the band registry (Vega zero points, photon zero points, extinction, solar
apparent magnitudes, per-regime sky-brightness fallbacks), the mirror-coating
reflectivity table, the roughness-scatter (TIS) model, and the sensor
zeropoint chain that turns (area, coating, roughness, QE, atmosphere) into
the single ``zeropoint`` magnitude the rest of sdasim consumes
(``fpa.mv_to_pe``: mag at which the sensor collects 1 e-/s).

Physics and sources
-------------------
* **Vega flux zero points** ``f0_w_m2_um`` (W/m^2/um at mag 0): UBVRI from
  Bessell, Castelli & Plez 1998 (A&A 333, 231); J/H/Ks are the 2MASS
  isophotal zero points of Cohen, Wheaton & Megeath 2003 (AJ 126, 1090).
  The photon zero point follows as ``photon_zp = f0 * lambda_c / (h c)``
  (photons/s/m^2/um at mag 0); for V this reproduces the classic
  ~1000 photons/s/cm^2/Angstrom (= 1.005e11 photons/s/m^2/um).
* **Solar apparent magnitudes** (Vega system): V = -26.74 is pinned to the
  v1 sdasim convention (modern reference: Willmer 2018, ApJS 236, 47 gives
  V_sun = -26.76 Vega; we keep -26.74 for v1 continuity). Other bands use
  measured solar colors: (V-R)_C = 0.352, (V-I)_C = 0.702 from Ramirez et
  al. 2012 (ApJ 752, 5, solar twins); (V-J) = 1.198, (V-H) = 1.484,
  (V-Ks) = 1.560 in the 2MASS system from Casagrande, Ramirez, Melendez &
  Asplund 2012 (ApJ 761, 16). SWIR_YJH is flux-weighted over 1.0-1.7 um
  (power law through the J/H anchors), giving -27.95.
* **Sky fallbacks** (mag/arcsec^2; ``sky.py`` owns the full sun-driven
  model, these are static per-regime defaults): night zenith V/R/I from
  Patat 2003 (A&A 400, 1183, Paranal); J/H/Ks broadband OH-airglow sky from
  Cuby, Lidman & Moutou 2000 (ESO Messenger 101, 2). Daytime references are
  mid-elevation, ~60-90 deg solar elongation: visible 2-5 mag/arcsec^2 with
  3.5-4 typical (Zimmer, McGraw & Ackermann 2020, AMOS), J/K along the GEO
  belt from MODTRAN5 (Jim, Gibson & Pier 2012, AMOS; IRTF measured J/H/K
  6-6.5 at 45 deg elongation), H interpolated between J and K, I between V
  and J (Thomas & Cobb 2019, AMOS).
* **Extinction** (mag/airmass): V/R/I from the Paranal FORS1 QC ranges and
  Mauna Kea-class sites (Krisciunas 1990: k_V 0.113 summit, 0.172 at
  2800 m; ESO FORS1 QC: V 0.10-0.18, R 0.07-0.11, I 0.03-0.11). NIR
  continuum extinction is a few x 0.01 mag/airmass; the broadband SWIR_YJH
  value is dominated by the saturated telluric H2O band at 1.35-1.45 um
  (~10-15% of the 1.0-1.7 um bandpass) crudely linearized in airmass.
* **Coating reflectivity** (fresh coatings, per surface, normal incidence,
  at band centers): bare/protected aluminum and gold from vendor curves
  (Thorlabs metallic-mirror specs; Edmund Optics metallic coatings note),
  protected silver from Gemini (Boccas et al. 2004, SPIE 5494: ~95-96% vis,
  >98% NIR) and the ARIEL flight coating on bare Al 6061 (Chioetto et al.
  2022, Exp. Astron.: 89-92% at 500 nm rising to ~98% in the NIR).
* **Roughness scatter**: total integrated scatter of a smooth surface,
  ``TIS = (4 pi sigma / lambda)^2`` (Davies 1954; Bennett & Porteus 1961,
  JOSA 51, 123; valid sigma/lambda < ~0.02), so the specular throughput of
  n reflections is ``(R_coat * exp(-TIS))^n``. This is the headline physics
  for cheap LPBF+SPDT apertures: sigma = 10 nm scatters 5.2% at 550 nm but
  only 0.58% at 1650 nm - a bare-aluminum mirror is a SWIR-grade optic.
  The scattered energy is not lost from the scene: it refills a
  sky-following background (v1 did this by hand in YAML comments; scene.py
  integration adds ``+ TIS x sky`` automatically).

Band conventions
----------------
``V`` here is the v2 "VIS" channel: an *unfiltered broadband silicon*
sensor calibrated to Johnson V magnitudes (exactly what the v1 campaigns
flew - IMX455/461/411 with no filter, zeropoints referenced to catalog mv).
Its ``width_nm = 600`` is therefore the *effective collection bandwidth* of
a BSI silicon CMOS (~400-1000 nm), NOT the Johnson V FWHM (~88 nm); with a
band-averaged QE it reproduces the v1 zeropoints (see
``compute_zeropoint``). R and I keep their Cousins filter widths, J/H/Ks
their 2MASS widths. ``SWIR_YJH`` is the daytime-SWIR workhorse: a 1.0-1.7
um InGaAs longpass band; its Vega zero point, solar magnitude, and sky
fallbacks are flux-weighted over the band via a power law through the J and
H anchors, and its TIS is the band average of (4 pi sigma / lambda)^2.

Zeropoint chain (validation anchor)
-----------------------------------
``ZP = 2.5 log10(photon_zp * width_um * area_m2 * throughput * qe * T_atm)``.
For the v1 0.7 m f/2 prime focus (obscuration 0.18, bare Al sigma = 9 nm,
one reflection, corrector losses folded into qe = 0.68, default atmosphere
T = 0.846 at airmass 1.3) this gives ZP_V = 25.14, matching the v1 YAML,
and pure area scaling reproduces the 0.35 m (23.57) and 1.0 m (25.92)
values. Pure python/math - no torch, no numpy required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "BANDS",
    "COATINGS",
    "DEFAULT_AIRMASS",
    "SUN_MV",
    "Band",
    "atm_transmission",
    "compute_zeropoint",
    "describe",
    "specular_throughput",
    "tis_fraction",
]

# Planck constant [J s] and speed of light [m/s] - exact SI values (2019
# redefinition, CODATA).
_H_PLANCK = 6.62607015e-34
_C_LIGHT = 2.99792458e8
_HC = _H_PLANCK * _C_LIGHT  # J m

#: Airmass at which the default atmospheric transmission is evaluated - a
#: typical mid-elevation (~50 deg) survey pointing, and the effective value
#: baked into the v1 zeropoints.
DEFAULT_AIRMASS = 1.3

#: Apparent V magnitude of the Sun (Vega system). Pinned to the v1 sdasim
#: convention (orbits.py Lambertian phase photometry); Willmer 2018
#: (ApJS 236, 47) gives -26.76.
SUN_MV = -26.74


@dataclass(frozen=True)
class Band:
    """A photometric band with its radiometric constants.

    Attributes:
        name: Registry key ("V", "R", "I", "J", "H", "KS", "SWIR_YJH").
        center_nm: Band center (isophotal/effective wavelength), nm.
        width_nm: Effective collection bandwidth, nm (see module docstring:
            for V this is the unfiltered-silicon effective width, not the
            Johnson FWHM).
        f0_w_m2_um: Vega flux zero point, W/m^2/um at mag 0 (Bessell et al.
            1998 for V/R/I; Cohen et al. 2003 for J/H/Ks; band-averaged for
            SWIR_YJH).
        extinction_mag_airmass: Default atmospheric extinction, mag/airmass
            (Mauna Kea/Paranal-class site).
        solar_mag: Apparent magnitude of the Sun in this band (Vega system;
            SUN_MV plus measured solar colors - see module docstring).
        sky_night: Fallback dark-night zenith sky brightness, mag/arcsec^2.
        sky_day_ref: Fallback daytime sky brightness at ~60-90 deg solar
            elongation, mid elevation, mag/arcsec^2. ``sky.py`` owns the
            full elongation/sun-altitude model; these are static defaults.
    """

    name: str
    center_nm: float
    width_nm: float
    f0_w_m2_um: float
    extinction_mag_airmass: float
    solar_mag: float
    sky_night: float
    sky_day_ref: float

    @property
    def photon_zp(self) -> float:
        """Photon zero point: photons/s/m^2/um at mag 0.

        ``f0 / (h c / lambda_c)`` - monochromatic photon energy at the band
        center. For V this is 1.005e11 photons/s/m^2/um, the classic
        ~1000 photons/s/cm^2/Angstrom rule of thumb.
        """
        return self.f0_w_m2_um * (self.center_nm * 1e-9) / _HC

    @property
    def lambda_min_nm(self) -> float:
        """Blue edge of the band (center - width/2), nm."""
        return self.center_nm - 0.5 * self.width_nm

    @property
    def lambda_max_nm(self) -> float:
        """Red edge of the band (center + width/2), nm."""
        return self.center_nm + 0.5 * self.width_nm


def _powerlaw_band_average(
    f_a: float,
    lam_a: float,
    f_b: float,
    lam_b: float,
    lam_lo: float,
    lam_hi: float,
) -> float:
    """Average over [lam_lo, lam_hi] of the power law through two anchors.

    Fits ``f(lam) = f_a * (lam / lam_a)**alpha`` through (lam_a, f_a) and
    (lam_b, f_b), then returns its mean over [lam_lo, lam_hi] (closed-form
    integral). Used to synthesize the broadband SWIR_YJH quantities (Vega
    zero point, solar flux, sky flux) from their J- and H-band anchors -
    a two-point spectral model, adequate for a ~0.7 um broadband fallback.

    Args:
        f_a: Flux at ``lam_a`` (any linear flux unit).
        lam_a: First anchor wavelength (same unit as lam_b/lam_lo/lam_hi).
        f_b: Flux at ``lam_b``.
        lam_b: Second anchor wavelength.
        lam_lo: Lower edge of the averaging interval.
        lam_hi: Upper edge of the averaging interval.

    Returns:
        Band-averaged flux, same unit as ``f_a``.
    """
    alpha = math.log(f_b / f_a) / math.log(lam_b / lam_a)
    p = alpha + 1.0
    if abs(p) < 1e-12:  # f ~ 1/lam: integral is logarithmic
        integral = math.log(lam_hi / lam_lo)
    else:
        integral = (lam_hi**p - lam_lo**p) / p
    return f_a * lam_a ** (-alpha) * integral / (lam_hi - lam_lo)


# --- SWIR_YJH (1.0-1.7 um InGaAs longpass) synthesis from J/H anchors -----
# 2MASS anchors (Cohen et al. 2003): isophotal wavelengths and zero points.
_LAM_J_UM, _F0_J = 1.235, 3.129e-9  # W/m^2/um
_LAM_H_UM, _F0_H = 1.662, 1.133e-9
_YJH_LO_UM, _YJH_HI_UM = 1.0, 1.7

#: Band-averaged Vega zero point of the 1.0-1.7 um band: 2.75e-9 W/m^2/um.
_F0_YJH = _powerlaw_band_average(
    _F0_J, _LAM_J_UM, _F0_H, _LAM_H_UM, _YJH_LO_UM, _YJH_HI_UM
)


def _yjh_mag_from_jh(mag_j: float, mag_h: float) -> float:
    """Broadband 1.0-1.7 um magnitude of a source with given J/H mags.

    Converts the J/H (2MASS) magnitudes to fluxes, band-averages the power
    law through them over 1.0-1.7 um, and references the result to the
    band-averaged Vega zero point ``_F0_YJH``.
    """
    f_j = _F0_J * 10.0 ** (-0.4 * mag_j)
    f_h = _F0_H * 10.0 ** (-0.4 * mag_h)
    f_band = _powerlaw_band_average(
        f_j, _LAM_J_UM, f_h, _LAM_H_UM, _YJH_LO_UM, _YJH_HI_UM
    )
    return -2.5 * math.log10(f_band / _F0_YJH)


# Solar colors (Vega system): (V-R)_C, (V-I)_C from Ramirez et al. 2012
# (ApJ 752, 5); (V-J), (V-H), (V-Ks) 2MASS from Casagrande et al. 2012
# (ApJ 761, 16).
_SUN_R = SUN_MV - 0.352  # -27.09
_SUN_I = SUN_MV - 0.702  # -27.44
_SUN_J = SUN_MV - 1.198  # -27.94
_SUN_H = SUN_MV - 1.484  # -28.22
_SUN_KS = SUN_MV - 1.560  # -28.30
_SUN_YJH = _yjh_mag_from_jh(_SUN_J, _SUN_H)  # -27.95 (flux-weighted)

# SWIR_YJH sky fallbacks, flux-weighted from the J/H values below:
# night 16.5/14.4 -> 16.0; day (60-90 deg elongation) 5.5/6.5 -> 5.4.
_SKY_NIGHT_YJH = _yjh_mag_from_jh(16.5, 14.4)
_SKY_DAY_YJH = _yjh_mag_from_jh(5.5, 6.5)

#: Band registry. Centers/widths: Bessell et al. 1998 (V/R/I; V width is the
#: unfiltered-silicon effective bandwidth, see module docstring), Cohen et
#: al. 2003 (2MASS J/H/Ks). Vega zero points, extinction, solar magnitudes,
#: and sky fallbacks per the module docstring sources.
BANDS: dict[str, Band] = {
    # v2 "VIS" channel: unfiltered broadband silicon calibrated to V mags.
    # width 600 nm = effective 400-1000 nm BSI CMOS bandpass (NOT Johnson
    # FWHM 88 nm); f0/solar/sky are Johnson-V-referenced. k_V = 0.14
    # mag/airmass (Mauna Kea-class: Krisciunas 1990; Paranal 0.10-0.18).
    # Sky: night 21.6 (Patat 2003); day 4.0 (Zimmer et al. 2020, 3.5-4
    # typical at 60-90 deg elongation).
    "V": Band("V", 550.0, 600.0, 3.63e-8, 0.14, SUN_MV, 21.6, 4.0),
    # Cousins R (Bessell 1998: lambda_eff 641 nm, FWHM 158 nm). Sky: night
    # 20.9 (Patat 2003); day 4.5 (between V and I, Zimmer 2020 range).
    "R": Band("R", 641.0, 158.0, 2.18e-8, 0.09, _SUN_R, 20.9, 4.5),
    # Cousins I (Bessell 1998: 798 nm, FWHM 154 nm). Sky: night 19.7 (Patat
    # 2003); day 5.0 (between V and J; Thomas & Cobb 2019 LEEDR I/J maps).
    "I": Band("I", 798.0, 154.0, 1.13e-8, 0.06, _SUN_I, 19.7, 5.0),
    # 2MASS J (Cohen 2003: 1.235 um, width 0.162 um). Sky: night 16.5
    # broadband OH (Cuby 2000); day 5.5 (Jim 2012, 60 deg elongation).
    "J": Band("J", 1235.0, 162.0, _F0_J, 0.05, _SUN_J, 16.5, 5.5),
    # 2MASS H (Cohen 2003: 1.662 um, width 0.251 um). Sky: night 14.4 (Cuby
    # 2000); day 6.5 (between J and K; IRTF measured 6-6.5 at 45 deg).
    "H": Band("H", 1662.0, 251.0, _F0_H, 0.03, _SUN_H, 14.4, 6.5),
    # 2MASS Ks (Cohen 2003: 2.159 um, width 0.262 um). Sky: night 13.0
    # (Cuby 2000, incl. thermal); day 7.5 (Jim 2012, 60 deg elongation).
    "KS": Band("KS", 2159.0, 262.0, 4.283e-10, 0.05, _SUN_KS, 13.0, 7.5),
    # Daytime-SWIR workhorse: 1.0-1.7 um InGaAs longpass. f0/solar/sky are
    # flux-weighted over the band (power law through the J/H anchors).
    # Extinction 0.12 mag/airmass: saturated telluric H2O at 1.35-1.45 um
    # removes ~10-15% of the bandpass (crude airmass linearization), plus
    # small NIR continuum extinction.
    "SWIR_YJH": Band(
        "SWIR_YJH", 1350.0, 700.0, _F0_YJH, 0.12, _SUN_YJH, _SKY_NIGHT_YJH, _SKY_DAY_YJH
    ),
}

#: Per-surface specular reflectivity at each band center (fresh coating,
#: normal incidence). Sources: bare Al / protected Al / gold from Thorlabs
#: metallic-mirror spec curves and the Edmund Optics metallic-coatings note
#: (bare Al interband dip: minimum ~86% at ~826 nm, inside I band);
#: protected Ag from Boccas et al. 2004 (Gemini, SPIE 5494: ~95-96%
#: visible, >98% NIR) and Chioetto et al. 2022 (ARIEL flight coating on Al
#: 6061: 89-92% at 500 nm, ~98% NIR) - we carry the catalog ">=97% from
#: 450 nm to 2 um" curve. Gold: ~79% at 550 nm, ~94% at 650 nm, ~97% at
#: 800 nm, 98.5-99% J through K - SWIR-only designs. SWIR_YJH is the J/H
#: mean (the band-average of the slowly varying NIR curves to ~0.5%).
COATINGS: dict[str, dict[str, float]] = {
    "bare_al": {"V": 0.915, "R": 0.905, "I": 0.865, "J": 0.960, "H": 0.974, "KS": 0.978},
    "protected_al": {"V": 0.870, "R": 0.860, "I": 0.840, "J": 0.940, "H": 0.950, "KS": 0.960},
    "protected_ag": {"V": 0.975, "R": 0.980, "I": 0.980, "J": 0.985, "H": 0.985, "KS": 0.985},
    "gold": {"V": 0.790, "R": 0.940, "I": 0.974, "J": 0.985, "H": 0.990, "KS": 0.990},
}
for _refl in COATINGS.values():
    _refl["SWIR_YJH"] = round(0.5 * (_refl["J"] + _refl["H"]), 4)
del _refl


def _band_lookup(band: Band | str) -> Band:
    """Resolve a Band instance or (case-insensitive) registry key."""
    if isinstance(band, Band):
        return band
    key = str(band).upper()
    try:
        return BANDS[key]
    except KeyError:
        raise KeyError(
            f"unknown band {band!r}; available: {', '.join(sorted(BANDS))}"
        ) from None


def _coating_lookup(coating: str) -> dict[str, float]:
    """Resolve a coating name to its per-band reflectivity table."""
    try:
        return COATINGS[coating.lower()]
    except KeyError:
        raise KeyError(
            f"unknown coating {coating!r}; available: {', '.join(sorted(COATINGS))}"
        ) from None


def tis_fraction(sigma_nm: float, band_or_wavelength_nm: Band | str | float) -> float:
    """Total integrated scatter of a surface with RMS micro-roughness sigma.

    Smooth-surface scatter at normal incidence (Davies 1954; Bennett &
    Porteus 1961): ``TIS = (4 pi sigma / lambda)^2``, the fraction of the
    specular beam scattered out of the core per reflection - valid for
    sigma/lambda < ~0.02 (sigma <= ~10 nm at 550 nm). Anchors (sigma =
    10 nm): 5.22% at 550 nm, 0.58% at 1650 nm - exactly 9x less in H than
    V, the headline physics for cheap bare-aluminum SWIR apertures.

    Args:
        sigma_nm: RMS surface micro-roughness, nm (per surface).
        band_or_wavelength_nm: A ``Band``, a band name, or a wavelength in
            nm. Bands evaluate at their center; the broadband SWIR_YJH
            band averages ``1/lambda^2`` over 1.0-1.7 um (closed form:
            ``<1/lambda^2> = 1/(lambda_min * lambda_max)``, i.e. the TIS at
            the geometric-mean wavelength, 1304 nm).

    Returns:
        Scattered energy fraction per reflection (dimensionless).
    """
    four_pi_sigma = 4.0 * math.pi * sigma_nm
    if isinstance(band_or_wavelength_nm, (int, float)):
        return (four_pi_sigma / float(band_or_wavelength_nm)) ** 2
    band = _band_lookup(band_or_wavelength_nm)
    if band.name == "SWIR_YJH":
        # Flat-weighted band average of 1/lambda^2 over [lo, hi]:
        # (1/(hi-lo)) * int 1/lam^2 dlam = 1/(lo*hi).
        return four_pi_sigma**2 / (band.lambda_min_nm * band.lambda_max_nm)
    return (four_pi_sigma / band.center_nm) ** 2


def specular_throughput(
    coating: str,
    sigma_nm: float,
    band: Band | str,
    n_reflections: int = 1,
) -> float:
    """Specular optical throughput of n mirror reflections.

    Per reflection: ``R_spec = R_coat(band) * exp(-(4 pi sigma/lambda)^2)``
    (Davies/Bennett-Porteus smooth-surface approximation; the scattered
    TIS energy is not destroyed - it feeds a sky-following scatter
    background, see ``tis_fraction``). Total: ``R_spec ** n_reflections``.

    Args:
        coating: One of ``COATINGS`` ("bare_al", "protected_al",
            "protected_ag", "gold").
        sigma_nm: RMS micro-roughness per surface, nm (e.g. 9 nm for
            porosity-pitted bare LPBF+SPDT aluminum, ~2 nm for NiP+MRF).
        band: Band instance or name (reflectivity at band center; SWIR_YJH
            uses the band-averaged TIS and J/H-mean reflectivity).
        n_reflections: Number of mirror reflections (1 = prime focus,
            2 = Cassegrain, 3 = TMA).

    Returns:
        End-to-end specular mirror throughput in [0, 1].
    """
    band = _band_lookup(band)
    reflectivity = _coating_lookup(coating)[band.name]
    per_surface = reflectivity * math.exp(-tis_fraction(sigma_nm, band))
    return per_surface**n_reflections


def atm_transmission(band: Band | str, airmass: float = DEFAULT_AIRMASS) -> float:
    """Atmospheric transmission from the band's default extinction.

    ``T = 10^(-0.4 * k * X)`` with k = ``band.extinction_mag_airmass``.

    Args:
        band: Band instance or name.
        airmass: Airmass X (default ``DEFAULT_AIRMASS`` = 1.3).

    Returns:
        Broadband atmospheric transmission in (0, 1].
    """
    band = _band_lookup(band)
    return 10.0 ** (-0.4 * band.extinction_mag_airmass * airmass)


def compute_zeropoint(
    area_m2: float,
    throughput: float,
    qe: float,
    band: Band | str,
    atm_transmission: float | None = None,
) -> float:
    """Sensor zeropoint: the magnitude that yields 1 photoelectron/s.

    ``ZP = 2.5 log10(photon_zp * width_um * area_m2 * throughput * qe *
    T_atm)`` - the electron rate a mag-0 source produces, on the Pogson
    scale, directly consumable as ``sensor.zeropoint`` by ``fpa.mv_to_pe``.

    Validation anchor (v1): 0.7 m f/2 prime focus, obscured area
    ``pi/4 * 0.49 * (1 - 0.18^2)`` m^2, bare Al sigma = 9 nm, one
    reflection (corrector losses folded into qe), band V with qe = 0.68
    (band-averaged BSI silicon QE ~0.72 x corrector transmission ~0.95)
    and the default atmosphere (k_V = 0.14 at airmass 1.3, T = 0.846;
    qe * T_atm = 0.575) gives ZP = 25.14 - the v1 YAML value - and pure
    area scaling reproduces the 0.35 m (23.57) and 1.0 m (25.92) YAMLs.

    Args:
        area_m2: Unobscured collecting area, m^2 (obscuration and gap
            fractions already applied).
        throughput: Specular optical throughput of the mirror train, e.g.
            from ``specular_throughput``.
        qe: Effective detector quantum efficiency, band-averaged, with any
            refractive-element (corrector/window) losses folded in.
        band: Band instance or name (supplies photon_zp and width).
        atm_transmission: Broadband atmospheric transmission; ``None``
            (default) evaluates the band's extinction at airmass 1.3.
            Pass 1.0 for a space sensor or an exoatmospheric zeropoint.

    Returns:
        Zeropoint magnitude (mag giving 1 e-/s).
    """
    band = _band_lookup(band)
    if atm_transmission is None:
        t_atm = 10.0 ** (-0.4 * band.extinction_mag_airmass * DEFAULT_AIRMASS)
    else:
        t_atm = atm_transmission
    rate = band.photon_zp * (band.width_nm * 1e-3) * area_m2 * throughput * qe * t_atm
    if rate <= 0.0:
        raise ValueError(
            f"non-positive mag-0 electron rate ({rate}); check area/throughput/qe/atm"
        )
    return 2.5 * math.log10(rate)


def describe(
    band: Band | str,
    coating: str | None = None,
    sigma_nm: float | None = None,
    n_reflections: int = 1,
    area_m2: float | None = None,
    qe: float | None = None,
    atm_transmission: float | None = None,
    airmass: float = DEFAULT_AIRMASS,
) -> dict[str, float | str | int | None]:
    """Resolve the full radiometric chain into a flat metadata dict.

    Intended for run metadata / FITS-header provenance: every number the
    zeropoint and background chain used, in one place. Keys whose inputs
    were not provided are ``None`` (e.g. no ``zeropoint`` without area and
    qe; no ``throughput`` without coating and sigma).

    Args:
        band: Band instance or name.
        coating: Optional coating name (see ``COATINGS``).
        sigma_nm: Optional RMS micro-roughness per surface, nm.
        n_reflections: Mirror reflection count (used with coating/sigma).
        area_m2: Optional unobscured collecting area, m^2.
        qe: Optional effective QE (with refractive losses folded in).
        atm_transmission: Optional explicit atmospheric transmission;
            ``None`` derives it from the band extinction at ``airmass``.
        airmass: Airmass for the derived atmospheric transmission.

    Returns:
        Dict with the band constants (center/width/f0/photon_zp/extinction/
        solar_mag/sky_night/sky_day_ref), the atmosphere (airmass,
        atm_transmission), and - when derivable - reflectivity, tis,
        throughput, and zeropoint.
    """
    band = _band_lookup(band)
    if atm_transmission is None:
        t_atm = 10.0 ** (-0.4 * band.extinction_mag_airmass * airmass)
    else:
        t_atm = atm_transmission

    reflectivity: float | None = None
    tis: float | None = None
    throughput: float | None = None
    if coating is not None and sigma_nm is not None:
        reflectivity = _coating_lookup(coating)[band.name]
        tis = tis_fraction(sigma_nm, band)
        throughput = specular_throughput(coating, sigma_nm, band, n_reflections)

    zeropoint: float | None = None
    if throughput is not None and area_m2 is not None and qe is not None:
        zeropoint = compute_zeropoint(area_m2, throughput, qe, band, t_atm)

    return {
        "band": band.name,
        "center_nm": band.center_nm,
        "width_nm": band.width_nm,
        "f0_w_m2_um": band.f0_w_m2_um,
        "photon_zp": band.photon_zp,
        "extinction_mag_airmass": band.extinction_mag_airmass,
        "solar_mag": band.solar_mag,
        "sky_night": band.sky_night,
        "sky_day_ref": band.sky_day_ref,
        "airmass": airmass,
        "atm_transmission": t_atm,
        "coating": coating,
        "sigma_nm": sigma_nm,
        "n_reflections": n_reflections,
        "reflectivity": reflectivity,
        "tis_fraction": tis,
        "throughput": throughput,
        "area_m2": area_m2,
        "qe": qe,
        "zeropoint": zeropoint,
    }
