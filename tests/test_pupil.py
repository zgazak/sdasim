"""Tests for aperture masks + diffraction PSF kernels (pupil.py).

Validation anchors (recon research_apertures.md, all literature-sourced and
Monte-Carlo verified there):

  * circular D: first zero at 1.22 lambda/D, EE(first ring) = 0.838
    (Born & Wolf; Schroeder, Astronomical Optics)
  * annular eps: EE(first ring) 0.838 -> 0.654 (eps 0.33) -> 0.479 (eps 0.5)
    (Schroeder's obscuration table)
  * ellipse: per-axis lambda/D scaling — FWHM ratio = axis ratio
  * rectangle: separable sinc^2, first zeros at lambda/L per axis, central-lobe
    2D energy 0.815 (Goodman)
  * segment piston Strehl S = exp(-s^2) + (1 - exp(-s^2))/N vs FFT Monte Carlo
    (Chanan & Troy 1999)
  * energy conservation: every kernel sums to 1
  * pupil_quadrature reproduces optics.pupil_samples statistics on the annulus
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sdasim.pupil import (
    airy_first_zero_arcsec,
    annular_ee_first_ring,
    annular_first_zero_lamD,
    aperture_area,
    circle_mask,
    ellipse_mask,
    ensquared_energy,
    gap_strehl,
    piston_strehl,
    psf_kernel,
    pupil_quadrature,
    rect_mask,
    segmented_mask,
)

LAM_NM = 550.0
FNUM = 8.0
LAM_FNUM_UM = LAM_NM * 1e-3 * FNUM  # lambda * F# on the detector: 4.4 um


def mono_kernel(mask_info, pitch_um, kernel_px, **kw):
    """Monochromatic 550 nm kernel at unit oversampling (fine-pitch probe)."""
    return psf_kernel(
        mask_info, [LAM_NM], [1.0], FNUM, pitch_um, kernel_px=kernel_px, oversample=1, **kw
    )


def first_min_radius_px(profile: np.ndarray) -> float:
    """Radius (samples) of the first local minimum of a center-out profile."""
    d = np.diff(profile)
    idx = int(np.nonzero(d > 0)[0][0])  # first uptick -> min at idx
    a, b, c = profile[idx - 1], profile[idx], profile[idx + 1]
    return idx + 0.5 * (a - c) / (a - 2 * b + c)  # parabolic vertex


def half_max_radius_px(profile: np.ndarray) -> float:
    """Radius (samples) where a center-out profile first crosses half its peak."""
    half = profile[0] / 2.0
    idx = int(np.nonzero(profile < half)[0][0])
    lo, hi = profile[idx - 1], profile[idx]
    return (idx - 1) + (lo - half) / (lo - hi)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def circle_kernel():
    """Filled 1.5 m circle, 0.25 um probe pitch, 1024 px window (58 lambda F#)."""
    return mono_kernel(circle_mask(1.5), 0.25, 1024)


@pytest.fixture(scope="module")
def annular_kernel():
    """1.5 m annulus, eps = 0.33 (the v1-like obscuration), same probe grid."""
    return mono_kernel(circle_mask(1.5, obscuration=0.33), 0.25, 1024)


# --------------------------------------------------------------------------- anchor 1


class TestCircleAiry:
    def test_first_zero_radius(self, circle_kernel):
        c = 512
        measured = first_min_radius_px(circle_kernel[c, c:])
        expected = annular_first_zero_lamD(0.0) * LAM_FNUM_UM / 0.25  # 21.47 px
        assert measured == pytest.approx(expected, rel=0.02)

    def test_ee_first_ring(self, circle_kernel):
        c = 512
        r1 = annular_first_zero_lamD(0.0) * LAM_FNUM_UM / 0.25
        ax = np.arange(1024) - c
        rr = np.hypot(ax[:, None], ax[None, :])
        ee = circle_kernel[rr <= r1].sum()
        # window captures ~99.4% of the total, biasing EE high by ~0.005
        assert ee == pytest.approx(0.838, abs=0.025)

    def test_airy_first_zero_arcsec(self):
        # 1.22 * 550 nm / 1.5 m = 0.0922 arcsec
        assert airy_first_zero_arcsec(550.0, 1.5) == pytest.approx(0.09224, rel=1e-3)

    def test_peak_centered(self, circle_kernel):
        assert np.unravel_index(np.argmax(circle_kernel), circle_kernel.shape) == (512, 512)


# --------------------------------------------------------------------------- anchor 2


class TestAnnular:
    def test_ee_table(self):
        # Schroeder, Astronomical Optics (recon-verified): eps -> EE(first ring)
        assert annular_ee_first_ring(0.0) == pytest.approx(0.838, abs=0.01)
        assert annular_ee_first_ring(0.33) == pytest.approx(0.654, abs=0.012)
        assert annular_ee_first_ring(0.5) == pytest.approx(0.479, abs=0.012)

    def test_first_zero_shrinks(self):
        # O'Neill 1956: 1.22 -> 1.00 lambda/D as eps -> 0.5
        assert annular_first_zero_lamD(0.0) == pytest.approx(1.2197, abs=0.005)
        assert annular_first_zero_lamD(0.33) == pytest.approx(1.098, abs=0.005)
        assert annular_first_zero_lamD(0.5) == pytest.approx(1.001, abs=0.005)

    def test_numeric_kernel_matches_analytic(self, annular_kernel):
        c = 512
        r1 = annular_first_zero_lamD(0.33) * LAM_FNUM_UM / 0.25
        measured_zero = first_min_radius_px(annular_kernel[c, c:])
        assert measured_zero == pytest.approx(r1, rel=0.025)
        ax = np.arange(1024) - c
        rr = np.hypot(ax[:, None], ax[None, :])
        ee = annular_kernel[rr <= r1].sum()
        assert ee == pytest.approx(annular_ee_first_ring(0.33), abs=0.02)


# --------------------------------------------------------------------------- anchor 3


class TestEllipse:
    def test_fwhm_ratio_matches_axis_ratio(self):
        kern = mono_kernel(ellipse_mask(1.5, 0.75), 0.5, 512)
        c = 256
        fwhm_x = 2 * half_max_radius_px(kern[c, c:])  # conjugate to d_major: narrow
        fwhm_y = 2 * half_max_radius_px(kern[c:, c])  # conjugate to d_minor: wide
        assert fwhm_y / fwhm_x == pytest.approx(2.0, rel=0.05)
        # absolute scale: Airy FWHM = 1.028 lambda/D_major along x
        assert fwhm_x == pytest.approx(1.028 * LAM_FNUM_UM / 0.5, rel=0.05)


# --------------------------------------------------------------------------- anchor 4


class TestRect:
    def test_first_zeros_per_axis(self):
        # 1.0 x 0.5 m: f = FNUM * 1.0, zeros at lambda f / L per axis
        kern = mono_kernel(rect_mask(1.0, 0.5), 0.25, 768)
        c = 384
        zero_x = LAM_NM * 1e-3 * FNUM * 1.0 / 1.0 / 0.25  # 17.6 px
        zero_y = LAM_NM * 1e-3 * FNUM * 1.0 / 0.5 / 0.25  # 35.2 px
        assert first_min_radius_px(kern[c, c:]) == pytest.approx(zero_x, rel=0.025)
        assert first_min_radius_px(kern[c:, c]) == pytest.approx(zero_y, rel=0.025)

    def test_central_lobe_energy(self):
        # square 1.05 m: central sinc^2 lobe holds 0.815 of the energy (Goodman)
        kern = mono_kernel(rect_mask(1.05, 1.05), 0.25, 1024)
        c = 512
        z = LAM_FNUM_UM / 0.25  # zero at lambda f / L = lambda F#(max-extent) = 17.6 px
        ax = np.abs(np.arange(1024) - c)
        inside = (ax[:, None] < z) & (ax[None, :] < z)
        assert kern[inside].sum() == pytest.approx(0.815, abs=0.02)


# --------------------------------------------------------------------------- anchor 5


SEGMENTS = [
    {"x": 0.6 * math.cos(a), "y": 0.6 * math.sin(a), "d": 1.0}
    for a in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3)
]


@pytest.fixture(scope="module")
def seg_info():
    return segmented_mask(SEGMENTS, n=384)


class TestSegmented:
    def test_meta(self, seg_info):
        _, meta = seg_info
        assert meta["d_res_m"]["phased"] == pytest.approx(2.2)
        assert meta["d_res_m"]["unphased"] == pytest.approx(1.0)
        assert meta["area_m2"] == pytest.approx(3 * math.pi / 4)

    def test_piston_strehl_monte_carlo(self, seg_info):
        # 60 nm rms piston OPD at 550 nm, N = 3: S = 0.750 (Chanan & Troy 1999).
        # 40 fixed-seed draws; sample std of the mean is ~0.03, tolerance 0.08.
        sigma = 2 * math.pi * 60.0 / 550.0
        analytic = piston_strehl(sigma, 3)
        kw = dict(pitch_um=0.5, kernel_px=256)
        ref = mono_kernel(seg_info, **kw)
        c = 128
        strehls = [
            mono_kernel(seg_info, seg_piston_rms_nm=60.0, seed=s, **kw)[c, c] / ref[c, c]
            for s in range(40)
        ]
        assert np.mean(strehls) == pytest.approx(analytic, rel=0.08)

    def test_unphased_is_single_segment(self, seg_info):
        # unphased segmented == the single-segment kernel (co-add = N x flux,
        # single-segment resolution); piston is ignored in this mode
        kern_u = mono_kernel(seg_info, 0.5, 256, phased=False, seg_piston_rms_nm=60.0)
        kern_1 = mono_kernel(circle_mask(1.0, n=384), 0.5, 256)
        assert np.allclose(kern_u, kern_1, atol=1e-12)

    def test_phased_core_narrower_than_unphased(self, seg_info):
        # ON-SKY the phased 2.2 m array core beats the 1.0 m segment core.
        # (In detector pixels both are ~lambda F#: the focal length is
        # f_number * d_res in each mode, so the comparison must be angular.)
        kern_p = mono_kernel(seg_info, 0.5, 256)
        kern_u = mono_kernel(seg_info, 0.5, 256, phased=False)
        c = 128
        _, meta = seg_info
        ang_p = 2 * half_max_radius_px(kern_p[c, c:]) * 0.5e-6 / (FNUM * meta["d_res_m"]["phased"])
        ang_u = 2 * half_max_radius_px(kern_u[c, c:]) * 0.5e-6 / (FNUM * 1.0)
        assert ang_p < 0.7 * ang_u  # 2.2 m array core vs 1.0 m segment core

    def test_unphased_tiptilt_blurs(self, seg_info):
        sharp = mono_kernel(seg_info, 2.0, 64, phased=False)
        # 0.2 arcsec rms tilt at f = 8 m -> ~7.8 um ~ 3.9 px blur
        blurred = mono_kernel(seg_info, 2.0, 64, phased=False, seg_tiptilt_rms_arcsec=0.2)
        assert blurred.max() < 0.7 * sharp.max()
        assert blurred.sum() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- anchor 6


ALL_SHAPES = [
    ("circle", lambda: circle_mask(1.5, n=256)),
    ("annulus", lambda: circle_mask(1.5, obscuration=0.33, n=256)),
    ("ellipse", lambda: ellipse_mask(1.5, 0.75, n=256)),
    ("rect", lambda: rect_mask(1.05, 1.05, n=256)),
    ("segmented", lambda: segmented_mask(SEGMENTS, n=256)),
]


class TestEnergyConservation:
    @pytest.mark.parametrize("name,builder", ALL_SHAPES, ids=[s[0] for s in ALL_SHAPES])
    def test_kernel_sums_to_one(self, name, builder):
        kern = psf_kernel(
            builder(),
            [500.0, 550.0, 600.0],
            [0.25, 0.5, 0.25],
            FNUM,
            5.0,
            kernel_px=64,
            oversample=4,
        )
        assert kern.sum() == pytest.approx(1.0, abs=1e-3)
        assert kern.min() >= 0.0
        r, c = np.unravel_index(np.argmax(kern), kern.shape)
        assert abs(r - 32) <= 1 and abs(c - 32) <= 1


# --------------------------------------------------------------------------- anchor 7


class TestQuadrature:
    def test_matches_optics_pupil_samples_on_annulus(self):
        from sdasim.optics import pupil_samples

        eps = 0.35
        mask, _ = circle_mask(1.0, obscuration=eps)
        pts, wts = pupil_quadrature(mask, 256)
        p_ref, w_ref = pupil_samples(256, eps)
        mean_r = float((wts * np.hypot(pts[:, 0], pts[:, 1])).sum())
        mean_r_ref = float((w_ref * p_ref.norm(dim=1)).sum())
        analytic = 2.0 / 3.0 * (1 - eps**3) / (1 - eps**2)
        assert wts.sum() == pytest.approx(1.0, abs=1e-9)
        assert mean_r == pytest.approx(mean_r_ref, rel=0.02)
        assert mean_r == pytest.approx(analytic, rel=0.02)

    def test_inversion_symmetric(self):
        mask, _ = circle_mask(1.0, obscuration=0.35)
        pts, _ = pupil_quadrature(mask, 200)
        d = np.linalg.norm(pts[:, None, :] + pts[None, :, :], axis=-1)
        assert d.min(axis=1).max() < 1e-8  # every point's antipode is present

    def test_support_and_count(self):
        mask, _ = circle_mask(1.0, obscuration=0.35)
        pts, wts = pupil_quadrature(mask, 256)
        r = np.hypot(pts[:, 0], pts[:, 1])
        assert r.max() <= 1.01 and r.min() >= 0.3  # nodes live on the annulus
        assert 128 <= len(wts) <= 512  # tracks the request approximately
        assert (wts > 0).all()


# --------------------------------------------------------------------------- masks + areas


class TestMasks:
    def test_mask_sum_matches_analytic_area(self):
        for mask, meta in [
            circle_mask(1.5, obscuration=0.33),
            ellipse_mask(1.5, 0.75),
            rect_mask(1.0, 0.5),
            segmented_mask(SEGMENTS),
        ]:
            mask_area = mask.sum() * meta["grid_m"] ** 2
            assert mask_area == pytest.approx(meta["area_m2"], rel=1e-3)

    def test_aperture_area(self):
        assert aperture_area("circle", d=2.0) == pytest.approx(math.pi)
        assert aperture_area("circle", d=2.0, obscuration=0.5) == pytest.approx(0.75 * math.pi)
        assert aperture_area("ellipse", d_major=2.0, d_minor=1.0) == pytest.approx(math.pi / 2)
        assert aperture_area("rect", width=3.0, height=0.5) == pytest.approx(1.5)
        segs = [{"x": 0, "y": 0, "d": 2.0}, {"x": 3, "y": 0, "d": 2.0}]
        assert aperture_area("segmented", segments=segs) == pytest.approx(2 * math.pi)

    def test_grid_meta_consistent(self):
        mask, meta = circle_mask(1.5, n=128)
        assert mask.shape == (128, 128)
        assert meta["grid_m"] * meta["n"] == pytest.approx(meta["extent_m"])
        assert meta["d_res_m"] == 1.5

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            circle_mask(-1.0)
        with pytest.raises(ValueError):
            circle_mask(1.0, obscuration=1.1)
        with pytest.raises(ValueError):
            ellipse_mask(0.5, 1.0)  # major < minor
        with pytest.raises(ValueError):
            segmented_mask([])
        with pytest.raises(ValueError):
            pupil_quadrature(np.zeros((16, 16)), 10)
        with pytest.raises(ValueError):
            psf_kernel(circle_mask(1.0, n=64), [550.0, 600.0], [1.0], FNUM, 5.0)
        with pytest.raises(ValueError):
            psf_kernel(circle_mask(1.0, n=64), [550.0], [1.0], FNUM, 5.0, wfe={"bogus_term": 0.1})


# --------------------------------------------------------------------------- analytic helpers


class TestAnalyticHelpers:
    def test_piston_strehl_limits(self):
        assert piston_strehl(0.0, 7) == 1.0
        assert piston_strehl(50.0, 4) == pytest.approx(0.25)  # 1/N incoherent floor
        assert piston_strehl(0.5, 6) < piston_strehl(0.25, 6)

    def test_gap_strehl(self):
        assert gap_strehl(0.0) == 1.0
        assert gap_strehl(0.01) == pytest.approx(0.99**4)

    def test_ensquared_energy(self):
        delta = np.zeros((33, 33))
        delta[16, 16] = 2.0
        assert ensquared_energy(delta, 0) == 1.0
        flat = np.ones((25, 25))
        flat[12, 12] = 2.0  # unique peak so the box is centered
        assert ensquared_energy(flat, 1) == pytest.approx(10 / 626)

    def test_ensquared_energy_on_airy(self, circle_kernel):
        # 2 lambda F# square (w = 2 in recon's ensquared table) holds ~0.83;
        # probe pitch 0.25 um -> half-width = lambda F# = 17.6 samples
        h = int(round(LAM_FNUM_UM / 0.25))
        assert ensquared_energy(circle_kernel, h) == pytest.approx(0.83, abs=0.04)


# --------------------------------------------------------------------------- wavefront error


class TestWfe:
    def test_defocus_broadens_and_dims(self):
        info = circle_mask(1.5, n=256)
        kw = dict(f_number=FNUM, pixel_pitch_um=5.0, kernel_px=32, oversample=4)
        kern0 = psf_kernel(info, [550.0], [1.0], **kw)
        kernd = psf_kernel(info, [550.0], [1.0], wfe={"defocus": 1.0}, **kw)
        assert kernd.max() < 0.7 * kern0.max()  # Strehl loss
        ax = np.arange(32) - 16
        rr2 = ax[:, None] ** 2 + ax[None, :] ** 2
        assert (kernd * rr2).sum() > (kern0 * rr2).sum()  # broadened
