"""Aperture geometry + Fraunhofer-diffraction PSF kernels for arbitrary pupil shapes.

Builds amplitude masks for the v2 aperture trade axes (circle/annulus, ellipse,
rectangle, segmented arrays), turns them into detector-sampled polychromatic PSF
kernels by Fourier transform of the complex pupil, and provides the closed-form
fast paths (Airy scale, annular encircled energy, segment-piston and gap Strehl)
the detection-mode bookkeeping and the tests rely on.

Physics and conventions (sources cited inline next to each constant):

  * Fraunhofer regime: the focal-plane amplitude is the Fourier transform of the
    complex pupil ``P(xi) = mask(xi) * exp(i 2 pi OPD(xi) / lambda)`` evaluated at
    spatial frequency ``x / (lambda f)`` (Goodman, *Introduction to Fourier
    Optics*). The transform is computed as a matrix Fourier transform (MFT,
    Soummer et al. 2007, Opt. Express 15, 15935) directly onto the oversampled
    detector grid — exact output pixel scale at any wavelength, no FFT-padding
    bookkeeping — then binned to the detector pitch. Internal oversampling
    defaults to 4 and broadband kernels are weighted sums of monochromatic PSFs,
    mirroring POPPY/WebbPSF practice (Perrin et al. 2012, SPIE 8442).
  * Circular aperture of diameter D: first PSF zero at 1.22 lambda/D and 83.8%
    of the energy inside the first dark ring (Born & Wolf; Schroeder,
    *Astronomical Optics*). Annular apertures follow the two-Bessel amplitude
    ``2 J1(v)/v - eps^2 2 J1(eps v)/(eps v)`` (O'Neill 1956, JOSA 46, 285;
    Mahajan 1982, JOSA 72, 1258): the core shrinks while energy moves to the
    rings — EE(first ring) falls 0.838 -> 0.479 for eps 0 -> 0.5 (Schroeder's
    table, reproduced numerically here to <0.5%).
  * Elliptical apertures scale the Airy pattern per-axis (Fourier scaling
    theorem): image-plane core width along each axis is lambda/D of the
    *conjugate* pupil axis. Rectangular apertures give separable sinc^2 with
    first zeros at lambda/L per axis and 81.5% of the energy in the 2D central
    lobe (Goodman; Shell 2010 AMOS uses exactly this EE-driven framing for SDA
    detection trades).
  * Segmented pupils: a co-phased array factorizes into (grid factor) x
    (single-segment PSF) (Yaitskova, Dohlen & Dierickx 2003, JOSA A 20, 1563).
    Random segment pistons of rms phase sigma (rad) cost ensemble Strehl
    ``S = exp(-sigma^2) + (1 - exp(-sigma^2)) / N`` with an incoherent floor of
    1/N (Chanan & Troy 1999, Appl. Opt. 38, 6642 — exact for Gaussian pistons);
    inter-segment gaps of relative size omega cost ``S = (1 - omega)^4``
    (Yaitskova et al. 2003). An *unphased* array is modeled as the
    single-segment PSF: co-adding N unphased segments gives N x flux at
    single-segment resolution (sqrt(N) SNR in the background limit — the
    cheap-array sweet spot; cf. Argus Array, Law et al. 2022).
  * Wavefront-error coefficients are RMS-normalized (Noll 1976) Zernike
    amplitudes in waves RMS at the weighted mean wavelength, the same
    convention and polynomial set as ``optics.py`` (circular-pupil Noll terms;
    on obscured or odd pupils the RMS reading is approximate).

Everything here is plain numpy in / numpy out — no torch, no scipy — so the
module can serve config-time Scene setup and offline analysis alike. The
geometric ``OpticsModel`` path and this diffraction path are alternative PSF
models; ``pupil_quadrature`` generalizes ``optics.pupil_samples`` to arbitrary
mask shapes for the former.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

ARCSEC_PER_RAD = 180.0 / math.pi * 3600.0  # 206264.8...

# Default mask grid resolution (samples across the full extent). 512 resolves a
# 1.5 m pupil to ~3 mm — far finer than any v2 feature (smallest segment ~0.6 m).
_DEFAULT_N = 512

# Supersampling factor for grey-pixel (antialiased) mask edges: each mask pixel
# is the mean of ss x ss subpixel indicator samples, so pupil area converges as
# O(grid^2) instead of O(grid) (standard grey-pixel practice, cf. POPPY).
_MASK_SUPERSAMPLE = 4


# --------------------------------------------------------------------------- geometry


def aperture_area(shape: str, **params) -> float:
    """Geometric collecting area (m^2) of an aperture — pure geometry, no throughput.

    Args:
        shape: One of ``"circle"``, ``"ellipse"``, ``"rect"``, ``"segmented"``.
        **params: Shape parameters —
            circle: ``d`` (m), optional ``obscuration`` (linear ratio eps; blocked
            area fraction is eps^2);
            ellipse: ``d_major``, ``d_minor`` (m);
            rect: ``width``, ``height`` (m);
            segmented: ``segments`` (list of ``{"x", "y", "d"}`` dicts, m),
            optional ``obscuration_per_segment``.

    Returns:
        Unobstructed geometric area in m^2.
    """
    if shape == "circle":
        eps = params.get("obscuration", 0.0)
        return math.pi / 4.0 * params["d"] ** 2 * (1.0 - eps**2)
    if shape == "ellipse":
        return math.pi / 4.0 * params["d_major"] * params["d_minor"]
    if shape == "rect":
        return params["width"] * params["height"]
    if shape == "segmented":
        eps = params.get("obscuration_per_segment", 0.0)
        return sum(math.pi / 4.0 * s["d"] ** 2 * (1.0 - eps**2) for s in params["segments"])
    raise ValueError(f"unknown aperture shape: {shape!r}")


# --------------------------------------------------------------------------- mask builders


def _subpixel_coords(n: int, extent_m: float) -> np.ndarray:
    """Centers of the n*ss supersampled cells across [-extent/2, extent/2] (m)."""
    m = n * _MASK_SUPERSAMPLE
    return (np.arange(m) + 0.5) * (extent_m / m) - extent_m / 2.0


def _bin_indicator(ind: np.ndarray, n: int) -> np.ndarray:
    """Average an (n*ss, n*ss) boolean indicator down to an (n, n) coverage mask."""
    ss = _MASK_SUPERSAMPLE
    return ind.reshape(n, ss, n, ss).mean(axis=(1, 3))


def circle_mask(
    D: float, obscuration: float = 0.0, n: int = _DEFAULT_N, extent_m: float | None = None
) -> tuple[np.ndarray, dict]:
    """Circular (optionally annular) aperture amplitude mask.

    Args:
        D: Outer diameter (m).
        obscuration: Central obscuration as a linear ratio eps in [0, 1)
            (inner diameter = eps * D), as in ``optics.OpticsConfig``.
        n: Grid size (mask is n x n).
        extent_m: Physical width the grid spans (m). Default: D (tight).

    Returns:
        ``(mask, meta)``: mask is an (n, n) float array in [0, 1] (amplitude
        transmission, grey-pixel antialiased edges; row index = y, column
        index = x); meta holds ``area_m2`` (analytic geometric area),
        ``d_res_m`` (resolution-driving extent = D), ``grid_m`` (per-sample
        spacing), ``extent_m``, ``n``, ``shape`` and the shape parameters.
    """
    if D <= 0:
        raise ValueError(f"D must be positive, got {D}")
    if not 0.0 <= obscuration < 1.0:
        raise ValueError(f"obscuration must be in [0, 1), got {obscuration}")
    extent = float(extent_m) if extent_m is not None else float(D)
    c = _subpixel_coords(n, extent)
    x, y = np.meshgrid(c, c)  # x varies along columns, y along rows
    r2 = x * x + y * y
    ind = r2 <= (D / 2.0) ** 2
    if obscuration > 0:
        ind &= r2 >= (obscuration * D / 2.0) ** 2
    meta = {
        "shape": "circle",
        "d": float(D),
        "obscuration": float(obscuration),
        "area_m2": aperture_area("circle", d=D, obscuration=obscuration),
        "d_res_m": float(D),
        "grid_m": extent / n,
        "extent_m": extent,
        "n": n,
    }
    return _bin_indicator(ind, n), meta


def ellipse_mask(
    d_major: float, d_minor: float, n: int = _DEFAULT_N, extent_m: float | None = None
) -> tuple[np.ndarray, dict]:
    """Elliptical aperture amplitude mask, major axis along x (columns).

    The PSF is the Airy pattern anisotropically scaled per-axis (Fourier scaling
    theorem): narrow along the image axis conjugate to ``d_major``, wide along
    the ``d_minor`` axis, with circular-aperture energy fractions.

    Args:
        d_major: Major-axis diameter (m), along x.
        d_minor: Minor-axis diameter (m), along y.
        n: Grid size.
        extent_m: Physical grid width (m). Default: d_major.

    Returns:
        ``(mask, meta)`` as in :func:`circle_mask`; ``d_res_m`` is the per-axis
        tuple ``(d_major, d_minor)``.
    """
    if d_minor <= 0 or d_major < d_minor:
        raise ValueError(f"need d_major >= d_minor > 0, got ({d_major}, {d_minor})")
    extent = float(extent_m) if extent_m is not None else float(d_major)
    c = _subpixel_coords(n, extent)
    x, y = np.meshgrid(c, c)
    ind = (x / (d_major / 2.0)) ** 2 + (y / (d_minor / 2.0)) ** 2 <= 1.0
    meta = {
        "shape": "ellipse",
        "d_major": float(d_major),
        "d_minor": float(d_minor),
        "area_m2": aperture_area("ellipse", d_major=d_major, d_minor=d_minor),
        "d_res_m": (float(d_major), float(d_minor)),
        "grid_m": extent / n,
        "extent_m": extent,
        "n": n,
    }
    return _bin_indicator(ind, n), meta


def rect_mask(
    width: float, height: float, n: int = _DEFAULT_N, extent_m: float | None = None
) -> tuple[np.ndarray, dict]:
    """Rectangular aperture amplitude mask, ``width`` along x (columns).

    The PSF is separable sinc^2 with first zeros at lambda/width (x axis) and
    lambda/height (y axis) and 81.5% of the energy in the 2D central lobe
    (Goodman, *Introduction to Fourier Optics*).

    Args:
        width: Aperture width along x (m).
        height: Aperture height along y (m).
        n: Grid size.
        extent_m: Physical grid width (m). Default: max(width, height).

    Returns:
        ``(mask, meta)`` as in :func:`circle_mask`; ``d_res_m`` is the per-axis
        tuple ``(width, height)``.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got ({width}, {height})")
    extent = float(extent_m) if extent_m is not None else float(max(width, height))
    c = _subpixel_coords(n, extent)
    x, y = np.meshgrid(c, c)
    ind = (np.abs(x) <= width / 2.0) & (np.abs(y) <= height / 2.0)
    meta = {
        "shape": "rect",
        "width": float(width),
        "height": float(height),
        "area_m2": aperture_area("rect", width=width, height=height),
        "d_res_m": (float(width), float(height)),
        "grid_m": extent / n,
        "extent_m": extent,
        "n": n,
    }
    return _bin_indicator(ind, n), meta


def segmented_mask(
    segments: list[dict],
    obscuration_per_segment: float = 0.0,
    n: int = _DEFAULT_N,
    extent_m: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Amplitude mask for an array of circular segments.

    Args:
        segments: List of ``{"x": m, "y": m, "d": m}`` dicts — segment center
            coordinates (x along columns, y along rows) and diameters.
        obscuration_per_segment: Linear central-obscuration ratio of each
            segment (an array of small on-axis telescopes).
        n: Grid size.
        extent_m: Physical grid width (m). Default: the bounding-circle
            diameter ``2 * max(|r_i| + d_i / 2)``.

    Returns:
        ``(mask, meta)`` as in :func:`circle_mask`. Because the
        resolution-driving extent depends on whether the array is co-phased,
        ``d_res_m`` is a dict: ``{"phased": full-array bounding extent,
        "unphased": first segment's diameter}`` (unphased co-add = N x flux at
        single-segment resolution; Yaitskova et al. 2003, Law et al. 2022).
        ``meta["segments"]`` keeps the input list for per-segment phase errors
        in :func:`psf_kernel`.
    """
    if not segments:
        raise ValueError("segments must be a non-empty list of {x, y, d} dicts")
    if not 0.0 <= obscuration_per_segment < 1.0:
        raise ValueError(f"obscuration_per_segment not in [0, 1): {obscuration_per_segment}")
    span = 2.0 * max(math.hypot(s["x"], s["y"]) + s["d"] / 2.0 for s in segments)
    extent = float(extent_m) if extent_m is not None else span
    c = _subpixel_coords(n, extent)
    x, y = np.meshgrid(c, c)
    ind = np.zeros(x.shape, dtype=bool)
    for s in segments:
        r2 = (x - s["x"]) ** 2 + (y - s["y"]) ** 2
        seg = r2 <= (s["d"] / 2.0) ** 2
        if obscuration_per_segment > 0:
            seg &= r2 >= (obscuration_per_segment * s["d"] / 2.0) ** 2
        ind |= seg
    meta = {
        "shape": "segmented",
        "segments": [dict(s) for s in segments],
        "obscuration_per_segment": float(obscuration_per_segment),
        "area_m2": aperture_area(
            "segmented", segments=segments, obscuration_per_segment=obscuration_per_segment
        ),
        "d_res_m": {"phased": span, "unphased": float(segments[0]["d"])},
        "grid_m": extent / n,
        "extent_m": extent,
        "n": n,
    }
    return _bin_indicator(ind, n), meta


# --------------------------------------------------------------------------- quadrature


def pupil_quadrature(mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """~k quadrature nodes over an arbitrary mask, for the geometric optics path.

    Generalizes ``optics.pupil_samples`` (annulus-only rings) to any mask shape:
    the support is tiled with square cells sized so ~k cells are occupied; each
    occupied cell contributes one node at its mask-weighted centroid with weight
    proportional to its mask mass (interior cells are equal-weight; edge cells
    carry their partial coverage). The cell grid is centered on the array
    center, so for an inversion-symmetric mask the node set is exactly
    inversion-symmetric (every p has its -p) — the same parity property
    ``optics.pupil_samples`` guarantees.

    Coordinates are "unit pupil": pixel positions divided by the array
    half-width, so a :func:`circle_mask` built at its default (tight) extent has
    its rim at ``|p| = 1``, matching the ``optics.py`` convention. Columns are
    ``(px, py)`` like ``optics.pupil_samples``; unlike it, nodes are not
    arranged in closed rings, so a defocused donut's rim renders slightly less
    smoothly at very coarse k.

    Args:
        mask: (n, n) float amplitude mask in [0, 1] (any builder's output).
        k: Requested node count; the actual count tracks it approximately.

    Returns:
        ``(points, weights)``: points is (K', 2) float64 ``(px, py)``; weights
        is (K',) float64 summing to exactly 1.
    """
    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError(f"mask must be square 2D, got shape {mask.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    total = mask.sum()
    if total <= 0:
        raise ValueError("mask has no support")
    n = mask.shape[0]
    iy, ix = np.nonzero(mask > 0)
    support_px = iy.size
    cell = max(1, round(math.sqrt(support_px / k)))
    # Half-integer pixel-center offsets from the array center: exactly odd under
    # inversion, and never on a cell boundary (boundaries are integers).
    ux = ix + 0.5 - n / 2.0
    uy = iy + 0.5 - n / 2.0
    cx = np.floor(ux / cell).astype(np.int64)
    cy = np.floor(uy / cell).astype(np.int64)
    key = (cy - cy.min()) * (cx.max() - cx.min() + 1) + (cx - cx.min())
    _, inv = np.unique(key, return_inverse=True)
    w_px = mask[iy, ix]
    w_cell = np.bincount(inv, weights=w_px)
    half = n / 2.0
    px = np.bincount(inv, weights=w_px * ux) / w_cell / half
    py = np.bincount(inv, weights=w_px * uy) / w_cell / half
    points = np.column_stack([px, py])
    weights = w_cell / total
    return points, weights / weights.sum()


# --------------------------------------------------------------------------- Zernike phase

# RMS-normalized (Noll 1976, JOSA 66, 207) Zernike polynomials on the unit
# pupil — the identical term set and normalization as optics._zernike_grad, so
# a wavefront report feeds either PSF path unchanged. Coefficients in waves RMS.
_WFE_TERMS = ("defocus", "astig_0", "astig_45", "coma_x", "coma_y", "spherical")


def _zernike(term: str, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Noll-normalized Zernike surface Z(p) on normalized pupil coords."""
    r2 = px * px + py * py
    s3, s5, s6, s8 = math.sqrt(3), math.sqrt(5), math.sqrt(6), math.sqrt(8)
    if term == "defocus":  # Z4
        return s3 * (2.0 * r2 - 1.0)
    if term == "astig_0":  # Z6
        return s6 * (px * px - py * py)
    if term == "astig_45":  # Z5
        return s6 * 2.0 * px * py
    if term == "coma_x":  # Z8
        return s8 * (3.0 * r2 - 2.0) * px
    if term == "coma_y":  # Z7
        return s8 * (3.0 * r2 - 2.0) * py
    if term == "spherical":  # Z11
        return s5 * (6.0 * r2 * r2 - 6.0 * r2 + 1.0)
    raise ValueError(f"unknown Zernike term: {term!r} (expected one of {_WFE_TERMS})")


# --------------------------------------------------------------------------- PSF kernel


def _resolution_extent(meta: dict, phased: bool) -> float:
    """Scalar resolution-driving extent d_res (m) for plate-scale purposes."""
    d = meta["d_res_m"]
    if meta["shape"] == "segmented":
        return d["phased"] if phased else d["unphased"]
    if isinstance(d, tuple):
        return max(d)
    return float(d)


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (sigma in samples), reflecting nothing — zero pad."""
    r = max(1, int(math.ceil(4.0 * sigma)))
    ax = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-0.5 * (ax / sigma) ** 2)
    g /= g.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 0, img)
    return np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 1, out)


def psf_kernel(
    mask_info: tuple[np.ndarray, dict],
    wavelengths_nm,
    weights,
    f_number: float,
    pixel_pitch_um: float,
    kernel_px: int = 64,
    wfe: dict | None = None,
    seg_piston_rms_nm: float = 0.0,
    seg_tiptilt_rms_arcsec: float = 0.0,
    phased: bool = True,
    oversample: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Polychromatic diffraction PSF kernel sampled on detector pixels.

    Fourier transform of the complex pupil (matrix Fourier transform, Soummer
    et al. 2007) onto an internally oversampled focal-plane grid, per
    wavelength; weighted sum; binned to the detector pitch; normalized to unit
    sum. The kernel peak is centered on pixel ``[kernel_px // 2] * 2``,
    matching the ``empirical._fft_convolve`` centering convention, so it drops
    into the existing FFT-convolve rendering seam.

    Plate scale: the focal length is taken as ``f_number * d_res`` where d_res
    is the mask's resolution-driving extent (circle D; ellipse d_major; rect
    max(width, height); segmented full-array extent when phased). Consequences:
    the first Airy zero of a filled circle lands at ``1.22 * lambda * f_number``
    on the detector regardless of D, and for an *unphased* segmented array
    ``f_number`` is each segment's working focal ratio (each segment an
    independent telescope of diameter d, images co-added).

    Segmented arrays: when ``phased``, each segment draws a random piston
    (Gaussian, ``seg_piston_rms_nm`` of OPD) and a random image-space tip-tilt
    per axis (Gaussian, ``seg_tiptilt_rms_arcsec``), realizing the Chanan &
    Troy (1999) ensemble Strehl ``exp(-s^2) + (1 - exp(-s^2)) / N``. When
    ``phased=False`` the kernel is that of a SINGLE segment (first segment's
    diameter): a co-add of N unphased segments is N x this kernel's flux at
    single-segment resolution — apply the N x in radiometry, not here. Piston
    is meaningless for a lone segment and is ignored; unphased tip-tilt jitter
    is applied as an equivalent Gaussian blur of the kernel.

    Args:
        mask_info: ``(mask, meta)`` tuple from a mask builder.
        wavelengths_nm: Wavelength sample(s), nm (3-9 samples for broadband;
            Perrin et al. 2012).
        weights: Relative spectral weights (source x QE x filter); normalized
            internally.
        f_number: Working focal ratio (see plate-scale note above).
        pixel_pitch_um: Detector pixel pitch, microns.
        kernel_px: Output kernel size K (K x K).
        wfe: Optional dict of Noll-normalized Zernike coefficients in waves RMS
            at the weighted mean wavelength — keys among ``defocus, astig_0,
            astig_45, coma_x, coma_y, spherical`` (same convention as
            ``optics.AberrationsConfig``).
        seg_piston_rms_nm: Per-segment piston rms, nm of OPD (segmented only).
        seg_tiptilt_rms_arcsec: Per-segment, per-axis tip-tilt rms, arcsec of
            image-space angle (segmented only).
        phased: Segmented arrays: co-phased (full-aperture core) vs unphased
            (single-segment kernel). Ignored for other shapes.
        oversample: Internal focal-plane oversampling factor before binning
            (POPPY/WebbPSF default 4).
        seed: Seed for the per-segment piston/tip-tilt draws.

    Returns:
        (K, K) float64 kernel, non-negative, summing to 1.
    """
    mask, meta = mask_info
    if oversample < 1:
        raise ValueError(f"oversample must be >= 1, got {oversample}")
    if f_number <= 0 or pixel_pitch_um <= 0 or kernel_px < 3:
        raise ValueError("f_number and pixel_pitch_um must be positive, kernel_px >= 3")
    wavelengths = np.atleast_1d(np.asarray(wavelengths_nm, dtype=np.float64))
    w = np.atleast_1d(np.asarray(weights, dtype=np.float64))
    if wavelengths.shape != w.shape:
        raise ValueError("wavelengths_nm and weights must have matching shapes")
    if np.any(wavelengths <= 0) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("wavelengths must be positive and weights non-negative, not all zero")
    w = w / w.sum()

    segmented = meta["shape"] == "segmented"
    unphased_tt_blur = 0.0
    if segmented and not phased:
        # Single-segment kernel: piston across ONE segment is a global phase
        # (ignored); tip-tilt jitter becomes an image-plane Gaussian blur.
        unphased_tt_blur = float(seg_tiptilt_rms_arcsec)
        seg = meta["segments"][0]
        mask, meta = circle_mask(seg["d"], meta["obscuration_per_segment"], n=meta["n"])
        segmented = False

    n = meta["n"]
    grid_m = meta["grid_m"]
    d_res = _resolution_extent(meta, phased)
    f_m = f_number * d_res
    lam_ref_m = float((wavelengths * w).sum()) * 1e-9

    # Pupil-plane sample coordinates (m), x along columns / y along rows.
    xi = (np.arange(n) + 0.5) * grid_m - meta["extent_m"] / 2.0
    xx, yy = np.meshgrid(xi, xi)

    # ---- OPD map (meters) ----------------------------------------------------
    opd = np.zeros((n, n), dtype=np.float64)
    if wfe:
        unknown = set(wfe) - set(_WFE_TERMS)
        if unknown:
            raise ValueError(f"unknown wfe terms: {sorted(unknown)} (expected among {_WFE_TERMS})")
        half = d_res / 2.0
        pxn, pyn = xx / half, yy / half
        for term, coeff in wfe.items():
            if coeff:
                opd += coeff * lam_ref_m * _zernike(term, pxn, pyn)
    if segmented and (seg_piston_rms_nm > 0 or seg_tiptilt_rms_arcsec > 0):
        rng = np.random.default_rng(seed)
        for s in meta["segments"]:
            member = (xx - s["x"]) ** 2 + (yy - s["y"]) ** 2 <= (s["d"] / 2.0) ** 2
            piston = rng.normal(0.0, seg_piston_rms_nm) * 1e-9
            # Image-space tilt theta displaces the segment PSF by theta; the
            # corresponding OPD ramp across the segment is theta * (xi - xi_c).
            tx, ty = rng.normal(0.0, seg_tiptilt_rms_arcsec / ARCSEC_PER_RAD, size=2)
            opd[member] += piston + tx * (xx[member] - s["x"]) + ty * (yy[member] - s["y"])

    # ---- MFT per wavelength, accumulate on the oversampled detector grid -----
    K, os_ = kernel_px, oversample
    S = K * os_
    pitch_m = pixel_pitch_um * 1e-6
    dx = pitch_m / os_
    # Focal-plane sample centers (m); PSF center on detector pixel K // 2.
    u = ((np.arange(S) + 0.5) / os_ - 0.5 - K // 2) * pitch_m
    lam_min_m = wavelengths.min() * 1e-9
    if dx > lam_min_m * f_number / 2.0:
        warnings.warn(
            f"internal focal-plane sampling ({dx * 1e6:.2f} um) is coarser than "
            f"lambda*F#/2 ({lam_min_m * f_number / 2e-6:.2f} um) — raise `oversample` "
            "to Nyquist-sample the diffraction core (Perrin et al. 2012 convention).",
            stacklevel=2,
        )
    mask_energy = (mask**2).sum()
    acc = np.zeros((S, S), dtype=np.float64)
    for lam_nm, wt in zip(wavelengths, w):
        lam_m = lam_nm * 1e-9
        pupil = mask * np.exp(2j * math.pi * opd / lam_m)
        # MFT: E[j, k] = sum_ab P[a, b] exp(-2 pi i (u_j y_a + u_k x_b) / (lam f))
        A = np.exp(-2j * math.pi * np.outer(u, xi) / (lam_m * f_m))
        E = A @ pupil @ A.T
        # Parseval normalization: each monochromatic PSF integrates to 1 over
        # the INFINITE focal plane, so acc.sum() is the true captured fraction.
        acc += wt * np.abs(E) ** 2 * (grid_m * dx / (lam_m * f_m)) ** 2 / mask_energy

    if unphased_tt_blur > 0:
        sigma_samples = unphased_tt_blur / ARCSEC_PER_RAD * f_m / dx
        if sigma_samples > 0.1:
            acc = _gaussian_blur(acc, sigma_samples)

    captured = acc.sum()
    if captured < 0.5:
        warnings.warn(
            f"PSF kernel window captures only {captured:.1%} of the energy — "
            "increase kernel_px or pixel_pitch_um.",
            stacklevel=2,
        )
    kern = acc.reshape(K, os_, K, os_).sum(axis=(1, 3))
    return kern / kern.sum()


# --------------------------------------------------------------------------- analytic helpers


def _bessel_j(order: int, x: np.ndarray) -> np.ndarray:
    """J_n(x) by the integral representation (1/pi) int_0^pi cos(n t - x sin t) dt.

    The integrand's derivative vanishes at both endpoints for integer n, so the
    trapezoid rule converges spectrally — ~1e-12 accurate for |x| < 50 at 801
    nodes. Keeps the module scipy-free.
    """
    x = np.asarray(x, dtype=np.float64)
    theta = np.linspace(0.0, math.pi, 801)
    integrand = np.cos(order * theta[None, :] - x[..., None] * np.sin(theta)[None, :])
    return np.trapezoid(integrand, theta, axis=-1) / math.pi


def _annular_amplitude(v: np.ndarray, eps: float) -> np.ndarray:
    """Normalized annular-pupil PSF amplitude a(v), a(0) = 1 (O'Neill 1956).

    v is the reduced radius pi * D * theta / lambda (first circular zero at
    v = 3.8317, i.e. theta = 1.22 lambda / D).
    """
    v = np.asarray(v, dtype=np.float64)
    a = 2.0 * _bessel_j(1, v) / v
    if eps > 0:
        a = a - eps**2 * 2.0 * _bessel_j(1, eps * v) / (eps * v)
    return a / (1.0 - eps**2)


def annular_first_zero_lamD(eps: float = 0.0) -> float:
    """Radius of the first PSF dark ring in lambda/D units for obscuration eps.

    1.2197 at eps = 0 (the Airy 1.22 lambda/D), shrinking to 1.0009 at
    eps = 0.5 (O'Neill 1956; Mahajan 1982; recon-verified table).

    Args:
        eps: Linear central-obscuration ratio in [0, ~0.9].

    Returns:
        First-zero radius in units of lambda/D (multiply by lambda/D for
        radians).
    """
    if not 0.0 <= eps < 0.95:
        raise ValueError(f"eps must be in [0, 0.95), got {eps}")
    v = np.arange(0.5, 8.0, 0.005)
    a = _annular_amplitude(v, eps)
    idx = int(np.nonzero(np.sign(a[:-1]) != np.sign(a[1:]))[0][0])
    lo, hi = float(v[idx]), float(v[idx + 1])
    s_lo = math.copysign(1.0, float(a[idx]))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if math.copysign(1.0, float(_annular_amplitude(np.array([mid]), eps)[0])) == s_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) / math.pi


def airy_first_zero_arcsec(wavelength_nm: float, D_m: float) -> float:
    """Angular radius of the first Airy dark ring, 1.22 lambda / D (Born & Wolf).

    Args:
        wavelength_nm: Wavelength, nm.
        D_m: Aperture diameter, m.

    Returns:
        First-zero radius, arcsec.
    """
    if wavelength_nm <= 0 or D_m <= 0:
        raise ValueError("wavelength_nm and D_m must be positive")
    return 1.2196698912665045 * wavelength_nm * 1e-9 / D_m * ARCSEC_PER_RAD


def annular_ee_first_ring(eps: float = 0.0) -> float:
    """Encircled energy inside the first dark ring of an annular aperture.

    0.838 for a filled circle, dropping to 0.654 at eps = 0.33 and 0.479 at
    eps = 0.5 as obscuration pushes energy into the rings (Schroeder,
    *Astronomical Optics*, obscuration table; reproduced numerically here from
    the O'Neill two-Bessel amplitude, agreement < 0.5%). Total energy comes
    from Parseval, so no infinite tail integration is needed.

    Args:
        eps: Linear central-obscuration ratio.

    Returns:
        Fraction of total PSF energy inside the first dark ring.
    """
    v0 = annular_first_zero_lamD(eps) * math.pi
    v = np.linspace(1e-6, v0, 4001)
    a = _annular_amplitude(v, eps)
    # EE(v0) = (1 - eps^2)/2 * int_0^v0 a^2 v dv (Parseval: int_0^inf = 2/(1-eps^2)).
    return float((1.0 - eps**2) / 2.0 * np.trapezoid(a * a * v, v))


def piston_strehl(sigma_rad: float, n_segments: int) -> float:
    """Ensemble Strehl of a segmented pupil with random piston errors.

    ``S = exp(-sigma^2) + (1 - exp(-sigma^2)) / N`` — exact for independent
    Gaussian segment pistons of rms phase sigma; floor 1/N for fully unphased
    segments (the coherent core dies, an incoherent single-segment halo
    survives). Chanan & Troy 1999, Appl. Opt. 38, 6642.

    Args:
        sigma_rad: Per-segment piston rms, radians of phase
            (= 2 pi * OPD_rms / lambda).
        n_segments: Number of segments N.

    Returns:
        Expected Strehl ratio.
    """
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1, got {n_segments}")
    e = math.exp(-sigma_rad * sigma_rad)
    return e + (1.0 - e) / n_segments


def gap_strehl(omega: float) -> float:
    """Strehl loss from inter-segment gaps: ``S = (1 - omega)^4``.

    omega is the gap width / segment pitch; the loss is exactly the square of
    the blocked-area fraction (Yaitskova, Dohlen & Dierickx 2003, JOSA A 20,
    1563 — percent-level gaps cost percent-level Strehl: omega = 0.005 gives
    S = 0.980, omega = 0.02 gives S = 0.922).

    Args:
        omega: Relative gap size (gap / pitch), in [0, 1).

    Returns:
        Strehl ratio.
    """
    if not 0.0 <= omega < 1.0:
        raise ValueError(f"omega must be in [0, 1), got {omega}")
    return (1.0 - omega) ** 4


def ensquared_energy(kernel: np.ndarray, half_width_px: int) -> float:
    """Fraction of kernel energy in a (2h+1) x (2h+1) box centered on the peak.

    The detection-mode figure of merit: SDA SNR depends on energy-in-pixel and
    background-per-pixel, not PSF shape (Shell 2010, AMOS).

    Args:
        kernel: (K, K) PSF kernel (any normalization).
        half_width_px: Box half-width h in pixels (h = 0 is the peak pixel).

    Returns:
        Ensquared energy fraction relative to the kernel's total.
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    if half_width_px < 0:
        raise ValueError(f"half_width_px must be >= 0, got {half_width_px}")
    r, c = np.unravel_index(int(np.argmax(kernel)), kernel.shape)
    h = int(half_width_px)
    block = kernel[max(0, r - h) : r + h + 1, max(0, c - h) : c + h + 1]
    return float(block.sum() / kernel.sum())
