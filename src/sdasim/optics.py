"""Geometric pupil-sampling defocus + aberration model.

Turns each point source into K sub-sources whose image-plane offsets are the
transverse ray aberrations of K fixed samples of the (annular) pupil — the
same flux-conserving expansion trick ``expand_motion()`` uses for motion blur,
applied over pupil area instead of time. Each sub-source keeps the in-focus
``psf_sigma``, so the splatting hot path is unchanged and stays FFT-free and
differentiable (offsets are linear in the commanded defocus and in every
aberration coefficient).

Physics (geometric optics; Mahajan conventions):

  * A wavefront error W(p) over normalized pupil coords p (|p| in [eps, 1],
    eps = obscuration ratio) displaces the ray from pupil point p by the
    transverse aberration  ``delta_r = 2 * N * grad_p W(p)``  in the image
    plane, where N is the working f-number and W is in length units.
  * A detector displaced ``defocus_um`` from best focus adds
    ``W020 = defocus_um / (8 N^2)`` of ``|p|^2`` wavefront error, giving the
    classic blur-circle radius ``defocus_um / (2 N)``. The defocused image of
    the annular pupil is a uniform annulus — the "donut", hole ratio eps.
  * Static aberrations are RMS-normalized (Noll) Zernike coefficients in
    waves RMS — the numbers an interferometer or a Zemax "Zernike Standard"
    analysis reports. 1 wave RMS of Z4 corresponds to a focal shift of
    ``16 * sqrt(3) * N^2 * lambda``. Coefficients are CIRCULAR-pupil Noll:
    on an obscured pupil the RMS reading is approximate (Mahajan annular
    Zernikes would be exact).
  * Field dependence follows the third-order wave-aberration expansion. The
    plain coefficients (defocus, astig_0/45, coma_x/y, spherical) are
    field-CONSTANT (miscollimation/design-residual signatures). The field
    terms scale the same pupil polynomials with per-source coefficient laws:
    ``field_curvature`` adds Z4 growing as |h|^2; ``coma_radial`` drives the
    (Z8, Z7) pair linearly along the field vector h; ``astig_radial`` drives
    (Z6, Z5) as the SQUARE of the field vector (doubled-angle rule — the astig
    axis tracks the field azimuth). Constant + quadratic astig yields binodal
    astigmatism. h is corner-normalized: each field term's value is its waves
    RMS at the field corner.

Conventions:

  * Pupil x maps to the detector column axis, pupil y to the row axis.
    ``astig_0`` (Z6) elongates along rows/columns; ``astig_45`` (Z5) along
    the diagonals. The astigmatic major axis flips 90 deg across best focus.
  * ``defocus_um > 0`` means the detector is beyond best focus (extra-focal).
    For a symmetric pupil the model is exactly blind to the sign of pure
    defocus (and of coma — a parity fact that matches real star tests);
    astigmatism and spherical aberration do distinguish the two sides.
  * The pupil sample set is exactly inversion-symmetric (every p has its -p),
    so that parity is exact rather than approximate.

Validity: this is ray optics regularized by the in-focus Gaussian. It is
faithful once the geometric blur dominates diffraction (W020 above ~1 wave);
near best focus the Gaussian ``psf_sigma`` floor stands in for the Airy core,
which is exactly the fidelity an autofocus V-curve needs — HFD follows
``sqrt(HFD0^2 + (k * defocus)^2)`` with ``k = sqrt((1 + eps^2) / 2) / N``.
Diffraction detail (Fresnel rings, Arago spot) is not modeled.
"""

from __future__ import annotations

import math
import time
import warnings

import torch
from torch import Tensor

from sdasim.config import OpticsConfig

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

# Order of the static aberration terms in the gradient basis (see _zernike_grad).
_STATIC_TERMS = ("astig_0", "astig_45", "coma_x", "coma_y", "spherical")

# max |grad Z| over the annulus per term (rim-dominated for every eps), for
# the worst-case blur-radius bound.
_GRAD_MAX = {
    "astig_0": 2 * math.sqrt(6),
    "astig_45": 2 * math.sqrt(6),
    "coma_x": 7 * math.sqrt(8),
    "coma_y": 7 * math.sqrt(8),
    "spherical": 12 * math.sqrt(5),
}


def pupil_samples(k: int, obscuration: float, device=None) -> tuple[Tensor, Tensor]:
    """~K deterministic annular-pupil quadrature nodes: (points (K', 2), weights (K',)).

    Concentric equal-width rings; each ring carries an even point count at a
    golden-angle phase offset from its neighbors. The outermost ring is a
    closed, evenly spaced circle, so a defocused donut's rim renders smoothly —
    a spiral's ragged boundary would scallop it into "iris blade" lumps. Even
    per-ring counts make the set exactly inversion-symmetric (every point's
    antipode is present), which keeps the model's parity properties exact.

    Each ring's node radius is its RMS radius and its weight is its exact area
    fraction (split evenly among its points), so the set integrates constants
    and rho^2 over the pupil exactly: flux is conserved at any K, and blur
    second moments are exact for offsets linear in p (defocus, astigmatism;
    coma/spherical moments converge with K). RMS-node placement puts the
    outermost node ~dr/2 inside the geometric rim, so a donut's outer edge
    reads a few percent small at coarse rungs (the deficit shrinks as K
    grows). Weights sum to 1. The actual count K' tracks the request at large
    K but floors at the 2-ring minimum (which grows with obscuration: ~14 at
    eps=0, ~50 at eps=0.6). Point columns are (px, py).

    The float64 quadrature is done on CPU (MPS has no float64) and only the
    float32 result moves to ``device``.
    """
    eps = obscuration
    area = math.pi * (1.0 - eps * eps)
    dr = math.sqrt(area / max(k, 4))
    # At least 2 rings: a single ring renders a blur DISC as a hollow circle
    # (flux stays exact, but the shape is wrong at coarse rungs).
    n_rings = max(2, round((1.0 - eps) / dr))
    dr = (1.0 - eps) / n_rings
    pts, wts = [], []
    for j in range(n_rings):
        lo = eps + j * dr
        hi = lo + dr
        rho = math.sqrt((lo * lo + hi * hi) / 2.0)  # RMS node: exact second moment
        m = max(2, 2 * round(math.pi * rho / dr))  # even count, ~dr azimuthal spacing
        theta = j * _GOLDEN_ANGLE + 2.0 * math.pi * torch.arange(m, dtype=torch.float64) / m
        pts.append(rho * torch.stack([torch.cos(theta), torch.sin(theta)], dim=1))
        ring_area = math.pi * (hi * hi - lo * lo)
        wts.append(torch.full((m,), ring_area / (area * m), dtype=torch.float64))
    p = torch.cat(pts, dim=0)
    w = torch.cat(wts, dim=0)
    return (
        p.to(device=device, dtype=torch.float32),
        (w / w.sum()).to(device=device, dtype=torch.float32),
    )


def _zernike_grad(term: str, p: Tensor) -> Tensor:
    """Gradient (d/dpx, d/dpy) of an RMS-normalized (Noll) Zernike term at p.

    ``p`` is (K, 2) normalized pupil coords; returns (K, 2). Coefficients that
    multiply these gradients are wavefront error in length units, so the
    coefficient equals the term's RMS contribution (Noll normalization).
    Constant/piston parts vanish under the gradient.
    """
    px, py = p[:, 0], p[:, 1]
    r2 = px * px + py * py
    s3, s5, s6, s8 = math.sqrt(3), math.sqrt(5), math.sqrt(6), math.sqrt(8)
    if term == "defocus":  # Z4 = sqrt(3)(2 rho^2 - 1)
        return 4.0 * s3 * p
    if term == "astig_0":  # Z6 = sqrt(6)(px^2 - py^2)
        return 2.0 * s6 * torch.stack([px, -py], dim=1)
    if term == "astig_45":  # Z5 = sqrt(6) * 2 px py
        return 2.0 * s6 * torch.stack([py, px], dim=1)
    if term == "coma_x":  # Z8 = sqrt(8)(3 rho^2 - 2) px
        return s8 * torch.stack([9 * px * px + 3 * py * py - 2, 6 * px * py], dim=1)
    if term == "coma_y":  # Z7 = sqrt(8)(3 rho^2 - 2) py
        return s8 * torch.stack([6 * px * py, 3 * px * px + 9 * py * py - 2], dim=1)
    if term == "spherical":  # Z11 = sqrt(5)(6 rho^4 - 6 rho^2 + 1)
        return s5 * ((24.0 * r2 - 12.0).unsqueeze(1) * p)
    raise ValueError(f"unknown Zernike term: {term}")


def _round_sig(x: float, digits: int) -> int:
    """Round to `digits` significant figures — keeps a timed budget from
    jittering the chosen sampling between otherwise identical runs."""
    if x <= 0:
        return 1
    mag = 10 ** (math.floor(math.log10(x)) - digits + 1)
    return max(1, int(round(x / mag) * mag))


def _sync(device) -> None:
    """Block until queued device work finishes, so timings are real."""
    if device is not None and getattr(device, "type", None) == "mps":
        torch.mps.synchronize()
    elif device is not None and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize()


def _measure_splat_rate(height: int, width: int, sigma: float, device, n: int = 20_000) -> float:
    """Sub-sources splatted per second for this scene on this machine.

    Splat cost is very nearly linear in sub-source count above a small fixed
    overhead, and the constant depends on the footprint (psf_sigma), the image
    size (scatter locality) and the device — all of which this measures
    directly rather than guessing. Costs ~10-40 ms, once, at Scene setup.
    """
    from sdasim.splat import splat_gaussians

    radius = max(1, math.ceil(3.0 * sigma))
    # Local generator: the benchmark must not consume the caller's seeded RNG
    # stream (it would shift every downstream noise draw when optics is on).
    gen = torch.Generator().manual_seed(0)
    scale = torch.tensor([float(height), float(width)])
    pos = (torch.rand(n, 2, generator=gen) * scale).to(device)
    ints = torch.ones(n, device=device)
    # Warm up first: lazy device init and kernel compilation would otherwise be
    # charged to the measurement.
    splat_gaussians(height, width, pos[:512], ints[:512], sigma, radius)
    _sync(device)
    t0 = time.perf_counter()
    splat_gaussians(height, width, pos, ints, sigma, radius)
    _sync(device)
    return n / max(time.perf_counter() - t0, 1e-6)


class OpticsModel:
    """Precomputed pupil geometry for ``expand()`` (built once per Scene).

    Precomputes the full ladder of pupil sample sets (K = 4, 8, 16, ... 4096;
    well under a megabyte in total, so there is nothing to gain by sizing it smaller) and,
    per rung, the image-plane offsets of the static non-defocus aberrations
    plus the defocus gradient direction. ``expand()`` then only scales/adds
    precomputed tensors, picks the smallest rung that keeps sub-source spacing
    below ~2*psf_sigma (no ring artifacts), and skips entirely when the total
    blur is negligible — so an in-focus, aberration-free frame costs nothing.

    The cost ceiling is expressed in seconds (``max_render_seconds``): this
    machine's splat throughput is measured once here and converted into a
    sub-source cap, so the same config behaves sanely on a laptop and a GPU
    box. Set ``max_sub_sources`` explicitly to pin the cap instead.
    """

    # Passthrough when total blur radius is below this many psf_sigma.
    SKIP_BLUR_SIGMA = 0.3

    def __init__(
        self,
        cfg: OpticsConfig,
        psf_sigma: float,
        height: int,
        width: int,
        device=None,
        budget_sub_sources: int | None = None,
        budget_seconds: float | None = None,
        splat_rate: float | None = None,
    ):
        if not 0.0 <= cfg.obscuration < 1.0:
            raise ValueError(f"obscuration must be in [0, 1), got {cfg.obscuration}")
        if cfg.f_number <= 0 or cfg.pixel_pitch_um <= 0 or cfg.wavelength_nm <= 0:
            raise ValueError("f_number, pixel_pitch_um and wavelength_nm must be positive")
        self.cfg = cfg
        self.sigma = float(psf_sigma)
        n = cfg.f_number
        lam_um = cfg.wavelength_nm * 1e-3
        # Displacement scale: wavefront-error microns -> image-plane pixels.
        self._px_per_um = 2.0 * n / cfg.pixel_pitch_um
        # Commanded focal-shift microns -> defocus (Z4) coefficient in microns.
        self._c4_per_defocus_um = 1.0 / (8.0 * n * n * 2.0 * math.sqrt(3.0))
        # Static coefficients in microns of wavefront error.
        ab = cfg.aberrations
        self._c4_static_um = ab.defocus * lam_um
        self._fc_um = ab.field_curvature * lam_um
        self._cr_um = ab.coma_radial * lam_um
        self._ar_um = ab.astig_radial * lam_um
        self._static_terms_um = [getattr(ab, t) * lam_um for t in _STATIC_TERMS]
        # Field radius normalization: the field terms (field_curvature,
        # coma_radial, astig_radial) are the values at the field corner.
        self._corner_r2 = (height / 2.0) ** 2 + (width / 2.0) ** 2

        # Full rung ladder: every power of two up to the sampling ceiling. The
        # whole ladder is ~163 KB, so there is no reason to size it from the
        # expected defocus and risk silently under-sampling a larger command.
        self._rungs: list[dict] = []
        k = 4
        while True:
            p, w = pupil_samples(k, cfg.obscuration, device=device)
            grad4 = _zernike_grad("defocus", p) * self._px_per_um  # px per micron of c4
            static = torch.zeros_like(p)
            for c_um, term in zip(self._static_terms_um, _STATIC_TERMS):
                if c_um != 0.0:
                    static = static + c_um * _zernike_grad(term, p) * self._px_per_um
            # (px, py) gradient components -> (row, col) image offsets. Ring
            # quantization means small requests collapse onto the same actual
            # count (4/8/16/32 all yield the 2-ring minimum) — keep one of each.
            if not self._rungs or p.shape[0] > self._rungs[-1]["k"]:
                rung = {"k": p.shape[0], "w": w, "grad4": grad4.flip(1), "static": static.flip(1)}
                if self._cr_um != 0.0:
                    # Field-linear coma needs the per-source coma bases too.
                    rung["grad8"] = (_zernike_grad("coma_x", p) * self._px_per_um).flip(1)
                    rung["grad7"] = (_zernike_grad("coma_y", p) * self._px_per_um).flip(1)
                if self._ar_um != 0.0:
                    # Field-quadratic astigmatism needs the astig bases.
                    rung["grad6"] = (_zernike_grad("astig_0", p) * self._px_per_um).flip(1)
                    rung["grad5"] = (_zernike_grad("astig_45", p) * self._px_per_um).flip(1)
                self._rungs.append(rung)
            if k >= 4096:
                break
            k *= 2

        # Cost ceiling: an explicit count wins; then a caller-shared budget
        # (Scene resolves the effective seconds and measures splat throughput
        # once for motion AND defocus); otherwise calibrate this machine
        # ourselves (standalone use). budget_seconds <= 0 disables the cap.
        if budget_seconds is None:
            budget_seconds = cfg.max_render_seconds if cfg.max_render_seconds is not None else 0.5
        self._budget_seconds = budget_seconds
        self.max_sub_sources: int | None
        if cfg.max_sub_sources is not None:
            self.max_sub_sources = int(cfg.max_sub_sources)
            self.splat_rate = None
        elif budget_seconds <= 0:
            self.max_sub_sources = None  # unlimited
            self.splat_rate = splat_rate
        elif budget_sub_sources is not None:
            self.max_sub_sources = int(budget_sub_sources)
            self.splat_rate = splat_rate
        else:
            self.splat_rate = _measure_splat_rate(height, width, self.sigma, device)
            self.max_sub_sources = _round_sig(self._budget_seconds * self.splat_rate, 2)

        # Set by each expand() call; read by Scene for render metadata.
        self.last_k: int = 1
        self.last_blur_px: float = 0.0
        self.last_clamped: bool = False
        self._warned_clamp = False

    def _warn_clamp(self, m: int, k_needed: int, blur_px: float, over_budget: bool) -> None:
        if self._warned_clamp:
            return
        self._warned_clamp = True
        what = f"rendering coarser (grainy donut, {self.last_k} samples)"
        if over_budget:
            what += " and exceeding the budget rather than dropping the defocus"
        # Two different constraints can bite; naming the wrong one sends the
        # reader to the wrong knob. The ladder is the binding one only when we
        # already handed back its top rung.
        if self.last_k == self._rungs[-1]["k"]:
            cause = (
                f"pupil sampling is capped at {self._rungs[-1]['k']} samples — this blur is "
                f"past what the model resolves; reduce the commanded defocus"
            )
        elif self.splat_rate is None:
            cause = (
                f"optics.max_sub_sources={self.max_sub_sources} caps sources x samples "
                f"— raise it or reduce defocus/star count"
            )
        else:
            cause = (
                f"max_render_seconds={self._budget_seconds:g} allows "
                f"{self.max_sub_sources:,} sub-sources at this machine's measured "
                f"{self.splat_rate / 1e6:.1f}M/s — raise the budget or reduce "
                f"defocus/star count"
            )
        warnings.warn(
            f"defocus blur of {blur_px:.0f} px across {m} sources wants {k_needed} "
            f"pupil samples; {cause}. {what.capitalize()} rather than allocating "
            f"unbounded memory/time.",
            stacklevel=3,
        )

    def blur_radius_px(self, defocus_um: float = 0.0) -> float:
        """Worst-case geometric blur radius (px) at a commanded focal shift.

        Includes the static aberrations and all field terms at their corner
        (worst-case) values. Useful to size pads/margins for sources just
        outside the frame whose defocused flux still lands in it.
        """
        c4 = abs(self._c4_static_um + defocus_um * self._c4_per_defocus_um) + abs(self._fc_um)
        return self._blur_radius_px(c4)

    def _blur_radius_px(self, c4_um: float) -> float:
        """Upper-bound blur radius (px) for a defocus coefficient plus statics."""
        r = abs(c4_um) * 4 * math.sqrt(3)
        for c_um, term in zip(self._static_terms_um, _STATIC_TERMS):
            r += abs(c_um) * _GRAD_MAX[term]
        r += abs(self._cr_um) * 7 * math.sqrt(8)  # field-linear coma, worst at the corner
        r += abs(self._ar_um) * 2 * math.sqrt(6)  # field-quadratic astig, worst at the corner
        return r * self._px_per_um

    def _k_needed(self, blur_px: float, sigma: float | None = None) -> int:
        """Samples needed so image-plane spacing stays <= 2*sigma (no rings)."""
        s = self.sigma if sigma is None else sigma
        e2 = self.cfg.obscuration**2
        return max(4, math.ceil(math.pi * (1.0 - e2) * blur_px**2 / (4.0 * s**2)))

    def expand(
        self,
        positions: Tensor,
        intensities: Tensor,
        defocus_um: float | Tensor,
        center: tuple[float, float],
        psf_sigma: float | Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Expand (M, 2)/(M,) sources into (M*K, 2)/(M*K,) defocused sub-sources.

        ``defocus_um`` may be a scalar tensor to keep gradients to the focus
        command; when it carries gradients the negligible-blur passthrough is
        disabled so the focus command always stays in the autograd graph.
        ``psf_sigma`` should be the sigma the sub-sources will be splatted with
        when it differs from the construction-time value — the passthrough gate
        and sampling-density (rung) choice depend on it. Flux is conserved
        exactly (sub-sources carry the pupil quadrature weights, which sum
        to 1). Returns the inputs unchanged when nothing needs expanding.
        """
        if isinstance(defocus_um, Tensor):
            defocus_um = defocus_um.to(dtype=positions.dtype, device=positions.device)
            d_um = float(defocus_um.detach().reshape(()).item())
            needs_grad = defocus_um.requires_grad and torch.is_grad_enabled()
        else:
            d_um = float(defocus_um)
            needs_grad = False
        if psf_sigma is None:
            sigma = self.sigma
        elif isinstance(psf_sigma, Tensor):
            sigma = float(psf_sigma.detach().reshape(()).item())
        else:
            sigma = float(psf_sigma)

        blur_px = self.blur_radius_px(d_um)
        self.last_blur_px = blur_px
        if blur_px < self.SKIP_BLUR_SIGMA * sigma and not needs_grad:
            self.last_k = 1
            self.last_clamped = False
            return positions, intensities

        m = positions.shape[0]
        k_needed = self._k_needed(blur_px, sigma)
        # Memory/time safety rail: cap sources * pupil_samples. Fast optics +
        # large defocus + motion expansion multiply into tens of millions of
        # splats otherwise. When the cap bites we coarsen the pupil sampling
        # (grainy donuts; flux stays exact) rather than blow up.
        if self.max_sub_sources is None:
            k_budget = self._rungs[-1]["k"]  # budget disabled
        else:
            k_budget = max(1, self.max_sub_sources // max(m, 1))
        allowed = [r for r in self._rungs if r["k"] <= k_budget]
        over_budget = not allowed
        if over_budget:
            # Even the coarsest sample set exceeds the budget. The budget is a
            # cost governor, NOT a physics switch: commanded defocus must never
            # silently vanish (and the threshold is a machine-load-dependent
            # calibration, so a skip here would flip nondeterministically
            # between runs). Render the coarsest sampling, exceed the time
            # target, and say so; peak memory stays bounded by the chunked
            # splat, and Scene relaxes motion sampling under heavy blur so
            # this path stays rare.
            allowed = [self._rungs[0]]
        rung = next((r for r in allowed if r["k"] >= k_needed), allowed[-1])
        self.last_k = rung["k"]
        self.last_clamped = over_budget or rung["k"] < k_needed
        if self.last_clamped and m > 0:
            self._warn_clamp(m, k_needed, blur_px, over_budget=over_budget)

        if m == 0:
            return positions, intensities

        # Per-source defocus coefficient (microns): static + commanded (+ field
        # curvature, quadratic in boresight-relative field radius).
        c4_um = self._c4_static_um + defocus_um * self._c4_per_defocus_um
        rel = None
        if self._fc_um != 0.0 or self._cr_um != 0.0 or self._ar_um != 0.0:
            rel = positions - positions.new_tensor(center)
        if self._fc_um != 0.0:
            r2 = (rel * rel).sum(dim=1)
            c4_um = c4_um + self._fc_um * (r2 / self._corner_r2)
        c4_um = torch.as_tensor(c4_um, dtype=positions.dtype, device=positions.device)
        c4_um = c4_um.reshape(-1, 1, 1)  # (M or 1, 1, 1)

        offsets = rung["static"].unsqueeze(0) + c4_um * rung["grad4"].unsqueeze(0)
        if self._cr_um != 0.0 or self._ar_um != 0.0:
            r_corner = math.sqrt(self._corner_r2)
            hx = rel[:, 1] / r_corner  # field vector, corner-normalized (x = col)
            hy = rel[:, 0] / r_corner  # (y = row)
        if self._cr_um != 0.0:
            # Field-linear coma: the coma vector tracks the field vector — the
            # Z8 (x/col) component scales with hx and Z7 (y/row) with hy.
            # Positive coma_radial points tails away from boresight.
            c8 = (self._cr_um * hx).reshape(-1, 1, 1)
            c7 = (self._cr_um * hy).reshape(-1, 1, 1)
            offsets = offsets + c8 * rung["grad8"].unsqueeze(0) + c7 * rung["grad7"].unsqueeze(0)
        if self._ar_um != 0.0:
            # Field-quadratic astigmatism with the doubled-angle rule: the astig
            # pair transforms as the SQUARE of the field vector, (c6 + i*c5) =
            # A * (hx + i*hy)^2 — magnitude grows as |h|^2 and the axis tracks
            # the field azimuth (radial/tangential line foci across focus).
            c6 = (self._ar_um * (hx * hx - hy * hy)).reshape(-1, 1, 1)
            c5 = (self._ar_um * 2.0 * hx * hy).reshape(-1, 1, 1)
            offsets = offsets + c6 * rung["grad6"].unsqueeze(0) + c5 * rung["grad5"].unsqueeze(0)
        expanded_pos = (positions.unsqueeze(1) + offsets).reshape(-1, 2)
        expanded_int = (intensities.unsqueeze(1) * rung["w"].unsqueeze(0)).reshape(-1)
        return expanded_pos, expanded_int
