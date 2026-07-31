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
    ``16 * sqrt(3) * N^2 * lambda``.

Conventions:

  * Pupil x maps to the detector column axis, pupil y to the row axis.
    ``astig_0`` (Z6) elongates along rows/columns; ``astig_45`` (Z5) along
    the diagonals. The astigmatic major axis flips 90 deg across best focus.
  * ``defocus_um > 0`` means the detector is beyond best focus (extra-focal).
    For a symmetric pupil the model is exactly blind to the sign of pure
    defocus (and of coma — a parity fact that matches real star tests);
    astigmatism and spherical aberration do distinguish the two sides.
  * Pupil samples come in antipodal pairs (p and -p), so that parity is exact
    rather than approximate.

Validity: this is ray optics regularized by the in-focus Gaussian. It is
faithful once the geometric blur dominates diffraction (W020 above ~1 wave);
near best focus the Gaussian ``psf_sigma`` floor stands in for the Airy core,
which is exactly the fidelity an autofocus V-curve needs — HFD follows
``sqrt(HFD0^2 + (k * defocus)^2)`` with ``k = sqrt((1 + eps^2) / 2) / N``.
Diffraction detail (Fresnel rings, Arago spot) is not modeled.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from sdasim.config import OpticsConfig

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

# Order of the static aberration terms in the gradient basis (see _zernike_grad).
_STATIC_TERMS = ("astig_0", "astig_45", "coma_x", "coma_y", "spherical")


def pupil_samples(k: int, obscuration: float, device=None) -> Tensor:
    """K deterministic, area-uniform samples of the annular pupil, shape (K, 2).

    Golden-angle spiral over pupil *area* (uniform in rho^2), restricted to
    the annulus [obscuration, 1]. Built as K/2 points plus their antipodes so
    the set is exactly inversion-symmetric (K is rounded up to even).
    Columns are (px, py).
    """
    half = max(1, (k + 1) // 2)
    i = torch.arange(half, dtype=torch.float32, device=device)
    e2 = obscuration**2
    rho = torch.sqrt(e2 + (1.0 - e2) * (i + 0.5) / half)
    theta = i * _GOLDEN_ANGLE
    p = torch.stack([rho * torch.cos(theta), rho * torch.sin(theta)], dim=1)
    return torch.cat([p, -p], dim=0)


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


class OpticsModel:
    """Precomputed pupil geometry for ``expand()`` (built once per Scene).

    Precomputes a small ladder of pupil sample sets (K = 4, 16, 64, ... up to
    what ``max_defocus_um`` needs) and, per rung, the image-plane offsets of
    the static non-defocus aberrations plus the defocus gradient direction.
    ``expand()`` then only scales/adds precomputed tensors, picks the smallest
    rung that keeps sub-source spacing below ~2*psf_sigma (no ring artifacts),
    and skips entirely when the total blur is negligible — so an in-focus,
    aberration-free frame costs nothing.
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
    ):
        if not 0.0 <= cfg.obscuration < 1.0:
            raise ValueError(f"obscuration must be in [0, 1), got {cfg.obscuration}")
        if cfg.fnumber <= 0 or cfg.pixel_pitch_um <= 0:
            raise ValueError("fnumber and pixel_pitch_um must be positive")
        self.cfg = cfg
        self.sigma = float(psf_sigma)
        n = cfg.fnumber
        lam_um = cfg.wavelength_nm * 1e-3
        # Displacement scale: wavefront-error microns -> image-plane pixels.
        self._px_per_um = 2.0 * n / cfg.pixel_pitch_um
        # Commanded focal-shift microns -> defocus (Z4) coefficient in microns.
        self._c4_per_defocus_um = 1.0 / (8.0 * n * n * 2.0 * math.sqrt(3.0))
        # Static coefficients in microns of wavefront error.
        ab = cfg.aberrations
        self._c4_static_um = ab.defocus * lam_um
        self._fc_um = ab.field_curvature * lam_um
        self._static_um = [getattr(ab, t) * lam_um for t in _STATIC_TERMS]
        # Field radius normalization: field_curvature is the extra defocus at
        # the field corner, growing quadratically from the boresight center.
        self._corner_r2 = (height / 2.0) ** 2 + (width / 2.0) ** 2

        # Rung ladder sized for the worst configured blur.
        r_max = self._blur_radius_px(
            abs(self._c4_static_um)
            + abs(self._fc_um)
            + cfg.max_defocus_um * self._c4_per_defocus_um
        )
        k_top = self._k_needed(r_max)
        self._rungs: list[dict] = []
        k = 4
        while True:
            p = pupil_samples(k, cfg.obscuration, device=device)
            grad4 = _zernike_grad("defocus", p) * self._px_per_um  # px per micron of c4
            static = torch.zeros_like(p)
            for c_um, term in zip(self._static_um, _STATIC_TERMS):
                if c_um != 0.0:
                    static = static + c_um * _zernike_grad(term, p) * self._px_per_um
            # (px, py) gradient components -> (row, col) image offsets.
            self._rungs.append({"k": p.shape[0], "grad4": grad4.flip(1), "static": static.flip(1)})
            if k >= k_top or k >= 4096:
                break
            k *= 4
        # Set by each expand() call; read by Scene for render metadata.
        self.last_k: int = 1
        self.last_blur_px: float = 0.0

    def _blur_radius_px(self, c4_um: float) -> float:
        """Upper-bound blur radius (px) for a defocus coefficient plus statics."""
        grad_max = {
            "astig_0": 2 * math.sqrt(6),
            "astig_45": 2 * math.sqrt(6),
            "coma_x": 7 * math.sqrt(8),
            "coma_y": 7 * math.sqrt(8),
            "spherical": 12 * math.sqrt(5),
        }
        r = abs(c4_um) * 4 * math.sqrt(3)
        for c_um, term in zip(self._static_um, _STATIC_TERMS):
            r += abs(c_um) * grad_max[term]
        return r * self._px_per_um

    def _k_needed(self, blur_px: float) -> int:
        """Samples needed so image-plane spacing stays <= 2*sigma (no rings)."""
        e2 = self.cfg.obscuration**2
        return max(4, math.ceil(math.pi * (1.0 - e2) * blur_px**2 / (4.0 * self.sigma**2)))

    def expand(
        self,
        positions: Tensor,
        intensities: Tensor,
        defocus_um: float | Tensor,
        center: tuple[float, float],
    ) -> tuple[Tensor, Tensor]:
        """Expand (M, 2)/(M,) sources into (M*K, 2)/(M*K,) defocused sub-sources.

        ``defocus_um`` may be a scalar tensor to keep gradients to the focus
        command. Flux is conserved (each sub-source carries 1/K). Returns the
        inputs unchanged when the total blur is negligible or M == 0.
        """
        m = positions.shape[0]
        if m == 0:
            return positions, intensities

        d_um = float(defocus_um) if not isinstance(defocus_um, Tensor) else float(defocus_um.item())
        c4_worst = abs(self._c4_static_um + d_um * self._c4_per_defocus_um) + abs(self._fc_um)
        blur_px = self._blur_radius_px(c4_worst)
        self.last_blur_px = blur_px
        if blur_px < self.SKIP_BLUR_SIGMA * self.sigma:
            self.last_k = 1
            return positions, intensities

        rung = next((r for r in self._rungs if r["k"] >= self._k_needed(blur_px)), self._rungs[-1])
        k = rung["k"]
        self.last_k = k

        # Per-source defocus coefficient (microns): static + commanded (+ field
        # curvature, quadratic in boresight-relative field radius).
        c4_um = self._c4_static_um + defocus_um * self._c4_per_defocus_um
        if self._fc_um != 0.0:
            rel = positions - positions.new_tensor(center)
            r2 = (rel * rel).sum(dim=1)
            c4_um = c4_um + self._fc_um * (r2 / self._corner_r2)
        c4_um = torch.as_tensor(c4_um, dtype=positions.dtype, device=positions.device)
        c4_um = c4_um.reshape(-1, 1, 1)  # (M or 1, 1, 1)

        offsets = rung["static"].unsqueeze(0) + c4_um * rung["grad4"].unsqueeze(0)
        expanded_pos = (positions.unsqueeze(1) + offsets).reshape(-1, 2)
        expanded_int = (intensities / k).unsqueeze(1).expand(m, k).reshape(-1)
        return expanded_pos, expanded_int
