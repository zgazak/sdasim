"""Empirical PSF + noise rendering (opt-in mode).

Calibrated from paired on-sky collects (sidereal point-PSF + rate-streak frames). This module
implements the SWITCHABLE empirical rendering path; the default Gaussian/basic path in
render.py is untouched.

Design:
  * Point-PSF: a measured mean PSF kernel (carries the real wings), optionally warped
    per frame by a size-scale + elliptical shear SAMPLED from the measured joint
    distribution (and extrapolatable past it by inflating the covariance) -- this is the
    generative knob set for pretraining-data diversity.
  * Rendering: bilinear-deposit all (sub-)sources into a point image, then convolve once
    with the kernel via FFT. Streaks (many motion sub-sources from expand_motion) cost
    nothing extra. Sub-pixel accurate and differentiable in position/intensity.
  * Noise: white Gaussian floor + shot (measured gain) + small background gradient +
    sparse hot pixels, then row/col-median subtraction to match the senpai processed stage.
    (Phase 2 found the noise is white -> no correlated-noise generator needed.)

All rendering is in ADU at the senpai processed stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from sdasim.render import expand_motion


# --------------------------------------------------------------------------- PSF
class EmpiricalPSF:
    """Mean empirical PSF + a samplable size/ellipticity distribution."""

    def __init__(self, path: str | Path):
        d = np.load(path)
        mp = np.clip(d["mean_psf"].astype("float32"), 0, None)
        mp /= mp.sum()
        self.mean_psf = torch.from_numpy(mp)
        self.K = self.mean_psf.shape[0]
        # eigen-PSF perturbation basis (for atmospheric-boiling streak structure)
        if "eigen_psfs" in d and "coeff_std" in d:
            self.eigen = torch.from_numpy(d["eigen_psfs"].astype("float32"))
            self.coeff_std = torch.from_numpy(d["coeff_std"].astype("float32"))
        else:
            self.eigen = None
            self.coeff_std = None
        sp = d["shape"]  # N x [fwhm_maj,fwhm_min,e1,e2,conc]
        size = 0.5 * (sp[:, 0] + sp[:, 1])
        self.base_fwhm = float(np.median(size))
        P = np.column_stack([size / self.base_fwhm, sp[:, 2], sp[:, 3]])  # size_scale,e1,e2
        self.param_mean = P.mean(0).astype("float64")
        self.param_cov = np.cov(P, rowvar=False).astype("float64")

    def sample_params(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        """Draw (size_scale, e1, e2). scale>1 extrapolates past the real distribution."""
        return rng.multivariate_normal(self.param_mean, self.param_cov * scale**2)

    def kernel(self, params=None, device="cpu", rng=None, scale: float = 1.0) -> Tensor:
        """Build a (K,K) kernel (sum=1) = mean PSF warped by (size_scale, e1, e2).

        params=None & scale==0 -> the nominal mean PSF (no variation).
        params=None & scale>0  -> a random draw from the (inflated) real distribution.
        """
        if params is None:
            if scale <= 0:
                params = self.param_mean
            else:
                params = self.sample_params(rng or np.random.default_rng(), scale)
        s, e1, e2 = (float(params[0]), float(params[1]), float(params[2]))
        s = max(s, 0.3)
        base = self.mean_psf.to(device)[None, None]
        K = self.K
        M = torch.tensor(
            [[s * (1 + e1), s * e2], [s * e2, s * (1 - e1)]], dtype=torch.float32, device=device
        )
        Minv = torch.linalg.inv(M)
        lin = torch.linspace(-1, 1, K, device=device)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        coords = torch.stack([gx, gy], -1).reshape(-1, 2)  # output coords (x,y)
        src = (coords @ Minv.T).reshape(1, K, K, 2)  # -> input coords
        out = F.grid_sample(base, src, mode="bilinear", align_corners=True, padding_mode="zeros")[
            0, 0
        ]
        out = out.clamp_min(0)
        return out / out.sum().clamp_min(1e-8)

    def structural_modes(
        self,
        n: int = 6,
        device="cpu",
        taper_in: float = 13.0,
        taper_out: float = 18.0,
        smooth_px: float = 0.5,
    ):
        """Eigen-PSF perturbation kernels + coeff stds for streak speckle.

        Skips mode 0 (resampling artifact); RADIALLY TAPERS each mode to the PSF footprint
        (corner support is noise); and lightly GAUSSIAN-SMOOTHS each mode (smooth_px) to strip
        the PCA pixel-noise that would otherwise give a 'crispy' rim — keeps the ~PSF-scale
        structural boil, drops the sub-pixel noise."""
        if self.eigen is None:
            return None, None
        n = min(n, self.eigen.shape[0] - 1)
        eig = self.eigen[1 : 1 + n].clone()
        K = eig.shape[-1]
        c = K // 2
        yy, xx = torch.meshgrid(torch.arange(K) - c, torch.arange(K) - c, indexing="ij")
        r = torch.hypot(xx.float(), yy.float())
        t = torch.clamp((taper_out - r) / (taper_out - taper_in), 0.0, 1.0)
        win = 0.5 - 0.5 * torch.cos(torch.pi * t)  # smooth cosine: 1 inside, 0 outside
        eig = eig * win[None]
        if smooth_px and smooth_px > 0:
            rr = int(max(1, round(3 * smooth_px)))
            ax = torch.arange(-rr, rr + 1, dtype=eig.dtype)
            g = torch.exp(-0.5 * (ax / smooth_px) ** 2)
            g = g / g.sum()
            k2 = (g[:, None] * g[None, :]).view(1, 1, 2 * rr + 1, 2 * rr + 1)
            eig = F.conv2d(eig.unsqueeze(1), k2, padding=rr).squeeze(1)
        return eig.to(device), self.coeff_std[1 : 1 + n].to(device)

    def size_mode(self, eps: float = 0.07, device="cpu") -> Tensor:
        """Symmetric PSF size-derivative kernel d(PSF)/d(size), for seeing-breathing: the PSF
        isotropically expands/contracts -> moves flux core<->wings without any one-sided edge
        artifact (unlike the asymmetric eigenmodes). Modulated along the track it makes the
        streak fatten/thin and brighten/dim in correlated patches."""
        return (
            self.kernel([1.0 + eps, 0.0, 0.0], device) - self.kernel([1.0, 0.0, 0.0], device)
        ) / eps


def gaussian_kernel(sigma: float, K: int, device="cpu") -> Tensor:
    """A (K,K) Gaussian kernel (sum=1) for the gaussian-PSF + empirical-noise combo."""
    r = K // 2
    ax = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    k = g[:, None] * g[None, :]
    return k / k.sum()


def deconvolve_gaussian(kernel: Tensor, sigma: float, reg: float = 1e-2) -> Tensor:
    """Wiener-deconvolve an isotropic Gaussian of width `sigma` from `kernel`.

    The measured sidereal PSF = instantaneous PSF ⊛ (wobble distribution over the
    exposure). Removing the wobble Gaussian recovers the sharper INSTANTANEOUS PSF, so
    that re-applying the wobble (to a stationary source) reproduces the measured PSF and
    a moving source produces a physically-consistent wobbly streak (no double-counting).
    """
    if sigma <= 0:
        return kernel
    K = kernel.shape[-1]
    ax = torch.arange(K, dtype=kernel.dtype, device=kernel.device) - (K // 2)
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    g2 = g[:, None] * g[None, :]
    g2 = torch.roll(g2, shifts=(-(K // 2), -(K // 2)), dims=(0, 1))  # center at (0,0)
    Kf = torch.fft.rfft2(kernel)
    Gf = torch.fft.rfft2(g2)
    Hf = torch.conj(Gf) / (Gf.real**2 + Gf.imag**2 + reg)
    out = torch.fft.irfft2(Kf * Hf, s=kernel.shape)
    out = out.clamp_min(0)
    return out / out.sum().clamp_min(1e-8)


# ------------------------------------------------------------------------- noise
class EmpiricalNoise:
    """White-noise floor + shot (gain) + background gradient + hot pixels."""

    def __init__(self, path: str | Path):
        j = json.load(open(path))
        self.gain = float(j["gain_e_per_adu"])
        self.floor = float(j["noise_floor_adu"])
        self.floor_std = float(j.get("noise_floor_std_adu", 0.0))  # sky-brightness variation
        self.grad_pp = float(j.get("background_gradient_pp_adu", 0.0))
        self.hot_frac = float(j.get("hot_pixel_fraction", 0.0))

    def apply(self, signal: Tensor, generator=None, hot_value: float = 55000.0) -> Tensor:
        H, W = signal.shape
        dev = signal.device
        # per-frame sky/noise level: vary the white floor (sky brightness changes frame to frame)
        floor = self.floor
        if self.floor_std > 0:
            floor = max(
                30.0,
                self.floor
                + self.floor_std * float(torch.randn((), generator=generator, device=dev)),
            )
        # smooth background gradient (random tilt), peak-to-peak ~ grad_pp
        a = torch.rand(2, generator=generator, device=dev) - 0.5
        yy = torch.linspace(-1, 1, H, device=dev)[:, None]
        xx = torch.linspace(-1, 1, W, device=dev)[None, :]
        bg = a[0] * yy + a[1] * xx
        rng_bg = (bg.max() - bg.min()).clamp_min(1e-9)
        s = signal + bg / rng_bg * self.grad_pp
        # white Gaussian: floor (sky+read) + shot (signal/gain)
        std = torch.sqrt(floor**2 + torch.clamp(s, min=0) / self.gain)
        s = s + std * torch.randn(s.shape, generator=generator, device=dev)
        # sparse hot pixels
        nh = int(self.hot_frac * H * W)
        if nh > 0:
            idx = torch.randint(0, H * W, (nh,), generator=generator, device=dev)
            sf = s.reshape(-1).clone()
            sf[idx] = hot_value
            s = sf.reshape(H, W)
        return s

    @staticmethod
    def match_stage(s: Tensor) -> Tensor:
        """Row + column median subtraction = the senpai processed pixel stage."""
        s = s - s.median(dim=1, keepdim=True).values
        s = s - s.median(dim=0, keepdim=True).values
        return s


class BasicWhiteNoise:
    """Minimal white read-noise floor (for the gaussian-PSF + empirical-render combo)."""

    def __init__(self, floor: float):
        self.floor = float(floor)

    def apply(self, signal: Tensor, generator=None, **_) -> Tensor:
        return signal + self.floor * torch.randn(
            signal.shape, generator=generator, device=signal.device
        )

    @staticmethod
    def match_stage(s: Tensor) -> Tensor:
        return EmpiricalNoise.match_stage(s)


# --------------------------------------------------------------------- rendering
def _bilinear_deposit(H: int, W: int, pos: Tensor, inten: Tensor) -> Tensor:
    """Deposit each source's flux into the 4 nearest pixels (sub-pixel point image)."""
    img = torch.zeros(H * W, dtype=pos.dtype, device=pos.device)
    if pos.shape[0] == 0:
        return img.reshape(H, W)
    r, c = pos[:, 0], pos[:, 1]
    r0 = torch.floor(r).long()
    c0 = torch.floor(c).long()
    fr, fc = r - r0.float(), c - c0.float()
    for dr, dc, w in (
        (0, 0, (1 - fr) * (1 - fc)),
        (0, 1, (1 - fr) * fc),
        (1, 0, fr * (1 - fc)),
        (1, 1, fr * fc),
    ):
        rr, cc = r0 + dr, c0 + dc
        valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        idx = rr.clamp(0, H - 1) * W + cc.clamp(0, W - 1)
        img.scatter_add_(0, idx, inten * w * valid.to(inten.dtype))
    return img.reshape(H, W)


def _fft_convolve(img: Tensor, kernel: Tensor) -> Tensor:
    """Convolve (H,W) image with a centered (K,K) kernel via FFT."""
    H, W = img.shape
    K = kernel.shape[0]
    r = K // 2
    ker = torch.zeros(H, W, dtype=img.dtype, device=img.device)
    ker[:K, :K] = kernel
    ker = torch.roll(ker, shifts=(-r, -r), dims=(0, 1))  # center kernel at (0,0)
    out = torch.fft.irfft2(torch.fft.rfft2(img) * torch.fft.rfft2(ker), s=(H, W))
    return out


def render_empirical_signal(H: int, W: int, pos: Tensor, inten: Tensor, kernel: Tensor) -> Tensor:
    return _fft_convolve(_bilinear_deposit(H, W, pos, inten), kernel)


def _smooth1d(x: Tensor, sigma: float) -> Tensor:
    """Gaussian-smooth along the last axis of (N, T)."""
    r = int(max(1, round(3 * sigma)))
    ax = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (ax / sigma) ** 2)
    k = k / k.sum()
    xp = F.pad(x.unsqueeze(1), (r, r), mode="reflect")
    return F.conv1d(xp, k.view(1, 1, -1)).squeeze(1)


def _alongtrack_process(n_src, t_osf, std, corr_px, L_px, generator, device, dtype):
    """A zero-mean, unit-then-scaled 1D process along the track (n_src, t_osf)."""
    m = torch.randn(n_src, t_osf, generator=generator, device=device, dtype=dtype)
    spacing = max(L_px / t_osf, 1e-3)
    sigma = max(corr_px / spacing / 2.0, 0.3)
    m = _smooth1d(m, sigma)
    m = m - m.mean(dim=1, keepdim=True)
    return m / m.std(dim=1, keepdim=True).clamp_min(1e-6) * std


def apply_along_track_texture(
    exp_int: Tensor,
    n_src: int,
    t_osf: int,
    L_px: float,
    rel_rms: float,
    corr_len_px: float,
    generator=None,
) -> Tensor:
    """Modulate per-sub-source intensities along the trail (scintillation + tracking
    jitter): I(s) = mean * (1 + m), m a 1D process with the measured rel-RMS and
    correlation length. exp_int is ordered (n_src, t_osf) flattened."""
    if rel_rms <= 0 or t_osf <= 1 or n_src <= 0:
        return exp_int
    m = torch.randn(n_src, t_osf, generator=generator, device=exp_int.device, dtype=exp_int.dtype)
    spacing = max(L_px / t_osf, 1e-3)
    sigma = max(corr_len_px / spacing / 2.0, 0.3)
    m = _smooth1d(m, sigma)
    m = m - m.mean(dim=1, keepdim=True)
    m = m / m.std(dim=1, keepdim=True).clamp_min(1e-6) * rel_rms
    return (exp_int.reshape(n_src, t_osf) * (1.0 + m).clamp_min(0.0)).reshape(-1)


def apply_cross_track_wobble(
    pos: Tensor,
    n_src: int,
    t_osf: int,
    velocity,
    wob_rms_px: float,
    wob_corr_px: float,
    exposure: float,
    generator=None,
) -> Tensor:
    """Displace sub-sources PERPENDICULAR to the track by a slow 1D wander (tracking
    jitter) -> the streak isn't perfectly straight ('flaring'). pos ordered (n_src, t_osf)."""
    if wob_rms_px <= 0 or t_osf <= 1 or n_src <= 0:
        return pos
    v = torch.as_tensor(velocity, dtype=pos.dtype, device=pos.device)
    if v.dim() == 1:
        v = v.unsqueeze(0).expand(n_src, 2)
    speed = torch.linalg.vector_norm(v, dim=1, keepdim=True).clamp_min(1e-6)  # (n,1)
    perp = torch.stack([-v[:, 1], v[:, 0]], dim=1) / speed  # (n,2) unit ⟂
    L_px = speed.squeeze(1) * exposure
    spacing = (L_px / t_osf).clamp_min(1e-3)
    sigma = float((wob_corr_px / spacing / 2.0).clamp_min(0.3).mean())
    w = torch.randn(n_src, t_osf, generator=generator, device=pos.device, dtype=pos.dtype)
    w = _smooth1d(w, sigma)
    w = w - w.mean(dim=1, keepdim=True)
    w = w / w.std(dim=1, keepdim=True).clamp_min(1e-6) * wob_rms_px
    disp = w.reshape(n_src, t_osf, 1) * perp.reshape(n_src, 1, 2)
    return pos + disp.reshape(-1, 2)


def apply_jitter(
    pos: Tensor,
    n_src: int,
    t_osf: int,
    velocity,
    exposure: float,
    along_rms: float,
    along_corr: float,
    cross_rms: float,
    cross_corr: float,
    generator=None,
) -> Tensor:
    """Jerky mount tracking: displace the motion sub-sources by a time-correlated 2D wander,
    with INDEPENDENT amplitude+correlation for the two axes (real mounts differ along vs cross):
      * ALONG-track (rate jitter): modulates path speed -> where it bunches the source dwells and
        photons pile into a smooth PSF-shaped knot. The real along-track lumps, from path DENSITY
        (not a painted multiplier), so each knot is a clean PSF. Shorter correlation ->
        tighter knots.
      * CROSS-track (pointing drift): a larger-but-SMOOTHER (longer-correlation) centroid wander ->
        the gentle band undulation + the double-humped 'structural fluctuation vs cross-track' seen
        in real stacked profiles. Longer correlation keeps it smooth, not jagged.
    pos ordered (n_src, t_osf) flattened."""
    if (along_rms <= 0 and cross_rms <= 0) or t_osf <= 1 or n_src <= 0:
        return pos
    v = torch.as_tensor(velocity, dtype=pos.dtype, device=pos.device)
    if v.dim() == 1:
        v = v.unsqueeze(0).expand(n_src, 2)
    speed = torch.linalg.vector_norm(v, dim=1, keepdim=True).clamp_min(1e-6)
    along = v / speed  # (n,2) unit ∥
    perp = torch.stack([-v[:, 1], v[:, 0]], dim=1) / speed  # (n,2) unit ⟂
    L_px = speed.squeeze(1) * exposure
    spacing = (L_px / t_osf).clamp_min(1e-3)

    def proc(rms, corr_px):
        if rms <= 0:
            return torch.zeros(n_src, t_osf, device=pos.device, dtype=pos.dtype)
        sigma = float((corr_px / spacing / 2.0).clamp_min(0.3).mean())
        w = torch.randn(n_src, t_osf, generator=generator, device=pos.device, dtype=pos.dtype)
        w = _smooth1d(w, sigma)
        w = w - w.mean(dim=1, keepdim=True)
        return w / w.std(dim=1, keepdim=True).clamp_min(1e-6) * rms

    ja = proc(along_rms, along_corr)
    jc = proc(cross_rms, cross_corr)
    disp = ja.reshape(n_src, t_osf, 1) * along.reshape(n_src, 1, 2) + jc.reshape(
        n_src, t_osf, 1
    ) * perp.reshape(n_src, 1, 2)
    return pos + disp.reshape(-1, 2)


def render_streak_speckle(
    H,
    W,
    pos,
    inten,
    base_kernel,
    eigen,
    coeff_std,
    n_src,
    t_osf,
    amp,
    corr_px,
    L_px,
    generator,
    noise_floor=0.0,
):
    """Streak with atmospheric-boiling STRUCTURE: the PSF shape fluctuates along the
    track via eigen-PSF coefficients c_k(s) (time-correlated). Linear in the eigenbasis:
        streak = deposit(I)⊛mean + Σ_k deposit(I·c_k(s))⊛eigen_k.
    Zero-mean c_k -> the sidereal (stationary) average is unchanged (consistent)."""
    base = render_empirical_signal(H, W, pos, inten, base_kernel)
    pert = torch.zeros_like(base)
    for k in range(eigen.shape[0]):
        std = amp * float(coeff_std[k])
        if std <= 0:
            continue
        ck = _alongtrack_process(
            n_src, t_osf, std, corr_px, L_px, generator, pos.device, inten.dtype
        ).reshape(-1)
        pert = pert + _fft_convolve(_bilinear_deposit(H, W, pos, inten * ck), eigen[k])
    # Keep the boil wherever the streak is genuinely above the noise (it's real, high-SNR
    # structure there) and fade it only in the far sub-noise wings, where the base streak has
    # vanished but the signed perturbation would otherwise leave coherent, rectified wisps that
    # the eye/asinh picks out. Cutoff keyed to the NOISE FLOOR, not a fraction of the peak.
    if noise_floor > 0:
        mask = (base / (3.0 * noise_floor)).clamp(0.0, 1.0)
    else:
        mask = (base / (0.05 * base.amax().clamp_min(1e-6))).clamp(0.0, 1.0)
    return base + pert * mask


def _red_process(n, t_osf, rms, beta, generator, device, dtype):
    """Broadband 1D process along the track with a power-law spectrum (power ~ f^-beta).
    beta~2 is red; lower=more high-freq. Unlike a single-correlation smooth (narrow-band),
    this has structure at ALL scales -> the multi-scale atmospheric character."""
    if rms <= 0 or t_osf <= 1 or n <= 0:
        return torch.zeros(n, t_osf, device=device, dtype=dtype)
    w = torch.randn(n, t_osf, generator=generator, device=device, dtype=dtype)
    W = torch.fft.rfft(w, dim=1)
    f = torch.fft.rfftfreq(t_osf, device=device, dtype=dtype).clone()
    f[0] = f[1]
    x = torch.fft.irfft(W * f.unsqueeze(0) ** (-beta / 2.0), n=t_osf, dim=1)
    x = x - x.mean(dim=1, keepdim=True)
    return x / x.std(dim=1, keepdim=True).clamp_min(1e-6) * rms


def _hf_field(H, W, rms, klo, khi, slope, generator, device, dtype):
    """Post-PSF band-limited (klo..khi cyc/px) high-frequency texture field. Fills the measured
    3-9px texture 'plateau' the PSF would otherwise erase if applied pre-convolution.
    A RED slope WITHIN the band (amp ~ k^-slope) makes it whispy/coherent rather than white
    salt-and-pepper; band-limiting (no content finer than ~1/khi px) keeps it smooth at the
    pixel scale. rms-normalized."""
    if rms <= 0:
        return torch.zeros(H, W, device=device, dtype=dtype)
    ky = torch.fft.fftfreq(H, device=device, dtype=dtype)[:, None]
    kx = torch.fft.rfftfreq(W, device=device, dtype=dtype)[None, :]
    k = torch.hypot(ky, kx)
    msk = ((k >= klo) & (k <= khi)).to(dtype)
    kk = k.clamp_min(klo)
    amp = msk * kk ** (-slope)
    re = torch.randn(msk.shape, generator=generator, device=device, dtype=dtype)
    im = torch.randn(msk.shape, generator=generator, device=device, dtype=dtype)
    field = torch.fft.irfft2(torch.complex(re, im) * amp, s=(H, W))
    return field / field.std().clamp_min(1e-6) * rms


def _smooth2d(img: Tensor, sigma: float) -> Tensor:
    """Isotropic Gaussian smooth of a 2D image (for the local streak-brightness envelope)."""
    r = int(max(1, round(3 * sigma)))
    ax = torch.arange(-r, r + 1, dtype=img.dtype, device=img.device)
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    k = (g[:, None] * g[None, :]).view(1, 1, 2 * r + 1, 2 * r + 1)
    return F.conv2d(img[None, None], k, padding=r)[0, 0]


def render_textured_streak(
    H,
    W,
    ep,
    ei,
    kernel,
    velocity,
    n_src,
    t_osf,
    exposure,
    scint_rms,
    tiptilt_rms,
    seeing_rms,
    hf_rms,
    beta,
    decon_px,
    hf_klo,
    hf_khi,
    hf_slope,
    size_mode,
    generator,
    floor_frac=0.0,
):
    """Streak with the 4 calibrated atmospheric texture components (all opt-in via amplitude):
    scintillation (intensity), tip/tilt (position), seeing-breathing (PSF size), and a
    post-PSF high-frequency plateau. Drives the structured knobs with broadband processes."""
    dev, dt = ep.device, ei.dtype
    ei0 = ei  # un-textured intensities (flux-floor reference)
    if scint_rms > 0:  # scintillation: intensity flicker
        s = _red_process(n_src, t_osf, scint_rms, beta, generator, dev, dt).reshape(-1)
        ei = ei * (1.0 + s).clamp_min(0.0)
    if tiptilt_rms > 0:  # tip/tilt: 2D position wander
        v = torch.as_tensor(velocity, dtype=dt, device=dev)
        if v.dim() == 1:
            v = v.unsqueeze(0).expand(n_src, 2)
        spd = torch.linalg.vector_norm(v, dim=1, keepdim=True).clamp_min(1e-6)
        av = v / spd
        pv = torch.stack([-v[:, 1], v[:, 0]], dim=1) / spd
        ja = _red_process(n_src, t_osf, tiptilt_rms, beta, generator, dev, dt)
        jc = _red_process(n_src, t_osf, tiptilt_rms, beta, generator, dev, dt)
        ep = ep + (
            ja.reshape(n_src, t_osf, 1) * av.reshape(n_src, 1, 2)
            + jc.reshape(n_src, t_osf, 1) * pv.reshape(n_src, 1, 2)
        ).reshape(-1, 2)
    k = deconvolve_gaussian(kernel, decon_px) if decon_px > 0 else kernel  # instantaneous PSF
    base = render_empirical_signal(H, W, ep, ei, k)
    if seeing_rms > 0 and size_mode is not None:  # seeing-breathing: PSF size pulse
        b = _red_process(n_src, t_osf, seeing_rms, beta, generator, dev, dt).reshape(-1)
        base = base + _fft_convolve(_bilinear_deposit(H, W, ep, ei * b), size_mode)
    if hf_rms > 0:  # post-PSF whispy high-freq plateau
        hf = _hf_field(H, W, hf_rms, hf_klo, hf_khi, hf_slope, generator, base.device, base.dtype)
        base = base * (1.0 + hf.clamp_min(-0.8))  # clamp -> can't punch black-pepper holes
    if floor_frac > 0:  # flux floor referenced to the UN-textured
        ref = render_empirical_signal(
            H, W, ep, ei0, k
        )  # streak (not the holey base), so it actually
        base = torch.maximum(base, floor_frac * ref)  # fills deep holes -> core keeps some flux
    return base


def render_frame_empirical(
    height: int,
    width: int,
    star_positions: Tensor,
    star_intensities: Tensor,
    target_positions: Tensor,
    target_intensities: Tensor,
    kernel: Tensor,
    noise=None,
    star_velocity=None,
    star_rotation: float = 0.0,
    target_velocities: Tensor | None = None,
    t_osf: int = 1,
    target_t_osf: int | None = None,
    exposure: float = 1.0,
    tex_scint_rms: float = 0.0,
    tex_tiptilt_rms: float = 0.0,
    tex_seeing_rms: float = 0.0,
    tex_hf_rms: float = 0.0,
    tex_beta: float = 2.0,
    tex_decon_px: float = 1.3,
    tex_hf_klo: float = 0.08,
    tex_hf_khi: float = 0.30,
    tex_hf_slope: float = 1.2,
    tex_floor_frac: float = 0.0,
    size_mode: Tensor | None = None,
    generator=None,
    match_stage: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Empirical-PSF render to the processed ADU stage. Moving sources get the 4 calibrated
    atmospheric texture components (scintillation, tip/tilt, seeing-breathing, hf-plateau)."""
    center = (height / 2.0, width / 2.0)
    txt = dict(
        scint_rms=tex_scint_rms,
        tiptilt_rms=tex_tiptilt_rms,
        seeing_rms=tex_seeing_rms,
        hf_rms=tex_hf_rms,
        beta=tex_beta,
        decon_px=tex_decon_px,
        hf_klo=tex_hf_klo,
        hf_khi=tex_hf_khi,
        hf_slope=tex_hf_slope,
        floor_frac=tex_floor_frac,
        size_mode=size_mode,
    )
    any_tex = tex_scint_rms > 0 or tex_tiptilt_rms > 0 or tex_seeing_rms > 0 or tex_hf_rms > 0

    has_star_motion = star_rotation != 0.0 or (
        star_velocity is not None
        and any(
            v != 0.0
            for v in (
                star_velocity
                if isinstance(star_velocity, (list, tuple))
                else star_velocity.tolist()
            )
        )
    )
    if has_star_motion and t_osf > 1 and star_positions.shape[0] > 0:
        ep, ei = expand_motion(
            star_positions,
            star_intensities,
            star_velocity,
            star_rotation,
            0.0,
            exposure,
            t_osf,
            center,
        )
        if any_tex:
            star_signal = render_textured_streak(
                height,
                width,
                ep,
                ei,
                kernel,
                star_velocity,
                star_positions.shape[0],
                t_osf,
                exposure,
                generator=generator,
                **txt,
            )
        else:
            star_signal = render_empirical_signal(height, width, ep, ei, kernel)
    else:
        star_signal = render_empirical_signal(
            height, width, star_positions, star_intensities, kernel
        )

    tgt_osf = target_t_osf
    if tgt_osf is None and target_velocities is not None and target_positions.shape[0] > 0:
        max_speed = target_velocities.abs().max().item()
        tgt_osf = max(1, int(max_speed * exposure * 2))
    if (
        target_velocities is not None
        and tgt_osf is not None
        and tgt_osf > 1
        and target_positions.shape[0] > 0
    ):
        tp, ti = expand_motion(
            target_positions,
            target_intensities,
            target_velocities,
            0.0,
            0.0,
            exposure,
            tgt_osf,
            center,
        )
        if any_tex:
            target_signal = render_textured_streak(
                height,
                width,
                tp,
                ti,
                kernel,
                target_velocities,
                target_positions.shape[0],
                tgt_osf,
                exposure,
                generator=generator,
                **txt,
            )
        else:
            target_signal = render_empirical_signal(height, width, tp, ti, kernel)
    else:
        target_signal = render_empirical_signal(
            height, width, target_positions, target_intensities, kernel
        )

    signal = star_signal + target_signal
    # Sources only ADD light: the signed eigen/speckle perturbations must never dig
    # sub-background holes in the wings (would suppress noise -> unphysical dark splotches).
    signal = signal.clamp_min(0.0)
    if noise is not None:
        signal = noise.apply(signal, generator=generator)
        if match_stage:
            signal = noise.match_stage(signal)
    return signal, star_signal, target_signal
