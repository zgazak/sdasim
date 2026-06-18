"""Real-vs-simulated streak visualization (talk-ready).

Walks a senpai reduction directory, and for each measured rate-frame streak renders a
matched sdasim streak (same length + orientation) from a calibration, side by side.

  * make_pairs()        -> list of {real, sim, length, angle, id}
  * save_contact_sheet()-> static grid of real|sim pairs (PNG)
  * save_video()        -> side-by-side animation over the dataset (GIF or MP4)

CLI:
    python -m sdasim.streak_vis REDUCTION_DIR CALIB_DIR -o out.gif [--n 40] [--fps 3]
                               [--contact pairs.png] [--exposure 2.0]
Needs the [calibrate] extras (astropy, scipy, matplotlib) + torch.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import torch

import sdasim
from sdasim.config import SceneConfig, SensorConfig, StarFieldConfig, StarMotionConfig


# ------------------------------------------------------------------ sim render
def _sim_scene(calib_dir, size, exposure, seed, extend=1.3):
    noise = json.load(open(os.path.join(calib_dir, "noise_model.json")))
    s = SensorConfig(
        height=size,
        width=size,
        exposure=exposure,
        psf_model="empirical",
        noise_model="empirical",
        empirical_psf_path=os.path.join(calib_dir, "psf_basis.npz"),
        empirical_noise_path=os.path.join(calib_dir, "noise_model.json"),
        empirical_streak_path=os.path.join(calib_dir, "streak_model.json"),
        psf_param_scale=extend,
        streak_param_sample=True,
        streak_param_extend=extend,
        gain=float(noise.get("gain_e_per_adu", 1.0)),
        apply_rowcol_median=False,
    )
    return sdasim.Scene(
        SceneConfig(
            sensor=s,
            stars=StarFieldConfig(mode="bins", density=[0.0] * 12),
            star_motion=StarMotionConfig(),
            seed=seed,
        )
    )


def render_sim_streak(calib_dir, length, angle_deg, size, exposure=2.0, seed=0, flux=8.0e7):
    sc = _sim_scene(calib_dir, size, exposure, seed)
    rate = length / exposure
    ang = math.radians(angle_deg)
    vel = torch.tensor([[rate * math.sin(ang), rate * math.cos(ang)]])
    cc = size / 2.0
    start = torch.tensor(
        [[cc - 0.5 * float(vel[0, 0]) * exposure, cc - 0.5 * float(vel[0, 1]) * exposure]]
    )
    img, _ = sc.render(
        0, target_positions=start, target_intensities=torch.tensor([flux]), target_velocities=vel
    )
    return img.detach().cpu().numpy()


def _psf_sigma(calib_dir):
    """Gaussian sigma matched to the empirical mean-PSF half-max FWHM (for the baseline)."""
    m = np.load(os.path.join(calib_dir, "psf_basis.npz"))["mean_psf"].astype("f8")
    m = m / m.max()
    K = m.shape[0]
    c = K // 2
    yy, xx = np.mgrid[0:K, 0:K]
    r = np.hypot(xx - c, yy - c)
    rb = np.arange(0, c, 1.0)
    prof = np.array([m[(r >= rb[k]) & (r < rb[k + 1])].mean() for k in range(len(rb) - 1)])
    rc = 0.5 * (rb[:-1] + rb[1:])
    prof = prof / prof[0]
    below = np.where(prof < 0.5)[0]
    if len(below) == 0 or below[0] == 0:
        return 4.0
    i = below[0]
    return 2 * np.interp(0.5, [prof[i], prof[i - 1]], [rc[i], rc[i - 1]]) / 2.3548


def render_gauss_streak(
    calib_dir, length, angle_deg, size, sigma, exposure=2.0, seed=0, flux=8.0e7
):
    """Gaussian-PSF baseline streak: same empirical noise/background, NO texture/wobble/speckle."""
    noise = json.load(open(os.path.join(calib_dir, "noise_model.json")))
    s = SensorConfig(
        height=size,
        width=size,
        exposure=exposure,
        psf_model="gaussian",
        noise_model="empirical",
        empirical_noise_path=os.path.join(calib_dir, "noise_model.json"),
        empirical_streak_path=None,
        psf_sigma=sigma,
        gain=float(noise.get("gain_e_per_adu", 1.0)),
        apply_rowcol_median=False,
    )
    sc = sdasim.Scene(
        SceneConfig(
            sensor=s,
            stars=StarFieldConfig(mode="bins", density=[0.0] * 12),
            star_motion=StarMotionConfig(),
            seed=seed,
        )
    )
    rate = length / exposure
    ang = math.radians(angle_deg)
    vel = torch.tensor([[rate * math.sin(ang), rate * math.cos(ang)]])
    cc = size / 2.0
    start = torch.tensor(
        [[cc - 0.5 * float(vel[0, 0]) * exposure, cc - 0.5 * float(vel[0, 1]) * exposure]]
    )
    img, _ = sc.render(
        0, target_positions=start, target_intensities=torch.tensor([flux]), target_velocities=vel
    )
    return img.detach().cpu().numpy()


# ------------------------------------------------------------------ real cutouts
def iter_real_streaks(reduction_dir, max_n=40, min_len=60, box=128):
    """Yield fixed `box`x`box` real-streak cutouts, centered on the trail CENTROID
    (intensity-weighted center of the bright pixels — not the brightest single pixel)."""
    from astropy.io import fits
    from scipy.ndimage import gaussian_filter

    h = box // 2
    out = []
    for cd in sorted(glob.glob(reduction_dir + "/*")):
        if len(out) >= max_n:
            break
        rfp = rL = ang = None
        for fn, fr in [("frame_1_rate.json", "frame2of3"), ("frame_2_rate.json", "frame3of3")]:
            p = cd + "/" + fn
            if not os.path.exists(p):
                continue
            s = (json.load(open(p)) or {}).get("streak")
            if s and s.get("pixel_length", 0) > min_len and s.get("sine_angle") is not None:
                c = glob.glob(cd + f"/*{fr}*_rate_processed.fits")
                if c:
                    rfp, rL = c[0], float(s["pixel_length"])
                    ang = float(np.degrees(np.arctan2(s["sine_angle"], s["cosine_angle"])))
                    break
        if not rfp:
            continue
        im = fits.getdata(rfp).astype("f4")
        H, W = im.shape
        win = int(rL * 0.7 + 20)
        sm = gaussian_filter(np.where(im > 40000, 0, im), 3)
        bw = win + h + 5
        sm[:bw] = 0
        sm[-bw:] = 0
        sm[:, :bw] = 0
        sm[:, -bw:] = 0
        sy, sx = np.unravel_index(np.argmax(sm), sm.shape)  # a point ON the trail
        sub = im[sy - win : sy + win + 1, sx - win : sx + win + 1].astype("f8")
        b = sub - np.median(sub)
        sg = 1.4826 * np.median(np.abs(b)) + 1e-6
        b[b < 4 * sg] = 0.0
        yy, xx = np.mgrid[0 : b.shape[0], 0 : b.shape[1]]
        t = b.sum()
        cy = sy - win + (yy * b).sum() / t if t > 0 else sy  # trail centroid
        cx = sx - win + (xx * b).sum() / t if t > 0 else sx
        cyi = int(np.clip(round(cy), h, H - h - 1))
        cxi = int(np.clip(round(cx), h, W - h - 1))
        cut = im[cyi - h : cyi + h, cxi - h : cxi + h].astype("f8")
        out.append(
            dict(
                real=cut - np.median(cut),
                length=rL,
                angle=ang,
                id=os.path.basename(cd)[:8],
                size=box,
            )
        )
    return out


def make_pairs(reduction_dir, calib_dir, n=40, exposure=2.0, min_len=60, box=128):
    pairs = iter_real_streaks(reduction_dir, max_n=n, min_len=min_len, box=box)
    sigma = _psf_sigma(calib_dir)
    for k, p in enumerate(pairs):
        real_peak = float(np.percentile(p["real"], 99.9))
        flux = max(2e7, real_peak * p["length"] / 0.011)  # rough brightness match
        p["sim"] = render_sim_streak(
            calib_dir, p["length"], p["angle"], size=box, exposure=exposure, seed=k, flux=flux
        )
        p["gauss"] = render_gauss_streak(
            calib_dir,
            p["length"],
            p["angle"],
            size=box,
            sigma=sigma,
            exposure=exposure,
            seed=k,
            flux=flux,
        )
    return pairs


# ------------------------------------------------------------------ figures
def _show(ax, a, title):
    from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval

    a = np.asarray(a, "f8")
    ax.imshow(
        a,
        norm=ImageNormalize(a, interval=PercentileInterval(99.6), stretch=AsinhStretch(0.04)),
        cmap="gray",
    )
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def save_contact_sheet(pairs, out_png, rows=6):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(pairs), rows)
    fig, ax = plt.subplots(n, 3, figsize=(10.5, n * 3.2))
    ax = np.atleast_2d(ax)
    for i in range(n):
        p = pairs[i]
        _show(ax[i, 0], p["real"], f"real  {p['length']:.0f}px {p['angle']:.0f}deg")
        _show(ax[i, 1], p["sim"], "empirical")
        _show(ax[i, 2], p["gauss"], "gaussian")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def save_video(pairs, out_path, fps=3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, (axR, axE, axG) = plt.subplots(1, 3, figsize=(12.6, 4.5))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.005, wspace=0.03)

    def update(i):
        p = pairs[i]
        axR.clear()
        axE.clear()
        axG.clear()
        _show(axR, p["real"], f"real   {p['length']:.0f}px  {p['angle']:.0f}deg")
        _show(axE, p["sim"], "empirical")
        _show(axG, p["gauss"], "gaussian")
        return []

    anim = FuncAnimation(fig, update, frames=len(pairs), interval=1000 / fps)
    if out_path.endswith(".mp4"):
        try:
            anim.save(out_path, fps=fps, dpi=110)
        except Exception:
            out_path = out_path[:-4] + ".gif"
            anim.save(out_path, writer=PillowWriter(fps=fps), dpi=110)
    else:
        anim.save(out_path, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Real-vs-simulated streak video / contact sheet.")
    ap.add_argument("reduction_dir")
    ap.add_argument("calib_dir", help="dir produced by sdasim.calibrate")
    ap.add_argument("-o", "--out", default="streaks.gif", help="video path (.gif or .mp4)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--exposure", type=float, default=2.0)
    ap.add_argument("--box", type=int, default=128, help="fixed cutout size in px")
    ap.add_argument("--contact", default=None, help="also write a static contact sheet here")
    a = ap.parse_args(argv)
    print(f"building {a.n} real/sim matched pairs ...")
    pairs = make_pairs(a.reduction_dir, a.calib_dir, n=a.n, exposure=a.exposure, box=a.box)
    print(f"  {len(pairs)} pairs")
    if a.contact:
        save_contact_sheet(pairs, a.contact)
        print("WROTE", a.contact)
    out = save_video(pairs, a.out, fps=a.fps)
    print("WROTE", out)


if __name__ == "__main__":
    main()
