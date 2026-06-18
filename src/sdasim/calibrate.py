"""sdasim.calibrate — ingest senpai reductions -> empirical bases + a ready-to-use config.

Consolidates the calibration measured from paired sidereal/rate collects:
  * point-PSF eigen-basis + interpretable shape params (sidereal frames)
  * white-noise floor / gain / background gradient / hot pixels
  * streak model: along-track intensity texture, cross-track wobble, atmospheric-boil
    speckle, and the rate-correlation reference
  * empirical parameter DISTRIBUTIONS (mean+std) for sampling, with an `extend` factor so
    generated data covers and reaches slightly past the measured range.

Writes into OUTDIR:  psf_basis.npz, noise_model.json, streak_model.json, calibration.yaml
which sdasim's empirical render mode (psf_model="empirical", noise_model="empirical")
consumes directly.

Pure numpy/scipy/astropy (no torch) — runs as an offline step; the outputs feed the
torch render path.

CLI:
    python -m sdasim.calibrate <reduction_dir> -o <outdir> --gain 0.022 [--max-collects N]
                               [--extend 1.3] [--flats <raw_flats_dir>]

`reduction_dir` is a directory of per-collect senpai outputs, each containing
frame_0_sidereal.json, frame_{1,2}_rate.json and the matching *_processed.fits.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

FLOOR_DEFAULT = 425.0


# ----------------------------------------------------------------- collect I/O
def _collect_dirs(root):
    return sorted(
        d
        for d in glob.glob(root + "/*")
        if os.path.isdir(d) and os.path.exists(d + "/frame_0_sidereal.json")
    )


def _sidereal_fits(cd):
    f = glob.glob(cd + "/*frame1of3_sidereal_processed.fits")
    return f[0] if f else None


def _rate_streak(cd):
    """Return (rate_fits, pixel_length, angle_deg, exptime) for a measured rate streak."""
    for fn, fr in [("frame_1_rate.json", "frame2of3"), ("frame_2_rate.json", "frame3of3")]:
        p = cd + "/" + fn
        if not os.path.exists(p):
            continue
        j = json.load(open(p)) or {}
        s = j.get("streak")
        if s and s.get("pixel_length") and s.get("sine_angle") is not None:
            c = glob.glob(cd + f"/*{fr}*_rate_processed.fits")
            if c:
                et = float((j.get("original_frame_header") or {}).get("EXPTIME", 1.0) or 1.0)
                ang = float(np.degrees(np.arctan2(s["sine_angle"], s["cosine_angle"])))
                return c[0], float(s["pixel_length"]), ang, et
    return None, None, None, None


# ----------------------------------------------------------------- Phase 1: PSF
def build_psf_basis(root, out, max_collects=200, half=20, nkeep=20):
    from astropy.io import fits
    from scipy.ndimage import shift as ndshift

    S = 2 * half + 1
    yy, xx = np.mgrid[0:S, 0:S]
    rmap = np.hypot(xx - half, yy - half)
    aper = rmap < 14.0

    def shape_params(rn):
        w = np.clip(rn, 0, None) * aper
        t = w.sum() + 1e-12
        mxx = ((xx - half) ** 2 * w).sum() / t
        myy = ((yy - half) ** 2 * w).sum() / t
        mxy = ((xx - half) * (yy - half) * w).sum() / t
        tr = mxx + myy + 1e-9
        rad = np.sqrt(max(0.0, (tr / 2) ** 2 - (mxx * myy - mxy**2)))
        return (
            2.3548 * np.sqrt(max(tr / 2 + rad, 1e-6)),
            2.3548 * np.sqrt(max(tr / 2 - rad, 1e-6)),
            (mxx - myy) / tr,
            2 * mxy / tr,
            rn[rmap < 3].sum() / (rn[rmap < 9].sum() + 1e-9),
        )

    def recenter(stamp):
        s = stamp - np.median(stamp)
        sg = 1.4826 * np.median(np.abs(s - np.median(s))) + 1e-6
        bad = (rmap > 12) & (s > 6 * sg)
        if bad.any():
            s = s.copy()
            s[bad] = 0.0
        cy, cx = float(half), float(half)
        for _ in range(6):
            d2 = (yy - cy) ** 2 + (xx - cx) ** 2
            wt = np.clip(s, 0, None) * np.exp(-d2 / (2 * 4.5**2))
            tot = wt.sum()
            if tot <= 0:
                return None
            ncy, ncx = (yy * wt).sum() / tot, (xx * wt).sum() / tot
            if abs(ncy - cy) + abs(ncx - cx) < 5e-3:
                cy, cx = ncy, ncx
                break
            cy, cx = ncy, ncx
        s = ndshift(s, (half - cy, half - cx), order=3, mode="constant", cval=0.0)
        f = s[aper].sum()
        return s / f if f > 0 else None

    stamps, M, SP = [], [], []
    cds = _collect_dirs(root)[:max_collects]
    for ci, cd in enumerate(cds):
        fp = _sidereal_fits(cd)
        if not fp:
            continue
        sj = json.load(open(cd + "/frame_0_sidereal.json"))
        dets = (sj.get("starfield") or {}).get("detections") or []
        if not dets:
            continue
        oh = sj.get("original_frame_header", {}) or {}
        alt, az, expt = oh.get("ALT", np.nan), oh.get("AZ", np.nan), oh.get("EXPTIME", np.nan)
        dx = np.array([d["x"] for d in dets])
        dy = np.array([d["y"] for d in dets])
        cc = np.array([(d["counts"] if d.get("counts") is not None else -1.0) for d in dets], "f8")
        with fits.open(fp, memmap=True) as hd:
            data = hd[0].data
            H, W = data.shape
            got = 0
            for i in np.argsort(cc)[::-1]:
                if got >= 25:
                    break
                x, y = dx[i], dy[i]
                if x < half + 4 or y < half + 4 or x > W - half - 4 or y > H - half - 4:
                    continue
                d2 = (dx - x) ** 2 + (dy - y) ** 2
                d2[i] = 1e18
                if np.sqrt(d2.min()) < 55:
                    continue
                iy, ix = int(round(y)), int(round(x))
                s33 = np.asarray(data[iy - 16 : iy + 17, ix - 16 : ix + 17], "f8")
                if s33.shape != (33, 33):
                    continue
                med = np.median(s33)
                pk = s33[14:19, 14:19].max() - med
                if pk <= 0 or pk >= 52000:
                    continue
                nb = (s33[15, 16] + s33[17, 16] + s33[16, 15] + s33[16, 17]) / 4 - med
                sg = 1.4826 * np.median(np.abs(s33 - med)) + 1e-6
                if nb / pk < 0.40 or pk / sg < 10:
                    continue
                rn = recenter(
                    np.asarray(data[iy - half : iy + half + 1, ix - half : ix + half + 1], "f8")
                )
                if rn is None or not np.all(np.isfinite(rn)):
                    continue
                sp = shape_params(rn)
                if (
                    not (6.0 < sp[0] < 18.0)
                    or sp[1] < 4.0
                    or (sp[2] ** 2 + sp[3] ** 2) ** 0.5 > 0.35
                ):
                    continue
                stamps.append(rn.astype("f4"))
                M.append([x, y, alt, az, pk / sg, expt, ci])
                SP.append(sp)
                got += 1
    stamps = np.array(stamps, "f4")
    M = np.array(M, "f8")
    SP = np.array(SP, "f8")
    N = len(stamps)
    if N < 30:
        raise RuntimeError(f"only {N} PSF stars found; need >=30")
    X = stamps.reshape(N, -1).astype("f8")
    mean_psf = X.mean(0)
    Xc = X - mean_psf
    gpy, gpx = np.gradient(mean_psf.reshape(S, S))
    G, _ = np.linalg.qr(np.column_stack([gpx.ravel(), gpy.ravel()]))
    Xc = Xc - (Xc @ G) @ G.T
    U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    expl = sv**2 / (sv**2).sum()
    C = (U * sv)[:, :nkeep]
    np.savez_compressed(
        out + "/psf_basis.npz",
        mean_psf=mean_psf.reshape(S, S).astype("f4"),
        eigen_psfs=Vt[:nkeep].reshape(nkeep, S, S).astype("f4"),
        explained=expl[:nkeep].astype("f4"),
        coeffs=C.astype("f4"),
        coeff_std=C.std(0).astype("f4"),
        shape=SP.astype("f4"),
        meta=M.astype("f4"),
        stamps=stamps,
        half=half,
        aper=14.0,
    )
    fwhm = 0.5 * (SP[:, 0] + SP[:, 1])
    return dict(
        n_stars=int(N),
        n_frames=int(len(np.unique(M[:, 6]))),
        fwhm=[float(fwhm.mean()), float(fwhm.std())],
        e1=[float(SP[:, 2].mean()), float(SP[:, 2].std())],
        e2=[float(SP[:, 3].mean()), float(SP[:, 3].std())],
        conc=[float(SP[:, 4].mean()), float(SP[:, 4].std())],
        var_explained=[float(e) for e in expl[:6]],
    )


# ----------------------------------------------------------------- Phase 2: noise
def measure_noise(root, out, gain, n_frames=10):
    from astropy.io import fits

    sids = []
    for cd in _collect_dirs(root):
        fp = _sidereal_fits(cd)
        if fp:
            sids.append((cd, fp))
        if len(sids) >= n_frames:
            break
    B = 224
    win = np.hanning(B)[:, None] * np.hanning(B)[None, :]
    fy, fx = np.mgrid[-B // 2 : B // 2, -B // 2 : B // 2]
    fr = np.hypot(fx, fy)
    frb = np.arange(0, B // 2, 2)
    psd = np.zeros((B, B))
    nb = 0
    rho1 = []
    sky = []
    grad = []
    frame_floors = []
    for cd, fp in sids:
        sj = json.load(open(cd + "/frame_0_sidereal.json"))
        dets = (sj.get("starfield") or {}).get("detections") or []
        xs = np.array([d["x"] for d in dets])
        ys = np.array([d["y"] for d in dets])
        d = fits.getdata(fp).astype("f4")
        H, W = d.shape
        got = 0
        frame_sky = []
        for yy0 in range(150, H - B - 150, 220):
            for xx0 in range(150, W - B - 150, 220):
                if (
                    len(xs)
                    and (
                        (xs > xx0 - 30)
                        & (xs < xx0 + B + 30)
                        & (ys > yy0 - 30)
                        & (ys < yy0 + B + 30)
                    ).any()
                ):
                    continue
                box = d[yy0 : yy0 + B, xx0 : xx0 + B].astype("f8")
                m = np.median(box)
                s = 1.4826 * np.median(np.abs(box - m))
                if (np.percentile(box, 99.95) - m) / s > 5:
                    continue
                b0 = box - box.mean()
                psd += np.abs(np.fft.fftshift(np.fft.fft2(b0 * win))) ** 2
                num = (b0[:, :-1] * b0[:, 1:]).sum() + (b0[:-1, :] * b0[1:, :]).sum()
                rho1.append(num / (2 * (b0**2).sum()))
                sky.append(s)
                frame_sky.append(s)
                nb += 1
                got += 1
                if got >= 8:
                    break
            if got >= 8:
                break
        if frame_sky:
            frame_floors.append(float(np.median(frame_sky)))
        # background gradient
        step = 16
        ds = d[::step, ::step].astype("f8")
        gy, gx = np.mgrid[0 : ds.shape[0], 0 : ds.shape[1]] * step
        mask = np.ones(ds.shape, bool)
        for x, y in zip(xs, ys):
            mask[(np.abs(gy - y) < 30) & (np.abs(gx - x) < 30)] = False
        mm = np.median(ds[mask])
        sg = 1.4826 * np.median(np.abs(ds[mask] - mm))
        mask &= np.abs(ds - mm) < 4 * sg
        Xn = gx[mask] / W * 2 - 1
        Yn = gy[mask] / H * 2 - 1
        A = np.column_stack([np.ones(mask.sum()), Xn, Yn, Xn**2, Xn * Yn, Yn**2])
        bt, *_ = np.linalg.lstsq(A, ds[mask], rcond=None)
        Xa = gx / W * 2 - 1
        Ya = gy / H * 2 - 1
        surf = bt[0] + bt[1] * Xa + bt[2] * Ya + bt[3] * Xa**2 + bt[4] * Xa * Ya + bt[5] * Ya**2
        grad.append(float(surf.max() - surf.min()))
    psd /= max(nb, 1)
    rps = np.array([psd[(fr >= frb[k]) & (fr < frb[k + 1])].mean() for k in range(len(frb) - 1)])
    white = np.median(rps[len(rps) // 2 :])
    floor = float(np.median(sky))
    rho = float(np.median(rho1))
    # hot pixels from a processed frame high-pass
    from scipy.ndimage import median_filter

    d0 = fits.getdata(sids[0][1]).astype("f8")
    resid = d0 - median_filter(d0, 3)
    sg = 1.4826 * np.median(np.abs(resid))
    hot = float((np.abs(resid) > 8 * sg).mean())
    floor_std = float(np.std(frame_floors)) if len(frame_floors) > 1 else 0.0
    model = dict(
        gain_e_per_adu=float(gain),
        noise_floor_adu=round(floor, 1),
        noise_floor_std_adu=round(floor_std, 1),
        lag1_autocorr=round(rho, 4),
        background_gradient_pp_adu=round(float(np.median(grad)), 1),
        hot_pixel_fraction=hot,
        psd_low_over_high=float(rps[:3].mean() / white),
        stage="row/col-median-subtracted ADU",
        render_recipe="noise_std=sqrt(noise_floor_adu^2 + max(signal_adu,0)/gain); white Gaussian; "
        "+ background gradient; + hot pixels",
    )
    json.dump(model, open(out + "/noise_model.json", "w"), indent=2)
    np.savez_compressed(out + "/noise_model_psd.npz", radial_psd=rps.astype("f4"))
    return model


# ----------------------------------------------------------------- Phase 3: streak
def measure_streak(root, out, max_collects=200, extend=1.3, speckle_amp=1.8):
    from astropy.io import fits
    from scipy.ndimage import gaussian_filter, gaussian_filter1d, rotate

    rate, trms, wob, sid_fw, str_fw, ang = [], [], [], [], [], []
    ic, wc = [], []
    cds = _collect_dirs(root)[:max_collects]
    for cd in cds:
        rfp, rL, a, et = _rate_streak(cd)
        if not rfp:
            continue
        sj = json.load(open(cd + "/frame_0_sidereal.json"))
        fw = ((sj.get("starfield") or {}).get("fwhm_stats") or {}).get("median_fwhm")
        rate.append(rL / et if et > 0 else np.nan)
        ang.append(a)
        rstk = None
        for fn in ("frame_1_rate.json", "frame_2_rate.json"):
            p = cd + "/" + fn
            if os.path.exists(p):
                st = (json.load(open(p)) or {}).get("streak")
                if st and st.get("fwhm"):
                    rstk = st
                    break
        if fw and rstk:
            sid_fw.append(fw)
            str_fw.append(float(rstk["fwhm"]))
        if rL < 70:
            continue
        im = fits.getdata(rfp).astype("f4")
        shalf = int(rL * 0.7 + 30)
        bw = shalf + 15
        sm = gaussian_filter(np.where(im > 40000, 0, im), 3)
        sm[:bw] = 0
        sm[-bw:] = 0
        sm[:, :bw] = 0
        sm[:, -bw:] = 0
        sy, sx = np.unravel_index(np.argmax(sm), sm.shape)
        cut = im[sy - shalf : sy + shalf + 1, sx - shalf : sx + shalf + 1].astype("f8")
        cut -= np.median(cut)
        R = rotate(cut, a, reshape=False, order=1)
        h = R.shape[0] // 2
        Is = R[h - 8 : h + 9].sum(0)
        m = Is > 0.55 * Is.max()
        idx = np.where(m)[0]
        if len(idx) < 30 or Is[idx].mean() < 10 * FLOOR_DEFAULT:
            continue
        a0, b0 = idx[0] + 3, idx[-1] - 3
        Ib = Is[a0:b0] / np.median(Is[a0:b0])
        yl = np.arange(17) - 8
        pc = []
        for j in range(a0, b0):
            col = np.clip(R[h - 8 : h + 9, j], 0, None)
            pc.append((yl * col).sum() / col.sum() if col.sum() > 0 else 0)
        pc = np.array(pc)
        pc = pc - np.polyval(np.polyfit(np.arange(len(pc)), pc, 1), np.arange(len(pc)))

        def clen(x):
            x = gaussian_filter1d(x, 4)
            x = x - x.mean()
            ac = np.correlate(x, x, "full")[len(x) - 1 :]
            ac /= ac[0]
            b = np.where(ac < 0.5)[0]
            return float(b[0]) if len(b) else float(len(ac))

        trms.append(float(np.std(gaussian_filter1d(Ib, 2))))
        wob.append(float(np.std(pc)))
        ic.append(clen(Ib))
        wc.append(clen(pc))
    rate = np.array(rate)
    broad = float(np.nanmedian(np.array(str_fw) / np.array(sid_fw))) if sid_fw else 1.0
    ref_rate = float(np.nanmedian([r for r in rate if r and r > 50]))

    def md(a):
        a = np.array([x for x in a if np.isfinite(x)])
        return [float(a.mean()), float(a.std())] if len(a) else [0.0, 0.0]

    model = dict(
        tracking_broadening_med=broad,
        reference_rate_px_s=ref_rate if ref_rate == ref_rate else 110.0,
        # --- 4-component atmospheric streak-texture model (calibrated to the measured 2D PSD) ---
        # Open-band silicon detectors chromatically wash out fine speckle, so the texture is
        # low-order/broadband: scintillation (intensity), tip/tilt (position), seeing-breathing
        # (symmetric PSF size pulse), and a post-PSF whispy high-frequency plateau (the measured
        # 3-9px band). Broadband (power-law) processes -> structure at all scales, no clean cutoff.
        tex_scint_rms=0.11,
        tex_tiptilt_rms=0.8,
        tex_seeing_rms=0.10,
        tex_hf_rms=0.13,
        tex_beta=2.0,
        tex_decon_px=1.3,
        tex_hf_klo=0.08,
        tex_hf_khi=0.30,
        tex_hf_slope=1.2,
        tex_floor_frac=0.45,
        # sampling distributions (mean, std) + extend factor;
        # orientation UNIFORM for pretraining
        dist_rate_px_s=md(rate),
        dist_length_px=md([r * 1.0 for r in rate if np.isfinite(r)]),
        orientation_deg_range=[0.0, 180.0],
        orientation_sample="uniform",
        extend_factor=float(extend),
        n_streaks=int(len(trms)),
        model="streak = instantaneous PSF along path + scintillation + tip/tilt + seeing-breathing "
        "+ whispy hf-plateau (post-PSF, band-limited red); calibrated to the measured 2D PSD",
    )
    json.dump(model, open(out + "/streak_model.json", "w"), indent=2)
    return model


# ----------------------------------------------------------------- config
def write_config(out, extend, psf, noise, streak):
    cfg = f"""# sdasim empirical calibration -- generated by sdasim.calibrate
# Point a Scene at these to render data matching (and slightly beyond) the measured instrument.
sensor:
  # --- empirical mode ---
  psf_model: empirical
  noise_model: empirical
  empirical_psf_path: {out}/psf_basis.npz
  empirical_noise_path: {out}/noise_model.json
  empirical_streak_path: {out}/streak_model.json
  psf_param_scale: {extend}        # sample PSF shape from real dist; >1 extends past measured
  apply_rowcol_median: true        # full-frame stage; set false for small patch renders
  gain: {noise["gain_e_per_adu"]}
  # --- set these to your detector / scene ---
  height: 512
  width: 512
  exposure: 2.0
# Data-generation policy (sample these per scene; +/- {extend}x std reaches past measured):
#   streak ORIENTATION: UNIFORM [0,180) deg  <-- full streak rotation; do NOT use measured angles
#   streak LENGTH px  : rate * exposure  (longer = faster at fixed exposure; texture scales w/ rate)
#   streak rate px/s  : {streak["dist_rate_px_s"]}   (extend slightly; clamp >0)
#   streak texture    : scint {streak["tex_scint_rms"]} / tip-tilt {streak["tex_tiptilt_rms"]} /
#                       seeing {streak["tex_seeing_rms"]} / hf {streak["tex_hf_rms"]} (4-knob)
#   PSF FWHM          : {psf["fwhm"]}   e1/e2: {psf["e1"]}/{psf["e2"]}
#   noise floor ADU   : {noise["noise_floor_adu"]}  (white; lag1={noise["lag1_autocorr"]})
stars:
  mode: bins
"""
    open(out + "/calibration.yaml", "w").write(cfg)


def make_param_dist_fig(out_dir, extend):
    """Overview figure: measured parameter distributions + the (extended) sampling dist."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = np.load(out_dir + "/psf_basis.npz")
    sh = b["shape"]
    fwhm = 0.5 * (sh[:, 0] + sh[:, 1])
    e1 = sh[:, 2]
    e2 = sh[:, 3]
    conc = sh[:, 4]
    streak = json.load(open(out_dir + "/streak_model.json"))
    fig, ax = plt.subplots(2, 4, figsize=(18, 9))

    def hist(ax, d, name, unit):
        mu, sd = d.mean(), d.std()
        ax.hist(d, 30, density=True, alpha=0.55, label="measured")
        x = np.linspace(mu - 4 * extend * sd, mu + 4 * extend * sd, 200)

        def gg(s):
            return np.exp(-((x - mu) ** 2) / (2 * s * s)) / (s * np.sqrt(2 * np.pi))

        ax.plot(x, gg(sd), "C0--", lw=1, label="fit")
        ax.plot(x, gg(extend * sd), "C3-", lw=2, label=f"sampling x{extend}")
        ax.set_title(f"{name}: {mu:.2f}+-{sd:.2f}{unit}", fontsize=9)
        ax.legend(fontsize=6)

    def curve(ax, ms, name, unit):
        mu, sd = ms
        x = np.linspace(max(0, mu - 4 * extend * sd), mu + 4 * extend * sd, 200)

        def gg(s):
            return np.exp(-((x - mu) ** 2) / (2 * s * s))

        ax.plot(x, gg(sd), "C0--", label="measured")
        ax.plot(x, gg(extend * sd), "C3-", lw=2, label=f"sampling x{extend}")
        ax.set_title(f"{name}: {mu:.2f}+-{sd:.2f}{unit}", fontsize=9)
        ax.legend(fontsize=6)

    hist(ax[0, 0], fwhm, "PSF FWHM", "px")
    hist(ax[0, 1], e1, "PSF e1", "")
    hist(ax[0, 2], e2, "PSF e2", "")
    hist(ax[0, 3], conc, "PSF conc", "")
    curve(ax[1, 0], streak["dist_rate_px_s"], "streak rate", "px/s")
    curve(ax[1, 1], streak["dist_length_px"], "streak length", "px")
    ax[1, 2].axis("off")
    ax[1, 2].text(
        0.0,
        0.95,
        "streak texture model:\n"
        + "\n".join(f"  {k} = {streak[k]}" for k in streak if k.startswith("tex_")),
        fontsize=7,
        va="top",
        family="monospace",
    )
    ax[1, 3].scatter(e1, e2, s=6, alpha=0.3)
    ax[1, 3].set_aspect("equal")
    ax[1, 3].set_xlabel("e1")
    ax[1, 3].set_ylabel("e2")
    ax[1, 3].set_title("ellipticity joint   (orientation: UNIFORM [0,180))", fontsize=8)
    fig.suptitle(
        "Empirical parameter distributions: measured (blue) + sampling, extended (red)", fontsize=13
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir + "/param_distributions.png", dpi=100)
    plt.close(fig)


def make_plots(reduction_dir, out_dir, extend, n_pairs=12):
    make_param_dist_fig(out_dir, extend)
    print(f"WROTE {out_dir}/param_distributions.png")
    try:
        from sdasim import streak_vis

        pairs = streak_vis.make_pairs(reduction_dir, out_dir, n=n_pairs)
        streak_vis.save_contact_sheet(pairs, out_dir + "/streak_pairs.png")
        print(f"WROTE {out_dir}/streak_pairs.png")
    except Exception as e:  # torch/render unavailable in this env
        print(f"  (skipped streak_pairs snapshot: {e})")


def calibrate(reduction_dir, out_dir, gain, max_collects=200, extend=1.3, speckle_amp=1.8):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[1/3] PSF basis from {reduction_dir} ...")
    psf = build_psf_basis(reduction_dir, out_dir, max_collects=max_collects)
    print(
        f"      {psf['n_stars']} stars / {psf['n_frames']} frames; "
        f"FWHM {psf['fwhm'][0]:.1f}+-{psf['fwhm'][1]:.1f}px"
    )
    print("[2/3] noise model ...")
    noise = measure_noise(reduction_dir, out_dir, gain=gain)
    print(f"      floor {noise['noise_floor_adu']} ADU  lag1 {noise['lag1_autocorr']}  gain {gain}")
    print("[3/3] streak model ...")
    streak = measure_streak(
        reduction_dir, out_dir, max_collects=max_collects, extend=extend, speckle_amp=speckle_amp
    )
    print(
        f"      {streak['n_streaks']} streaks; rate {streak['dist_rate_px_s'][0]:.0f}px/s "
        f"ref {streak['reference_rate_px_s']:.0f}; "
        f"broadening {streak['tracking_broadening_med']:.2f}"
    )
    write_config(out_dir, extend, psf, noise, streak)
    print(
        f"WROTE {out_dir}/{{psf_basis.npz, noise_model.json, streak_model.json, calibration.yaml}}"
    )
    return dict(psf=psf, noise=noise, streak=streak)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Calibrate sdasim empirical bases from senpai reductions."
    )
    ap.add_argument(
        "reduction_dir",
        help="dir of per-collect senpai outputs (frame_*_*.json + *_processed.fits)",
    )
    ap.add_argument("-o", "--out", required=True, help="output dir for bases + config")
    ap.add_argument(
        "--gain", type=float, default=1.0, help="detector gain e-/ADU (e.g. senpai measure_gain)"
    )
    ap.add_argument("--max-collects", type=int, default=200)
    ap.add_argument(
        "--extend", type=float, default=1.3, help="sampling reach past measured (std multiplier)"
    )
    ap.add_argument(
        "--speckle-amp", type=float, default=1.8, help="atmospheric-boil speckle amplitude"
    )
    ap.add_argument(
        "--plots",
        action="store_true",
        help="also write param_distributions.png + streak_pairs.png (real vs sim)",
    )
    a = ap.parse_args(argv)
    if a.gain == 1.0:
        print(
            "WARNING: --gain not set; using 1.0. "
            "Measure it (e.g. senpai measure_gain) for correct shot noise."
        )
    calibrate(
        a.reduction_dir,
        a.out,
        gain=a.gain,
        max_collects=a.max_collects,
        extend=a.extend,
        speckle_amp=a.speckle_amp,
    )
    if a.plots:
        print("[plots] overview distributions + real/sim streak snapshots ...")
        make_plots(a.reduction_dir, a.out, a.extend)


if __name__ == "__main__":
    main()
