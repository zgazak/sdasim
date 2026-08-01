"""Tests for the geometric pupil-sampling defocus/aberration model.

The closed-form expectations come from the same geometric-optics relations the
model implements (see optics.py docstring): blur radius = defocus/(2N), donut
hole ratio = obscuration, HFD slope = sqrt((1+eps^2)/2)/N, astigmatic axis
flip, and the parity facts (pure defocus and coma are side-blind; spherical
is not).
"""

from __future__ import annotations

import math

import pytest
import torch

from sdasim.config import AberrationsConfig, OpticsConfig, SceneConfig
from sdasim.optics import OpticsModel, pupil_samples
from sdasim.scene import Scene
from sdasim.splat import splat_gaussians

FNUM = 8.0
PITCH = 9.0
LAM_UM = 0.550
SIGMA = 1.5
H = W = 256
CENTER = (H / 2.0, W / 2.0)


def make_model(obscuration=0.35, **aberrations) -> OpticsModel:
    cfg = OpticsConfig(
        enabled=True,
        fnumber=FNUM,
        obscuration=obscuration,
        wavelength_nm=LAM_UM * 1e3,
        pixel_pitch_um=PITCH,
        max_sub_sources=10_000_000,
        aberrations=AberrationsConfig(**aberrations),
    )
    return OpticsModel(cfg, SIGMA, H, W)


def render_star(model: OpticsModel, defocus_um, pos=None, intensity=1e6):
    positions = torch.tensor([pos or [H / 2.0, W / 2.0]])
    intensities = torch.tensor([float(intensity)])
    epos, eint = model.expand(positions, intensities, defocus_um, CENTER)
    return splat_gaussians(H, W, epos, eint, SIGMA)


def moments(img: torch.Tensor):
    """(total, centroid_r, centroid_c, Mrr, Mcc) flux-weighted image moments."""
    total = img.sum()
    rows = torch.arange(img.shape[0], dtype=img.dtype).unsqueeze(1)
    cols = torch.arange(img.shape[1], dtype=img.dtype).unsqueeze(0)
    cr = (img * rows).sum() / total
    cc = (img * cols).sum() / total
    mrr = (img * (rows - cr) ** 2).sum() / total
    mcc = (img * (cols - cc) ** 2).sum() / total
    return total.item(), cr.item(), cc.item(), mrr.item(), mcc.item()


def hfd(img: torch.Tensor) -> float:
    """Half-flux diameter about the flux centroid, in pixels."""
    total, cr, cc, _, _ = moments(img)
    rows = torch.arange(img.shape[0], dtype=img.dtype).unsqueeze(1)
    cols = torch.arange(img.shape[1], dtype=img.dtype).unsqueeze(0)
    r = torch.sqrt((rows - cr) ** 2 + (cols - cc) ** 2).reshape(-1)
    f = img.reshape(-1)
    order = torch.argsort(r)
    cum = torch.cumsum(f[order], dim=0)
    idx = int(torch.searchsorted(cum, torch.tensor(total / 2.0)))
    return 2.0 * r[order][idx].item()


def blur_radius_px(defocus_um: float) -> float:
    """Geometric blur-circle radius delta/(2N), in pixels."""
    return defocus_um / (2.0 * FNUM) / PITCH


class TestPupilSamples:
    def test_quadrature_is_exact_for_flux_and_second_moment(self):
        p, w = pupil_samples(256, obscuration=0.35)
        assert abs(p.shape[0] - 256) < 24  # ring rounding lands near the request
        assert abs(w.sum().item() - 1.0) < 1e-6
        rho = p.norm(dim=1)
        assert rho.min() >= 0.35
        assert rho.max() <= 1.0 + 1e-6
        # RMS nodes + area weights integrate rho^2 exactly: (eps^2 + 1)/2
        weighted = (w * rho**2).sum().item()
        assert abs(weighted - (0.35**2 + 1.0) / 2.0) < 1e-6

    def test_antipodal_symmetry(self):
        """Every sample's antipode is in the set (exact parity across focus).

        Distances via explicit broadcasting — torch.cdist's matmul path loses
        ~3e-4 of precision on near-zero distances in float32.
        """
        for eps in (0.0, 0.2, 0.35):
            p, _ = pupil_samples(64, obscuration=eps)
            gap = (p.unsqueeze(1) + p.unsqueeze(0)).norm(dim=-1).min(dim=1).values.max()
            assert gap.item() < 1e-5

    def test_outer_ring_is_closed_circle(self):
        """The rim samples are evenly spaced — no iris-blade scalloping."""
        p, _ = pupil_samples(256, obscuration=0.35)
        rho = p.norm(dim=1)
        rim = p[rho > rho.max() - 1e-5]
        theta = torch.atan2(rim[:, 1], rim[:, 0]).sort().values
        gaps = torch.diff(theta)
        assert torch.allclose(gaps, gaps[0].expand_as(gaps), atol=1e-4)


class TestExpand:
    def test_in_focus_is_passthrough(self):
        model = make_model()
        pos = torch.tensor([[100.0, 100.0]])
        ints = torch.tensor([1000.0])
        epos, eint = model.expand(pos, ints, 0.0, CENTER)
        assert epos is pos and eint is ints
        assert model.last_k == 1

    def test_flux_conserved(self):
        model = make_model()
        pos = torch.rand(50, 2) * 200 + 28
        ints = torch.rand(50) * 1e5
        epos, eint = model.expand(pos, ints, 1000.0, CENTER)
        assert epos.shape[0] == eint.shape[0] == 50 * model.last_k
        assert torch.allclose(eint.sum(), ints.sum(), rtol=1e-5)
        img = render_star(model, 1000.0)
        assert abs(img.sum().item() - 1e6) < 100.0

    def test_k_scales_with_defocus(self):
        model = make_model()
        render_star(model, 300.0)
        k_small = model.last_k
        render_star(model, 3000.0)
        assert model.last_k > k_small


class TestDefocusGeometry:
    def test_rms_radius_clear_aperture(self):
        """Uniform-disk blur (eps=0): rms^2 = R^2/2 + 2*sigma^2."""
        model = make_model(obscuration=0.0)
        defocus = 1152.0  # -> R = 8 px
        img = render_star(model, defocus)
        _, _, _, mrr, mcc = moments(img)
        r = blur_radius_px(defocus)
        expected = r**2 / 2.0 + 2.0 * SIGMA**2
        assert abs((mrr + mcc) - expected) / expected < 0.05

    def test_donut_hole(self):
        """Obscured pupil far from focus: central intensity well below the ring."""
        model = make_model(obscuration=0.4)
        img = render_star(model, 3000.0)  # R ~ 20.8 px, eps*R ~ 8.3 px >> sigma
        center_val = img[H // 2, W // 2].item()
        assert center_val < 0.5 * img.max().item()

    def test_hfd_slope(self):
        """HFD ~ (defocus/N) * sqrt((1+eps^2)/2) far from focus."""
        eps = 0.35
        model = make_model(obscuration=eps)
        defocus = 3240.0  # R = 22.5 px = 15 sigma
        expected = 2.0 * blur_radius_px(defocus) * math.sqrt((1 + eps**2) / 2.0)
        assert abs(hfd(render_star(model, defocus)) - expected) / expected < 0.07

    def test_waves_rms_defocus_chain(self):
        """a waves RMS of static Z4 == commanded defocus of 16*sqrt(3)*N^2*a*lambda."""
        a = 0.5
        static = make_model(defocus=a)
        commanded = make_model()
        delta = 16.0 * math.sqrt(3.0) * FNUM**2 * a * LAM_UM
        img_a = render_star(static, 0.0)
        img_b = render_star(commanded, delta)
        assert torch.allclose(img_a, img_b, rtol=1e-4, atol=1e-3)


class TestAberrations:
    def test_astigmatism_axis_flips_across_focus(self):
        model = make_model(obscuration=0.0, astig_0=0.4)
        _, _, _, mrr_in, mcc_in = moments(render_star(model, -1000.0))
        _, _, _, mrr_out, mcc_out = moments(render_star(model, 1000.0))
        assert (mcc_in - mrr_in) * (mcc_out - mrr_out) < 0
        assert abs(mcc_in - mrr_in) > 1.0

    def test_coma_is_side_blind(self):
        """Parity: coma gradient is even in p, so +/- defocus give the same image."""
        model = make_model(coma_x=0.3)
        img_in = render_star(model, -800.0)
        img_out = render_star(model, 800.0)
        assert torch.allclose(img_in, img_out, rtol=1e-4, atol=1e-2)

    def test_spherical_breaks_side_blindness(self):
        model = make_model(spherical=0.4)
        img_in = render_star(model, -1000.0)
        img_out = render_star(model, 1000.0)
        rel_diff = (img_in - img_out).abs().sum() / img_in.sum()
        assert rel_diff.item() > 0.05

    def test_field_curvature_blurs_corner(self):
        model = make_model(obscuration=0.0, field_curvature=1.5)
        img_center = render_star(model, 0.0, pos=[H / 2.0, W / 2.0])
        img_corner = render_star(model, 0.0, pos=[40.0, 40.0])
        _, _, _, mrr_ctr, mcc_ctr = moments(img_center)
        _, _, _, mrr_cor, mcc_cor = moments(img_corner)
        assert (mrr_cor + mcc_cor) > (mrr_ctr + mcc_ctr) + 1.0


class TestDifferentiability:
    def test_gradient_flows_to_defocus_command(self):
        model = make_model()
        defocus = torch.tensor(500.0, requires_grad=True)
        img = render_star(model, defocus)
        loss = (img**2).sum()
        loss.backward()
        assert defocus.grad is not None
        assert torch.isfinite(defocus.grad)
        assert defocus.grad.abs().item() > 0


class TestSceneIntegration:
    def _scene_config(self, enabled=True, **aberrations) -> SceneConfig:
        cfg = SceneConfig()
        cfg.seed = 7
        cfg.sensor.height = cfg.sensor.width = 128
        cfg.sensor.psf_sigma = SIGMA
        cfg.optics = OpticsConfig(
            enabled=enabled,
            fnumber=FNUM,
            obscuration=0.35,
            pixel_pitch_um=PITCH,
            aberrations=AberrationsConfig(**aberrations),
        )
        return cfg

    def test_enabled_in_focus_matches_disabled(self):
        _, sig_off, _, _ = Scene(self._scene_config(enabled=False)).render_signals(0)
        _, sig_on, _, _ = Scene(self._scene_config(enabled=True)).render_signals(0)
        assert torch.equal(sig_off, sig_on)

    def test_defocus_override_and_metadata(self):
        scene = Scene(self._scene_config())
        _, meta = scene.render(0, defocus_um=800.0)
        assert meta["defocus_um"] == 800.0
        assert meta["pupil_samples"] >= 4
        assert meta["defocus_blur_px"] > 0
        _, sharp, _, _ = scene.render_signals(0)
        _, blurred, _, _ = scene.render_signals(0, defocus_um=800.0)
        assert not torch.allclose(sharp, blurred)

    def test_defocus_override_requires_optics(self):
        scene = Scene(self._scene_config(enabled=False))
        with pytest.raises(ValueError, match="optics.enabled"):
            scene.render(0, defocus_um=100.0)

    def test_optics_rejects_empirical_path(self):
        cfg = self._scene_config()
        cfg.sensor.psf_model = "empirical"
        with pytest.raises(ValueError, match="optics"):
            Scene(cfg)


class TestReviewRegressions:
    """Pins for defects found in adversarial review of the initial diff."""

    def test_batch_render_applies_optics(self):
        """render_scene_batch must not bypass the optics model (was: silent sharp stars)."""
        from sdasim.batch import render_scene_batch

        def build():
            cfg = SceneConfig()
            cfg.seed = 3
            cfg.sensor.height = cfg.sensor.width = 128
            cfg.sensor.psf_sigma = SIGMA
            cfg.optics = OpticsConfig(
                enabled=True,
                fnumber=FNUM,
                obscuration=0.35,
                pixel_pitch_um=PITCH,
                aberrations=AberrationsConfig(defocus=2.0),
            )
            return Scene(cfg)

        scene = build()
        _, star_ref, _, _ = scene.render_signals(0)
        batch = render_scene_batch([build()])
        assert torch.allclose(batch.star_signal[0], star_ref)

    def test_gradient_survives_near_focus(self):
        """A grad-tracking defocus inside the skip zone must stay in the graph."""
        model = make_model()
        defocus = torch.tensor(30.0, requires_grad=True)  # blur < 0.3*sigma
        img = render_star(model, defocus)
        assert model.last_k > 1  # passthrough disabled for grad-tracking input
        (img**2).sum().backward()
        assert defocus.grad is not None and torch.isfinite(defocus.grad)

    def test_psf_sigma_override_drives_gate_and_k(self):
        """The per-render psf_sigma must inform the skip gate and rung choice."""
        model = make_model()
        pos = torch.tensor([[H / 2.0, W / 2.0]])
        ints = torch.tensor([1000.0])
        # blur ~0.28 px: negligible vs sigma=1.5, significant vs sigma=0.5
        model.expand(pos, ints, 40.0, CENTER)
        assert model.last_k == 1
        model.expand(pos, ints, 40.0, CENTER, psf_sigma=0.5)
        assert model.last_k > 1
        # Sharper rendered PSF needs denser pupil sampling at the same defocus
        model.expand(pos, ints, 2000.0, CENTER)
        k_default = model.last_k
        model.expand(pos, ints, 2000.0, CENTER, psf_sigma=0.75)
        assert model.last_k > k_default

    def test_defocus_tensor_dtype_normalized(self):
        model = make_model(field_curvature=1.0)
        pos = torch.tensor([[40.0, 40.0]])
        ints = torch.tensor([1000.0])
        epos, _ = model.expand(pos, ints, torch.tensor(800.0, dtype=torch.float64), CENTER)
        assert epos.dtype == torch.float32

    def test_sub_source_budget_caps_k(self):
        """f/2.25-style blowup: N x K must respect max_sub_sources (OOM guard)."""
        cfg = OpticsConfig(
            enabled=True,
            fnumber=2.25,
            obscuration=0.6,
            pixel_pitch_um=10.0,
            max_sub_sources=200_000,
            aberrations=AberrationsConfig(),
        )
        model = OpticsModel(cfg, SIGMA, H, W)
        m = 9000  # e.g. 450 stars x t_osf 20 motion expansion
        pos = torch.rand(m, 2) * 200
        ints = torch.rand(m)
        with pytest.warns(UserWarning, match="max_sub_sources"):
            epos, eint = model.expand(pos, ints, 5000.0, CENTER)
        # The coarsest sample set may overshoot a *time* budget slightly rather
        # than drop the defocus entirely, but the overshoot is bounded.
        assert epos.shape[0] <= OpticsModel.SKIP_OVERSHOOT * cfg.max_sub_sources
        assert model.last_k <= model._rungs[0]["k"]
        assert model.last_clamped
        assert torch.allclose(eint.sum(), ints.sum(), rtol=1e-4)  # flux still exact
        # Small frames at the same defocus are NOT clamped
        few = model.expand(pos[:10], ints[:10], 5000.0, CENTER)[0]
        assert not model.last_clamped
        assert few.shape[0] > 10 * 256

    def test_time_budget_calibrates_to_a_sub_source_cap(self):
        """max_render_seconds is converted via a measured splat rate."""
        def build(seconds):
            cfg = OpticsConfig(
                enabled=True, fnumber=FNUM, obscuration=0.35, pixel_pitch_um=PITCH,
                max_render_seconds=seconds, aberrations=AberrationsConfig(),
            )
            return OpticsModel(cfg, SIGMA, H, W)

        slim, generous = build(0.05), build(2.0)
        assert slim.splat_rate is not None and slim.splat_rate > 0
        assert slim.max_sub_sources > 0
        # A 40x larger budget must buy materially more sub-sources (allowing
        # generous slack for timing noise on a loaded machine).
        assert generous.max_sub_sources > 4 * slim.max_sub_sources
        # ... and the tighter budget clamps where the generous one does not.
        pos, ints = torch.rand(400, 2) * 200, torch.ones(400)
        with pytest.warns(UserWarning, match="max_render_seconds"):
            slim.expand(pos, ints, 4000.0, CENTER)
        assert slim.last_clamped
        assert slim.last_k <= generous.last_k or not generous.last_clamped

    def test_explicit_count_skips_calibration(self):
        cfg = OpticsConfig(
            enabled=True, fnumber=FNUM, obscuration=0.35, pixel_pitch_um=PITCH,
            max_sub_sources=123_456, aberrations=AberrationsConfig(),
        )
        model = OpticsModel(cfg, SIGMA, H, W)
        assert model.splat_rate is None  # no benchmark run
        assert model.max_sub_sources == 123_456

    def test_ladder_covers_full_range_regardless_of_config(self):
        """No max_defocus_um: the ladder always reaches the sampling ceiling.

        Actual counts land within a few of each power of two (ring rounding),
        and duplicate rungs are collapsed.
        """
        ks = [r["k"] for r in make_model()._rungs]
        assert ks[-1] > 4000
        assert ks == sorted(set(ks))  # strictly increasing, no duplicates

    def test_empty_frame_reports_frame_level_k(self):
        """m==0 must not leave stale (or misleadingly reset) sampling metadata."""
        model = make_model()
        empty_pos = torch.zeros(0, 2)
        empty_int = torch.zeros(0)
        model.expand(empty_pos, empty_int, 2000.0, CENTER)
        assert model.last_k > 1  # the K this frame's blur would use
        model.expand(empty_pos, empty_int, 0.0, CENTER)
        assert model.last_k == 1


class TestConfigLoading:
    def test_yaml_round_trip(self, tmp_path):
        yaml_text = """
sensor: {height: 64, width: 64}
optics:
  enabled: true
  fnumber: 10.0
  obscuration: 0.3
  aberrations: {astig_0: 0.08, spherical: 0.1, field_curvature: 0.5}
"""
        path = tmp_path / "scene.yaml"
        path.write_text(yaml_text)
        from sdasim.config import load_config

        cfg = load_config(path)
        assert cfg.optics.enabled is True
        assert cfg.optics.fnumber == 10.0
        assert cfg.optics.aberrations.astig_0 == 0.08
        assert cfg.optics.aberrations.field_curvature == 0.5
