"""Tests for the opt-in empirical PSF/noise rendering path."""

import json

import numpy as np
import pytest
import torch

import sdasim
from sdasim.config import SceneConfig, SensorConfig, StarFieldConfig, StarMotionConfig
from sdasim.empirical import EmpiricalPSF, gaussian_kernel


@pytest.fixture
def basis(tmp_path):
    """A minimal synthetic PSF basis + noise model on disk."""
    K = 21
    ax = np.arange(K) - K // 2
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * 3.0**2))
    g = (g / g.sum()).astype("f4")
    rng = np.random.default_rng(0)
    n = 200
    shape = np.column_stack(
        [
            rng.normal(9.0, 0.8, n),
            rng.normal(8.5, 0.8, n),
            rng.normal(0.0, 0.05, n),
            rng.normal(0.0, 0.05, n),
            rng.normal(0.25, 0.02, n),
        ]
    ).astype("f4")
    pb = tmp_path / "psf.npz"
    np.savez(pb, mean_psf=g, shape=shape)
    nb = tmp_path / "noise.json"
    json.dump(
        dict(
            gain_e_per_adu=0.022,
            noise_floor_adu=425.0,
            background_gradient_pp_adu=27.0,
            hot_pixel_fraction=0.0,
        ),
        open(nb, "w"),
    )
    return str(pb), str(nb)


def _scene(pb, nb, psf, noise, **kw):
    s = SensorConfig(
        height=100,
        width=100,
        exposure=1.0,
        psf_model=psf,
        noise_model=noise,
        empirical_psf_path=pb,
        empirical_noise_path=nb,
        **kw,
    )
    return sdasim.Scene(
        SceneConfig(
            sensor=s, stars=StarFieldConfig(mode="bins"), star_motion=StarMotionConfig(), seed=0
        )
    )


def test_kernels_sum_to_one(basis):
    pb, _ = basis
    ep = EmpiricalPSF(pb)
    assert abs(float(ep.kernel(scale=0.0).sum()) - 1) < 1e-5
    assert abs(float(ep.kernel(rng=np.random.default_rng(1), scale=1.0).sum()) - 1) < 1e-5
    assert abs(float(gaussian_kernel(3.0, 21).sum()) - 1) < 1e-5


def test_empirical_render_runs(basis):
    pb, nb = basis
    img, meta = _scene(pb, nb, "empirical", "empirical").render(0)
    assert img.shape == (100, 100)
    assert torch.isfinite(img).all()
    assert meta["psf_model"] == "empirical" and meta["noise_model"] == "empirical"


def test_default_path_unchanged():
    s = SensorConfig(height=64, width=64)
    img, meta = sdasim.Scene(
        SceneConfig(sensor=s, stars=StarFieldConfig(mode="bins"), seed=0)
    ).render(0)
    assert img.shape == (64, 64)
    assert meta["psf_model"] == "gaussian" and meta["noise_model"] == "basic"


def test_empirical_differs_from_gaussian(basis):
    pb, nb = basis
    ig, _ = _scene(pb, nb, "gaussian", "basic").render(0)
    ie, _ = _scene(pb, nb, "empirical", "empirical").render(0)
    assert not torch.allclose(ig, ie)


def test_psf_param_sampling_and_extrapolation(basis):
    pb, nb = basis
    _, m = _scene(pb, nb, "empirical", "empirical", psf_param_scale=1.5).render(0)
    assert m["psf_params"] is not None and len(m["psf_params"]) == 3


def test_empirical_streak_runs(basis):
    pb, nb = basis
    sc = _scene(pb, nb, "empirical", "empirical")
    img, _ = sc.render(
        0,
        target_positions=torch.tensor([[50.0, 30.0]]),
        target_intensities=torch.tensor([5.0e5]),
        target_velocities=torch.tensor([[0.0, 20.0]]),
    )
    assert torch.isfinite(img).all()


def test_missing_path_raises():
    s = SensorConfig(height=32, width=32, psf_model="empirical")
    with pytest.raises(ValueError):
        sdasim.Scene(SceneConfig(sensor=s, stars=StarFieldConfig(mode="bins")))
