"""sdasim — Speed-optimized differentiable satellite scene simulator."""

from sdasim._version import __version__
from sdasim.batch import BatchRenderResult, render_scene_batch
from sdasim.config import (
    SceneConfig,
    SensorConfig,
    StarFieldConfig,
    StarMotionConfig,
    TargetConfig,
    load_config,
)
from sdasim.device import get_device, resolve_device, set_device
from sdasim.empirical import (
    EmpiricalNoise,
    EmpiricalPSF,
    render_frame_empirical,
)
from sdasim.fpa import analog_to_digital, eod_to_sigma, mv_to_pe, pe_to_mv
from sdasim.noise import gaussian_noise, poisson_noise
from sdasim.render import expand_motion, render_frame
from sdasim.scene import Scene
from sdasim.splat import (
    splat_elliptical_gaussian_batched,
    splat_gaussians,
    splat_gaussians_batched,
    splat_moffat_batched,
)


def __getattr__(name: str):
    if name == "io":
        import importlib

        _io = importlib.import_module("sdasim.io")
        globals()["io"] = _io  # cache so __getattr__ isn't called again
        return _io
    if name in ("sampler", "SceneDistribution", "random_scene"):
        import importlib

        _sampler = importlib.import_module("sdasim.sampler")
        globals()["sampler"] = _sampler
        globals()["SceneDistribution"] = _sampler.SceneDistribution
        globals()["random_scene"] = _sampler.random_scene
        return globals()[name]
    raise AttributeError(f"module 'sdasim' has no attribute {name!r}")


__all__ = [
    "__version__",
    # Device
    "get_device",
    "set_device",
    "resolve_device",
    # Core
    "splat_gaussians",
    "splat_gaussians_batched",
    "splat_moffat_batched",
    "splat_elliptical_gaussian_batched",
    "poisson_noise",
    "gaussian_noise",
    "analog_to_digital",
    "mv_to_pe",
    "pe_to_mv",
    "eod_to_sigma",
    # Render
    "render_frame",
    "expand_motion",
    "render_scene_batch",
    "BatchRenderResult",
    # Empirical (opt-in)
    "EmpiricalPSF",
    "EmpiricalNoise",
    "render_frame_empirical",
    # Scene
    "Scene",
    # Config
    "SceneConfig",
    "SensorConfig",
    "StarFieldConfig",
    "StarMotionConfig",
    "TargetConfig",
    "load_config",
    # I/O (lazy)
    "io",
    # Sampler (lazy)
    "sampler",
    "SceneDistribution",
    "random_scene",
]
