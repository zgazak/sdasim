# sdasim — Speed-Optimized Differentiable Satellite Scene Simulator

## What is sdasim?

A from-scratch reimplementation of satsim optimized for speed and differentiability.
Uses analytical Gaussian splatting instead of FFT convolution — ~800x fewer ops for
the PSF step on typical scenes.

## Quick Start

```bash
uv sync --extra dev   # Install with dev deps
uv run pytest         # Run tests (71 tests, <1s)
```

## Architecture

```
src/sdasim/
  splat.py       # Core: analytical Gaussian splatting kernel (THE hot path)
  render.py      # Pipeline: splat → noise → A/D. Contains expand_motion() for motion blur
  scene.py       # Scene class: slow setup (catalog load) → fast render (pure tensor ops)
  config.py      # Flat dataclasses + YAML loader (no $sample/$ref)
  noise.py       # Differentiable Poisson (STE) + Gaussian (reparam trick)
  fpa.py         # A/D conversion, mv↔pe, eod_to_sigma
  stars.py       # Star catalogs: random bins, SSTR7, Gaia (lazy import)
  targets.py     # Target trajectory computation
  device.py      # GPU-first device management
  io.py          # Optional FITS/JSON writers (not in hot path)
  _compat.py     # satsim config converter
```

## Key Design Decisions

1. **Gaussian splatting, not FFT**: For each source, compute a small (2R+1)² Gaussian
   footprint and scatter_add onto the image. O(N × K²) vs O(HW × log(HW)).

2. **Native resolution**: No oversampling. Sub-pixel accuracy via analytical Gaussian
   evaluation at integer pixel centers.

3. **Motion blur via sub-source expansion**: `expand_motion()` converts N sources with
   velocities into N×t_osf stationary sub-sources. Fully vectorized, no Python loops.

4. **Differentiable through everything**: Positions, intensities, sigma all have gradients.
   Poisson noise uses STE, Gaussian noise uses reparameterization trick.
   A/D `torch.floor` breaks gradients — use pre-A/D signal for training losses.

5. **Scene separates setup from rendering**: `Scene.__init__()` is slow (catalogs, allocation).
   `Scene.render()` is fast (pure tensor ops, GPU-ready).

## Dependencies

- **Required**: torch>=2.2, numpy>=1.26, pyyaml>=6.0
- **Optional**: astropy (SSTR7/Gaia/FITS), satsim (converter only)
- **Dev**: pytest, ruff

## torch.compile() Notes

The hot path (splat → noise → A/D) uses `clamp` chains instead of `torch.where` for
compile compatibility. One graph break remains in `splat_gaussians` when `radius=None`
(auto-computed via `.item()`). Pass `radius` explicitly to avoid it:

```python
splat_gaussians(H, W, pos, ints, sigma=1.5, radius=5)  # no graph break
```

## Testing

```bash
uv run pytest tests/ -v           # All tests
uv run pytest tests/test_splat.py # Just splatting
uv run pytest tests/test_gradients.py  # Gradient flow tests
```

## Config

sdasim uses flat YAML config (no dynamic keys). See `config.py` for dataclass definitions.
Use `Scene.from_satsim()` or `_compat.from_satsim_config()` to convert satsim configs.

## Common Patterns

```python
import sdasim

# Differentiable rendering in training loop
scene = sdasim.Scene.from_yaml("config.yaml")
img, meta = scene.render(frame_idx=0, psf_sigma=learned_sigma)

# Batch generation
imgs, metas = scene.render_sequence()
sdasim.io.write_sequence(imgs, metas, "output/")
```
