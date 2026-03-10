# sdasim

Speed-optimized differentiable satellite scene simulator.

A from-scratch reimplementation of [satsim](../satsim/) built for two use cases:

1. **In-the-loop rendering** inside neural network training (differentiable, GPU-first)
2. **Fast batch generation** of training datasets

## How it works

satsim renders by oversampling 3-5x, scattering point sources, FFT-convolving the full image with a PSF, then downsampling. This is high-fidelity but slow.

sdasim replaces that with **analytical Gaussian splatting**: for each source, directly compute its Gaussian PSF footprint on nearby pixels and accumulate. On a typical 512x512 scene with 1000 stars:

- satsim: O(1536x1536 x log(1536^2)) ~ **94M ops** for the PSF step
- sdasim: O(1000 x 11x11) ~ **121K ops**, fully parallel on GPU

Motion blur is handled by expanding each source into K sub-sources along its trajectory (vectorized, no Python loops), then splatting all at once.

## Install

```bash
# With uv (recommended)
uv sync --extra dev

# With pip
pip install -e ".[dev]"
```

**Required dependencies (3 total):** `torch>=2.2`, `numpy>=1.26`, `pyyaml>=6.0`

**Optional:** `astropy` (SSTR7/Gaia catalogs, FITS output), `satsim` (config converter)

## Quick start

### Render a sequence from config

```python
import sdasim

scene = sdasim.Scene.from_yaml("examples/rate_track.yaml")
images, annotations = scene.render_sequence()  # (8, 512, 512) tensor

# Write to disk (optional)
sdasim.io.write_sequence(images, annotations, "output/")
```

### Differentiable rendering in a training loop

```python
import torch
import sdasim

scene = sdasim.Scene.from_yaml("examples/speed_test.yaml")

# PSF sigma as a learnable parameter
learned_sigma = torch.tensor(1.5, requires_grad=True)
optimizer = torch.optim.Adam([learned_sigma], lr=0.01)

for epoch in range(100):
    img, meta = scene.render(frame_idx=0, psf_sigma=learned_sigma)
    loss = my_criterion(img)
    loss.backward()       # gradients flow through the renderer
    optimizer.step()
    optimizer.zero_grad()
```

### From a satsim config

```python
scene = sdasim.Scene.from_satsim("path/to/satsim_config.json", seed=42)
images, annotations = scene.render_sequence()
```

## Example configs

| Config | Scenario | Description |
|---|---|---|
| [`examples/rate_track.yaml`](examples/rate_track.yaml) | Rate-track | Sensor tracks target. Stars streak, targets are near-stationary. 8 frames, 512x512. |
| [`examples/sidereal.yaml`](examples/sidereal.yaml) | Sidereal | Sensor tracks star field. Stars are sharp, targets streak. 8 frames, 512x512. |
| [`examples/speed_test.yaml`](examples/speed_test.yaml) | Speed benchmark | 1000 frames at 256x256. No disk I/O. Simulates in-training renderer throughput. |

### Rate-track vs sidereal

In **rate-track** mode, the sensor slews to follow a target. Stars have non-zero `star_motion.translation` so they blur, while targets have near-zero velocity:

```yaml
star_motion:
  translation: [3.0, -5.0]   # stars streak
  temporal_osf: 100
targets:
  - velocity: [0.2, -0.1]    # target nearly stationary
```

In **sidereal** mode, the sensor is fixed on the sky. Stars have zero motion, targets streak:

```yaml
star_motion:
  translation: [0.0, 0.0]    # stars are sharp
  temporal_osf: 1
targets:
  - velocity: [8.0, 12.0]    # target streaks across FOV
```

## Config reference

```yaml
sensor:
  height: 512               # image height (pixels)
  width: 512                # image width (pixels)
  y_fov: 0.5                # vertical FOV (degrees)
  x_fov: 0.5                # horizontal FOV (degrees)
  exposure: 2.0              # exposure time (seconds)
  gap: 0.5                   # inter-frame gap (seconds)
  num_frames: 8              # frames in sequence
  zeropoint: 23.5            # sensor zeropoint (magnitude)
  psf_sigma: 1.5             # Gaussian PSF sigma (pixels)
  dark_current: 10.0         # dark current (e-/pixel/sec)
  read_noise: 10.0           # read noise (e- RMS)
  electronic_noise: 5.0      # electronic noise (e- RMS)
  background_mv: 21.0        # sky background brightness (mag)
  bias: 50.0                 # bias level (e-)
  gain: 8.0                  # conversion gain (e-/DN)
  fwc: 100000.0              # full-well capacity (e-)
  a2d_bias: 500.0            # A/D bias (DN)
  a2d_dtype: uint16          # output dtype

stars:
  mode: bins                 # "bins" (random), "sstr7", or "gaia"
  mv_bins: [6, 7, ..., 18]  # N+1 magnitude bin edges
  density: [0.04, ..., 285] # N star densities (stars/deg^2/bin)

star_motion:
  rotation: 0.0              # rotation rate (rad/sec)
  translation: [0.0, 0.0]   # [row, col] drift rate (px/sec)
  temporal_osf: 100          # sub-steps for motion blur

targets:
  - mode: line               # trajectory mode
    origin: [0.5, 0.5]      # start position (fractional image coords)
    velocity: [8.0, 12.0]   # [row, col] velocity (px/sec)
    mv: 12.0                 # visual magnitude

seed: 42                     # random seed (null for non-deterministic)
device: auto                 # "auto", "cpu", or "cuda"
enable_shot_noise: true      # Poisson photon noise
enable_read_noise: true      # Gaussian read + electronic noise
```

## Differentiability

The renderer is differentiable with respect to:

- **Source positions** (row, col) — for astrometric fitting
- **Source intensities** (PE) — for photometric calibration
- **PSF sigma** — for PSF model learning

Noise models preserve gradients:
- Poisson noise: straight-through estimator (STE)
- Gaussian noise: reparameterization trick

`torch.floor` in A/D conversion has zero gradient. For training, use the pre-A/D signal (the star/target signal tensors returned by `render_frame()`).

## Architecture

```
src/sdasim/
  splat.py       # Gaussian splatting kernel (core hot path)
  render.py      # Full pipeline: splat + noise + A/D + expand_motion()
  scene.py       # Scene class: slow setup -> fast render
  config.py      # Flat dataclasses + YAML loader
  noise.py       # Differentiable Poisson (STE) + Gaussian (reparam)
  fpa.py         # A/D, mv<->pe, eod_to_sigma
  stars.py       # Star catalogs: random bins, SSTR7, Gaia
  targets.py     # Target trajectories
  device.py      # GPU-first device management
  io.py          # Optional FITS/JSON writers
  _compat.py     # satsim config converter
```

## Tests

```bash
uv run pytest              # 71 tests, <1s
uv run pytest -v           # verbose
uv run pytest -k splat     # just splatting tests
uv run pytest -k gradient  # gradient flow tests
```

## License

MIT
