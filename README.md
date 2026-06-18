# sdasim

Speed-optimized differentiable satellite scene simulator.

A from-scratch, GPU-first satellite scene simulator built for two use cases:

1. **In-the-loop rendering** inside neural network training (differentiable, GPU-first)
2. **Fast batch generation** of training datasets

## How it works

Conventional high-fidelity simulators render by oversampling 3-5x, scattering point sources, FFT-convolving the full image with a PSF, then downsampling. This is accurate but slow.

sdasim replaces that with **analytical Gaussian splatting**: for each source, directly compute its Gaussian PSF footprint on nearby pixels and accumulate. On a typical 512x512 scene with 1000 stars:

- FFT-convolution baseline: O(1536x1536 x log(1536^2)) ~ **94M ops** for the PSF step
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

**Optional:** `astropy` (SSTR7/Gaia catalogs, FITS output), `satsim` (config converter),
`[calibrate]` extra = `astropy`+`scipy`+`matplotlib` (empirical-mode calibration; see
[Empirical rendering](#empirical-data-calibrated-rendering--for-sim-to-real-pretraining))

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

## Empirical (data-calibrated) rendering — for sim-to-real pretraining

The default path renders Gaussian PSFs with parametric noise. **Empirical mode** instead
calibrates the forward model from *real paired collects* (sidereal point-PSF frames + rate
streak frames) so generated streaks carry the instrument's real PSF wings, noise, and
atmospheric streak texture. The goal is **pretraining a streak detector** (e.g. starcsp)
without the sim-to-real gap. It is fully **opt-in** (`psf_model="empirical"`,
`noise_model="empirical"`); the default Gaussian/basic path is unchanged.

What gets calibrated:

- **PSF** — the measured mean PSF (real diffraction wings), warpable by a size/ellipticity
  distribution sampled from the data (and extrapolatable past it for diversity).
- **Noise** — gain, white floor (with per-frame sky-brightness variation), background
  gradient, hot-pixel rate. (The noise measured white, so no correlated-noise generator.)
- **Streak texture** — four broadband atmospheric components, tuned to the measured 2D
  texture power spectrum (open-band silicon → fine speckle is chromatically washed out, so
  the texture is low-order/broadband):
  - *scintillation* — along-track intensity flicker
  - *tip/tilt* — 2D position wander
  - *seeing-breathing* — symmetric PSF size pulse (no one-sided edge artifacts)
  - *hf-plateau* — post-PSF, band-limited "whispy" texture filling the 3–9 px band the PSF
    would otherwise erase
  …with a **flux floor** so texture dims but never guts the streak core.

### Install (adds astropy/scipy/matplotlib for the offline calibration step)

```bash
uv sync --extra calibrate          # torch + astropy + scipy + matplotlib
```

### 1. Calibrate (offline): real reductions → bases + config

```bash
python -m sdasim.calibrate <senpai_reduction_dir> -o calib --gain 0.022 --plots
```

`<senpai_reduction_dir>` is per-collect senpai outputs (`frame_*_*.json` + `*_processed.fits`).
`--gain` is the detector gain in e-/ADU (e.g. from `senpai.cli.measure_gain`). Writes into
`calib/`: `psf_basis.npz`, `noise_model.json`, `streak_model.json`, `calibration.yaml`, and
(with `--plots`) `param_distributions.png` + `streak_pairs.png` (real-vs-sim snapshots).

### 2. Generate training data (matched + slightly extended past measured)

```python
import sdasim
from sdasim.sampler import SceneDistribution, random_scene
from sdasim.batch import render_scene_batch

dist = SceneDistribution.from_calibration("calib", extend=1.3)   # extend>1 reaches past real
scenes = [sdasim.Scene(random_scene(dist, seed=k)) for k in range(64)]
batch = render_scene_batch(scenes)        # (64, H, W) empirical, GPU, per-scene reproducible

# …or a single labeled frame straight from the generated config:
img, meta = sdasim.Scene.from_yaml("calib/calibration.yaml").render(0)
```

Streak **orientation is sampled uniform [0, 180)** (the full rotation — a streak at θ and
θ+180 are identical) rather than the campaign's measured angles; rate/length and per-streak
texture amplitudes are drawn from the measured distributions, widened by `extend`.

### 3. Visualize (sanity-check realism)

```bash
python -m sdasim.streak_vis <senpai_reduction_dir> calib -o streaks.mp4 --n 60
```

Side-by-side **real | empirical | gaussian** matched-streak video (or `--contact sheet.png`
for a static grid). MP4 if `ffmpeg` is present, else GIF.

### Does it actually help? (the test that matters)

The model is tuned to *look* right, but the arbiter is the **A/B transfer test**: pretrain
the streak detector on empirical vs gaussian vs ImageNet-init data and compare downstream
detection metrics. Plain Gaussian sdasim did not help pretraining in earlier experiments;
the empirical texture model exists to close that gap — verify it on your data.

### Notes

- The calibration step is numpy/scipy/astropy (no torch needed); the render step is torch.
  The `[calibrate]` extra installs both, so one env runs everything.
- Console scripts (after install): `sdasim-calibrate`, `sdasim-streak-vis`.
- Texture knobs live in `streak_model.json` as `tex_*` (scint/tiptilt/seeing/hf rms, the hf
  band + slope, and `tex_floor_frac`); edit there to retune without recalibrating.

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
  scene.py       # Scene class: slow setup -> fast render (default + empirical dispatch)
  config.py      # Flat dataclasses + YAML loader
  noise.py       # Differentiable Poisson (STE) + Gaussian (reparam)
  fpa.py         # A/D, mv<->pe, eod_to_sigma
  stars.py       # Star catalogs: random bins, SSTR7, Gaia
  targets.py     # Target trajectories
  sampler.py     # Random scene generation (+ SceneDistribution.from_calibration)
  batch.py       # Batched multi-scene rendering (Gaussian fused + empirical)
  device.py      # GPU-first device management
  io.py          # Optional FITS/JSON writers
  _compat.py     # satsim config converter
  # --- empirical (data-calibrated) mode ---
  empirical.py   # measured PSF + noise + 4-component streak texture renderer
  calibrate.py   # CLI: real reductions -> psf_basis/noise/streak bases + config + plots
  streak_vis.py  # real-vs-empirical-vs-gaussian matched video / contact sheet
```

## Tests

```bash
uv run pytest              # 102 tests, ~1-3s
uv run pytest -v           # verbose
uv run pytest -k splat     # just splatting tests
uv run pytest -k gradient  # gradient flow tests
uv run pytest -k empirical # empirical PSF/noise/streak-texture tests
```

## License

MIT
