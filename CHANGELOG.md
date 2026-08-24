# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file starts after 0.3.0. Earlier history was not tracked here — see the
[git tags](https://github.com/zgazak/sdasim/tags) for prior versions.

## [Unreleased]

## [0.5.0] - 2026-08-23

### Added

- **Geometric defocus and aberration model** (`src/sdasim/optics.py`, opt-in via
  `optics.enabled`). Each source is expanded into sub-sources displaced by the
  transverse ray aberration of K annular-pupil samples — the same trick motion
  blur already used, applied over pupil area instead of time. Produces donuts
  with central obscuration, and supports field-constant aberrations
  (`defocus`, `astig_0`/`astig_45`, `coma_x`/`coma_y`, `spherical`) plus field
  terms (`field_curvature`, `coma_radial`, `astig_radial`) in waves RMS
  (Noll-normalized). Commanded focus is the `defocus_um` render override and is
  differentiable end-to-end. Costs nothing when disabled or in focus.
  New exports: `OpticsModel`, `pupil_samples`, `OpticsConfig`,
  `AberrationsConfig`.
- **Calibration frame types** — `Scene.render(frame_type=...)` accepts
  `"light"` (default), `"dark"`, `"bias"`, and `"flat"`. Calibration frames
  render only what the shutter or lamp admits: dark is dark current + bias,
  bias uses the zero-second convention (no dark current regardless of the
  commanded exposure), and flat is uniform illumination from the new
  `sensor.flat_radiance` (pe/pixel/s) times exposure. Sources, sky, and the
  optics expansion are all skipped. `sdasim.FRAME_TYPES` is exported so callers
  can feature-detect, and render metadata carries `frame_type` (written as
  `IMAGETYP` by the FITS writer). The empirical PSF/noise path raises for
  non-light frames — it is a holistic fit and cannot separate components.
- **Wall-clock cost budget** — scene-level `max_render_seconds` (default 0.5)
  bounds motion and defocus sub-source work against a splat-throughput
  measurement taken at construction. Motion sampling is served first; the
  optics stage divides what remains. Degradations are flagged in render
  metadata (`star_osf_clamped`, `pupil_sampling_clamped`). The budget never
  drops physics: the coarsest pupil sampling always renders, over budget and
  warned, rather than silently returning an in-focus frame.

### Changed

- **`star_motion.temporal_osf` now defaults to automatic** (`int | None`, was a
  fixed `int = 100`). Motion sampling is derived per frame by inverting the
  Gaussian-comb ripple law against that frame's own noise floor, so bright
  streaks on dark sky sample finely and faint streaks on bright sky sample
  coarsely. Set an int to pin the old fixed behavior, or `sampling_quality` to
  pin the spacing in units of `psf_sigma` for reproducible renders. Existing
  configs will render slightly differently, and usually faster.

### Fixed

- **Bias no longer receives shot noise.** `bias` was added before the Poisson
  draw in both composition paths (`render.render_frame` and the batch
  renderer), so the electronic readout offset was picking up photon variance it
  cannot physically have. It is now added after. Frames render with slightly
  less variance than before — with the default `bias = 50`, about 50 pe² less,
  against a `read_noise = 10` floor of 100 pe².

## [0.4.0] - 2026-08-11

### Added

- **Continuous integration** (`.github/workflows/ci.yml`) — lint, tests on
  Python 3.10/3.11/3.12, and a build check on every push and pull request to
  `main`/`dev`. The repository previously had no CI.
- **Automated PyPI publishing** (`.github/workflows/python-publish.yml`) via
  Trusted Publishing (OIDC), so no API token is stored. It triggers only when a
  GitHub Release is published — never on a merge, and never for draft or
  prerelease releases — and pauses for manual approval on the `pypi`
  environment before uploading. See `docs/RELEASING.md`.

### Removed

- **The `compat` extra.** `pip install sdasim[compat]` no longer installs
  satsim. satsim pins `astropy<6`, which cannot coexist with the `catalogs`,
  `fits`, or `calibrate` extras (all require `astropy>=6`), so the combination
  was never installable together and had to be declared as a uv conflict.

  **No functionality was removed.** `Scene.from_satsim()` and
  `sdasim._compat` remain and still convert flat satsim configs with satsim
  absent. Only dynamic configs (`$sample`/`$ref`/`$generator`) need satsim, and
  those now raise an ImportError directing you to install it in a separate
  environment.

### Changed

- With satsim no longer declared as an extra, all remaining extras resolve
  together: the `[tool.uv] conflicts` block is gone and `make sync` / `make
  test` now use `--all-extras` (which additionally picks up `orbital`, so the
  satkit-dependent orbit tests run rather than skip).

- The MIT license copyright holder is now "sdasim maintainers" rather than an
  individual.
- `CLAUDE.md` is no longer tracked; it is local editor state and is now
  gitignored. It was already excluded from the sdist.

### Fixed

- Import sort in `tests/test_sstrc7.py`, which `make lint` flagged.
