# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file starts after 0.3.0. Earlier history was not tracked here — see the
[git tags](https://github.com/zgazak/sdasim/tags) for prior versions.

## [Unreleased]

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

### Fixed

- Import sort in `tests/test_sstrc7.py`, which `make lint` flagged.
