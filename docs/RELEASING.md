# Releasing sdasim

Publishing to PyPI is automated by `.github/workflows/python-publish.yml`. It
runs **only when a GitHub Release is published** — never on a merge to `main`,
never for draft releases, never for prereleases.

Authentication uses **PyPI Trusted Publishing** (OIDC). There is no
`PYPI_API_TOKEN` secret, and there should never be one.

Note this repository lives under the personal account **`zgazak`**, not the
`ssc-ai` organization — the trusted-publisher owner field must match.

## One-time setup

Both steps are required before the first automated release. Until they are
done, the publish job will fail with a 403 or hang waiting on a missing
environment.

### 1. Register the trusted publisher on PyPI

On <https://pypi.org/manage/project/sdasim/settings/publishing/>, add a GitHub
publisher with **exactly** these values:

| Field             | Value                |
| ----------------- | -------------------- |
| Owner             | `zgazak`             |
| Repository        | `sdasim`             |
| Workflow name     | `python-publish.yml` |
| Environment name  | `pypi`               |

The workflow name is the filename only — not the full path. The environment
name must match the workflow's `environment.name` exactly or PyPI will reject
the OIDC token.

### 2. Create the `pypi` environment with required reviewers

In **Settings → Environments → New environment**, name it `pypi`, then:

- Enable **Required reviewers** and add whoever is allowed to approve a PyPI
  upload.
- Optionally restrict **Deployment branches and tags** to tags matching `v*`.

This is the human gate. Publishing a GitHub Release starts the build, but the
upload to PyPI pauses at "Waiting for review" until a listed reviewer approves
it in the Actions run. Approval is per-run.

On a personal repository you are likely the only reviewer, so leave **"Prevent
self-review"** off — enabling it with a single reviewer deadlocks the release.

## Cutting a release

1. Bump `version` in `pyproject.toml` and move the `CHANGELOG.md`
   `[Unreleased]` section under the new version heading. Merge to `main`.
2. Tag the release commit as `v<version>` — e.g. `v0.3.1` for version `0.3.1`.
   `make tag` does this from `pyproject.toml` and pushes it.
3. On GitHub, create a Release pointing at that tag and click **Publish
   release**. Leave "Set as a pre-release" unchecked — prereleases are skipped
   by design.
4. The `release-build` job runs unattended: it verifies the tag, lints, tests,
   builds, and checks the artifacts.
5. Approve the `pypi` environment in the Actions run. The upload then happens.

## What the workflow verifies before uploading

- The release tag equals `v` + the `version` in `pyproject.toml`. A mismatch
  fails loudly instead of silently republishing the old version.
- That version is not already on PyPI (PyPI rejects re-uploads, and it is
  better to find out before the release is public).
- `ruff check src/ tests/` passes — the same as `make lint`.
- `pytest` passes — the full suite, the same as `make test`.
- The built wheel and sdist filenames carry the expected version.
- `twine check` passes on both artifacts.

## If something goes wrong

**Tag/version mismatch.** Fix `pyproject.toml` or the tag, delete the release
and tag, and redo. The `published` event does not re-fire when you edit an
existing release — you must publish a new one.

**Version already on PyPI.** Bump the version. PyPI never allows re-uploading a
filename, even after a delete; a bad release can only be *yanked*, not
unpublished.

**Publish job stuck on "Waiting for review".** That is the environment gate
working as intended. Approve it from the run page.

## A note on satsim

`satsim` is deliberately **not** an extra. It pins `astropy<6`, which cannot
coexist with the `catalogs`, `fits`, or `calibrate` extras (all require
`astropy>=6`). `Scene.from_satsim()` still converts flat configs with no satsim
installed; dynamic configs (`$sample`/`$ref`/`$generator`) raise an ImportError
telling the user to install satsim in a separate environment.

## Manual fallback

`make publish` still exists (clean tree → version check → test → build →
`twine upload` with a token from `~/.pypirc`), as does `make publish-test` for
TestPyPI. Prefer the workflow; the manual path bypasses the reviewer gate.
