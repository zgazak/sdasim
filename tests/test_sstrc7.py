"""Tests for the SSTRC7 wrapper.

The reader itself now lives in the ``sstrc7`` package, which has its own
exactness tests against a brute-force scan. What matters here is the contract
sdasim depends on: the structured-array return type, radian units, the
magnitude sentinel, and catalog path resolution. A small synthetic catalog is
written to a temp directory so these need no downloaded data.
"""

from __future__ import annotations

import numpy as np
import pytest

from sdasim import sstrc7 as wrapper
from sdasim.sstrc7 import _STAR_DTYPE

sstrc7 = pytest.importorskip("sstrc7", reason="needs the sstrc7 package")
pytest.importorskip("astropy", reason="query_by_los needs astropy")

from sstrc7._format import (  # noqa: E402
    BAND_INDEX,
    INDEX_FILENAME,
    MAG_ABSENT,
    MAS_PER_DEG,
    N_DEC_ZONES,
    N_RA_ZONES,
    RECORD_DTYPE,
    ZONE_HEIGHT_DEG,
    ZONE_WIDTH_DEG,
    zone_filename,
)


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    """A synthetic catalog: a 2-degree grid of stars with known magnitudes."""
    directory = tmp_path_factory.mktemp("sstrc7") / "catalog"
    directory.mkdir()

    ra_grid, dec_grid = np.meshgrid(np.arange(0.5, 360, 2.0), np.arange(-80.0, 80.1, 2.0))
    ra = ra_grid.ravel()
    dec = dec_grid.ravel()

    records = np.zeros(ra.size, dtype=RECORD_DTYPE)
    records["ra"] = np.round(ra * MAS_PER_DEG)
    records["dec"] = np.round(dec * MAS_PER_DEG)
    records["mag"] = MAG_ABSENT
    records["mag"][:, BAND_INDEX["Johnson_V"]] = 12000  # 12.0 mag
    records["mag"][:, BAND_INDEX["2MASS_J"]] = 10000  # 10.0 mag, for interpolation

    index = np.zeros((N_DEC_ZONES, N_RA_ZONES, 2), dtype="<u4")
    zone_of = np.clip(((dec + 90.0) / ZONE_HEIGHT_DEG).astype(int), 0, N_DEC_ZONES - 1)

    for zone_id in range(N_DEC_ZONES):
        in_zone = records[zone_of == zone_id]
        in_zone = in_zone[np.argsort(in_zone["ra"], kind="stable")]
        ra_zone = np.clip(
            (in_zone["ra"] / MAS_PER_DEG / ZONE_WIDTH_DEG).astype(int), 0, N_RA_ZONES - 1
        )
        offset = 0
        for r in range(N_RA_ZONES):
            count = int((ra_zone == r).sum())
            index[zone_id, r] = (offset, count)
            offset += count
        (directory / zone_filename(zone_id)).write_bytes(in_zone.tobytes())

    (directory / INDEX_FILENAME).write_bytes(index.tobytes())
    return directory


def test_query_by_min_max_returns_the_expected_dtype(catalog):
    out = wrapper.query_by_min_max(
        np.radians(10.0), np.radians(20.0), np.radians(-5.0), np.radians(5.0), str(catalog)
    )
    assert out.dtype == _STAR_DTYPE
    assert out.shape[0] > 0


def test_query_by_min_max_is_in_radians(catalog):
    out = wrapper.query_by_min_max(
        np.radians(10.0), np.radians(20.0), np.radians(-5.0), np.radians(5.0), str(catalog)
    )
    assert np.all(out["ra"] >= np.radians(10.0)) and np.all(out["ra"] <= np.radians(20.0))
    assert np.all(np.abs(out["dec"]) <= np.radians(5.0))


def test_query_by_min_max_clips_exactly(catalog):
    """The grid is 2 degrees apart, so the count is exactly predictable."""
    out = wrapper.query_by_min_max(
        np.radians(0.0), np.radians(10.0), np.radians(-4.0), np.radians(4.0), str(catalog)
    )
    # RA 0.5, 2.5, 4.5, 6.5, 8.5 and dec -4, -2, 0, 2, 4 -> 5 x 5.
    assert out.shape[0] == 25


def test_query_by_min_max_wraps_through_zero(catalog):
    out = wrapper.query_by_min_max(
        np.radians(358.0), np.radians(2.0), np.radians(-1.0), np.radians(1.0), str(catalog)
    )
    # RA 358.5, 0.5 at dec 0.
    assert out.shape[0] == 2
    assert np.allclose(np.degrees(np.sort(out["ra"])), [0.5, 358.5])


def test_magnitude_comes_from_the_priority_band(catalog):
    out = wrapper.query_by_min_max(
        np.radians(0.0), np.radians(10.0), np.radians(-4.0), np.radians(4.0), str(catalog)
    )
    assert np.allclose(out["mv"], 12.0, atol=1e-3)


def test_filter_center_interpolates(catalog):
    out = wrapper.query_by_min_max(
        np.radians(0.0),
        np.radians(10.0),
        np.radians(-4.0),
        np.radians(4.0),
        str(catalog),
        filter_center=1235.0,  # exactly 2MASS_J
    )
    assert np.allclose(out["mv"], 10.0, atol=1e-3)


def test_empty_region_returns_an_empty_typed_array(catalog):
    out = wrapper.query_by_min_max(
        np.radians(0.0), np.radians(0.001), np.radians(85.0), np.radians(86.0), str(catalog)
    )
    assert out.dtype == _STAR_DTYPE
    assert out.shape == (0,)


def test_clip_min_max_flag_is_accepted(catalog):
    """Kept in the signature for compatibility; clipping is always exact."""
    args = (np.radians(0.0), np.radians(10.0), np.radians(-4.0), np.radians(4.0), str(catalog))
    assert wrapper.query_by_min_max(*args, clip_min_max=False).shape == (
        wrapper.query_by_min_max(*args, clip_min_max=True).shape
    )


def test_query_by_los_returns_three_aligned_arrays(catalog):
    rows, cols, mv = wrapper.query_by_los(
        512, 512, 20.0, 20.0, 100.0, 10.0, rootPath=str(catalog), pad_mult=1.0
    )
    assert rows.shape == cols.shape == mv.shape
    assert rows.size > 0
    assert np.all(np.isfinite(rows)) and np.all(np.isfinite(cols))


def test_query_by_los_centres_the_boresight(catalog):
    rows, cols, _ = wrapper.query_by_los(
        512, 512, 20.0, 20.0, 100.0, 0.0, rootPath=str(catalog), pad_mult=0.0
    )
    # origin="center" shifts pixel coordinates so the boresight sits at h/2.
    assert rows.min() < 256 < rows.max()
    assert cols.min() < 256 < cols.max()


def test_query_by_los_rotation_moves_stars(catalog):
    """A rotated field covers a larger RA/Dec box, so counts differ too."""
    base = wrapper.query_by_los(512, 512, 20.0, 20.0, 100.0, 10.0, rootPath=str(catalog))
    turned = wrapper.query_by_los(512, 512, 20.0, 20.0, 100.0, 10.0, 30.0, rootPath=str(catalog))
    assert base[0].size and turned[0].size
    common = min(base[0].size, turned[0].size)
    assert not np.allclose(np.sort(base[0])[:common], np.sort(turned[0])[:common])


def test_missing_catalog_raises_a_clear_error(tmp_path):
    from sstrc7.query import CatalogNotFound

    with pytest.raises(CatalogNotFound):
        wrapper.query_by_min_max(0.0, 0.1, 0.0, 0.1, str(tmp_path / "nope"))


def test_path_resolution_prefers_the_environment(catalog, monkeypatch):
    monkeypatch.setenv("SDASIM_SSTRC7_PATH", str(catalog))
    out = wrapper.query_by_min_max(
        np.radians(0.0), np.radians(10.0), np.radians(-4.0), np.radians(4.0)
    )
    assert out.shape[0] == 25
