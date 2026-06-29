"""Tests for the SSTRC7 zone reader's compact (numpy) representation + RA clip.

These don't need the on-disk catalog: ``query_by_min_max``'s index lookup, zone
selection, and per-zone loading are monkeypatched with controlled data so the
boolean-mask RA clip and structured-array concatenation can be verified directly
(this is the behaviour that replaced the old list-of-dict / break-on-first-violation
loop).
"""

from __future__ import annotations

import numpy as np

from sdasim import sstrc7
from sdasim.sstrc7 import _STAR_DTYPE, _BytesBoundedCache


def _zone(ra, dec=0.0, mv=10.0):
    """Build a zone structured array from a list of RA values (radians)."""
    ra = np.asarray(ra, dtype="f8")
    out = np.empty(ra.shape[0], dtype=_STAR_DTYPE)
    out["ra"] = ra
    out["dec"] = dec
    out["mv"] = mv
    return out


def _patch(monkeypatch, zones, zone_arrays):
    """Stub the index/zone-selection/loader so query_by_min_max uses our data."""
    monkeypatch.setattr(sstrc7, "load_index", lambda *a, **k: object())
    monkeypatch.setattr(sstrc7, "select_zone", lambda *a, **k: zones)

    def fake_loader(zone_id, pos, length, bound, rootPath, filter_center=None):
        return zone_arrays[zone_id]

    monkeypatch.setattr(sstrc7, "_load_stars_for_zone", fake_loader)


def test_returns_structured_array(monkeypatch):
    zones = [{"id": 0, "pos": 0, "length": 3, "bound": "inside"}]
    _patch(monkeypatch, zones, {0: _zone([1.0, 1.1, 1.2])})
    out = sstrc7.query_by_min_max(0.0, 2.0, -1.0, 1.0)
    assert out.dtype == _STAR_DTYPE
    assert out.shape == (3,)
    assert np.allclose(out["ra"], [1.0, 1.1, 1.2])


def test_maxRA_truncates_at_first_violation(monkeypatch):
    # Ascending RA; clip should drop everything from the first ra > ra_max.
    zones = [{"id": 0, "pos": 0, "length": 5, "bound": "maxRA"}]
    _patch(monkeypatch, zones, {0: _zone([1.0, 1.5, 2.0, 2.5, 3.0])})
    out = sstrc7.query_by_min_max(0.0, 2.0, -1.0, 1.0)
    # 2.0 is not > 2.0; 2.5 is the first violation -> keep [1.0, 1.5, 2.0]
    assert np.allclose(out["ra"], [1.0, 1.5, 2.0])


def test_minRA_truncates_at_first_violation(monkeypatch):
    # _load_stars_for_zone returns minRA zones reversed (descending RA); clip
    # drops from the first ra < ra_min.
    zones = [{"id": 0, "pos": 0, "length": 5, "bound": "minRA"}]
    _patch(monkeypatch, zones, {0: _zone([3.0, 2.5, 2.0, 1.5, 1.0])})
    out = sstrc7.query_by_min_max(2.0, 5.0, -1.0, 1.0)
    # 2.0 is not < 2.0; 1.5 is the first violation -> keep [3.0, 2.5, 2.0]
    assert np.allclose(out["ra"], [3.0, 2.5, 2.0])


def test_inside_zone_not_clipped(monkeypatch):
    zones = [{"id": 0, "pos": 0, "length": 3, "bound": "inside"}]
    _patch(monkeypatch, zones, {0: _zone([0.1, 5.0, 9.9])})  # ignores ra bounds
    out = sstrc7.query_by_min_max(1.0, 2.0, -1.0, 1.0)
    assert out.shape == (3,)


def test_no_violation_keeps_all(monkeypatch):
    zones = [{"id": 0, "pos": 0, "length": 3, "bound": "maxRA"}]
    _patch(monkeypatch, zones, {0: _zone([1.0, 1.2, 1.4])})
    out = sstrc7.query_by_min_max(0.0, 2.0, -1.0, 1.0)
    assert out.shape == (3,)


def test_concatenates_multiple_zones(monkeypatch):
    zones = [
        {"id": 0, "pos": 0, "length": 2, "bound": "inside"},
        {"id": 1, "pos": 0, "length": 2, "bound": "maxRA"},
    ]
    arrays = {0: _zone([0.5, 0.6]), 1: _zone([1.0, 9.0])}  # zone 1: 9.0 clipped
    _patch(monkeypatch, zones, arrays)
    out = sstrc7.query_by_min_max(0.0, 2.0, -1.0, 1.0)
    assert np.allclose(out["ra"], [0.5, 0.6, 1.0])


def test_empty_when_no_zones(monkeypatch):
    _patch(monkeypatch, [], {})
    out = sstrc7.query_by_min_max(0.0, 2.0, -1.0, 1.0)
    assert out.dtype == _STAR_DTYPE
    assert out.shape == (0,)


# ---------------------------------------------------------------------------
# byte-bounded zone cache
# ---------------------------------------------------------------------------


def _arr(n):
    return np.zeros(n, dtype=_STAR_DTYPE)  # n stars = n*24 bytes


def test_bytes_cache_hit_and_miss():
    calls = []

    def load(z):
        calls.append(z)
        return _arr(10)

    cache = _BytesBoundedCache(load, max_bytes=10_000)
    cache(0)
    cache(0)  # served from cache
    cache(1)
    assert calls == [0, 1]
    info = cache.cache_info()
    assert info.hits == 1 and info.misses == 2 and info.currsize == 2
    assert info.currbytes == 2 * _arr(10).nbytes


def test_bytes_cache_evicts_over_budget():
    # Each zone is 100 stars = 2400 bytes; cap at 5000 -> holds 2 at a time.
    cap = 5000
    cache = _BytesBoundedCache(lambda z: _arr(100), max_bytes=cap)
    for z in range(5):
        cache(z)
        assert cache.cache_info().currbytes <= cap
    info = cache.cache_info()
    assert info.currsize == 2  # 2 * 2400 = 4800 <= 5000 < 7200


def test_bytes_cache_lru_order():
    cache = _BytesBoundedCache(lambda z: _arr(100), max_bytes=5000)  # holds 2
    cache(0)
    cache(1)
    cache(0)  # touch 0 -> now 1 is LRU
    cache(2)  # evicts 1, keeps 0
    # 0 should still be cached (a hit, no recompute); 1 evicted (a miss).
    calls = []
    cache2 = _BytesBoundedCache(lambda z: (calls.append(z) or _arr(100)), max_bytes=5000)
    cache2(0)
    cache2(1)
    cache2(0)
    cache2(2)
    cache2(0)  # still resident -> no recompute
    cache2(1)  # was evicted -> recompute
    assert calls == [0, 1, 2, 1]


def test_bytes_cache_keeps_one_when_single_zone_exceeds_cap():
    cache = _BytesBoundedCache(lambda z: _arr(1000), max_bytes=100)  # 24000 > 100
    out = cache(0)
    assert out.shape == (1000,)
    assert cache.cache_info().currsize == 1  # never evicts below 1


def test_bytes_cache_clear():
    cache = _BytesBoundedCache(lambda z: _arr(10), max_bytes=10_000)
    cache(0)
    cache.cache_clear()
    info = cache.cache_info()
    assert info.currsize == 0 and info.currbytes == 0 and info.hits == 0
