"""Tests for catalog discovery of streaking field satellites (offline).

These exercise discover_streaks / load_catalog and the Scene catalog hook using
a small local TLE file — no network, no Space-Track credentials.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sdasim.config import (
    CatalogConfig,
    SceneConfig,
    SensorConfig,
    SiteConfig,
    StarFieldConfig,
    StarMotionConfig,
)

satkit = pytest.importorskip("satkit")

from sdasim.orbits import (  # noqa: E402
    LoadedCatalog,
    discover_streaks,
    load_catalog,
    propagate,
)

# ISS (NORAD 25544) — hardcoded for offline tests
ISS_L1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9002"
ISS_L2 = "2 25544  51.6400 200.0000 0007417  35.0000 325.0000 15.49000000000000"

SITE = SiteConfig(latitude=40.0, longitude=-105.0, altitude=1600.0)


def _iss_catalog(diameter_m: float = 2.0) -> LoadedCatalog:
    tle = satkit.TLE.from_lines([ISS_L1, ISS_L2])
    return LoadedCatalog(tles=[tle], diameters=np.array([diameter_m]))


def _write_tle_file(tmp_path) -> str:
    f = tmp_path / "catalog.3le"
    f.write_text(ISS_L1 + "\n" + ISS_L2 + "\n")
    return str(f)


# ---------------------------------------------------------------------------
# load_catalog (local-file source)
# ---------------------------------------------------------------------------


class TestLoadCatalog:
    def test_load_local_file(self, tmp_path):
        from sdasim.orbits import _cache_dir

        cfg = CatalogConfig(enabled=True, source=_write_tle_file(tmp_path), default_diameter=1.5)
        cat = load_catalog(cfg)
        assert len(cat.tles) == 1
        assert cat.diameters.shape == (1,)
        # Without a SATCAT cache, diameter falls back to the configured default.
        # (Guarded so a real SATCAT left in the temp dir by a live fetch can't flake.)
        if not (_cache_dir() / "spacetrack_satcat.csv").exists():
            assert cat.diameters[0] == pytest.approx(1.5)
        else:
            assert cat.diameters[0] > 0.0

    def test_load_is_cached(self, tmp_path):
        cfg = CatalogConfig(enabled=True, source=_write_tle_file(tmp_path))
        assert load_catalog(cfg) is load_catalog(cfg)


# ---------------------------------------------------------------------------
# discover_streaks round-trip
# ---------------------------------------------------------------------------


class TestDiscoverStreaks:
    def _find_visible_instant(self, cat, sensor):
        """Scan a day for an epoch where ISS is sunlit + above the horizon."""
        base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        for k in range(0, 24 * 60, 5):
            t = base + timedelta(minutes=k)
            ra, dec, *_ = propagate(ISS_L1, ISS_L2, t, SITE)
            streaks = discover_streaks(
                cat, t, SITE, sensor, ra, dec, 0.0, 0.0, albedo=0.2, psf_pad_px=5.0
            )
            if streaks:
                return t, ra, dec, streaks
        return None

    def test_roundtrip_centered(self):
        """Pointing at the satellite → it lands near image center, as a streak."""
        sensor = SensorConfig(height=512, width=512, y_fov=2.0, x_fov=2.0, exposure=2.0)
        cat = _iss_catalog()
        hit = self._find_visible_instant(cat, sensor)
        assert hit is not None, "no sunlit/above-horizon ISS pass found in a day scan"
        _, _, _, streaks = hit

        assert len(streaks) == 1
        s = streaks[0]
        # Pointing AT the object → centered to within sub-degree projection error
        assert abs(s.pixel_row - 256.0) < 30.0
        assert abs(s.pixel_col - 256.0) < 30.0
        # ISS moves fast → a real streak, with a finite magnitude and positive range
        assert math.hypot(s.row_rate, s.col_rate) > 1.0
        assert math.isfinite(s.mv)
        assert s.range_m > 0.0

    def test_roundtrip_centered_multi_object(self):
        """Regression: with N>1 TLEs, the pointed-at object still lands at center.

        The TEME->GCRF rotation must be applied row-wise. A quaternion*array bug
        only manifests for N>1 (N==1 collapses to the scalar path), so this uses a
        multi-object catalog and asserts the centered object is actually centered.
        """
        import numpy as np

        sensor = SensorConfig(height=512, width=512, y_fov=2.0, x_fov=2.0, exposure=2.0)
        tle = satkit.TLE.from_lines([ISS_L1, ISS_L2])
        cat = LoadedCatalog(tles=[tle, tle, tle], diameters=np.full(3, 2.0))  # N=3
        hit = self._find_visible_instant(cat, sensor)
        assert hit is not None
        _, _, _, streaks = hit
        # All three are the same object → all should sit within a few px of center.
        nearest = min(streaks, key=lambda s: (s.pixel_row - 256) ** 2 + (s.pixel_col - 256) ** 2)
        assert abs(nearest.pixel_row - 256.0) < 30.0
        assert abs(nearest.pixel_col - 256.0) < 30.0

    def test_empty_when_pointing_away(self):
        """Pointing at the opposite sky returns nothing (object far outside FOV)."""
        sensor = SensorConfig(height=256, width=256, y_fov=1.0, x_fov=1.0, exposure=2.0)
        cat = _iss_catalog()
        t = datetime(2024, 1, 1, 3, 0, 0, tzinfo=timezone.utc)
        ra, dec, *_ = propagate(ISS_L1, ISS_L2, t, SITE)
        opp_ra = (ra + 180.0) % 360.0
        opp_dec = -dec
        streaks = discover_streaks(
            cat, t, SITE, sensor, opp_ra, opp_dec, 0.0, 0.0, albedo=0.2, psf_pad_px=5.0
        )
        assert streaks == []

    def test_brighter_when_larger(self):
        """A larger assumed diameter yields a brighter (smaller) magnitude."""
        sensor = SensorConfig(height=512, width=512, y_fov=2.0, x_fov=2.0, exposure=2.0)
        hit = self._find_visible_instant(_iss_catalog(2.0), sensor)
        assert hit is not None
        t, ra, dec, small = hit
        big = discover_streaks(
            _iss_catalog(20.0), t, SITE, sensor, ra, dec, 0.0, 0.0,
            albedo=0.2, psf_pad_px=5.0,
        )
        assert big and small
        assert big[0].mv < small[0].mv


# ---------------------------------------------------------------------------
# Mount-rate resolution
# ---------------------------------------------------------------------------


class TestMountRate:
    def _scene(self, tmp_path, **scene_kw):
        from sdasim.scene import Scene

        cfg = SceneConfig(
            sensor=SensorConfig(height=64, width=64, y_fov=1.0, x_fov=1.0),
            stars=StarFieldConfig(mode="bins", ra=180.0, dec=12.0),
            star_motion=StarMotionConfig(temporal_osf=50),
            site=SITE,
            catalog=CatalogConfig(enabled=True, source=_write_tle_file(tmp_path)),
            obs_time="2024-01-01T03:00:00Z",
            seed=42,
            **scene_kw,
        )
        return Scene(cfg)

    def test_sidereal_default(self, tmp_path):
        scene = self._scene(tmp_path)
        assert scene._resolve_mount_rate() == (0.0, 0.0)

    def test_config_mount_rate(self, tmp_path):
        scene = self._scene(tmp_path, mount_ra_rate=0.01, mount_dec_rate=-0.02)
        mr, md = scene._resolve_mount_rate()
        assert mr == pytest.approx(math.radians(0.01))
        assert md == pytest.approx(math.radians(-0.02))

    def test_override_beats_config(self, tmp_path):
        scene = self._scene(tmp_path, mount_ra_rate=0.01)
        mr, md = scene._resolve_mount_rate(mount_ra_rate=0.05)
        assert mr == pytest.approx(math.radians(0.05))


# ---------------------------------------------------------------------------
# Scene integration
# ---------------------------------------------------------------------------


class TestSceneCatalog:
    def test_catalog_render_runs(self, tmp_path):
        """A catalog-only scene renders; mode label is 'catalog'."""
        from sdasim.scene import Scene

        cfg = SceneConfig(
            sensor=SensorConfig(height=128, width=128, y_fov=2.0, x_fov=2.0),
            stars=StarFieldConfig(mode="bins", ra=180.0, dec=12.0),
            star_motion=StarMotionConfig(temporal_osf=50),
            site=SITE,
            catalog=CatalogConfig(enabled=True, source=_write_tle_file(tmp_path)),
            obs_time="2024-01-01T03:00:00Z",
            seed=42,
        )
        scene = Scene(cfg)
        assert scene._catalog is not None
        img, meta = scene.render(0)
        assert img.shape == (128, 128)
        assert meta["frame_mode"] == "catalog"

    def test_catalog_requires_site(self, tmp_path):
        from sdasim.scene import Scene

        cfg = SceneConfig(
            sensor=SensorConfig(height=64, width=64),
            stars=StarFieldConfig(mode="bins"),
            site=None,
            catalog=CatalogConfig(enabled=True, source=_write_tle_file(tmp_path)),
        )
        with pytest.raises(ValueError, match="requires site"):
            Scene(cfg)

    def test_catalog_streak_appears_in_frame(self, tmp_path):
        """At a sunlit pass with the satellite pointed at, a target is rendered."""
        from sdasim.scene import Scene

        sensor = SensorConfig(height=256, width=256, y_fov=3.0, x_fov=3.0, exposure=2.0)
        # find a sunlit, above-horizon epoch + pointing
        base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        cat = _iss_catalog()
        chosen = None
        for k in range(0, 24 * 60, 5):
            t = base + timedelta(minutes=k)
            ra, dec, *_ = propagate(ISS_L1, ISS_L2, t, SITE)
            if discover_streaks(cat, t, SITE, sensor, ra, dec, 0.0, 0.0, 0.2, 6.0):
                chosen = (t, ra, dec)
                break
        assert chosen is not None
        t, ra, dec = chosen

        cfg = SceneConfig(
            sensor=sensor,
            stars=StarFieldConfig(mode="bins", ra=ra, dec=dec),
            star_motion=StarMotionConfig(temporal_osf=50),
            site=SITE,
            catalog=CatalogConfig(enabled=True, source=_write_tle_file(tmp_path)),
            obs_time=t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            seed=42,
        )
        scene = Scene(cfg)
        _, meta = scene.render(0)
        assert meta["num_targets"] >= 1


# ---------------------------------------------------------------------------
# Config nesting
# ---------------------------------------------------------------------------


def test_catalog_config_yaml_nesting():
    from sdasim.config import _dataclass_from_dict

    data = {
        "catalog": {"enabled": True, "source": "spacetrack", "albedo": 0.3},
        "mount_ra_rate": 0.01,
    }
    cfg = _dataclass_from_dict(SceneConfig, data)
    assert isinstance(cfg.catalog, CatalogConfig)
    assert cfg.catalog.enabled is True
    assert cfg.catalog.albedo == 0.3
    assert cfg.mount_ra_rate == 0.01
