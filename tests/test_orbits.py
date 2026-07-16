"""Tests for orbital mechanics module."""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from sdasim.config import (
    ObjectConfig,
    SceneConfig,
    SensorConfig,
    SiteConfig,
    StarFieldConfig,
    StarMotionConfig,
)
from sdasim.orbits import (
    ObjectState,
    _fix_tle_checksum,
    angular_to_pixel_rates,
    dither_tle,
    propagate,
)

# ---------------------------------------------------------------------------
# Sample TLE for ISS (NORAD 25544) — hardcoded for offline tests
# ---------------------------------------------------------------------------

ISS_TLE_L1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9002"
ISS_TLE_L2 = "2 25544  51.6400 200.0000 0007417  35.0000 325.0000 15.49000000000000"

SITE = SiteConfig(latitude=40.0, longitude=-105.0, altitude=1600.0)
OBS_TIME = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# angular_to_pixel_rates (pure math, no external deps)
# ---------------------------------------------------------------------------


class TestAngularToPixelRates:
    def test_known_conversion(self):
        """1 rad/s dec rate with 1 deg/512px IFOV → predictable px/s."""
        y_fov = 1.0  # degrees
        x_fov = 1.0
        height = 512
        width = 512

        # 1 rad/s dec rate
        row_rate, col_rate = angular_to_pixel_rates(
            ra_rate=0.0, dec_rate=1.0, dec_deg=0.0,
            y_fov=y_fov, x_fov=x_fov, height=height, width=width,
        )

        ifov_rad = math.radians(y_fov / height)
        expected_row = 1.0 / ifov_rad
        assert abs(row_rate - expected_row) < 1e-6
        assert col_rate == 0.0

    def test_ra_rate_with_cos_dec(self):
        """RA rate is scaled by cos(dec)."""
        row_rate, col_rate = angular_to_pixel_rates(
            ra_rate=1.0, dec_rate=0.0, dec_deg=60.0,
            y_fov=1.0, x_fov=1.0, height=512, width=512,
        )

        ifov_rad = math.radians(1.0 / 512)
        expected_col = 1.0 * math.cos(math.radians(60.0)) / ifov_rad
        assert row_rate == 0.0
        assert abs(col_rate - expected_col) < 1e-6

    def test_zero_rates(self):
        row_rate, col_rate = angular_to_pixel_rates(
            ra_rate=0.0, dec_rate=0.0, dec_deg=0.0,
            y_fov=0.5, x_fov=0.5, height=256, width=256,
        )
        assert row_rate == 0.0
        assert col_rate == 0.0


# ---------------------------------------------------------------------------
# TLE checksum
# ---------------------------------------------------------------------------


class TestTLEChecksum:
    def test_fix_checksum(self):
        line = ISS_TLE_L2
        # Corrupt checksum
        corrupted = line[:68] + "0"
        fixed = _fix_tle_checksum(corrupted)
        # Recompute manually
        s = 0
        for c in fixed[:68]:
            if c.isdigit():
                s += int(c)
            elif c == "-":
                s += 1
        assert int(fixed[68]) == s % 10


# ---------------------------------------------------------------------------
# SGP4 propagation (requires satkit, no network)
# ---------------------------------------------------------------------------


class TestPropagate:
    def test_reasonable_ra_dec(self):
        """Propagation should produce valid RA/Dec values."""
        ra, dec, ra_rate, dec_rate, range_m = propagate(
            ISS_TLE_L1, ISS_TLE_L2, OBS_TIME, SITE,
        )

        assert 0.0 <= ra < 360.0, f"RA out of range: {ra}"
        assert -90.0 <= dec <= 90.0, f"Dec out of range: {dec}"
        assert range_m > 0, "Range should be positive"
        # ISS is in LEO: slant range should be < 10000 km = 1e7 m
        assert range_m < 1e7, f"Range too large for ISS: {range_m}"

    def test_rates_nonzero(self):
        """ISS should have non-negligible angular rates."""
        _, _, ra_rate, dec_rate, _ = propagate(
            ISS_TLE_L1, ISS_TLE_L2, OBS_TIME, SITE,
        )
        # ISS moves fast — at least one rate should be significant
        total_rate = math.sqrt(ra_rate**2 + dec_rate**2)
        assert total_rate > 1e-6, "Angular rates suspiciously small for ISS"


# ---------------------------------------------------------------------------
# Observer velocity in apparent rates (requires satkit, no network)
# ---------------------------------------------------------------------------

# GEO satellite (NORAD 09855), where the observer's velocity is ~11% of the rate.
GEO_TLE_L1 = "1 09855U 77007C   26170.33306599 -.00000012  00000-0  00000-0 0  9996"
GEO_TLE_L2 = "2 09855   3.3870 267.8596 0016186  48.2675 312.2250  1.00336374 25387"
GEO_OBS_TIME = datetime(2026, 6, 21, 5, 56, 53, 826355, tzinfo=timezone.utc)
GEO_SITE = SiteConfig(latitude=20.82028, longitude=-156.27944, altitude=1000.0)


class TestObserverVelocity:
    """propagate()/discover_streaks() must subtract the observer's velocity
    (Earth rotation) so apparent rates are relative to the inertial star field.
    Dropping it overstated GEO rates by ~1.7"/s (~11%)."""

    @staticmethod
    def _finite_diff_rates_rad_s(l1, l2, obs, site, dt=1.0):
        """Independent truth: finite-difference topocentric RA/Dec using the real
        co-rotating observer geometry (so observer motion is included)."""
        from datetime import timedelta

        import numpy as np
        import satkit

        tle = satkit.TLE.from_lines([l1, l2])

        def radec(when):
            tm = satkit.time(
                when.year, when.month, when.day, when.hour, when.minute,
                when.second + when.microsecond / 1e6,
            )
            rot = np.asarray(satkit.frametransform.qteme2gcrf(tm).as_rotation_matrix(), float)
            p = np.asarray(satkit.sgp4(tle, tm)[0], float).reshape(3) @ rot.T
            ri = np.asarray(satkit.frametransform.qitrf2gcrf(tm).as_rotation_matrix(), float)
            o = np.asarray(
                satkit.itrfcoord(
                    latitude_deg=site.latitude,
                    longitude_deg=site.longitude,
                    altitude=site.altitude,
                ).vector, float,
            ).reshape(3) @ ri.T
            x, y, z = p - o
            return (
                math.degrees(math.atan2(y, x)) % 360.0,
                math.degrees(math.atan2(z, math.hypot(x, y))),
            )

        def dwrap(a, b):
            return (b - a + 180.0) % 360.0 - 180.0

        ra0, dec0 = radec(obs)
        ra1, dec1 = radec(obs + timedelta(seconds=dt))
        return math.radians(dwrap(ra0, ra1) / dt), math.radians(dwrap(dec0, dec1) / dt)

    def test_geo_rate_matches_finite_diff(self):
        pytest.importorskip("satkit")
        _, _, ra_rate, dec_rate, range_m = propagate(
            GEO_TLE_L1, GEO_TLE_L2, GEO_OBS_TIME, GEO_SITE,
        )
        # Sanity: this really is a GEO-range object (~37,000 km, in meters).
        assert 3.5e7 < range_m < 4.2e7

        ra_truth, dec_truth = self._finite_diff_rates_rad_s(
            GEO_TLE_L1, GEO_TLE_L2, GEO_OBS_TIME, GEO_SITE,
        )
        # arcsec/s agreement well under 0.1" (pre-fix the RA rate was ~1.7" high).
        d_ra = abs(math.degrees(ra_rate - ra_truth) * 3600.0)
        d_dec = abs(math.degrees(dec_rate - dec_truth) * 3600.0)
        assert d_ra < 0.1, f"RA rate off by {d_ra:.3f}\"/s"
        assert d_dec < 0.1, f"Dec rate off by {d_dec:.3f}\"/s"


# ---------------------------------------------------------------------------
# Dithering (requires satkit, no network)
# ---------------------------------------------------------------------------


class TestDither:
    def test_dither_changes_position(self):
        """Dithered TLE should produce a different position."""
        ra_orig, dec_orig, _, _, _ = propagate(
            ISS_TLE_L1, ISS_TLE_L2, OBS_TIME, SITE,
        )

        _, l2_dithered = dither_tle(
            ISS_TLE_L1, ISS_TLE_L2, 100.0, SITE, OBS_TIME,
        )

        ra_dith, dec_dith, _, _, _ = propagate(
            ISS_TLE_L1, l2_dithered, OBS_TIME, SITE,
        )

        # Position should be different
        offset = math.sqrt((ra_dith - ra_orig)**2 + (dec_dith - dec_orig)**2)
        assert offset > 1e-6, "Dithered position should differ from original"

    def test_zero_dither_unchanged(self):
        """Zero dither should return original TLE."""
        l1, l2 = dither_tle(ISS_TLE_L1, ISS_TLE_L2, 0.0, SITE, OBS_TIME)
        assert l1 == ISS_TLE_L1
        assert l2 == ISS_TLE_L2


# ---------------------------------------------------------------------------
# Tracking mode integration tests (mock TLE fetch, no network)
# ---------------------------------------------------------------------------


def _mock_object_states():
    """Create mock ObjectStates for tracking mode tests."""
    primary = ObjectState(
        norad_id="25544", ra=180.0, dec=12.0,
        ra_rate=0.001, dec_rate=0.002,
        row_rate=5.0, col_rate=-3.0,
        mv=2.0, pixel_row=256.0, pixel_col=256.0,
    )
    secondary = ObjectState(
        norad_id="25994", ra=180.1, dec=12.05,
        ra_rate=0.0005, dec_rate=0.001,
        row_rate=2.0, col_rate=-1.0,
        mv=14.0, pixel_row=280.0, pixel_col=230.0,
    )
    return [primary, secondary]


class TestTrackingModes:
    """Test that Scene._init_from_objects sets star motion and target velocities correctly."""

    def _make_config(self, tracking="rate_track"):
        return SceneConfig(
            sensor=SensorConfig(height=512, width=512, y_fov=0.5, x_fov=0.5),
            stars=StarFieldConfig(mode="bins"),
            star_motion=StarMotionConfig(temporal_osf=100),
            site=SiteConfig(latitude=40.0, longitude=-105.0, altitude=1600.0),
            objects=[
                ObjectConfig(norad_id="25544", mv=2.0),
                ObjectConfig(norad_id="25994", mv=14.0),
            ],
            obs_time="2024-01-01T12:00:00Z",
            tracking=tracking,
            seed=42,
        )

    @patch("sdasim.orbits.compute_object_states")
    def test_rate_track_star_motion(self, mock_compute):
        """In rate_track, star motion = -primary pixel rates."""
        states = _mock_object_states()
        mock_compute.return_value = states

        config = self._make_config("rate_track")
        from sdasim.scene import Scene
        Scene(config)

        assert config.star_motion.translation == [-5.0, 3.0]

    @patch("sdasim.orbits.compute_object_states")
    def test_rate_track_primary_stationary(self, mock_compute):
        """In rate_track, primary target velocity = [0, 0]."""
        states = _mock_object_states()
        mock_compute.return_value = states

        config = self._make_config("rate_track")
        from sdasim.scene import Scene
        Scene(config)

        assert config.targets[0].velocity == [0.0, 0.0]

    @patch("sdasim.orbits.compute_object_states")
    def test_rate_track_secondary_relative(self, mock_compute):
        """In rate_track, secondary velocity = secondary - primary rates."""
        states = _mock_object_states()
        mock_compute.return_value = states

        config = self._make_config("rate_track")
        from sdasim.scene import Scene
        Scene(config)

        expected = [2.0 - 5.0, -1.0 - (-3.0)]  # [-3.0, 2.0]
        assert config.targets[1].velocity == expected

    @patch("sdasim.orbits.compute_object_states")
    def test_sidereal_stars_stationary(self, mock_compute):
        """In sidereal, star motion = [0, 0] and osf = 1."""
        states = _mock_object_states()
        mock_compute.return_value = states

        config = self._make_config("sidereal")
        from sdasim.scene import Scene
        Scene(config)

        assert config.star_motion.translation == [0.0, 0.0]
        assert config.star_motion.temporal_osf == 1

    @patch("sdasim.orbits.compute_object_states")
    def test_sidereal_full_rates(self, mock_compute):
        """In sidereal, targets have full pixel rates."""
        states = _mock_object_states()
        mock_compute.return_value = states

        config = self._make_config("sidereal")
        from sdasim.scene import Scene
        Scene(config)

        assert config.targets[0].velocity == [5.0, -3.0]
        assert config.targets[1].velocity == [2.0, -1.0]

    @patch("sdasim.orbits.compute_object_states")
    def test_star_pointing_from_primary(self, mock_compute):
        """Star field pointing derives from the primary target.

        In rate_track mode the pointing is back-shifted by half an exposure so
        that streak centroids (sampled at mid-exposure) align with the satellite
        at image center, so the expected RA/Dec is the shifted value.
        """
        states = _mock_object_states()
        mock_compute.return_value = states

        config = self._make_config("rate_track")
        from sdasim.scene import Scene
        Scene(config)

        half_exp = config.sensor.exposure / 2.0
        expected_ra = 180.0 - math.degrees(0.001) * half_exp
        expected_dec = 12.0 - math.degrees(0.002) * half_exp
        assert config.stars.ra == pytest.approx(expected_ra)
        assert config.stars.dec == pytest.approx(expected_dec)


class TestRawRaDec:
    """Test raw RA/Dec objects (no TLE fetch)."""

    def test_raw_object_pixel_rates(self):
        """Raw RA/Dec object should produce correct pixel rates."""
        from sdasim.orbits import compute_object_states

        obj = ObjectConfig(
            ra=180.0, dec=12.0,
            ra_rate=0.01, dec_rate=0.02,  # deg/s
            mv=10.0,
        )
        sensor = SensorConfig(height=512, width=512, y_fov=0.5, x_fov=0.5)
        obs_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        states = compute_object_states([obj], None, sensor, obs_time)
        assert len(states) == 1
        st = states[0]

        assert st.ra == 180.0
        assert st.dec == 12.0
        # Rates stored internally as rad/s
        assert abs(st.ra_rate - math.radians(0.01)) < 1e-12
        assert abs(st.dec_rate - math.radians(0.02)) < 1e-12
        # Pixel rates should be nonzero
        assert st.row_rate != 0.0
        assert st.col_rate != 0.0

    def test_raw_object_primary_at_center(self):
        """Primary raw object should be placed at image center."""
        from sdasim.orbits import compute_object_states

        obj = ObjectConfig(ra=90.0, dec=45.0, ra_rate=0.0, dec_rate=0.0, mv=8.0)
        sensor = SensorConfig(height=256, width=256)
        obs_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

        states = compute_object_states([obj], None, sensor, obs_time)
        assert states[0].pixel_row == 128.0
        assert states[0].pixel_col == 128.0

    def test_raw_zero_rates(self):
        """Object with zero rates should have zero pixel rates."""
        from sdasim.orbits import compute_object_states

        obj = ObjectConfig(ra=0.0, dec=0.0, ra_rate=0.0, dec_rate=0.0, mv=12.0)
        sensor = SensorConfig(height=512, width=512)
        obs_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

        states = compute_object_states([obj], None, sensor, obs_time)
        assert states[0].row_rate == 0.0
        assert states[0].col_rate == 0.0

    def test_raw_scene_rate_track(self):
        """Raw RA/Dec objects should work end-to-end with Scene in rate_track."""
        config = SceneConfig(
            sensor=SensorConfig(height=64, width=64, y_fov=0.5, x_fov=0.5),
            stars=StarFieldConfig(mode="bins"),
            star_motion=StarMotionConfig(temporal_osf=50),
            objects=[
                ObjectConfig(ra=180.0, dec=12.0, ra_rate=0.01, dec_rate=0.02, mv=10.0),
                ObjectConfig(ra=180.05, dec=12.01, ra_rate=0.005, dec_rate=0.01, mv=14.0),
            ],
            tracking="rate_track",
            obs_time="2024-01-01T12:00:00Z",
            seed=42,
        )
        from sdasim.scene import Scene
        scene = Scene(config)

        # Primary should be stationary in rate_track
        assert config.targets[0].velocity == [0.0, 0.0]
        # Star motion should be opposite of primary
        assert config.star_motion.translation[0] != 0.0

        img, meta = scene.render(0)
        assert img.shape == (64, 64)

    def test_raw_scene_sidereal(self):
        """Raw RA/Dec objects should work end-to-end with Scene in sidereal."""
        config = SceneConfig(
            sensor=SensorConfig(height=64, width=64, y_fov=0.5, x_fov=0.5),
            stars=StarFieldConfig(mode="bins"),
            star_motion=StarMotionConfig(temporal_osf=50),
            objects=[
                ObjectConfig(ra=180.0, dec=12.0, ra_rate=0.01, dec_rate=0.02, mv=10.0),
            ],
            tracking="sidereal",
            obs_time="2024-01-01T12:00:00Z",
            seed=42,
        )
        from sdasim.scene import Scene
        scene = Scene(config)

        assert config.star_motion.translation == [0.0, 0.0]
        # Target should have nonzero velocity in sidereal
        assert config.targets[0].velocity[0] != 0.0

        img, meta = scene.render(0)
        assert img.shape == (64, 64)

    def test_mixed_raw_and_norad_not_fetched(self):
        """Raw objects should not trigger TLE fetch."""
        from sdasim.orbits import _is_raw_radec

        raw_obj = ObjectConfig(ra=180.0, dec=12.0, ra_rate=0.0, dec_rate=0.0, mv=10.0)
        assert _is_raw_radec(raw_obj) is True

        norad_obj = ObjectConfig(norad_id="25544", mv=2.0)
        assert _is_raw_radec(norad_obj) is False


class TestBackwardCompat:
    """Legacy config (with targets, no objects) still works."""

    def test_legacy_config_no_objects(self):
        """Scene with targets and no objects should work as before."""
        config = SceneConfig(
            sensor=SensorConfig(height=64, width=64),
            stars=StarFieldConfig(mode="bins"),
            star_motion=StarMotionConfig(translation=[3.0, -5.0], temporal_osf=100),
            targets=[],
            seed=42,
        )
        from sdasim.scene import Scene
        scene = Scene(config)

        assert scene._object_states is None
        img, meta = scene.render(0)
        assert img.shape == (64, 64)


# ---------------------------------------------------------------------------
# Network-dependent tests (TLE fetch)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("SDASIM_RUN_NETWORK_TESTS"),
    reason="live TLE fetch; set SDASIM_RUN_NETWORK_TESTS=1 to run",
)
class TestTLEFetch:
    def test_fetch_iss(self):
        from sdasim.orbits import fetch_tles_sync

        tles = fetch_tles_sync(["25544"])
        assert "25544" in tles
        l1, l2 = tles["25544"]
        assert l1.startswith("1 ")
        assert l2.startswith("2 ")
