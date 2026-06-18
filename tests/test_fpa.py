"""Tests for focal plane array utilities."""

from __future__ import annotations

import torch

from sdasim.fpa import analog_to_digital, eod_to_sigma, mv_to_pe, pe_to_mv


class TestMagnitudeConversion:
    """Tests for magnitude <-> photoelectron conversions."""

    def test_mv_to_pe_roundtrip(self):
        """mv -> pe -> mv should be identity."""
        zp = 23.5
        mv = 12.0
        pe = mv_to_pe(zp, mv)
        mv_back = pe_to_mv(zp, pe)
        assert abs(mv_back - mv) < 1e-10

    def test_brighter_star_more_pe(self):
        """Brighter (lower mv) stars should produce more PE."""
        zp = 23.5
        pe_bright = mv_to_pe(zp, 10.0)
        pe_dim = mv_to_pe(zp, 15.0)
        assert pe_bright > pe_dim

    def test_five_mag_factor(self):
        """5 magnitudes should correspond to 100x in flux."""
        zp = 23.5
        pe1 = mv_to_pe(zp, 10.0)
        pe2 = mv_to_pe(zp, 15.0)
        assert abs(pe1 / pe2 - 100.0) < 0.01

    def test_tensor_input(self):
        """Should work with tensor inputs."""
        zp = 23.5
        mv = torch.tensor([10.0, 12.0, 15.0])
        pe = mv_to_pe(zp, mv)
        assert pe.shape == (3,)
        # Verify ordering
        assert pe[0] > pe[1] > pe[2]

    def test_zeropoint_star(self):
        """A star at zeropoint magnitude should produce 1 PE/sec."""
        zp = 20.0
        pe = mv_to_pe(zp, 20.0)
        assert abs(pe - 1.0) < 1e-10


class TestAnalogToDigital:
    """Tests for A/D conversion."""

    def test_basic_conversion(self):
        """Basic conversion with known values."""
        fpa = torch.tensor([[100.0, 500.0], [1000.0, 50000.0]])
        result = analog_to_digital(fpa, gain=8.0, fwc=100000.0, bias=500.0, dtype="uint16")
        # (100 + 500) / 8 = 75
        assert result[0, 0].item() == 75.0
        # (500 + 500) / 8 = 125
        assert result[0, 1].item() == 125.0

    def test_full_well_clipping(self):
        """Signal above FWC should be clipped."""
        fpa = torch.tensor([[200000.0]])
        result = analog_to_digital(fpa, gain=1.0, fwc=100000.0, bias=0.0, dtype="uint16")
        assert result[0, 0].item() == 65535.0  # uint16 max

    def test_dtype_range_clipping(self):
        """Output should be clipped to dtype range."""
        fpa = torch.tensor([[60000.0]])
        result = analog_to_digital(fpa, gain=1.0, fwc=100000.0, bias=0.0, dtype="uint8")
        assert result[0, 0].item() == 255.0

    def test_negative_clamped_to_zero(self):
        """Negative signal after bias should clamp to zero."""
        fpa = torch.tensor([[-1000.0]])
        result = analog_to_digital(fpa, gain=8.0, fwc=100000.0, bias=0.0, dtype="uint16")
        assert result[0, 0].item() == 0.0

    def test_floor_division(self):
        """A/D should use floor (not round)."""
        fpa = torch.tensor([[7.9]])
        result = analog_to_digital(fpa, gain=1.0, fwc=100000.0, bias=0.0, dtype="uint16")
        assert result[0, 0].item() == 7.0


class TestEodToSigma:
    """Tests for EOD to sigma conversion."""

    def test_known_value(self):
        """Verify against known EOD->sigma mapping."""
        # EOD=0.68 corresponds to approximately sigma=1.0 for osf=1
        # (since a 2D Gaussian has ~68% within 1 sigma)
        # Actually the formula is sigma = 1/(2*sqrt(2)*erfinv(sqrt(eod)))
        # For eod=0.5: erfinv(sqrt(0.5)) = erfinv(0.7071) ≈ 0.7549
        # sigma = 1/(2*1.4142*0.7549) = 1/2.1349 ≈ 0.4684
        sigma = eod_to_sigma(0.5, osf=1.0)
        assert 0.4 < sigma < 0.6

    def test_higher_eod_smaller_sigma(self):
        """Higher EOD means more concentrated PSF -> smaller sigma."""
        s1 = eod_to_sigma(0.3)
        s2 = eod_to_sigma(0.7)
        assert s2 < s1

    def test_osf_scaling(self):
        """Sigma should scale linearly with OSF."""
        s1 = eod_to_sigma(0.5, osf=1.0)
        s2 = eod_to_sigma(0.5, osf=3.0)
        assert abs(s2 / s1 - 3.0) < 0.01
