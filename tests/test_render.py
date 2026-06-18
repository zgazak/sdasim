"""Tests for the render pipeline."""

from __future__ import annotations

import torch

from sdasim.render import expand_motion, render_frame


class TestExpandMotion:
    """Tests for motion blur expansion."""

    def test_no_motion(self):
        """No motion should return original positions."""
        pos = torch.tensor([[16.0, 16.0]])
        ints = torch.tensor([1000.0])
        exp_pos, exp_int = expand_motion(
            pos,
            ints,
            velocity=None,
            rotation=0.0,
            t_start=0.0,
            t_end=1.0,
            t_osf=1,
            center=(16.0, 16.0),
        )
        assert exp_pos.shape == (1, 2)
        assert exp_int.shape == (1,)

    def test_translation_expansion(self):
        """Translation should expand into sub-sources."""
        pos = torch.tensor([[16.0, 16.0]])
        ints = torch.tensor([1000.0])
        exp_pos, exp_int = expand_motion(
            pos,
            ints,
            velocity=[10.0, 0.0],
            rotation=0.0,
            t_start=0.0,
            t_end=1.0,
            t_osf=10,
            center=(16.0, 16.0),
        )
        assert exp_pos.shape == (10, 2)
        assert exp_int.shape == (10,)
        # Each sub-source gets 1/10 of intensity
        assert abs(exp_int.sum().item() - 1000.0) < 0.1
        # First sub-source should be near origin, last near origin + velocity
        assert exp_pos[0, 0].item() < exp_pos[-1, 0].item()

    def test_intensity_conservation(self):
        """Total intensity should be conserved."""
        pos = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
        ints = torch.tensor([500.0, 300.0])
        exp_pos, exp_int = expand_motion(
            pos,
            ints,
            velocity=[5.0, 3.0],
            rotation=0.0,
            t_start=0.0,
            t_end=2.0,
            t_osf=50,
            center=(16.0, 16.0),
        )
        assert abs(exp_int.sum().item() - 800.0) < 0.1

    def test_rotation_expansion(self):
        """Rotation should create arc pattern."""
        pos = torch.tensor([[0.0, 16.0]])  # Off-center
        ints = torch.tensor([1000.0])
        exp_pos, exp_int = expand_motion(
            pos,
            ints,
            velocity=None,
            rotation=0.1,
            t_start=0.0,
            t_end=1.0,
            t_osf=20,
            center=(16.0, 16.0),
        )
        assert exp_pos.shape == (20, 2)
        # Column should change due to rotation
        assert not torch.allclose(exp_pos[:, 1], exp_pos[0:1, 1].expand(20))

    def test_empty_sources(self):
        """Empty input should return empty output."""
        pos = torch.zeros(0, 2)
        ints = torch.zeros(0)
        exp_pos, exp_int = expand_motion(
            pos,
            ints,
            velocity=[1.0, 1.0],
            rotation=0.0,
            t_start=0.0,
            t_end=1.0,
            t_osf=10,
            center=(16.0, 16.0),
        )
        assert exp_pos.shape[0] == 0


class TestRenderFrame:
    """Tests for full render pipeline."""

    def _make_inputs(self, n_stars=5, n_targets=1, h=32, w=32):
        """Create basic inputs for render_frame."""
        torch.manual_seed(42)
        star_pos = torch.rand(n_stars, 2) * torch.tensor([h, w], dtype=torch.float32)
        star_int = torch.rand(n_stars) * 1000 + 100
        tgt_pos = torch.tensor([[h / 2.0, w / 2.0]]) if n_targets else torch.zeros(0, 2)
        tgt_int = torch.tensor([5000.0]) if n_targets else torch.zeros(0)
        return star_pos, star_int, tgt_pos, tgt_int

    def test_basic_render(self):
        """Basic render should produce valid output."""
        star_pos, star_int, tgt_pos, tgt_int = self._make_inputs()
        digital, star_sig, tgt_sig = render_frame(
            32,
            32,
            star_pos,
            star_int,
            tgt_pos,
            tgt_int,
            psf_sigma=1.5,
            background_pe=100.0,
            dark_current_pe=20.0,
            bias_pe=50.0,
            read_noise=10.0,
            electronic_noise=5.0,
            gain=8.0,
            fwc=100000.0,
            a2d_bias=500.0,
            a2d_dtype="uint16",
        )
        assert digital.shape == (32, 32)
        assert star_sig.shape == (32, 32)
        assert tgt_sig.shape == (32, 32)
        assert (digital >= 0).all()

    def test_no_noise_deterministic(self):
        """Without noise, same inputs should produce same output."""
        star_pos, star_int, tgt_pos, tgt_int = self._make_inputs()
        kwargs = dict(
            height=32,
            width=32,
            star_positions=star_pos,
            star_intensities=star_int,
            target_positions=tgt_pos,
            target_intensities=tgt_int,
            psf_sigma=1.5,
            background_pe=100.0,
            dark_current_pe=20.0,
            bias_pe=50.0,
            read_noise=10.0,
            electronic_noise=5.0,
            gain=8.0,
            fwc=100000.0,
            a2d_bias=500.0,
            a2d_dtype="uint16",
            enable_shot_noise=False,
            enable_read_noise=False,
        )
        d1, _, _ = render_frame(**kwargs)
        d2, _, _ = render_frame(**kwargs)
        assert torch.equal(d1, d2)

    def test_no_targets(self):
        """Render without targets should still work."""
        star_pos, star_int, _, _ = self._make_inputs(n_targets=0)
        digital, _, tgt_sig = render_frame(
            32,
            32,
            star_pos,
            star_int,
            torch.zeros(0, 2),
            torch.zeros(0),
            psf_sigma=1.5,
            background_pe=100.0,
            dark_current_pe=20.0,
            bias_pe=50.0,
            read_noise=10.0,
            electronic_noise=5.0,
            gain=8.0,
            fwc=100000.0,
            a2d_bias=500.0,
            a2d_dtype="uint16",
        )
        assert tgt_sig.sum().item() == 0.0

    def test_star_signal_separation(self):
        """Star and target signals should be separate."""
        star_pos, star_int, tgt_pos, tgt_int = self._make_inputs()
        _, star_sig, tgt_sig = render_frame(
            32,
            32,
            star_pos,
            star_int,
            tgt_pos,
            tgt_int,
            psf_sigma=1.5,
            background_pe=0.0,
            dark_current_pe=0.0,
            bias_pe=0.0,
            read_noise=0.0,
            electronic_noise=0.0,
            gain=1.0,
            fwc=1e9,
            a2d_bias=0.0,
            a2d_dtype="uint16",
            enable_shot_noise=False,
            enable_read_noise=False,
        )
        # Star signal should have non-zero pixels where stars are
        assert star_sig.sum().item() > 0
        # Target signal should have non-zero pixels where target is
        assert tgt_sig.sum().item() > 0
