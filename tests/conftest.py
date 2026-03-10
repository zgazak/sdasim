"""Shared test fixtures for sdasim."""

from __future__ import annotations

import pytest
import torch

from sdasim.device import set_device


@pytest.fixture(autouse=True)
def force_cpu():
    """Force CPU for deterministic tests."""
    set_device("cpu")


@pytest.fixture
def rng():
    """Seeded torch generator for reproducible tests."""
    return torch.Generator().manual_seed(42)
