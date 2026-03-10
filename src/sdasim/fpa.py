"""Focal plane array utilities: A/D conversion, magnitude/PE conversions.

Ports satsim's image/fpa.py analog_to_digital, mv_to_pe, pe_to_mv
and image/psf.py eod_to_sigma.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# Maximum pixel values for integer dtypes
MAX_PIXEL_VALUE: dict[str, float] = {
    "int8": 127.0,
    "uint8": 255.0,
    "int16": 32767.0,
    "uint16": 65535.0,
    "int32": 2147483647.0,
    "uint32": 4294967295.0,
}

# Mapping from string dtype names to torch dtypes (for output)
_TORCH_DTYPE: dict[str, torch.dtype] = {
    "uint8": torch.uint8,
    "int16": torch.int16,
    "int32": torch.int32,
    "uint16": torch.int32,  # torch has no uint16; store in int32
    "uint32": torch.int64,
    "float32": torch.float32,
}


def mv_to_pe(zeropoint: float, mv: float | Tensor) -> float | Tensor:
    """Convert visual magnitude to photoelectrons per second.

    Args:
        zeropoint: Sensor zeropoint magnitude.
        mv: Visual magnitude(s).

    Returns:
        Photoelectrons per second.
    """
    if isinstance(mv, Tensor):
        return 10.0 ** ((zeropoint - mv) / 2.5)
    return math.pow(10.0, (zeropoint - mv) / 2.5)


def pe_to_mv(zeropoint: float, pe: float | Tensor) -> float | Tensor:
    """Convert photoelectrons per second to visual magnitude.

    Args:
        zeropoint: Sensor zeropoint magnitude.
        pe: Photoelectrons per second.

    Returns:
        Visual magnitude.
    """
    if isinstance(pe, Tensor):
        return -2.5 * torch.log10(pe) + zeropoint
    return -2.5 * math.log10(pe) + zeropoint


def analog_to_digital(
    fpa: Tensor,
    gain: float,
    fwc: float,
    bias: float,
    dtype: str = "uint16",
) -> Tensor:
    """Convert photoelectron image to digital counts.

    Pipeline: add bias -> clip to full-well -> divide by gain -> floor -> clip to dtype range.

    Args:
        fpa: Image in photoelectrons.
        gain: Electrons per digital count (e-/DN).
        fwc: Full-well capacity in electrons.
        bias: Bias offset in electrons.
        dtype: Output dtype name ('uint8', 'uint16', 'int16', 'uint32', 'int32').

    Returns:
        Digital image (integer-valued tensor).
    """
    # Add bias and clip to full-well capacity
    signal = (fpa + bias).clamp(min=0.0, max=fwc)
    # Convert to digital counts
    dn = torch.floor(signal / gain)
    # Clip to output dtype range
    max_val = MAX_PIXEL_VALUE.get(dtype, 65535.0)
    dn = dn.clamp(min=0.0, max=max_val)
    return dn


def eod_to_sigma(eod: float, osf: float = 1.0) -> float:
    """Convert Energy-on-Detector fraction to Gaussian PSF sigma.

    Uses the inverse error function: sigma = osf / (2 * sqrt(2) * erfinv(sqrt(eod)))

    Args:
        eod: Energy-on-detector fraction (0 < eod < 1).
        osf: Oversampling factor (default 1.0 for native resolution).

    Returns:
        Gaussian sigma in pixels.
    """
    return osf / (2.0 * math.sqrt(2.0) * torch.erfinv(torch.tensor(math.sqrt(eod))).item())
