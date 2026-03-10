"""GPU-first device management."""

from __future__ import annotations

import torch

_device: torch.device | None = None


def get_device() -> torch.device:
    """Return the global device, defaulting to CUDA if available."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def set_device(device: str | torch.device) -> None:
    """Set the global device."""
    global _device
    _device = torch.device(device) if isinstance(device, str) else device


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve a device argument: None -> global default, 'auto' -> auto-detect."""
    if device is None:
        return get_device()
    if isinstance(device, str):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    return device


def ensure_tensor(
    x,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert input to a tensor on the specified device."""
    dev = resolve_device(device)
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=dev)
    return torch.as_tensor(x, dtype=dtype, device=dev)
