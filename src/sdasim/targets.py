"""Target trajectory computation.

Computes target positions and intensities for each frame.
"""

from __future__ import annotations

import torch
from torch import Tensor

from sdasim.config import SensorConfig, TargetConfig
from sdasim.device import resolve_device
from sdasim.fpa import mv_to_pe


def compute_target_positions(
    targets: list[TargetConfig],
    sensor: SensorConfig,
    frame_idx: int = 0,
    device: str | torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute target positions, intensities, and velocities for a given frame.

    For 'line' mode targets, position at frame start is:
        pos = origin_pixels + velocity * (frame_idx * (exposure + gap))

    Args:
        targets: List of target configurations.
        sensor: Sensor configuration.
        frame_idx: Frame index.
        device: Target device.

    Returns:
        (positions, intensities, velocities): (T, 2) pixel positions, (T,) PE, (T, 2) px/sec.
    """
    dev = resolve_device(device)

    if not targets:
        return (
            torch.zeros(0, 2, dtype=torch.float32, device=dev),
            torch.zeros(0, dtype=torch.float32, device=dev),
            torch.zeros(0, 2, dtype=torch.float32, device=dev),
        )

    positions = []
    intensities = []
    velocities = []

    frame_time = frame_idx * (sensor.exposure + sensor.gap)

    for tgt in targets:
        # Origin in fractional coordinates -> pixel coordinates
        row_origin = tgt.origin[0] * sensor.height
        col_origin = tgt.origin[1] * sensor.width

        if tgt.mode == "line":
            # Position at frame start
            row = row_origin + tgt.velocity[0] * frame_time
            col = col_origin + tgt.velocity[1] * frame_time
            velocities.append(tgt.velocity)
        else:
            row = row_origin
            col = col_origin
            velocities.append([0.0, 0.0])

        pe = mv_to_pe(sensor.zeropoint, tgt.mv) * sensor.exposure
        positions.append([row, col])
        intensities.append(pe)

    pos_tensor = torch.tensor(positions, dtype=torch.float32, device=dev)
    int_tensor = torch.tensor(intensities, dtype=torch.float32, device=dev)
    vel_tensor = torch.tensor(velocities, dtype=torch.float32, device=dev)

    return pos_tensor, int_tensor, vel_tensor
