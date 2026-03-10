"""Optional I/O writers for FITS, annotations, and sequences.

These are convenience utilities for batch workflow, never in the hot path.
Requires astropy for FITS writing (lazy import).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor


def write_fits(image: Tensor, path: str | Path, header: dict | None = None) -> None:
    """Write a 2D image tensor to a FITS file.

    Args:
        image: (H, W) image tensor.
        path: Output file path.
        header: Optional FITS header keywords.
    """
    try:
        from astropy.io import fits
    except ImportError:
        raise ImportError("FITS writing requires astropy. Install with: pip install astropy")

    import numpy as np

    data = image.detach().cpu().numpy().astype(np.float32)
    hdu = fits.PrimaryHDU(data)
    if header:
        for key, val in header.items():
            hdu.header[key] = val
    hdu.writeto(str(path), overwrite=True)


def write_annotations(metadata: dict | list[dict], path: str | Path) -> None:
    """Write annotation metadata to a JSON file.

    Args:
        metadata: Single frame metadata dict or list of dicts.
        path: Output file path.
    """
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def write_sequence(
    images: Tensor,
    metadata: list[dict],
    output_dir: str | Path,
    prefix: str = "frame",
    fmt: str = "fits",
) -> list[Path]:
    """Write a sequence of rendered frames and annotations.

    Args:
        images: (N, H, W) tensor of rendered frames.
        metadata: List of metadata dicts, one per frame.
        output_dir: Output directory.
        prefix: Filename prefix.
        fmt: Output format ('fits' or 'npy').

    Returns:
        List of written file paths.
    """
    import numpy as np

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(images.shape[0]):
        frame = images[i]
        if fmt == "fits":
            fpath = out / f"{prefix}_{i:04d}.fits"
            write_fits(frame, fpath, header={"FRAMEIDX": i})
        elif fmt == "npy":
            fpath = out / f"{prefix}_{i:04d}.npy"
            np.save(str(fpath), frame.detach().cpu().numpy())
        else:
            raise ValueError(f"Unknown format: {fmt}")
        paths.append(fpath)

    # Write annotations
    ann_path = out / f"{prefix}_annotations.json"
    write_annotations(metadata, ann_path)
    paths.append(ann_path)

    return paths
