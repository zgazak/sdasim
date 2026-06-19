"""SSTRC7 star catalog reader.

Ported from senpai.catalog.sstrc7 — provides query_by_los() for rendering
star fields from the SSTRC7 binary catalog.

The catalog uses zone-indexed binary files:
  - sstrc.acc: index file (60 RA zones x 1800 Dec zones)
  - s0000.cat .. s1799.cat: per-Dec-zone star records (60 bytes each)
"""

from __future__ import annotations

import math
import os
import struct
from functools import lru_cache

import numpy as np
from astropy import wcs

# SSTRC7 catalog location. Override with the SDASIM_SSTRC7_PATH environment
# variable, or by passing `catalog_path` in the scene config.
DEFAULT_SSTRC7_PATH = os.environ.get("SDASIM_SSTRC7_PATH", "sstrc7")

RECORD_LEN = 30  # 30 unsigned shorts = 60 bytes
RECORD_LEN_BYTES = RECORD_LEN * 2


# ---------------------------------------------------------------------------
# Index and zone I/O
# ---------------------------------------------------------------------------


@lru_cache(maxsize=10)
def load_index(filename: str, numRaZones: int = 60, numDecZones: int = 1800):
    """Load SSTRC index file into a 2-D list of {pos, length} dicts."""
    zoneIndex = [[{}] * numRaZones for _ in range(numDecZones)]
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.dtype("<u4"))
    i = 0
    for decIndex in range(numDecZones):
        for raIndex in range(numRaZones):
            zoneIndex[decIndex][raIndex] = {"pos": data[i], "length": data[i + 1]}
            i += 2
    return zoneIndex


def select_zone(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    zoneIndex: list,
    numRaZones: int = 60,
    numDecZones: int = 1800,
):
    """Select zones that intersect with the given RA/Dec bounds (radians)."""
    decZoneLimit = numDecZones - 1
    raZoneLimit = numRaZones - 1
    zoneHeight = math.pi / numDecZones
    zoneWidth = 2.0 * math.pi / numRaZones
    selectZoneList = []

    spd_min = math.pi / 2.0 + dec_min
    spd_max = math.pi / 2.0 + dec_max
    minSPDIndex = max(int(spd_min / zoneHeight), 0)
    maxSPDIndex = min(int(spd_max / zoneHeight), decZoneLimit)

    def append_zones(minRAIndex, maxRAIndex, bound_func):
        for spd in range(minSPDIndex, maxSPDIndex + 1):
            for ra in range(minRAIndex, maxRAIndex + 1):
                selectZone = {
                    "id": int(spd),
                    "pos": zoneIndex[spd][ra]["pos"],
                    "length": zoneIndex[spd][ra]["length"],
                }
                if bound_func == 0:
                    selectZone["bound"] = (
                        "maxRA"
                        if ((ra + 1) * zoneWidth > ra_max)
                        else ("minRA" if (ra * zoneWidth < ra_min) else "inside")
                    )
                elif bound_func == 1:
                    selectZone["bound"] = "minRA" if (ra * zoneWidth < ra_max) else "inside"
                elif bound_func == 2:
                    selectZone["bound"] = "maxRA" if (ra * zoneWidth > ra_min) else "inside"
                if selectZone["length"] > 0:
                    selectZoneList.append(selectZone)

    if ra_min <= ra_max:
        minRAIndex = max(int(ra_min / zoneWidth), 0)
        maxRAIndex = min(int(ra_max / zoneWidth), raZoneLimit)
        append_zones(minRAIndex, maxRAIndex, 0)
    else:
        minRAIndex = max(int(ra_min / zoneWidth) - 1, 0)
        maxRAIndex = raZoneLimit
        append_zones(minRAIndex, maxRAIndex, 1)
        minRAIndex = 0
        maxRAIndex = min(int(ra_max / zoneWidth) + 1, raZoneLimit)
        append_zones(minRAIndex, maxRAIndex, 2)

    return selectZoneList


def load_zone(currentZone, rootPath):
    """Load a zone's star records from disk."""
    filename = os.path.join(rootPath, "s{:04d}.cat".format(currentZone["id"]))
    with open(filename, "rb") as f:
        offset = currentZone["pos"] * RECORD_LEN_BYTES
        f.seek(offset)
        nbytes = currentZone["length"] * RECORD_LEN_BYTES
        buf = f.read(nbytes)
    return buf, offset, offset + nbytes


# ---------------------------------------------------------------------------
# Star record parsing
# ---------------------------------------------------------------------------


def get_star_mv(mv: np.ndarray) -> float:
    """Pick the best available magnitude for open-band (silicon) response.

    Priority: Johnson_V > Johnson_R > Sloan_r > Gaia_G > Sloan_g > Johnson_B.
    """
    if mv[4] < 32:
        return mv[4]  # Johnson_V
    if mv[5] < 32:
        return mv[5]  # Johnson_R
    if mv[8] < 32:
        return mv[8]  # Sloan_r
    if mv[0] < 32:
        return mv[0]  # Gaia_G
    if mv[7] < 32:
        return mv[7]  # Sloan_g
    if mv[3] < 32:
        return mv[3]  # Johnson_B
    return 32.0


def read_star(buffer: bytes, filter_center: float | None = None):
    """Parse a 60-byte SSTRC7 star record.

    Returns dict with 'ra', 'dec' (radians), and 'mv' (magnitude).
    """
    mas2rad = 4.84813681109535993589914102358e-9
    angleScale = mas2rad

    raw = struct.unpack("=iihhh", buffer[0:14])
    raw_mv = struct.unpack("=hhhhhhhhhhhhhhhhhh", buffer[14:50])
    raw_mv_array = np.asarray(raw_mv) * 1.0e-3

    ra_raw, dec_raw = raw[0], raw[1]

    if filter_center is not None:
        centers = np.array(
            [
                600,
                500,
                800,
                440,
                548,
                700,
                900,
                477,
                622,
                762,
                913,
                1235,
                1662,
                2159,
                3400,
                4600,
                12000,
                22000,
            ]
        )
        valid = (raw_mv_array < 32) & (raw_mv_array > -32)
        c = centers[valid]
        m = raw_mv_array[valid]
        mv = float(np.interp(filter_center, c, m)) if len(c) > 0 else 32.0
        if mv < -32 or mv > 32:
            mv = 32.0
    else:
        mv = float(get_star_mv(raw_mv_array))

    return {
        "ra": ra_raw * angleScale,
        "dec": dec_raw * angleScale,
        "mv": mv,
    }


@lru_cache(maxsize=1024)
def _load_stars_for_zone(
    zone_id: int,
    pos: int,
    length: int,
    bound: str,
    rootPath: str,
    filter_center: float | None = None,
):
    """Load and parse stars for a single zone (cached)."""
    buf, start, end = load_zone({"id": zone_id, "pos": pos, "length": length}, rootPath)
    stars = []
    nrec = (end - start) // RECORD_LEN_BYTES
    indices = range(nrec)
    if bound == "minRA":
        indices = reversed(range(nrec))
    for i in indices:
        s = i * RECORD_LEN_BYTES
        star = read_star(buf[s : s + RECORD_LEN_BYTES], filter_center=filter_center)
        stars.append(star)
    return stars


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------


def query_by_min_max(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    rootPath: str = DEFAULT_SSTRC7_PATH,
    clip_min_max: bool = True,
    filter_center: float | None = None,
):
    """Query catalog by RA/Dec bounds (radians)."""
    zoneIndex = load_index(
        os.path.join(rootPath, "sstrc.acc"),
        numRaZones=60,
        numDecZones=1800,
    )
    zones = select_zone(
        ra_min,
        ra_max,
        dec_min,
        dec_max,
        zoneIndex,
        numRaZones=60,
        numDecZones=1800,
    )

    stars = []
    for z in zones:
        ss = _load_stars_for_zone(
            z["id"],
            z["pos"],
            z["length"],
            z["bound"],
            rootPath,
            filter_center=filter_center,
        )
        if clip_min_max:
            if z["bound"] == "minRA":
                for i in range(len(ss)):
                    if ss[i]["ra"] < ra_min:
                        ss = ss[:i]
                        break
            elif z["bound"] == "maxRA":
                for i in range(len(ss)):
                    if ss[i]["ra"] > ra_max:
                        ss = ss[:i]
                        break
        stars += ss

    return stars


def _get_wcs(height, width, y_ifov, x_ifov, ra, dec, rot=0.0):
    """Build an astropy WCS for focal-plane ↔ sky projection."""
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [1, 1]
    w.wcs.cdelt = np.array([x_ifov, y_ifov])
    w.wcs.crval = np.array([ra, dec])
    w.wcs.crota = [rot, rot]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _get_min_max_ra_dec(
    height,
    width,
    y_ifov,
    x_ifov,
    ra,
    dec,
    rot=0.0,
    pad_mult=0.0,
    origin="center",
):
    """Compute RA/Dec bounding box for the focal plane."""
    w = _get_wcs(height, width, y_ifov, x_ifov, ra, dec, rot)

    hp = height * pad_mult
    wp = width * pad_mult
    pixcrd = np.array(
        [
            [-wp, -hp],
            [-wp, height * 0.5],
            [-wp, height + hp],
            [width * 0.5, height + hp],
            [width + wp, height + hp],
            [width + wp, height * 0.5],
            [width + wp, -hp],
            [width * 0.5, -hp],
        ],
        np.float64,
    )

    center = np.array([[width / 2.0, height / 2.0]])

    if origin == "center":
        pixcrd[:, 0] -= width / 2.0
        pixcrd[:, 1] -= height / 2.0
        center[:, 0] -= width / 2.0
        center[:, 1] -= height / 2.0

    world = w.wcs_pix2world(pixcrd, 1)
    cworld = w.wcs_pix2world(center, 1)
    cra, cdec = cworld[0]

    minTheta, minPhi = np.min(world, axis=0)
    maxTheta, maxPhi = np.max(world, axis=0)

    northpole = w.wcs_world2pix([[0, 89.99999]], 1)[0]
    southpole = w.wcs_world2pix([[0, -89.99999]], 1)[0]

    if not np.any(np.isnan(northpole)) and 0 < northpole[0] < width and 0 < northpole[1] < height:
        cmin = [0, minPhi]
        cmax = [360.0, 90.0]
    elif not np.any(np.isnan(southpole)) and 0 < southpole[0] < width and 0 < southpole[1] < height:
        cmin = [0, -90.0]
        cmax = [360.0, maxPhi]
    elif cra > maxTheta or cra < minTheta or (maxTheta - minTheta) > 180:
        cmin = [maxTheta, minPhi]
        cmax = [minTheta, maxPhi]
    else:
        cmin = [minTheta, minPhi]
        cmax = [maxTheta, maxPhi]

    return cmin, cmax, w


def query_by_los(
    height: int,
    width: int,
    y_fov: float,
    x_fov: float,
    ra: float,
    dec: float,
    rot: float = 0.0,
    rootPath: str = DEFAULT_SSTRC7_PATH,
    pad_mult: float = 0.0,
    origin: str = "center",
    filter_ob: bool = True,
    filter_center: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query SSTRC7 catalog and project stars onto the focal plane.

    Args:
        height, width: image dimensions in pixels.
        y_fov, x_fov: field of view in degrees.
        ra, dec: pointing direction in degrees.
        rot: focal-plane rotation in degrees.
        rootPath: path to SSTRC7 catalog directory.
        pad_mult: padding multiplier for star field.
        origin: 'center' or 'corner'.
        filter_ob: if True, remove stars outside padded FOV.
        filter_center: optional wavelength (nm) for magnitude interpolation.

    Returns:
        (rows, cols, magnitudes) — pixel positions and visual magnitudes.
    """
    y_ifov = y_fov / height
    x_ifov = x_fov / width

    cmin, cmax, w = _get_min_max_ra_dec(
        height,
        width,
        y_ifov,
        x_ifov,
        ra,
        dec,
        rot,
        pad_mult,
        origin,
    )

    cmin_rad = np.radians(cmin)
    cmax_rad = np.radians(cmax)
    stars = query_by_min_max(
        cmin_rad[0],
        cmax_rad[0],
        cmin_rad[1],
        cmax_rad[1],
        rootPath,
        filter_center=filter_center,
    )

    rra = np.array([s["ra"] for s in stars])
    ddec = np.array([s["dec"] for s in stars])
    mm = np.array([s["mv"] for s in stars])

    cc, rr = w.wcs_world2pix(np.degrees(rra), np.degrees(ddec), 0)

    if filter_ob:
        hp = height * (1 + pad_mult)
        wp = width * (1 + pad_mult)
        in_bounds = (rr <= hp) & (rr >= -hp) & (cc <= wp) & (cc >= -wp)
        rr = rr[in_bounds]
        cc = cc[in_bounds]
        mm = mm[in_bounds]

    if origin == "center":
        rr += height / 2.0
        cc += width / 2.0

    return rr, cc, mm
