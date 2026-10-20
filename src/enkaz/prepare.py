"""Preparation: pair each BRIGHT tile with an aligned post-event optical image, list its
buildings and cut building-centred pre/post crops.

Alignment. BRIGHT's pre image and the Maxar post scene are orthorectified separately, and
post-event scenes were taken 30–45° off-nadir, so they can be offset by several metres. We
resample the post scene onto the BRIGHT tile grid, estimate a residual translation with phase
correlation on gradient magnitude (robust to sensor and season differences), and resample
again with the corrected grid. Building lean on tall buildings is *not* corrected; it is a
documented limitation.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

CROP = 128  # pixels; at 0.35 m this is ~45 m of context around the building
MIN_AREA = 20  # px; smaller components are rasterisation noise


# --------------------------------------------------------------------------- alignment


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    gx = ndimage.sobel(g, axis=1)
    gy = ndimage.sobel(g, axis=0)
    return np.hypot(gx, gy)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def phase_correlation(
    ref: np.ndarray, mov: np.ndarray, max_shift: int = 64, valid: np.ndarray | None = None
) -> tuple[int, int, float]:
    """Integer shift (dy, dx) such that ``mov[y + dy, x + dx] ≈ ref[y, x]``.

    Returns the shift and a peak-to-mean ratio (higher = more trustworthy). Pixels outside
    ``valid`` are set to the image mean before correlating.
    """
    a = ref.astype(np.float64)
    b = mov.astype(np.float64)
    if valid is not None:
        a = np.where(valid, a, a[valid].mean() if valid.any() else 0)
        b = np.where(valid, b, b[valid].mean() if valid.any() else 0)
    a -= a.mean()
    b -= b.mean()
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.fft2(a * win)
    fb = np.fft.fft2(b * win)
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-12
    corr = np.real(np.fft.ifft2(r))
    corr = np.fft.fftshift(corr)
    cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
    sub = corr[cy - max_shift : cy + max_shift + 1, cx - max_shift : cx + max_shift + 1]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    peak = float(sub[iy, ix])
    ratio = peak / (float(np.abs(corr).mean()) + 1e-12)
    return int(iy - max_shift), int(ix - max_shift), ratio


def shifted_transform(transform, dy: int, dx: int):
    """Grid whose pixel (y, x) samples the old grid's pixel (y + dy, x + dx)."""
    from affine import Affine

    return transform * Affine.translation(dx, dy)


# --------------------------------------------------------------------------- buildings


@dataclass
class Building:
    tile: str
    idx: int
    label: int  # 1 intact, 2 damaged, 3 destroyed
    cy: float
    cx: float
    y0: int
    x0: int
    y1: int
    x1: int
    area: int
    source: str  # "instance" (BRIGHT polygon) or "component" (connected component of mask)
    polygon: list | None = None

    def row(self) -> dict:
        d = self.__dict__.copy()
        d.pop("polygon")
        d["id"] = f"{self.tile}_{self.idx:04d}"
        return d


def load_instances(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def polygon_mask(polygon: Sequence[float], shape: tuple[int, int]) -> np.ndarray:
    img = Image.new("L", (shape[1], shape[0]), 0)
    pts = list(zip(polygon[0::2], polygon[1::2]))
    if len(pts) >= 3:
        ImageDraw.Draw(img).polygon(pts, fill=1)
    return np.asarray(img, dtype=bool)


def buildings_from_instances(tile: str, inst: dict, shape: tuple[int, int]) -> list[Building]:
    out = []
    for i, a in enumerate(inst.get("annotations", [])):
        if not a.get("segmentation") or a.get("category_id") not in (1, 2, 3):
            continue
        poly = max(a["segmentation"], key=len)  # outer ring
        xs, ys = np.asarray(poly[0::2]), np.asarray(poly[1::2])
        x0, y0, x1, y1 = (
            int(np.floor(xs.min())),
            int(np.floor(ys.min())),
            int(np.ceil(xs.max())),
            int(np.ceil(ys.max())),
        )
        area = int(a.get("area") or 0)
        if area < MIN_AREA:
            continue
        out.append(
            Building(
                tile,
                i,
                int(a["category_id"]),
                float(ys.mean()),
                float(xs.mean()),
                max(y0, 0),
                max(x0, 0),
                min(y1, shape[0]),
                min(x1, shape[1]),
                area,
                "instance",
                poly,
            )
        )
    return out


def buildings_from_mask(tile: str, mask: np.ndarray) -> list[Building]:
    """Fallback when no instance polygons exist: 4-connected components of each class."""
    out = []
    for cls in (1, 2, 3):
        lab, n = ndimage.label(mask == cls)
        if n == 0:
            continue
        slices = ndimage.find_objects(lab)
        for k, sl in enumerate(slices, 1):
            comp = lab[sl] == k
            area = int(comp.sum())
            if area < MIN_AREA:
                continue
            ys, xs = np.nonzero(comp)
            out.append(
                Building(
                    tile,
                    len(out),
                    cls,
                    float(ys.mean() + sl[0].start),
                    float(xs.mean() + sl[1].start),
                    sl[0].start,
                    sl[1].start,
                    sl[0].stop,
                    sl[1].stop,
                    area,
                    "component",
                )
            )
    return out


def footprint(b: Building, mask: np.ndarray) -> np.ndarray:
    """Boolean footprint of one building in tile coordinates."""
    if b.polygon is not None:
        return polygon_mask(b.polygon, mask.shape)
    fp = np.zeros(mask.shape, bool)
    sub = mask[b.y0 : b.y1, b.x0 : b.x1] == b.label
    lab, _ = ndimage.label(sub)
    cy, cx = int(round(b.cy)) - b.y0, int(round(b.cx)) - b.x0
    k = lab[min(max(cy, 0), sub.shape[0] - 1), min(max(cx, 0), sub.shape[1] - 1)]
    fp[b.y0 : b.y1, b.x0 : b.x1] = lab == k if k else sub
    return fp


# --------------------------------------------------------------------------- crops


def crop_box(b: Building, shape: tuple[int, int], crop: int = CROP) -> tuple[int, int, int]:
    """Square window (y0, x0, size) centred on the building; grows for large buildings so
    the whole footprint plus a margin is visible. May extend past the tile (zero-padded)."""
    size = max(crop, int(1.5 * max(b.y1 - b.y0, b.x1 - b.x0)))
    y0 = int(round(b.cy)) - size // 2
    x0 = int(round(b.cx)) - size // 2
    return y0, x0, size


def cut(img: np.ndarray, y0: int, x0: int, size: int, out: int = CROP) -> np.ndarray:
    h, w = img.shape[:2]
    pad = np.zeros((size, size) + img.shape[2:], img.dtype)
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y0 + size, h), min(x0 + size, w)
    if sy1 > sy0 and sx1 > sx0:
        pad[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    if size == out:
        return pad
    resample = Image.NEAREST if img.dtype == bool or img.ndim == 2 else Image.BILINEAR
    mode_img = Image.fromarray(pad.astype(np.uint8) * (255 if img.dtype == bool else 1))
    res = np.asarray(mode_img.resize((out, out), resample))
    return res > 127 if img.dtype == bool else res


def jpeg_bytes(rgb: np.ndarray, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def png_bytes(mask: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def decode_jpeg(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def decode_png(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b))) > 127


def pack(blobs: Sequence[bytes]) -> tuple[np.ndarray, np.ndarray]:
    """Variable-length byte strings -> (uint8 buffer, int64 offsets of length n + 1)."""
    offs = np.zeros(len(blobs) + 1, np.int64)
    offs[1:] = np.cumsum([len(b) for b in blobs])
    return np.frombuffer(b"".join(blobs), np.uint8).copy(), offs


def unpack(buf: np.ndarray, offs: np.ndarray, i: int) -> bytes:
    return buf[offs[i] : offs[i + 1]].tobytes()


def tile_crops(
    pre: np.ndarray, post: np.ndarray, post_valid: np.ndarray, mask: np.ndarray, buildings
) -> dict[str, np.ndarray]:
    """Encode crops of every building of one tile into packed arrays (npz-ready)."""
    pre_b, post_b, fp_b, valid_frac = [], [], [], []
    for b in buildings:
        y0, x0, size = crop_box(b, mask.shape)
        fp = footprint(b, mask)
        pre_b.append(jpeg_bytes(cut(pre, y0, x0, size)))
        post_b.append(jpeg_bytes(cut(post, y0, x0, size)))
        fp_b.append(png_bytes(cut(fp, y0, x0, size)))
        v = post_valid[fp]
        valid_frac.append(float(v.mean()) if v.size else 0.0)
    out: dict[str, np.ndarray] = {}
    for name, blobs in (("pre", pre_b), ("post", post_b), ("fp", fp_b)):
        out[f"{name}_buf"], out[f"{name}_off"] = pack(blobs)
    out["post_valid"] = np.asarray(valid_frac, np.float32)
    return out
