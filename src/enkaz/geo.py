"""Small geographic helpers that do not need raster libraries."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

BBox = Sequence[float]  # (min_lon, min_lat, max_lon, max_lat)


def bbox_intersects(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def bbox_contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def nearest_city(
    lat: float, lon: float, centers: Mapping[str, tuple[float, float]], max_km: float = 15.0
) -> str | None:
    """Return the closest city name, or ``None`` if nothing is within ``max_km``."""
    best, best_d = None, float("inf")
    for name, (clat, clon) in centers.items():
        d = haversine_km(lat, lon, clat, clon)
        if d < best_d:
            best, best_d = name, d
    return best if best_d <= max_km else None


def utm_zone_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def union_bbox(boxes: Iterable[BBox]) -> tuple[float, float, float, float]:
    boxes = list(boxes)
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
