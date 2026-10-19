"""Maxar Open Data STAC catalogue for the 2023 Kahramanmaraş earthquake event.

Imagery licence: CC BY-NC 4.0 (stated in the catalogue itself). Attribution required.
"""

from __future__ import annotations

import concurrent.futures as cf
import urllib.parse
from collections.abc import Iterable, Sequence

from ..geo import BBox, bbox_contains
from .http import get_json


def _child_links(doc: dict, rel: str) -> list[str]:
    return [link["href"] for link in doc.get("links", []) if link.get("rel") == rel]


def _item_row(url: str) -> dict:
    d = get_json(url)
    p = d["properties"]
    assets = d.get("assets", {})

    def href(key: str) -> str | None:
        a = assets.get(key)
        return urllib.parse.urljoin(url, a["href"]) if a else None

    return {
        "id": d["id"],
        "acquisition": d["id"].split("/")[-1],
        "datetime": p.get("datetime"),
        "date": (p.get("datetime") or "")[:10],
        "bbox": d["bbox"],
        "platform": p.get("platform"),
        "gsd": p.get("gsd"),
        "off_nadir": p.get("view:off_nadir"),
        "sun_elevation": p.get("view:sun_elevation"),
        "cloud_percent": p.get("tile:clouds_percent"),
        "epsg": p.get("proj:epsg"),
        "visual": href("visual"),
        "pan": href("pan_analytic"),
        "data_mask": href("data-mask"),
        "item_url": url,
    }


def crawl_event(collection_url: str, workers: int = 32, log=print) -> list[dict]:
    """Return one row per STAC item (one Maxar ARD tile) of the event."""
    col = get_json(collection_url)
    acq_urls = [urllib.parse.urljoin(collection_url, h) for h in _child_links(col, "child")]

    def acq_items(u: str) -> list[str]:
        return [urllib.parse.urljoin(u, h) for h in _child_links(get_json(u), "item")]

    with cf.ThreadPoolExecutor(workers) as ex:
        item_urls = [u for us in ex.map(acq_items, acq_urls) for u in us]
    log(f"{len(acq_urls)} acquisitions, {len(item_urls)} items")
    rows: list[dict] = []
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(_item_row, item_urls), 1):
            rows.append(row)
            if i % 500 == 0:
                log(f"  {i}/{len(item_urls)} items")
    rows.sort(key=lambda r: (r["date"], r["id"]))
    return rows


def phase(row: dict, event_date: str) -> str:
    return "pre" if row["date"] < event_date else "post"


def covering(rows: Iterable[dict], bbox: BBox) -> list[dict]:
    """Items whose footprint bbox fully contains ``bbox``."""
    return [r for r in rows if bbox_contains(r["bbox"], bbox)]


def choose_post(
    rows: Sequence[dict],
    bbox: BBox,
    event_date: str,
    max_off_nadir: float = 45.0,
    max_cloud: float = 10.0,
    max_days_after: int = 10,
) -> dict | None:
    """Pick the post-event tile for ``bbox``.

    Earliest acquisitions are preferred: later scenes increasingly show debris removal and
    demolition, which would no longer match the damage labels. Among scenes within
    ``max_days_after`` of the earliest acceptable one, the lowest off-nadir angle wins
    (less building lean and occlusion).
    """
    import datetime as dt

    cands = [
        r
        for r in covering(rows, bbox)
        if phase(r, event_date) == "post"
        and (r["off_nadir"] or 0) <= max_off_nadir
        and (r["cloud_percent"] or 0) <= max_cloud
    ]
    if not cands:
        return None
    first = min(dt.date.fromisoformat(r["date"]) for r in cands)
    window = [r for r in cands if (dt.date.fromisoformat(r["date"]) - first).days <= max_days_after]
    return min(window, key=lambda r: (r["off_nadir"] or 0, r["date"]))
