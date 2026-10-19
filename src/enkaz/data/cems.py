"""Copernicus EMS Rapid Mapping activation EMSR648 (grading products).

Used as an independent reference, not as training labels: gradings are at building-block
level and the first products (7–8 Feb 2023, Pléiades imagery) under-report damage in several
AOIs. Attribution: © European Union, Copernicus Emergency Management Service.
"""

from __future__ import annotations

import html
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from .http import download, get_bytes

_ZIP_RE = re.compile(r"https://cems-mapping-website[^\"'<> ]*?_GRA_[^\"'<> ]*?_vector\.zip")


def vector_zip_urls(activation_page_url: str) -> list[str]:
    page = html.unescape(get_bytes(activation_page_url).decode("utf-8", "replace"))
    return sorted(set(_ZIP_RE.findall(page)))


def parse_product(url: str) -> dict:
    """'.../EMSR648_AOI10_GRA_MONIT02_r1_RTP01_v1_vector.zip' -> aoi/product/version."""
    m = re.search(r"EMSR\d+_AOI(\d+)_GRA_([A-Z0-9]+)_r(\d+)_\w+?_v(\d+)_vector\.zip", url)
    if not m:
        raise ValueError(url)
    return {
        "aoi": int(m[1]),
        "product": m[2],
        "release": int(m[3]),
        "version": int(m[4]),
        "url": url,
    }


def download_all(activation_page_url: str, dest_dir: Path, log=print) -> list[dict]:
    dest_dir = Path(dest_dir)
    out = []
    for url in vector_zip_urls(activation_page_url):
        meta = parse_product(url)
        z = download(url, dest_dir / Path(url).name)
        folder = dest_dir / z.stem
        if not folder.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(folder)
        meta["folder"] = str(folder)
        out.append(meta)
    log(f"{len(out)} grading vector packages in {dest_dir}")
    return out


def summarize_builtup(folder: Path) -> dict:
    """Locality and damage-grade counts of the builtUpA layer of one product."""
    folder = Path(folder)
    aoi = next(folder.glob("*areaOfInterestA*.json"), None)
    locality = None
    if aoi is not None:
        feats = json.loads(aoi.read_text())["features"]
        locality = feats[0]["properties"].get("locality") if feats else None
    built = next(folder.glob("*builtUpA*.json"), None)
    grades: Counter = Counter()
    notation: Counter = Counter()
    if built is not None:
        for f in json.loads(built.read_text())["features"]:
            grades[f["properties"].get("damage_gra")] += 1
            notation[f["properties"].get("notation")] += 1
    return {"locality": locality, "grades": dict(grades), "notation": dict(notation)}
