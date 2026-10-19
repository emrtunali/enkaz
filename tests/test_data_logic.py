import json

from enkaz.config import EVENT_DATE
from enkaz.data.bright import tile_id
from enkaz.data.cems import parse_product
from enkaz.data.maxar import choose_post, phase
from enkaz.data.provinces import (
    STATUS_LABELED,
    STATUS_NONE,
    STATUS_PAIRED,
    STATUS_POST_ONLY,
    Provinces,
    coverage_table,
)


def _item(date, bbox, off_nadir=20.0, cloud=0.0):
    return {"date": date, "bbox": bbox, "off_nadir": off_nadir, "cloud_percent": cloud}


AOI = (36.15, 36.19, 36.16, 36.20)
BIG = (36.0, 36.0, 36.5, 36.5)


def test_phase():
    assert phase(_item("2023-02-05", BIG), EVENT_DATE) == "pre"
    assert phase(_item("2023-02-06", BIG), EVENT_DATE) == "post"


def test_choose_post_prefers_early_then_low_off_nadir():
    rows = [
        _item("2022-12-22", BIG, 5),  # pre-event: never chosen
        _item("2023-02-08", BIG, 40),
        _item("2023-02-11", BIG, 30),  # within 10 days of first, lower off-nadir -> chosen
        _item("2023-02-28", BIG, 2),  # too late: debris removal
        _item("2023-02-09", BIG, 10, cloud=60),  # cloudy
        _item("2023-02-09", (37, 37, 38, 38), 1),  # does not cover the AOI
    ]
    best = choose_post(rows, AOI, EVENT_DATE)
    assert best["date"] == "2023-02-11"
    assert choose_post(rows, (40, 40, 41, 41), EVENT_DATE) is None


def _square(x0, y0, x1, y1):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def test_coverage_table_statuses(tmp_path):
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"shapeName": n}, "geometry": _square(*b)}
            for n, b in {
                "Hatay": (36, 36, 37, 37),
                "Kilis": (37, 36, 38, 37),
                "Adana": (35, 36, 36, 37),
                "Ankara": (32, 39, 33, 40),
            }.items()
        ],
    }
    p = tmp_path / "adm1.geojson"
    p.write_text(json.dumps(fc))
    prov = Provinces(p)
    assert prov.at(36.5, 36.5) == "Hatay"
    items = [
        _item("2022-01-01", (36.1, 36.1, 36.2, 36.2)),
        _item("2023-02-08", (36.1, 36.1, 36.2, 36.2)),
        _item("2022-01-01", (37.1, 36.1, 37.2, 36.2)),
        _item("2023-02-08", (37.1, 36.1, 37.2, 36.2)),
        _item("2023-02-08", (35.1, 36.1, 35.2, 36.2)),
        _item("2023-02-08", (50, 50, 50.1, 50.1)),  # outside every province
    ]
    rows = {
        r["province"]: r
        for r in coverage_table(prov, items, EVENT_DATE, {"antakya": (36.2, 36.16)})
    }
    assert rows["Hatay"]["status"] == STATUS_LABELED
    assert rows["Kilis"]["status"] == STATUS_PAIRED
    assert rows["Adana"]["status"] == STATUS_POST_ONLY
    assert rows["Ankara"]["status"] == STATUS_NONE
    assert len(rows) == 4


def test_name_parsers():
    assert tile_id("turkey-earthquake_00000052_pre_disaster.tif") == "turkey-earthquake_00000052"
    meta = parse_product(
        "https://x/static/activations/EMSR648/EMSR648_AOI10_GRA_MONIT02_r1_RTP01_v1_vector.zip"
    )
    assert (meta["aoi"], meta["product"], meta["version"]) == (10, "MONIT02", 1)
    meta = parse_product("https://x/EMSR648_AOI05_GRA_PRODUCT_r1_VECTORS_v1_vector.zip")
    assert (meta["aoi"], meta["product"]) == (5, "PRODUCT")
