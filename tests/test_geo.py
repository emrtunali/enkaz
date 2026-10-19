import pytest

from enkaz.config import CITY_CENTERS
from enkaz.geo import bbox_contains, bbox_intersects, haversine_km, nearest_city, utm_zone_epsg


def test_bbox_predicates():
    outer = (36.0, 36.0, 37.0, 37.0)
    assert bbox_contains(outer, (36.2, 36.2, 36.4, 36.4))
    assert not bbox_contains(outer, (36.9, 36.9, 37.1, 37.1))
    assert bbox_intersects(outer, (36.9, 36.9, 37.1, 37.1))
    assert not bbox_intersects(outer, (37.1, 37.1, 37.2, 37.2))


def test_haversine_known_distance():
    # Antakya -> Kahramanmaraş is roughly 160 km as the crow flies
    d = haversine_km(*CITY_CENTERS["antakya"], *CITY_CENTERS["kahramanmaras"])
    assert 150 < d < 175


def test_nearest_city():
    assert nearest_city(36.205, 36.165, CITY_CENTERS) == "antakya"
    assert nearest_city(39.92, 32.85, CITY_CENTERS) is None  # Ankara: no labeled city nearby


@pytest.mark.parametrize("lon,epsg", [(36.16, 32637), (38.3, 32637), (35.3, 32636), (29.0, 32635)])
def test_utm_zone(lon, epsg):
    assert utm_zone_epsg(lon, 37.0) == epsg
