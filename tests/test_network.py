"""Live checks against the real sources. Run with ``pytest -m network``."""

import pytest

from enkaz.config import BRIGHT_BASE, CEMS_ACTIVATION_PAGE, MAXAR_EVENT_COLLECTION
from enkaz.data import bright, cems
from enkaz.data.http import get_json
from enkaz.data.remotezip import RemoteZip

pytestmark = pytest.mark.network


def test_maxar_collection_licence():
    col = get_json(MAXAR_EVENT_COLLECTION)
    assert col["license"] == "CC-BY-NC-4.0"
    assert sum(link["rel"] == "child" for link in col["links"]) >= 70


def test_cems_grading_packages_listed():
    urls = cems.vector_zip_urls(CEMS_ACTIVATION_PAGE)
    assert len(urls) >= 30
    assert {cems.parse_product(u)["aoi"] for u in urls} >= {4, 10, 11, 16}


def test_bright_turkey_members():
    rz = RemoteZip.from_url(f"{BRIGHT_BASE}/target.zip")
    assert len(bright.members(rz)) == 1114
