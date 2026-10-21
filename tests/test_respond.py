import numpy as np
import pytest

from enkaz import istanbul as I
from enkaz import respond as R

# --------------------------------------------------------------------------- istanbul


@pytest.mark.parametrize(
    "a, b",
    [
        ("HEYBELİADA", "Heybeliada Mahallesi"),
        ("ŞİLE", "Şile"),
        ("YENİ MAHALLE", "Yenimahalle"),
        ("EYÜP", "Eyüpsultan"),
        ("ÇAKMAKLI", "Çakmaklı Mah."),
    ],
)
def test_norm_matches_ibb_and_osm_spellings(a, b):
    assert I.norm(a) == I.norm(b)


def test_norm_keeps_koy_inside_a_name():
    assert I.norm("DURSUNKÖY") == "DURSUNKOY"
    assert I.norm("Dursunköy Mahallesi") == "DURSUNKOY"
    assert I.norm("Kızılca Köyü") == "KIZILCA"


def test_read_ibb_csv_turkish_code_page(tmp_path):
    p = tmp_path / "s.csv"
    p.write_bytes("ilce_adi;mahalle_adi;agir\r\nADALAR;HEYBELİADA;175\r\n;;\r\n".encode("cp1254"))
    rows = I.read_ibb_csv(p)
    assert rows == [{"ilce_adi": "ADALAR", "mahalle_adi": "HEYBELİADA", "agir": 175}]


def test_join_leaves_ambiguous_names_unmatched():
    mah = [
        {"district": "Fatih", "name": "Balat Mahallesi"},
        {"district": "Kartal", "name": "Yalı Mahallesi"},
        {"district": "Kartal", "name": "Yalı"},  # duplicate key after normalising
    ]
    rows = [
        {"ilce_adi": "FATİH", "mahalle_adi": "BALAT"},
        {"ilce_adi": "KARTAL", "mahalle_adi": "YALI"},
    ]
    assert I.join_ibb(mah, rows) == {0: rows[0]}


def test_ring_area():
    # 0.001° x 0.001° at 41° N is ~111.3 m x ~84.0 m
    lon = np.array([29.0, 29.001, 29.001, 29.0])
    lat = np.array([41.0, 41.0, 41.001, 41.001])
    assert I.ring_area_m2(lon, lat) == pytest.approx(111.32 * 84.02, rel=0.01)


# --------------------------------------------------------------------------- respond


def test_cell_raster_contains_its_buildings_with_margin():
    x, y = np.array([1234.5, 1799.0]), np.array([600.1, 1199.9])
    keys = R.cell_of(x, y)
    assert keys.tolist() == [[2, 1], [2, 1]]
    left, top, h, w = R.cell_grid(keys[0])
    assert left == 1200 - R.MARGIN_M and top == 1200 + R.MARGIN_M
    assert h == w == round((R.CELL_M + 2 * R.MARGIN_M) / R.RES)
    col, row = (x - left) / R.RES, (top - y) / R.RES
    assert (col > R.MARGIN_M / R.RES - 1).all() and (row < h).all()


def test_footprints_in_pixel_grid():
    ring = np.array(
        [[100.0, 200.0], [110.5, 200.0], [110.5, 189.5], [100.0, 189.5], [100.0, 200.0]]
    )
    (b,) = R.footprints([ring], left=93.0, top=207.0)
    assert b.x0 == 20 and b.y0 == 20  # 7 m / 0.35 m
    assert b.area == pytest.approx((10.5 / 0.35) ** 2, rel=0.01)
    mask = P_mask(b, (128, 128))
    assert mask[35, 35] and not mask[10, 10]


def P_mask(b, shape):
    from enkaz import prepare as P

    return P.polygon_mask(b.polygon, shape)


def test_rank_puts_unusable_last_and_severe_first():
    sev = np.array([0.1, 1.5, 0.9, 2.0])
    usable = np.array([True, True, True, False])
    assert R.rank(sev, usable).tolist() == [3, 1, 2, 4]


def test_calibrate_uses_offsets_for_class_but_not_for_severity():
    logits = np.array([[2.0, 1.5, 0.0]])
    _, pred0, s0 = R.calibrate(logits, 1.0, None)
    _, pred1, s1 = R.calibrate(logits, 1.0, [0, 1.0, 0])
    assert pred0[0] == 0 and pred1[0] == 1 and s0 == pytest.approx(s1)


def test_run_end_to_end_with_fake_imagery_and_model():
    """Two buildings ~300 m apart; the image is bright only on the second. A fake model that
    reads brightness inside the footprint must rank the second one first."""
    pytest.importorskip("pyproj")
    from pyproj import Transformer

    to_ll = Transformer.from_crs("EPSG:32635", "EPSG:4326", always_xy=True)

    def square(x, y, s=12.0):
        pts = np.array([[x, y], [x + s, y], [x + s, y - s], [x, y - s], [x, y]])
        lon, lat = to_ll.transform(pts[:, 0], pts[:, 1])
        return np.column_stack([lon, lat])

    x0, y0 = 668_200.0, 4_543_500.0  # central Istanbul in UTM 35N
    rings = [square(x0, y0), square(x0 + 300, y0 - 50)]
    inv = {
        "rings": rings,
        "lon": np.array([r[:-1, 0].mean() for r in rings]),
        "lat": np.array([r[:-1, 1].mean() for r in rings]),
        "osm_id": np.array([1, 2]),
        "mahalle": np.array([0, 0]),
    }

    def images(post, pre, crs, left, top, shape):
        img = np.full(shape + (3,), 40, np.uint8)
        c0 = int((x0 + 300 - left) / R.RES)
        r0 = int((top - (y0 - 50)) / R.RES)
        img[r0 : r0 + 34, c0 : c0 + 34] = 230  # the second building's roof
        return img, img, np.ones(shape, bool), (0, 0)

    seen = []

    def predict(x_pre, x_post):
        fp = x_post[:, 3] > 0
        assert fp[:, 64, 64].all()  # footprint centred in every crop
        bright = np.array([x[0][f].mean() for x, f in zip(x_post, fp)])
        seen.append(len(bright))
        return np.column_stack([-bright, np.zeros_like(bright), bright])

    res = R.run(inv, ["post.tif"], predict, images=images, log=lambda *_: None)
    assert res["usable"].all()
    assert res["rank"].tolist() == [2, 1]
    assert res["pred"].tolist() == [0, 2]
    assert sum(seen) == 2


def test_cell_images_reads_geotiff_and_aligns_post_to_pre(tmp_path):
    """Real raster path: a post scene offset by 3 m from the pre scene is shifted back."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    rng = np.random.default_rng(0)
    scene = (rng.random((900, 900)) * 255).astype(np.uint8)
    from scipy import ndimage

    scene = ndimage.gaussian_filter(scene, 2).astype(np.uint8)
    scene = (scene - scene.min()) * (255 // max(1, scene.max() - scene.min()))
    left, top = 668_000.0, 4_544_000.0

    def write(path, x0):
        prof = dict(driver="GTiff", width=900, height=900, count=3, dtype="uint8",
                    crs="EPSG:32635", transform=from_origin(x0, top, R.RES, R.RES))  # fmt: skip
        with rasterio.open(path, "w", **prof) as ds:
            for b in (1, 2, 3):
                ds.write(np.maximum(scene, 1), b)

    write(tmp_path / "pre.tif", left)
    write(tmp_path / "post.tif", left + 3.0)  # same ground, georeferenced 3 m east
    shape = (400, 400)
    pre, post, valid, shift = R.cell_images(
        [str(tmp_path / "post.tif")], [str(tmp_path / "pre.tif")], "EPSG:32635",
        left + 50, top - 50, shape,
    )  # fmt: skip
    assert valid.all()
    assert shift[1] == pytest.approx(3.0 / R.RES, abs=1)  # grid moved east onto the content
    a, b = pre[..., 0].astype(float), post[..., 0].astype(float)
    assert np.corrcoef(a[20:-20, 20:-20].ravel(), b[20:-20, 20:-20].ravel())[0, 1] > 0.9
