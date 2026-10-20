import numpy as np
import pytest

from enkaz import prepare as P
from enkaz.config import EVENT_DATE
from enkaz.data.maxar import post_items_for


def _texture(h=256, w=256, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.random((h, w))
    from scipy import ndimage

    return ndimage.gaussian_filter(img, 2.0)


@pytest.mark.parametrize("dy,dx", [(0, 0), (5, -3), (-12, 17)])
def test_phase_correlation_recovers_shift(dy, dx):
    big = _texture(320, 320)
    ref = big[32:288, 32:288]
    mov = big[32 + dy : 288 + dy, 32 + dx : 288 + dx]  # mov[y, x] = ref[y + dy, x + dx]
    # mov[y - dy, x - dx] = ref[y, x], so the function must report (-dy, -dx)
    got_dy, got_dx, ratio = P.phase_correlation(ref, mov, max_shift=24)
    assert (got_dy, got_dx) == (-dy, -dx)
    assert ratio > 5


def test_shifted_transform_samples_offset_pixel():
    affine = pytest.importorskip("affine")
    t = affine.Affine(0.35, 0, 1000.0, 0, -0.35, 2000.0)
    t2 = P.shifted_transform(t, dy=4, dx=-2)
    # pixel (0, 0) of the new grid is pixel (4, -2) of the old grid
    assert t2 * (0, 0) == pytest.approx(t * (-2, 4))


def test_buildings_from_mask_components_and_classes():
    m = np.zeros((64, 64), np.uint8)
    m[5:15, 5:15] = 1  # intact
    m[30:40, 30:50] = 3  # destroyed
    m[50:52, 50:52] = 2  # too small: dropped
    bs = P.buildings_from_mask("t", m)
    assert sorted((b.label, b.area) for b in bs) == [(1, 100), (3, 200)]
    b = next(b for b in bs if b.label == 3)
    assert (b.y0, b.x0, b.y1, b.x1) == (30, 30, 40, 50)
    assert P.footprint(b, m).sum() == 200


def test_buildings_from_instances():
    inst = {
        "annotations": [
            {"category_id": 2, "segmentation": [[10, 10, 30, 10, 30, 20, 10, 20]], "area": 200},
            {"category_id": 1, "segmentation": [[0, 0, 2, 0, 2, 2]], "area": 2},  # tiny
            {"category_id": 9, "segmentation": [[0, 0, 9, 0, 9, 9]], "area": 50},  # unknown
        ]
    }
    bs = P.buildings_from_instances("t", inst, (64, 64))
    assert len(bs) == 1 and bs[0].label == 2 and bs[0].source == "instance"
    fp = P.footprint(bs[0], np.zeros((64, 64), np.uint8))
    assert 180 <= fp.sum() <= 240


def test_cut_pads_at_border_and_resizes_large_windows():
    img = np.full((50, 50, 3), 7, np.uint8)
    c = P.cut(img, -10, -10, 32, out=32)
    assert c.shape == (32, 32, 3)
    assert c[0, 0].tolist() == [0, 0, 0] and c[20, 20].tolist() == [7, 7, 7]
    big = P.cut(img, 0, 0, 64, out=32)
    assert big.shape == (32, 32, 3)
    fp = P.cut(np.ones((50, 50), bool), 0, 0, 64, out=32)
    assert fp.dtype == bool and fp[:20, :20].all()


def test_crop_box_grows_for_large_buildings():
    b = P.Building("t", 0, 1, 100, 100, 0, 0, 200, 150, 30000, "component")
    y0, x0, size = P.crop_box(b, (1024, 1024))
    assert size == 300 and (y0, x0) == (-50, -50)
    small = P.Building("t", 0, 1, 100, 100, 95, 95, 105, 105, 100, "component")
    assert P.crop_box(small, (1024, 1024))[2] == P.CROP


def test_tile_crops_roundtrip():
    rng = np.random.default_rng(1)
    pre = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    post = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    mask = np.zeros((200, 200), np.uint8)
    mask[90:110, 90:120] = 2
    valid = np.ones((200, 200), bool)
    valid[:, :105] = False
    bs = P.buildings_from_mask("t", mask)
    out = P.tile_crops(pre, post, valid, mask, bs)
    assert len(out["pre_off"]) == len(bs) + 1
    crop = P.decode_jpeg(P.unpack(out["pre_buf"], out["pre_off"], 0))
    fp = P.decode_png(P.unpack(out["fp_buf"], out["fp_off"], 0))
    assert crop.shape == (P.CROP, P.CROP, 3) and fp.shape == (P.CROP, P.CROP)
    assert fp.sum() == 600  # 20x30 footprint, crop not resized
    assert out["post_valid"][0] == pytest.approx(15 / 30)


def test_post_items_for_mosaics_one_acquisition():
    def it(acq, date, bbox, on=20.0):
        return {"acquisition": acq, "date": date, "bbox": bbox, "off_nadir": on, "cloud_percent": 0}

    tile = (36.09, 36.0, 36.11, 36.01)  # straddles lon 36.1
    rows = [
        it("A", "2023-02-08", (36.0, 35.9, 36.1, 36.1), 40),
        it("A", "2023-02-08", (36.1, 35.9, 36.2, 36.1), 40),
        it("B", "2023-02-11", (36.0, 35.9, 36.1, 36.1), 30),  # lower off-nadir, half coverage
        it("C", "2023-02-09", (36.05, 35.95, 36.15, 36.05), 35),  # one item covers it all
    ]
    got = post_items_for(rows, tile, EVENT_DATE)
    assert [r["acquisition"] for r in got] == ["C"]
    got = post_items_for(rows[:3], tile, EVENT_DATE)
    assert {r["acquisition"] for r in got} == {"A"} and len(got) == 2
    assert post_items_for(rows, (40, 40, 41, 41), EVENT_DATE) == []
