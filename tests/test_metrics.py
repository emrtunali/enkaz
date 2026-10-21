import numpy as np
import pytest

from enkaz import metrics as M


def test_confusion_and_f1():
    y = np.array([0, 0, 1, 2, 2])
    p = np.array([0, 1, 1, 2, 0])
    cm = M.confusion(y, p)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 1]]
    f1 = M.per_class_f1(cm)
    assert f1 == pytest.approx([0.5, 2 / 3, 2 / 3])


def test_perfect_calibration_has_zero_ece():
    probs = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    assert M.ece(probs, np.array([0, 1])) == pytest.approx(0.0)


def test_overconfidence_is_measured():
    probs = np.tile([0.99, 0.005, 0.005], (100, 1))
    y = np.array([0] * 50 + [1] * 50)  # right only half the time
    assert M.ece(probs, y) == pytest.approx(0.49, abs=1e-6)


def test_temperature_softens_overconfident_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 3000)
    logits = rng.normal(0, 1, (3000, 3))
    logits[np.arange(3000), y] += 1.0
    logits *= 6  # same ranking, far too confident
    t = M.fit_temperature(logits, y)
    assert t > 3
    assert M.ece(M.softmax(logits, t), y) < M.ece(M.softmax(logits), y)


def test_auroc_matches_definition():
    s = np.array([0.1, 0.4, 0.35, 0.8])
    pos = np.array([0, 0, 1, 1], bool)
    assert M.binary_auroc(s, pos) == pytest.approx(0.75)
    assert M.binary_auroc(np.zeros(4), pos) == pytest.approx(0.5)


def test_recall_at_fraction():
    score = np.arange(10.0)
    y = np.array([0] * 8 + [2, 2])
    assert M.recall_at_fraction(score, y, 2, 0.2) == 1.0
    assert M.recall_at_fraction(-score, y, 2, 0.2) == 0.0


def test_report_keys():
    logits = np.eye(3)[[0, 1, 2, 0]] * 5
    r = M.report(logits, np.array([0, 1, 2, 0]))
    assert r["macro_f1"] == pytest.approx(1.0)
    assert set(r["f1"]) == set(M.CLASSES)


def test_crop_position_is_row_order_within_tile():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("torch")
    from enkaz.learn import select

    bdf = pd.DataFrame(
        {
            "tile": ["a", "a", "a", "b", "b"],
            "idx": [0, 5, 9, 2, 3],  # instance ids with gaps
            "label": [1, 2, 3, 1, 1],
            "city": ["x"] * 5,
            "post_valid": [1.0, 0.1, 1.0, 1.0, 1.0],
        }
    )
    rows = select(bdf, ["x"])
    assert rows["crop"].tolist() == [
        0,
        2,
        0,
        1,
    ]  # the 0.1-valid building is dropped, positions kept
