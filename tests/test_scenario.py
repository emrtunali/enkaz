import numpy as np
import pytest

from enkaz import scenario as S


def test_vs30_from_slope_follows_wald_allen_edges():
    v = S.vs30_from_slope(np.array([1e-6, 1e-4, 2.2e-3, 0.018, 0.138, 0.5]))
    assert v.tolist() == pytest.approx([180, 180, 240, 360, 760, 760])
    mid = S.vs30_from_slope(np.array([np.sqrt(0.05 * 0.10)]))  # geometric middle of a bin
    assert mid[0] == pytest.approx(555, abs=1)


def test_mmi_from_pga_worden_2012():
    assert S.mmi_from_pga(np.array([0.1]))[0] == pytest.approx(
        -1.60 + 3.70 * np.log10(98.0665), abs=0.01
    )
    assert S.mmi_from_pga(np.array([0.01]))[0] == pytest.approx(
        1.78 + 1.55 * np.log10(9.80665), abs=0.01
    )
    assert S.mmi_from_pga(np.array([5.0]))[0] == 10.0


@pytest.mark.parametrize(
    "tax, fam",
    [
        ("CR/LFINF(CL)+CDL+LFC:15.0/HBET:4-6", "CR_LFINF-CDL-15"),
        ("CR/LWAL+CDH+LFC:13.0/HBET:1-3", "CR_LWAL-DUH"),
        ("CR/LDUAL+CDH+LFC:13.0/HBET:7-", "CR_LDUAL-DUH"),
        ("CR/LWAL+CDL+LFC:15.0/HBET:1-3", "CR_LWAL-DUL"),
        ("CR/LFINF+CDN/HBET:1-3", "CR_LFINF-CDN-0"),
        ("MUR+CL/LWAL+CDN/HBET:1-3", "MUR-CL99_LWAL-DNO"),
        ("MUR+CBH/LWAL+CDN/HBET:1-3", "MUR-CB99_LWAL-DNO"),
        ("W/LWAL+CDN/HBET:1-3", "W_LFM-DUL"),
        ("CR+PC/LWAL+CDL+LFC:15.0/HBET:1-3", "CR_LWAL-DUL"),
        ("S/LFM+CDL/HBET:1-3", "S_LFM-DUM"),
        ("CR/LFINF(CBH)+CDL+LFC:15.0/HBET:4-6/SOS", "CR_LFINF-CDL-15"),
    ],
)
def test_fragility_family(tax, fam):
    assert S.fragility_family(tax) == fam


def test_class_mix_by_age_and_band():
    rows = [
        ("CR/LFINF(CL)+CDL+LFC:15.0/HBET:1-3", 60.0),
        ("MUR+CL/LWAL+CDN/HBET:1-3", 40.0),
        ("CR/LWAL+CDH+LFC:13.0/HBET:1-3", 30.0),
        ("CR/LFINF(CL)+CDL+LFC:15.0/HBET:4-6", 10.0),
        ("CR/LDUAL+CDH+LFC:13.0/HBET:4-6", 10.0),
        ("CR/LDUAL+CDH+LFC:13.0/HBET:7-", 5.0),
        ("CR/LFINF(CBH)+CDL+LFC:15.0/HBET:4-6/SOS", 0.0),
    ]
    mix = S.class_mix(rows)
    assert mix[(1, 0)] == pytest.approx({"CR_LFINF-CDL-15": 0.6, "MUR-CL99_LWAL-DNO": 0.4})
    assert mix[(0, 0)] == mix[(1, 0)]
    assert mix[(2, 0)] == {"CR_LWAL-DUH": 1.0}
    assert mix[(1, 2)] == {"CR_LFINF-CDL-15": 1.0}  # no old high-rise rows: band 1 is used
    for d in mix.values():
        assert sum(d.values()) == pytest.approx(1.0)


def test_damage_state_probabilities_sum_to_one():
    ex = S.lognormal_exceed(np.array([0.05, 0.3, 2.0]), np.array([0.1, 0.2, 0.4, 0.8]), 0.6)
    assert (np.diff(ex, axis=1) <= 0).all()
    p = S.ds_probabilities(ex)
    assert p.shape == (3, 5) and p.sum(1) == pytest.approx(np.ones(3))
    assert p[0, 0] > 0.8 and p[2, 4] > 0.8


def test_debris_and_blocking():
    # 7 m street, facade 5 m from the centreline (1.5 m pavement): a 4-storey collapse (12 m)
    # throws 6 m of debris, covering 4.5 m of the carriageway
    assert S.free_width(np.array([5.0]), np.array([7.0]), np.array([12.0]))[0] == pytest.approx(2.5)
    assert S.free_width(np.array([30.0]), np.array([7.0]), np.array([12.0]))[0] == pytest.approx(
        7.0
    )
    edges = S.blocked_edges(
        3,
        np.array([0, 1, 1, 2]),
        np.array([2.5, 6.0, 1.0, 0.0]),
        np.array([True, True, False, False]),
    )
    assert edges.tolist() == [True, False, False]
