"""Istanbul earthquake scenario: shaking, building damage, deaths, blocked roads, hospital access.

This is a *scenario*, not a forecast. It answers "if a Mw 7.5 earthquake ruptured the Main Marmara
Fault south of Istanbul, what would the published models say happens?", with the uncertainty
those models carry. Every ingredient is a published, citable model or open dataset:

* **Rupture.** The Main Marmara Fault segment from the Central Marmara Basin to Çınarcık (GEM
  Global Active Faults, EMME catalogue, 104 km), vertical, 0–20 km deep, right-lateral, Mw 7.5,
  the magnitude İBB uses for its own loss scenario.
* **Ground motion.** Two GMPEs, Akkar, Sandıkkaya & Bommer (2014) and Boore et al. (2014), equally
  weighted, through OpenQuake; spatially correlated within-event residuals (Jayaram & Baker, 2009).
* **Site conditions.** Vs30 from topographic slope (Wald & Allen, 2007) on the Copernicus DEM.
* **Buildings.** The OSM + Microsoft footprint inventory. Each building gets a mix of building
  classes from the ESRM20 exposure model for Istanbul, re-weighted per mahalle by İBB's building
  age and height statistics; storeys from OSM where tagged.
* **Damage and deaths.** ESRM20 fragility and loss-of-life functions (CC BY 4.0) per class.
* **Roads.** A collapsed building's debris reaches about half its height into the street
  (Argyroudis et al. 2015 use 0.5–0.7 H); a street is blocked when less than 3 m stay free.

The functions here are NumPy only; ``scripts/istanbul_scenario.py`` wires them to OpenQuake.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np

MAG = 7.5
RAKE = 180.0  # right-lateral strike-slip
UPPER_KM, LOWER_KM = 0.0, 20.0
# Main Marmara Fault, GEM Global Active Faults (EMME catalogue, ME_TRCS013), lon/lat
FAULT_TRACE = [
    (28.056, 40.797), (28.117, 40.808), (28.177, 40.819), (28.238, 40.829), (28.298, 40.840),
    (28.360, 40.846), (28.421, 40.852), (28.483, 40.858), (28.545, 40.864), (28.607, 40.866),
    (28.669, 40.869), (28.731, 40.872), (28.793, 40.875), (28.854, 40.866), (28.915, 40.857),
    (28.968, 40.833), (29.021, 40.809), (29.075, 40.785), (29.128, 40.761), (29.228, 40.716),
]  # fmt: skip
IMTS = ("PGA", "SA(0.3)", "SA(0.6)", "SA(1.0)")
DEBRIS_K = 0.5  # debris reach as a fraction of building height
STOREY_M = 3.0
MIN_FREE_M = 3.0  # an emergency vehicle needs about one lane
ROAD_WIDTH_M = {
    "motorway": 25, "trunk": 20, "primary": 16, "secondary": 13, "tertiary": 10,
    "unclassified": 7, "residential": 7, "living_street": 5, "service": 5,
    "motorway_link": 8, "trunk_link": 8, "primary_link": 8, "secondary_link": 8, "tertiary_link": 7,
}  # fmt: skip
ROAD_KMH = {
    "motorway": 70, "trunk": 55, "primary": 40, "secondary": 35, "tertiary": 30,
    "unclassified": 25, "residential": 20, "living_street": 10, "service": 15,
    "motorway_link": 40, "trunk_link": 35, "primary_link": 30, "secondary_link": 25,
    "tertiary_link": 25,
}  # fmt: skip


# --------------------------------------------------------------------------- site and intensity

# Wald & Allen (2007), active tectonic regions: slope (m/m) bin edges and Vs30 (m/s) at the edges
_WA_SLOPE = np.array([1.0e-4, 2.2e-3, 6.3e-3, 0.018, 0.050, 0.10, 0.138])
_WA_VS30 = np.array([180.0, 240.0, 300.0, 360.0, 490.0, 620.0, 760.0])


def vs30_from_slope(slope: np.ndarray) -> np.ndarray:
    """Vs30 (m/s) from topographic slope on a 30 arc-second grid (Wald & Allen, 2007), linear in
    log-slope between the published class edges, clamped to 180–760 m/s."""
    s = np.clip(np.asarray(slope, float), _WA_SLOPE[0], _WA_SLOPE[-1])
    return np.interp(np.log10(s), np.log10(_WA_SLOPE), _WA_VS30)


def slope_m_per_m(dem: np.ndarray, lat: float, dlon_deg: float, dlat_deg: float) -> np.ndarray:
    """Gradient magnitude of a lon/lat DEM (metres) in m/m."""
    dy = dlat_deg * 111_320.0
    dx = dlon_deg * 111_320.0 * math.cos(math.radians(lat))
    gy, gx = np.gradient(dem.astype(float), dy, dx)
    return np.hypot(gx, gy)


def mmi_from_pga(pga_g: np.ndarray) -> np.ndarray:
    """Modified Mercalli intensity from PGA (g) after Worden et al. (2012)."""
    lg = np.log10(np.maximum(np.asarray(pga_g, float), 1e-6) * 980.665)
    return np.clip(np.where(lg <= 1.57, 1.78 + 1.55 * lg, -1.60 + 3.70 * lg), 1.0, 10.0)


# --------------------------------------------------------------------------- building classes


def fragility_family(taxonomy: str) -> str:
    """ESRM20 exposure taxonomy (GEM v2) -> ESRM20 fragility family (storeys appended later)."""
    t = taxonomy.split("/HBET")[0]
    if t.startswith("MUR"):
        for key, fam in (("+CL", "MUR-CL99"), ("+CBH", "MUR-CB99"), ("+STRUB", "MUR-STRUB"),
                         ("+ADO", "MUR-ADO")):  # fmt: skip
            if key in t.split("/")[0]:
                return f"{fam}_LWAL-DNO"
        return "MUR_LWAL-DNO"
    if t.startswith("W/"):
        return "W_LFM-DUL"
    if t.startswith("S/"):  # steel: ESRM20 has no low-ductility steel class, take the moderate one
        return "S_LFM-DUM"
    if t.startswith("CR/") or t.startswith("CR+"):  # CR+PC: precast concrete
        code = re.search(r"\+CD([NLMH])", t)
        code = code.group(1) if code else "N"
        if "LFINF" in t or "LFM" in t:
            if code == "N":
                return "CR_LFINF-CDN-0"
            lfc = re.search(r"LFC:([\d.]+)", t)
            lfc = (
                min((0, 5, 10, 15, 20, 25, 30), key=lambda v: abs(v - float(lfc.group(1))))
                if lfc
                else 15
            )
            return f"CR_{'LFM' if 'LFM' in t else 'LFINF'}-CD{code}-{lfc}"
        system = "LDUAL" if "LDUAL" in t else "LWAL"
        return f"CR_{system}-DU{'L' if code in 'NL' else code}"
    raise ValueError(f"no fragility family for {taxonomy}")


def height_band(storeys) -> np.ndarray:
    """0: 1–4 storeys, 1: 5–9, 2: 10+ (İBB's classes)."""
    s = np.asarray(storeys, float)
    return np.where(s >= 10, 2, np.where(s >= 5, 1, 0))


SMALL_M2 = 50.0  # footprints this small without a storey tag are sheds, kiosks, annexes
SMALL_STOREYS = {1: 0.6, 2: 0.4}
ESRM_BAND = {"1-3": 0, "4-6": 1, "7-": 2}  # ESRM20 HBET classes, mapped onto İBB's bands
# storeys assumed inside a band when a building's own storeys are unknown
BAND_STOREYS = (
    {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.2},
    {5: 0.4, 6: 0.3, 7: 0.2, 8: 0.1},
    {10: 0.5, 12: 0.5},
)
MODERN = re.compile(r"CD[MH]|DU[MH]")


def class_mix(esrm_rows: Sequence[tuple[str, float]]) -> dict:
    """Shares of fragility families per (age group, height band) from ESRM20's Istanbul rows
    ``(taxonomy, buildings)``.

    Age groups follow İBB: 0 before 1980, 1 1980–2000, 2 after 2000. Buildings after 2000 take
    ESRM's moderate/high-code classes, older ones its low/no-code classes and masonry. (ESRM20's
    no-code infilled frame is not more fragile than its low-code one, so pre-1980 buildings get
    the same mix as 1980–2000 ones rather than a guess of ours.)
    """
    acc: dict[tuple[int, int], dict[str, float]] = {}
    for tax, n in esrm_rows:
        band = ESRM_BAND[tax.split("HBET:")[1].split("/")[0]] if "HBET:" in tax else 0
        fam = fragility_family(tax)
        modern = bool(MODERN.search(fam))
        for age in (2,) if modern else (0, 1):
            d = acc.setdefault((age, band), {})
            d[fam] = d.get(fam, 0.0) + n
    out = {}
    for age in range(3):
        for band in range(3):
            # a class ESRM does not list for this height falls back to the next lower band
            d = next(acc[(age, b)] for b in range(band, -1, -1) if (age, b) in acc)
            tot = sum(d.values())
            out[(age, band)] = {f: v / tot for f, v in d.items()}
    return out


def lognormal_exceed(im: np.ndarray, medians: np.ndarray, beta: float) -> np.ndarray:
    """P(DS >= k | IM) for k = 1..4; ``im`` shape (n,), ``medians`` (4,) -> (n, 4)."""
    from scipy.special import ndtr

    z = (np.log(np.maximum(im, 1e-9))[:, None] - np.log(medians)[None, :]) / beta
    return ndtr(z)


def ds_probabilities(exceed: np.ndarray) -> np.ndarray:
    """P(DS >= k), k=1..4 -> P(DS = k), k=0..4 (none, slight, moderate, extensive, complete)."""
    e = np.concatenate(
        [np.ones(exceed.shape[:-1] + (1,)), exceed, np.zeros(exceed.shape[:-1] + (1,))], -1
    )
    return np.clip(e[..., :-1] - e[..., 1:], 0, 1)


# --------------------------------------------------------------------------- roads


def free_width(dist_m: np.ndarray, road_w: np.ndarray, height_m: np.ndarray, k: float = DEBRIS_K):
    """Street width left free by one collapsed building whose facade is ``dist_m`` from the road
    centreline: its debris covers ``k * height`` outward from the facade."""
    near_edge = dist_m - road_w / 2  # facade to the near kerb
    covered = np.clip(k * height_m - np.maximum(near_edge, 0), 0, road_w)
    return road_w - covered


def blocked_edges(n_edges: int, pair_edge: np.ndarray, pair_free: np.ndarray,
                  collapsed: np.ndarray, min_free: float = MIN_FREE_M) -> np.ndarray:  # fmt: skip
    """Edges with less than ``min_free`` metres left by at least one collapsed neighbour.

    ``pair_edge``, ``pair_free`` and ``collapsed`` are aligned on building–edge pairs."""
    hit = collapsed & (pair_free < min_free)
    out = np.zeros(n_edges, bool)
    out[pair_edge[hit]] = True
    return out


# --------------------------------------------------------------------------- animation

VP_KMS, VS_KMS = 6.0, 3.5  # crustal P and S wave speeds
VR_KMS = 0.8 * VS_KMS  # rupture speed, a typical 80 % of the S wave speed
CENTROID_KM = 10.0  # depth of the rupture's energy release used for travel times


def rupture_length_km(mag: float) -> float:
    """Subsurface rupture length of a strike-slip earthquake (Wells & Coppersmith, 1994)."""
    return 10 ** (-2.57 + 0.62 * mag)


def rupture_width_km(mag: float) -> float:
    """Down-dip rupture width of a strike-slip earthquake (Wells & Coppersmith, 1994)."""
    return 10 ** (-0.76 + 0.27 * mag)


def along(xy: np.ndarray) -> np.ndarray:
    """Cumulative distance along a polyline (same units as ``xy``)."""
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])


def sub_trace(xy: np.ndarray, center: float, length: float) -> tuple[np.ndarray, float]:
    """The piece of a polyline of ``length`` centred at along-distance ``center``, shifted to stay
    on the line. Returns the densified piece (1 km steps) and where its centre ended up."""
    s = along(xy)
    half = min(length, s[-1]) / 2
    c = float(np.clip(center, half, s[-1] - half))
    q = np.linspace(c - half, c + half, max(int(2 * half) + 1, 2))
    pts = np.column_stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])])
    return pts, c - (c - half)


def arrival_times(sites: np.ndarray, rup: np.ndarray, hypo: int, depth: float = CENTROID_KM):
    """First P arrival (from the hypocentre) and first S arrival (from whichever ruptured patch
    gets its S waves there first, counting the time the rupture takes to reach the patch), in s.

    ``sites`` (n, 2) and ``rup`` (m, 2) are in km; ``hypo`` indexes ``rup``.
    """
    t_rup = np.abs(along(rup) - along(rup)[hypo]) / VR_KMS
    d = np.sqrt(((sites[:, None, :] - rup[None, :, :]) ** 2).sum(-1) + depth**2)
    tp = d[:, hypo] / VP_KMS
    ts = (t_rup[None, :] + d / VS_KMS).min(axis=1)
    return tp, ts, d.min(axis=1)


def strong_duration(length_km: float, r_km: np.ndarray) -> np.ndarray:
    """Approximate duration of strong shaking (s): the bilateral source duration, plus a path term
    of 0.1 s per km and a 2 s floor; the order of magnitude of significant-duration models."""
    return length_km / (2 * VR_KMS) + 0.1 * np.asarray(r_km) + 2.0
