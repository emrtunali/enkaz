"""Project-wide constants and storage locations.

Two storage roots are used:

* ``work``    – fast, ephemeral disk (``/content`` on Colab). Raw downloads live here; they can
  always be fetched again.
* ``persist`` – Google Drive on Colab. Stage markers, catalogues, processed shards and
  checkpoints live here so that a disconnected session can resume.

Both can be overridden with the ``ENKAZ_WORK`` / ``ENKAZ_PERSIST`` environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

EVENT_DATE = "2023-02-06"  # Mw 7.8 Pazarcık and Mw 7.5 Elbistan earthquakes (04:17 / 13:24 local)

MAXAR_EVENT_COLLECTION = (
    "https://maxar-opendata.s3.amazonaws.com/events/"
    "Kahramanmaras-turkey-earthquake-23/collection.json"
)
BRIGHT_BASE = "https://huggingface.co/datasets/Kullervo/BRIGHT/resolve/main"
CEMS_ACTIVATION_PAGE = "https://mapping.emergency.copernicus.eu/activations/EMSR648/"
GEOBOUNDARIES_ADM1_API = "https://www.geoboundaries.org/api/current/gbOpen/TUR/ADM1/"
GEOBOUNDARIES_ADM2_API = "https://www.geoboundaries.org/api/current/gbOpen/TUR/ADM2/"

# Approximate urban centres (lat, lon) used to group BRIGHT tiles into cities.
CITY_CENTERS: dict[str, tuple[float, float]] = {
    "antakya": (36.20, 36.16),
    "kahramanmaras": (37.58, 36.93),
    "islahiye": (37.025, 36.63),
    "gaziantep": (37.07, 37.38),
    "adiyaman": (37.765, 38.27),
    "malatya": (38.35, 38.31),
    "osmaniye": (37.07, 36.25),
}

# Geographic split: the test city is never seen during training or model selection.
SPLITS: dict[str, dict[str, list[str]]] = {
    "main": {
        "train": ["kahramanmaras", "adiyaman", "malatya", "gaziantep", "osmaniye"],
        "val": ["islahiye"],
        "test": ["antakya"],
    },
    "alt_kahramanmaras": {
        "train": ["antakya", "adiyaman", "malatya", "gaziantep", "osmaniye"],
        "val": ["islahiye"],
        "test": ["kahramanmaras"],
    },
}

DAMAGE_CLASSES = ("intact", "damaged", "destroyed")  # BRIGHT values 1, 2, 3 (0 = background)


@dataclass(frozen=True)
class Paths:
    work: Path
    persist: Path

    @property
    def raw(self) -> Path:
        return self.work / "raw"

    @property
    def catalog(self) -> Path:
        return self.persist / "catalog"

    @property
    def processed(self) -> Path:
        return self.persist / "processed"

    @property
    def runs(self) -> Path:
        return self.persist / "runs"

    @property
    def markers(self) -> Path:
        return self.persist / "stages"

    def ensure(self) -> Paths:
        for p in (self.raw, self.catalog, self.processed, self.runs, self.markers):
            p.mkdir(parents=True, exist_ok=True)
        return self


def in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or Path("/content").is_dir()


def default_paths() -> Paths:
    if in_colab():
        work = Path("/content/enkaz_work")
        persist = Path("/content/drive/MyDrive/enkaz")
    else:
        work = Path("data/work")
        persist = Path("data/persist")
    work = Path(os.environ.get("ENKAZ_WORK", work))
    persist = Path(os.environ.get("ENKAZ_PERSIST", persist))
    return Paths(work=work, persist=persist)
