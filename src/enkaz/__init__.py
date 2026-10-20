"""Enkaz: building-level damage assessment from pre/post satellite imagery."""

import logging

__version__ = "0.1.0"

# Maxar COGs carry both an internal mask and a nodata value; GDAL warns about it on every read,
# which buries the progress output of the preparation stage under thousands of identical lines.
logging.getLogger("rasterio._env").setLevel(logging.ERROR)
