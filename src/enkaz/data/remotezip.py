"""Read individual members of a large remote ZIP via HTTP range requests.

BRIGHT ships one multi-GB archive per modality for all events. We only need the Türkiye
members, so we read the central directory from the end of the file and fetch members one by
one. Supports stored and deflated members and ZIP64 archives.
"""

from __future__ import annotations

import struct
import urllib.request
import zlib
from collections.abc import Callable
from dataclasses import dataclass

from .http import get_bytes, open_url

Fetch = Callable[[int, int], bytes]  # inclusive byte range -> bytes


@dataclass(frozen=True)
class ZipEntry:
    name: str
    method: int
    compressed_size: int
    size: int
    header_offset: int


def http_fetcher(url: str, timeout: float = 120) -> tuple[Fetch, int]:
    """Return a range fetcher and the total size, following redirects (e.g. HF -> CDN).

    A one-byte ranged GET is used instead of HEAD because urllib turns redirected HEAD
    requests into full GETs.
    """
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with open_url(req, timeout) as r:
        final_url = r.url
        content_range = r.headers.get("Content-Range", "")
        r.read()
    if "/" not in content_range:
        raise ValueError(f"server does not support range requests: {url}")
    size = int(content_range.rsplit("/", 1)[1])

    def fetch(a: int, b: int) -> bytes:
        return get_bytes(final_url, timeout, headers={"Range": f"bytes={a}-{b}"})

    return fetch, size


class RemoteZip:
    def __init__(self, fetch: Fetch, size: int):
        self.fetch = fetch
        self.size = size
        self._entries: list[ZipEntry] | None = None

    @classmethod
    def from_url(cls, url: str) -> RemoteZip:
        return cls(*http_fetcher(url))

    def entries(self) -> list[ZipEntry]:
        if self._entries is None:
            self._entries = self._read_central_directory()
        return self._entries

    def _read_central_directory(self) -> list[ZipEntry]:
        tail_len = min(self.size, 65536 + 22 + 20 + 56)
        tail = self.fetch(self.size - tail_len, self.size - 1)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise ValueError("end of central directory not found")
        n = struct.unpack("<H", tail[i + 10 : i + 12])[0]
        cd_size, cd_off = struct.unpack("<II", tail[i + 12 : i + 20])
        if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n == 0xFFFF:
            j = tail.rfind(b"PK\x06\x07")
            z64_off = struct.unpack("<Q", tail[j + 8 : j + 16])[0]
            rec = self.fetch(z64_off, z64_off + 55)
            n, cd_size, cd_off = struct.unpack("<QQQ", rec[32:56])
        cd = self.fetch(cd_off, cd_off + cd_size - 1)
        out: list[ZipEntry] = []
        o = 0
        while o + 46 <= len(cd) and cd[o : o + 4] == b"PK\x01\x02":
            method = struct.unpack("<H", cd[o + 10 : o + 12])[0]
            csz, usz = struct.unpack("<II", cd[o + 20 : o + 28])
            fl, el, cl = struct.unpack("<HHH", cd[o + 28 : o + 34])
            lho = struct.unpack("<I", cd[o + 42 : o + 46])[0]
            name = cd[o + 46 : o + 46 + fl].decode("utf-8")
            extra = cd[o + 46 + fl : o + 46 + fl + el]
            k = 0
            while k + 4 <= len(extra):
                hid, hl = struct.unpack("<HH", extra[k : k + 4])
                if hid == 1:  # ZIP64 extended information
                    q, p = extra[k + 4 : k + 4 + hl], 0
                    if usz == 0xFFFFFFFF:
                        usz = struct.unpack("<Q", q[p : p + 8])[0]
                        p += 8
                    if csz == 0xFFFFFFFF:
                        csz = struct.unpack("<Q", q[p : p + 8])[0]
                        p += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", q[p : p + 8])[0]
                k += 4 + hl
            out.append(ZipEntry(name, method, csz, usz, lho))
            o += 46 + fl + el + cl
        return out

    def read(self, e: ZipEntry) -> bytes:
        h = self.fetch(e.header_offset, e.header_offset + 29)
        if h[:4] != b"PK\x03\x04":
            raise ValueError(f"bad local header for {e.name}")
        fl, el = struct.unpack("<HH", h[26:30])
        start = e.header_offset + 30 + fl + el
        data = self.fetch(start, start + e.compressed_size - 1) if e.compressed_size else b""
        if e.method == 0:
            out = data
        elif e.method == 8:
            out = zlib.decompress(data, -15)
        else:
            raise NotImplementedError(f"compression method {e.method}")
        if len(out) != e.size:
            raise ValueError(f"size mismatch for {e.name}: {len(out)} != {e.size}")
        return out
