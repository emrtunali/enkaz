"""HTTP helpers with retries and resumable, atomic downloads (stdlib only)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

USER_AGENT = "enkaz/0.1 (research; +https://github.com/emrtunali)"


def open_url(req: urllib.request.Request | str, timeout: float):
    if isinstance(req, str):
        req = urllib.request.Request(req)
    req.add_header("User-Agent", USER_AGENT)
    return urllib.request.urlopen(req, timeout=timeout)


def with_retries(fn, retries: int = 4, backoff: float = 2.0):
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 429:
                raise
            last = e
            time.sleep(backoff * (2**attempt))
    assert last is not None
    raise last


def get_bytes(url: str, timeout: float = 60, headers: dict[str, str] | None = None) -> bytes:
    def go():
        req = urllib.request.Request(url, headers=headers or {})
        with open_url(req, timeout) as r:
            return r.read()

    return with_retries(go)


def get_json(url: str, timeout: float = 60) -> Any:
    return json.loads(get_bytes(url, timeout))


def download(url: str, dest: Path, timeout: float = 120, chunk: int = 1 << 20) -> Path:
    """Download ``url`` to ``dest``; resume a partial ``.part`` file; skip if ``dest`` exists."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    def go():
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        with open_url(req, timeout) as r:
            mode = "ab" if have and r.status == 206 else "wb"
            with open(part, mode) as f:
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)

    with_retries(go)
    part.replace(dest)
    return dest
