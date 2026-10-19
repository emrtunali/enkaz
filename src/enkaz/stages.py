"""Resumable pipeline stages.

Each stage writes a small JSON marker when it finishes. Re-running a notebook cell skips any
stage whose marker exists with the same ``version``; bump the version to force a rerun after
changing the stage's logic.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable


class StageRunner:
    def __init__(self, marker_dir: Path, log: Callable[[str], None] = print):
        self.marker_dir = Path(marker_dir)
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        self.log = log

    def marker(self, name: str) -> Path:
        return self.marker_dir / f"{name}.json"

    def done(self, name: str, version: str = "1") -> bool:
        m = self.marker(name)
        if not m.exists():
            return False
        try:
            return json.loads(m.read_text()).get("version") == version
        except json.JSONDecodeError:
            return False

    def result(self, name: str) -> Any:
        return json.loads(self.marker(name).read_text())["result"]

    def run(
        self, name: str, fn: Callable[[], Any], *, version: str = "1", force: bool = False
    ) -> Any:
        if not force and self.done(name, version):
            self.log(f"[skip] {name} (already done, v{version})")
            return self.result(name)
        self.log(f"[run ] {name} (v{version})")
        t0 = time.time()
        result = fn()
        elapsed = time.time() - t0
        payload = {"version": version, "elapsed_s": round(elapsed, 1), "result": result}
        tmp = self.marker(name).with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=str))
        tmp.replace(self.marker(name))
        self.log(f"[done] {name} in {elapsed / 60:.1f} min")
        return result
