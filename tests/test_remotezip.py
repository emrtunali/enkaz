import io
import zipfile

import pytest

from enkaz.data.remotezip import RemoteZip


def _make_zip(members: dict[str, bytes], compression: int) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compression) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _local(blob: bytes) -> RemoteZip:
    calls = []

    def fetch(a: int, b: int) -> bytes:
        assert 0 <= a <= b < len(blob)
        calls.append((a, b))
        return blob[a : b + 1]

    rz = RemoteZip(fetch, len(blob))
    rz.calls = calls  # type: ignore[attr-defined]
    return rz


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_reads_members(compression):
    members = {
        "target/turkey-earthquake_00000001_building_damage.tif": bytes(range(256)) * 40,
        "target/beirut-explosion_00000001_building_damage.tif": b"x" * 1000,
        "target/": b"",
    }
    rz = _local(_make_zip(members, compression))
    entries = {e.name: e for e in rz.entries()}
    assert set(entries) == set(members)
    for name, data in members.items():
        assert rz.read(entries[name]) == data


def test_only_requested_bytes_are_fetched():
    big = {f"m{i}.bin": bytes([i]) * 50_000 for i in range(10)}
    blob = _make_zip(big, zipfile.ZIP_STORED)
    rz = _local(blob)
    e = next(e for e in rz.entries() if e.name == "m3.bin")
    assert rz.read(e) == bytes([3]) * 50_000
    fetched = sum(b - a + 1 for a, b in rz.calls)
    assert fetched < len(blob) / 2
