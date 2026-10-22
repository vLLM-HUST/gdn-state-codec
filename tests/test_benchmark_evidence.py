import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_evidence", ROOT / "tools/benchmark_evidence.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "value", ["../outside", "/absolute", "a/../outside", "", "./a"]
)
def test_unsafe_source_paths_are_rejected(value):
    with pytest.raises(ValueError, match="Unsafe"):
        MODULE.relative_path(value)


def test_empty_original_preserved_and_corruption_rejected():
    record = {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    assert MODULE.verify_bytes(b"", record) == b""
    with pytest.raises(ValueError, match="mismatch"):
        MODULE.verify_bytes(b"{}", record)


def test_wrong_hash_rejected_even_when_length_matches():
    record = {"bytes": 1, "sha256": hashlib.sha256(b"a").hexdigest()}
    with pytest.raises(ValueError, match="mismatch"):
        MODULE.verify_bytes(b"b", record)


def test_refuse_in_repository_cache(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        MODULE.fetch(ROOT / "local-evidence")


def test_corrupt_cache_is_not_overwritten(tmp_path, monkeypatch):
    original = b"original"
    path = "evidence/result.json"
    record = {
        "path": path,
        "bytes": len(original),
        "sha256": hashlib.sha256(original).hexdigest(),
    }
    monkeypatch.setattr(
        MODULE,
        "load_manifest",
        lambda: ("https://unused/", {"files": [record], "source_commit": "frozen"}),
    )
    target = tmp_path / path
    target.parent.mkdir()
    target.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="mismatch"):
        MODULE.fetch(tmp_path)
    assert target.read_bytes() == b"corrupted"


def test_no_embedded_data_copies():
    evidence = ROOT / "evidence"
    assert all(
        path.relative_to(evidence).as_posix()
        == "all-head-negative-m2/vectorized-prototype.patch"
        for path in evidence.rglob("*")
        if path.is_file()
    )
