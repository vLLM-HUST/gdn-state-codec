"""Verify immutable benchmark originals in an external cache; never execute them."""

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "vLLM-HUST/vllm-hust-benchmark"


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("Unsafe evidence path")
    return path


def verify_bytes(raw, record):
    if (
        len(raw) != record["bytes"]
        or hashlib.sha256(raw).hexdigest() != record["sha256"]
    ):
        raise ValueError("Evidence length/SHA256 mismatch")
    return raw


def download(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return response.read()
        except urllib.error.URLError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def load_manifest():
    source = json.loads((ROOT / "EVIDENCE_SOURCES.json").read_text())
    if source["repository"] != REPOSITORY or not re.fullmatch(
        r"[a-f0-9]{40}", source["commit"]
    ):
        raise ValueError("Evidence must use a fixed benchmark commit")
    base = f"https://raw.githubusercontent.com/{REPOSITORY}/{source['commit']}/"
    relative_path(source["manifest"]["path"])
    raw = verify_bytes(download(base + source["manifest"]["path"]), source["manifest"])
    manifest = json.loads(raw)
    if (
        manifest["source_repository"] != source["source_repository"]
        or manifest["source_commit"] != source["source_commit"]
    ):
        raise ValueError("Source identity mismatch")
    paths = [r["path"] for r in manifest["files"]]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate source path")
    for path in paths:
        relative_path(path)
    relative_path(source["snapshot"])
    return base + source["snapshot"] + "/", manifest


def fetch(destination, selected=None):
    destination = destination.resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("Evidence cache must be outside the MOD repository")
    base, manifest = load_manifest()
    records = {r["path"]: r for r in manifest["files"]}
    paths = list(records) if selected is None else selected
    if not set(paths) <= records.keys():
        raise ValueError("Requested path is not in the frozen source manifest")

    def one(path):
        target = destination / str(relative_path(path))
        if not target.resolve().is_relative_to(destination):
            raise ValueError("Evidence cache path escapes destination")
        raw = target.read_bytes() if target.exists() else download(base + path)
        verify_bytes(raw, records[path])
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        return path

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        verified = list(pool.map(one, paths))
    return {
        "verified_files": len(verified),
        "source_commit": manifest["source_commit"],
        "destination": str(destination),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--path", action="append")
    args = parser.parse_args()
    print(json.dumps(fetch(args.destination, None if args.all else args.path)))


if __name__ == "__main__":
    main()
