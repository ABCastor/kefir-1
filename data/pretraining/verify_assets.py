"""Verify the scientific inputs bundled with the run 003 corpus recipe."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"


def _verify(root: Path, expected: dict[str, str]) -> int:
    for name, digest in expected.items():
        path = root / name
        if not path.is_file():
            if name == "students.pkl":
                from student_gate import require_student_asset
                require_student_asset(path)
            raise RuntimeError(f"missing corpus input: {path}")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            raise RuntimeError(f"corpus input hash mismatch: {path}")
    return len(expected)


def verify_sources(root: Path = ROOT) -> int:
    manifest = json.loads((root / "assets/manifest.json").read_text())
    return _verify(root, manifest["curated_source_sha256"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-only", action="store_true")
    args = parser.parse_args()
    sources = verify_sources()
    assets = 0 if args.sources_only else _verify(
        ASSETS, json.loads((ASSETS / "manifest.json").read_text())["assets"])
    print(f"verified {sources} corpus sources and {assets} assets")


if __name__ == "__main__":
    main()
