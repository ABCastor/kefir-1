"""Fetch the pinned external fastText language identification model."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


ASSETS = Path(__file__).resolve().parent / "assets"
URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


def main() -> None:
    expected = json.loads((ASSETS / "manifest.json").read_text())["assets"]["lid.176.bin"]
    target = ASSETS / "lid.176.bin"
    temporary = ASSETS / "lid.176.bin.part"
    digest = hashlib.sha256()
    if target.is_file():
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() == expected:
            print(f"verified {target}")
            return
        raise SystemExit(f"existing model has wrong SHA-256: {target}")
    with urlopen(URL, timeout=60) as source, temporary.open("wb") as output:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != expected:
        temporary.unlink(missing_ok=True)
        raise SystemExit("downloaded fastText model has wrong SHA-256")
    temporary.replace(target)
    print(f"verified {target}")


if __name__ == "__main__":
    main()
