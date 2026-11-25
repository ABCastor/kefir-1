"""Small filesystem and hashing helpers used by the frozen corpus recipe."""

import hashlib
import json
import time
from pathlib import Path

from normalization import normalized
from kefir.data.src.storage import canonical_json_bytes


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def filehash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def paragraphs(text: str) -> set[str]:
    return {digest(normalized(p)) for p in text.split("\n") if len(p.split()) >= 40}


def stable_hash(value) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def upload_with_retry(attempt, name, *, attempts=6, backoff=(5, 15, 45, 90, 180), sleep=time.sleep):
    for index in range(attempts):
        try:
            attempt()
            return index
        except Exception as error:
            if index == attempts - 1:
                print(json.dumps({"event": "upload_failed", "shard": name,
                                  "attempts": attempts, "error": str(error)[:300]}), flush=True)
                raise
            delay = backoff[min(index, len(backoff) - 1)]
            print(json.dumps({"event": "upload_retry", "shard": name,
                              "attempt": index + 1, "sleep_seconds": delay,
                              "error": str(error)[:200]}), flush=True)
            sleep(delay)
