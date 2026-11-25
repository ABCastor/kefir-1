"""New outputs appear whole or not at all: the one rule for every command that writes a result."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil


@contextmanager
def staged_output(final: str | Path):
    """Yield a temporary sibling path that becomes the new ``final`` only on success.

    An existing ``final`` is refused before any work; a failure removes the partial output,
    so a retry needs no cleanup.
    """
    final = Path(final)
    if final.exists():
        raise FileExistsError(f"output already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name(f".{final.name}.{os.getpid()}.partial")
    try:
        yield temporary
        os.replace(temporary, final)
    finally:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        elif temporary.exists():
            temporary.unlink()


__all__ = ["staged_output"]
