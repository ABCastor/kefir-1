"""Frozen Kefir 1 chat benchmark suite (kefir-bench/1): build, run and score.

``authored``, ``io`` and ``scoring`` are byte-identical to the snapshot used during the
Kefir 1 SFT runs; ``public_data`` differs only in the location of its data directory.
The selected public cases are not redistributed: ``kefir bench build`` reconstructs them
from the pinned upstream files listed in ``data/source-manifest.json``.
"""
