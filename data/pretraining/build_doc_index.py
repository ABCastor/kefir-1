"""Index raw region metadata for random-access stratified packing.

Each local gzipped JSONL is streamed once. The compact NumPy table records document
location, length, cluster, and filter scores; text is discarded. A region is checked
against its receipt before its index is installed atomically.
"""
import argparse
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SCORES = ('st_not_prose', 'st_poor_english', 'st_reasoning_dense', 'st_reasoning_none', 'st_symbolic',
          'v4_bad_score', 'v4_english_mean', 'v4_english_min')
DTYPE = np.dtype([('region', 'u2'), ('doc', 'u4'), ('offset', 'u4'), ('length', 'u4'), ('cluster', 'u1'),
                  ('source_tokens', 'u4'), ('axes', 'u1')] + [(s, 'f2') for s in SCORES])
AXES = {'soft': 1, 'code': 2, 'html': 4, 'math': 8, 'script': 16}


def region_map(root: Path):
    """region index -> worker dir name, from the local worker contracts/receipts."""
    mapping = {}
    for receipt in root.glob('worker-*/region-*.receipt.json'):
        idx = int(receipt.stem.split('-')[1].split('.')[0])
        mapping[idx] = receipt.parent.name
    return mapping


def load_region(idx: int, worker: str, root: Path) -> Path:
    stem = f'region-{idx:06d}.jsonl.gz'
    path = root / worker / stem
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def index_region(path: Path, idx: int, expected_docs: int | None) -> np.ndarray:
    rows = []
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        for doc, line in enumerate(fh):
            r = json.loads(line)
            axes = 0
            for a in r.get('v4_top_axes') or []:
                axes |= AXES.get(a, 0)
            rows.append((idx, doc, r['token_offset_in_shard'], r['target_token_count'] + 1, r['cluster_id'],
                         r.get('source_token_count', 0), axes,
                         *[float(r.get(s, float('nan'))) for s in SCORES]))
    table = np.array(rows, dtype=DTYPE)
    if expected_docs is not None and len(table) != expected_docs:
        raise RuntimeError(f'region {idx}: {len(table)} docs, receipt says {expected_docs}')
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT / 'corpus-13b-v2')
    ap.add_argument('--regions', type=int, default=2100)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--start', type=int, default=0)
    a = ap.parse_args()
    root = a.root
    out = root / 'index'
    out.mkdir(parents=True, exist_ok=True)
    workers = region_map(root)
    started = time.time()
    done = 0
    for idx in range(a.start + a.offset, a.regions, a.stride):
        target = out / f'region-{idx:06d}.npy'
        if target.exists():
            continue
        worker = workers[idx]
        receipt = json.loads((root / worker / f'region-{idx:06d}.receipt.json').read_text())
        expected = receipt['counts'].get('accepted_documents')
        path = load_region(idx, worker, root)
        table = index_region(path, idx, expected)
        tmp = target.with_suffix('.tmp.npy')
        np.save(tmp, table)
        os.replace(tmp, target)
        done += 1
        if done % 10 == 0:
            el = time.time() - started
            print(json.dumps({'region': idx, 'docs': int(len(table)), 'done': done, 's_per_region': round(el / done, 1)}), flush=True)
    print(json.dumps({'event': 'done', 'regions': a.regions}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
