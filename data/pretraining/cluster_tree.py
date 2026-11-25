"""Derive the run 003 cluster tree from filtered corpus text and token marginals."""
from pathlib import Path
import argparse
import collections
import gzip
import hashlib
import json
import re

import numpy as np

ROOT = Path(__file__).resolve().parent
DIM = 1 << 18
TOKEN = re.compile(r"[a-z]{2,}")
STOP = set('the of and to in is for that with as on by are it this be from or an at was which can not have has their will its also more one these other than but they may used use all such into two when if been each between about your you we our some most who his her he she were had do does how any where there what been'.split())


def marginal_and_regions(root: Path):
    """cluster -> token share (from index files) and one region id per cluster."""
    tokens = collections.Counter()
    sample = {}
    for f in sorted((root / 'index').glob('region-*.npy')):
        t = np.load(f, mmap_mode='r')
        if not len(t):
            continue
        c = int(t['cluster'][0])
        tokens[c] += int(t['length'].sum())
        sample.setdefault(c, int(t['region'][0]))
    tot = sum(tokens.values())
    return {c: tokens[c] / tot for c in tokens}, sample, len(list((root / 'index').glob('*.npy')))


def region_jsonl(root: Path, idx: int) -> Path:
    worker = None
    for r in root.glob(f'worker-*/region-{idx:06d}.receipt.json'):
        worker = r.parent
    if worker is None:
        raise FileNotFoundError(f'region {idx}: worker receipt absent')
    local = worker / f'region-{idx:06d}.jsonl.gz'
    if not local.is_file():
        raise FileNotFoundError(local)
    return local


WORDS = {}

def centroid(path: Path, max_docs: int, cluster: int = -1):
    counts = np.zeros(DIM, dtype=np.float64)
    df = np.zeros(DIM, dtype=np.float64)
    n = 0
    vocab = collections.Counter()
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        for line in fh:
            text = json.loads(line)['text'].lower()
            ids = np.fromiter((int(hashlib.blake2b(w.encode(), digest_size=4).hexdigest(), 16) % DIM
                               for w in TOKEN.findall(text)), dtype=np.int64)
            if not len(ids):
                continue
            vocab.update(set(TOKEN.findall(text)))
            b = np.bincount(ids, minlength=DIM).astype(np.float64)
            counts += b / b.sum()
            df += b > 0
            n += 1
            if n >= max_docs:
                break
    WORDS[cluster] = vocab
    return counts / max(n, 1), df / max(n, 1), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs-per-cluster', type=int, default=4000)
    ap.add_argument('--root', type=Path, default=ROOT / 'corpus-13b-v2')
    a = ap.parse_args()
    root = a.root
    marginal, sample, nfiles = marginal_and_regions(root)
    clusters = sorted(marginal)
    print(json.dumps({'index_files': nfiles, 'clusters_seen': clusters}), flush=True)
    tf = {}
    df_all = np.zeros(DIM)
    ndocs = 0
    for c in clusters:
        path = region_jsonl(root, sample[c])
        tf[c], df_c, n = centroid(path, a.docs_per_cluster, c)
        df_all += df_c * n
        ndocs += n
        print(json.dumps({'cluster': c, 'region': sample[c], 'docs': n}), flush=True)
    idf = np.log((1 + ndocs) / (1 + df_all)) + 1.0
    M = np.stack([tf[c] * idf for c in clusters])
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    D = 1.0 - M @ M.T
    np.fill_diagonal(D, 0.0)
    nodes = {i: [clusters[i]] for i in range(len(clusters))}
    mass = {i: marginal[clusters[i]] for i in range(len(clusters))}
    dist = {(i, j): D[i, j] for i in range(len(clusters)) for j in range(i + 1, len(clusters))}
    merges = []
    nxt = len(clusters)
    while len(nodes) > 1:
        (i, j), d = min(dist.items(), key=lambda kv: kv[1])
        merged = nodes[i] + nodes[j]
        nodes[nxt] = merged
        merges.append((i, j, nxt, d))
        mass[nxt] = mass[i] + mass[j]
        for k in list(nodes):
            if k in (i, j, nxt):
                continue
            wi, wj, wk = mass[i], mass[j], mass[k]
            dik, djk, dij = dist[(min(k, i), max(k, i))], dist[(min(k, j), max(k, j))], d
            dist[(min(k, nxt), max(k, nxt))] = ((wi + wk) * dik + (wj + wk) * djk - wk * dij) / (wi + wj + wk)
        for k in list(dist):
            if i in k or j in k:
                del dist[k]
        del nodes[i], nodes[j]
        nxt += 1
    # cut the tree into 2, 4, 8, ... groups by undoing the last merges (largest distances first)
    def cut(k):
        leafsets = {i: [clusters[i]] for i in range(len(clusters))}
        for m in merges:
            leafsets[m[2]] = leafsets[m[0]] + leafsets[m[1]]
        current = [nxt - 1]
        order = sorted(merges, key=lambda m: -m[3])
        for m in order:
            if len(current) >= k:
                break
            if m[2] in current:
                current.remove(m[2])
                current += [m[0], m[1]]
        return [sorted(leafsets[n]) for n in current]
    levels = []
    k = 2
    while k < len(clusters):
        levels.append({'groups': cut(k)})
        k *= 2
    levels.append({'groups': [[c] for c in clusters]})
    out = {
        'leaves': clusters,
        'levels': levels,
        'marginal': {str(c): marginal[c] for c in clusters},
        'method': f'hashed TF-IDF centroid ({DIM} dims, {a.docs_per_cluster} docs/cluster), cosine distance, Ward (Lance-Williams) weighted by token marginal',
        'top_words': {
            str(c): [w for w, _ in WORDS[c].most_common(400) if w not in STOP][:12]
            for c in clusters
        },
        'index_files_used': nfiles,
        'distance': {
            f'{clusters[i]}-{clusters[j]}': round(float(D[i, j]), 4)
            for i in range(len(clusters)) for j in range(i + 1, len(clusters))
        },
    }
    (root / 'cluster_tree.json').write_text(json.dumps(out, indent=1))
    for lvl in levels[:3]:
        print('livello', len(lvl['groups']), 'gruppi:', lvl['groups'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
