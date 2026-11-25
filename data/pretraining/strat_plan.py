#!/usr/bin/env python3
"""Stratified packing plan: documents -> sequences -> blocks -> stream order.

This module never touches a .bin: it works only on the global per-document index written by
build_doc_index.py (region, doc, offset, length, cluster, scores) and produces a *plan* that
pack_stratified.py then materialises into the trainer's shards.

Three levels, exactly the recipe used for run 003 (method and references in data/pretraining/README.md):

  1. document -> sequence.  Best-fit-decreasing packing (Ding 2024, arXiv 2404.10830) of documents
     of the *same cluster* into 2049-token records.  Every document already ends with the builder's
     EOD (the index `length` counts it), so a record is a concatenation of whole documents with EOD
     between them and EOD padding at the end.  Because every region of this corpus is mono-cluster
     (verified: receipt counts have a single cluster_NN key), packing region by region is what makes
     records cluster-pure, which in turn makes the per-block cluster quotas exact.
     A document longer than 2048 tokens is cut into fragments; the full-capacity fragments become
     records of their own and the tail fragment re-enters the best-fit pool.

  2. sequence -> block of 64.  Each block reproduces the global cluster marginal with exact
     integer largest-remainder / largest-deficit allocation (the arithmetic of the trainer's
     TokenDeficitScheduler; the scheduler object itself is driven by pack_stratified.py so the
     manifest's `scheduler` block stays truthful).  Inside a cluster the slots walk the
     st_reasoning_dense quartile x mean-document-length quartile strata round-robin, so the
     sequences of one cluster inside one block come from different score and length strata.

  3. block -> stream.  A single global permutation with a fixed seed.  We have the global index, so
     this is a true global shuffle rather than the sliding-window / just-in-time shuffle Dai 2026
     (arXiv 2605.30334) measures as strictly worse.  No curriculum.

Cluster weights are the corpus marginal, computed from the index, never re-optimised: ClimbMix is
already the output of a mixture search (Diao 2025, arXiv 2504.13161).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
from kefir.data import config                                                          # noqa: E402
config.EOD_TOKEN_ID = 7992   # the trainer's classes insert config.EOD_TOKEN_ID; its default is GPT-2's
from kefir.data.src.split import is_validation                                         # noqa: E402

EOD = 7992
CONTEXT = 2048
STORED = CONTEXT + 1          # 2049: one record, capacity of one bin
SEQ_PER_BLOCK = 64
N_SCORE_STRATA = 4
N_LENGTH_STRATA = 4
N_STRATA = N_SCORE_STRATA * N_LENGTH_STRATA
PLAN_VERSION = 3              # bump whenever the plan arithmetic changes; part of the cache key
SCORE_FIELD = 'st_reasoning_dense'


# --------------------------------------------------------------------------------------------
# corpus layout
# --------------------------------------------------------------------------------------------
def region_receipts(root: Path) -> dict[int, dict]:
    """region index -> receipt (source_region identity, counts, hashes).

    The worker receipt is the canonical one: index/ and bins/ hold the *worker* artefacts, so its
    target_tokens and tokens_sha256 are the ones that describe the local files.  merged/ describes
    the post-cross-region-dedup artefact and its target_tokens is smaller (region 48: 44,951,335
    against the worker 46,151,427).  Both carry the same `source_region`, which is the only field
    the validation identity reads, and that agreement is checked here.
    """
    out: dict[int, dict] = {}
    for p in root.glob('worker-*/region-*.receipt.json'):
        idx = int(re.search(r'region-(\d+)\.receipt\.json', p.name).group(1))
        out[idx] = json.loads(p.read_text())
    for p in (root / 'merged').glob('region-*.receipt.json'):
        idx = int(re.search(r'region-(\d+)\.receipt\.json', p.name).group(1))
        merged = json.loads(p.read_text())
        if idx in out and merged.get('source_region') != out[idx].get('source_region'):
            raise RuntimeError(f'region {idx}: worker and merged receipts disagree on source_region')
        out.setdefault(idx, merged)
    return out


def merge_dedup_stats(root: Path) -> dict:
    """How many documents the merge pass would drop as cross-region duplicates.

    Measured, not assumed: over the 130 regions that already have a merged receipt,
    1,459 of 9,197,170 documents (0.016%) and 0.022% of the tokens are cross-region duplicates.
    Region 48 alone accounts for 1,019 of them, so the rate is lumpy.
    """
    dropped = documents = worker_tokens = merged_tokens = regions = 0
    for p in (root / 'merged').glob('region-*.receipt.json'):
        idx = int(re.search(r'region-(\d+)\.receipt\.json', p.name).group(1))
        merged = json.loads(p.read_text())
        worker = next((json.loads(w.read_text())
                       for w in root.glob(f'worker-*/region-{idx:06d}.receipt.json')), None)
        if worker is None:
            continue
        regions += 1
        dropped += int(merged.get('merge', {}).get('cross_region_duplicates_dropped', 0))
        documents += int(worker['counts'].get('accepted_documents', 0))
        worker_tokens += int(worker['target_tokens'])
        merged_tokens += int(merged['target_tokens'])
    return {'regions_compared': regions, 'documents': documents,
            'cross_region_duplicates': dropped,
            'document_fraction': round(dropped / documents, 6) if documents else None,
            'token_fraction': round((worker_tokens - merged_tokens) / worker_tokens, 6)
            if worker_tokens else None}


def region_bins(root: Path) -> dict[int, Path]:
    """region index -> a local .bin, preferring the flat bins/ dir over the worker dirs."""
    out: dict[int, Path] = {}
    for p in root.glob('worker-*/region-*.bin'):
        out[int(re.search(r'region-(\d+)\.bin', p.name).group(1))] = p
    for p in (root / 'bins').glob('region-*.bin'):
        out[int(re.search(r'region-(\d+)\.bin', p.name).group(1))] = p
    return out


def region_indexes(root: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in (root / 'index').glob('region-*.npy'):
        out[int(re.search(r'region-(\d+)\.npy', p.name).group(1))] = p
    return out


def bin_is_complete(path: Path, receipt: dict) -> bool:
    """A region .bin must hold exactly the tokens its receipt promises.

    fetch_bins.py only publishes a .bin under bins/ after checking tokens_sha256, but the copies
    still sitting in the worker dirs can be half-written, and a truncated file would be read through
    a memmap without any error until the very last document of the region.
    """
    try:
        return path.stat().st_size == 2 * int(receipt['target_tokens'])
    except OSError:
        return False


def available_regions(root: Path, *, need_bin: bool = True) -> list[int]:
    """Regions this machine can actually pack: index + receipt (+ a complete token .bin)."""
    idx, bins, receipts = region_indexes(root), region_bins(root), region_receipts(root)
    keys = set(idx) & set(receipts)
    if need_bin:
        keys = {r for r in keys & set(bins) if bin_is_complete(bins[r], receipts[r])}
    return sorted(keys)


# --------------------------------------------------------------------------------------------
# validation split
# --------------------------------------------------------------------------------------------
def _component(value: str) -> bytes:
    encoded = value.encode('utf-8')
    return len(encoded).to_bytes(8, 'big') + encoded


def validation_mask(receipt: dict, offsets: np.ndarray, *, seed: str, probability: float) -> np.ndarray:
    """is_validation() for a whole region, with the identity the *index* can carry.

    dataset/src/split.py hashes (SPLIT_HASH_VERSION, seed, revision, filename, record_start).  The
    per-document index has no record_start and no source filename: those live only in the 207 GB of
    jsonl.gz on the hub, and re-streaming them to recover 0.1% of the documents is not affordable.
    The identity used here is the region's own (revision, source file) from its merge receipt plus
    the document's *token* offset inside the region, which is permanent, unique and already in the
    index.  Everything else is unchanged: same function, same SPLIT_HASH_VERSION, same seed, same
    0.001 probability, order-independent, identical after a resume or on another machine.

    Consequence, recorded in the manifest as validation.rule = 'index_identity_v1': this selects a
    *different* 0.1% of documents than pack_shards.py would.  Exact parity would need three more
    columns (source_revision, source_file, record_start) in build_doc_index.py.
    """
    if probability <= 0.0:
        return np.zeros(len(offsets), dtype=bool)
    revision, filename = receipt['source_region'].split(':')[0], receipt['source_region'].split(':')[1]
    prefix = hashlib.sha256()
    prefix.update(_component(config.SPLIT_HASH_VERSION))
    prefix.update(_component(seed))
    prefix.update(_component(revision))
    prefix.update(_component(filename))
    threshold = int(probability * (1 << 64))
    out = np.zeros(len(offsets), dtype=bool)
    for i, off in enumerate(offsets.tolist()):
        h = prefix.copy()
        h.update(int(off).to_bytes(8, 'big', signed=False))
        if int.from_bytes(h.digest()[:8], 'big') < threshold:
            out[i] = True
    return out


def validation_identity(receipt: dict, token_offset: int) -> dict:
    """The (revision, filename, record_start) triple validation_mask() feeds to is_validation."""
    revision, filename = receipt['source_region'].split(':')[0], receipt['source_region'].split(':')[1]
    return {'revision': revision, 'filename': filename, 'record_start': int(token_offset)}


# --------------------------------------------------------------------------------------------
# level 1: documents -> sequences (best-fit decreasing, Ding 2024)
# --------------------------------------------------------------------------------------------
def best_fit_decreasing(sizes: np.ndarray, capacity: int = STORED) -> tuple[np.ndarray, np.ndarray]:
    """Best-Fit-Decreasing bin packing.  Returns (items_in_bin_order, bin_ptr).

    Items are placed largest first into the bin whose residual capacity is the smallest one that
    still fits (Ding 2024's best-fit packing; their segment tree is replaced by a 2050-wide count
    vector, which is faster here because the capacity is tiny and numpy does the scan).
    """
    n = len(sizes)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64)
    order = np.argsort(-sizes.astype(np.int64), kind='stable')
    resid_count = np.zeros(capacity + 1, dtype=np.int64)
    by_resid: dict[int, list[int]] = defaultdict(list)
    bin_items: list[list[int]] = []
    for i in order.tolist():
        s = int(sizes[i])
        if s <= 0 or s > capacity:
            raise ValueError(f'item size {s} outside 1..{capacity}')
        window = resid_count[s:] > 0
        if window.any():
            r = s + int(window.argmax())
            b = by_resid[r].pop()
            resid_count[r] -= 1
            nr = r - s
            resid_count[nr] += 1
            by_resid[nr].append(b)
            bin_items[b].append(i)
        else:
            b = len(bin_items)
            bin_items.append([i])
            nr = capacity - s
            resid_count[nr] += 1
            by_resid[nr].append(b)
    ptr = np.zeros(len(bin_items) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(x) for x in bin_items])
    flat = np.fromiter((i for items in bin_items for i in items), dtype=np.int64, count=int(ptr[-1]))
    return flat, ptr


def _fragments(length: int, capacity: int = STORED) -> list[tuple[int, int, bool]]:
    """(start, size, terminal) pieces of one document whose stored size (incl. EOD) is `length`."""
    if length <= capacity:
        return [(0, length, True)]
    nfrag = math.ceil(length / capacity)
    out = [(j * capacity, capacity, False) for j in range(nfrag - 1)]
    tail = length - capacity * (nfrag - 1)
    out.append((capacity * (nfrag - 1), tail, True))
    return out


def plan_region_split(table: np.ndarray, keep: np.ndarray) -> dict[str, np.ndarray]:
    """Pack one split of one region into cluster-pure sequences.

    Almost every region holds a single NVIDIA cluster, but not all of them: 3 of the 1,373 regions
    indexed on 2026-09-22 straddle a cluster boundary (770 -> {3,4}, 1230 -> {15,16},
    1312 -> {17,18}).  Records must stay cluster-pure for the per-block quotas to be exact, so the
    packing runs once per cluster present and the cluster is carried per sequence, not per region.
    """
    clusters = np.unique(np.asarray(table['cluster'])[keep]) if keep.any() else np.zeros(0, np.uint8)
    if len(clusters) <= 1:
        out = _plan_one_cluster(table, keep)
        out['seq_cluster'] = np.full(len(out['seq_ptr']) - 1,
                                     int(clusters[0]) if len(clusters) else 0, dtype=np.uint8)
        return out
    parts = [(_plan_one_cluster(table, keep & (np.asarray(table['cluster']) == c)), int(c))
             for c in clusters.tolist()]
    merged: dict[str, np.ndarray] = {}
    seq_off = ent_off = 0
    doc_offsets, running = {}, 0
    for part, c in parts:
        doc_offsets[c] = running
        running += len(part['doc_index'])
    ptrs, seq_cl = [np.zeros(1, dtype=np.uint64)], []
    for part, c in parts:
        nseq = len(part['seq_ptr']) - 1
        ptrs.append((part['seq_ptr'][1:].astype(np.int64) + ent_off).astype(np.uint64))
        seq_cl.append(np.full(nseq, c, dtype=np.uint8))
        for k in ('ent_doc', 'ent_off', 'ent_len', 'ent_term', 'doc_index', 'seq_src', 'seq_tokens',
                  'seq_meanlen', 'seq_score'):
            merged.setdefault(k, []).append(part[k])
        merged.setdefault('frag_seq', []).append(
            (part['frag_seq'].astype(np.int64) + seq_off).astype(np.uint32))
        # frag_parent is a document index local to this cluster's part, like ent_doc: it must be
        # shifted by the same doc_index offset or two documents of different clusters would share
        # one parent key and be treated as fragments of each other
        merged.setdefault('frag_parent', []).append(
            (part['frag_parent'].astype(np.int64) + doc_offsets[c]).astype(np.uint32))
        seq_off += nseq
        ent_off += len(part['ent_len'])
    out = {k: np.concatenate(v) for k, v in merged.items()}
    out['seq_ptr'] = np.concatenate(ptrs)
    out['seq_cluster'] = np.concatenate(seq_cl)
    # doc_index is per-cluster; ent_doc indexes into it, so shift it by each part's offset
    out['ent_doc'] = np.concatenate(
        [part['ent_doc'].astype(np.int64) + doc_offsets[c] for part, c in parts]).astype(np.uint32)
    return out


def _plan_one_cluster(table: np.ndarray, keep: np.ndarray) -> dict[str, np.ndarray]:
    """Best-fit-decreasing packing of the documents selected by `keep` (all of one cluster)."""
    doc_idx = np.flatnonzero(keep).astype(np.int64)
    length = np.asarray(table['length'], dtype=np.int64)[doc_idx]
    offset = np.asarray(table['offset'], dtype=np.int64)[doc_idx]
    score = np.asarray(table[SCORE_FIELD], dtype=np.float32)[doc_idx]
    score = np.nan_to_num(score, nan=0.0)

    # ---- fragments ---------------------------------------------------------------------------
    solo_rows: list[tuple[int, int, int, int]] = []      # (doc_local, abs_offset, size, terminal)
    pool_doc: list[int] = []
    pool_off: list[int] = []
    pool_size: list[int] = []
    multi = np.zeros(len(doc_idx), dtype=bool)
    long_mask = length > STORED
    for k in np.flatnonzero(long_mask).tolist():
        multi[k] = True
        for start, size, terminal in _fragments(int(length[k])):
            if terminal:
                pool_doc.append(k); pool_off.append(int(offset[k]) + start); pool_size.append(size)
            else:
                solo_rows.append((k, int(offset[k]) + start, size, 0))
    short = np.flatnonzero(~long_mask)
    pool_doc.extend(short.tolist())
    pool_off.extend(offset[short].tolist())
    pool_size.extend(length[short].tolist())

    pool_doc_a = np.asarray(pool_doc, dtype=np.int64)
    pool_off_a = np.asarray(pool_off, dtype=np.int64)
    pool_size_a = np.asarray(pool_size, dtype=np.int64)
    flat, ptr = best_fit_decreasing(pool_size_a)

    # ---- sequences ---------------------------------------------------------------------------
    n_solo = len(solo_rows)
    n_bins = len(ptr) - 1
    nseq = n_solo + n_bins
    ent_total = n_solo + len(flat)
    seq_ptr = np.zeros(nseq + 1, dtype=np.int64)
    ent_doc = np.zeros(ent_total, dtype=np.uint32)
    ent_off = np.zeros(ent_total, dtype=np.uint32)
    ent_len = np.zeros(ent_total, dtype=np.uint16)
    ent_term = np.zeros(ent_total, dtype=np.uint8)
    for j, (d, o, s, t) in enumerate(solo_rows):
        seq_ptr[j + 1] = j + 1
        ent_doc[j] = d; ent_off[j] = o; ent_len[j] = s; ent_term[j] = t
    base = n_solo
    seq_ptr[n_solo + 1:] = base + ptr[1:]
    if len(flat):
        ent_doc[base:] = pool_doc_a[flat]
        ent_off[base:] = pool_off_a[flat]
        ent_len[base:] = pool_size_a[flat]
        ent_term[base:] = 1

    # per-sequence statistics, vectorised over the ragged structure
    seq_of_ent = np.repeat(np.arange(nseq, dtype=np.int64), np.diff(seq_ptr))
    lens = ent_len.astype(np.int64)
    src = lens - ent_term.astype(np.int64)                  # the trailing EOD is not a source token
    seq_src = np.bincount(seq_of_ent, weights=src, minlength=nseq).astype(np.int64)
    seq_tokens = np.bincount(seq_of_ent, weights=lens, minlength=nseq).astype(np.int64)
    if nseq and seq_tokens.max() > STORED:
        raise RuntimeError('a planned sequence exceeds the record capacity')
    counts = np.diff(seq_ptr)
    doc_len_of_ent = length[ent_doc.astype(np.int64)] if ent_total else np.zeros(0, dtype=np.int64)
    seq_meanlen = np.zeros(nseq, dtype=np.float64)
    seq_score = np.zeros(nseq, dtype=np.float64)
    if ent_total:
        seq_meanlen = np.bincount(seq_of_ent, weights=doc_len_of_ent, minlength=nseq) / np.maximum(counts, 1)
        seq_score = np.bincount(seq_of_ent, weights=score[ent_doc.astype(np.int64)], minlength=nseq) / np.maximum(counts, 1)

    # (sequence, parent document) pairs for every sequence holding a fragment of a split document
    frag_ent = np.flatnonzero(multi[ent_doc.astype(np.int64)]) if ent_total else np.zeros(0, dtype=np.int64)
    frag_seq = seq_of_ent[frag_ent].astype(np.uint32)
    frag_parent = ent_doc[frag_ent].astype(np.uint32)

    return {
        'seq_ptr': seq_ptr.astype(np.uint64),
        'ent_doc': ent_doc, 'ent_off': ent_off, 'ent_len': ent_len, 'ent_term': ent_term,
        'doc_index': doc_idx.astype(np.uint32),          # local sequence doc -> row in the region index
        'seq_src': seq_src.astype(np.uint32),
        'seq_tokens': seq_tokens.astype(np.uint32),
        'seq_meanlen': seq_meanlen.astype(np.float32),
        'seq_score': seq_score.astype(np.float32),
        'frag_seq': frag_seq, 'frag_parent': frag_parent,
    }


def _save_split(dst: Path, arrays: dict[str, np.ndarray]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        tmp = dst / f'{name}.npy.swap'
        with open(tmp, 'wb') as fh:      # np.save would append a second .npy to the swap name
            np.save(fh, arr)
        os.replace(tmp, dst / f'{name}.npy')


def build_region_plan(root: Path, plan_dir: Path, region: int, *,
                      validation_seed: str = config.SELECTION_SEED,
                      validation_probability: float = config.VALIDATION_PROBABILITY,
                      force: bool = False) -> dict:
    """Level 1 for one region, cached under plan_dir/region-NNNNNN/.

    Nothing here depends on the packing seed: best-fit packing is deterministic and the split is the
    trainer's own is_validation with config.SELECTION_SEED, so changing --seed reshuffles the blocks
    without moving a single document between train and validation.
    """
    dst = plan_dir / f'region-{region:06d}'
    meta_path = dst / 'meta.json'
    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if meta.get('plan_version') == PLAN_VERSION and meta.get('validation_seed') == validation_seed:
            return meta
    table = np.load(root / 'index' / f'region-{region:06d}.npy')
    receipts = region_receipts(root)
    receipt = receipts[region]
    if len(table) == 0:
        meta = {'region': region, 'plan_version': PLAN_VERSION, 'validation_seed': validation_seed,
                'cluster': -1, 'documents': 0, 'empty': True,
                'splits': {'train': {'sequences': 0, 'source_tokens': 0, 'padding_tokens': 0, 'documents': 0},
                           'validation': {'sequences': 0, 'source_tokens': 0, 'padding_tokens': 0, 'documents': 0}}}
        dst.mkdir(parents=True, exist_ok=True)
        (dst / 'meta.json.swap').write_text(json.dumps(meta))
        os.replace(dst / 'meta.json.swap', meta_path)
        return meta
    clusters = np.unique(np.asarray(table['cluster']))
    bad = sorted(int(c) for c in clusters if int(c) not in config.ACCEPTED_CLUSTER_IDS)
    if bad:
        raise RuntimeError(f'region {region} has unaccepted clusters {bad}')
    cluster = int(clusters[0]) if len(clusters) == 1 else -1
    offsets = np.asarray(table['offset'], dtype=np.int64)
    val = validation_mask(receipt, offsets, seed=validation_seed, probability=validation_probability)
    meta = {'region': region, 'plan_version': PLAN_VERSION, 'validation_seed': validation_seed,
            'validation_probability': validation_probability, 'cluster': cluster,
            'clusters': [int(c) for c in clusters],
            'documents': int(len(table)), 'empty': False,
            'source_region': receipt['source_region'], 'splits': {}}
    for split, keep in (('train', ~val), ('validation', val)):
        arrays = plan_region_split(table, keep)
        _save_split(dst / split, arrays)
        nseq = len(arrays['seq_ptr']) - 1
        meta['splits'][split] = {
            'sequences': int(nseq),
            'documents': int(keep.sum()),
            'source_tokens': int(arrays['seq_src'].astype(np.int64).sum()),
            'stored_tokens': int(nseq) * STORED,
            'padding_tokens': int(nseq) * STORED - int(arrays['seq_tokens'].astype(np.int64).sum()),
        }
    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'meta.json.swap').write_text(json.dumps(meta))
    os.replace(dst / 'meta.json.swap', meta_path)
    return meta


# --------------------------------------------------------------------------------------------
# levels 2 and 3: sequences -> blocks -> stream
# --------------------------------------------------------------------------------------------
class SequenceTable:
    """Per-sequence metadata of one split, concatenated over the planned regions."""

    def __init__(self, regions: list[int], parts: list[dict[str, np.ndarray]]):
        self.regions = regions
        sizes = [len(p['seq_score']) for p in parts]
        self.bounds = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        total = int(self.bounds[-1])
        self.region = np.zeros(total, dtype=np.uint16)
        self.lseq = np.zeros(total, dtype=np.uint32)
        self.cluster = np.zeros(total, dtype=np.uint8)
        self.score = np.zeros(total, dtype=np.float32)
        self.meanlen = np.zeros(total, dtype=np.float32)
        self.src = np.zeros(total, dtype=np.uint32)
        self.parent = np.full(total, -1, dtype=np.int64)
        pair_seq: list[np.ndarray] = []
        pair_key: list[np.ndarray] = []
        for k, (r, part) in enumerate(zip(regions, parts)):
            a, b = int(self.bounds[k]), int(self.bounds[k + 1])
            self.region[a:b] = r
            self.lseq[a:b] = np.arange(b - a, dtype=np.uint32)
            self.cluster[a:b] = part['seq_cluster']
            self.score[a:b] = part['seq_score']
            self.meanlen[a:b] = part['seq_meanlen']
            self.src[a:b] = part['seq_src']
            if len(part['frag_seq']):
                # one global key per split document (region << 32 | local document row)
                keys = (np.int64(r) << np.int64(32)) | part['frag_parent'].astype(np.int64)
                gseq = a + part['frag_seq'].astype(np.int64)
                self.parent[gseq] = keys          # last writer wins; frag_pairs keeps them all
                pair_seq.append(gseq)
                pair_key.append(keys)
        self.frag_pairs = (np.concatenate(pair_seq) if pair_seq else np.zeros(0, dtype=np.int64),
                           np.concatenate(pair_key) if pair_key else np.zeros(0, dtype=np.int64))
        # ragged sequence -> parent-document keys: one bin can hold the tail fragments of two
        # different long documents, so a single key per sequence would silently drop one of them
        ps, pk = self.frag_pairs
        srt = np.argsort(ps, kind='stable')
        self.parent_key = pk[srt]
        self.parent_ptr = np.searchsorted(ps[srt], np.arange(total + 1))

    def __len__(self) -> int:
        return len(self.region)


def strata(table: "SequenceTable", clusters: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-cluster quartiles of the score and of the mean document length -> stratum id 0..15."""
    sid = np.zeros(len(table), dtype=np.uint8)
    bounds: dict[str, dict] = {'score': {}, 'length': {}}
    for c in clusters.tolist():
        sel = np.flatnonzero(table.cluster == c)
        if not len(sel):
            continue
        sq = np.quantile(table.score[sel].astype(np.float64), [0.25, 0.5, 0.75])
        lq = np.quantile(table.meanlen[sel].astype(np.float64), [0.25, 0.5, 0.75])
        bounds['score'][str(c)] = [float(x) for x in sq]
        bounds['length'][str(c)] = [float(x) for x in lq]
        q = np.searchsorted(sq, table.score[sel].astype(np.float64), side='right')
        b = np.searchsorted(lq, table.meanlen[sel].astype(np.float64), side='right')
        sid[sel] = (np.minimum(q, 3) * N_LENGTH_STRATA + np.minimum(b, 3)).astype(np.uint8)
    return sid, bounds


def score_stratum_of(score: np.ndarray, bounds: list[float]) -> np.ndarray:
    """The same score-quartile rule, exposed so the tests can re-derive it independently."""
    return np.minimum(np.searchsorted(np.asarray(bounds, dtype=np.float64),
                                      np.asarray(score, dtype=np.float64), side='right'), 3)


def _stratum_walk_order() -> list[int]:
    """A Latin-square walk: every 4 consecutive steps cover all 4 score and all 4 length strata."""
    order = []
    for k in range(N_STRATA):
        q = k % N_SCORE_STRATA
        b = (k // N_SCORE_STRATA + q) % N_LENGTH_STRATA
        order.append(q * N_LENGTH_STRATA + b)
    return order


WALK = _stratum_walk_order()


def load_cluster_tree(path: Path | None, leaves: list[int]) -> dict:
    """Hierarchical grouping of the NVIDIA clusters, innermost level = the leaves themselves.

    Format (produced outside this repo):
      {"leaves":[...], "levels":[{"groups":[[...],[...]]}, ...], "marginal":{...}, "method":"..."}
    levels[0] is the coarsest partition; the last level must be the singletons.  A missing or
    unreadable file falls back to the trivial one-level tree, which makes the allocation exactly the
    flat largest-remainder over the 19 leaves.
    """
    tree = {'levels': [], 'source': None, 'sha256': None, 'method': 'flat (fallback: leaves only)',
            'marginal': None}
    if path is not None and Path(path).is_file():
        raw = json.loads(Path(path).read_text())
        levels = [list(map(list, lvl['groups'])) for lvl in raw['levels']]
        tree.update({'levels': levels, 'source': str(path), 'method': raw.get('method', 'unspecified'),
                     'marginal': raw.get('marginal'), 'declared_leaves': raw.get('leaves'),
                     'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    levels = [[list(g) for g in lvl] for lvl in tree['levels']]
    # keep only the clusters this corpus actually has; add unknown ones as their own group
    present = set(leaves)
    fixed = []
    for lvl in levels:
        groups = [[c for c in g if c in present] for g in lvl]
        covered = {c for g in groups for c in g}
        for c in sorted(present - covered):
            groups.append([c])
        groups = [g for g in groups if g]
        if groups:
            fixed.append(groups)
    if not fixed or sorted(len(g) for g in fixed[-1]) != [1] * len(present):
        fixed.append([[c] for c in sorted(present)])
    tree['levels'] = fixed
    tree['effective_leaves'] = sorted(present)
    return tree


def _children_of(parent: list[int], level: list[list[int]]) -> list[list[int]]:
    """The groups of `level` whose members lie inside `parent` (a partition refinement)."""
    ps = set(parent)
    out = [g for g in level if set(g) & ps]
    covered = {c for g in out for c in g}
    missing = sorted(ps - covered)
    if missing:
        out = out + [[c] for c in missing]
    return [[c for c in g if c in ps] for g in out if set(g) & ps]


def _apportion(budget: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Split a per-block budget among k children: exact sum, error carried across blocks.

    Per block the child gets floor(B*w/W) or that plus one, and never anything else, so its count
    can differ from its exact share by strictly less than one slot -- the property test 2 checks.
    Which children get the surplus slot rotates by error diffusion on the fractional parts, so the
    long-run share is exact instead of being frozen at the rounded value (a cluster whose share is
    7.2 sequences per block would otherwise get 7 forever, i.e. 2.8% too little over 760k blocks).
    """
    nb, k = len(budget), len(weights)
    alloc = np.zeros((nb, k), dtype=np.int64)
    W = int(weights.sum())
    if nb == 0 or k == 0:
        return alloc
    if W == 0:
        alloc[:, 0] = budget
        return alloc
    if k == 1:
        alloc[:, 0] = budget
        return alloc
    bw = budget.astype(np.int64)[:, None] * weights.astype(np.int64)[None, :]
    base = bw // W
    frac = bw - base * W
    surplus = budget.astype(np.int64) - base.sum(axis=1)
    acc = np.zeros(k, dtype=np.int64)
    idx0 = np.arange(k)
    for b in range(nb):
        acc += frac[b]
        r = int(surplus[b])
        row = base[b]
        if r > 0:
            pick = np.lexsort((idx0, -acc))[:r]      # largest accumulated fraction, ties by index
            row = row.copy()
            row[pick] += 1
            acc[pick] -= W
        alloc[b] = row
    return alloc


def hierarchical_quotas(tree: dict, clusters: np.ndarray, counts: np.ndarray,
                        nblocks: int) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Per-block sequence quota per cluster, allocated down the cluster tree.

    Level 0 first (so the rounding error cannot pile up on clusters that are similar to each other),
    then inside each group on the next level, down to the leaves.  Returns
    (quota[nblocks, ncluster], level-0 group index per cluster, level-0 groups).
    """
    cl_pos = {int(c): i for i, c in enumerate(clusters.tolist())}
    weight = {int(c): int(n) for c, n in zip(clusters.tolist(), counts.tolist())}
    quota = np.zeros((nblocks, len(clusters)), dtype=np.int64)
    level0 = _children_of(sorted(cl_pos), tree['levels'][0]) if tree['levels'] else [sorted(cl_pos)]
    group_of = np.zeros(len(clusters), dtype=np.int64)
    for gi, g in enumerate(level0):
        for c in g:
            group_of[cl_pos[c]] = gi
    if nblocks == 0 or int(counts.sum()) == 0:
        return quota.astype(np.int16), group_of, level0

    stack = [(sorted(cl_pos), np.full(nblocks, SEQ_PER_BLOCK, dtype=np.int64), 0)]
    while stack:
        members, budget, depth = stack.pop()
        if len(members) == 1:
            quota[:, cl_pos[members[0]]] = budget
            continue
        children = None
        while depth < len(tree['levels']):
            cand = _children_of(members, tree['levels'][depth])
            depth += 1
            if len(cand) > 1:
                children = cand
                break
        if children is None:
            children = [[c] for c in members]
        w = np.array([sum(weight[c] for c in g) for g in children], dtype=np.int64)
        alloc = _apportion(budget, w)
        for gi, g in enumerate(children):
            stack.append((sorted(g), alloc[:, gi], depth))
    return quota.astype(np.int16), group_of, level0


def block_quotas(counts: np.ndarray, nblocks: int) -> np.ndarray:
    """Flat largest-remainder allocation (the trivial one-level tree)."""
    clusters = np.arange(len(counts))
    tree = {'levels': [[[int(c)] for c in clusters]]}
    return hierarchical_quotas(tree, clusters, counts, nblocks)[0]


def interleave_groups(block_seq: np.ndarray, group_of_seq: np.ndarray, ngroups: int) -> np.ndarray:
    """Reorder the 64 slots of a block so the level-0 groups are spread evenly through it.

    A plain round robin front-loads the alternation and leaves the majority group in one long run at
    the end.  Each sequence is placed at the fraction (2*rank+1)/(2*group_size) of the block instead,
    which is the even-spread / Bresenham placement: the longest same-group run is then about
    ceil(largest group / smallest group).
    """
    if block_seq.size == 0 or ngroups <= 1:
        return block_seq
    g = group_of_seq[block_seq]
    srt = np.argsort(g, axis=1, kind='stable')
    gs = np.take_along_axis(g, srt, axis=1)
    nb, k = gs.shape
    cols = np.arange(k)
    change = np.ones_like(gs, dtype=bool)
    change[:, 1:] = gs[:, 1:] != gs[:, :-1]
    start = np.maximum.accumulate(np.where(change, cols, 0), axis=1)
    counts = (g[:, :, None] == np.arange(ngroups)).sum(axis=1)       # (blocks, groups)
    size = np.take_along_axis(counts, gs, axis=1)
    rank = cols - start
    key = (2 * rank + 1) * k // (2 * size)                # even-spread position inside the block
    composite = key * ngroups + gs                        # ties inside a slot resolved by group id
    final = np.take_along_axis(srt, np.argsort(composite, axis=1, kind='stable'), axis=1)
    return np.take_along_axis(block_seq, final, axis=1)


def assemble_blocks(table: "SequenceTable", sid: np.ndarray, *, seed: str, split: str,
                    tree: dict, max_blocks: int | None = None) -> dict:
    """Levels 2 and 3.  Returns the block -> sequence assignment and the emission order."""
    n = len(table)
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(f'{seed}\0{split}\0blocks'.encode()).digest()[:8], 'big'))
    clusters = np.unique(table.cluster)
    cl_pos = {int(c): i for i, c in enumerate(clusters)}
    counts = np.array([int((table.cluster == c).sum()) for c in clusters], dtype=np.int64)
    ncl = len(clusters)

    # stratum queues: sequences grouped by (cluster, stratum), shuffled inside each group
    key = np.zeros(n, dtype=np.int64)
    for c, i in cl_pos.items():
        key[table.cluster == c] = i
    key = key * N_STRATA + sid.astype(np.int64)
    order = np.lexsort((rng.random(n), key))
    gptr = np.searchsorted(key[order], np.arange(ncl * N_STRATA + 1))
    cursor = gptr[:-1].copy()
    gend = gptr[1:].copy()

    nblocks = n // SEQ_PER_BLOCK
    if max_blocks is not None:
        nblocks = min(nblocks, max_blocks)
    quota, group_of, level0 = hierarchical_quotas(tree, clusters, counts, nblocks)
    parent = table.parent
    deferred: dict[int, list[int]] = defaultdict(list)

    pkey, pptr = table.parent_key, table.parent_ptr

    def keys_of(seq: int):
        return pkey[pptr[seq]:pptr[seq + 1]]

    def clashes(seq: int, seen: set[int]) -> bool:
        for key in keys_of(seq).tolist():
            if key in seen:
                return True
        return False

    def take(g: int, seen: set[int], hold: list[tuple[int, int]]) -> int:
        q = deferred[g]
        for i, cand in enumerate(q):
            if not clashes(cand, seen):
                q.pop(i)
                return cand
        while cursor[g] < gend[g]:
            cand = int(order[cursor[g]])
            cursor[g] += 1
            if clashes(cand, seen):
                hold.append((g, cand))       # two fragments of one document: not in this block
                continue
            return cand
        return -1

    block_seq = np.full((nblocks, SEQ_PER_BLOCK), -1, dtype=np.int64)
    remaining = counts.copy()
    short_slots = 0
    used_blocks = nblocks
    for b in range(nblocks):
        want = np.minimum(quota[b].astype(np.int64), remaining)
        over = int(quota[b].sum()) - int(want.sum())
        if over:
            for ci in np.argsort(-(remaining - want)).tolist():
                take_n = min(over, int(remaining[ci] - want[ci]))
                want[ci] += take_n
                over -= take_n
                if not over:
                    break
        slot = 0
        seen: set[int] = set()
        hold: list[tuple[int, int]] = []
        for ci in range(ncl):
            need = int(want[ci])
            if need <= 0:
                continue
            step = (b * 7 + ci * 5) % N_STRATA
            taken = misses = 0
            while taken < need and misses < N_STRATA:
                g = ci * N_STRATA + WALK[step % N_STRATA]
                step += 1
                pick = take(g, seen, hold)
                if pick < 0:
                    misses += 1
                    continue
                misses = 0
                seen.update(keys_of(pick).tolist())
                block_seq[b, slot] = pick
                slot += 1
                taken += 1
                remaining[ci] -= 1
            short_slots += need - taken
        if slot < SEQ_PER_BLOCK:                      # end of corpus: fill from anything left
            for g in range(ncl * N_STRATA):
                while slot < SEQ_PER_BLOCK:
                    pick = take(g, seen, hold)
                    if pick < 0:
                        break
                    seen.update(keys_of(pick).tolist())
                    block_seq[b, slot] = pick
                    slot += 1
                    remaining[g // N_STRATA] -= 1
                if slot >= SEQ_PER_BLOCK:
                    break
        for g, cand in hold:
            deferred[g].append(cand)
        if slot < SEQ_PER_BLOCK:
            block_seq = block_seq[:b]
            used_blocks = b
            break
    nblocks = used_blocks
    quota = quota[:nblocks]
    block_seq = block_seq[:nblocks]
    # inside a block the 64 slots alternate the level-0 groups instead of running group by group
    if nblocks:
        group_of_seq = group_of[np.array([cl_pos[int(c)] for c in table.cluster], dtype=np.int64)]
        block_seq = interleave_groups(block_seq, group_of_seq, len(level0))

    # ---- level 3: global shuffle, then repair fragment adjacency -----------------------------
    emit_pos = rng.permutation(nblocks) if nblocks else np.zeros(0, dtype=np.int64)
    emit_pos = emit_pos.astype(np.int64)
    seq_block = np.full(n, -1, dtype=np.int64)
    if nblocks:
        flat = block_seq.reshape(-1)
        seq_block[flat] = np.repeat(np.arange(nblocks), SEQ_PER_BLOCK)
    repairs = _repair_fragment_adjacency(table, seq_block, emit_pos, rng)
    violations = _fragment_violations(table, seq_block, emit_pos)
    order_out = np.empty(nblocks, dtype=np.int64)
    if nblocks:
        order_out[emit_pos] = np.arange(nblocks)

    return {'block_seq': block_seq, 'order': order_out, 'emit_pos': emit_pos, 'clusters': clusters,
            'counts': counts, 'quota': quota, 'repairs': repairs, 'short_slots': short_slots,
            'leftover': n - nblocks * SEQ_PER_BLOCK, 'group_of_cluster': group_of,
            'level0_groups': level0, 'fragment_violations': violations}


def _fragment_violations(table: "SequenceTable", seq_block: np.ndarray,
                         emit_pos: np.ndarray) -> dict:
    """Fragments of one document still in the same or an adjacent emitted block.

    With a few hundred thousand blocks this is empty; it can stay non-zero only on a tiny split,
    where placing k fragments in k pairwise non-adjacent blocks is pigeonhole-impossible.
    """
    pair_seq, pair_key = table.frag_pairs
    if not len(pair_seq) or not len(emit_pos):
        return {'same_block': 0, 'adjacent_blocks': 0, 'pairs': 0}
    used = seq_block[pair_seq] >= 0
    pair_seq, pair_key = pair_seq[used], pair_key[used]
    if not len(pair_seq):
        return {'same_block': 0, 'adjacent_blocks': 0, 'pairs': 0}
    pos = emit_pos[seq_block[pair_seq]]
    srt = np.lexsort((pos, pair_key))
    k, p = pair_key[srt], pos[srt]
    same = k[1:] == k[:-1]
    d = np.abs(np.diff(p))
    return {'same_block': int((same & (d == 0)).sum()),
            'adjacent_blocks': int((same & (d == 1)).sum()), 'pairs': int(same.sum())}


def _repair_fragment_adjacency(table: "SequenceTable", seq_block: np.ndarray, emit_pos: np.ndarray,
                               rng: np.random.Generator, max_passes: int = 32) -> int:
    """Fragments of one document must land in different, non-consecutive emitted blocks."""
    pair_seq, pair_key = table.frag_pairs
    if not len(pair_seq) or not len(emit_pos):
        return 0
    used = seq_block[pair_seq] >= 0
    pair_seq, pair_key = pair_seq[used], pair_key[used]
    if not len(pair_seq):
        return 0
    nblocks = len(emit_pos)
    repairs = 0
    for _ in range(max_passes):
        pos = emit_pos[seq_block[pair_seq]]
        srt = np.lexsort((pos, pair_key))
        k, p = pair_key[srt], pos[srt]
        bad = np.flatnonzero((k[1:] == k[:-1]) & (np.abs(np.diff(p)) <= 1))
        if not len(bad):
            return repairs
        moved: set[int] = set()
        for i in bad.tolist():
            b = int(seq_block[pair_seq[srt[i + 1]]])
            if b in moved:
                continue
            other = int(rng.integers(nblocks))
            if other == b or other in moved:
                continue
            pb, po = int(emit_pos[b]), int(emit_pos[other])
            emit_pos[b], emit_pos[other] = po, pb
            moved.add(b)
            moved.add(other)
            repairs += 1
    return repairs
