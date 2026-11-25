#!/usr/bin/env python3
"""Pack the 100B v2 corpus into the trainer's schema-v2 shards with *stratified* blocks.

pack_shards.py emits documents in corpus order.  Every region of this corpus holds ~50M tokens of a
single NVIDIA cluster (~378 blocks), so every 64-sequence training block was mono-topic and the loss
sawtoothed at each region change.  This producer keeps the output format byte-for-byte identical --
2049-token uint16 records, 64 per block, ~1 GiB immutable shards, the same manifest / run_contract /
frontier artefacts, all written by the trainer's own ImmutableShardWriter -- and replaces the order:

  level 1  documents -> 2049-token records, best-fit-decreasing inside one cluster (Ding 2024)
  level 2  records   -> blocks of 64 whose cluster histogram follows the corpus marginal, allocated
                        down the NVIDIA cluster tree (coarse groups first, then leaves), with the
                        records of one cluster drawn round-robin from different st_reasoning_dense
                        and mean-length strata, and the 64 slots interleaving the coarse groups
  level 3  blocks    -> one global shuffle with a fixed seed (Dai 2026: global beats just-in-time)

The whole plan is computed from the document index before block 0 is written, so the output is a
*stream*: ~1 GiB shards are finalized, uploaded and published through publish_frontier as they are
produced, and the trainer can start as soon as the frontier passes its threshold.

    .venv/bin/python pack_stratified.py --out corpus-13b-v2/shards-strat --no-upload \
        --regions 0-5 --max-blocks 200
    .venv/bin/python pack_stratified.py --out corpus-13b-v2/shards-strat \
        --run-id moe-100b-superbpe-b64-dataset-003 --bucket OWNER/NAME --public
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import strat_plan as sp                                                              # noqa: E402
from strat_plan import (CONTEXT, EOD, SEQ_PER_BLOCK, STORED, SequenceTable,           # noqa: E402
                        assemble_blocks, available_regions, build_region_plan,
                        load_cluster_tree, region_bins, strata)

from kefir.data import config                                                           # noqa: E402
config.EOD_TOKEN_ID = EOD
from kefir.data.src.storage import write_json_atomic                                     # noqa: E402
from kefir.data.src.streaming import (SEQUENCE_FORMAT, STREAM_CACHE_SCHEMA_VERSION,      # noqa: E402
                                   ImmutableShardWriter, PreparedBlock, TokenDeficitScheduler)
from kefir.data.src.verify import verify                                                 # noqa: E402
from kefir.data.incremental_frontier import (build_run_contract, publish_frontier,        # noqa: E402
                                          publish_run_contract)
from _helpers import filehash as sha256_file, stable_hash, upload_with_retry          # noqa: E402

VOCAB = 8000
SHARD_BYTES = 1024 ** 3
STATE_FILE = 'pack-state.json'
DEFAULT_RUN_ID = 'moe-100b-superbpe-b64-dataset-003'
DEFAULT_SEED = 'small-lm-stratified-blocks-v1'
DEFECTS = ('none', 'mono-cluster', 'offset-plus-one')
SPLITS = ('validation', 'train')          # validation first: it is 0.1%, so it is READY immediately


def tokenizer_identity(sha256: str) -> dict:
    return {'source_tokenizer_id': 'gpt2', 'output_tokenizer_id': 'superbpe_8000',
            'tokenizer_sha256': sha256, 'semantic_vocab_size': VOCAB, 'eod_token_id': EOD}


# ============================================================================================
# plan: level 1 per region, levels 2-3 globally, both persisted so a resume never recomputes
# ============================================================================================
def _region_plan_job(args):
    root, plan_dir, region, val_seed, prob = args
    return build_region_plan(Path(root), Path(plan_dir), region, validation_seed=val_seed,
                             validation_probability=prob)


def build_level1(root: Path, plan_dir: Path, regions: list[int], *, val_seed: str, prob: float,
                 workers: int) -> list[dict]:
    jobs = [(str(root), str(plan_dir), r, val_seed, prob) for r in regions]
    todo = [j for j in jobs
            if not (Path(plan_dir) / f'region-{j[2]:06d}' / 'meta.json').exists()]
    if workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_region_plan_job, todo, chunksize=1))
    else:
        for j in todo:
            _region_plan_job(j)
    return [json.loads((plan_dir / f'region-{r:06d}' / 'meta.json').read_text()) for r in regions]


def _load_seq_parts(plan_dir: Path, regions: list[int], split: str):
    keep, parts = [], []
    for r in regions:
        meta = json.loads((plan_dir / f'region-{r:06d}' / 'meta.json').read_text())
        if meta.get('empty') or meta['splits'][split]['sequences'] == 0:
            continue
        d = plan_dir / f'region-{r:06d}' / split
        parts.append({'seq_cluster': np.load(d / 'seq_cluster.npy'),
                      'seq_score': np.load(d / 'seq_score.npy'),
                      'seq_meanlen': np.load(d / 'seq_meanlen.npy'),
                      'seq_src': np.load(d / 'seq_src.npy'),
                      'frag_seq': np.load(d / 'frag_seq.npy'),
                      'frag_parent': np.load(d / 'frag_parent.npy')})
        keep.append(r)
    return keep, parts


def build_global_plan(plan_dir: Path, regions: list[int], split: str, *, seed: str, tree: dict,
                      max_blocks: int | None, defect: str, force: bool = False) -> dict:
    """Levels 2 and 3 for one split, persisted under plan_dir/global/<split>/."""
    dst = plan_dir / 'global' / split
    key = stable_hash({'v': sp.PLAN_VERSION, 'seed': seed, 'regions': regions, 'split': split,
                       'max_blocks': max_blocks, 'defect': defect,
                       'tree': tree['levels'], 'tree_sha256': tree.get('sha256')})
    meta_path = dst / 'meta.json'
    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if meta.get('key') == key:
            return meta
    kept, parts = _load_seq_parts(plan_dir, regions, split)
    table = SequenceTable(kept, parts)
    clusters = np.unique(table.cluster) if len(table) else np.zeros(0, np.uint8)
    sid, bounds = strata(table, clusters)
    if len(table) >= SEQ_PER_BLOCK:
        plan = assemble_blocks(table, sid, seed=seed, split=split, tree=tree, max_blocks=max_blocks)
    else:
        plan = {'block_seq': np.zeros((0, SEQ_PER_BLOCK), np.int64), 'order': np.zeros(0, np.int64),
                'emit_pos': np.zeros(0, np.int64), 'clusters': clusters,
                'counts': np.zeros(len(clusters), np.int64),
                'quota': np.zeros((0, len(clusters)), np.int16), 'repairs': 0, 'short_slots': 0,
                'leftover': len(table), 'group_of_cluster': np.zeros(len(clusters), np.int64),
                'level0_groups': [clusters.tolist()]}
    if defect == 'mono-cluster':
        plan = _mono_cluster_plan(table, clusters, max_blocks)
    dst.mkdir(parents=True, exist_ok=True)
    arrays = {'block_seq': plan['block_seq'].astype(np.int64),
              'order': plan['order'].astype(np.int64),
              'quota': plan['quota'].astype(np.int16),
              'seq_region': table.region, 'seq_lseq': table.lseq, 'seq_cluster': table.cluster,
              'seq_score': table.score, 'seq_meanlen': table.meanlen, 'seq_stratum': sid,
              'seq_parent': table.parent,
              'group_of_cluster': plan['group_of_cluster'].astype(np.int64)}
    for name, arr in arrays.items():
        tmp = dst / f'{name}.npy.swap'
        with open(tmp, 'wb') as fh:
            np.save(fh, arr)
        os.replace(tmp, dst / f'{name}.npy')
    meta = {'key': key, 'split': split, 'sequences': int(len(table)),
            'blocks': int(len(plan['order'])), 'regions': kept,
            'clusters': [int(c) for c in plan['clusters']],
            'counts': [int(x) for x in plan['counts']],
            'level0_groups': [[int(c) for c in g] for g in plan['level0_groups']],
            'stratum_bounds': bounds, 'repairs': int(plan['repairs']),
            'short_slots': int(plan['short_slots']), 'leftover': int(plan['leftover']),
            'fragment_violations': plan.get('fragment_violations',
                                            {'same_block': 0, 'adjacent_blocks': 0, 'pairs': 0}),
            'source_tokens': int(table.src.astype(np.int64).sum()),
            'plan_version': sp.PLAN_VERSION, 'seed': seed, 'defect': defect,
            'tree': {'levels': tree['levels'], 'method': tree.get('method'),
                     'source': tree.get('source'), 'sha256': tree.get('sha256')}}
    (dst / 'meta.json.swap').write_text(json.dumps(meta))
    os.replace(dst / 'meta.json.swap', meta_path)
    return meta


def _mono_cluster_plan(table, clusters, max_blocks):
    """TEST ONLY planted defect: every block is filled from a single cluster (the old behaviour)."""
    n = len(table)
    order = np.lexsort((np.arange(n), table.cluster.astype(np.int64)))
    nblocks = n // SEQ_PER_BLOCK
    if max_blocks is not None:
        nblocks = min(nblocks, max_blocks)
    block_seq = order[:nblocks * SEQ_PER_BLOCK].reshape(nblocks, SEQ_PER_BLOCK)
    rng = np.random.default_rng(7)
    emit_pos = rng.permutation(nblocks).astype(np.int64)
    out = np.empty(nblocks, dtype=np.int64)
    out[emit_pos] = np.arange(nblocks)
    counts = np.array([int((table.cluster == c).sum()) for c in clusters], np.int64)
    return {'block_seq': block_seq, 'order': out, 'emit_pos': emit_pos, 'clusters': clusters,
            'counts': counts, 'quota': np.zeros((nblocks, len(clusters)), np.int16), 'repairs': 0,
            'short_slots': 0, 'leftover': n - nblocks * SEQ_PER_BLOCK,
            'group_of_cluster': np.zeros(len(clusters), np.int64), 'level0_groups': [clusters.tolist()]}


# ============================================================================================
# reading: turn planned blocks into payload bytes
# ============================================================================================
def _ragged(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Concatenated ranges [start_i, start_i+len_i) without a Python loop."""
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    ends = np.cumsum(lens)
    return np.repeat(starts - (ends - lens), lens) + np.arange(total, dtype=np.int64)


class BlockReader:
    """Materialises planned blocks into payload bytes.  Cheap to build inside a pool worker."""

    def __init__(self, plan_dir: Path, split: str, bins: dict[int, Path], defect: str = 'none'):
        d = Path(plan_dir) / 'global' / split
        self.split = split
        self.defect = defect
        self.plan_dir = Path(plan_dir)
        self.block_seq = np.load(d / 'block_seq.npy', mmap_mode='r')
        self.order = np.load(d / 'order.npy')
        self.seq_region = np.load(d / 'seq_region.npy')
        self.seq_lseq = np.load(d / 'seq_lseq.npy')
        self.seq_cluster = np.load(d / 'seq_cluster.npy')
        self.bins = bins
        self._ent: dict[int, dict[str, np.ndarray]] = {}
        self._mm: dict[int, np.memmap] = {}

    @property
    def nblocks(self) -> int:
        return len(self.order)

    def entries(self, region: int) -> dict[str, np.ndarray]:
        got = self._ent.get(region)
        if got is None:
            d = self.plan_dir / f'region-{region:06d}' / self.split
            got = {k: np.load(d / f'{k}.npy', mmap_mode='r')
                   for k in ('seq_ptr', 'ent_doc', 'ent_off', 'ent_len', 'ent_term', 'doc_index')}
            self._ent[region] = got
        return got

    def mm(self, region: int) -> np.memmap:
        got = self._mm.get(region)
        if got is None:
            got = np.memmap(self.bins[region], dtype='<u2', mode='r')
            self._mm[region] = got
        return got

    def block_sequences(self, emit_index: int) -> np.ndarray:
        return np.asarray(self.block_seq[self.order[emit_index]])

    def materialise(self, lo: int, hi: int) -> dict:
        """Read blocks [lo, hi) of the emission order, region-sequentially, into one staging array.

        Everything is vectorised over the whole batch: the per-sequence Python loop this replaces
        cost about 70% of the wall clock at 64 blocks per batch (64 x 64 sequences x ~4.5 entries).
        """
        nb = hi - lo
        nseq = nb * SEQ_PER_BLOCK
        stage = np.full((nseq, STORED), EOD, dtype='<u2')
        flat = stage.reshape(-1)
        if nb <= 0:
            return {'stage': stage, 'entries': None, 'cluster': np.zeros(0, np.int64)}
        gseq = np.concatenate([self.block_sequences(e) for e in range(lo, hi)])
        regions = np.asarray(self.seq_region[gseq], dtype=np.int64)
        lseq = np.asarray(self.seq_lseq[gseq], dtype=np.int64)
        order = np.argsort(regions, kind='stable')
        cuts = np.flatnonzero(np.diff(regions[order])) + 1
        chunks = []
        for a, b in zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [len(order)]])):
            slots = order[a:b]
            r = int(regions[slots[0]])
            ent = self.entries(r)
            ptr = np.asarray(ent['seq_ptr'], dtype=np.int64)
            starts, n = ptr[lseq[slots]], ptr[lseq[slots] + 1] - ptr[lseq[slots]]
            idx = _ragged(starts, n)
            lens = np.asarray(ent['ent_len'], dtype=np.int64)[idx]
            csum = np.cumsum(lens)
            group_end = np.cumsum(n)
            base = np.concatenate([[0], csum[group_end[:-1] - 1]]) if len(n) > 1 else np.zeros(1, np.int64)
            within = csum - lens - np.repeat(base, n)
            slot_of = np.repeat(slots, n)
            chunks.append({
                'region': np.full(len(idx), r, dtype=np.int64),
                'off': np.asarray(ent['ent_off'], dtype=np.int64)[idx],
                'len': lens, 'dest': slot_of * STORED + within,
                'term': np.asarray(ent['ent_term'], dtype=np.int64)[idx],
                'doc': np.asarray(ent['doc_index'], dtype=np.int64)[
                    np.asarray(ent['ent_doc'], dtype=np.int64)[idx]],
                'seq': slot_of})
        c = {k: np.concatenate([ch[k] for ch in chunks]) for k in chunks[0]}
        read_off = c['off']
        if self.defect == 'offset-plus-one':          # TEST ONLY planted defect
            read_off = read_off + 1
        edges = np.flatnonzero(np.diff(c['region'])) + 1
        for a, b in zip(np.concatenate([[0], edges]), np.concatenate([edges, [len(c['region'])]])):
            srt = a + np.argsort(read_off[a:b], kind='stable')      # sequential inside the region
            mm = self.mm(int(c['region'][a]))
            src = _ragged(read_off[srt], c['len'][srt])
            if self.defect != 'none':
                src = src % len(mm)                   # the planted offset can run past the last doc
            flat[_ragged(c['dest'][srt], c['len'][srt])] = mm[src]
        srt = np.argsort(c['seq'], kind='stable')                   # back to slot order
        c = {k: v[srt] for k, v in c.items()}
        return {'stage': stage, 'entries': c, 'cluster': np.asarray(self.seq_cluster[gseq], np.int64),
                'gseq': gseq}

    def batch(self, lo: int, hi: int) -> dict:
        """Everything the producer needs for blocks [lo, hi): payloads plus their bookkeeping."""
        m = self.materialise(lo, hi)
        c, cluster, stage = m['entries'], m['cluster'], m['stage']
        if c is None:
            return {'blocks': [], 'sidecar': None}
        nb = hi - lo
        block_of_entry = c['seq'] // SEQ_PER_BLOCK
        cl_of_entry = cluster[c['seq']]
        src = c['len'] - c['term']
        # per (block, cluster) source tokens, per-block documents / padding / cluster histogram
        per_bc = np.bincount(block_of_entry * 21 + cl_of_entry, weights=src,
                             minlength=nb * 21).astype(np.int64).reshape(nb, 21)
        docs = np.bincount(block_of_entry, weights=c['term'], minlength=nb).astype(np.int64)
        used = np.bincount(block_of_entry, weights=c['len'], minlength=nb).astype(np.int64)
        hist = np.bincount(np.repeat(np.arange(nb), SEQ_PER_BLOCK) * 21 + cluster,
                           minlength=nb * 21).astype(np.int64).reshape(nb, 21)
        bounds = np.searchsorted(c['seq'], np.arange(nb + 1) * SEQ_PER_BLOCK)
        blocks = []
        for k in range(nb):
            nz = np.flatnonzero(per_bc[k])
            blocks.append({
                'payload': stage[k * SEQ_PER_BLOCK:(k + 1) * SEQ_PER_BLOCK].tobytes(),
                'per_cluster': {int(x): int(per_bc[k, x]) for x in nz},
                'documents': int(docs[k]),
                'padding': SEQ_PER_BLOCK * STORED - int(used[k]),
                'hist': hist[k],
                'first_source_id': _sid(c, int(bounds[k])),
                'last_source_id': _sid(c, int(bounds[k + 1]) - 1)})
        # compact dtypes: at 100B tokens this is ~221M entries, 15 bytes each (3.3 GiB) instead of
        # 56; it is the only artefact that lets anyone re-derive a block from the source afterwards
        sidecar = {'entry_block': (block_of_entry + lo).astype(np.int32),
                   'entry_slot': (c['seq'] % SEQ_PER_BLOCK).astype(np.uint8),
                   'entry_region': c['region'].astype(np.uint16),
                   'entry_doc': c['doc'].astype(np.uint32), 'entry_off': c['off'].astype(np.uint32),
                   'entry_len': c['len'].astype(np.uint16), 'entry_term': c['term'].astype(np.uint8),
                   'seq_block': np.repeat(np.arange(lo, hi), SEQ_PER_BLOCK).astype(np.int32),
                   'seq_slot': np.tile(np.arange(SEQ_PER_BLOCK), nb).astype(np.uint8),
                   'seq_global': m['gseq'].astype(np.int64),
                   'seq_cluster': cluster.astype(np.uint8)}
        return {'blocks': blocks, 'sidecar': sidecar}


def _sid(c, i: int) -> str:
    return '<padding>' if i < 0 else f"r{int(c['region'][i])}:d{int(c['doc'][i])}"


_WORKER: dict[str, BlockReader] = {}


def _worker_init(plan_dir, bins, defect):
    for split in SPLITS:
        try:
            _WORKER[split] = BlockReader(Path(plan_dir), split, bins, defect)
        except FileNotFoundError:
            pass


def _worker_batch(args):
    split, lo, hi = args
    return split, lo, hi, _WORKER[split].batch(lo, hi)


# ============================================================================================
# producer
# ============================================================================================
class _Doc:
    """What TokenDeficitScheduler.emit() actually reads off a SourceDocument."""

    __slots__ = ('cluster_id', 'source_token_count')

    def __init__(self, cluster_id: int, source_token_count: int):
        self.cluster_id = cluster_id
        self.source_token_count = source_token_count


class _Row:
    def __init__(self, d):
        self.__dict__.update(d)


class Producer:
    def __init__(self, out: Path, *, sidecar: bool = True, plan_key: str = ''):
        self.out = out
        self.sidecar = sidecar
        self.plan_key = plan_key
        self.scheduler = TokenDeficitScheduler({c: 1 for c in config.ACCEPTED_CLUSTER_IDS},
                                               'stratified-global-shuffle')
        self.validation_source_tokens = 0
        self.documents = 0
        self.padding_tokens = 0
        self.emitted = {'train': 0, 'validation': 0}
        self.block_id = {'train': 0, 'validation': 0}
        self.entropy_sum = {'train': 0.0, 'validation': 0.0}
        self.distinct_sum = {'train': 0, 'validation': 0}
        self.cluster_hist = {s: np.zeros(21, dtype=np.int64) for s in SPLITS}
        self.writers = {s: ImmutableShardWriter(out, split=s, target_bytes=SHARD_BYTES,
                                                context_length=CONTEXT) for s in SPLITS}
        self.finalized: list[dict] = []

    def consume(self, split: str, lo: int, batch: dict) -> list:
        rows = []
        for k, blk in enumerate(batch['blocks']):
            per_cluster = blk['per_cluster']
            self.documents += blk['documents']
            self.padding_tokens += blk['padding']
            if split == 'train':
                for c, v in sorted(per_cluster.items()):
                    if v > 0:
                        self.scheduler.emit(_Doc(c, v))
                cumulative = self.scheduler.total_emitted_source_tokens
            else:
                self.validation_source_tokens += int(sum(per_cluster.values()))
                cumulative = self.validation_source_tokens
            hist = np.asarray(blk['hist'])
            self.cluster_hist[split] += hist
            p = hist[hist > 0] / SEQ_PER_BLOCK
            self.entropy_sum[split] += float(-(p * np.log(p)).sum())
            self.distinct_sum[split] += int((hist > 0).sum())
            block = PreparedBlock(block_id=self.block_id[split], split=split,
                                  sequence_count=SEQ_PER_BLOCK, token_count=SEQ_PER_BLOCK * STORED,
                                  payload=blk['payload'], cumulative_source_tokens=cumulative,
                                  per_cluster_source_tokens=per_cluster,
                                  first_source_id=blk['first_source_id'],
                                  last_source_id=blk['last_source_id'])
            self.block_id[split] += 1
            rows += self.writers[split].write_block(block)
        if self.sidecar and batch['sidecar'] is not None:
            d = self.out / 'plan' / split
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f'blocks-{lo:09d}.npz.swap'
            with open(tmp, 'wb') as fh:
                np.savez(fh, **{k: np.asarray(v) for k, v in batch['sidecar'].items()})
            os.replace(tmp, d / f'blocks-{lo:09d}.npz')
        self.emitted[split] = lo + len(batch['blocks'])
        return rows

    # -- checkpoint / resume -------------------------------------------------------------------
    def state(self) -> dict:
        return {'plan_key': self.plan_key, 'emitted': self.emitted, 'documents': self.documents,
                'validation_source_tokens': self.validation_source_tokens,
                'padding_tokens': self.padding_tokens, 'scheduler': self.scheduler.state_dict(),
                'block_id': self.block_id, 'entropy_sum': self.entropy_sum,
                'distinct_sum': self.distinct_sum,
                'cluster_hist': {k: v.tolist() for k, v in self.cluster_hist.items()},
                'writer_index': {s: self.writers[s]._index for s in SPLITS},
                'cumulative': {s: {str(k): v for k, v in self.writers[s]._cumulative_counts.items()}
                               for s in SPLITS},
                'finalized': self.finalized}

    def load_state(self, st: dict) -> None:
        if st.get('plan_key') and self.plan_key and st['plan_key'] != self.plan_key:
            raise SystemExit(
                'REFUSING TO RESUME: the plan this output directory was written with no longer '
                'matches the plan for the current arguments (a different --seed, --regions, '
                '--cluster-tree, or more regions having appeared on disk since the run started). '
                'Every block already written follows the old global shuffle; continuing would mix '
                'two streams. Start a fresh --out, or restore the original arguments.\n'
                f"  written with: {st['plan_key']}\n  now:          {self.plan_key}")
        self.emitted = {k: int(v) for k, v in st['emitted'].items()}
        self.documents = int(st['documents'])
        self.validation_source_tokens = int(st['validation_source_tokens'])
        self.padding_tokens = int(st['padding_tokens'])
        self.scheduler = TokenDeficitScheduler({c: 1 for c in config.ACCEPTED_CLUSTER_IDS},
                                               st['scheduler']['tie_break_seed'])
        self.scheduler = TokenDeficitScheduler.from_state({c: 1 for c in config.ACCEPTED_CLUSTER_IDS},
                                                          st['scheduler'])
        self.block_id = {k: int(v) for k, v in st['block_id'].items()}
        self.entropy_sum = {k: float(v) for k, v in st['entropy_sum'].items()}
        self.distinct_sum = {k: int(v) for k, v in st['distinct_sum'].items()}
        self.cluster_hist = {k: np.array(v, dtype=np.int64) for k, v in st['cluster_hist'].items()}
        self.writers = {s: ImmutableShardWriter(self.out, split=s, target_bytes=SHARD_BYTES,
                                                context_length=CONTEXT,
                                                start_index=int(st['writer_index'][s])) for s in SPLITS}
        for s in SPLITS:
            self.writers[s]._cumulative_counts = {int(k): v for k, v in st['cumulative'][s].items()}
        self.finalized = st['finalized']
        for s in SPLITS:
            d = self.out / s
            if d.exists():
                for p in d.glob('.*.bin.tmp'):
                    p.unlink()
                for p in d.glob(f'{s}-*.bin'):
                    if int(p.stem.split('-')[-1]) >= self.writers[s]._index:
                        p.unlink()
        for row in self.finalized:
            self.writers[row['split']].shards.append(_Row(row))


# ============================================================================================
def parse_regions(spec: str | None, root: Path) -> list[int]:
    have = available_regions(root)
    if not spec:
        return have
    want: set[int] = set()
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            want.update(range(int(a), int(b) + 1))
        else:
            want.add(int(part))
    missing = sorted(want - set(have))
    if missing:
        raise SystemExit(f'{len(missing)} requested regions have no index, no receipt, or an '
                         f'incomplete .bin: {missing[:10]}')
    return sorted(want)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=ROOT / 'corpus-13b-v2')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--plan-dir', type=Path, default=None, help='default: <out>/plan-cache')
    ap.add_argument('--regions', default=None, help='e.g. 0-5,10 ; default: every region with index+bin')
    ap.add_argument('--run-id', default=DEFAULT_RUN_ID)
    ap.add_argument('--seed', default=DEFAULT_SEED)
    ap.add_argument('--cluster-tree', type=Path, default=None,
                    help='hierarchical cluster grouping; default <root>/cluster_tree.json when present')
    ap.add_argument('--tokenizer', type=Path, default=ROOT.parents[1] / 'src/kefir/assets/superbpe_8000_v2.json')
    ap.add_argument('--bucket', default=None)
    ap.add_argument('--public', action='store_true')
    ap.add_argument('--no-upload', action='store_true')
    ap.add_argument('--nominal-training-tokens', type=int, default=100_000_000_000)
    ap.add_argument('--stop-margin-blocks', type=int, default=64)
    ap.add_argument('--max-blocks', type=int, default=None, help='TEST: cap the planned train blocks')
    ap.add_argument('--gather-blocks', type=int, default=256,
                    help='blocks per region-sorted read sweep (bigger = denser reads, more RAM: '
                         '64 blocks of staging is 8.4 MiB)')
    ap.add_argument('--prefetch', type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)),
                    help='worker processes reading and assembling block batches ahead of the writer')
    ap.add_argument('--checkpoint-blocks', type=int, default=16384,
                    help='finalize the active shards and save a resumable state every N train blocks')
    ap.add_argument('--crash-after', type=int, default=None,
                    help='TEST ONLY: exit(3) at the first checkpoint at or after this many train blocks')
    ap.add_argument('--plan-workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument('--no-sidecar', action='store_true', help='skip out/plan/<split>/blocks-*.npz')
    ap.add_argument('--inject-defect', choices=DEFECTS, default='none',
                    help='TEST ONLY: plant a known defect so the tests can be shown to fail')
    return ap


def main(argv=None) -> int:
    a = build_argparser().parse_args(argv)
    from verify_assets import verify_sources
    verify_sources(ROOT)
    root, out = a.root, a.out
    regions = parse_regions(a.regions, root)
    if not regions:
        raise SystemExit('no region has both an index and a .bin locally')
    plan_dir = a.plan_dir or (out / 'plan-cache')
    out.mkdir(parents=True, exist_ok=True)
    plan_dir.mkdir(parents=True, exist_ok=True)
    tree_path = a.cluster_tree if a.cluster_tree is not None else (ROOT / 'assets/cluster_tree.json')

    t0 = time.time()
    metas = build_level1(root, plan_dir, regions, val_seed=config.SELECTION_SEED,
                         prob=config.VALIDATION_PROBABILITY, workers=a.plan_workers)
    leaves = sorted({c for m in metas if not m.get('empty') for c in m.get('clusters', [m['cluster']])})
    tree = load_cluster_tree(tree_path if Path(tree_path).is_file() else None, leaves)
    print(json.dumps({'event': 'level1', 'regions': len(regions), 'seconds': round(time.time() - t0, 1),
                      'sequences': sum(m['splits']['train']['sequences'] +
                                       m['splits']['validation']['sequences'] for m in metas),
                      'cluster_tree': tree['method'],
                      'tree_levels': [len(l) for l in tree['levels']]}), flush=True)

    t1 = time.time()
    plans = {s: build_global_plan(plan_dir, regions, s, seed=a.seed, tree=tree,
                                  max_blocks=a.max_blocks if s == 'train' else None,
                                  defect=a.inject_defect) for s in SPLITS}
    print(json.dumps({'event': 'level23', 'seconds': round(time.time() - t1, 1),
                      'train_blocks': plans['train']['blocks'],
                      'validation_blocks': plans['validation']['blocks'],
                      'fragment_repairs': plans['train']['repairs']}), flush=True)

    tok_sha = sha256_file(a.tokenizer)
    tokenizer_block = tokenizer_identity(tok_sha)
    schema_h = stable_hash({'stream_cache_schema_version': STREAM_CACHE_SCHEMA_VERSION,
                            'sequence_format': 'context_plus_one', 'context_length': CONTEXT,
                            'stored_sequence_tokens': STORED, 'sequences_per_block': SEQ_PER_BLOCK,
                            'int_type': config.INT_TYPE, 'byte_order': config.BYTE_ORDER,
                            'tokenizer': tokenizer_block})
    plan_h = plans['train']['key']
    config_h = stable_hash({'producer': 'pack_stratified.py', 'regions': len(regions),
                            'context_length': CONTEXT, 'sequences_per_block': SEQ_PER_BLOCK,
                            'target_shard_bytes': SHARD_BYTES, 'tokenizer_sha256': tok_sha,
                            'ordering': 'stratified blocks, global shuffle', 'plan_hash': plan_h})
    contract = build_run_contract(run_id=a.run_id, nominal_training_tokens=a.nominal_training_tokens,
                                  target_source_tokens=a.nominal_training_tokens,
                                  minimum_source_tokens=int(a.nominal_training_tokens * 0.9),
                                  maximum_source_tokens=int(a.nominal_training_tokens * 1.1),
                                  checkpoint_source_tokens=500_000_000, context_length=CONTEXT,
                                  sequences_per_block=SEQ_PER_BLOCK, target_shard_bytes=SHARD_BYTES,
                                  configuration_hash=config_h, schema_hash=schema_h, work_plan_hash=plan_h)
    planned_blocks = int(contract['planned_train_blocks'])
    write_json_atomic(out / 'run_contract.json', contract)

    store = None
    if not a.no_upload:
        from kefir.data.src.hf_bucket_shards import HuggingFaceBucketShardStore
        store = HuggingFaceBucketShardStore(a.bucket, token=os.environ.get('HF_TOKEN'),
                                            private=not a.public, create_bucket=True)
        publish_run_contract(store, run_id=a.run_id, contract=contract)
        print(json.dumps({'event': 'contract', 'planned_train_blocks': planned_blocks,
                          'sha256': contract['contract_sha256'][:16]}), flush=True)

    producer = Producer(out, sidecar=not a.no_sidecar, plan_key=plan_h)
    st_path = out / STATE_FILE
    if st_path.exists():
        producer.load_state(json.loads(st_path.read_text()))
        print(json.dumps({'event': 'resume', 'emitted': producer.emitted}), flush=True)

    bins = region_bins(root)
    up_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='upload')
    up_futs: list = []

    def _upload_task(d):
        local = out / d['filename']
        if store is not None:
            def attempt():
                from kefir.data.src.hf_bucket_shards import HuggingFaceBucketShardStore as S
                fresh = S(a.bucket, token=os.environ.get('HF_TOKEN'), private=not a.public,
                          create_bucket=False)
                info = fresh.upload_finalized_shard(run_id=a.run_id, logical_name=d['filename'],
                                                    local_path=local)
                d.update({'remote_durable': True, 'file_id': info['file_id'],
                          'remote_sha256': info['sha256']})
                local.unlink()
            upload_with_retry(attempt, d['filename'])
        producer.finalized.append(d)
        print(json.dumps({'shard': d['filename'], 'blocks': [d['first_block_id'], d['last_block_id']],
                          'bytes': d['byte_size'], 'uploaded': store is not None}), flush=True)
        if store is not None:
            fr = publish_frontier(store, run_id=a.run_id, contract=contract,
                                  durability_manifest={'shards': [x for x in producer.finalized
                                                                  if x.get('remote_durable')]})
            print(json.dumps({'event': 'frontier',
                              'last_ready_train_block_id': fr['last_ready_train_block_id'],
                              'ready_train_shards': len(fr['ready_train_shards']),
                              'validation_ready': fr['validation_ready']}), flush=True)
        return d['filename']

    def durable(rows):
        for m in rows:
            d = m.as_dict() if hasattr(m, 'as_dict') else dict(m)
            up_futs.append(up_pool.submit(_upload_task, d))
        for f in [f for f in up_futs if f.done()]:
            f.result()

    def drain():
        for f in up_futs:
            f.result()
        up_futs.clear()

    def checkpoint(tag):
        rows = [w.finalize_active() for w in producer.writers.values() if w._blocks]
        durable(rows)
        drain()
        write_json_atomic(st_path, producer.state())
        print(json.dumps({'event': 'checkpoint', 'at': tag, 'emitted': producer.emitted,
                          'shards': len(producer.finalized)}), flush=True)

    limits = {'validation': plans['validation']['blocks'],
              'train': min(plans['train']['blocks'], planned_blocks + a.stop_margin_blocks)}
    work: list[tuple[str, int, int]] = []
    for split in SPLITS:
        pos = producer.emitted[split]
        while pos < limits[split]:
            hi = min(pos + a.gather_blocks, limits[split])
            work.append((split, pos, hi))
            pos = hi

    t2 = time.time()
    crashed = False
    next_ckpt = producer.emitted['train'] + a.checkpoint_blocks
    readers: dict[str, BlockReader] = {}
    pool = None
    if a.prefetch > 1 and len(work) > 1:
        pool = ProcessPoolExecutor(max_workers=a.prefetch, initializer=_worker_init,
                                   initargs=(str(plan_dir), bins, a.inject_defect))
    try:
        pending: list = []
        cursor = 0
        i = 0
        while i < len(work):
            if pool is not None:
                while cursor < len(work) and len(pending) < a.prefetch + 1:
                    pending.append(pool.submit(_worker_batch, work[cursor]))
                    cursor += 1
                split, lo, hi, batch = pending.pop(0).result()
            else:
                split, lo, hi = work[i]
                if split not in readers:
                    readers[split] = BlockReader(plan_dir, split, bins, a.inject_defect)
                batch = readers[split].batch(lo, hi)
            i += 1
            durable(producer.consume(split, lo, batch))
            if producer.emitted['train'] >= next_ckpt:
                checkpoint(f"train:{producer.emitted['train']}")
                next_ckpt += a.checkpoint_blocks
                if a.crash_after is not None and producer.emitted['train'] >= a.crash_after:
                    print(json.dumps({'event': 'simulated_crash',
                                      'train_blocks': producer.emitted['train']}), flush=True)
                    crashed = True
                    break
            if i % 64 == 0:
                done = sum(producer.emitted.values())
                print(json.dumps({'split': split, 'blocks': producer.emitted[split],
                                  'ms_per_block': round(1000 * (time.time() - t2) / max(1, done), 2)}),
                      flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    if crashed:
        up_pool.shutdown(wait=True)
        return 3

    for w in producer.writers.values():
        durable(w.finish())
    drain()
    up_pool.shutdown(wait=True)
    seconds = time.time() - t2
    total_blocks = sum(producer.emitted.values())
    horizon_ok = producer.emitted['train'] >= planned_blocks

    manifest = build_manifest(producer, plans, tree, a, tok_sha, plan_h, regions,
                              planned_blocks, horizon_ok, seconds)
    write_json_atomic(out / config.MANIFEST_FILENAME, manifest)

    if store is None:
        rep = verify(out, full_scan=False)
        print(json.dumps({'verify_passed': rep.passed, 'problems': rep.problems,
                          'train_tokens': rep.train_token_count,
                          'validation_tokens': rep.validation_token_count}), flush=True)
        if not rep.passed:
            return 2
    else:
        ready = store.publish_dataset_manifest(run_id=a.run_id,
                                               manifest_path=out / config.MANIFEST_FILENAME)
        fr = publish_frontier(store, run_id=a.run_id, contract=contract,
                              durability_manifest={'shards': [x for x in producer.finalized
                                                              if x.get('remote_durable')]},
                              producer_complete=horizon_ok,
                              final_manifest_sha256=sha256_file(out / config.MANIFEST_FILENAME))
        print(json.dumps({'published': ready, 'frontier_complete': fr['producer_complete']}), flush=True)
    print(json.dumps({'event': 'done', 'blocks': total_blocks, 'documents': producer.documents,
                      'seconds': round(seconds, 1),
                      'ms_per_block': round(1000 * seconds / max(1, total_blocks), 3),
                      'blocks_per_hour': int(3600 * total_blocks / max(1e-9, seconds))}), flush=True)
    return 0


def build_manifest(producer, plans, tree, a, tok_sha, plan_h, regions, planned_blocks,
                   horizon_ok, seconds) -> dict:
    tr = plans['train']
    counts = {int(c): int(n) for c, n in zip(tr['clusters'], tr['counts'])}
    total_seq = sum(counts.values()) or 1
    emitted = producer.emitted
    hist = producer.cluster_hist['train']
    stored = sum(emitted.values()) * SEQ_PER_BLOCK * STORED
    w = producer.writers
    return {
        'schema_version': STREAM_CACHE_SCHEMA_VERSION, 'sequence_format': SEQUENCE_FORMAT,
        'context_length': CONTEXT, 'stored_tokens_per_sequence': STORED,
        'final_partial_sequence_policy': 'pad_eod', 'sequences_per_block': SEQ_PER_BLOCK,
        'target_shard_bytes': SHARD_BYTES,
        'weights': {'supplied': {str(k): v for k, v in producer.scheduler.supplied_weights.items()},
                    'normalized_integer_units': {str(k): v for k, v in producer.scheduler.weight_units.items()}},
        'shards': [dict(s) for s in producer.finalized],
        'scheduler': producer.scheduler.state_dict(),
        'accepted_source_tokens': producer.scheduler.total_emitted_source_tokens + producer.validation_source_tokens,
        'validation_source_tokens': producer.validation_source_tokens,
        'accepted_document_count': producer.documents,
        'mixture': {'emitted_source_tokens_per_cluster':
                    {str(k): v for k, v in producer.scheduler.emitted_source_tokens.items()}},
        'last_durable_block_id': w['train'].shards[-1].last_block_id if w['train'].shards else -1,
        'last_durable_train_block_id': w['train'].shards[-1].last_block_id if w['train'].shards else -1,
        'last_durable_validation_block_id': w['validation'].shards[-1].last_block_id if w['validation'].shards else -1,
        'tokenizer': tokenizer_identity(tok_sha),
        'semantic_vocab_size': VOCAB, 'eod_token_id': EOD, 'source_token_unit': 'superbpe_8000',
        'work_plan_hash': plan_h,
        'production': {
            'run_id': a.run_id, 'target_reached': horizon_ok, 'producer': 'pack_stratified.py',
            'ordering': 'stratified: cluster-pure best-fit records, hierarchical largest-remainder '
                        'cluster quotas per block with round-robin score/length strata, '
                        'level-0 groups interleaved inside the block, global block shuffle',
            'planned_train_blocks': planned_blocks, 'train_blocks': emitted['train'],
            'validation_blocks': emitted['validation'],
            'completion_reason': 'train_horizon_reached' if horizon_ok else 'source_exhausted_before_horizon',
            'seconds': round(seconds, 1),
            'ms_per_block': round(1000 * seconds / max(1, sum(emitted.values())), 3),
            'source_corpus': {'root': str(a.root), 'regions': len(regions),
                              'region_ids': regions if len(regions) <= 64 else [regions[0], regions[-1]]}},
        'remote_transport': {'backend': 'local_only' if a.no_upload else 'hf_bucket',
                             'evict_local_finalized_shards': not a.no_upload,
                             'hf_bucket_id': a.bucket,
                             'hf_bucket_private': None if a.no_upload else (not a.public),
                             'incremental_frontier': True},
        'stratification': {
            'recipe_version': sp.PLAN_VERSION, 'seed': a.seed, 'plan_hash': plan_h,
            'level_1_document_to_sequence': {
                'method': 'best-fit-decreasing bin packing, cluster-pure, capacity 2049 tokens',
                'reference': 'Ding 2024, Fewer Truncations Improve Language Modeling (arXiv 2404.10830)',
                'eod_between_documents': True, 'padding': 'EOD up to 2049',
                'long_document_rule': 'a document longer than 2048 tokens is cut into fragments that are '
                                      'placed in different, non-adjacent emitted blocks',
                'padding_tokens': producer.padding_tokens,
                'padding_fraction': round(producer.padding_tokens / max(1, stored), 6)},
            'level_2_sequence_to_block': {
                'method': 'hierarchical largest-remainder (largest-deficit) cluster quotas over the corpus '
                          'marginal, coarse tree groups first then leaves; inside a cluster the slots walk '
                          'the score x length strata round-robin; the 64 slots interleave the level-0 groups',
                'score_field': sp.SCORE_FIELD, 'score_strata': sp.N_SCORE_STRATA,
                'length_strata': sp.N_LENGTH_STRATA, 'stratum_walk': sp.WALK,
                'quantiles_are_per_cluster': True, 'stratum_bounds': tr['stratum_bounds'],
                'cluster_tree': {'method': tree.get('method'), 'source': tree.get('source'),
                                 'sha256': tree.get('sha256'),
                                 'levels': [[[int(c) for c in g] for g in lvl] for lvl in tree['levels']],
                                 'group_count_per_level': [len(lvl) for lvl in tree['levels']],
                                 'level0_groups': tr['level0_groups'],
                                 'declared_marginal': tree.get('marginal')},
                'scheduler_note': 'TokenDeficitScheduler.emit() is called once per (block, cluster) group '
                                  'with that group source-token total: the per-cluster and total counters '
                                  'are exact; tie_break_counter counts groups, not documents',
                'unfilled_slots': tr['short_slots']},
            'level_3_block_to_stream': {
                'method': 'single global permutation of every planned block, fixed seed, computed before '
                          'block 0 is written so the output is a stream',
                'reference': 'Dai 2026, Demystifying Data Organization (arXiv 2605.30334)',
                'curriculum': None, 'fragment_adjacency_repairs': tr['repairs'],
                'fragment_violations': tr.get('fragment_violations'),
                'validation_fragment_violations': plans['validation'].get('fragment_violations')},
            'cluster_weights': {
                'source': 'corpus marginal computed from the document index; never re-optimised '
                          '(ClimbMix is already an optimised mixture, Diao 2025 arXiv 2504.13161)',
                'sequences_per_cluster': {str(k): v for k, v in counts.items()},
                'sequence_share': {str(k): round(v / total_seq, 6) for k, v in counts.items()},
                'emitted_sequences_per_cluster': {str(i): int(hist[i]) for i in range(len(hist)) if hist[i]}},
            'diversity': {
                'mean_distinct_clusters_per_block': round(producer.distinct_sum['train'] / max(1, emitted['train']), 4),
                'mean_cluster_entropy_nats_per_block': round(producer.entropy_sum['train'] / max(1, emitted['train']), 5),
                'max_possible_entropy_nats': round(float(np.log(max(1, min(SEQ_PER_BLOCK, len(counts))))), 5)},
            'validation': {
                'rule': 'index_identity_v1', 'function': 'kefir.data.src.split.is_validation',
                'seed': config.SELECTION_SEED, 'probability': config.VALIDATION_PROBABILITY,
                'identity': '(SPLIT_HASH_VERSION, SELECTION_SEED, region revision, region source file, '
                            'document token offset inside the region)',
                'deviation': 'pack_shards.py uses the per-document record_start byte offset; the document '
                             'index does not carry it, so a different 0.1% of documents is selected. Same '
                             'function, same seed, same probability, order-independent, resume-stable.',
                'stratified': True},
            'dedup_scope': 'region',
            'dedup_note': 'intra-region exact dedup was done by the corpus builder; the cross-region pass of '
                          'pack_shards.py (sha256 of the jsonl id) is skipped because the document index '
                          'carries no ids. The rate below is measured from the merge receipts that already '
                          'exist, not assumed: it is higher than the 19-in-2.25M figure the brief quoted, '
                          'and it is lumpy (region 48 alone accounts for 1,019 of 1,459).',
            'dedup_measured': sp.merge_dedup_stats(a.root),
            'defect_injected': a.inject_defect},
    }


if __name__ == '__main__':
    raise SystemExit(main())
