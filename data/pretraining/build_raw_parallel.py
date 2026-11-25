"""Build the raw Kefir v1 corpus from pinned GPT-2-tokenized NVIDIA regions.

Each worker filters, decodes and re-encodes its own regions. Deduplication is
per region, as in the run 003 source. Validation overlap is checked against the
frozen seed hashes bundled in assets/validation-overlap.json.gz. Output is a
flat SuperBPE-v2 uint16 stream with one EOD 7992 after each document, plus
metadata and receipts consumed by build_doc_index.py and pack_stratified.py.
"""
import argparse
import array
import collections
import gzip
import hashlib
import io
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if 'TIKTOKEN_CACHE_DIR' not in os.environ and (ROOT/'tiktoken-cache').is_dir():
    os.environ['TIKTOKEN_CACHE_DIR'] = str(ROOT/'tiktoken-cache')

from kefir.data import config
from kefir.data.src.bytesource import HttpRangeReader
from kefir.data.src.workplan import load_work_plan
from kefir.data.src.records import iter_owned_records, validate_record
from _helpers import digest, filehash, write_json, paragraphs
import filters_v4_adapter
from student_gate import require_student_asset

EOD = 7992
REGION_BYTES = 256 * 1024 * 1024
SEED_NAME = 'english-next-20260913-v1'
SOURCE_PIN = ('nvidia/Nemotron-ClimbMix', '5eaa64b9c0c85b7f56af01d7dffdb0795816b12b')

def exclusive_lock(path):
    f = path.open('w')
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        raise RuntimeError(f'Another builder holds {path}: {error}') from error
    return f

def parse_regions(spec):
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out

def load_filter():
    mod = filters_v4_adapter
    source = Path(mod.__file__)
    name = 'filters_v4_adapter'
    info = {'name': name, 'file': source.name, 'sha256': filehash(source),
            'version': mod.VERSION, 'second_pass': name + '.reject_reason'}
    return mod, mod.reject_reason, info

def load_validation_overlap(path):
    # Hashes were derived from the frozen seed validation JSONL with filters.normalized.
    if filehash(path) != '8c69271d0f001b07a38fa3f384d163085d841b57467a20c6171a38ff18ef2b27':
        raise RuntimeError('Frozen validation overlap asset drift')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        data = json.load(stream)
    if data.get('version') != 1 or not data.get('source_validation_files'):
        raise RuntimeError('Frozen validation overlap asset invalid')
    return data

def open_gzip_text(path):
    """gzip text writer with fixed header (mtime=0, no FNAME) so bytes are reproducible."""
    raw = open(path, 'wb')
    gz = gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=1, mtime=0)
    return io.TextIOWrapper(gz, encoding='utf-8'), gz, raw

def build_region(item,plan,source_files,out,tools,total_hint,started,state_extra):
    """Produce one region with the original filtering and token boundaries."""
    tok, gpt2, filt, second_pass, normalized, val, val_paras, db = tools
    stem = f'region-{item.index:06d}'
    texts = out / (stem + '.jsonl.gz')
    binary = out / (stem + '.bin')
    for path in (texts, binary):
        if path.exists():
            path.rename(path.with_name(path.name + f'.orphan-{time.time_ns()}'))
    counts = collections.Counter()
    target_count = 0
    source_hash = hashlib.sha256()
    seen = set()
    t0 = time.time()
    reader = HttpRangeReader(source_files[item.filename], plan.dataset, plan.revision)
    textout, gz, rawf = open_gzip_text(str(texts) + '.part')
    with open(str(binary) + '.part', 'wb') as binout:
        for rec in iter_owned_records(item, reader):
            counts['input_documents'] += 1
            source_hash.update(rec.raw + b'\n')
            v = validate_record(rec)
            reason = None
            if not v.valid:
                reason = 'source_schema_' + str(v.rejection_reason)
            elif v.cluster_id == 11:
                reason = 'programming_cluster'
            else:
                ids = list(v.tokens)
                if ids and ids[-1] == 50256:
                    ids.pop()
                if 50256 in ids:
                    reason = 'embedded_source_eod'
                else:
                    try:
                        text = gpt2.decode_bytes(ids).decode('utf-8', errors='strict')
                    except (ValueError, KeyError, UnicodeDecodeError):
                        reason = 'source_decode_error'
                if not reason:
                    reason, info = filt.assess(text)
                    reason = reason or second_pass(text)
                    if not reason:
                        h = digest(normalized(text))
                        if h in val or paragraphs(text) & val_paras:
                            reason = 'validation_overlap'
                        elif h in seen:
                            reason = 'exact_duplicate'
            if reason:
                counts['reject_' + reason] += 1
                continue
            new = tok.encode(text, add_special_tokens=False).ids
            if not new or any(token >= EOD for token in new):
                raise RuntimeError('Invalid source token IDs')
            if tok.decode(new, skip_special_tokens=False) != text:
                raise RuntimeError('Roundtrip failure')
            packed = array.array('H', new + [EOD])
            if sys.byteorder != 'little':
                packed.byteswap()
            binout.write(packed.tobytes())
            metadata = {
                'text': text, 'id': h, 'source_revision': plan.revision,
                'source_file': item.filename, 'record_start': rec.record_start,
                'cluster_id': v.cluster_id, 'source_token_count': len(ids),
                'target_token_count': len(new), 'token_offset_in_shard': target_count,
                **info,
            }
            textout.write(json.dumps(metadata, ensure_ascii=False) + '\n')
            seen.add(h)
            target_count += len(new) + 1
            counts['accepted_documents'] += 1
            counts['accepted_utf8_bytes'] += len(text.encode())
            counts['cluster_' + str(v.cluster_id)] += 1
            if counts['input_documents'] % 1000 == 0:
                write_json(out / 'state.json', {
                    'phase': 'filtering_tokenizing', 'pid': os.getpid(),
                    'region': item.index, 'committed_target_tokens': total_hint,
                    'inflight_target_tokens': target_count, 'counts': dict(counts),
                    'elapsed_seconds': time.time() - started,
                    'updated_at': time.time(), **state_extra,
                })
        binout.flush()
        os.fsync(binout.fileno())
    textout.flush()
    gz.close()
    rawf.flush()
    os.fsync(rawf.fileno())
    rawf.close()
    Path(str(texts) + '.part').replace(texts)
    Path(str(binary) + '.part').replace(binary)
    receipt = {
        'index': item.index, 'source_region': item.work_item_id,
        'owned_record_stream_sha256': source_hash.hexdigest(),
        'texts_sha256': filehash(texts), 'tokens_sha256': filehash(binary),
        'target_tokens': target_count, 'counts': dict(counts), 'dedup_scope': 'region',
    }
    db.execute('INSERT INTO regions VALUES (?,?)', (item.index, json.dumps(receipt)))
    db.commit()
    write_json(out / (stem + '.receipt.json'), receipt)
    seconds = time.time() - t0
    write_json(out / (stem + '.timing.json'), {
        'index': item.index, 'seconds': seconds, 'target_tokens': target_count,
        'target_tokens_per_second': target_count / seconds if seconds else None,
        'process_cpu_seconds_so_far': time.process_time(), 'finished_at': time.time(),
    })
    print(json.dumps({
        'region': item.index, 'seconds': round(seconds, 1),
        'target_tokens': target_count,
        'tokens_per_s': round(target_count / seconds) if seconds else None,
        'counts': dict(counts),
    }), flush=True)
    return receipt

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker', type=int, required=True)
    ap.add_argument('--workers', type=int, required=True)
    ap.add_argument('--regions', default='', help='explicit plan indices, e.g. 0,5,9-12')
    ap.add_argument('--max-regions', type=int, default=0)
    ap.add_argument('--target-tokens', type=int, default=13_107_200_000,
                    help='recorded target; workers process complete assigned regions')
    ap.add_argument('--seed', type=Path, default=ROOT / 'assets')
    ap.add_argument('--work-plan', type=Path, default=ROOT / 'assets/work-plan.json')
    ap.add_argument('--plan-hash', default='')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--min-free-gib', type=float, default=100.0)
    args = ap.parse_args(argv)
    if args.workers <= 0 or not 0 <= args.worker < args.workers:
        raise ValueError('Require 0 <= worker < workers')
    if args.max_regions < 0 or args.target_tokens <= 0:
        raise ValueError('Require nonnegative region limit and positive token target')
    out = args.out or ROOT / f'corpus-13b-v2/worker-{args.worker}'
    out.mkdir(parents=True, exist_ok=True)
    lock = exclusive_lock(out / 'build.lock')
    try:
        run(args, out)
    finally:
        lock.close()

def run(a,out):
    from verify_assets import verify_sources
    verify_sources(ROOT)
    seed = a.seed
    tokpath = ROOT.parents[1] / 'src/kefir/assets/superbpe_8000_v2.json'
    import fasttext
    import tiktoken
    from tokenizers import Tokenizer
    evaluation = json.loads((seed / 'evaluation.json').read_text())
    if evaluation['tokenizer_sha256'] != filehash(tokpath):
        raise RuntimeError('Tokenizer identity drift')
    filter_contract = json.loads((seed / 'contract.json').read_text())
    filter_module, second_pass, filter_info = load_filter()
    filter_info['seed_contract_check'] = 'skipped_module_differs_from_seed_filter'
    provenance = json.loads((ROOT / 'assets/FILTER-PROVENANCE.json').read_text())
    source_names = {
        'filters_v4.py': 'filters_v4.py',
        'assets/students.pkl': 'teacher/student/students.pkl',
    }
    require_student_asset()
    for local_name, historical_name in source_names.items():
        if filehash(ROOT / local_name) != provenance['sha256'][historical_name]:
            raise RuntimeError('Filter dependency drift: ' + local_name)
    lid = ROOT / 'assets/lid.176.bin'
    if filehash(lid) != filter_contract['language_model_sha256']:
        raise RuntimeError('Language model drift')
    if (config.DATASET_REPOSITORY, config.DATASET_REVISION) != SOURCE_PIN:
        raise RuntimeError('Source pin drift')
    partition = {'regions': a.regions} if a.regions else {'worker': a.worker, 'workers': a.workers}
    overlap_path = seed / 'validation-overlap.json.gz'
    overlap = load_validation_overlap(overlap_path)
    contract = {
        'source': config.DATASET_REPOSITORY,
        'revision': config.DATASET_REVISION,
        'license': 'CC-BY-NC-4.0',
        'filter': filter_contract,
        'filter_module': filter_info,
        'tokenizer_sha256': filehash(tokpath),
        'builder_sha256': filehash(Path(__file__)),
        'builder': 'build_raw_parallel.py',
        'partition': partition,
        'dedup_scope': 'region; no cross-region replay in run 003',
        'target_tokens_including_EOD': a.target_tokens,
        'seed': SEED_NAME,
        'current_run_changed': False,
        'output_format': 'uint16 little endian flat token stream; EOD7992, document ends in metadata; NOT schema-v2 training blocks',
        'eval_overlap_guard': 'normalized exact docs and paragraphs>=40words from seedvalidation; semantic/near duplicate leakage not certified',
        'validation_inputs': overlap['source_validation_files'],
        'validation_overlap_sha256': filehash(overlap_path),
        'refinement_sha256': filter_contract['refinement_sha256'],
    }
    contract_path = out / 'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError('Output contract drift')
    write_json(contract_path, contract)
    plan_path = out / 'work-plan.json'
    if plan_path.exists():
        plan = load_work_plan(plan_path)
    else:
        if not a.work_plan.is_file():
            raise FileNotFoundError(a.work_plan)
        shutil.copyfile(a.work_plan, plan_path.with_suffix('.json.tmp'))
        plan_path.with_suffix('.json.tmp').replace(plan_path)
        plan = load_work_plan(plan_path)
    if (plan.dataset, plan.revision, plan.selection_seed, plan.region_bytes) != (
        contract['source'], contract['revision'], contract['seed'], REGION_BYTES
    ):
        raise RuntimeError('Work plan identity drift')
    if a.plan_hash and plan.hash != a.plan_hash:
        raise RuntimeError('Work plan hash differs from --plan-hash')
    plan_pin = out / 'plan-pin.json'
    if plan_pin.exists():
        if json.loads(plan_pin.read_text())['hash'] != plan.hash:
            raise RuntimeError('Work plan hash drift')
    else:
        write_json(plan_pin, {'hash': plan.hash})
    source_files = {source.path: source for source in plan.source_files}
    payload = json.loads(tokpath.read_text())
    payload['added_tokens'] = []
    payload['model']['vocab'] = {key: value for key, value in payload['model']['vocab'].items() if value < EOD}
    tokenizer = Tokenizer.from_str(json.dumps(payload))
    gpt2 = tiktoken.get_encoding('gpt2')
    prose_filter = filter_module.ProseFilter(fasttext.load_model(str(lid)))
    db = sqlite3.connect(out / 'index.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS regions (idx INTEGER PRIMARY KEY, receipt TEXT)')
    db.commit()
    validation_ids = set(overlap['ids'])
    validation_paragraphs = set(overlap['paragraph_sha256'])
    completed = {idx: json.loads(receipt) for idx, receipt in db.execute('SELECT idx,receipt FROM regions')}
    for idx, receipt in completed.items():
        for suffix, key in (('.jsonl.gz', 'texts_sha256'), ('.bin', 'tokens_sha256')):
            path = out / (f'region-{idx:06d}' + suffix)
            if not path.exists() or filehash(path) != receipt[key]:
                raise RuntimeError('Committed artifact missing or corrupt: ' + str(path))
    wanted = parse_regions(a.regions) if a.regions else None
    mine = [item for item in plan.work_items
            if (item.index in wanted if wanted else item.index % a.workers == a.worker)
            and (not a.max_regions or item.index < a.max_regions)]
    total = sum(receipt['target_tokens'] for receipt in completed.values())
    started = time.time()
    extra = {'partition': partition, 'assigned_regions': len(mine)}
    tools = (tokenizer, gpt2, prose_filter, second_pass, filter_module.normalized,
             validation_ids, validation_paragraphs, db)
    for item in mine:
        if item.index in completed:
            continue
        if shutil.disk_usage(out).free < a.min_free_gib * 1024**3:
            raise RuntimeError(f'Less than {a.min_free_gib} GiB free; worker paused')
        receipt = build_region(item, plan, source_files, out, tools, total, started, extra)
        total += receipt['target_tokens']
        completed[item.index] = receipt
        write_json(out / 'state.json', {
            'phase': 'building', 'pid': os.getpid(), 'completed_regions': len(completed),
            'committed_target_tokens': total, 'elapsed_seconds': time.time() - started,
            'updated_at': time.time(), **extra,
        })
    done = all(item.index in completed for item in mine)
    write_json(out / 'manifest.json', {
        'worker_complete': done, 'partition': partition, 'max_regions': a.max_regions,
        'target_tokens': total, 'contract_sha256': filehash(contract_path),
        'regions': [completed[key] for key in sorted(completed)],
        'order': 'per-worker subset of work-plan order', 'merge_required': False,
    })
    write_json(out / 'state.json', {
        'phase': 'complete' if done else 'partial', 'pid': os.getpid(),
        'committed_target_tokens': total, 'completed_regions': len(completed),
        'elapsed_seconds': time.time() - started, 'updated_at': time.time(),
        'worker_complete': done, **extra,
    })
    db.close()

if __name__ == '__main__':
    main()
