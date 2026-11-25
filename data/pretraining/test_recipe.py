"""Small fixed-output checks for the run 003 corpus boundary."""

import hashlib
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import build_raw_parallel as builder
from _helpers import stable_hash
from normalization import normalized
from verify_assets import verify_sources


class FakeGPT2:
    texts = {(1, 2): b'Hello science', (3, 4): b'Hold out', (5, 6): b'BAD text'}

    def decode_bytes(self, tokens):
        return self.texts[tuple(tokens)]


class FakeTokenizer:
    encodings = {'Hello science': [10, 11, 12], 'Hold out': [13, 14], 'BAD text': [15, 16]}

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=self.encodings[text])

    def decode(self, tokens, skip_special_tokens=False):
        return next(text for text, ids in self.encodings.items() if ids == tokens)


class FakeFilter:
    def assess(self, text):
        if text == 'BAD text':
            return 'bad_shape', {}
        return None, {'st_reasoning_dense': 0.42}


def record(raw, offset, cluster, tokens, *, valid=True):
    checked = SimpleNamespace(valid=valid, cluster_id=cluster, tokens=tokens,
                              rejection_reason=None if valid else 'schema')
    return SimpleNamespace(raw=raw, record_start=offset, validated=checked)


class CorpusRecipeTest(unittest.TestCase):
    def test_curated_sources_match_the_recorded_recipe(self):
        self.assertGreaterEqual(verify_sources(), 10)

    def test_adapter_and_student_threshold_drift_is_refused(self):
        for name in ('filters_v4_adapter.py', 'student_gate.py'):
            with self.subTest(source=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'assets').mkdir()
                source = (builder.ROOT / name).read_bytes()
                (root / name).write_bytes(source)
                (root / 'assets/manifest.json').write_text(json.dumps({
                    'curated_source_sha256': {name: hashlib.sha256(source).hexdigest()},
                }))
                self.assertEqual(verify_sources(root), 1)
                (root / name).write_bytes(source + b'\nLANG_MEAN_MIN = 0.01\n')
                with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                    verify_sources(root)

    def test_region_bytes_and_decisions(self):
        records = [
            record(b'a', 0, 1, [1, 2, 50256]),
            record(b'b', 1, 1, [1, 2, 50256]),
            record(b'c', 2, 1, [3, 4]),
            record(b'd', 3, 1, [5, 6]),
            record(b'e', 4, 1, [50256, 1]),
            record(b'f', 5, 11, [1, 2]),
            record(b'g', 6, None, [], valid=False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            db = sqlite3.connect(out / 'index.sqlite')
            db.execute('CREATE TABLE regions (idx INTEGER PRIMARY KEY, receipt TEXT)')
            tools = (FakeTokenizer(), FakeGPT2(), FakeFilter(), lambda text: None,
                     normalized, {builder.digest(normalized('Hold out'))}, set(), db)
            with patch.object(builder, 'HttpRangeReader', return_value=object()), \
                 patch.object(builder, 'iter_owned_records', return_value=iter(records)), \
                 patch.object(builder, 'validate_record', side_effect=lambda rec: rec.validated):
                receipt = builder.build_region(
                    SimpleNamespace(index=0, filename='part_0', work_item_id='regionid'),
                    SimpleNamespace(dataset='nvidia', revision='pin'),
                    {'part_0': object()}, out, tools, 0, time.time(), {},
                )
            db.close()
            self.assertEqual(receipt['target_tokens'], 4)
            self.assertEqual(receipt['counts']['accepted_documents'], 1)
            self.assertEqual(receipt['counts']['reject_exact_duplicate'], 1)
            self.assertEqual(receipt['counts']['reject_validation_overlap'], 1)
            self.assertEqual(receipt['counts']['reject_bad_shape'], 1)
            self.assertEqual(receipt['counts']['reject_embedded_source_eod'], 1)
            self.assertEqual(receipt['counts']['reject_programming_cluster'], 1)
            self.assertEqual(receipt['counts']['reject_source_schema_schema'], 1)
            self.assertEqual(receipt['tokens_sha256'],
                             '4e193b13c2fd0ef1ac7f13b3d241b848ee2455f6ff0825a7f40425620ca3bb00')
            self.assertEqual(receipt['texts_sha256'],
                             'bdde5ac60cc712c4783826edba57d8a0dc535a2a7368e26260bf8ec9bfc53344')

    def test_frozen_overlap_and_canonical_hash(self):
        overlap = builder.load_validation_overlap(builder.ROOT / 'assets/validation-overlap.json.gz')
        self.assertEqual(len(overlap['ids']), 11671)
        self.assertEqual(len(overlap['paragraph_sha256']), 53244)
        self.assertEqual(stable_hash({'v': 3, 'seed': 'kefir', 'regions': [0, 1, 2], 'split': 'train'}),
                         '56015784dcca02ae80878f1bdebd9c87b893bfefc810a535bae0d1735fc13310')


if __name__ == '__main__':
    unittest.main()
