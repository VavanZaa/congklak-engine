"""Meaningful persistence adversaries using no engine or model imports."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from congklak import benchmark_loss_review_v1 as review


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.binding = dict(path='manifest.json', sha256='a'*64)

    def receipt(self, index, previous=None):
        return dict(schema=review.SCHEMA+'-game-receipt', manifest_ref=self.binding,
                    index=index, previous_sha256=previous, result=dict(game=index))

    def save(self, body):
        data = review.canonical(body)
        h = hashlib.sha256(data).hexdigest()
        p = self.output/'receipts'/f"{body['index']:03d}-{h}.json"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(data)
        return p

    def test_contiguous_prefix_and_no_duplicate(self):
        first = self.save(self.receipt(0))
        self.save(self.receipt(1, review.sha(first)))
        self.assertEqual(len(review.load_receipts(self.output, self.binding)), 2)
        self.save(self.receipt(1, 'wrong'))
        with self.assertRaises(ValueError):
            review.load_receipts(self.output, self.binding)

    def test_missing_committed_row_fails_closed(self):
        first = self.save(self.receipt(0))
        self.save(self.receipt(1, review.sha(first)))
        first.unlink()
        with self.assertRaises(ValueError):
            review.load_receipts(self.output, self.binding)

    def test_receipt_corruption_fails_closed(self):
        first = self.save(self.receipt(0))
        first.write_bytes(first.read_bytes().replace(b'"game":0', b'"game":1'))
        with self.assertRaises(ValueError):
            review.load_receipts(self.output, self.binding)

    def test_stale_manifest_fails_closed(self):
        self.save(self.receipt(0))
        with self.assertRaises(ValueError):
            review.load_receipts(self.output, dict(self.binding, sha256='b'*64))

    def test_truncated_published_tail_fails_closed(self):
        first = self.save(self.receipt(0))
        (self.output/'progress.json').write_bytes(review.canonical(dict(completed=1,
            manifest_ref=self.binding, last_committed_sha256=review.sha(first))))
        first.unlink()
        with self.assertRaises(ValueError):
            review.load_receipts(self.output, self.binding)

    def test_rehashed_semantic_cache_substitution_fails_closed(self):
        ref = dict(path='archive.json', sha256='c'*64)
        state = dict(pits=[[0]*7,[0]*7], stores=[45,53], controller=0)
        audited = dict(root_state=state, profile='candidate', baseline='H_v1', seat=0,
                       margin=-8, final=state, choices=1)
        result = dict(game=0, root_state=state, profile='candidate', baseline='H_v1', seat=0,
                      margin=8, final=state, decisions=1, moves=[dict(archive_ref=ref)])
        body = self.receipt(0)
        body.update(input_archive_refs=[ref], result=result)
        self.save(body)
        manifest = dict(games=[dict(archives=[ref], audited=audited,
            schedule=dict(profile='candidate', baseline='H_v1', seat=0, meta=dict(state=state)))])
        with self.assertRaises(ValueError):
            review.load_receipts(self.output, self.binding, manifest)

    def test_changed_input_bytes_fails_even_with_same_length(self):
        p = review.ROOT/'build/benchmark-loss-review-v1/analysis/fixtures/input-hash.bin'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'abc')
        ref = review.reference(p)
        p.write_bytes(b'xyz')
        with self.assertRaises(ValueError):
            review.checked(ref)

    def test_atomic_immutable_existing_bytes_not_rewritten(self):
        p = review.HOME/'fixtures/atomic-immutable.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            p.unlink()
        review.atomic(p, dict(a=1))
        before = (p.read_bytes(), p.stat().st_mtime_ns)
        review.atomic(p, dict(a=1))
        self.assertEqual((p.read_bytes(), p.stat().st_mtime_ns), before)
        with self.assertRaises(ValueError):
            review.atomic(p, dict(a=2))

    def test_uncommitted_pending_file_is_never_a_receipt(self):
        directory = self.output/'receipts'
        directory.mkdir()
        (directory/'000-incomplete.pending').write_text('{')
        self.assertEqual(review.load_receipts(self.output, self.binding), [])


if __name__ == '__main__':
    unittest.main()
