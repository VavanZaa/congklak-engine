"""Finite proof lifetime and byte-tamper tests on synthetic tiny artifacts."""
from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from congklak import guarded_value_pilot_v2_common as stage
from congklak import guarded_value_pilot_v2_job_views as views


class JobViewTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        base = stage.FOLDER / 'fixtures'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='job-view-owner-', dir=base) as folder:
            path = Path(folder).resolve()
            self.assertTrue(path.is_relative_to(base.resolve()))
            closing, raw = path / 'closing.json', path / 'archive.bin'
            closing.write_bytes(b'{"complete":true}\n')
            raw.write_bytes(b'captured raw bytes')
            with patch.object(views, '_dependencies', return_value={closing, raw}), \
                 patch.object(views, '_compact', return_value={closing}):
                yield path, closing, raw

    def test_cold_proof_once_per_scope_and_returned_copies_are_isolated(self):
        with self.fixture() as (path, closing, raw):
            produce = Mock(return_value=dict(publication_gap=False, report=dict(state='complete'), rows=[1, 2]))
            with views.admission_scope() as scope:
                first = views.completed(path, 'arena', produce)
                first['rows'].append(99)
                self.assertEqual(views.completed(path, 'arena', produce)['rows'], [1, 2])
                self.assertEqual((scope.cold, scope.hits), (1, 1))
            self.assertEqual(produce.call_count, 1)
            self.assertTrue(views.admission_receipt()['full_bytes_verified_before_and_after'])
            # A new caller gets a new full validation; tokens never escape scope.
            with views.admission_scope():
                views.completed(path, 'arena', produce)
            self.assertEqual(produce.call_count, 2)

    def test_same_size_same_mtime_raw_tamper_is_caught_before_scope_acceptance(self):
        with self.fixture() as (path, closing, raw):
            produce = Mock(return_value=dict(publication_gap=False, report=dict(state='complete')))
            with self.assertRaisesRegex(ValueError, 'Scoped completed evidence changed'):
                with views.admission_scope():
                    views.completed(path, 'arena', produce)
                    stat = raw.stat()
                    raw.write_bytes(b'X' * stat.st_size)
                    os.utime(raw, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                    # Compact reuse is allowed inside the finite view; its
                    # required closing full-byte proof still refuses this run.
                    views.completed(path, 'arena', produce)

    def test_changed_closing_binding_is_refused_immediately(self):
        with self.fixture() as (path, closing, raw):
            produce = Mock(return_value=dict(publication_gap=False, report=dict(state='complete')))
            with self.assertRaisesRegex(ValueError, 'Scoped completed evidence changed'):
                with views.admission_scope():
                    views.completed(path, 'arena', produce)
                    closing.write_bytes(b'{"complete":false}')
                    views.completed(path, 'arena', produce)

    def test_added_file_and_unissued_token_are_refused(self):
        with self.fixture() as (path, closing, raw):
            produce = Mock(return_value=dict(publication_gap=False, report=dict(state='complete')))
            with self.assertRaisesRegex(ValueError, 'not issued'):
                with views.admission_scope() as scope:
                    views.completed(path, 'arena', produce)
                    valid = scope.tokens[(path, 'arena')]
                    scope.tokens[(path, 'arena')] = views._VerifiedJob(valid.result, valid.compact, valid.full,
                        valid.inventory, path, 'arena', object())
                    views.completed(path, 'arena', produce)
            with self.assertRaisesRegex(ValueError, 'physical inventory changed'):
                with views.admission_scope():
                    views.completed(path, 'arena', produce)
                    (path / 'extra.json').write_bytes(b'{}')
                    views.completed(path, 'arena', produce)

    def test_prior_scope_token_cannot_replace_a_new_cold_validation(self):
        with self.fixture() as (path, closing, raw):
            produce = Mock(return_value=dict(publication_gap=False, report=dict(state='complete')))
            with views.admission_scope() as first:
                views.completed(path, 'arena', produce)
                old = first.tokens[(path, 'arena')]
            with self.assertRaisesRegex(ValueError, 'not issued'):
                with views.admission_scope() as second:
                    second.tokens[(path, 'arena')] = old
                    views.completed(path, 'arena', produce)


if __name__ == '__main__':
    unittest.main()
