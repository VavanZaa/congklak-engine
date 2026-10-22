"""Read shared immutable evidence; live observations avoid mutable targets."""
import json
import os
from pathlib import Path
import unittest
import uuid
from unittest import mock
from congklak.value_run_reader import open_shared, read
from congklak import value_run_progress_v3 as progress


class SharedReaderTests(unittest.TestCase):
    def test_shared_immutable_evidence_read(self):
        root = Path(__file__).resolve().parents[1] / 'build/value-run-v1'
        folder = root / ('reader-fixture-' + uuid.uuid4().hex[:8])
        folder.mkdir()
        path = folder / 'immutable.json'
        path.write_text('{"version":1}', encoding='utf-8')
        with open_shared(path) as old:
            self.assertEqual(json.loads(old.read()), {'version': 1})
            self.assertEqual(read(path), {'version': 1})

    def test_unfinished_observer_never_opens_replaceable_worker_files(self):
        root = Path(__file__).resolve().parents[1] / 'build/value-run-v1'
        folder = root / ('reader-live-fixture-' + uuid.uuid4().hex[:8]); folder.mkdir()
        for name in ('games', 'decisions'): (folder / name).mkdir()
        (folder / 'games/000.json').write_text('{}')
        (folder / 'decisions/000.json').write_text('{}')
        (folder / 'progress.json').write_text('{"do_not_read":true}')
        (folder / 'head.json').write_text('{"do_not_read":true}')
        with mock.patch.object(progress, 'read', side_effect=AssertionError('Mutable worker file opened')):
            observed = progress.arena_live(folder, certified=dict(games=0, decisions=0))
        self.assertEqual(observed['physical_saved_game_files'], 1)
        self.assertEqual(observed['last_verified_committed']['games'], 0)
        self.assertTrue(observed['physical_counts_are_provisional'])
        pending = folder / 'progress.pending'; pending.write_text('{"version":2}')
        os.replace(pending, folder / 'progress.json')


if __name__ == '__main__': unittest.main()
