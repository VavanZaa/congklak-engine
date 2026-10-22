"""Physical subprocess regressions for non-object JSON diagnostic lines."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, ordinary_value_review_progress as progress
from congklak.ordinary_value_review_common import pins, publish

FIXTURE = ROOT / 'build/ordinary-value-review-progress-author-fixtures' / uuid.uuid4().hex


class OrdinaryValueReviewProgressAuthor(unittest.TestCase):
    def child(self, code, expected):
        folder = FIXTURE / str(expected)
        marker = folder / 'child-finished.txt'
        with patch.object(progress, 'FOLDER', folder), contextlib.redirect_stdout(io.StringIO()):
            result = progress.run_command([sys.executable, '-c', code, str(marker)], 'scalar-diagnostic-child')
        self.assertEqual(result, expected)
        self.assertEqual(marker.read_text(), 'complete')
        rows = list(folder.glob('run-*.json')); self.assertEqual(len(rows), 1)
        row = json.loads(rows[0].read_text())
        self.assertEqual(row['returncode'], expected)
        self.assertEqual(row['state'], 'complete' if expected == 0 else 'failed')
        self.assertIsInstance(row['last_event'], str)
        return row, (ROOT / row['log']).read_text()

    def test_01_scalar_diagnostics_do_not_terminate_a_successful_child(self):
        code = "import json,pathlib,sys,time\nprint('1.23456789e-15',flush=True)\nprint('null',flush=True)\nprint('false',flush=True)\ntime.sleep(.05)\npathlib.Path(sys.argv[1]).write_text('complete')\nprint(json.dumps({'event':'child_finished'}),flush=True)"
        row, log = self.child(code, 0)
        self.assertEqual(row['last_event'], 'child_finished')
        self.assertTrue(all(value in log for value in ('1.23456789e-15','null','false','child_finished')))
        self.assertNotIn('torch', sys.modules)

    def test_02_nonobject_diagnostics_preserve_failed_child_exit_and_full_log(self):
        code = "import pathlib,sys,time\nprint('[1,2]',flush=True)\nprint('{\"event\":9}',flush=True)\nprint('\"trace string\"',flush=True)\nprint('7',flush=True)\ntime.sleep(.05)\npathlib.Path(sys.argv[1]).write_text('complete')\nprint('expected failure diagnostic',file=sys.stderr,flush=True)\nsys.exit(7)"
        row, log = self.child(code, 7)
        self.assertTrue(all(value in log for value in ('[1,2]','{"event":9}','"trace string"','expected failure diagnostic')))
        self.assertIn('expected failure diagnostic', row['last_event'])
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OrdinaryValueReviewProgressAuthor)
    methods = [test.id() for test in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    body = dict(schema='ordinary-value-review-progress-author-v1', passed=result.wasSuccessful(), tests_run=result.testsRun,
                covered_methods=methods, errors=len(result.errors), failures=len(result.failures), skips=len(result.skipped),
                fixture_path=FIXTURE.relative_to(ROOT).as_posix(),
                inputs=pins(['tests/ordinary_value_review_progress_author.py','congklak/ordinary_value_review_progress.py',
                             'tools/ordinary-value-review-dashboard.html','congklak/ordinary_value_review_common.py',
                             'build/ordinary-value-review-baseline.json','docs/ORDINARY_VALUE_REVIEW_SCOPE.md']))
    publish(ROOT / 'build/ordinary-value-review-progress-author.json', body)
    print(config.canonical(dict(event='ordinary_value_review_progress_author', passed=body['passed'], tests_run=result.testsRun)), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
