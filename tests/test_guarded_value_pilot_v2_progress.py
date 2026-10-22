"""Pure observer checks: no HTTP, model, worker, or raw archive access."""
import builtins,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from congklak import guarded_value_pilot_v2_progress as progress

class ProgressChecks(unittest.TestCase):
    def test_snapshot_never_opens_archive_or_imports_torch(self):
        original_open=Path.open;original_import=builtins.__import__
        def safe_open(path,*args,**kwargs):
            self.assertNotIn('archives',path.parts);self.assertNotIn('attempts',path.parts)
            self.assertNotEqual(path.suffix,'.pt');self.assertNotEqual(path.suffix,'.sqlite3')
            self.assertFalse(args and any(c in args[0] for c in 'wax+'))
            return original_open(path,*args,**kwargs)
        def safe_import(name,*args,**kwargs):
            self.assertFalse(name=='torch' or name.startswith('torch.'))
            return original_import(name,*args,**kwargs)
        with patch.object(Path,'open',safe_open),patch.object(builtins,'__import__',safe_import):
            body=progress.snapshot()
        self.assertEqual(len(body['tools']),9);self.assertIs(body['policy_fixed'],True)
        self.assertIs(body['accepted'],False);self.assertIs(body['model_promotion'],False)

    def test_bad_training_cursor_is_visible_and_never_reported_complete(self):
        def compact(path,errors,limit=4*1024*1024):
            if path==progress.s.DATA/'training/catalog.json':return dict(step=999,complete=True,checkpoints=[])
            return None
        with patch.object(progress,'_optional',side_effect=compact):body=progress.snapshot()
        self.assertEqual(body['production_training_steps'],0)
        self.assertTrue(any('invalid committed step' in text for text in body['errors']))
        self.assertEqual(body['training']['state'],'not_started')

    def test_tampered_compact_envelope_is_an_error(self):
        progress.s.FOLDER.joinpath('fixtures').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='pure-observer-',dir=progress.s.FOLDER/'fixtures') as folder:
            path=Path(folder)/'catalog.json'
            path.write_text('{"body":{"step":400},"sha256":"bad"}',encoding='utf-8')
            errors=[];self.assertIsNone(progress._optional(path,errors));self.assertEqual(len(errors),1)

if __name__=='__main__':unittest.main()
