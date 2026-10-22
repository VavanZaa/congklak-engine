"""Independent review of author evidence, without claiming a new CUDA kill."""
import ast
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config
from congklak.neural_relay import Model
from phase6_audit_acceptance import sha,load,EVIDENCE

class Phase6AuthorWarmEvidenceReview(unittest.TestCase):
    def test_09_independent_author_warm_cuda_recovery_evidence_review(self):
        path=ROOT/'build/phase6-training-acceptance.json';record=load(path)
        source=ROOT/'tests/phase6_training_acceptance.py';trainer=ROOT/'congklak/phase6_training.py'
        self.assertEqual(record['source_sha256'],sha(source));self.assertEqual(record['trainer_sha256'],sha(trainer))
        self.assertTrue(record['passed']);self.assertEqual(record['tests'],6);self.assertEqual(record['failures']+record['errors']+record['skipped'],0)
        self.assertTrue(record['real_CUDA_precommit_kill']);self.assertTrue(record['real_CUDA_final_commit_pre_export_kill']);self.assertTrue(record['all_state_recovery_bitwise'])
        self.assertEqual([(r['kind'],r['restored_step'],r['steps']) for r in record['records']],[('precommit',10,30),('preexport',20,20)])
        for r in record['records']:self.assertTrue(r['actual_cuda_kill'] and r['full_state_bitwise'] and r['completed_bytes_and_mtimes_unchanged'])
        text=source.read_text();parsed=ast.parse(text)
        self.assertTrue(any(isinstance(n,ast.FunctionDef) and n.name=='isolated_driver' for n in ast.walk(parsed)))
        # Source review checks that the author's reported record is backed by
        # actual serial subprocess barriers, kills and whole-state assertions.
        for token in ('subprocess.Popen','process.kill()','checkpoint_file_ready','training_final_checkpoint_ready','m.equal(m.latest(a),m.latest(b))','len(state[\'optimizer\'][\'state\'])==2','snapshot(resumed)==before'):
            self.assertIn(token,text)
        run=ROOT/'build/phase6-training-acceptance/serial-cuda-warm-fixture';model=Model(run/'model.json');result=load(run/'result.json');identity=model.identity
        self.assertEqual(identity['source_sha256'],sha(trainer));self.assertEqual(identity['stage'],'warm');self.assertEqual(result['status'],'complete');self.assertEqual(result['steps_completed'],20)
        self.assertEqual(result['weights_sha256'],model.id);self.assertEqual(result['best_step'],model.envelope['model']['step'])
        digest=hashlib.sha256()
        for name,value in sorted(model.weights.items()):
            if not name.startswith('layers.'):continue
            flat=[v for row in value for v in row] if isinstance(value[0],list) else value
            digest.update(name.encode());digest.update(struct.pack('<'+'f'*len(flat),*flat))
        self.assertEqual(digest.hexdigest(),'29f90512e59ebaeb93eedf74d247684161e6b211a9b21640dc4f9333d213d69f')
        files=[source,trainer,path,run/'model.json',run/'result.json',run/'catalog.sqlite3'];history=[]
        with closing(sqlite3.connect((run/'catalog.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
            configbody,configsha=db.execute('SELECT body,hash FROM config').fetchone();self.assertEqual(json.loads(configbody),identity);self.assertEqual(config.digest(identity),configsha)
            for step,name,checksum,body,bodyhash in db.execute('SELECT step,name,sha,body,hash FROM checkpoints ORDER BY step'):
                meta=json.loads(body);self.assertEqual(config.digest(meta),bodyhash);file=run/name;self.assertEqual(sha(file),checksum);files.append(file)
                history.append((step,meta['metrics']['policy']['uniform_stratum_teacher_cross_entropy']))
        self.assertEqual([s for s,v in history],[10,20]);self.assertEqual(min(history,key=lambda r:r[1])[0],result['best_step'])
        failure1=load(ROOT/'build/phase6-audit/warm-resource-failure.json');failure2=load(ROOT/'build/phase6-audit/warm-batch32-host-memory-failure.json')
        self.assertFalse(failure1['passed']);self.assertFalse(failure2['passed']);files += [ROOT/'build/phase6-audit/warm-resource-failure.json',ROOT/'build/phase6-audit/warm-batch32-host-memory-failure.json']
        EVIDENCE['author_warm_recovery_evidence_review']=dict(author_suite_reported_actual_cuda_kills=2,author_source_and_record_hashes_verified=True,
            completed_standalone_fixture_catalog_and_export_verified=True,standalone_fixture_is_not_kill_proof=True,raw_successful_kill_fixtures_retained=False,
            independent_raw_killed_state_recomparison=False,independent_live_warm_kill_passed=False,
            limitation='Successful author kill fixtures were temporary and removed; review verifies source/published record and a separate complete20-step fixture, not raw fresh/resumed states. Both independent live warm attempts failed before kill markers under resource pressure.',
            inputs={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files})
