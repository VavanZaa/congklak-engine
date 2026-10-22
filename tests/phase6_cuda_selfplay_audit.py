"""Actual CUDA selfplay kills with a stdlib-only process coordinator."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config,phase6_common as common
from phase6_audit_acceptance import fixture_plan,decisions,semantic,load,normalize_literal,json_position,WORK
import reference as literal
EVIDENCE={}

class Phase6CudaSelfplayAcceptance(unittest.TestCase):
    def test_14_cuda_selfplay_actual_pre_postcommit_kills_and_legal_resume(self):
        with tempfile.TemporaryDirectory(dir=WORK) as temporary:
            root=Path(temporary);assert root.resolve().is_relative_to(WORK.resolve());plan=fixture_plan();p=root/'plan.json';p.write_text(config.canonical(plan))
            code="import sys,json;from pathlib import Path;from congklak import phase6_selfplay as sp;sp.run(sys.argv[1],sys.argv[2],'cuda',plan=json.loads(Path(sys.argv[3]).read_text()))"
            def invoke(path):
                completed=subprocess.run([sys.executable,'-c',code,str(path),str(common.WARM),str(p)],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
                return load(path/'manifest.json')
            fresh=root/'fresh';invoke(fresh);expected=semantic(decisions(fresh));proofs=[]
            for event in ('phase6_selfplay_precommit','phase6_selfplay_committed'):
                run=root/event;marker=root/(event+'.json');log=root/(event+'.log')
                childcode="""import sys,json,time
from pathlib import Path
from congklak import phase6_selfplay as sp
def callback(v):
 if v['event']==sys.argv[5] and v.get('committed_decisions',0)>=4:
  Path(sys.argv[4]).write_text(json.dumps(v))
  while True:time.sleep(.1)
sp.run(sys.argv[1],sys.argv[2],'cuda',plan=json.loads(Path(sys.argv[3]).read_text()),on_event=callback)
"""
                with log.open('w',encoding='utf-8') as output:
                    child=subprocess.Popen([sys.executable,'-c',childcode,str(run),str(common.WARM),str(p),str(marker),event],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
                    try:
                        deadline=time.monotonic()+180
                        while not marker.exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.05)
                        self.assertTrue(marker.exists(),log.read_text(encoding='utf-8'));child.kill();child.wait(timeout=10)
                    finally:
                        if child.poll() is None:child.kill();child.wait(timeout=10)
                prefix=decisions(run);self.assertGreaterEqual(len(prefix),4);manifest=invoke(run);after=decisions(run);indexed={(g,c):(b,h) for g,c,b,h in after}
                for g,c,b,h in prefix:self.assertEqual(indexed[g,c],(b,h))
                self.assertEqual(semantic(after),expected)
                with closing(sqlite3.connect(run/'work.sqlite3')) as db:
                    for body, in db.execute('SELECT body FROM sp_games'):
                        game=json.loads(body);q=normalize_literal(game['start'])
                        for row, in db.execute('SELECT body FROM sp_decisions WHERE gameid=? ORDER BY choice',(game['id'],)):q=literal.apply(q,json.loads(row)['action']).position
                        self.assertTrue(literal.normalize(q).terminal);self.assertEqual(json_position(q),game['position']);self.assertEqual(literal.evaluate(q),game['terminal_margin'])
                files={f.name:(hashlib.sha256(f.read_bytes()).hexdigest(),f.stat().st_mtime_ns) for f in run.iterdir() if f.is_file() and not f.name.endswith(('-wal','-shm'))};invoke(run)
                self.assertEqual(files,{f.name:(hashlib.sha256(f.read_bytes()).hexdigest(),f.stat().st_mtime_ns) for f in run.iterdir() if f.is_file() and not f.name.endswith(('-wal','-shm'))})
                durable=WORK/'cuda-selfplay-kills';durable.mkdir(exist_ok=True);(durable/log.name).write_bytes(log.read_bytes());(durable/marker.name).write_bytes(marker.read_bytes())
                proofs.append(dict(event=event,actual_cuda_hard_kill=True,committed_decisions_before=len(prefix),decisions_after=len(after),committed_bytes_preserved=True,fixed_forest_schedule_fresh_semantic_equal=True,literal_terminal_replay=True,finished_bytes_mtime_stable=True,marker=load(marker),log_path=str((durable/log.name).relative_to(ROOT)).replace('\\','/')))
            EVIDENCE['cuda_selfplay_recovery']=dict(tests=proofs,model_file_sha256=hashlib.sha256(common.WARM.read_bytes()).hexdigest(),fixture_games=4,fixture_simulations=8,production_claim=False)

if __name__=='__main__':unittest.main(verbosity=2)
