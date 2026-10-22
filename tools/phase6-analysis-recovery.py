"""Actual native PUCT recommendation kill/recompute and completed-file checks."""
from __future__ import annotations
from contextlib import closing
import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import phase6_common as common,phase6_analysis as analyzer,config

OUTPUT=ROOT/'build/phase6-analysis-recovery.json';RUN=ROOT/'data/phase6-analysis-demo';FRESH=ROOT/'build/phase6-analysis-recovery/fresh'

def files(folder):return {str(p.relative_to(folder)).replace('\\','/'):[common.sha(p),p.stat().st_mtime_ns] for p in folder.rglob('*') if p.is_file()}

def semantic(value):
    value=copy.deepcopy(value);value['search'].pop('timing');value['search']['work'].pop('inference_ms');return value

def main():
    if OUTPUT.exists():
        saved=json.loads(OUTPUT.read_text(encoding='utf-8'))
        for p,digest in saved['inputs'].items():common.require(common.sha(ROOT/p)==digest,'Stale recovery evidence '+p)
        print(config.canonical(dict(event='phase6_analysis_recovery_already_complete',passed=True)),flush=True);return
    p=dict(pits=[[7]*7,[7]*7],stores=[0,0],controller=0);model=common.UPDATE;RUN.parent.mkdir(parents=True,exist_ok=True)
    prior=analyzer.analyze(RUN,p,model,'native',64);common.require(prior['search']['completed_simulations']==64,'Expected saved64 root')
    marker=ROOT/'build/phase6-analysis-recovery/precommit.marker';marker.parent.mkdir(parents=True,exist_ok=True);log=marker.with_suffix('.log')
    child_code="""import sys,time
from pathlib import Path
from congklak import phase6_analysis as a
def event(row):
 if row['event']=='phase6_analysis_precommit':
  Path(sys.argv[1]).write_text('completed128 before recommendation transaction')
  while True:time.sleep(.05)
a.analyze(Path(sys.argv[2]),dict(pits=[[7]*7,[7]*7],stores=[0,0],controller=0),Path(sys.argv[3]),'native',128,on_event=event)
"""
    with log.open('w',encoding='utf-8') as stream:
        child=subprocess.Popen([sys.executable,'-c',child_code,str(marker),str(RUN),str(model)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+30
            while not marker.exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.05)
            common.require(marker.exists(),'Analysis kill marker not reached: '+log.read_text(encoding='utf-8'))
        finally:
            if child.poll() is None:child.kill()
            child.wait(timeout=10)
    with closing(sqlite3.connect((RUN/'work.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
        body,digest=db.execute("SELECT body,hash FROM meta WHERE key='analysis_result'").fetchone();saved=json.loads(body)
        common.require(config.digest(saved)==digest and saved==prior,'Interrupted recommendation overwrote committed64')
    resumed=analyzer.analyze(RUN,p,model,'native',128);fresh=analyzer.analyze(FRESH,p,model,'native',128)
    common.require(semantic(resumed)==semantic(fresh),'Native recomputed128 differs from fresh')
    before=files(RUN);cached=analyzer.analyze(RUN,p,model,'native',128);common.require(cached==resumed and files(RUN)==before,'Completed analysis files changed')
    paths=[Path(__file__),ROOT/'congklak/phase6_analysis.py',ROOT/'congklak/phase6_selfplay.py',ROOT/'congklak/phase6_store.py',ROOT/'congklak/puct.py',ROOT/'congklak/phase6_inference.py',model,marker,log]
    paths += [q for folder in (RUN,FRESH) for q in folder.rglob('*') if q.is_file()]
    result=dict(schema='phase6-analysis-recovery-v1',passed=True,retained_prior_after_kill=True,fresh_semantic_equal=True,completed_bytes_mtimes_unchanged=True,
                requested_simulations=[64,128],chosen_one_based=resumed['search']['chosen']+1,proof_status=resumed['proof_status'],model_weights_sha256=json.loads(model.read_text(encoding='utf-8'))['model']['weights_sha256'],
                inputs={str(q.relative_to(ROOT)).replace('\\','/'):common.sha(q) for q in paths},scope='Native fixed simulations; saved completed recommendation survives interruption; unfinished trees recomputed; no exact-game solution claim')
    common.publish(OUTPUT,result);print(config.canonical(dict(event='phase6_analysis_recovery_passed',chosen_one_based=result['chosen_one_based'])),flush=True)

if __name__=='__main__':main()
