"""Check the report sweep through the standalone literal verifier process."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
OUT=ROOT/'build/report-openings-v1'

def main():
    sweep=json.loads((OUT/'sweep.json').read_text())
    for file,expected in sweep['source_sha256'].items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==expected,file
    verified=[]
    for row in sweep['rows']:
        if row['state']!='found':continue
        path=ROOT/row['run_dir']/'certificate.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['certificate_sha256']
        result=subprocess.run([sys.executable,'-B','-X','utf8','-m','congklak.dakon_opening_verify',str(path)],
                              cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=30,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        assert result.returncode==0,result.stderr
        checked=json.loads(result.stdout)
        assert checked['passed'] and checked['n']==row['n']
        assert checked['starter_store']==row['starter_store'] and checked['required']==row['required']
        assert checked['opponent_choices_before_goal']==0 and checked['shortest_proved'] is False
        cert=json.loads(path.read_text())
        assert cert['body']['recipe']['sources']=={
            'searcher':sweep['source_sha256']['congklak/dakon_opening.py'],
            'verifier':sweep['source_sha256']['congklak/dakon_opening_verify.py']}
        verified.append(dict(n=row['n'],passed=True,certificate_sha256=row['certificate_sha256'],verification=checked))
        print('verified',row['n'],flush=True)
    # Check n=1's only initial action with literal sowing, independently of DFS.
    from congklak.dakon_opening_verify import literal
    n1=literal([1,0,1,0],1,0,0)
    assert n1['cells']==[0,1,1,0] and n1['controller']==1 and n1['passes']==[[0,1]]
    assert sweep['rows'][0]['state']=='exhausted'
    n7=json.loads((ROOT/sweep['rows'][6]['run_dir']/'certificate.json').read_text())['body']
    assert n7['actions']==[0,1,4,1,0,4,1,1,2,3,2,0,0]
    assert n7['proof']['starter_store']==54 and n7['steps'][-1]['reason']=='own_capture'
    trace=[]
    previous=[7]*7+[0]+[7]*7+[0]
    for index,(action,step) in enumerate(zip(n7['actions'],n7['steps']),1):
        trace.append(dict(choice=index,display_pit=7-action,canonical_action=action,
                          store=step['cells'][7],store_gain=step['cells'][7]-previous[7],
                          relays=step['relays'],deposits=step['deposits'],captured=step['captured'],
                          reason=step['reason'],cells=step['cells']))
        previous=step['cells']
    (OUT/'seven-pit-trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    result=dict(passed=True,verification_mode='63 separate standalone literal verifier processes',
                verifier_does_not_import_searcher=True,verified_count=len(verified),verified=verified,
                n1_only_action_literal_replay=n1,n1_scope='No first-turn majority witness; not a whole-game loss claim',
                source_sha256=sweep['source_sha256'])
    assert len(verified)==63 and [item['n'] for item in verified]==list(range(2,65))
    (OUT/'independent-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,verified_count=63,n1='only action checked')),flush=True)

if __name__=='__main__':main()
