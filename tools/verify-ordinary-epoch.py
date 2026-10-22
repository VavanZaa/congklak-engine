"""Strict full verification or final publication of the fixed ordinary pilot."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak import ordinary_epoch_common as s,ordinary_epoch_results as r,ordinary_epoch_finish as finish
p=argparse.ArgumentParser();p.add_argument('operation',choices=('verify','report','seal'));a=p.parse_args()
if a.operation=='report':
    report=finish.report();answer=dict(schema=report['schema'],decision=report['decision'],metrics=report['metrics'],
        report_sha256=s.sha(s.FOLDER/'report.json'))
elif a.operation=='seal':
    sealed=finish.seal();answer=dict(schema=sealed['schema'],passed=sealed['passed'],decision=sealed['decision'],
        files_preserved=len(sealed['inputs']),acceptance_sha256=s.sha(s.FOLDER/'acceptance.json'))
else:
    accepted=s.read(s.FOLDER/'acceptance.json');s.need(r.valid_closure(accepted),'Live pilot closure is invalid')
    s.common.current(accepted['inputs'])
    answer=dict(schema=s.SCHEMA+'-full-verification',passed=True,files_checked=len(accepted['inputs']),
        previous_files_preserved=accepted['previous_files_preserved'],policy_fixed=True,
        new_queries=0,new_optimizer_steps=0,promotion=False,decision=accepted['decision'])
print(json.dumps(answer,sort_keys=True,allow_nan=False))
