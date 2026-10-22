"""Freeze consumed timing roots by metadata before any timing results exist."""
from pathlib import Path
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,neurodiag_common as common,tracing_speed_common as stage,tracing_speed_contract as contract
from congklak import value_run_arena as literal

def main():
    source=ROOT/'data/neural-value-pilot-v2/reservations/inventory.json';inventory=common.read(source)
    panel=[];reference=literal._literal()
    for cell in contract.CELLS:
        candidates=[r for r in inventory['primary_pool'] if r['cell']==cell and r['legal_mask'].bit_count()>=2]
        candidates.sort(key=lambda r:(hashlib.sha256(('tracing-speed-v1/panel/20261016/'+r['state_key']).encode()).hexdigest(),r['state_key']))
        common.require(candidates,'Missing consumed panel cell '+cell);row=candidates[0]
        normal=literal.normalize(row['state'],reference)
        common.require(normal['state']==row['state'] and not normal['terminal'] and len(normal['legal'])>=2,'Panel root normalization differs')
        panel.append(dict(index=len(panel),cell=cell,state=row['state'],state_key=row['state_key'],source_family=row['family'],
            source_path=source.relative_to(ROOT).as_posix(),source_pointer='/primary_pool/'+str(inventory['primary_pool'].index(row)),
            diagnostic_origin='timing-consumed-panel',legal=normal['legal'],metadata_selection_only=True))
    schedule=[]
    for rep in range(5):
        for row in panel:
            offset=(rep+row['index'])%4;engines=contract.ENGINES[offset:]+contract.ENGINES[:offset]
            for engine in engines:schedule.append(dict(index=len(schedule),root=row['index'],repetition=rep,engine=engine))
    common.require(len(schedule)==240,'Timing panel size differs')
    body=dict(schema='trace-speed-timing-panel-v1',clock_contract=contract.definition(),
        clock_contract_sha256=config.digest(contract.definition()),panel=panel,schedule=schedule,
        selection_salt='tracing-speed-v1/panel/20261016',origin_scope='One shared consumed diagnostic origin, not independent arena origins.',
        setup_claims='Fixed4096 canonical high keys from the frozen engineering supplement, all assigned to the same diagnostic origin; restore before each independent trial. Real worker tests keep cumulative claims separately.',
        source_inputs=common.pins([source,__file__,'congklak/tracing_speed_contract.py','congklak/tracing_speed_common.py','tests/reference.py']),
        policy_fixed=True,models_fixed=True,targets_or_predictions_used=False,training_counters=stage.ZERO)
    stage.atomic(stage.FOLDER/'timing/panel.json',body)
    print(config.canonical(dict(passed=True,panel_roots=12,planned_attempts=240,panel_sha256=stage.sha(stage.FOLDER/'timing/panel.json'))),flush=True)

if __name__=='__main__':main()
