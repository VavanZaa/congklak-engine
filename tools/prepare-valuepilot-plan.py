"""Record plan-only metadata feasibility and protect the preceding accepted stage."""
from collections import Counter
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,neurodiag_common as common
FOLDER=ROOT/'build/valuepilot-plan-v1'


def publish_draft(path,body):
    """The metadata worksheet may change during plan review, before sealing."""
    common.require(not (FOLDER/'acceptance.json').exists(),'Accepted planning package is immutable')
    text=config.canonical(body)+'\n';temporary=path.with_suffix('.tmp')
    temporary.write_text(text,encoding='utf-8',newline='\n');os.replace(temporary,path)
    return body


def main():
    accepted=common.read(ROOT/'build/neurodiag-acceptance.json')
    common.require(accepted['passed'] is True and accepted['tests_run']==14,'Prior accepted diagnostics missing')
    protected=dict(accepted['inputs'])
    protected.update(common.pins(['build/neurodiag-acceptance.json','build/neurodiag-acceptance-rerun.json','build/neurodiag-dashboard-final.json']))
    common.current(protected)
    common.publish(FOLDER/'prior.json',dict(schema='neural-value-pilot-plan-prior-v1',passed=True,
        inputs_count=len(protected),inputs=protected,models_and_policy_fixed=True,training_performed=False))
    inv=common.read(ROOT/'build/neurodiag-inventory.json');panel=common.read(ROOT/'build/neurodiag-panel.json')
    plan=common.read(FOLDER/'plan.json');cells=plan['data']['cells'];quota=plan['data']['train_per_cell']+plan['data']['fit_check_per_cell']
    excluded={row['component'] for group in ('primary','paired','MC') for row in panel[group]}
    rows=[row for row in inv['primary_pool'] if row['component'] not in excluded and row['legal_mask'].bit_count()>=2]
    options={cell:sorted([row for row in rows if row['cell']==cell],
        key=lambda row:(config.digest(plan['data']['hash_salt']+row['state_key']),row['state_key'])) for cell in cells}
    assigned={};chosen={}
    def place(slot,seen):
        for row in options[slot[0]]:
            component=row['component']
            if component in seen:continue
            seen.add(component)
            if component not in assigned or place(assigned[component],seen):
                assigned[component]=slot;chosen[slot]=row;return True
        return False
    slots=[(cell,index) for cell in cells for index in range(quota)]
    feasible=all(place(slot,set()) for slot in slots)
    result=dict(schema='neural-value-pilot-plan-metadata-feasibility-v1',passed=feasible,
        inventory_roots=len(inv['primary_pool']),inventory_components=len({row['component'] for row in inv['primary_pool']}),
        excluded_diagnostic_components=len(excluded),requested_components=len(slots),
        matching_components=len(assigned),matched_by_cell=dict(Counter(cell for cell,_ in chosen)),
        available_by_cell={cell:dict(roots=len(options[cell]),components=len({r['component'] for r in options[cell]})) for cell in cells},
        actual_selection_published=False,new_reservations=False,full_current_ownership_rebuild_still_required=True,
        target_bodies_read=0,model_forwards=0,searches=0,optimizer_steps=0,new_games=0,
        interpretation='Existing frozen metadata feasibility only. No selected memberships saved; no unseen/generalization claim.',
        inputs=common.pins([__file__,FOLDER/'plan.json','build/neurodiag-inventory.json','build/neurodiag-panel.json']))
    common.require(feasible and len(assigned)==len(slots)==640,'Metadata component quota shortage')
    publish_draft(FOLDER/'feasibility.json',result)
    print(config.canonical(dict(event='valuepilot_plan_metadata_checked',passed=True,prior_inputs=len(protected),
        feasible_components=len(assigned),target_bodies_read=0,model_forwards=0,optimizer_steps=0)),flush=True)


if __name__=='__main__':main()
