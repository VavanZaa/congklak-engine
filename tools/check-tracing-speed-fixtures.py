"""Compare complete consumed fixtures with accepted raw moves and endgames."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,tracing_speed_common as stage,ownership_protocol as protocol

def main():
    folder=stage.FOLDER;oldpath=ROOT/'build/ownership-implementation-v1/handoff/engineering-fixed/report.json'
    fixedpath=folder/'jobs/engineering-fixed/report.json';timedpath=folder/'jobs/engineering-timed/report.json'
    old=stage.read(oldpath);fixed=stage.read(fixedpath);timed=stage.read(timedpath)
    baseline=stage.read(folder/'baseline.json')
    stage.need(baseline['inputs'][oldpath.relative_to(ROOT).as_posix()]==stage.sha(oldpath),'Accepted old fixture changed')
    keys=('action','after','before','engine','trace_raw_sha256');cases=[]
    for prior,current in ((4,0),(7,1)):
        left=[{k:r[k] for k in keys} for r in old['logical_results'] if r['game']==prior]
        right=[{k:r[k] for k in keys} for r in fixed['logical_results'] if r['game']==current]
        stage.need(left==right and left,'Complete frozen fixture diverged')
        stage.need(old['final_states'][prior]==fixed['final_states'][current],'Fixed final state differs')
        cases.append(dict(old_game=prior,new_game=current,decisions=len(right),exact_raw_events=True,
            exact_actions_states_engines=True,final_state=fixed['final_states'][current]))
    stage.need(old['final_states'][8]==fixed['final_states'][2] and
        fixed['terminal_boundary_events']==timed['terminal_boundary_events']==1,'Terminal-start regression')
    low_checks={}
    for name,report,specpath in (('fixed',fixed,folder/'jobs/engineering-spec.json'),
                                ('timed',timed,folder/'jobs/engineering-timed-spec.json')):
        stage.need(report['models_fixed'] is True and report['policy_fixed'] is True and
            report['training_counters']==stage.ZERO and report['strength_interpretation_allowed'] is False,
            'Fixture scope differs')
        spec=stage.read(specpath);count=0
        for decision in report['logical_results']:
            row=spec['schedule'][decision['game']];state=decision['before']
            if protocol.active(state)<=24 and state['controller']==row['seat']:
                stage.need(decision['engine']=='control','Learned-seat shared endgame did not retain frozen control')
                count+=1
        stage.need(count>0,'No shared endgame guard exercised');low_checks[name]=count
    stage.need(timed['mode']=='timed25' and timed['overshoot_decisions']==0 and timed['maximum_critical_ms']<=25,
        'Timed game contains a counterexample')
    sources=stage.common.pins([__file__,oldpath,fixedpath,timedpath,folder/'jobs/engineering-spec.json',
        folder/'jobs/engineering-timed-spec.json','congklak/ownership_protocol.py'])
    body=dict(schema='trace-speed-primary-fixture-conformance-v1',passed=True,source_inputs=sources,
        fixed_cases=cases,fixed_decisions_exact=sum(c['decisions'] for c in cases),terminal_start_preserved=True,
        learned_seat_shared_endgame_checks=low_checks,timed_games=timed['games'],timed_decisions=timed['decisions'],
        timed_maximum_critical_ms=timed['maximum_critical_ms'],timed_overshoots=0,models_fixed=True,policy_fixed=True,
        training_counters=stage.ZERO,production_admission=False,strength_claim=False)
    stage.atomic(folder/'main-fixtures-conformance.json',body)
    print(config.canonical(dict(passed=True,fixed_decisions_exact=body['fixed_decisions_exact'],
        learned_seat_shared_endgame_checks=low_checks,timed_maximum_critical_ms=body['timed_maximum_critical_ms'])),flush=True)

if __name__=='__main__':main()
