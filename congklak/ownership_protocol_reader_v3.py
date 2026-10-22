"""Strict semantic reader over the unchanged, recovery-tested V2 publisher.

No new producer, game, search, replay, model forward or training occurs here.
Validation checks persisted flags/counters and their committed source records.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from . import ownership_protocol_flow_v2 as producer

FOLDER,DEFAULT,ROOT,ZERO=producer.FOLDER,producer.DEFAULT,producer.ROOT,producer.ZERO
baseline,atomic=producer.baseline,producer.atomic
SOURCES=(*producer.SOURCES,'congklak/ownership_protocol_reader_v3.py','tools/run-ownership-protocol-v3.py')
need=producer.need
WORK=('model_forwards','feature_simulations','searches','new_games','new_teacher_labels','optimizer_steps','optimizer_draws')


def zero(body,keys,message):
    need(all(type(body.get(key)) is int and body[key]==0 for key in keys),message)


def validate(result,path):
    reports=result['reports'];plan=reports['plan'];projection=reports['projection'];checks=reports['contract-checks']
    zero(projection,WORK,'Projection work counters differ')
    need(projection['schema']=='ownership-domain-projection-v1-report' and
         projection['models_and_policy_fixed'] is True and projection['training_authorized'] is False and
         projection['existing_ownership_hold_remains'] is True and projection['strength_interpretation_allowed'] is False and
         projection['production_admission'] is False and projection['fresh_domain_components'] is None,
         'Projection scope differs')
    inventory=projection['inventory_projection']
    zero(inventory,WORK,'Inventory work counters differ')
    need(inventory['schema']=='ownership-domain-projection-v1-unit' and
         inventory['family_to_quotient_incidence_present'] is False and inventory['complete_incidence_rebuilt'] is False and
         inventory['production_admission'] is False and inventory['fresh_domain_components'] is None and
         inventory['existing_ownership_hold_remains'] is True and inventory['current_strength_claim_allowed'] is False and
         inventory['current_arena_remains_exploratory'] is True and inventory['retained_upper_band']==[75,98] and
         inventory['old_component_ids_cannot_be_reused_after_domain_cut'] is True and
         inventory['retain_all_in_domain_aliases'] is True and inventory['prospective_graph_status'].startswith('HOLD:'),
         'Inventory scope differs')
    zero(checks,('production_engine_calls','optimizer_steps','new_games','searches','new_teacher_labels','feature_simulations'),
         'Contract fixture work counters differ')
    need(checks['schema']=='prospective-ownership-contract-v1-fixture-checks' and
         checks['live_suffix_worker_tested'] is False and checks['production_admission'] is False,
         'Contract fixture scope differs')
    path=producer.path_of(path);identity=producer.read(path/'identity.json')
    units=producer.prefix(path,identity)
    need(checks==units[0],'Contract report differs from committed fixture unit')
    need(inventory==units[193],'Inventory report differs from committed metadata unit')
    compiled=producer.read(FOLDER/'design/compiled-section-spec.json')
    need(plan['section_spec']==units[194]==compiled and
         plan['pending_gates']==compiled['design']['pending_gates'] and
         plan['current_projection_is_not_new_reservations'] is True and
         plan['recovery_scope']=='protocol publication and scratch contracts; live handoff recovery pending',
         'Plan differs from committed and source-bound specification')
    need(all(identity['inputs'].get(k)==v for k,v in projection['inputs'].items()),'Projection input binding differs')
    games=units[1:193];bands=Counter()
    for row in games:bands.update(row['state_band_counts'])
    expected=dict(units=193,existing_saved_games_replayed=192,
        literal_replayed_actions=sum(row['literal_replayed_actions'] for row in games),
        observed_complete_states=sum(row['observed_states'] for row in games),
        state_band_counts=dict(sorted(bands.items())),
        mass_increases=sum(row['mass_increases'] for row in games),
        shared_to_owned_reentries=sum(row['shared_to_owned_reentries'] for row in games),
        owned_to_shared_boundaries=sum(row['first_owned_to_shared'] is not None for row in games),
        equal_active_mass=sum(row['equal_active_mass'] for row in games),
        decreasing_active_mass=sum(row['decreasing_active_mass'] for row in games))
    need(all(projection[k]==v for k,v in expected.items()) and
         projection['all_complete_transitions_checked'] is True and
         projection['mass_increases']==projection['shared_to_owned_reentries']==0,
         'Projection aggregate differs from committed unit counters')
    return result


def completed(path=DEFAULT):
    return validate(producer.completed(path),path)


def run(path=DEFAULT,**kwargs):
    path=producer.path_of(path)
    if (path/'catalog.json').exists():return completed(path)
    return validate(producer.run(path,**kwargs),path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('run','verify'));parser.add_argument('--path',type=Path,default=DEFAULT)
    args=parser.parse_args();result=completed(args.path) if args.operation=='verify' else run(args.path)
    print(json.dumps(dict(passed=True,units=result['catalog']['committed_units'],strict_reader_version=3,
                         producer_version=2,execution_ready=False,training_authorized=False,counters=ZERO),sort_keys=True),flush=True)


if __name__=='__main__':main()
