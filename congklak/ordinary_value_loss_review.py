"""Deterministic saved-game review: legal replay and literal forwards, no search."""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import closing, contextmanager, ExitStack
import json
from pathlib import Path
import sqlite3
import statistics
import sys
from unittest.mock import patch

from . import config, ordinary_value_review_common as common, ordinary_value_benchmark as bench, pilot
from .native import Native, ROOT, flatten
from .neural_relay import Model, NeuralNative

OUTPUT=ROOT/'build/ordinary-value-loss-review.json'
CLI=ROOT/'tools/review-ordinary-value-losses.py'
JOURNAL=ROOT/'data/ordinary-value-benchmark-v1/work.sqlite3'
SOURCE_REPORT=ROOT/'build/ordinary-value-benchmark.json'
RESOURCE=ROOT/'build/ordinary-value-benchmark-resources.json'
CONTROL=ROOT/'data/quality-train-v1/model.json'
CANDIDATE=ROOT/'data/ordinary-value-pilot-v1/model.json'
SCHEMA='ordinary-value-loss-review-v1'
BASELINES=('H_v1','H8_relay_CPU')
PROFILES=('ordinary_control','ordinary_value')
require=common.require


def raw_key(state):
    return bytes(flatten(state)+[state['controller']]).hex()


@contextmanager
def no_search():
    """Make accidentally reaching any old search entry point fail closed."""
    from . import quality_search
    def forbidden(*_args,**_kwargs):raise AssertionError('Saved-data review attempted new search/decision')
    with ExitStack() as stack:
        for owner,name in ((Native,'analyze'),(NeuralNative,'analyze'),(quality_search,'search'),(bench,'decide')):
            stack.enter_context(patch.object(owner,name,side_effect=forbidden))
        yield


def _ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)


def _hashed(text,digest):
    body=json.loads(text);require(config.digest(body)==digest,'Corrupt retained JSON row');return body


def label_index(plan):
    """Only explicitly consumed train/development/diagnostic and exact references.

    The lookup uses raw normalized JSON equality, including stores/controller.
    Sealed final-test bodies and transformed quotient aliases are never read.
    """
    result=defaultdict(list);sources=[]
    def add(state,label,source):
        require(isinstance(label,dict) and isinstance(label.get('action_scores'),list),'Invalid retained label')
        result[config.canonical(state)].append(dict(state=state,label=label,source=source))
    file='build/value-pilot-rows.json';groups=common.read(ROOT/file)['groups'];count=0
    require(set(groups)=={'train','development','ordinary_development','diagnostic'},'Unexpected historical consumed groups')
    for group,rows in groups.items():
        for index,row in enumerate(rows):
            add(row['state'],row['label'],dict(path=file,group=group,row=index,row_sha256=config.digest(row)));count+=1
    sources.append(dict(path=file,entries=count,scope='Previously consumed train/development/ordinary-development/diagnostic groups only'))
    for index,row in enumerate(plan['cases']):
        require(row['reference']['kind']=='exact_optimal_margin','Non-exact case unexpectedly indexed as exact')
        add(row['state'],row['reference'],dict(path='build/ordinary-value-benchmark-plan.json',case=index,row_sha256=config.digest(row)))
    sources.append(dict(path='build/ordinary-value-benchmark-plan.json',entries=len(plan['cases']),scope='Known exact guards; six validation references are indexed from their closed journal'))
    tables={
        'ordinary-data-pilot-v1':('labels',),
        'natural-study-v1':('h8_labels','h10_labels'),
        'horizon-finish-v1':('h8_references','h10_references','h12_references','prior_h14_labels','tier1_labels','tier2_labels'),
        'midgame-label-pilot-v1':('old_h8_references','h8_labels','h10_labels','h12_labels'),
        'ordinary-value-labels-v1':('h8_labels','h10_labels','h12_labels'),
        'deeper-9000-v2':('labels',),
    }
    for directory,names in tables.items():
        file='data/'+directory+'/work.sqlite3';count=0
        with closing(_ro(ROOT/file)) as db:
            roots={}
            # Split filtering is on root metadata before any associated label body is read.
            for key,text,digest in db.execute('SELECT state,body,hash FROM roots ORDER BY state'):
                root=_hashed(text,digest)
                if directory=='deeper-9000-v2' and root['split'] not in ('train','development'):continue
                roots[key]=(root,digest)
            for table in names:
                for key in sorted(roots):
                    row=db.execute('SELECT body,hash FROM '+table+' WHERE state=?',(key,)).fetchone()
                    if row is None:continue
                    item=_hashed(*row);root,root_sha=roots[key]
                    add(root['state'],item['label'],dict(path=file,table=table,state_key=key,item_sha256=row[1],root_row_sha256=root_sha,split=root.get('split'),cohort=root.get('cohort')));count+=1
        sources.append(dict(path=file,entries=count,scope='Consumed full requested labels/references; deeper-9000 train/development only; no auxiliary iterations'))
    return result,dict(match='Literal normalized pits, stores and controller; no player swap, quotient or store-offset Q reuse',sources=sources,
        indexed_entries=sum(len(values) for values in result.values()),indexed_unique_full_states=len(result),sealed_final_test_bodies_read=0,
        miss_semantics='uncovered_in_index does not assert that no reference exists elsewhere',auxiliary_reports_indexed=False)


def existing_Q(state,left_action,right_action,index):
    found=index.get(config.canonical(state),[]);grouped={};sign=1 if state['controller']==0 else -1
    for item in found:
        require(item['state']==state,'Stored Q state orientation differs')
        label=item['label'];q={row['pit']:row['value_player0'] for row in label['action_scores']}
        require(left_action in q and right_action in q,'Indexed full-root Q lacks a selected action')
        key=config.digest(label)
        if key in grouped:
            grouped[key]['sources'].append(item['source']);continue
        left=sign*(label['target_player0']-q[left_action]);right=sign*(label['target_player0']-q[right_action])
        require(left>=0 and right>=0,'Stored Q actor perspective differs')
        grouped[key]=dict(sources=[item['source']],label_sha256=key,label_kind=label.get('kind'),choice_depth=label.get('choice_depth'),
            target_player0=label['target_player0'],best_moves=label['best_moves'],display_best_moves=[pit+1 for pit in label['best_moves']],
            control_action_Q_player0=q[left_action],candidate_action_Q_player0=q[right_action],control_actor_regret=left,candidate_actor_regret=right,
            candidate_minus_control_actor_regret=right-left)
    return dict(coverage='exact_full_state_stored_Q' if found else 'uncovered_in_index',comparisons=[grouped[key] for key in sorted(grouped)],
        orientation='Original normalized full state, player0 values and controller-signed regret; no transforms',
        interpretation='Finite-horizon disagreement is not a proven game blunder; an index miss is not globally absent labels')


def _snapshot(record,choice):
    row=record['records'][choice]['row'];env=row['decision']['decision'];report=env['report']
    return dict(source=dict(game_index=record['index'],choice_index=choice,decision_sha256=config.digest(record['records'][choice])),saved_decision=row,
        summary=dict(engine=row['engine'],actor=row['actor'],chosen=env['chosen'],display_chosen=env['chosen']+1,kind=env['kind'],
            completed_depth=env['completed_depth'],completed_simulations=env['completed_simulations'],nodes=env['nodes'],
            requested_ms=env['requested_ms'],wall_ms=env['wall_ms'],overshoot_ms=env['overshoot_ms'],
            root_priors=report.get('root_priors'),visits=report.get('visits'),visit_policy=report.get('policy'),
            estimated_value_player0=report.get('estimated_value_player0',report.get('value_player0')),
            neural_bootstrap_player0=report.get('unvisited_root_q_player0'),action_mean_value_player0=report.get('action_mean_value_player0'),
            timing_scope='Historical saved whole-decision clock, not newly measured',evaluator=row['decision']['evaluator']))


def _replay(record,plan,native):
    game=record['body'];index=record['index'];comp=dict(id=game['comparison'],candidate=game['comparison'].split('_vs_')[0],opponent=game['comparison'].split('_vs_')[1])
    unit=dict(pair=game['pair'],seat=game['candidate_seat']);start=plan['arena_starts'][game['pair']]
    require(game['start']==start['state'] and game['family']==start['family'],'Game start/family differs from frozen schedule')
    records=record['records'];require(game['moves']==[row['action'] for row in records] and game['decisions']==[row['row'] for row in records],'Game versus durable decision ledger differs')
    require(game['status']=='terminal' and 0<len(records)<=plan['arena_max_choices'] and not any(game['fallbacks'].values()),'Unfinished/fallback game cannot be silently admitted')
    state=game['start'];steps=[];costs=defaultdict(lambda:dict(decisions=0,wall_ms=0.,overshoot_ms=0.,nodes=0,completed_simulations=0,physical_forwards=0,feature_simulations=0,feature_deposits=0))
    for choice,row in enumerate(records):
        after=bench.validate_record(row,index,choice,state,unit,comp,plan,native)
        require(not row['row']['fallback'],'Fallback decision in retained game')
        before=row['state'];transition=native.apply(flatten(before),actor=before['controller'],action=row['action'],total=98,trace=True)
        require(transition['ok'] and transition['position']==after,'Literal trace versus saved state differs')
        gains=[after['stores'][i]-before['stores'][i] for i in (0,1)];captures=[event for event in transition['events'] if event['type']=='capture']
        require(gains==transition['store_delta'] and all(event['after']-event['before']==event['count'] for event in captures),'Literal capture/store facts differ')
        env=row['row']['decision']['decision'];telemetry=row['row']['decision']['evaluator'];raw=env['report'];cost=costs[row['row']['engine']]
        cost['decisions']+=1;cost['wall_ms']+=env['wall_ms'];cost['overshoot_ms']+=env['overshoot_ms'];cost['nodes']+=env['nodes'];cost['completed_simulations']+=env['completed_simulations'] or 0
        cost['physical_forwards']+=telemetry['physical_forward_executions'] if telemetry else raw.get('nn_evaluations',0)
        for field in ('feature_simulations','feature_deposits'):cost[field]+=telemetry['counts'][field] if telemetry else raw.get(field,0)
        steps.append(dict(choice_index=choice,choice=choice+1,before_state_key=raw_key(before),after_state_key=raw_key(after),actor=before['controller'],engine=row['row']['engine'],pit=row['action'],display_pit=row['action']+1,
            store_gain=gains,capture_events=captures,capture_transfer=sum(event['count'] for event in captures),bonus_choice=transition['reason']=='own_store' and not transition['terminal'],
            forced_passes_after=transition['passes'],terminal_after=transition['terminal'],reason=transition['reason'],relays=transition['relays'],deposits=transition['drops']))
        state=after
    require(state==game['final'] and not sum(map(sum,state['pits'])),'Nonterminal or changed final state')
    margin=(state['stores'][0]-state['stores'][1])*(1 if game['candidate_seat']==0 else -1)
    require(margin==game['candidate_final_margin'] and game['candidate_match_score']==(1 if margin>0 else 0 if margin<0 else .5),'Terminal margin/match score differs')
    return dict(game_index=index,game_sha256=record['sha256'],comparison=game['comparison'],pair=game['pair'],candidate_seat=game['candidate_seat'],family=game['family'],start_stratum=start['stratum'],
        status='terminal',truncated=False,margin_seeds=margin,match_score=game['candidate_match_score'],choices=len(steps),fallbacks=0,final=state,cost_by_engine=dict(costs),steps=steps)


def _divergence(left,right,a,b,index,native):
    shared=0
    while shared<min(len(left['records']),len(right['records'])) and left['records'][shared]['action']==right['records'][shared]['action']:
        require(left['records'][shared]['state']==right['records'][shared]['state'] and left['records'][shared]['after']==right['records'][shared]['after'],'Shared prefix state differs');shared+=1
    if shared==min(len(left['records']),len(right['records'])):
        require(len(left['records'])==len(right['records']) and left['body']['final']==right['body']['final'],'A terminal game cannot be a strict prefix')
        return dict(shared_choices=shared,identical_action_trajectory=True,first_different_choice=None)
    l=left['records'][shared];r=right['records'][shared];state=l['state'];require(state==r['state'],'First divergence does not share the exact raw state')
    actor=state['controller'];seat=left['body']['candidate_seat'];active,stratum=pilot._mechanism(state,native);candidate_first=actor==seat
    leftsnap=_snapshot(left,shared);rightsnap=_snapshot(right,shared)
    priors_equal=(leftsnap['summary']['root_priors']==rightsnap['summary']['root_priors']) if candidate_first else None
    if candidate_first:require(priors_equal is True and leftsnap['summary']['root_priors'] is not None,'Frozen same-state candidate priors differ')
    return dict(shared_choices=shared,identical_action_trajectory=False,first_different_choice=shared+1,choice_index=shared,state=state,state_key=raw_key(state),actor=actor,candidate_seat=seat,
        first_changed_actor_role='candidate_profile' if candidate_first else 'opponent_'+left['body']['comparison'].split('_vs_')[1],opponent_first=not candidate_first,
        active_seeds=active,divergence_stratum=stratum,start_stratum=a['start_stratum'],control=leftsnap,candidate=rightsnap,candidate_priors_equal=priors_equal,
        existing_Q_comparison=existing_Q(state,l['action'],r['action'],index),
        control_nearby_observations=a['steps'][max(0,shared-2):shared+4],candidate_nearby_observations=b['steps'][max(0,shared-2):shared+4],
        attribution='First distinct saved action after the shared prefix. Candidate-owned divergence combines changed values with independently timed search/visits; opponent-owned divergence is also timing dependent. Later trajectories are unmatched; neither local captures nor this game delta identifies a causal blunder.')


def _wdl(values):
    return dict(wins=sum(x>0 for x in values),draws=sum(x==0 for x in values),losses=sum(x<0 for x in values))


def build():
    require('torch' not in sys.modules,'Review must run without Torch')
    prior=common.baseline(verify_all=True)
    paths=(Path(__file__),CLI,ROOT/'tests/ordinary_value_loss_review_author.py',common.BASELINE,common.SCOPE,ROOT/'congklak/ordinary_value_review_common.py')
    inputs={**prior['inputs'],**common.pins(paths)};common.current(inputs)
    with no_search():
        plan=common.read(ROOT/'build/ordinary-value-benchmark-plan.json');report=common.read(SOURCE_REPORT);resource=common.read(RESOURCE)
        require(report['schema']=='ordinary-value-benchmark-results-v1' and report['stage']=='evaluation' and report['fixture'] is False,'Wrong saved benchmark')
        require(report['games']==96 and report['decisions']==3268 and report['cases']==38,'Saved benchmark geometry differs')
        require(resource['report_file_sha256']==common.sha(SOURCE_REPORT) and resource['torch_imported'] is False,'Saved resource identity differs')
        plan={**plan,'stage':'evaluation','fixture':False}
        models={'control':Model(CONTROL),'candidate':Model(CANDIDATE)};model_identity=bench.verify_candidate(models['control'],models['candidate'])
        model_identity.update(models={name:dict(path=str(path.relative_to(ROOT)).replace('\\','/'),file_sha256=model.file_sha256,weights_sha256=model.id) for name,path,model in (('control',CONTROL,models['control']),('candidate',CANDIDATE,models['candidate']))},
            causal_limit='Frozen trunk/policy proves identical same-state raw policy. It does not identify the value update as the cause of every timed move or outcome difference. The sealed historical draw-support proof establishes positive ancestral draw exposure for the encoder-equivalent source pattern; its exact old count and causal gradient effect are not reconstructed here.')
        labels,index_scope=label_index(plan);native=Native();records={};replays={}
        with closing(_ro(JOURNAL)) as db:
            meta={key:_hashed(text,digest) for key,text,digest in db.execute('SELECT key,body,hash FROM meta')}
            marker=meta['ordinary_value_benchmark_complete']
            require(meta['identity']==report['identity'] and marker['report_sha256']==config.digest(report) and marker['resource_file_sha256']==common.sha(RESOURCE),'Saved completion binding differs')
            require((marker['cases'],marker['games'],marker['decisions'])==(38,96,3268),'Saved completion marker geometry differs')
            for expected,(idx,name,pair,seat,text,digest) in enumerate(db.execute('SELECT idx,comparison,pair,seat,body,hash FROM games ORDER BY idx')):
                require(idx==expected,'Noncontiguous saved game IDs');game=_hashed(text,digest)
                require((game['comparison'],game['pair'],game['candidate_seat'])==(name,pair,seat),'Saved game key differs')
                record=dict(index=idx,sha256=digest,body=game,records=bench.records_for(db,idx));key=(name,pair,seat);require(key not in records,'Duplicate saved game');records[key]=record;replays[key]=_replay(record,plan,native)
            require(db.execute('SELECT COUNT(*) FROM decisions').fetchone()[0]==3268 and len(records)==96,'Orphan/missing saved choices/games')
        expected={(profile+'_vs_'+base,pair,seat) for profile in PROFILES for base in BASELINES for pair in range(12) for seat in (0,1)}
        require(set(records)==expected,'Incomplete twelve-family matched geometry')
        seats=[];families=[];summary={};cache={}
        for base in BASELINES:
            matched=[]
            for pair in range(12):
                group=[]
                for seat in (0,1):
                    ck=('ordinary_control_vs_'+base,pair,seat);tk=('ordinary_value_vs_'+base,pair,seat);c=records[ck];t=records[tk];a=replays[ck];b=replays[tk]
                    require(c['body']['start']==t['body']['start'] and c['body']['family']==t['body']['family'],'Matched seat roots differ')
                    divergence=_divergence(c,t,a,b,labels,native)
                    if not divergence['identical_action_trajectory']:
                        key=divergence['state_key']
                        if key not in cache:
                            require(len(cache)<10,'Literal forward state budget exceeded');state=divergence['state'];predictions={}
                            for name,model in models.items():
                                value,quantized,logits=model.reference(state);predictions[name]=dict(value_player0=value,quantized_value_units=quantized,policy_logits=logits)
                            require(predictions['control']['policy_logits']==predictions['candidate']['policy_logits'],'Frozen literal policy differs')
                            cache[key]=dict(state=state,control=predictions['control'],candidate=predictions['candidate'],candidate_minus_control_value_player0=predictions['candidate']['value_player0']-predictions['control']['value_player0'],policy_logits_identical=True,new_timing_measured=False)
                        divergence['literal_prediction_state_key']=key
                    item=dict(baseline=base,pair=pair,candidate_seat=seat,family=a['family'],start_stratum=a['start_stratum'],control_game_index=a['game_index'],candidate_game_index=b['game_index'],
                        control_margin=a['margin_seeds'],candidate_margin=b['margin_seeds'],candidate_minus_control_margin=b['margin_seeds']-a['margin_seeds'],control_match_score=a['match_score'],candidate_match_score=b['match_score'],first_divergence=divergence)
                    seats.append(item);matched.append(item);group.append(item)
                families.append(dict(baseline=base,pair=pair,family=group[0]['family'],start_stratum=group[0]['start_stratum'],
                    control_mean_margin=statistics.mean(x['control_margin'] for x in group),candidate_mean_margin=statistics.mean(x['candidate_margin'] for x in group),paired_mean_candidate_minus_control=statistics.mean(x['candidate_minus_control_margin'] for x in group)))
            div=[x['first_divergence'] for x in matched if not x['first_divergence']['identical_action_trajectory']];family_values=[x['paired_mean_candidate_minus_control'] for x in families if x['baseline']==base]
            summary[base]=dict(control_wdl=_wdl([x['control_margin'] for x in matched]),candidate_wdl=_wdl([x['candidate_margin'] for x in matched]),control_mean_margin=statistics.mean(x['control_margin'] for x in matched),candidate_mean_margin=statistics.mean(x['candidate_margin'] for x in matched),paired_mean_candidate_minus_control=statistics.mean(family_values),
                identical_action_trajectories=sum(x['first_divergence']['identical_action_trajectory'] for x in matched),first_divergences=len(div),candidate_first_divergences=sum(not x['opponent_first'] for x in div),opponent_first_divergences=sum(x['opponent_first'] for x in div),
                negative_delta_seats=sum(x['candidate_minus_control_margin']<0 for x in matched),positive_delta_seats=sum(x['candidate_minus_control_margin']>0 for x in matched),unchanged_delta_seats=sum(x['candidate_minus_control_margin']==0 for x in matched),
                absolute_candidate_losses=sum(x['candidate_margin']<0 for x in matched),absolute_losses_with_identical_trajectory=sum(x['candidate_margin']<0 and x['first_divergence']['identical_action_trajectory'] for x in matched),
                leave_one_family_out_means=[dict(omitted_pair=i,paired_mean_candidate_minus_control=statistics.mean(v for j,v in enumerate(family_values) if i!=j)) for i in range(12)],
                inherited_exploratory_family_interval=report['secondary_factor_contrasts'][base],new_confidence_intervals_computed=False)
        divergences=[x for x in seats if not x['first_divergence']['identical_action_trajectory']];covered=sum(x['first_divergence']['existing_Q_comparison']['coverage']=='exact_full_state_stored_Q' for x in divergences)
        losses=sorted([dict(pair=x['pair'],candidate_seat=x['candidate_seat'],family=x['family'],start_stratum=x['start_stratum'],control_margin=x['control_margin'],candidate_margin=x['candidate_margin'],candidate_minus_control_margin=x['candidate_minus_control_margin'],identical_action_trajectory=x['first_divergence']['identical_action_trajectory'],first_different_choice=x['first_divergence']['first_different_choice']) for x in seats if x['baseline']=='H_v1' and x['candidate_margin']<0],key=lambda x:(x['candidate_margin'],x['pair'],x['candidate_seat']))
        replay_by_id={row['game_index']:row for row in replays.values()};hold_context=[]
        for collision in report['trajectory_replay_overlap']['collisions']:
            game=replay_by_id[collision['game']];matched=next(row for row in seats if row['baseline']=='H_v1' and row['pair']==collision['pair'] and row['candidate_seat']==collision['candidate_seat'])
            hold_context.append(dict(saved_collision=collision,game_margin=game['margin_seeds'],game_choices=game['choices'],matched_margin_delta=matched['candidate_minus_control_margin'],is_candidate_H_v1_loss=game['margin_seeds']<0,is_matched_H_v1_regression=matched['candidate_minus_control_margin']<0))
        summary['H_v1']['historical_admitted_pool_hold_context']=dict(observations=hold_context,
            interpretation='These original HOLD observations are a separate cohort fact, not a diagnosis that they caused any of the three adverse H_v1 seat deltas. The original allocation is unchanged.')
    common.current(inputs);require('torch' not in sys.modules,'Torch imported during review')
    return dict(schema=SCHEMA,passed=True,inputs=inputs,prior_inputs_preserved=len(prior['inputs']),
        counts=dict(games=len(records),choices=sum(len(x['steps']) for x in replays.values()),source_families=12,matched_family_units=len(families),matched_seats=len(seats),first_divergences=len(divergences),unique_divergence_states=len(cache),literal_model_forwards=2*len(cache),indexed_Q_covered_divergences=covered,indexed_Q_uncovered_divergences=len(divergences)-covered),
        summary=summary,model_identity=model_identity,label_index_scope=index_scope,literal_divergence_predictions=cache,game_replays=sorted(replays.values(),key=lambda x:x['game_index']),families=families,seats=seats,absolute_H_v1_losses=losses,
        limits=['Exploratory consumed development games, not new or sealed release evidence. All saved games terminate; no truncated game was dropped.',
            'A start mechanism and the later divergence mechanism are separate facts. Captures, store gains and unmatched later paths are observations, not a causal blunder proof.',
            'Independent whole-decision clocks change simulations/visits and can change opponent moves. Frozen policy plus changed values does not isolate a deterministic value-head effect.',
            'Twelve of thirteen absolute H_v1 losses share the complete control trajectory. Shared losses are not proved unavoidable or optimal.',
            'Finite-depth Q is only reused at the exact original full state. Missing indexed Q remains uncovered; no new teacher/search/fixed-simulation counterfactual was run.',
            'Original admitted-pool allocation HOLD is unchanged. Zero current optimizer draws coexists with positive ancestral draw exposure to the intentionally encoder-equivalent late source pattern, established by the sealed all-7669-root draw-support report; it does not establish a causal gradient or loss effect.'],
        work=dict(new_games=0,new_teacher_searches=0,new_PUCT_searches=0,new_fixed_simulation_diagnostics=0,new_optimizer_steps=0,torch_imported=False,literal_model_forwards=2*len(cache),literal_forward_states=len(cache),new_forward_timing_measured=False,historical_clocks_retained=True))


def run(path=OUTPUT):
    return common.publish(Path(path),build())


def verify(path=OUTPUT):
    return validate_report(common.read(path))


def validate_report(body):
    require(body==build(),'Saved loss review differs from recomputed current evidence');return body
