"""Prospectively fixed paired-origin uncertainty and strict pilot closure."""
import math,random,statistics
from . import config,ordinary_epoch_common as s

AUDIT_METHODS=('recipe_reservations','data_targets_sampler','policy_optimizer_recovery',
 'native_trace_guard_conformance','arena_recovery_ownership','fit_capture_endgame',
 'equal_time_both_baselines','resources_prior_preservation','observer_closure')
GATES=('fixed_policy_and_base','final400_training','fit_coverage','fit_huber','fit_seed_mae','capture_seed_mae',
 'capture_case_regret','case_coverage','native_conformance','endgame_guard','identical_low_fixed_work',
 'arena_coverage','arena_no_caps_or_fallbacks','whole_component_ownership','equal_time_25ms',
 'all_four_strength_lower_bounds_positive')

def derive_gates(metrics,statistics_report):
    """Derive a fixed gate set; missing/HOLD measurements can never pass."""
    s.need(set(metrics)=={'training','labels','fit','cases','conformance','endgame','arena'},'Metric groups differ')
    t,l,f,c,n,e,a=[metrics[k] for k in ('training','labels','fit','cases','conformance','endgame','arena')]
    def count(row,key,value):return type(row.get(key)) is int and row[key]==value
    def number(row,key):return type(row.get(key)) in (int,float) and math.isfinite(row[key])
    def nonincrease(row,control,candidate):return number(row,control) and number(row,candidate) and row[candidate]<=row[control]+1e-9
    result=dict(fixed_policy_and_base=t.get('policy_fixed') is True and t.get('base_frozen') is True,
        final400_training=count(t,'steps',400) and count(t,'draws',51200),
        fit_coverage=count(l,'roots',128) and count(l,'complete_h8_roots',128) and
            all(count(f,k,v) for k,v in (('roots',128),('ordinary_roots',64),('capture_roots',32),('bonus_roots',32))),
        fit_huber=number(f,'control_huber') and number(f,'candidate_huber') and f['control_huber']-f['candidate_huber']>1e-12,
        fit_seed_mae=nonincrease(f,'control_mae','candidate_mae'),
        capture_seed_mae=nonincrease(f,'control_capture_mae','candidate_capture_mae'),
        capture_case_regret=all(nonincrease(c,'control_capture_regret_'+mode,'candidate_capture_regret_'+mode) for mode in ('timed25','fixed128')),
        case_coverage=all(count(c,k,v) for k,v in (('roots',32),('ordinary_roots',16),('capture_roots',8),('bonus_roots',8))),
        native_conformance=n.get('passed') is True and n.get('policy_fixed') is True,
        endgame_guard=count(e,'roots',32) and e.get('control_optimal') is True and e.get('candidate_optimal') is True,
        identical_low_fixed_work=e.get('fixed_work_equal') is True and count(e,'learned_branch_calls',0),
        arena_coverage=count(a,'games',192) and count(a,'terminal_games',192),
        arena_no_caps_or_fallbacks=count(a,'caps',0) and count(a,'fallbacks',0),
        whole_component_ownership=all(row.get('ownership_passed') is True for row in (l,f,a,c,n,e)),
        equal_time_25ms=all(count(row,'critical_overshoots',0) for row in (a,c,n,e)),
        all_four_strength_lower_bounds_positive=bool(statistics_report) and statistics_report.get('all_four_lower_bounds_positive') is True)
    s.need(set(result)==set(GATES) and all(type(v) is bool for v in result.values()),'Gate derivation differs')
    return result

def paired_statistics(rows):
    """192 source-validated game rows, averaging seats before origin resampling.

    Each row has origin, profile(control/candidate), baseline(H_v1/H8_relay_CPU),
    seat(0/1), integer learned-seat margin and terminal=True.  Validation of the
    underlying game journals and timing belongs to the independent arena reader.
    """
    s.need(len(rows)==192,'All192 game rows required for paired statistics')
    origins=sorted({r['origin'] for r in rows});s.need(len(origins)==24,'Exactly24 origins required')
    lookup={}
    for r in rows:
        s.need(type(r['origin']) is str and r['profile'] in ('control','candidate') and
            r['baseline'] in ('H_v1','H8_relay_CPU') and type(r['seat']) is int and r['seat'] in (0,1) and
            type(r['margin']) is int and -98<=r['margin']<=98 and r['terminal'] is True,'Invalid terminal margin row')
        key=(r['origin'],r['baseline'],r['profile'],r['seat']);s.need(key not in lookup,'Repeated game cell')
        lookup[key]=r['margin']
    contrasts={}
    for baseline in ('H_v1','H8_relay_CPU'):
        control=[sum(lookup[(o,baseline,'control',seat)] for seat in (0,1))/2 for o in origins]
        candidate=[sum(lookup[(o,baseline,'candidate',seat)] for seat in (0,1))/2 for o in origins]
        contrasts['candidate/'+baseline]=candidate
        contrasts['candidate-minus-control/'+baseline]=[b-a for a,b in zip(control,candidate)]
    rng=random.Random(20261016);boot={k:[] for k in contrasts}
    for _ in range(20000):
        indices=[rng.randrange(24) for _ in range(24)]
        for key,values in contrasts.items():boot[key].append(sum(values[i] for i in indices)/24)
    def quantile(values,p):
        values.sort();rank=(len(values)-1)*p;lo=math.floor(rank);hi=math.ceil(rank)
        return values[lo]+(values[hi]-values[lo])*(rank-lo)
    out={}
    for key,values in contrasts.items():
        out[key]=dict(mean=statistics.mean(values),lower=quantile(boot[key],.00625),
            upper=quantile(boot[key],.99375),origin_values=values,
            all_origins_equal_weight=True,seats_averaged=True)
    return dict(schema=s.SCHEMA+'-paired-statistics',origins=origins,origin_count=24,games=192,
        resamples=20000,seed=20261016,method='paired percentile bootstrap/linear quantiles',
        interval_two_sided_coverage=.9875,simultaneous_nominal_coverage=.95,contrasts=out,
        all_four_lower_bounds_positive=all(v['lower']>0 for v in out.values()),
        uncertainty_calibrated=False,natural_play_representativeness=False)

def audit_valid(a):
    s.need(a['schema']==s.SCHEMA+'-independent-audit' and a['passed'] is True and
        a['independent_of_implementation'] is True and a['plan_sha256']==s.sha(s.PLAN) and
        a['unresolved_findings']==[] and set(a['methods'])==set(AUDIT_METHODS) and
        a['policy_fixed'] is True and a['base_frozen'] is True and a['promotion'] is False and
        a['inherited_unknown_background'] is True and a['historically_unseen'] is False,
        'Independent final audit incomplete')
    s.need(bool(a['source_inputs']),'Independent source closure missing');s.common.current(a['source_inputs'])
    expected_recovery=dict(training_process_kills=2,native_label_process_kills=6,native_handoff_process_kills=2,
        native_zero_query_intent_process_kills=1,uninterrupted_training_fixture_steps=6,
        successful_independent_optimizer_replay_steps=418)
    actual=a['recovery_counts']
    s.need(set(actual)==set(expected_recovery) and all(type(actual[k]) is int and actual[k]>=v for k,v in expected_recovery.items()) and
           actual['successful_independent_optimizer_replay_steps']==418,'Actual required recovery matrix is incomplete')
    from . import ordinary_epoch_query_contract as contract
    expected_boundaries=dict(training=['before_checkpoint_catalog','after_checkpoint_catalog'],
        native_label=list(contract.LABEL_KILL_HOOKS),native_handoff=list(contract.HANDOFF_KILL_HOOKS),
        native_zero_query_intent=['before_query'])
    s.need(config.canonical(a['accepted_recovery_boundaries'])==config.canonical(expected_boundaries),
           'Accepted recovery boundary coverage differs')
    s.need(type(a['engineering_counters']) is dict and bool(a['engineering_counters']) and
           all(type(v) is int and v>=0 for v in a['engineering_counters'].values()),'Physical engineering activity was not disclosed')
    for method in a['methods'].values():
        s.need(method['passed'] is True and type(method['evidence']) is list and bool(method['evidence']),
               'Independent method missing evidence')
        for ref in method['evidence']:
            s.need(set(ref)=={'path','sha256'} and s.sha(s.ROOT/ref['path'])==ref['sha256'],'Independent method evidence changed')
    return True

def valid_closure(body):
    """Small live closure; full old/new byte preservation is done once at seal."""
    try:
        s.admit_plan();s.need(body['schema']==s.SCHEMA+'-acceptance' and body['passed'] is True and
            body['policy_fixed'] is True and body['base_frozen'] is True and
            body['plan_sha256']==s.sha(s.PLAN) and body['default_engine']=='H_v1' and
            body['promotion'] is False and body['historically_unseen'] is False and
            body['inherited_unknown_background'] is True,'Acceptance scope differs')
        b=s.read(s.FOLDER/'baseline.json')
        s.need(type(body['previous_files_preserved']) is int and body['previous_files_preserved']==b['inputs_count'] and
            all(body['inputs'].get(k)==v for k,v in b['inputs'].items()),'Prior immutable closure differs')
        report=s.read(s.FOLDER/'report.json');audit=s.read(s.FOLDER/'audit/audit.json');audit_valid(audit)
        s.need(body['report_sha256']==s.sha(s.FOLDER/'report.json') and body['audit_sha256']==s.sha(s.FOLDER/'audit/audit.json') and
            config.canonical(body['decision'])==config.canonical(report['decision']), 'Final report/audit decision differs')
        s.need(bool(report['archival_inputs']) and all(body['inputs'].get(k)==v for k,v in report['archival_inputs'].items()),
               'Sealed archival expectations differ from the report')
        required=s.common.pins([s.PLAN,s.FOLDER/'baseline.json',s.FOLDER/'report.json',s.FOLDER/'audit/audit.json',
            s.PREFLIGHT,__file__,s.ROOT/'congklak/ordinary_epoch_progress.py',
            s.ROOT/'tools/ordinary-epoch-dashboard.html'])
        event=sorted((s.FOLDER/'events').glob('*.json'))[-1]
        s.need(body['progress_event']==s.relative(event) and body['progress_event_sha256']==s.sha(event),
            'Latest observer progress is outside the accepted closure')
        required.update(s.common.pins([event]))
        required.update(s.read(s.PLAN)['source_inputs'])
        required.update(report['live_inputs'])
        required.update(audit['source_inputs'])
        for method in audit['methods'].values():required.update({r['path']:r['sha256'] for r in method['evidence']})
        s.need(config.canonical(body['live_inputs'])==config.canonical(required) and
            all(body['inputs'].get(k)==v for k,v in required.items()), 'Required live closure missing')
        s.common.current(required)
        d=report['decision'];g=report['scientific_gates'];derived=derive_gates(report['metrics'],report['statistics'])
        s.need(config.canonical(g)==config.canonical(derived) and
            d['all_scientific_gates_passed'] is all(g.values()) and d['promotion'] is False and
            d['default_engine']=='H_v1','Scientific result contradicts measured gates')
        if report['statistics']:
            s.need(all(g[k] for k in ('fit_coverage','case_coverage','native_conformance','endgame_guard','identical_low_fixed_work',
                'arena_coverage','arena_no_caps_or_fallbacks','whole_component_ownership','equal_time_25ms')),
                'Strength intervals require complete eligible execution')
            s.need(config.canonical(report['statistics'])==config.canonical(paired_statistics(report['game_rows'])),
                'Reported paired uncertainty differs from recorded margins')
        return True
    except (KeyError,TypeError,ValueError,OSError,RuntimeError,IndexError):return False
