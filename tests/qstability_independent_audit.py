"""Independent prospective max-action-Q, ownership and recovery audit."""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import closing
from fractions import Fraction as F
import argparse,ast,copy,hashlib,json,os,shutil,sqlite3,sys,time,unittest,uuid
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from capture_independent_audit import read,sha,save,rel,pins,current,durable,files,qread,rows,physical_rows,backup,mutate_row,run_child,kill,frac,rational,pos,state,raw,bits,report_literal,digest,canonical
import stratum_independent_audit as ancestor_audit
import reference as literal

WORK=ROOT/'build/qstability-audit';FREEZE=ROOT/'build/qstability-audit-source-freeze.json';BASELINE=ROOT/'build/qstability-baseline.json'
STRATA=('50-74/ordinary','25-49/ordinary','25-49/bonus','25-49/capture')
CELLS=tuple(s+'/actor'+str(a) for s in STRATA for a in (0,1))
METRICS=('H10_best_H12_regret_mean','root_drift','action_Q_drift','H10_best_H12_regret_min','H10_best_H12_regret_max','H10_lexicographic_H12_regret','max_action_Q_drift','H10_best_action_Q_drift_contribution','H10_nonbest_action_Q_drift_contribution')
EVIDENCE={}
GROUPS={
 'preflight':['test_01_prior5050_blind_support_components_and_fixed_rule','test_02_exact_max_action_weights_guards_ties_and_censoring','test_03_real_teacher_pre_post_commit_kills_and_exact_resume'],
 'reference':['test_04_all32_references_PVs_and_pre_H12_realized_freeze'],
 'final':['test_05_all_H12_Q_regret_coverage_and_full_histories','test_06_rehashed_corruption_closed_API_CLI_and_prior5050']}


def qvalues(item,actor):
    label=item['label'];q={v['pit']:v['value_player0'] for v in label['action_scores']};sign=1 if actor==0 else -1
    assert len(q)==len(label['action_scores']) and all(type(v) is int for v in q.values())
    best=max(sign*v for v in q.values());assert label['target_player0']==sign*best
    assert label['best_moves']==[a for a in sorted(q) if sign*q[a]==best]
    return q


def worst_weight(row,h8,h10):
    q8=qvalues(h8,row['actor']);q10=qvalues(h10,row['actor']);assert set(q8)==set(q10)
    maximum=max(abs(q10[a]-q8[a]) for a in q8)
    return max(F(1,4),F(4,4+maximum)),maximum


def normalize(raw_weights,rows):
    totals=defaultdict(F)
    for key,w in raw_weights.items():assert w>0;totals[rows[key]['cell']]+=w
    assert set(totals)==set(CELLS)
    return {key:F(1,8)*w/totals[rows[key]['cell']] for key,w in raw_weights.items()}


def independent_root(key,row,projection,h12):
    entry=projection['rows'][key];before=entry['endpoints']['h10'];q10=qvalues(before,row['actor']);best10=before['label']['best_moves']
    result=dict(state_key=key,root_sha256=digest(row),cell=row['cell'],stratum=row['stratum'],actor=row['actor'],family=row['family'],component=row['component'],
        weighted_probability=entry['weighted_probability'],uniform_probability=entry['uniform_probability'],H12_complete=h12 is not None,
        h8_item_sha256=digest(entry['endpoints']['h8']),h10_item_sha256=digest(before),h12_item_sha256=digest(h12) if h12 else None,
        H10_best_moves=best10,H12_best_moves=None,best_sets_changed=None,best_sets_disjoint=None,regrets_by_H10_best_move=None,action_changes=None,ranking=None,metrics=None)
    if h12 is None:return result
    q12=qvalues(h12,row['actor']);assert set(q10)==set(q12);sign=1 if row['actor']==0 else -1;target=h12['label']['target_player0'];best12=h12['label']['best_moves']
    regrets={a:sign*(target-q12[a]) for a in best10};assert min(regrets.values())>=0
    delta={a:q12[a]-q10[a] for a in q10};mean=F(sum(delta.values()),len(delta));change=frac(entry['weighted_probability'])-F(1,16)
    metrics=dict(H10_best_H12_regret_mean=F(sum(regrets.values()),len(regrets)),root_drift=F(abs(target-before['label']['target_player0'])),
        action_Q_drift=F(sum(abs(v) for v in delta.values()),len(delta)),H10_best_H12_regret_min=F(min(regrets.values())),H10_best_H12_regret_max=F(max(regrets.values())),
        H10_lexicographic_H12_regret=F(regrets[min(best10)]),max_action_Q_drift=F(max(abs(v) for v in delta.values())),
        H10_best_action_Q_drift_contribution=F(sum(abs(delta[a]) for a in best10),len(delta)),
        H10_nonbest_action_Q_drift_contribution=F(sum(abs(v) for a,v in delta.items() if a not in best10),len(delta)))
    actions=[];ranks=[];legal=sorted(q10)
    for a in legal:
        actions.append(dict(pit=a,Q10_player0=q10[a],Q12_player0=q12[a],Q10_actor=sign*q10[a],Q12_actor=sign*q12[a],delta_player0=delta[a],delta_actor=sign*delta[a],absolute_delta=abs(delta[a]),
            centered_delta_player0=rational(delta[a]-mean),H10_best=a in best10,H12_best=a in best12,H10_actor_gap=sign*(before['label']['target_player0']-q10[a]),H12_actor_gap=sign*(target-q12[a]),
            H10_best_anchor_gap_changes=[dict(anchor=b,change_actor=sign*(delta[b]-delta[a])) for b in best10],global_weighted_minus_uniform_Q_drift=rational(change*abs(delta[a])/len(delta))))
    for i,a in enumerate(legal):
        for b in legal[i+1:]:
            old=sign*(q10[a]-q10[b]);new=sign*(q12[a]-q12[b]);ranks.append(dict(actions=[a,b],H10_actor_difference=old,H12_actor_difference=new,gap_change_actor=new-old,strict_reversal=old*new<0,tie_changed=(old==0)!=(new==0)))
    result.update(H12_best_moves=best12,best_sets_changed=best10!=best12,best_sets_disjoint=not(set(best10)&set(best12)),
        regrets_by_H10_best_move=[dict(pit=a,regret_seeds=regrets[a]) for a in best10],action_changes=actions,
        ranking=dict(pairs=ranks,strict_reversals=sum(v['strict_reversal'] for v in ranks),tie_changes=sum(v['tie_changed'] for v in ranks),numeric_gap_changes=sum(v['gap_change_actor']!=0 for v in ranks),
            action_shift_mean_player0=rational(mean),action_shift_mean_actor=rational(sign*mean),interpretation='Signed Q/rank diagnostics only; finite H12 is not game truth. Nonbest contributions retain the full legal-action denominator.'),
        metrics={k:rational(v) for k,v in metrics.items()})
    return result


def scope(keys,results):
    keys=sorted(keys);complete=[k for k in keys if results[k]['H12_complete']]
    out=dict(selected_roots=len(keys),complete_roots=len(complete),missing_roots=len(keys)-len(complete),selected_state_keys=keys,complete_state_keys=complete)
    for name in ('uniform','weighted'):
        p={k:frac(results[k][name+'_probability']) for k in keys};mass=sum(p.values(),F());covered=sum((p[k] for k in complete),F())
        partial={metric:sum((p[k]*frac(results[k]['metrics'][metric]) for k in complete),F()) for metric in METRICS}
        risks={metric:rational(partial[metric]/mass) if len(complete)==len(keys) and mass else None for metric in METRICS}
        out[name]=dict(**risks,original_selected_mass=rational(mass),original_complete_mass=rational(covered),original_missing_mass=rational(mass-covered),
            ESS=rational(mass*mass/sum((v*v for v in p.values()),F())) if p else None,max_conditional_probability=rational(max(p.values())/mass) if mass else None,
            complete_risk=risks,covered_unnormalized_totals={k:rational(v) for k,v in partial.items()},
            conditional_complete_diagnostic={k:rational(v/covered) if covered else None for k,v in partial.items()},
            conditional_diagnostic_warning='Explicit covered-support diagnostic only; not the fixed full-cell objective. No missing mass is redistributed.')
    out['weighted_minus_uniform']={metric:rational(frac(out['weighted'][metric])-frac(out['uniform'][metric])) if out['weighted'][metric] is not None else None for metric in METRICS}
    return out


def groups(rows,results):
    return dict(all=scope(rows,results),ordinary=scope([k for k,r in rows.items() if r['stratum'].endswith('/ordinary')],results),
        cells={c:scope([k for k,r in rows.items() if r['cell']==c],results) for c in CELLS},strata={s:scope([k for k,r in rows.items() if r['stratum']==s],results) for s in STRATA},
        actors={str(a):scope([k for k,r in rows.items() if r['actor']==a],results) for a in (0,1)})


def independent_screen(g,projection):
    all_=g['all'];complete=all_['complete_roots']==16
    def guard(x,metric):return frac(x['weighted'][metric])<=frac(x['uniform'][metric]) if complete else None
    masses=dict(cells={c:rational(F(1,8)) for c in CELLS},strata={s:rational(F(1,4)) for s in STRATA},actors={str(a):rational(F(1,2)) for a in (0,1)},ordinary=rational(F(1,2)))
    varies={c:len({frac(projection['rows'][k]['raw_rule']['raw_weight']) for k in g['cells'][c]['selected_state_keys']})>1 for c in CELLS}
    ordinary_strata={k:v for k,v in g['strata'].items() if k.endswith('/ordinary')};ordinary_cells={k:v for k,v in g['cells'].items() if '/ordinary/' in k}
    actor_mass={a:sum((frac(projection['rows'][k]['weighted_probability']) for k in g['ordinary']['selected_state_keys'] if k in g['actors'][a]['selected_state_keys']),F()) for a in ('0','1')}
    c=dict(all16_requested_H8_H10_H12_complete=complete,fixed_cell_stratum_actor_ordinary_masses=projection['masses']['uniform']==projection['masses']['weighted']==masses and all(v==F(1,4) for v in actor_mass.values()),
        nontrivial_within_cell_rule=any(varies.values()),primary_fulltie_regret_nonincrease_overall=guard(all_,'H10_best_H12_regret_mean'),
        primary_fulltie_regret_nonincrease_ordinary_strata={k:guard(v,'H10_best_H12_regret_mean') for k,v in ordinary_strata.items()},primary_fulltie_regret_nonincrease_ordinary_actor_cells={k:guard(v,'H10_best_H12_regret_mean') for k,v in ordinary_cells.items()},
        worst_tie_regret_nonincrease_overall=guard(all_,'H10_best_H12_regret_max'),worst_tie_regret_nonincrease_ordinary_strata={k:guard(v,'H10_best_H12_regret_max') for k,v in ordinary_strata.items()},worst_tie_regret_nonincrease_ordinary_actor_cells={k:guard(v,'H10_best_H12_regret_max') for k,v in ordinary_cells.items()},
        root_drift_nonincrease_overall=guard(all_,'root_drift'),max_action_Q_drift_nonincrease_overall=guard(all_,'max_action_Q_drift'),
        primary_action_Q_drift_strictly_lower_overall=frac(all_['weighted']['action_Q_drift'])<frac(all_['uniform']['action_Q_drift']) if complete else None,
        capture_stratum_action_Q_drift_nonincrease=guard(g['strata']['25-49/capture'],'action_Q_drift'),
        ESS_at_least_four_fifths_overall=frac(all_['weighted']['ESS'])>=F(4,5)*all_['selected_roots'],
        ESS_at_least_four_fifths_all_actor_cells={k:frac(v['weighted']['ESS'])>=F(4,5)*v['selected_roots'] for k,v in g['cells'].items()})
    def flatten(value):return [x for v in value.values() for x in (flatten(v) if isinstance(v,dict) else [v])]
    passed=all(v is True for v in flatten(c))
    return dict(conditions=c,passes=passed,status='passed' if passed else 'not_passed' if complete else 'incomplete_reference_hold',nontrivial_cells=varies,
        equal_score_cells=[cell for cell,changed in varies.items() if not changed],screen_scope='Predeclared descriptive safeguards on selected16assessment roots; not training admission, uncertainty calibration, population inference or promotion',weights_applied=False)


def metric_check(test,rows_,projection,h12):
    from congklak import qstability_metrics as m
    actual=m.evaluate(rows_,projection,h12);results={k:independent_root(k,r,projection,h12.get(k)) for k,r in sorted(rows_.items())};g=groups(rows_,results)
    test.assertEqual(actual['roots'],results)
    test.assertEqual(actual['metrics'],dict(overall=g['all'],ordinary=g['ordinary'],by_cell=g['cells'],by_stratum=g['strata'],by_actor=g['actors']))
    test.assertEqual(actual['screen'],independent_screen(g,projection));return actual


def independent_inventory():
    """Reconstruct the complete ancestor graph, then apply the new full overlay."""
    old=ancestor_audit.independent_inventory();sealed=read(ROOT/'build/stratum-inventory.json')
    for name,value in old.items():assert sealed[name]==value,name
    overlay=read(ROOT/'build/stratum-reservations.json');families=set(overlay['origin_families']);quotients=set(overlay['eligible_quotients']);reserved=set(overlay['components'])
    components={};eligible={};excluded={}
    for cid,body in old['components'].items():
        reasons=list(body['blocked_reasons'])
        if cid in reserved or set(body['families'])&families or set(body['quotients'])&quotients:reasons.append('prior_stratum_consumed_component')
        components[cid]=dict(body,blocked_reasons=reasons)
    for key,row in old['eligible'].items():
        c=components[row['component']]
        if c['blocked_reasons']:excluded[key]=dict(component=row['component'],reasons=c['blocked_reasons'])
        else:eligible[key]=row
    origins=[o for r in eligible.values() for o in r['all_origins']]
    names=('build/stratum-inventory.json','build/stratum-reservations.json','build/stratum-registry.json','build/allocation-weighting-reservations.json')
    return dict(schema='qstability-fixed-cell-max-action-assessment-v1-metadata-inventory',eligible=dict(sorted(eligible.items())),components=components,excluded_roots=dict(sorted(excluded.items())),
        eligible_roots=len(eligible),representative_origin_families=len({r['family'] for r in eligible.values()}),all_origin_families=len({o['source_candidate']['family'] for o in origins}),
        eligible_components=len({r['component'] for r in eligible.values()}),eligible_cells={c:dict(roots=sum(r['cell']==c for r in eligible.values()),origin_families=len({r['family'] for r in eligible.values() if r['cell']==c}),components=len({r['component'] for r in eligible.values() if r['cell']==c})) for c in CELLS},
        duplicate_raw_observations=len(origins)-len(eligible),source_families=old['source_families'],inherited_original_eligible_observations=old['eligible_candidate_observations'],inherited_eligible_roots=old['eligible_roots'],
        prior_stratum_reserved_components=sorted(reserved),prior_stratum_reserved_families=sorted(families),prior_stratum_reserved_quotients=sorted(quotients),ancestral_metadata_inputs={n:sha(ROOT/n) for n in names},H8_H10_target_bodies_read=False,H12_H14_target_bodies_read=False,shared_prefix_independence_claim=False)


def selection(inv):
    rank=lambda key:hashlib.sha256(('qstability-saved-metadata-panel/20261101/v1/'+key).encode()).hexdigest()
    choices={c:sorted((r for r in inv['eligible'].values() if r['cell']==c),key=lambda r:(rank(r['state_key']),r['state_key'])) for c in CELLS};owners={};chosen={}
    def augment(slot,seen):
        for row in choices[slot[0]]:
            component=row['component']
            if component in seen:continue
            seen.add(component)
            if component not in owners or augment(owners[component],seen):owners[component]=slot;chosen[slot]=row;return True
        return False
    for slot in [(c,i) for c in CELLS for i in (0,1)]:assert augment(slot,set()),'No complete independent matching'
    return dict(sorted((r['state_key'],dict(r,group='assessment',future_optimizer_reserved=True)) for r in chosen.values()))


def teacher_semantic(value):
    if isinstance(value,dict):return {k:teacher_semantic(v) for k,v in value.items() if k not in ('elapsed_ms','peak_process_bytes','label_sha256')}
    if isinstance(value,list):return [teacher_semantic(v) for v in value]
    return value


class QStabilityIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True,exist_ok=True);cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()

    def command(self,path,operation,variant='normal'):
        return [sys.executable,str(ROOT/'tools/validate-qstability.py'),operation,'--fixture','--roots','2','--run',str(path),'--variant',variant]

    def clone(self,source,name):
        target=self.base/name;shutil.copytree(source,target);return target

    def worker(self,spec,target):
        target.mkdir();spec=dict(spec,output=str(target/'proof.json'));save(target/'spec.json',spec)
        run_child([sys.executable,str(ROOT/'tools/qstability-audit.py'),str(target/'spec.json')],target/'worker.log',timeout=1200)
        proof=read(target/'proof.json');self.assertTrue(proof['passed']);return dict(path=rel(target/'proof.json'),**proof)

    def closed_worker(self,path,target,fixture=False,variant='normal',watch_prior=False):
        from congklak import qstability_weighting as p
        held=read(path/'manifest.json')['status']=='references_incomplete_hold'
        calls=[dict(function='run',args=[str(path)],kwargs=dict(fixture=fixture,variant=variant)),dict(function='load_rows',args=[str(path)])]
        if not held:calls.append(dict(function='realize',args=[str(path)]))
        extra=[]
        if not fixture:
            calls += [dict(function='build'),dict(function='assess'),dict(function='verify')];extra=[rel(p.OUTPUT)]
            if not held:extra.append(rel(p.RULE))
        args=['--run',str(path)]+(['--fixture'] if fixture else [])+['--variant',variant]
        clis=[dict(argv=['all',*args]),dict(argv=['verify-labels',*args])]
        if not fixture:clis.append(dict(argv=['verify']))
        return self.worker(dict(watch_directories=[str(path)],extra_files=extra,watch_baseline=watch_prior,calls=calls,clis=clis,
            input_names=[rel(p.PLAN),rel(p.IMPLEMENTATION),rel(path/'work.sqlite3'),rel(path/'manifest.json'),rel(FREEZE),'tests/qstability_independent_audit.py']),target)

    def check_work(self,rootmap,stores,stages,settings):
        from congklak import horizon_probe as h
        total=histories=pathcount=choices=complete=capped=endpoint=0;overlap={}
        for stage in stages:
            self.assertEqual(set(stores[stage+'_attempts']),set(rootmap));depth=settings[stage]['depth'];cap=settings[stage]['node_budget']
            for key,row in rootmap.items():
                a=stores[stage+'_attempts'][key];item=stores[stage+'_labels'].get(key)
                self.assertEqual(a['study_row_sha256'],digest(row));self.assertEqual(a['source_root_sha256'],row['source_root_sha256']);self.assertEqual((a['stage'],a['requested_depth']),(stage,depth))
                self.assertEqual([r['choice_depth'] for r in a['history']],list(range(1,a['completed_depth']+1)));self.assertGreater(a['work']['nodes'],0);self.assertLessEqual(a['work']['nodes'],cap);total+=a['work']['nodes']
                if a['status']=='complete':
                    complete+=1;self.assertEqual(a['completed_depth'],depth);self.assertEqual(a['stop_reason'],'max_depth');self.assertEqual(item,h.item_for(a['history'][-1],row));self.assertEqual(a['label_sha256'],digest(item));endpoint+=len(item['label']['action_scores'])
                else:
                    capped+=1;self.assertEqual(a['status'],'budget_rejected');self.assertEqual(a['stop_reason'],'node_budget');self.assertEqual(a['work']['nodes'],cap);self.assertIsNone(item);self.assertIsNone(a['label_sha256']);self.assertLess(a['completed_depth'],depth)
                previous={field:0 for field in a['work']}
                for report in a['history']:
                    literal_report=report_literal(row,report);histories+=1;pathcount+=1+len(literal_report['actions']);choices+=literal_report['root']['choices']+sum(v['choices'] for v in literal_report['actions'].values())
                    for field,value in previous.items():self.assertGreaterEqual(report[field],value);self.assertLessEqual(report[field],a['work'][field])
                    previous={field:report[field] for field in previous};signature=h.q_signature(h.item_for(report,row));at=(key,report['choice_depth'])
                    if at in overlap:self.assertEqual(signature,overlap[at])
                    else:overlap[at]=signature
        return dict(attempts=sum(len(stores[s+'_attempts']) for s in stages),complete=complete,capped=capped,actual_nodes=total,completed_iteration_reports=histories,literal_PV_paths=pathcount,literal_choices=choices,requested_Q_entries=endpoint)

    def test_01_prior5050_blind_support_components_and_fixed_rule(self):
        from congklak import qstability_weighting as p,qstability_metrics as m
        current(read(BASELINE)['inputs']);self.assertEqual(len(read(BASELINE)['inputs']),5050)
        real=sqlite3.connect
        def guarded(*args,**kwargs):
            db=real(*args,**kwargs)
            def authorize(action,table,column,database,trigger):
                if action!=sqlite3.SQLITE_READ or column!='body':return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY if table=='labels' or any(s in table for s in ('h8','h10','h12','h14')) else sqlite3.SQLITE_OK
            db.set_authorizer(authorize);return db
        with mock.patch.object(sqlite3,'connect',side_effect=guarded),mock.patch.object(p.Native,'__init__',side_effect=AssertionError('Blind inventory activated Native')):
            inv=independent_inventory();self.assertEqual(inv,read(p.INVENTORY));self.assertEqual(inv,p.inventory())
            with closing(qread(ROOT/'data/stratum-weighting-v1/work.sqlite3')) as db:
                with self.assertRaises(sqlite3.DatabaseError):db.execute('SELECT body FROM h12_labels LIMIT 1').fetchone()
        chosen=selection(inv);registry=read(p.REGISTRY);self.assertEqual(chosen,registry['rows']);self.assertEqual((inv['eligible_roots'],inv['all_origin_families'],inv['eligible_components']),(3772,536,531))
        self.assertEqual(Counter(r['cell'] for r in chosen.values()),Counter({c:2 for c in CELLS}));self.assertEqual(len({r['component'] for r in chosen.values()}),16);self.assertEqual(len({r['family'] for r in chosen.values()}),16)
        reserve=read(p.RESERVATIONS);self.assertEqual(reserve['registry_sha256'],sha(p.REGISTRY));self.assertEqual(reserve['components'],sorted(r['component'] for r in chosen.values()))
        self.assertEqual(reserve['origin_families'],sorted({f for r in chosen.values() for f in inv['components'][r['component']]['families']}));self.assertEqual(reserve['eligible_quotients'],sorted({q for r in chosen.values() for q in inv['components'][r['component']]['quotients']}))
        old=read(ROOT/'build/stratum-reservations.json');self.assertFalse(set(reserve['origin_families'])&set(old['origin_families']));self.assertFalse(set(reserve['eligible_quotients'])&set(old['eligible_quotients']))
        histories=0
        for key,row in chosen.items():
            self.assertEqual(row['source_cohort'],'training_candidate');self.assertEqual(row['group'],'assessment');self.assertTrue(row['future_optimizer_reserved']);self.assertEqual(raw(row['state']),key)
            self.assertEqual(ancestor_audit.mechanism(row['state'])[1],row['stratum']);self.assertEqual(row['legal_mask'],sum(1<<a for a in literal.legal_actions(pos(row['state']))))
            for origin in row['all_origins']:
                candidate=origin['source_candidate'];self.assertEqual(digest(candidate),origin['source_candidate_sha256']);meta=candidate['provenance'];position=pos(meta['start'])
                for action in meta['moves_prefix']:position=literal.apply(position,action).position;histories+=1
                self.assertEqual(state(position),meta['raw_state']);expected=state(position)
                if meta['swapped']:expected=dict(pits=list(reversed(expected['pits'])),stores=list(reversed(expected['stores'])),controller=1-expected['controller'])
                self.assertEqual(expected,row['state'])
        m.validate_rows(chosen);p.admit();self.assertEqual(p.CONTRACT['maximum_new_nodes'],480000000)
        for stage,depth in [('h8',8),('h10',10),('h12',12)]:self.assertEqual(p.CONTRACT[stage],dict(depth=depth,node_budget=10000000))
        for bad in (1,False):
            if bad is False:continue
            with self.assertRaises(ValueError):p.settings_for(bad)
        with self.assertRaises(ValueError):p.run(self.base/'invalid-production-cap',variant='reference-cap')
        self.assertFalse((self.base/'invalid-production-cap').exists())
        with mock.patch.object(p,'initial_gate',side_effect=ValueError('No independent preflight')):
            with self.assertRaises(ValueError):p.run(stage='h8')
        self.assertFalse(p.DEFAULT.exists());self.assertFalse(p.RULE.exists());self.assertNotIn('torch',sys.modules)
        EVIDENCE['metadata_and_blinding']=dict(prior5050_current=True,full_saved_families=1280,eligible_raw_states=3772,full_origin_families=536,eligible_components=531,
            all_allocation_and_stratum_reservation_components_aliases_excluded=True,full_graph_reconstructed_under_SQL_target_body_denial=True,two_per_cell_distinct16_components=True,literal_saved_history_choices=histories,
            formula_before_new_measurements=True,future_reservation_requires_explicit_optimizer_admission=True,ancestry_independence_not_claimed=True)

    def test_02_exact_max_action_weights_guards_ties_and_censoring(self):
        from congklak import qstability_weighting as p,qstability_metrics as m
        rootmap=read(p.REGISTRY)['rows'];h8=ancestor_audit.synthetic_labels(rootmap,8);h10=ancestor_audit.synthetic_labels(rootmap,10,variable=True);h12=ancestor_audit.synthetic_labels(rootmap,12)
        projection=m.weight_projection(rootmap,h8,h10);rawweights={k:worst_weight(r,h8[k],h10[k])[0] for k,r in rootmap.items()};prob=normalize(rawweights,rootmap)
        for key,row in rootmap.items():
            self.assertEqual(frac(projection['rows'][key]['raw_rule']['raw_weight']),rawweights[key]);self.assertEqual(frac(projection['rows'][key]['weighted_probability']),prob[key]);self.assertEqual(frac(projection['rows'][key]['uniform_probability']),F(1,16))
        actual=metric_check(self,rootmap,projection,h12)
        self.assertTrue(any(len(v['H10_best_moves'])>1 and frac(v['metrics']['H10_best_H12_regret_max'])>frac(v['metrics']['H10_best_H12_regret_min']) for v in actual['roots'].values()))
        key=next(iter(rootmap));missing=dict(h12);missing.pop(key);held=metric_check(self,rootmap,projection,missing)
        self.assertEqual(held['screen']['status'],'incomplete_reference_hold');self.assertFalse(held['screen']['passes']);self.assertGreater(frac(held['metrics']['overall']['weighted']['original_missing_mass']),0)
        for metric in METRICS:self.assertIsNone(held['metrics']['overall']['weighted'][metric])
        # Correct requested-depth metadata is mandatory even when numeric Qs match.
        identical=copy.deepcopy(h8)
        for item in identical.values():item['label']['choice_depth']=10
        uniform=m.weight_projection(rootmap,h8,identical);equal=metric_check(self,rootmap,uniform,h12)
        self.assertFalse(equal['screen']['conditions']['nontrivial_within_cell_rule']);self.assertFalse(equal['screen']['conditions']['primary_action_Q_drift_strictly_lower_overall'])
        for name in ('missing','depth','action'):
            bad=copy.deepcopy(h10)
            if name=='missing':bad.pop(key)
            elif name=='depth':bad[key]['label']['choice_depth']=10.0
            else:bad[key]['label']['action_scores'].pop()
            with self.assertRaises(ValueError):m.weight_projection(rootmap,h8,bad)
        changed=copy.deepcopy(projection);changed['rows'][key]['weighted_probability']=rational(prob[key]+F(1,100))
        with self.assertRaises(ValueError):m.validate_projection(rootmap,changed)
        EVIDENCE['prospective_exact_rule']=dict(max_all_legal_shift_only=True,root_and_mean_diagnostics_not_rule_inputs=True,fixed_cell_stratum_actor_ordinary_masses=True,
            full_bestset_mean_max_min_lowest_pit_actor_correct=True,strict_primary_and_capture_ordinary_max_all8ESS_guards_rederived=True,original_missing_mass_and_null_full_risk=True,uniform_does_not_pass_strict_screen=True,
            exact_Fraction_arithmetic=True,no_calibration_or_future_guarantee=True)

    def test_03_real_teacher_pre_post_commit_kills_and_exact_resume(self):
        from congklak import qstability_weighting as p
        seed=self.base/'old-fixture-h8-seed';run_child(self.command(seed,'h8'),self.base/'seed-h8.log',timeout=360)
        seedhash=sha(seed/'work.sqlite3');record=self.base/'fixture-seed.json';save(record,dict(run=rel(seed),journal_sha256=seedhash,source='First2 previously consumed old diagnostic roots',actual_requested_depths=[2,3,4]))
        fresh=self.clone(seed,'fresh');run_child(self.command(fresh,'h10'),self.base/'fresh-h10.log',timeout=360);p.realize(fresh);run_child(self.command(fresh,'h12'),self.base/'fresh-h12.log',timeout=360)
        rootmap,reference,manifest=p.load_rows(fresh);workproof=self.check_work(rootmap,reference,p.STAGES,manifest['identity']['settings']);tests=[]
        for event,aggregate,count in [('qstability_h10_precommit',3,1),('qstability_h10_committed',4,2)]:
            run=self.clone(seed,'resume-'+str(count));probe=self.base/('probe-'+str(count))
            marker=kill(self.command(run,'h10'),probe,event,aggregate,dict(QSTABILITY_BLOCK_EVENT=event,QSTABILITY_BLOCK_STEP=str(aggregate),QSTABILITY_MARKER=str(probe/'declared-marker.json')))
            backup(run/'work.sqlite3',probe/'validated-prefix.sqlite3');prefix={t:physical_rows(probe/'validated-prefix.sqlite3',t,'state') for t in ('roots',*p.TABLES)}
            self.assertEqual(marker['stage_completed'],count);self.assertEqual(len(prefix['h8_attempts']),2);self.assertEqual(len(prefix['h8_labels']),2);self.assertEqual(len(prefix['h10_attempts']),count);self.assertEqual(len(prefix['h10_labels']),count);self.assertFalse(prefix['h12_attempts']);self.assertFalse(prefix['h12_labels'])
            run_child(self.command(run,'h10'),probe/'resume-h10.log',timeout=360);p.realize(run);run_child(self.command(run,'h12'),probe/'resume-h12.log',timeout=360)
            actualroots,actual,body=p.load_rows(run);self.assertEqual(actualroots,rootmap)
            for table,values in prefix.items():self.assertEqual(values,physical_rows(run/'work.sqlite3',table,'state')[:len(values)])
            self.assertEqual(teacher_semantic(actual),teacher_semantic(reference))
            tests.append(dict(event=event,actual_process_kill=True,committed_prefix=count,committed_aggregate_prefix=aggregate,marker=marker,probe=rel(probe),run=rel(run),seed_record=rel(record),source_seed_journal_sha256=seedhash,committed_prefix_bytes_preserved=True,fresh_semantics_equal=True))
        capped=self.base/'actual-reference-cap';run_child(self.command(capped,'all','reference-cap'),self.base/'reference-cap.log',timeout=360)
        _,capstores,held=p.load_rows(capped);self.assertEqual(held['status'],'references_incomplete_hold');self.assertTrue(held['protocol_closed']);self.assertFalse(held['assessment_available']);self.assertTrue(held['H12_held'])
        for stage in ('h8','h10'):
            self.assertEqual((len(capstores[stage+'_attempts']),len(capstores[stage+'_labels'])),(2,0))
            for attempt in capstores[stage+'_attempts'].values():self.assertEqual(attempt['work']['nodes'],1);self.assertEqual(attempt['status'],'budget_rejected')
        self.assertFalse(capstores['h12_attempts']);self.assertFalse((capped/'rule.json').exists());self.assertFalse((capped/'references-prefix.sqlite3').exists())
        with self.assertRaises(ValueError):p.realize(capped)
        closed=self.closed_worker(fresh,self.base/'fixture-closed',True);capclosed=self.closed_worker(capped,self.base/'cap-closed',True,'reference-cap')
        self.assertFalse(p.DEFAULT.exists());self.assertFalse(p.RULE.exists())
        EVIDENCE['live_teacher_recovery']=dict(tests=tests,fresh=rel(fresh),cap_fixture=rel(capped),seed_record=rel(record),source_seed_journal_sha256=seedhash,actual_fixture_depths=[2,3,4],literal_fixture_work=workproof,
            normal_completed=closed,genuine_one_node_cap_HOLD_completed=capclosed,fixture_caps_not_production=True)

    def test_04_all32_references_PVs_and_pre_H12_realized_freeze(self):
        from congklak import qstability_weighting as p,qstability_metrics as m
        rootmap,stores,manifest=p.load_rows(allow_partial=True);registry=read(p.REGISTRY)['rows'];self.assertEqual(len(rootmap),16);self.assertEqual(set(rootmap),set(registry));self.assertFalse(stores['h12_attempts']);self.assertFalse(stores['h12_labels'])
        for key,row in rootmap.items():
            self.assertEqual(row['root'],registry[key]['source']['source_candidate']);self.assertEqual(row['source_root_sha256'],digest(row['root']));self.assertEqual(row['source_metadata_sha256'],digest(registry[key]));self.assertEqual(row['state'],registry[key]['state']);self.assertEqual(row['source_cohort'],'training_candidate');self.assertEqual(row['cohort'],'assessment')
        proof=self.check_work(rootmap,stores,('h8','h10'),manifest['identity']['settings']);self.assertEqual(proof['attempts'],32);self.assertLessEqual(proof['actual_nodes'],320000000);ready=proof['complete']==32
        refs=dict(H12_ready=ready,H8_labels=len(stores['h8_labels']),H10_labels=len(stores['h10_labels']),endpoint_item_sha256={s:{k:digest(v) for k,v in stores[s+'_labels'].items()} for s in ('h8','h10')},realized_rule=None,references_prefix=None,references_manifest=dict(path=rel(p.DEFAULT/'references-manifest.json'),sha256=sha(p.DEFAULT/'references-manifest.json')))
        if ready:
            rule=read(p.RULE);projection=rule['projection'];self.assertFalse(rule['H12_used']);self.assertEqual(rule['endpoint_item_sha256'],refs['endpoint_item_sha256']);self.assertEqual(projection,m.weight_projection(rootmap,stores['h8_labels'],stores['h10_labels']))
            weights={k:worst_weight(r,stores['h8_labels'][k],stores['h10_labels'][k])[0] for k,r in rootmap.items()};probs=normalize(weights,rootmap)
            for key in rootmap:self.assertEqual(frac(projection['rows'][key]['weighted_probability']),probs[key])
            prefix=p.DEFAULT/'references-prefix.sqlite3'
            for table in ('roots',*p.TABLES):self.assertEqual(physical_rows(prefix,table,'state'),physical_rows(p.DEFAULT/'work.sqlite3',table,'state'))
            self.assertEqual(set(rows(prefix,'meta','key')),{'identity','references_complete'});self.assertNotIn('h12_gate',rows(p.DEFAULT/'work.sqlite3','meta','key'))
            self.assertEqual(rule['references_prefix_sha256'],sha(prefix));self.assertEqual(rule['references_manifest_sha256'],refs['references_manifest']['sha256'])
            refs.update(realized_rule=dict(path=rel(p.RULE),sha256=sha(p.RULE)),references_prefix=dict(path=rel(prefix),sha256=sha(prefix)))
        else:
            self.assertEqual(manifest['status'],'references_incomplete_hold');self.assertTrue(manifest['protocol_closed']);self.assertFalse(manifest['assessment_available']);self.assertFalse(p.RULE.exists());self.assertFalse((p.DEFAULT/'references-prefix.sqlite3').exists())
        EVIDENCE['references']=refs;EVIDENCE['reference_validation']=dict(**proof,H12_attempts_before_GO=0,live_DB_not_current_pinned=True,all32_endpoint_hashes_and_fixed_probabilities_verified=ready,finite_reference_not_game_truth=True)

    def test_05_all_H12_Q_regret_coverage_and_full_histories(self):
        from congklak import qstability_weighting as p
        rootmap,stores,manifest=p.load_rows();reference=read(WORK/'reference.json');current(reference['inputs']);self.assertNotIn(rel(p.DEFAULT/'work.sqlite3'),reference['inputs'])
        report=p.verify();self.assertFalse(report['training_performed']);self.assertFalse(report['weights_applied']);self.assertFalse(report['promotion']);self.assertTrue(report['future_optimizer_reserved']);self.assertFalse(report['calibrated_confidence_claim']);self.assertFalse(report['H12_is_game_truth'])
        if not reference['evidence']['references']['H12_ready']:
            self.assertEqual(manifest['status'],'references_incomplete_hold');self.assertFalse(stores['h12_attempts']);self.assertFalse(p.RULE.exists());self.assertIsNone(report['metrics']);self.assertIsNone(report['assessment']['weighted_probability']);self.assertFalse(report['screen']['available']);self.assertFalse(report['assessment_available'])
            EVIDENCE['full_assessment']=dict(HOLD_missing_H8_H10=True,H12_attempts=0,weighted_distribution_available=False,weights_applied=False);return
        proof=self.check_work(rootmap,stores,p.STAGES,manifest['identity']['settings']);self.assertEqual(proof['attempts'],48);self.assertLessEqual(proof['actual_nodes'],480000000);self.assertEqual(manifest['total_new_nodes'],proof['actual_nodes']);self.assertEqual(manifest['auxiliary_completed_reports'],proof['completed_iteration_reports'])
        for table in ('roots','h8_attempts','h8_labels','h10_attempts','h10_labels'):self.assertEqual(physical_rows(p.DEFAULT/'references-prefix.sqlite3',table,'state'),physical_rows(p.DEFAULT/'work.sqlite3',table,'state'))
        actual=metric_check(self,rootmap,read(p.RULE)['projection'],stores['h12_labels']);self.assertEqual(report['assessment'],actual);self.assertEqual(report['metrics'],actual['metrics']);self.assertEqual(report['screen'],actual['screen']);current(report['inputs'])
        self.assertEqual(report['realized_rule_sha256'],sha(p.RULE));self.assertEqual(report['manifest_sha256'],sha(p.DEFAULT/'manifest.json'))
        contribution=sum((frac(a['global_weighted_minus_uniform_Q_drift']) for r in actual['roots'].values() if r['H12_complete'] for a in r['action_changes']),F())
        if actual['complete_roots']==16:self.assertEqual(contribution,frac(actual['metrics']['overall']['weighted_minus_uniform']['action_Q_drift']))
        EVIDENCE['full_assessment']=dict(**proof,H12_attempts=16,H12_complete=len(stores['h12_labels']),H12_capped=16-len(stores['h12_labels']),all_exact_root_action_rank_cell_stratum_actor_and_ordinary_metrics=True,screen=actual['screen'],
            original_missing_mass_not_renormalized=True,per_action_global_contribution=rational(contribution),full_ties_mean_worst_and_lowest_pit_reported=True,no_weights_applied=True)

    def test_06_rehashed_corruption_closed_API_CLI_and_prior5050(self):
        from congklak import qstability_weighting as p
        pre=read(WORK/'preflight.json');fresh=ROOT/pre['evidence']['live_teacher_recovery']['fresh'];rootmap,stores,manifest=p.load_rows(fresh);key=next(iter(rootmap));checks=[]
        for name,table,ident,column,mutate in [
            ('identity-bool','meta','identity','key',lambda b:b.update(fixture=1)),
            ('ownership','roots',key,'state',lambda b:b.update(source_cohort='invalid')),
            ('history-gap','h12_attempts',key,'state',lambda b:b['history'].pop(0)),
            ('history-PVS','h12_attempts',key,'state',lambda b:b['history'][0]['search_options'].update(use_pvs=False)),
            ('label-Q','h12_labels',key,'state',lambda b:b['label']['action_scores'][0].update(value_player0=999)),
            ('complete-marker','meta','complete','key',lambda b:b.update(total_new_nodes=0))]:
            bad=self.clone(fresh,'bad-'+name);(bad/'manifest.json').unlink();mutate_row(bad/'work.sqlite3',table,ident,column,mutate)
            with self.assertRaises((ValueError,FileNotFoundError,sqlite3.OperationalError)):p.run(bad,fixture=True)
            self.assertFalse((bad/'manifest.json').exists());checks.append(name)
        for name in ('references-prefix.sqlite3','references-manifest.json','rule.json'):
            bad=self.clone(fresh,'missing-'+name.replace('.','-'));(bad/name).unlink();(bad/'manifest.json').unlink()
            with self.assertRaises((ValueError,FileNotFoundError,sqlite3.OperationalError)):p.run(bad,fixture=True)
            self.assertFalse((bad/'manifest.json').exists());checks.append('missing-required-'+name)
        repaired=self.clone(fresh,'valid-final-export-repair');expected=(repaired/'manifest.json').read_bytes();(repaired/'manifest.json').unlink()
        with mock.patch.object(p.history,'execute',side_effect=AssertionError('Repair searched')):p.run(repaired,fixture=True)
        self.assertEqual((repaired/'manifest.json').read_bytes(),expected)
        original=p.read;report=read(p.OUTPUT)
        for kind in ('metric','source'):
            bad=copy.deepcopy(report)
            if kind=='source':bad['inputs'][rel(p.REGISTRY)]='0'*64
            elif bad['metrics'] is None:bad['assessment']['weighted_probability']=rational(F(1,16))
            else:bad['metrics']['overall']['weighted']['action_Q_drift']=rational(999)
            save(self.base/('rehashed-'+kind+'.json'),dict(body=bad,sha256=digest(bad)))
            with mock.patch.object(p,'read',side_effect=lambda path,b=bad:b if Path(path).resolve()==p.OUTPUT.resolve() else original(path)):
                with self.assertRaises(ValueError):p.verify()
        alternate=self.base/'missing-new-assessment.json'
        with mock.patch.object(p,'OUTPUT',alternate):
            p.assess();first=(sha(alternate),alternate.stat().st_size,alternate.stat().st_mtime_ns);p.assess();p.verify();self.assertEqual(first,(sha(alternate),alternate.stat().st_size,alternate.stat().st_mtime_ns))
        self.assertEqual(read(alternate),report)
        if p.RULE.exists():
            altered=copy.deepcopy(read(p.RULE));altered['H12_used']=True;altered['projection']['H12_used']=True
            with mock.patch.object(p,'read',side_effect=lambda path:altered if Path(path).resolve()==p.RULE.resolve() else original(path)):
                with self.assertRaises(ValueError):p.load_rows()
        proof=self.closed_worker(p.DEFAULT,self.base/'production-completed',watch_prior=True)
        for field in ('new_teacher_searches','new_PUCT_searches','new_model_forwards','new_training_updates','new_games','new_attempt_commits'):self.assertEqual(proof[field],0)
        self.assertTrue(proof['all_bytes_sizes_mtimes_unchanged']);self.assertFalse(proof['torch_imported'])
        old=ast.parse((ROOT/'congklak/capture_progress.py').read_text(encoding='utf-8-sig'));new=ast.parse((ROOT/'congklak/qstability_progress.py').read_text(encoding='utf-8-sig'))
        for name in ('save','event_message','runs','run_command'):
            before=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name);after=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name);self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(after,include_attributes=False))
        current(read(BASELINE)['inputs']);self.assertNotIn('torch',sys.modules)
        EVIDENCE['completed_refusal_and_preservation']=dict(rehashed_SQL_refusals=checks,rehashed_metrics_sources_and_H12_lookahead_refused=True,required_phase_files_not_rebuilt=True,valid_final_export_repaired=True,missing_new_assessment_stable=True,
            completed_worker=proof,prior5050_bytes_sizes_mtimes_preserved=True,model_policy_fixed=True,no_training_search_forwards_or_new_games=True,saved_Native_transition_validation_allowed=True,zero_total_Native_calls_claim=False,logger4_AST_identity=True,old_logger_tests_not_recounted=True)


def source_inputs():
    from congklak import qstability_weighting as p
    mapping=dict(read(BASELINE)['inputs']);names={rel(__file__),'tools/qstability-audit.py',rel(BASELINE),*p.SOURCES,*p.DEPENDENCIES,rel(p.INVENTORY),rel(p.REGISTRY),rel(p.RESERVATIONS),rel(p.PLAN),rel(p.IMPLEMENTATION),
        'build/qstability-metrics-implementation.json','build/qstability-metrics-author.json','build/qstability-weighting-author.json','congklak/qstability_progress.py','tools/qstability-dashboard.html'}
    mapping.update({n:sha(ROOT/n) for n in names});return dict(sorted(mapping.items()))


def collect():
    current(read(FREEZE)['inputs']);inputs={};evidence={};covered=[];groupentries=[]
    for name,methods in GROUPS.items():
        path=WORK/(name+'.json');body=read(path);selfsource=dict(path=rel(__file__),sha256=sha(__file__))
        assert body['schema']=='qstability-independent-group-v1' and body['group']==name and body['passed'] is True and body['tests_run']==len(methods)
        assert body['covered_methods']==methods and body['covered_checks']==['__main__.QStabilityIndependentAudit.'+m for m in methods] and not any(body[k] for k in ('errors','failures','skips'))
        current(body['inputs'])
        for key,value in body['inputs'].items():assert key not in inputs or inputs[key]==value,key;inputs[key]=value
        assert not set(evidence)&set(body['evidence']);evidence.update(body['evidence']);covered+=methods;inputs[rel(path)]=sha(path)
        groupentries.append(dict(path=rel(path),sha256=sha(path),tests_run=len(methods),covered_methods=methods,executed_source=selfsource))
    assert len(covered)==len(set(covered))==6
    body=dict(schema='qstability-independent-audit-v1',passed=True,tests_run=6,errors=0,failures=0,skips=0,covered_methods=covered,covered_checks=['__main__.QStabilityIndependentAudit.'+m for m in covered],groups=groupentries,inputs=dict(sorted(inputs.items())),evidence=evidence)
    save(ROOT/'build/qstability-audit.json',body);return body


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('freeze',*GROUPS,'collect'));args=parser.parse_args()
    if args.operation=='freeze':save(FREEZE,dict(schema='qstability-independent-executed-source-v1',groups=GROUPS,inputs=source_inputs()));return 0
    if args.operation=='collect':collect();return 0
    frozen=read(FREEZE);assert frozen['groups']==GROUPS;current(frozen['inputs']);output=WORK/(args.operation+'.json');assert not output.exists(),'Immutable group exists'
    started=time.monotonic();methods=GROUPS[args.operation];result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(QStabilityIndependentAudit(name) for name in methods))
    from congklak import qstability_weighting as p
    inputs={**frozen['inputs'],rel(FREEZE):sha(FREEZE)};retained=getattr(QStabilityIndependentAudit,'base',None);captured=files(retained) if retained else []
    if args.operation=='reference':
        captured+=[WORK/'preflight.json',p.DEFAULT/'references-manifest.json'];captured += [x for x in (p.RULE,p.DEFAULT/'references-prefix.sqlite3') if x.exists()]
    if args.operation=='final':captured+=files(p.DEFAULT)+[p.OUTPUT,WORK/'preflight.json',WORK/'reference.json'];captured += [p.RULE] if p.RULE.exists() else []
    inputs.update(pins(captured));body=dict(schema='qstability-independent-group-v1',group=args.operation,passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),covered_methods=methods,covered_checks=['__main__.QStabilityIndependentAudit.'+m for m in methods],duration_seconds=time.monotonic()-started,retained_directory=rel(retained) if retained else None,inputs=dict(sorted(inputs.items())),evidence=EVIDENCE,failure_details=[dict(method=str(test),trace=trace) for test,trace in result.errors+result.failures])
    save(output,body);print(canonical(dict(event='qstability_independent_group',group=args.operation,passed=body['passed'],tests=result.testsRun)),flush=True);return int(not body['passed'])


if __name__=='__main__':raise SystemExit(main())
