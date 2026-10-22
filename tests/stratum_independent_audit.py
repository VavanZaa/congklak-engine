"""Independent metadata, exact within-cell mass and finite-reference audit."""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import closing
from fractions import Fraction as F
import argparse, ast, copy, hashlib, json, os, shutil, sqlite3, sys, time, unittest, uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
from congklak import config
from data_pilot_independent_audit import sha, load, save, relative, pins, dbopen, norm, position, mechanism, run_child
from divergence_independent_audit import backup, kill, sql_rows, patch_sql, bits, teacher_semantic
from natural_study_independent_audit import check_pvs
from allocation_independent_audit import raw, quotient, metadata_rows, requested_keys, exact_weight, rational, from_rational
import reference as literal

WORK = ROOT / 'build/stratum-audit'
BASELINE = ROOT / 'build/stratum-baseline.json'
FREEZE = ROOT / 'build/stratum-audit-source-freeze.json'
STRATA = ('50-74/ordinary', '25-49/ordinary', '25-49/bonus', '25-49/capture')
CELLS = tuple(s + '/actor' + str(a) for s in STRATA for a in (0, 1))
EVIDENCE = {}
GROUPS = {
 'preflight': ['test_01_prior4546_metadata_components_panel_and_blind_formula',
               'test_02_exact_cell_mass_bestset_and_missing_reference_probes',
               'test_03_real_teacher_pre_post_commit_kills_and_exact_resume'],
 'rule': ['test_04_all32_H8_H10_labels_PVs_and_blind_realized_weights'],
 'labels': ['test_05_all16_H12_labels_histories_and_literal_PVs',
            'test_06_exact_within_cell_regret_coverage_and_missing_mass'],
 'final': ['test_07_rehashed_corruption_lookahead_and_missing_export_refusal',
           'test_08_closed_API_CLI_no_work_models_and_prior4546_preserved']}


def probabilities(raw_weights, cells):
    """Fixed 1/8 cell mass; no label-completion argument can affect weights."""
    assert set(raw_weights) == set(cells)
    totals = defaultdict(F)
    for key, value in raw_weights.items():
        assert value > 0; totals[cells[key]] += value
    assert set(totals) == set(CELLS)
    return {key: F(1, 8) * value / totals[cells[key]] for key, value in raw_weights.items()}


def endpoint_errors(h10, h12, actor):
    """Player-0 full-Q values, selected by the actual root actor."""
    a, b = h10['label'], h12['label']
    q10 = {r['pit']: r['value_player0'] for r in a['action_scores']}
    q12 = {r['pit']: r['value_player0'] for r in b['action_scores']}
    assert set(q10) == set(q12) and type(actor) is int and actor in (0, 1)
    best = (max if actor == 0 else min)(q12.values())
    regrets = {str(p): (best - q12[p] if actor == 0 else q12[p] - best) for p in a['best_moves']}
    assert regrets and min(regrets.values()) >= 0
    return dict(root_drift=F(abs(b['target_player0'] - a['target_player0'])),
                action_Q_drift=F(sum(abs(q12[p] - q10[p]) for p in q10), len(q10)),
                H10_best_H12_regret_min=F(min(regrets.values())),
                H10_best_H12_regret_max=F(max(regrets.values())),
                H10_best_H12_regret_mean=F(sum(regrets.values()), len(regrets)),
                H10_best_H12_regrets=regrets)


METRICS = ('H10_best_H12_regret_mean', 'root_drift', 'action_Q_drift',
           'H10_best_H12_regret_min', 'H10_best_H12_regret_max', 'H10_lexicographic_H12_regret')


def independent_scope(keys, results):
    """Selected-mass risks; missing references cannot acquire zero errors."""
    keys = sorted(keys); complete = [k for k in keys if results[k]['H12_complete']]
    out = dict(selected_roots=len(keys), complete_roots=len(complete), missing_roots=len(keys)-len(complete),
               selected_state_keys=keys, complete_state_keys=complete)
    for comparator in ('uniform', 'weighted'):
        p = {k: from_rational(results[k][comparator+'_probability']) for k in keys}
        mass = sum(p.values(), F()); covered = sum((p[k] for k in complete), F())
        totals = {field: sum((p[k]*from_rational(results[k]['metrics'][field]) for k in complete), F()) for field in METRICS}
        full={field:rational(totals[field]/mass) if len(complete)==len(keys) and mass else None for field in METRICS}
        out[comparator] = dict(**full, original_selected_mass=rational(mass), original_complete_mass=rational(covered),
            original_missing_mass=rational(mass-covered), ESS=rational(mass*mass/sum((v*v for v in p.values()), F())) if p else None,
            max_conditional_probability=rational(max(p.values())/mass) if p and mass else None,
            complete_risk=full,
            covered_unnormalized_totals={field:rational(totals[field]) for field in METRICS},
            conditional_complete_diagnostic={field:rational(totals[field]/covered) if covered else None for field in METRICS},
            conditional_diagnostic_warning='Explicit covered-support diagnostic only; not the fixed full-cell objective. No missing mass is redistributed.')
    out['weighted_minus_uniform'] = {field:rational(from_rational(out['weighted']['complete_risk'][field])-from_rational(out['uniform']['complete_risk'][field]))
                                      if out['weighted']['complete_risk'][field] is not None else None for field in METRICS}
    return out


def independent_groups(rows, results):
    return dict(all=independent_scope(rows, results),
        ordinary=independent_scope([k for k,r in rows.items() if r['stratum'].endswith('/ordinary')],results),
        cells={c:independent_scope([k for k,r in rows.items() if r['cell']==c],results) for c in CELLS},
        strata={s:independent_scope([k for k,r in rows.items() if r['stratum']==s],results) for s in STRATA},
        actors={str(a):independent_scope([k for k,r in rows.items() if r['actor']==a],results) for a in (0,1)})


def independent_screen(groups, projection):
    overall = groups['all']; complete = overall['complete_roots'] == 16
    strata = {s:groups['strata'][s] for s in STRATA if s.endswith('/ordinary')}
    cells = {c:groups['cells'][c] for c in CELLS if '/ordinary/' in c}
    varies = {c:len({from_rational(row['raw_rule']['raw_weight']) for k,row in projection['rows'].items()
                         if k in groups['cells'][c]['selected_state_keys']}) > 1 for c in CELLS}
    def guard(scope, field):
        return from_rational(scope['weighted']['complete_risk'][field]) <= from_rational(scope['uniform']['complete_risk'][field]) if complete else None
    expected = dict(cells={c:rational(F(1,8)) for c in CELLS}, strata={s:rational(F(1,4)) for s in STRATA},
                    actors={str(a):rational(F(1,2)) for a in (0,1)}, ordinary=rational(F(1,2)))
    actual_ordinary = {str(a):sum((from_rational(v['weighted_probability']) for k,v in projection['rows'].items()
                        if k in groups['ordinary']['selected_state_keys'] and k in groups['actors'][str(a)]['selected_state_keys']),F()) for a in (0,1)}
    condition = dict(all16_requested_H8_H10_H12_complete=complete,
        fixed_cell_stratum_actor_ordinary_masses=projection['masses']['uniform']==projection['masses']['weighted']==expected
                                                   and all(v==F(1,4) for v in actual_ordinary.values()),
        nontrivial_within_cell_rule=any(varies.values()),
        primary_fulltie_regret_nonincrease_overall=guard(overall,'H10_best_H12_regret_mean'),
        primary_fulltie_regret_nonincrease_ordinary_strata={k:guard(v,'H10_best_H12_regret_mean') for k,v in strata.items()},
        primary_fulltie_regret_nonincrease_ordinary_actor_cells={k:guard(v,'H10_best_H12_regret_mean') for k,v in cells.items()},
        worst_tie_regret_nonincrease_overall=guard(overall,'H10_best_H12_regret_max'),
        worst_tie_regret_nonincrease_ordinary_strata={k:guard(v,'H10_best_H12_regret_max') for k,v in strata.items()},
        worst_tie_regret_nonincrease_ordinary_actor_cells={k:guard(v,'H10_best_H12_regret_max') for k,v in cells.items()},
        root_drift_nonincrease_overall=guard(overall,'root_drift'),action_Q_drift_nonincrease_overall=guard(overall,'action_Q_drift'),
        at_least_one_overall_risk_strictly_improves=any(from_rational(overall['weighted']['complete_risk'][m]) < from_rational(overall['uniform']['complete_risk'][m])
              for m in ('H10_best_H12_regret_mean','H10_best_H12_regret_max','root_drift','action_Q_drift')) if complete else None,
        ESS_at_least_four_fifths_overall=from_rational(overall['weighted']['ESS']) >= F(4,5)*overall['selected_roots'],
        ESS_at_least_four_fifths_ordinary_actor_cells={k:from_rational(v['weighted']['ESS']) >= F(4,5)*v['selected_roots'] for k,v in cells.items()})
    def flatten(obj):
        return [value for v in obj.values() for value in (flatten(v) if isinstance(v,dict) else [v])]
    passes = all(v is True for v in flatten(condition))
    return dict(conditions=condition, passes=passes, status='passed' if passes else 'not_passed' if complete else 'incomplete_reference_hold',
        nontrivial_cells=varies, screen_scope='Predeclared descriptive safeguards on selected16assessment roots; not training admission, uncertainty calibration, population inference or promotion', weights_applied=False)


def synthetic_labels(rows, depth, variable=False, tied=True):
    out = {}
    for i,(key,row) in enumerate(sorted(rows.items())):
        legal=[a for a in range(7) if row['legal_mask']&(1<<a)]
        d=row['state']['stores'][0]-row['state']['stores'][1];sign=1 if row['actor']==0 else -1
        q={a:d-sign for a in legal}
        best=legal[:2] if tied else legal[:1]
        for a in best:q[a]=d+sign*(2+(i%2 if variable else 0))
        if depth==12:
            q[best[-1]]=d-sign*3
        target=(max if row['actor']==0 else min)(q.values())
        out[key]=dict(label=dict(choice_depth=depth,target_player0=target,
            action_scores=[dict(pit=a,value_player0=q[a]) for a in legal],best_moves=[a for a in legal if q[a]==target]))
    return out


def expected_metrics(rows, projection, h12):
    results={}
    for key,row in sorted(rows.items()):
        entry=projection['rows'][key];q10=entry['endpoints']['h10']['label'];target=h12.get(key)
        result=dict(state_key=key,root_sha256=config.digest(row),cell=row['cell'],stratum=row['stratum'],actor=row['actor'],family=row['family'],
            component=row['component'],weighted_probability=entry['weighted_probability'],uniform_probability=entry['uniform_probability'],H12_complete=target is not None,
            h8_item_sha256=config.digest(entry['endpoints']['h8']),h10_item_sha256=config.digest(entry['endpoints']['h10']),h12_item_sha256=config.digest(target) if target else None,
            H10_best_moves=q10['best_moves'],H12_best_moves=None,best_sets_changed=None,best_sets_disjoint=None,regrets_by_H10_best_move=None,metrics=None)
        if target is not None:
            values=endpoint_errors(entry['endpoints']['h10'],target,row['actor']);regrets=values.pop('H10_best_H12_regrets')
            values['H10_lexicographic_H12_regret']=F(regrets[str(min(q10['best_moves']))])
            best12=target['label']['best_moves']
            result.update(H12_best_moves=best12,best_sets_changed=q10['best_moves']!=best12,best_sets_disjoint=not(set(q10['best_moves'])&set(best12)),
                regrets_by_H10_best_move=[dict(pit=a,regret_seeds=regrets[str(a)]) for a in q10['best_moves']],metrics={k:rational(v) for k,v in values.items()})
        results[key]=result
    groups=independent_groups(rows,results)
    return dict(roots=results,metrics=dict(overall=groups['all'],ordinary=groups['ordinary'],by_cell=groups['cells'],by_stratum=groups['strata'],by_actor=groups['actors']),
                screen=independent_screen(groups,projection))


class StratumIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True,exist_ok=True);cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()

    def current(self, values):
        for name,checksum in values.items(): self.assertEqual(sha(ROOT/name),checksum,name)

    def worker(self,spec,path):
        path.mkdir();spec=dict(spec,output=str(path/'proof.json'));save(path/'spec.json',spec)
        run_child([sys.executable,str(ROOT/'tools/stratum-audit.py'),str(path/'spec.json')],path/'worker.log',timeout=900)
        proof=load(path/'proof.json');self.assertTrue(proof['passed']);return dict(path=relative(path/'proof.json'),**proof)

    def clone(self,source,name):
        target=self.base/name;shutil.copytree(source,target);return target

    def command(self,path,stage):
        return [sys.executable,str(ROOT/'tools/validate-stratum-weighting.py'),stage,'--fixture','--run',str(path)]

    def closed_worker(self,path,target,fixture=False,watch_prior=False):
        from congklak import stratum_weighting as w
        incomplete=not fixture and not w.RULE.exists()
        calls=[dict(function='run',args=[str(path)],kwargs=dict(fixture=fixture,stage='h10' if incomplete else 'all')),
               dict(function='load_rows',args=[str(path)],kwargs=dict(allow_partial=incomplete))]
        if not incomplete:calls.append(dict(function='realize',args=[str(path)]))
        extra=[]
        if not fixture and not incomplete:calls += [dict(function='assess'),dict(function='verify')];extra=[relative(w.OUTPUT),relative(w.RULE)]
        argv=['verify-labels','--run',str(path)]+(['--allow-partial'] if incomplete else [])
        manifest=path/('references-manifest.json' if incomplete else 'manifest.json')
        return self.worker(dict(module=w.__name__,watch_directories=[str(path)],extra_files=extra,watch_baseline=watch_prior,
            calls=calls,cli='tools/validate-stratum-weighting.py',argv=argv,forbidden_attributes=['history.execute'],
            input_names=[relative(w.PLAN),relative(w.IMPLEMENTATION),relative(path/'work.sqlite3'),relative(manifest),relative(FREEZE),
                         'tests/stratum_independent_audit.py']),target)

    def check_label_work(self,rootmap,stores,stages,settings):
        from congklak import reports,horizon_probe as history
        from congklak.native import Native
        native=Native();total=histories=paths=choices=complete=capped=endpointQ=0;overlaps={}
        for stage in stages:
            attempts=stores[stage+'_attempts'];labels=stores[stage+'_labels'];self.assertEqual(set(attempts),set(rootmap))
            depth=settings[stage]['depth'];cap=settings[stage]['node_budget']
            for key,row in rootmap.items():
                a=attempts[key];item=labels.get(key);self.assertEqual(a['study_row_sha256'],config.digest(row));self.assertEqual(a['source_root_sha256'],row['source_root_sha256'])
                self.assertEqual((a['stage'],a['requested_depth']),(stage,depth));self.assertEqual([r['choice_depth'] for r in a['history']],list(range(1,a['completed_depth']+1)))
                self.assertGreater(a['work']['nodes'],0);self.assertLessEqual(a['work']['nodes'],cap);total+=a['work']['nodes']
                if a['status']=='complete':
                    complete+=1;self.assertEqual(a['completed_depth'],depth);self.assertEqual(a['stop_reason'],'max_depth');self.assertEqual(item,history.item_for(a['history'][-1],row));endpointQ+=len(item['label']['action_scores'])
                else:
                    capped+=1;self.assertEqual(a['status'],'budget_rejected');self.assertEqual(a['stop_reason'],'node_budget');self.assertEqual(a['work']['nodes'],cap)
                    self.assertIsNone(item);self.assertIsNone(a['label_sha256']);self.assertLess(a['completed_depth'],depth)
                last={k:0 for k in a['work']}
                for report in a['history']:
                    reports.validate(report,row['state'],config.identity(native.build_id),native)
                    self.assertEqual((report['search_options']['tt_mb'],report['search_options']['use_pvs']),(16,True))
                    for field,value in last.items():self.assertGreaterEqual(report[field],value);self.assertLessEqual(report[field],a['work'][field])
                    last={k:report[k] for k in last};v=history.item_for(report,row)
                    n,m=check_pvs(self,row,v,report['choice_depth']);paths+=n;choices+=m;histories+=1
                    signature=history.q_signature(v);at=(key,report['choice_depth'])
                    if at in overlaps:self.assertEqual(signature,overlaps[at])
                    else:overlaps[at]=signature
                    for oldstage in stages:
                        if oldstage!=stage and settings[oldstage]['depth']==report['choice_depth'] and key in stores[oldstage+'_labels']:
                            self.assertEqual(history.q_signature(v),history.q_signature(stores[oldstage+'_labels'][key]))
        return dict(attempts=complete+capped,complete=complete,capped=capped,actual_nodes=total,completed_iteration_reports=histories,
                    endpoint_legal_Q_entries=endpointQ,literal_PV_paths=paths,literal_PV_choices=choices,
                    full_Q_bonus_actors_and_source_links_verified=True,auxiliary_not_requested_labels=True)

    def metric_check(self,rows,projection,h12):
        from congklak import stratum_metrics as m
        actual=m.evaluate(rows,projection,h12);expected=expected_metrics(rows,projection,h12)
        for field,value in expected.items():self.assertEqual(actual[field],value,field)
        self.assertEqual(actual['counts'],dict(selected_roots=16,H8_complete=16,H10_complete=16,H12_complete=len(h12),H12_missing=16-len(h12),complete_pairs=len(h12)))
        self.assertFalse(actual['weights_applied']);self.assertFalse(actual['training_performed'])
        return actual

    def test_01_prior4546_metadata_components_panel_and_blind_formula(self):
        from congklak import stratum_weighting as w,stratum_metrics as m
        from congklak import allocation_weighting as ancestor
        baseline=load(BASELINE);self.assertEqual((baseline['inputs_count'],baseline['prior_accepted_inputs'],len(baseline['inputs'])),(4546,4543,4546));self.current(baseline['inputs'])
        a=load(ROOT/'data/quality-train-v1/model.json')['model']['weights'];b=load(ROOT/'data/ordinary-value-pilot-v1/model.json')['model']['weights']
        self.assertEqual({k for k in a if bits(a[k])!=bits(b[k])},{'layers.4.weight','layers.4.bias'})
        inv=independent_inventory();actual=load(w.INVENTORY)
        for key,value in inv.items():self.assertEqual(actual[key],value,key)
        self.assertEqual((inv['eligible_roots'],inv['all_origin_families'],inv['eligible_components']),(3890,553,547))
        selected=independent_selection(inv);registry=load(w.REGISTRY);self.assertEqual(registry['rows'],selected)
        self.assertEqual(Counter(r['cell'] for r in selected.values()),Counter({c:2 for c in CELLS}))
        self.assertEqual(len({r['component'] for r in selected.values()}),16);self.assertEqual(len({r['family'] for r in selected.values()}),16)
        reserved=load(w.RESERVATIONS);self.assertEqual(reserved['registry_sha256'],sha(w.REGISTRY));self.assertFalse(reserved['historical_roles_rewritten'])
        self.assertEqual(reserved['selected_state_keys'],sorted(selected));self.assertEqual(reserved['components'],sorted(r['component'] for r in selected.values()))
        self.assertEqual(reserved['origin_families'],sorted({f for r in selected.values() for f in inv['components'][r['component']]['families']}))
        self.assertEqual(reserved['eligible_quotients'],sorted({q for r in selected.values() for q in inv['components'][r['component']]['quotients']}))
        replay_choices=0
        for key,row in selected.items():
            self.assertEqual(row['source_cohort'],'training_candidate');self.assertEqual(row['group'],'assessment');self.assertTrue(row['future_optimizer_reserved'])
            self.assertEqual(raw(row['state']),key);self.assertEqual(mechanism(row['state'])[1],row['stratum'])
            legal=list(literal.legal_actions(norm(row['state'])));self.assertEqual(row['legal_mask'],sum(1<<a for a in legal));self.assertGreaterEqual(len(legal),2)
            source=row['source'];self.assertEqual(config.digest(source['source_candidate']),source['source_candidate_sha256'])
            meta=source['source_candidate']['provenance'];state=norm(meta['start'])
            for action in meta['moves_prefix']:
                self.assertIn(action,literal.legal_actions(state));state=literal.apply(state,action).position;replay_choices+=1
            self.assertEqual(position(state),meta['raw_state'])
            expected=position(state)
            if meta['swapped']:expected=dict(pits=list(reversed(expected['pits'])),stores=list(reversed(expected['stores'])),controller=1-expected['controller'])
            self.assertEqual(expected,row['state'])
        # Explicit denial covers any target bodies, including old H8/H10, while
        # allowing metadata roots/candidates and key-only exposure queries.
        def guard(db):
            def authorize(action,table,column,database,trigger):
                if action!=sqlite3.SQLITE_READ or column!='body':return sqlite3.SQLITE_OK
                target=table=='labels' or any(s in table for s in ('h8','h10','h12','h14'))
                return sqlite3.SQLITE_DENY if target else sqlite3.SQLITE_OK
            db.set_authorizer(authorize);return db
        aro=ancestor.ro;wro=w.ro
        with mock.patch.object(ancestor,'ro',side_effect=lambda p:guard(aro(p))),mock.patch.object(w,'ro',side_effect=lambda p:guard(wro(p))):
            self.assertEqual(w.prepare(),registry)
        self.assertEqual(w.CONTRACT['maximum_new_nodes'],480000000)
        for s,d in [('h8',8),('h10',10),('h12',12)]:self.assertEqual(w.CONTRACT[s],dict(depth=d,node_budget=10000000))
        w.admit();m.validate_rows(selected)
        with mock.patch.object(w,'initial_gate',side_effect=ValueError('Independent preflight absent')):
            with self.assertRaises(ValueError):w.run()
        self.assertFalse(w.DEFAULT.exists());self.assertFalse(w.RULE.exists());self.assertNotIn('torch',sys.modules)
        EVIDENCE['metadata_and_blinding']=dict(prior4546_current=True,full_saved_families=1280,eligible_raw_states=3890,full_origin_families=553,clean_components=547,
            all_allocation16_components_aliases_excluded=True,all16_assessment_distinct_components=True,two_per_cell=True,
            selected_literal_history_choices=replay_choices,all_target_body_reads_denied_for_prepare=True,
            models_policy_and_HOLD_fixed=True,formula_before_new_measurements=True,future_overlay_requires_admission=True)

    def test_02_exact_cell_mass_bestset_and_missing_reference_probes(self):
        from congklak import stratum_weighting as w,stratum_metrics as m
        rows=load(w.REGISTRY)['rows'];h8=synthetic_labels(rows,8);h10=synthetic_labels(rows,10,variable=True);h12=synthetic_labels(rows,12)
        projection=m.weight_projection(rows,h8,h10)
        rawweights={k:exact_weight(h8[k],h10[k],[a for a in range(7) if r['legal_mask']&(1<<a)])[2] for k,r in rows.items()}
        p=probabilities(rawweights,{k:r['cell'] for k,r in rows.items()})
        for key in rows:
            self.assertEqual(projection['rows'][key]['weighted_probability'],rational(p[key]));self.assertEqual(projection['rows'][key]['uniform_probability'],rational(F(1,16)))
        actual=self.metric_check(rows,projection,h12)
        self.assertTrue(any(len(r['H10_best_moves'])>1 and from_rational(r['metrics']['H10_best_H12_regret_max'])>from_rational(r['metrics']['H10_best_H12_regret_min']) for r in actual['roots'].values()))
        missing=dict(h12);missing.pop(next(iter(rows)));result=self.metric_check(rows,projection,missing)
        self.assertFalse(result['screen']['passes']);self.assertEqual(result['screen']['status'],'incomplete_reference_hold')
        self.assertTrue(all(v is None for v in result['metrics']['overall']['weighted']['complete_risk'].values()))
        self.assertGreater(from_rational(result['metrics']['overall']['weighted']['original_missing_mass']),0)
        empty=self.metric_check(rows,projection,{})
        self.assertTrue(all(v is None for v in empty['metrics']['overall']['weighted']['conditional_complete_diagnostic'].values()))
        uniform=m.weight_projection(rows,h8,synthetic_labels(rows,10))
        identity=self.metric_check(rows,uniform,h12);self.assertFalse(identity['screen']['passes'])
        self.assertFalse(identity['screen']['conditions']['at_least_one_overall_risk_strictly_improves'])
        # Missing H8/H10 cannot silently become a one-root or zero-weight cell.
        bad=dict(h8);bad.pop(next(iter(rows)))
        with self.assertRaises(ValueError):m.weight_projection(rows,bad,h10)
        # Rehashed metadata/probability/type and H12 endpoint lookahead are refused.
        for field,value in [('H12_used',True),('roots',True)]:
            bad=copy.deepcopy(projection);bad[field]=value
            with self.assertRaises(ValueError):m.validate_projection(rows,bad)
        extra={k:dict(v,h12=h12[k]) for k,v in {k:dict(h8=h8[k],h10=h10[k]) for k in rows}.items()}
        with self.assertRaises(ValueError):m.freeze_weights(rows,extra)
        groups=independent_groups(rows,actual['roots'])
        probes=[('all',None,'H10_best_H12_regret_max'),('all',None,'root_drift'),('all',None,'action_Q_drift')]
        probes += [('cells',c,field) for c in CELLS if '/ordinary/' in c for field in ('H10_best_H12_regret_mean','H10_best_H12_regret_max')]
        probes += [('strata',s,field) for s in STRATA if s.endswith('/ordinary') for field in ('H10_best_H12_regret_mean','H10_best_H12_regret_max')]
        for group,key,field in probes:
            bad=copy.deepcopy(groups);scope=bad[group] if key is None else bad[group][key]
            higher=rational(from_rational(scope['uniform']['complete_risk'][field])+1)
            scope['weighted']['complete_risk'][field]=higher;scope['weighted'][field]=higher
            screen=m.favorable_screen(bad,projection);self.assertEqual(screen,independent_screen(bad,projection));self.assertFalse(screen['passes'])
        EVIDENCE['prospective_math']=dict(exact_cell_mass=True,full_bestset_mean_min_max_actor_verified=True,synthetic_ties=True,
            uniform_identity_not_benefit=True,missing_H8_H10_refused=True,missing_H12_null_primary_not_renormalized=True,H12_lookahead_refused=True)

    def test_03_real_teacher_pre_post_commit_kills_and_exact_resume(self):
        from congklak import stratum_weighting as w
        seed=self.base/'old-fixture-h8-seed';run_child(self.command(seed,'h8'),self.base/'seed-h8.log',timeout=240)
        # Retain the exact closed H8 seed; each killed H10 run starts from it.
        seed_db=sha(seed/'work.sqlite3');seed_record=self.base/'fixture-seed.json'
        save(seed_record,dict(run=relative(seed),journal_sha256=seed_db,source='two previously consumed diagnostic roots',fixture_depths=[2,3,4]))
        fresh=self.clone(seed,'teacher-fresh');run_child(self.command(fresh,'h10'),self.base/'fresh-h10.log',timeout=240)
        w.realize(fresh);run_child(self.command(fresh,'h12'),self.base/'fresh-h12.log',timeout=240)
        rootmap,reference,manifest=w.load_rows(fresh);self.assertEqual(len(rootmap),2);self.assertEqual(manifest['identity']['settings']['h12']['depth'],4)
        tests=[]
        for event,aggregate,stage_count in [('stratum_weighting_h10_precommit',3,1),('stratum_weighting_h10_committed',4,2)]:
            run=self.clone(seed,'teacher-'+str(stage_count));probe=self.base/('teacher-probe-'+str(stage_count))
            marker=kill(self.command(run,'h10'),probe,event,aggregate,dict(STRATUM_WEIGHTING_BLOCK_EVENT=event,
                STRATUM_WEIGHTING_BLOCK_STEP=str(aggregate),STRATUM_WEIGHTING_MARKER=str(probe/'declared-marker.json')))
            backup(run/'work.sqlite3',probe/'validated-prefix.sqlite3')
            prefix={t:sql_rows(probe/'validated-prefix.sqlite3',t,'state') for t in ('roots',*w.TABLES)}
            self.assertEqual(len(prefix['h8_attempts']),2);self.assertEqual(len(prefix['h8_labels']),2)
            self.assertEqual(len(prefix['h10_attempts']),stage_count);self.assertEqual(len(prefix['h10_labels']),stage_count);self.assertFalse(prefix['h12_attempts'])
            run_child(self.command(run,'h10'),probe/'resume-h10.log',timeout=240);w.realize(run)
            run_child(self.command(run,'h12'),probe/'resume-h12.log',timeout=240);actualroots,actual,body=w.load_rows(run)
            self.assertEqual(actualroots,rootmap)
            for table,values in prefix.items():self.assertEqual(values,sql_rows(run/'work.sqlite3',table,'state')[:len(values)])
            self.assertEqual(teacher_semantic(actual),teacher_semantic(reference))
            tests.append(dict(event=event,actual_process_kill=True,committed_prefix=stage_count,committed_aggregate_prefix=aggregate,
                marker=marker,probe=relative(probe),run=relative(run),seed_record=relative(seed_record),source_seed_journal_sha256=seed_db,
                resumed_attempts=6,committed_prefix_bytes_preserved=True,fresh_semantics_equal=True))
        proof=self.check_label_work(rootmap,reference,w.STAGES,manifest['identity']['settings'])
        bad=self.clone(fresh,'teacher-corrupt-history');key=next(iter(rootmap))
        patch_sql(bad/'work.sqlite3','h12_attempts',key,'state',lambda a:a['history'].pop(0))
        with self.assertRaises(ValueError):w.load_rows(bad)
        bad=self.clone(fresh,'teacher-corrupt-marker');(bad/'manifest.json').unlink()
        patch_sql(bad/'work.sqlite3','meta','complete','key',lambda m:m.update(total_new_nodes=0))
        with self.assertRaises(ValueError):w.run(bad,fixture=True)
        self.assertFalse((bad/'manifest.json').exists())
        closed=self.closed_worker(fresh,self.base/'fixture-completed-closed',True)
        self.assertFalse(w.DEFAULT.exists());self.assertFalse(w.RULE.exists())
        EVIDENCE['live_teacher_recovery']=dict(tests=tests,old_consumed_fixture_roots=2,actual_fixture_depths=[2,3,4],
            source_seed_journal_sha256=seed_db,seed_record=relative(seed_record),literal_fixture_work=proof,
            rehashed_history_and_complete_marker_refused_before_export=True,completed_no_work=closed)

    def test_04_all32_H8_H10_labels_PVs_and_blind_realized_weights(self):
        from congklak import stratum_weighting as w,stratum_metrics as m
        roots,stores,body=w.load_rows(allow_partial=True);registry=load(w.REGISTRY)['rows']
        self.assertEqual(set(roots),set(registry));self.assertEqual(len(roots),16)
        for key,row in roots.items():
            expected=registry[key];self.assertEqual(row['root'],expected['source']['source_candidate']);self.assertEqual(row['source_root_sha256'],config.digest(row['root']))
            self.assertEqual(row['source_metadata_sha256'],config.digest(expected));self.assertEqual(row['actor'],expected['actor'])
            self.assertEqual(row['state'],expected['state']);self.assertEqual(row['cohort'],'assessment');self.assertEqual(row['source_cohort'],'training_candidate')
        proof=self.check_label_work(roots,stores,('h8','h10'),body['identity']['settings']);self.assertEqual(proof['attempts'],32);self.assertLessEqual(proof['actual_nodes'],320000000)
        self.assertFalse(stores['h12_attempts']);self.assertFalse(stores['h12_labels'])
        if proof['complete']==32:
            realized=load(w.RULE);projection=realized['projection'];self.assertFalse(realized['H12_used'])
            self.assertEqual(realized['references_prefix_sha256'],sha(w.DEFAULT/'references-prefix.sqlite3'))
            self.assertEqual(realized['references_manifest_sha256'],sha(w.DEFAULT/'references-manifest.json'))
            self.assertEqual(realized['endpoint_item_sha256'],{s:{k:config.digest(v) for k,v in stores[s+'_labels'].items()} for s in ('h8','h10')})
            self.assertEqual(projection,m.weight_projection(roots,stores['h8_labels'],stores['h10_labels']))
            weights={};cells={}
            for key,row in roots.items():
                legal=[a for a in range(7) if row['legal_mask']&(1<<a)];dv,dq,r=exact_weight(stores['h8_labels'][key],stores['h10_labels'][key],legal)
                entry=projection['rows'][key];self.assertEqual(entry['raw_rule'],dict(root_shift_seeds=dv,mean_action_shift_seeds=rational(dq),raw_weight=rational(r),
                    formula=w.CONTRACT['raw_rule'],requested_depths=[8,10],H12_used=False))
                self.assertEqual(entry['root_sha256'],config.digest(row));weights[key]=r;cells[key]=row['cell']
            p=probabilities(weights,cells)
            for key in roots:self.assertEqual(projection['rows'][key]['weighted_probability'],rational(p[key]));self.assertEqual(projection['rows'][key]['uniform_probability'],rational(F(1,16)))
            prefix=w.DEFAULT/'references-prefix.sqlite3'
            for table in ('roots',*w.TABLES):
                self.assertEqual(sql_rows(prefix,table,'state'),sql_rows(w.DEFAULT/'work.sqlite3',table,'state'))
            with closing(dbopen(prefix)) as db:self.assertEqual({r[0] for r in db.execute('SELECT key FROM meta')},{'identity','references_complete'})
            with closing(dbopen(w.DEFAULT/'work.sqlite3')) as db:self.assertIsNone(db.execute("SELECT body FROM meta WHERE key='h12_gate'").fetchone())
        else:
            self.assertEqual(body['status'],'references_incomplete_hold');self.assertFalse(w.RULE.exists())
        EVIDENCE['H8_H10_phase']=dict(**proof,all32_complete=proof['complete']==32,H12_attempts_before_GO=0,
            immutable_prefix_snapshot='data/stratum-weighting-v1/references-prefix.sqlite3' if proof['complete']==32 else None,
            all32_endpoint_hashes_and_exact_probabilities_verified=proof['complete']==32,
            future_mutable_DB_sha256_capture_only=sha(w.DEFAULT/'work.sqlite3'),mutable_DB_not_current_input=True,
            incomplete_references_hold_without_fallback=proof['complete']!=32)

    def test_05_all16_H12_labels_histories_and_literal_PVs(self):
        from congklak import stratum_weighting as w
        roots,stores,body=w.load_rows(allow_partial=True)
        if not w.RULE.exists():
            self.assertEqual(body['status'],'references_incomplete_hold');self.assertFalse(stores['h12_attempts']);self.assertFalse(stores['h12_labels'])
            EVIDENCE['H12_phase']=dict(attempts=0,H12_held_for_missing_H8_H10=True);return
        roots,stores,body=w.load_rows();proof=self.check_label_work(roots,stores,w.STAGES,body['identity']['settings'])
        self.assertEqual(proof['attempts'],48);self.assertLessEqual(proof['actual_nodes'],480000000)
        self.assertEqual(body['total_new_nodes'],proof['actual_nodes']);self.assertEqual(body['auxiliary_completed_reports'],proof['completed_iteration_reports'])
        self.assertEqual(len(stores['h12_attempts']),16)
        prefix=w.DEFAULT/'references-prefix.sqlite3'
        for table in ('roots','h8_attempts','h8_labels','h10_attempts','h10_labels'):
            self.assertEqual(sql_rows(prefix,table,'state'),sql_rows(w.DEFAULT/'work.sqlite3',table,'state'))
        rule_proof=load(WORK/'rule.json');self.current(rule_proof['inputs']);self.assertNotIn(relative(w.DEFAULT/'work.sqlite3'),rule_proof['inputs'])
        self.assertFalse(body['weights_applied']);self.assertFalse(body['training_performed']);self.assertFalse(body['promotion'])
        EVIDENCE['H12_phase']=dict(**proof,H12_attempts=16,H12_complete=len(stores['h12_labels']),H12_capped=16-len(stores['h12_labels']),
            node_ceiling=480000000,reference_prefix_current_and_live_tables_exact=True,H12_auxiliary_not_game_truth=True)

    def test_06_exact_within_cell_regret_coverage_and_missing_mass(self):
        from congklak import stratum_weighting as w,stratum_metrics as m
        roots,stores,body=w.load_rows(allow_partial=True)
        if not w.RULE.exists():
            self.assertFalse(stores['h12_attempts']);self.assertEqual(body['status'],'references_incomplete_hold')
            EVIDENCE['within_cell_assessment']=dict(screen_unavailable_missing_H8_H10=True,weights_applied=False);return
        rule=load(w.RULE);result=load(w.OUTPUT);actual=self.metric_check(roots,rule['projection'],stores['h12_labels'])
        self.assertEqual(result['assessment'],actual);self.assertEqual(result['metrics'],actual['metrics']);self.assertEqual(result['screen'],actual['screen'])
        self.assertEqual(result['realized_rule_sha256'],sha(w.RULE));self.assertEqual(result['manifest_sha256'],sha(w.DEFAULT/'manifest.json'));self.current(result['inputs'])
        self.assertFalse(result['weights_applied']);self.assertFalse(result['training_performed']);self.assertFalse(result['promotion']);self.assertEqual(result['previous_allocation'],'hold')
        self.assertTrue(result['future_optimizer_reserved']);self.assertFalse(result['H12_is_game_truth']);self.assertFalse(result['calibrated_confidence_claim'])
        EVIDENCE['within_cell_assessment']=dict(exact_all_root_cell_stratum_actor_and_ordinary_metrics=True,all_full_bestset_mean_min_max_and_lowest_pit_verified=True,
            mass_cells_1_8_strata_1_4_actors_1_2_ordinary_1_2=True,missing_original_mass_not_optimistically_renormalized=True,
            exact_screen=result['screen'],finite_reference_not_calibrated_confidence=True,weights_applied=False)

    def test_07_rehashed_corruption_lookahead_and_missing_export_refusal(self):
        from congklak import stratum_weighting as w,stratum_metrics as m
        fresh=self.base/'corruption-old-fixture'
        for stage in ('h8','h10'):run_child(self.command(fresh,stage),self.base/('corruption-'+stage+'.log'),timeout=240)
        w.realize(fresh);run_child(self.command(fresh,'h12'),self.base/'corruption-h12.log',timeout=240)
        rootmap,stores,manifest=w.load_rows(fresh);key=next(iter(rootmap));cases=[]
        for name,table,ident,field,mutate in [
            ('identity-bool','meta','identity','key',lambda b:b.update(fixture=1)),
            ('root-ownership','roots',key,'state',lambda b:b.update(source_cohort='training_candidate')),
            ('history-gap','h12_attempts',key,'state',lambda b:b['history'].pop(0)),
            ('history-PVS','h12_attempts',key,'state',lambda b:b['history'][0]['search_options'].update(use_pvs=False)),
            ('label-Q','h12_labels',key,'state',lambda b:b['label']['action_scores'][0].update(value_player0=999))]:
            bad=self.clone(fresh,'rehashed-'+name);patch_sql(bad/'work.sqlite3',table,ident,field,mutate)
            with self.assertRaises(ValueError):w.load_rows(bad)
            cases.append(name)
        missing=self.clone(fresh,'valid-missing-final-manifest');expected=load(missing/'manifest.json');(missing/'manifest.json').unlink()
        with mock.patch.object(w.history,'execute',side_effect=AssertionError('Repair searched')):w.run(missing,fixture=True)
        self.assertEqual(load(missing/'manifest.json'),expected)
        for name in ('references-prefix.sqlite3','references-manifest.json','rule.json'):
            bad=self.clone(fresh,'missing-required-'+name.replace('.','-'));(bad/name).unlink();(bad/'manifest.json').unlink()
            with self.assertRaises((ValueError,FileNotFoundError)):w.run(bad,fixture=True)
            self.assertFalse((bad/'manifest.json').exists());cases.append('missing-required-'+name)
        bad=self.clone(fresh,'corrupt-prefix-source');patch_sql(bad/'references-prefix.sqlite3','roots',key,'state',lambda b:b.update(family='wrong'))
        with self.assertRaises(ValueError):w.load_rows(bad)
        if w.OUTPUT.exists():
            original=w.read
            for name in ('metric','source'):
                bad=copy.deepcopy(load(w.OUTPUT))
                if name=='metric':bad['metrics']['overall']['weighted']['root_drift']=rational(999)
                else:bad['inputs'][relative(w.REGISTRY)]='0'*64
                save(self.base/('rehashed-report-'+name+'.json'),dict(body=bad,sha256=config.digest(bad)))
                with mock.patch.object(w,'read',side_effect=lambda p,b=bad:b if Path(p).resolve()==w.OUTPUT.resolve() else original(p)):
                    with self.assertRaises(ValueError):w.verify()
                cases.append('assessment-'+name)
            bad=copy.deepcopy(load(w.RULE));bad['H12_used']=True;bad['projection']['H12_used']=True
            save(self.base/'rehashed-rule-lookahead.json',dict(body=bad,sha256=config.digest(bad)))
            with mock.patch.object(w,'read',side_effect=lambda p:bad if Path(p).resolve()==w.RULE.resolve() else original(p)):
                with self.assertRaises(ValueError):w.load_rows()
            cases.append('H12-lookahead-rule')
        EVIDENCE['corruption_and_publication']=dict(cases=cases,all_payloads_rehashed=True,all_refused=True,
            complete_marker_before_export_verified=True,valid_final_export_repaired_without_search=True,
            immutable_phase_evidence_required_not_repairable=True,no_production_artifacts_modified=True)

    def test_08_closed_API_CLI_no_work_models_and_prior4546_preserved(self):
        from congklak import stratum_weighting as w
        proof=self.closed_worker(w.DEFAULT,self.base/'production-completed-no-work',watch_prior=True)
        oldtree=ast.parse((ROOT/'congklak/allocation_progress.py').read_text());newtree=ast.parse((ROOT/'congklak/stratum_progress.py').read_text())
        for name in ('save','event_message','runs','run_command'):
            before=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            after=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(after,include_attributes=False))
        self.current(load(BASELINE)['inputs']);self.assertNotIn('torch',sys.modules)
        doc=(ROOT/'docs/STRATUM_RESEARCH.md').read_text()
        for term in ('population','density','regret','concentration','Missing','unit loss'):self.assertIn(term,doc)
        EVIDENCE['final_no_work']=dict(proof=proof,prior4546_bytes_sizes_mtimes_unchanged=True,models_and_policy_fixed=True,
            no_training_PUCT_new_games_or_model_forwards=True,proven_logger4_AST_identical=True,old_logger_checks_not_recounted=True,
            research_primary_sources_independently_opened=True,chosen_mixture_not_natural_population=True,
            future_optimizer_requires_explicit_reservation_admission=True,previous_allocation='hold')


def source_inputs():
    from congklak import stratum_weighting as w
    result=dict(load(BASELINE)['inputs']);names={relative(Path(__file__)),'tools/stratum-audit.py',relative(BASELINE),
        'docs/STRATUM_SCOPE.md','docs/STRATUM_RESEARCH.md','congklak/stratum_common.py','congklak/stratum_progress.py','tools/stratum-dashboard.html',
        relative(w.INVENTORY),relative(w.REGISTRY),relative(w.RESERVATIONS),relative(w.PLAN),relative(w.IMPLEMENTATION),
        'build/stratum-metrics-implementation.json','build/stratum-weighting-author.json','build/stratum-metrics-author.json',*w.SOURCES,*w.DEPENDENCIES}
    result.update({name:sha(ROOT/name) for name in names});return dict(sorted(result.items()))


def collect():
    inputs={};evidence={};covered=[];groups=[]
    for group,methods in GROUPS.items():
        path=WORK/(group+'.json');body=load(path)
        assert body['passed'] is True and body['covered_methods']==methods and body['tests_run']==len(methods)
        assert not any(body[k] for k in ('errors','failures','skips'))
        for name,checksum in body['inputs'].items():
            assert sha(ROOT/name)==checksum,name;assert name not in inputs or inputs[name]==checksum;inputs[name]=checksum
        assert not(set(evidence)&set(body['evidence']));evidence.update(body['evidence']);covered+=methods
        groups.append(dict(path=relative(path),sha256=sha(path),tests_run=len(methods),covered_methods=methods));inputs[relative(path)]=sha(path)
    assert len(covered)==len(set(covered))==8
    body=dict(schema='stratum-independent-audit-v1',passed=True,tests_run=8,errors=0,failures=0,skips=0,covered_methods=covered,
        covered_checks=['__main__.StratumIndependentAudit.'+m for m in covered],groups=groups,inputs=inputs,evidence=evidence)
    output=ROOT/'build/stratum-audit.json'
    if output.exists():assert load(output)==body,'Immutable aggregate differs'
    else:save(output,body)
    return body


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['freeze',*GROUPS,'collect']);args=parser.parse_args()
    if args.operation=='freeze':
        assert not FREEZE.exists(),'Audit source already frozen';save(FREEZE,dict(schema='stratum-independent-executed-source-v1',groups=GROUPS,inputs=source_inputs()));return 0
    if args.operation=='collect':collect();return 0
    frozen=load(FREEZE);assert frozen['groups']==GROUPS
    for name,checksum in frozen['inputs'].items():assert sha(ROOT/name)==checksum,'Executed source changed: '+name
    output=WORK/(args.operation+'.json');assert not output.exists(),'Immutable audit group exists'
    started=time.monotonic();methods=GROUPS[args.operation]
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(StratumIndependentAudit(name) for name in methods))
    from congklak import stratum_weighting as w
    inputs={**frozen['inputs'],relative(FREEZE):sha(FREEZE)};files=[p for p in StratumIndependentAudit.base.rglob('*') if durable(p)]
    if args.operation=='rule':
        # The live DB is deliberately absent: it mutates when H12 starts.
        files += [p for p in (w.RULE,w.DEFAULT/'references-prefix.sqlite3',w.DEFAULT/'references-manifest.json') if p.is_file()]
        files += [WORK/'preflight.json']
    if args.operation in ('labels','final'):
        files += [p for p in w.DEFAULT.rglob('*') if durable(p)]
        files += [p for p in (w.OUTPUT,w.RULE,WORK/'preflight.json',WORK/'rule.json') if p.is_file()]
    if args.operation=='final':files += [WORK/'labels.json']
    inputs.update(pins(files))
    body=dict(schema='stratum-independent-group-v1',group=args.operation,passed=result.wasSuccessful(),tests_run=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),covered_methods=methods,
        covered_checks=['__main__.StratumIndependentAudit.'+m for m in methods],duration_seconds=time.monotonic()-started,
        retained_directory=relative(StratumIndependentAudit.base),inputs=inputs,evidence=EVIDENCE,
        failure_details=[dict(method=str(test),trace=traceback) for test,traceback in result.errors+result.failures])
    save(output,body);print(config.canonical(dict(event='stratum_independent_group',group=args.operation,passed=body['passed'],tests=result.testsRun)),flush=True)
    return int(not body['passed'])


def independent_inventory():
    """BFS component closure and literal saved candidate projection, no labels."""
    from allocation_independent_audit import independent_inventory as previous_inventory
    previous = previous_inventory()
    reservation = load(ROOT / 'build/allocation-weighting-reservations.json')
    reserved_q = {q for group in reservation['roles'].values() for q in group['eligible_quotients']}
    reserved_f = {f for group in reservation['roles'].values() for f in group['origin_families']}
    families = {}
    graph = defaultdict(set); owners = {}; provenance = {}
    for journal in ('ordinary-data-pilot-v1', 'natural-study-v1', 'midgame-data-pilot-v1'):
        for index, family in metadata_rows(journal, 'families', 'idx').items():
            fid = family['family']; assert fid not in families; families[fid] = family
            owners[fid] = family['cohort']; provenance[fid] = (journal, index)
            node = ('family', fid); graph[node]
            for candidate in family['candidates']:
                if candidate['stratum'] in STRATA and candidate['legal_mask'].bit_count() >= 2:
                    other = ('quotient', quotient(candidate['state']))
                    graph[node].add(other); graph[other].add(node)
    components = {}; by_family = {}; seen = set()
    old_by_family = {fid: c for c in previous['components'].values() for fid in c['families']}
    for first in graph:
        if first in seen: continue
        todo = [first]; seen.add(first); found = []
        while todo:
            node = todo.pop(); found.append(node)
            for other in graph[node]:
                if other not in seen: seen.add(other); todo.append(other)
        fs = sorted(v for kind, v in found if kind == 'family')
        qs = sorted(v for kind, v in found if kind == 'quotient')
        reasons = list(old_by_family[fs[0]]['blocked_reasons'])
        assert old_by_family[fs[0]]['families'] == fs and old_by_family[fs[0]]['quotients'] == qs
        if set(fs) & reserved_f or set(qs) & reserved_q: reasons.append('prior_allocation_consumed_component')
        body = dict(families=fs, quotients=qs, source_cohorts=sorted({owners[f] for f in fs}), blocked_reasons=reasons)
        cid = config.digest(old_by_family[fs[0]]); components[cid] = body
        by_family.update({f: cid for f in fs})
    # The owner defines the exact serialized provenance projection. This audit
    # returns the complete literal candidate occurrences for independent matching.
    candidates = []
    for fid, family in families.items():
        for ordinal, item in enumerate(family['candidates']):
            if item['stratum'] not in STRATA or item['legal_mask'].bit_count() < 2: continue
            candidates.append(dict(source=provenance[fid][0], family_index=provenance[fid][1],
                candidate_index=ordinal, family_body_sha256=config.digest(family), candidate_sha256=config.digest(item),
                candidate=item, state_key=raw(item['state']), state=item['state'], family=fid,
                stratum=item['stratum'], actor=item['state']['controller'], legal_mask=item['legal_mask'],
                source_cohort=family['cohort'], component=by_family[fid], quotient=quotient(item['state'])))
    exposure={s:dict(raw=set(),quotient=set()) for s in ('h8','h10')}
    for journal in ('ordinary-data-pilot-v1','natural-study-v1','midgame-label-pilot-v1'):
        rootmap=metadata_rows(journal,'roots')
        for stage in ('h8','h10'):
            keys=requested_keys(journal,stage)
            if journal=='ordinary-data-pilot-v1' and stage=='h8':
                with closing(dbopen(ROOT/'data'/journal/'work.sqlite3')) as db:keys.update(r[0] for r in db.execute('SELECT state FROM labels'))
            for key in keys:
                if key in rootmap:
                    exposure[stage]['raw'].add(raw(rootmap[key]['state']));exposure[stage]['quotient'].add(quotient(rootmap[key]['state']))
    origins=defaultdict(list);excluded=Counter()
    for row in candidates:
        if components[row['component']]['blocked_reasons']:
            excluded.update(components[row['component']]['blocked_reasons']);continue
        if row['source_cohort']!='training_candidate':excluded['source_not_training_candidate']+=1;continue
        origins[row['state_key']].append(dict(source_journal=row['source'],source_family_index=row['family_index'],source_family_sha256=row['family_body_sha256'],
            source_candidate_index=row['candidate_index'],source_candidate_sha256=row['candidate_sha256'],source_candidate=row['candidate'],component=row['component']))
    eligible={}
    for key,values in origins.items():
        values.sort(key=lambda o:(o['source_journal'],o['source_candidate']['family'],o['source_candidate_index']))
        self_origin=values[0];r=self_origin['source_candidate'];q=quotient(r['state']);actor=r['state']['controller']
        assert len({o['component'] for o in values})==1
        eligible[key]=dict(state_key=key,state=r['state'],family=r['family'],component=self_origin['component'],stratum=r['stratum'],actor=actor,cell=r['stratum']+'/actor'+str(actor),
            legal_mask=r['legal_mask'],quotient=q,source_cohort='training_candidate',source=self_origin,all_origins=values,
            previous_requested_reference_exposure={s:dict(raw=key in exposure[s]['raw'],quotient=q in exposure[s]['quotient']) for s in ('h8','h10')})
    inventory=dict(eligible=dict(sorted(eligible.items())),components=components,source_families=len(families),eligible_candidate_observations=len(candidates),
        eligible_roots=len(eligible),representative_origin_families=len({r['family'] for r in eligible.values()}),
        all_origin_families=len({o['source_candidate']['family'] for values in origins.values() for o in values}),eligible_components=len({r['component'] for r in eligible.values()}),
        eligible_cells={c:dict(roots=sum(r['cell']==c for r in eligible.values()),origin_families=len({r['family'] for r in eligible.values() if r['cell']==c}),
                             components=len({r['component'] for r in eligible.values() if r['cell']==c})) for c in CELLS},
        duplicate_raw_observations=sum(len(v)-1 for v in origins.values()),exclusion_observation_reasons=dict(sorted(excluded.items())),
        prior_allocation_reserved_components=sorted({r['component'] for r in load(ROOT/'build/allocation-weighting-registry.json')['rows'].values()}),
        prior_inventory_sha256=sha(ROOT/'build/allocation-weighting-inventory.json'),prior_reservations_sha256=sha(ROOT/'build/allocation-weighting-reservations.json'),
        H8_H10_target_bodies_read=False,H12_H14_target_bodies_read=False,shared_prefix_independence_claim=False)
    return inventory


def independent_selection(inv):
    salt='stratum-weighting-saved-metadata-panel/20261031/v1/'
    rank=lambda key:hashlib.sha256((salt+key).encode()).hexdigest()
    options={c:sorted((r for r in inv['eligible'].values() if r['cell']==c),key=lambda r:(rank(r['state_key']),r['state_key'])) for c in CELLS}
    owners={};chosen={}
    def augment(slot,seen):
        for row in options[slot[0]]:
            comp=row['component']
            if comp in seen:continue
            seen.add(comp)
            if comp not in owners or augment(owners[comp],seen):owners[comp]=slot;chosen[slot]=row;return True
        return False
    for slot in [(c,i) for c in CELLS for i in range(2)]:assert augment(slot,set()),'Metadata matching shortage'
    return dict(sorted((r['state_key'],dict(r,group='assessment',future_optimizer_reserved=True)) for r in chosen.values()))


def durable(path):
    return (path.is_file() and path.suffix != '.pyc' and not path.name.endswith('-shm')
            and (not path.name.endswith('-wal') or path.stat().st_size > 0))


if __name__=='__main__':raise SystemExit(main())

