"""Independent read-only pilot loss, ordinary-error and readiness audit."""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import closing
import bisect
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import sys
import time
import unittest
from unittest import mock
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config
from congklak.native import Native,flatten
from data_pilot_independent_audit import sha,load,save,relative,pins,dbopen,norm,position,snapshot
from quality_audit_acceptance import module
import reference as literal
WORK=ROOT/'build/pilot-loss-review-audit';EVIDENCE={}
ORDINARY=ROOT/'build/pilot-loss-review-ordinary.json';ARENA=ROOT/'build/pilot-loss-review-arena.json';BASELINE=ROOT/'build/pilot-loss-review-baseline.json'
CONTROL=ROOT/'data/quality-train-v1/model.json';PILOT=ROOT/'data/value-pilot-v1/model.json'
FROZEN={'layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias','policy.weight','policy.bias'}


def dense(model,row):
    w=model['weights'];x=row['cached_features']['x']
    for name in ('layers.0','layers.2'):x=[max(0.,sum(a*b for a,b in zip(weights,x))+bias) for weights,bias in zip(w[name+'.weight'],w[name+'.bias'])]
    z=math.tanh(sum(a*b for a,b in zip(w['layers.4.weight'][0],x))+w['layers.4.bias'][0]);logits=[sum(a*b for a,b in zip(weights,x))+bias for weights,bias in zip(w['policy.weight'],w['policy.bias'])]
    f=row['cached_features'];return f['d']+f['r']*z,logits


def stat(rows):
    n=len(rows);ds=[r['absolute_error_delta_seeds'] for r in rows];worse=sorted([r for r in rows if r['absolute_error_delta_seeds']>0],key=lambda r:(-r['absolute_error_delta_seeds'],r['state_key']));better=sorted([r for r in rows if r['absolute_error_delta_seeds']<0],key=lambda r:(r['absolute_error_delta_seeds'],r['state_key']));gross=math.fsum(r['absolute_error_delta_seeds'] for r in worse)
    def contribution(r):return dict(state_key=r['state_key'],family=r['family'],band=r['band'],teacher_depth=r['teacher_depth'],delta_absolute_error_seeds=r['absolute_error_delta_seeds'],control_absolute_error_seeds=r['control']['absolute_error_seeds'],pilot_absolute_error_seeds=r['pilot']['absolute_error_seeds'])
    total=lambda model:math.fsum(r[model]['absolute_error_seeds'] for r in rows)
    return dict(rows=n,families=len({r['family'] for r in rows}),control_mae_seeds=total('control')/n,pilot_mae_seeds=total('pilot')/n,delta_mae_seeds=math.fsum(ds)/n,total_absolute_error_delta_seeds=math.fsum(ds),improved=sum(x<0 for x in ds),worse=sum(x>0 for x in ds),tied=sum(x==0 for x in ds),gross_worsening_seeds=gross,gross_improvement_seeds=-math.fsum(r['absolute_error_delta_seeds'] for r in better),top_worsening=[contribution(r) for r in worse[:5]],top_improvement=[contribution(r) for r in better[:5]],top1_worsening_fraction=worse[0]['absolute_error_delta_seeds']/gross if worse else 0.,top3_worsening_fraction=math.fsum(r['absolute_error_delta_seeds'] for r in worse[:3])/gross if worse else 0.,top3_pilot_absolute_error_fraction=math.fsum(sorted([r['pilot']['absolute_error_seeds'] for r in rows],reverse=True)[:3])/total('pilot') if total('pilot') else 0.,control_mean_error_player0=math.fsum(r['control']['error_player0'] for r in rows)/n,pilot_mean_error_player0=math.fsum(r['pilot']['error_player0'] for r in rows)/n,control_mean_error_actor=math.fsum(r['control']['error_actor'] for r in rows)/n,pilot_mean_error_actor=math.fsum(r['pilot']['error_actor'] for r in rows)/n,pilot_actor_overestimates=sum(r['pilot']['error_actor']>0 for r in rows),pilot_actor_underestimates=sum(r['pilot']['error_actor']<0 for r in rows),leave_one_out_delta_mae=[dict(state_key=r['state_key'],delta_mae_seeds=(math.fsum(ds)-r['absolute_error_delta_seeds'])/(n-1)) for r in rows] if n>1 else [])


def summaries(rows):
    answer={'all':stat(rows)}
    for key,field in (('by_band','band'),('by_teacher_depth','teacher_depth'),('by_actor','actor'),('by_legal_count','legal_count')):
        bins=defaultdict(list)
        for r in rows:bins[str(r[field])].append(r)
        answer[key]={k:stat(v) for k,v in sorted(bins.items())}
    return answer


def errors(value,y,d,r,actor):
    sign=1 if actor==0 else -1;e=value-y
    return dict(prediction_player0=value,prediction_actor=sign*value,residual_actor=(value-d)/r,error_player0=e,error_actor=sign*e,error_residual_actor=e/r,absolute_error_seeds=abs(e))


def game_records(stage,candidate):
    path=ROOT/('data/value-pilot-benchmark-'+stage+'-v1/work.sqlite3');answer={}
    with closing(dbopen(path)) as db:
        for index,text,digest in db.execute('SELECT idx,body,hash FROM games WHERE comparison=? ORDER BY pair,seat',(candidate+'_vs_H8_relay_CPU',)):
            g=json.loads(text);assert config.digest(g)==digest;answer[g['pair'],g['candidate_seat']]=dict(body=g,source=dict(path=relative(path),index=index,comparison=g['comparison'],pair=g['pair'],candidate_seat=g['candidate_seat'],game_sha256=digest,stage=stage))
    assert len(answer)==24;return answer


def replay(test,record):
    g=record['body'];q=norm(g['start']);steps=[];costs=defaultdict(lambda:dict(decisions=0,requested_ms=0,wall_ms=0.,overshoot_ms=0.,max_overshoot_ms=0.,nodes=0,completed_simulations=0,physical_forwards=0,feature_simulations=0,feature_deposits=0))
    for index,(action,row) in enumerate(zip(g['moves'],g['decisions'])):
        before=position(q);t=literal.apply(q,action);test.assertEqual(t.actor,row['actor']);q=t.position;after=position(q);gain=[after['stores'][a]-before['stores'][a] for a in (0,1)];store=7 if t.actor==0 else 15;store_before=t.before.counts[store]+sum(e['kind']=='deposit' and e['index']==store for e in t.events)
        captures=[dict(actor=t.actor,after=store_before+e['seeds'],before=store_before,count=e['seeds'],hand=0,opposite=e['opposite'],slot=e['index'],type='capture') for e in t.events if e['kind']=='capture'];last=next(e for e in reversed(t.events) if e['kind']=='deposit')['index'];reason='own_store' if t.bonus else 'own_capture' if t.captured else 'own_empty' if last in literal.row_indices(7,t.actor) else 'opponent_empty'
        steps.append(dict(choice_index=index,choice=index+1,before=before,after=after,actor=t.actor,engine=row['engine'],pit=action,display_pit=action+1,store_gain=gain,stores_after=after['stores'],capture_events=captures,capture_transfer=t.captured,bonus_choice=t.bonus and not t.terminal,forced_passes_after=[dict(zip(('from','to'),v)) for v in t.final_passes],terminal_after=t.terminal,reason=reason,relays=t.relay_pickups,deposits=t.deposits))
        test.assertFalse(row['fallback']);test.assertEqual(row['decision']['decision']['chosen'],action);d=row['decision']['decision'];w=costs[row['engine']]
        for name,value in (('decisions',1),('requested_ms',d['requested_ms']),('wall_ms',d['wall_ms']),('overshoot_ms',d['overshoot_ms']),('nodes',d['nodes']),('completed_simulations',d['completed_simulations'] or 0)):w[name]+=value
        w['max_overshoot_ms']=max(w['max_overshoot_ms'],d['overshoot_ms']);report=d['report'];telemetry=row['decision']['evaluator'];w['physical_forwards']+=telemetry['physical_forward_executions'] if telemetry else report.get('nn_evaluations',0);w['feature_simulations']+=telemetry['counts']['feature_simulations'] if telemetry else report.get('feature_simulations',0);w['feature_deposits']+=telemetry['counts']['feature_deposits'] if telemetry else report.get('feature_deposits',0)
    test.assertTrue(literal.normalize(q).terminal);test.assertEqual(position(q),g['final']);margin=literal.evaluate(q)*(1 if g['candidate_seat']==0 else -1);score=1 if margin>0 else 0 if margin<0 else .5;test.assertEqual((g['candidate_final_margin'],g['candidate_match_score']),(margin,score))
    return dict(steps=steps,summary=dict(source=record['source'],family=g['family'],candidate_seat=g['candidate_seat'],margin_seeds=margin,match_score=score,choices=len(steps),fallbacks=g['fallbacks'],cost_by_engine=dict(costs),capture_events=sum(len(s['capture_events']) for s in steps),capture_transfer=sum(s['capture_transfer'] for s in steps),bonus_choices=sum(s['bonus_choice'] for s in steps),literal_deposits=sum(s['deposits'] for s in steps),largest_capture_observations=sorted([s for s in steps if s['capture_events']],key=lambda s:(-s['capture_transfer'],s['choice_index']))[:3]))


class PilotLossReviewIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):WORK.mkdir(parents=True,exist_ok=True);cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()

    def test_01_prior2600_pins_and_global_fixed_policy(self):
        baseline=load(BASELINE);self.assertEqual((baseline['inputs_count'],baseline['prior_accepted_inputs']),(2600,2598));self.assertEqual(len(baseline['inputs']),2600)
        for name,digest in baseline['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        self.assertEqual(sha(CONTROL),baseline['fixed_model_sha256']);self.assertEqual(sha(PILOT),baseline['fixed_candidate_model_sha256']);a=load(CONTROL)['model'];b=load(PILOT)['model'];self.assertEqual(a['encoder'],b['encoder'])
        def raw(x):return b''.join(raw(v) for v in x) if isinstance(x,list) else struct.pack('<f',x)
        for name in FROZEN:self.assertEqual(raw(a['weights'][name]),raw(b['weights'][name]))
        self.assertEqual(sum(len(raw(a['weights'][name]))//4 for name in FROZEN),12807)
        EVIDENCE['immutability']=dict(prior_inputs=2600,old_accepted_inputs=2598,all_current=True,source_and_pilot_models_fixed=True,six_frozen_tensors_raw_bytes_identical=True,no_new_training_or_weight_update=True)

    def test_02_all342_ordinary_errors_sources_horizons_and_native_literal_predictions(self):
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        data=load(ROOT/'build/value-pilot-rows.json')['groups'];report=load(ORDINARY);source=load(CONTROL)['model'];pilot=load(PILOT)['model'];selected=dict(ordinary_validation=data['ordinary_development'],ordinary_training=[r for r in data['train'] if r['pool']=='ordinary'],protected_diagnostic=data['diagnostic'],base_ordinary_development=[r for r in data['development'] if r['stratum'].endswith('/ordinary')]);count=depths=0
        self.assertEqual({k:len(v) for k,v in selected.items()},dict(ordinary_validation=14,ordinary_training=61,protected_diagnostic=8,base_ordinary_development=259))
        with CpuBatchEvaluator(CONTROL,max_batch=32,backend='scalar') as left,CpuBatchEvaluator(PILOT,max_batch=32,backend='scalar') as right:
            for name,rows_ in selected.items():
                records=report['groups'][name]['rows'];self.assertEqual([r['state_key'] for r in records],[r['state_key'] for r in rows_])
                for raw,record in zip(rows_,records):
                    state=raw['state'];actor=state['controller'];y=raw['label']['target_player0'];d=raw['cached_features']['d'];scale=raw['cached_features']['r'];sign=1 if actor==0 else -1
                    self.assertEqual(record['source'],dict(path='build/value-pilot-rows.json',row_sha256=config.digest(raw),label_sha256=config.digest(raw['label']),original_root_sha256=raw.get('original_root_sha256'),label_item_sha256=raw.get('label_item_sha256'),label_source=raw['label_source']))
                    self.assertEqual((record['family'],record['cohort'],record['pool'],record['stratum']),(raw['family'],raw['cohort'],raw['pool'],raw['stratum']));self.assertEqual((record['actor'],record['active'],record['legal_mask'],record['legal_count']),(actor,sum(map(sum,state['pits'])),raw['legal_mask'],raw['legal_mask'].bit_count()));self.assertEqual((record['teacher_depth'],record['target_player0'],record['target_actor'],record['d'],record['signed_active'],record['target_residual_actor']),(raw['teacher_depth'],y,sign*y,d,scale,(y-d)/scale))
                    a=left.evaluate_many([state])[0];b=right.evaluate_many([state])[0];self.assertEqual(a['policy_logits'],b['policy_logits'])
                    for model,pred,key in ((source,a,'control'),(pilot,b,'pilot')):
                        value,logits=dense(model,raw);self.assertAlmostEqual(value,pred['value_player0'],delta=1e-10);self.assertLessEqual(max(abs(x-z) for x,z in zip(logits,pred['policy_logits'])),1e-10);self.assertEqual(record[key],errors(pred['value_player0'],y,d,scale,actor))
                    self.assertEqual(record['absolute_error_delta_seeds'],record['pilot']['absolute_error_seeds']-record['control']['absolute_error_seeds']);targets=record['requested_depth_sensitivity'];self.assertIn(8,[t['depth'] for t in targets]);self.assertLessEqual({t['depth'] for t in targets},{8,10,12,14})
                    for target in targets:
                        depths+=1;v=target['target_player0'];self.assertEqual(target['control_absolute_error_seeds'],abs(a['value_player0']-v));self.assertEqual(target['pilot_absolute_error_seeds'],abs(b['value_player0']-v));self.assertEqual(target['absolute_error_delta_seeds'],abs(b['value_player0']-v)-abs(a['value_player0']-v))
                        for link in target['sources']:
                            if link['path'].endswith('.sqlite3'):
                                with closing(dbopen(ROOT/link['path'])) as db:text,digest=db.execute('SELECT body,hash FROM '+link['table']+' WHERE state=?',(link['state_key'],)).fetchone()
                                item=json.loads(text);self.assertEqual(config.digest(item),digest);self.assertEqual(digest,target['item_sha256']);self.assertEqual((item['root_sha256'],item['label']['choice_depth'],item['label']['target_player0']),(raw['original_root_sha256'],target['depth'],v));self.assertEqual(item['label']['target_player0'],item['report']['depth_result']['value_player0']);self.assertEqual(item['label']['action_scores'],[dict(pit=c['pit'],value_player0=c['depth_result']['value_player0']) for c in item['report']['candidates']])
                            else:self.assertEqual(link,dict(path='build/value-pilot-rows.json',group='development',state_key=raw['state_key'],row_sha256=config.digest(raw)));self.assertIsNone(target['item_sha256']);self.assertEqual(v,y)
                    count+=1
        self.assertEqual(count,342);self.assertEqual(report['inference']['model_dense_forwards'],684)
        EVIDENCE['ordinary_per_root_audit']=dict(rows=count,ordinary_validation14_training61_diagnostic8=True,separate_base_ordinary_development=259,native_model_forwards=684,literal_model_forwards=684,requested_depth_comparisons=depths,all_source_SQL_and_original_ownership_links_checked=True,all_actor_player0_residual_errors_and_absolute_deltas_rederived=True,policy_logits_equal=True,finite_targets_not_game_truth=True)

    def test_03_all_summary_sensitivity_and_other_family_feature_neighbors(self):
        report=load(ORDINARY);data=load(ROOT/'build/value-pilot-rows.json')['groups'];training=data['train'];ordinary=[r for r in training if r['pool']=='ordinary'];allrows={r['state_key']:r for group in data.values() for r in group};nearest_count=0
        for name,group in report['groups'].items():
            self.assertEqual(group['summary'],summaries(group['rows']));h8=[]
            for record in group['rows']:
                y=next(t['target_player0'] for t in record['requested_depth_sensitivity'] if t['depth']==8);a=record['control']['prediction_player0'];b=record['pilot']['prediction_player0'];r=dict(record,control=errors(a,y,record['d'],record['signed_active'],record['actor']),pilot=errors(b,y,record['d'],record['signed_active'],record['actor']),absolute_error_delta_seeds=abs(b-y)-abs(a-y));h8.append(r)
                if record['nearest_training'] is None:self.assertEqual(name,'base_ordinary_development');continue
                raw=allrows[record['state_key']]
                for pool_name,chosen in record['nearest_training'].items():
                    pool=training if pool_name=='all_optimizer' else ordinary;eligible=[v for v in pool if v['family']!=raw['family']];distance=lambda v:math.fsum(abs(a-b) for a,b in zip(raw['cached_features']['x'],v['cached_features']['x']))/127
                    candidate=min(eligible,key=lambda v:(distance(v),v['state_key']));expected=dict(state_key=candidate['state_key'],family=candidate['family'],pool=candidate['pool'],stratum=candidate['stratum'],teacher_depth=candidate['teacher_depth'],actor=candidate['state']['controller'],active=sum(map(sum,candidate['state']['pits'])),store_difference=candidate['cached_features']['d'],mean_l1_raw127=distance(candidate),eligible_rows=len(eligible),source_row_sha256=config.digest(candidate));self.assertEqual(chosen,expected);self.assertNotEqual(chosen['family'],raw['family']);nearest_count+=1
            self.assertEqual(report['original_H8_same_prediction_sensitivity'][name],summaries(h8))
        val=report['groups']['ordinary_validation']['summary']['all'];self.assertEqual((val['rows'],val['improved'],val['worse'],val['tied']),(14,6,8,0));self.assertTrue(all(r['absolute_error_delta_seeds']>0 for r in report['groups']['ordinary_validation']['rows'] if r['band']=='50-74'))
        EVIDENCE['ordinary_aggregation_audit']=dict(all_four_groups_and_all_stratum_depth_actor_legal_bins_rederived=True,requested_H8_fixed_prediction_sensitivity_rederived=True,nearest_other_family_queries=nearest_count,all_nearest_distances_eligible_counts_and_ties_recomputed=True,validation_improved=6,validation_worse=8,all_six_50_74_validation_rows_worsen=True,leave_one_out_is_sensitivity_not_new_result=True,feature_similarity_not_coverage_or_capacity_proof=True)

    def test_04_exact51200_draw_CDF_root_family_pool_ledger(self):
        rows_=load(ROOT/'build/value-pilot-rows.json')['groups']['train'];sampling=load(ORDINARY)['sampling'];cumulative=[];total=0.
        for row in rows_:total+=row['sampling_probability'];cumulative.append(total)
        rng=random.Random(20261022);counts=[0]*len(rows_)
        for _ in range(51200):counts[bisect.bisect_right(cumulative,rng.random()*total)]+=1
        families=defaultdict(lambda:dict(roots=0,draws=0,probability=0.));pools=defaultdict(lambda:dict(roots=0,draws=0,probability=0.));expected=[]
        for row,count in zip(rows_,counts):
            p=row['sampling_probability'];expected.append(dict(state_key=row['state_key'],family=row['family'],pool=row['pool'],stratum=row['stratum'],probability=p,expected_draws=51200*p,observed_draws=count))
            for target in (families[row['family']],pools[row['pool']]):target['roots']+=1;target['draws']+=count;target['probability']+=p
        ff=[dict(family=name,**v,expected_draws=51200*v['probability']) for name,v in sorted(families.items())];self.assertEqual(sampling['roots'],expected);self.assertEqual(sampling['families'],ff);self.assertEqual(sampling['pools'],dict(sorted(pools.items())));self.assertEqual(sampling['root_draw_ledger_sha256'],config.digest([[r['state_key'],n] for r,n in zip(rows_,counts)]));self.assertEqual(sampling['family_draw_ledger_sha256'],config.digest([[v['family'],v['draws']] for v in ff]));self.assertEqual(sum(counts),51200)
        draws={r['state_key']:n for r,n in zip(rows_,counts)}
        for group in load(ORDINARY)['groups'].values():
            for row in group['rows']:self.assertEqual(row['sampler_observed_root_draws'],draws.get(row['state_key'],0));self.assertEqual(row['sampler_observed_family_draws'],families.get(row['family'],{}).get('draws',0))
        self.assertEqual(sum(n>0 for n in counts),7616);self.assertEqual(sum(v['draws']>0 for v in ff),4589);self.assertEqual(pools['ordinary']['draws'],2512);self.assertEqual(pools['base']['draws'],48688)
        EVIDENCE['sampler_audit']=dict(draws=51200,unique_roots_drawn=7616,unique_family_IDs_drawn=4589,ordinary_draws=2512,base_draws=48688,all7624_root_and4592_family_ledger_entries_reconstructed=True,row_order_CDF_and_digests_exact=True,reconstruction_not_separate_historical_draw_log=True,no_new_sampling_weights_applied=True)

    def arena_inputs(self):
        cls=type(self)
        if not hasattr(cls,'arena_sources'):
            cls.arena_sources=dict(pretraining_control=game_records('pretraining','policy_control'),control=game_records('posttraining','policy_control'),pilot=game_records('posttraining','value_pilot'))
            cls.arena_replays={name:{key:replay(self,record) for key,record in records.items()} for name,records in cls.arena_sources.items()}
        return cls.arena_sources,cls.arena_replays

    def test_05_all72_H8_games_family_WDL_tails_drift_and_leave_one_out(self):
        source,replayed=self.arena_inputs();report=load(ARENA);seats=report['seats'];self.assertEqual([(r['pair'],r['candidate_seat']) for r in seats],sorted(source['control']));families=[];plan=load(ROOT/'build/value-pilot-benchmark-plan.json')
        for row in seats:
            key=row['pair'],row['candidate_seat'];self.assertEqual(row['family'],source['control'][key]['body']['family']);self.assertEqual(row['stratum'],plan['arena_starts'][key[0]]['stratum'])
            for name in source:self.assertEqual(row[name],replayed[name][key]['summary'])
            self.assertEqual(row['pilot_minus_control_margin'],row['pilot']['margin_seeds']-row['control']['margin_seeds']);self.assertEqual(row['same_control_post_minus_pre_margin'],row['control']['margin_seeds']-row['pretraining_control']['margin_seeds'])
        for pair in range(12):
            rs=[r for r in seats if r['pair']==pair];families.append(dict(pair=pair,family=rs[0]['family'],stratum=rs[0]['stratum'],control_mean_margin=statistics.mean(r['control']['margin_seeds'] for r in rs),pilot_mean_margin=statistics.mean(r['pilot']['margin_seeds'] for r in rs),pretraining_control_mean_margin=statistics.mean(r['pretraining_control']['margin_seeds'] for r in rs),pilot_minus_control_paired_mean_margin=statistics.mean(r['pilot_minus_control_margin'] for r in rs),same_control_post_minus_pre_paired_mean_margin=statistics.mean(r['same_control_post_minus_pre_margin'] for r in rs),seat_contributions=[dict(seat=r['candidate_seat'],delta=r['pilot_minus_control_margin'],control_margin=r['control']['margin_seeds'],pilot_margin=r['pilot']['margin_seeds'],first_changed_actor_role=r['first_divergence'].get('first_changed_actor_role'),shared_choices=r['first_divergence']['shared_choices']) for r in rs]))
        self.assertEqual(report['families'],families);delta=[r['pilot_minus_control_paired_mean_margin'] for r in families];negative=sorted([r for r in families if r['pilot_minus_control_paired_mean_margin']<0],key=lambda r:(r['pilot_minus_control_paired_mean_margin'],r['pair']));positive=[r for r in families if r['pilot_minus_control_paired_mean_margin']>0];self.assertEqual(report['adverse_families_ranked'],[dict(rank=i+1,**r) for i,r in enumerate(negative)])
        loo=[dict(excluded_pair=r['pair'],excluded_family=r['family'],families_remaining=11,mean_pilot_minus_control_margin=statistics.mean(v for i,v in enumerate(delta) if i!=r['pair'])) for r in families];self.assertEqual(report['leave_one_family_out'],loo);gross=-sum(r['pilot_minus_control_paired_mean_margin'] for r in negative);agg=report['aggregate']
        for name in source:
            values=[r[name]['margin_seeds'] for r in seats];expected=dict(wdl=dict(wins=sum(v>0 for v in values),draws=sum(v==0 for v in values),losses=sum(v<0 for v in values)),mean_margin_seeds=statistics.mean(values))
            if name!='pretraining_control':expected['match_score']=statistics.mean(r[name]['match_score'] for r in seats)
            self.assertEqual(agg[name],expected)
        expected=dict(pilot_minus_control_paired_mean_margin=statistics.mean(delta),sum_seat_margin_deltas=sum(r['pilot_minus_control_margin'] for r in seats),same_control_post_minus_pre_mean_margin=statistics.mean(r['control']['margin_seeds'] for r in seats)-statistics.mean(r['pretraining_control']['margin_seeds'] for r in seats),negative_families=len(negative),positive_families=len(positive),unchanged_families=sum(v==0 for v in delta),gross_negative_paired_margin_mass=gross,gross_positive_paired_margin_mass=sum(r['pilot_minus_control_paired_mean_margin'] for r in positive),worst_family_pair=negative[0]['pair'] if negative else None,top_two_fraction_of_gross_adverse_mass=-sum(r['pilot_minus_control_paired_mean_margin'] for r in negative[:2])/gross if gross else None,leave_one_family_out_min=min(r['mean_pilot_minus_control_margin'] for r in loo),leave_one_family_out_max=max(r['mean_pilot_minus_control_margin'] for r in loo),leave_one_family_out_positive=sum(r['mean_pilot_minus_control_margin']>0 for r in loo))
        for name,v in expected.items():self.assertEqual(agg[name],v)
        choices=sum(r['summary']['choices'] for group in replayed.values() for r in group.values());self.assertEqual((report['counts']['families'],report['counts']['matched_seats'],report['counts']['replayed_games'],report['counts']['literal_choices']),(12,24,72,choices));self.assertEqual(choices,2460)
        EVIDENCE['arena_outcome_audit']=dict(families=12,matched_seats=24,games=72,literal_choices=choices,all_full_transitions_captures_bonus_passes_and_game_costs_independently_replayed=True,all_WDL_margin_tail_concentration_drift_and12_leave_one_family_out_means_rederived=True,paired_delta=statistics.mean(delta),gross_adverse_mass=gross,gross_favorable_mass=expected['gross_positive_paired_margin_mass'],LOO_not_a_confidence_interval_or_selection=True)

    def test_06_all_same_state_divergence_snapshots_Q_coverage_and_forward_facts(self):
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        source,replayed=self.arena_inputs();report=load(ARENA);groups=load(ROOT/'build/value-pilot-rows.json')['groups'];plan=load(ROOT/'build/value-pilot-benchmark-plan.json');labels=defaultdict(list)
        for group,rs in groups.items():
            for index,r in enumerate(rs):labels[config.canonical(r['state'])].append(dict(state=r['state'],source=dict(path='build/value-pilot-rows.json',group=group,row=index,state_key=r['state_key'],row_sha256=config.digest(r)),label=r['label'],depth=r['teacher_depth']))
        for index,r in enumerate(plan['cases']):labels[config.canonical(r['state'])].append(dict(state=r['state'],source=dict(path='build/value-pilot-benchmark-plan.json',case=index,state_key=r['state_key'],row_sha256=config.digest(r)),label=r['reference'],depth=r['reference'].get('choice_depth')))
        direct={};counts=Counter();observations=0
        with CpuBatchEvaluator(CONTROL,cache_capacity=0,max_batch=1) as left,CpuBatchEvaluator(PILOT,cache_capacity=0,max_batch=1) as right:
            for row in report['seats']:
                key=row['pair'],row['candidate_seat']
                for field,ln,rn in (('first_divergence','control','pilot'),('same_control_drift_divergence','pretraining_control','control')):
                    a=replayed[ln][key]['steps'];b=replayed[rn][key]['steps'];i=0
                    while i<min(len(a),len(b)) and (a[i]['actor'],a[i]['pit'])==(b[i]['actor'],b[i]['pit']):self.assertEqual((a[i]['before'],a[i]['after']),(b[i]['before'],b[i]['after']));i+=1
                    d=row[field];self.assertEqual(d['shared_choices'],i)
                    if i==min(len(a),len(b)):
                        self.assertEqual(d,dict(shared_choices=i,identical_action_trajectory=True,first_different_choice=None));self.assertEqual(len(a),len(b));continue
                    self.assertFalse(d['identical_action_trajectory']);self.assertEqual(a[i]['before'],b[i]['before']);state=a[i]['before'];actor=state['controller'];role='candidate_profile' if actor==key[1] else 'opponent_H8_relay';self.assertEqual((d['state'],d['first_different_choice'],d['actor'],d['candidate_seat'],d['first_changed_actor_role'],d['opponent_first']),(state,i+1,actor,key[1],role,actor!=key[1]));counts[(field,role)]+=1
                    names=('control','pilot') if field=='first_divergence' else ('pretraining_control','posttraining_control')
                    for output_name,source_name,steps in zip(names,(ln,rn),(a,b)):
                        record=source[source_name][key];raw=record['body']['decisions'][i];snapshot=d[output_name];self.assertEqual(snapshot['source'],dict(**record['source'],choice_index=i,decision_sha256=config.digest(raw)));self.assertEqual(snapshot['saved_decision'],raw);decision=raw['decision']['decision'];r=decision['report'];is_puct=decision['kind']=='puct'
                        expected=dict(engine=raw['engine'],actor=raw['actor'],chosen=decision['chosen'],display_chosen=decision['chosen']+1,kind=decision['kind'],wall_ms=decision['wall_ms'],requested_ms=decision['requested_ms'],overshoot_ms=decision['overshoot_ms'],nodes=decision['nodes'],completed_depth=decision['completed_depth'],completed_simulations=decision['completed_simulations'],estimated_value_player0=r['estimated_value_player0'] if is_puct else r['value_player0'],neural_bootstrap_player0=r['unvisited_root_q_player0'] if is_puct else None,visits=r['visits'] if is_puct else None,root_priors=r['root_priors'] if is_puct else None,visit_policy=r['policy'] if is_puct else None,action_mean_value_player0=r['action_mean_value_player0'] if is_puct else None,work=r['work'] if is_puct else {k:r.get(k) for k in ('deposits','feature_simulations','feature_deposits','nn_evaluations')},evaluator=raw['decision']['evaluator']);self.assertEqual(snapshot['summary'],expected);self.assertEqual(d[output_name+'_nearby_observations'],steps[max(0,i-2):i+4]);observations+=len(steps[max(0,i-2):i+4])
                    self.assertIn('unmatched',d['attribution'])
                    if field!='first_divergence':continue
                    found=labels.get(config.canonical(state),[]);comparisons=[];sign=1 if actor==0 else -1
                    for item in found:
                        self.assertEqual(item['state'],state);label=item['label'];q={v['pit']:v['value_player0'] for v in label['action_scores']};v=label['target_player0'];l=sign*(v-q[a[i]['pit']]);r=sign*(v-q[b[i]['pit']]);comparisons.append(dict(source=item['source'],teacher_depth=item['depth'],label_kind=label['kind'],label_sha256=config.digest(label),target_player0=v,best_moves=label['best_moves'],display_best_moves=[p+1 for p in label['best_moves']],control_action_Q_player0=q[a[i]['pit']],pilot_action_Q_player0=q[b[i]['pit']],control_actor_regret=l,pilot_actor_regret=r,pilot_minus_control_actor_regret=r-l))
                    self.assertEqual(d['existing_Q_comparison']['coverage'],'exact_full_state_stored_label' if found else 'uncovered');self.assertEqual(d['existing_Q_comparison']['comparisons'],comparisons);counts['covered' if found else 'uncovered']+=1
                    sk=config.canonical(state)
                    if sk not in direct:
                        before=left.evaluate_many([state])[0];after=right.evaluate_many([state])[0];before.pop('inference_ms');after.pop('inference_ms');self.assertEqual(before['policy_logits'],after['policy_logits']);direct[sk]=dict(control=before,pilot=after,pilot_minus_control_value_player0=after['value_player0']-before['value_player0'],policy_logits_identical=True)
                    actual=d['analysis_model_forwards']
                    for name,v in direct[sk].items():self.assertEqual(actual[name],v)
                    self.assertNotIn('inference_ms',actual['control']);self.assertNotIn('inference_ms',actual['pilot'])
        expected=dict(first_divergent_seats=sum(v for k,v in counts.items() if isinstance(k,tuple) and k[0]=='first_divergence'),opponent_first_divergences=counts['first_divergence','opponent_H8_relay'],candidate_first_divergences=counts['first_divergence','candidate_profile'],same_control_drift_divergent_seats=sum(v for k,v in counts.items() if isinstance(k,tuple) and k[0]=='same_control_drift_divergence'),divergent_seats_with_exact_state_Q=counts['covered'],divergent_seats_without_stored_Q=counts['uncovered'],analysis_forward_states=len(direct),analysis_CPU_model_forwards=2*len(direct))
        for name,v in expected.items():self.assertEqual(report['counts'][name],v)
        self.assertEqual(report['counts']['identical_action_trajectories'],24-expected['first_divergent_seats'])
        for field,aggregate in (('first_divergence','first_changed_actor_counts'),('same_control_drift_divergence','same_control_drift_first_changed_actor_counts')):self.assertEqual(report['aggregate'][aggregate],{role:counts[field,role] for role in ('candidate_profile','opponent_H8_relay') if counts[field,role]})
        EVIDENCE['divergence_audit']=dict(**expected,nearby_literal_step_observations=observations,all_saved_envelopes_source_hashes_visits_Q_priors_bootstrap_work_and_clocks_exact=True,raw_normalized_pits_stores_controller_match_only=True,no_player_or_store_transform=True,all_direct_CPU_forwards_recomputed_without_new_timing=True,policy_logits_unchanged=True,opponent_first_not_game_first=True,no_causal_blunder_or_counterfactual_search_claim=True)

    def test_07_readiness_claims_and_deterministic_readonly_reruns(self):
        import contextlib
        import io
        import runpy
        from congklak import value_pilot_learning as learning,value_pilot_benchmark as benchmark,quality_search
        from congklak.neural_relay import NeuralNative
        readiness=load(ROOT/'build/engine-readiness-current.json');self.assertEqual(readiness['schema'],'congklak-current-readiness-v1');self.assertTrue(readiness['passed']);self.assertEqual(readiness['larger_data']['engineering'],'ready_for_bounded_versioned_collection');self.assertEqual(readiness['larger_data']['learning_evidence'],'not_ready_for_blind_100k_or_long_training');self.assertEqual(readiness['product']['default'],'H_v1');self.assertEqual(readiness['product']['finished_strong_analyzer'],'not_release_ready')
        self.assertTrue(readiness['no_completion_percentage']);self.assertTrue(readiness['recommendations_are_unexecuted']);self.assertFalse(readiness['both_baseline_strength_gate_passed']);self.assertFalse(readiness['default_promotion']);self.assertFalse(readiness['new_training']);self.assertFalse(readiness['new_games']);self.assertFalse(readiness['new_teacher_labels']);gates={g['id']:g for g in readiness['gates']};self.assertEqual(len(gates),12);self.assertEqual(gates['gui_client']['status'],'optional_scope_decision');self.assertEqual(gates['full_game_proof']['status'],'future_goal_not_first_product_gate');self.assertEqual(gates['exact_retrieval']['status'],'implemented_separate_interface')
        for name,digest in readiness['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        for gate in gates.values():
            for name in gate['evidence']:self.assertTrue((ROOT/name).is_file(),name)
        configsource=(ROOT/'congklak/config.py').read_text();allowed=configsource.split('allowed=')[1].split('\n')[0];self.assertNotIn('"model"',allowed);self.assertNotIn('"backend"',allowed);self.assertIn('5,462',(ROOT/'docs/K4_ORACLE.md').read_text());self.assertIn('no natural strict ordinary observations in the 75-98 band',(ROOT/'docs/DATA_PILOT_RESULTS.md').read_text())
        post=load(ROOT/'build/value-pilot-benchmark-posttraining.json');self.assertFalse(post['both_baseline_primary_gates']['value_pilot']);self.assertGreater(load(ORDINARY)['groups']['ordinary_validation']['summary']['all']['delta_mae_seeds'],0)
        doc=(ROOT/'docs/ENGINE_READINESS_CURRENT.md').read_text();self.assertIn('single-writer SQLite',doc);self.assertIn('144 pilot game trajectories',doc);self.assertIn('61 training candidates',doc);self.assertIn('14 validation and 8 diagnostic',doc);self.assertIn('259 ordinary development',doc);self.assertIn('Hash',doc.replace('hash','Hash'));self.assertIn('not solve the 98-seed opening',doc)
        scripts=('tools/analyze-pilot-ordinary.py','tools/analyze-pilot-losses.py','tools/assess-engine-readiness.py');watched=[ORDINARY,ARENA,ROOT/'build/engine-readiness-current.json',CONTROL,PILOT,ROOT/'data/value-pilot-v1/catalog.sqlite3',ROOT/'build/value-pilot-rows.json',ROOT/'data/value-pilot-benchmark-pretraining-v1/work.sqlite3',ROOT/'data/value-pilot-benchmark-posttraining-v1/work.sqlite3'];before={relative(p):(sha(p),p.stat().st_size,p.stat().st_mtime_ns) for p in watched};stream=io.StringIO()
        with mock.patch.object(Native,'analyze',side_effect=AssertionError('New teacher search forbidden')) as native,mock.patch.object(NeuralNative,'analyze',side_effect=AssertionError('New neural search forbidden')) as neural,mock.patch.object(learning,'runtime',side_effect=AssertionError('Training runtime forbidden')) as runtime,mock.patch.object(learning,'train',side_effect=AssertionError('Training forbidden')) as training,mock.patch.object(benchmark,'decide',side_effect=AssertionError('New game decisions forbidden')) as decide,mock.patch.object(quality_search,'search',side_effect=AssertionError('New PUCT forbidden')) as puct:
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                for name in scripts:
                    with mock.patch.object(sys,'argv',[str(ROOT/name)]):
                        try:runpy.run_path(str(ROOT/name),run_name='__main__')
                        except SystemExit as exc:self.assertIn(exc.code,(0,None))
            for patched in (native,neural,runtime,training,decide,puct):self.assertEqual(patched.call_count,0)
        (self.base/'readonly-rerun.log').write_text(stream.getvalue(),encoding='utf-8');after={relative(p):(sha(p),p.stat().st_size,p.stat().st_mtime_ns) for p in watched};self.assertEqual(before,after);self.assertNotIn('torch',sys.modules)
        for name,digest in load(BASELINE)['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        EVIDENCE['readiness_and_no_work_audit']=dict(readiness_gates=12,all_claimed_artifacts_current=True,bounded_versioned_collection_distinct_from_mass_training_release=True,sharding_not_claimed_implemented=True,K4_not_full98seed_solving=True,GUI_optional_and_proof_stages_later=True,future_role_aware_ownership_gate_explicit=True,no_completion_percentage=True,readiness_recommendations_unexecuted=True,rerun_outputs=3,watched_artifacts=len(watched),all_bytes_sizes_mtimes_unchanged=True,new_native_neural_PUCT_searches=0,new_training_runtime_or_calls=0,new_benchmark_decisions=0,Torch_imported=False,all2600_prior_pins_current=True)


def main():
    started=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(PilotLossReviewIndependentAudit);methods=unittest.defaultTestLoader.getTestCaseNames(PilotLossReviewIndependentAudit);result=unittest.TextTestRunner(verbosity=2).run(suite)
    names=('tests/pilot_loss_review_independent_audit.py','tools/analyze-pilot-ordinary.py','tools/analyze-pilot-losses.py','tools/assess-engine-readiness.py','tools/verify-pilot-loss-review-rerun.py','build/pilot-loss-review-baseline.json','build/pilot-loss-review-ordinary.json','build/pilot-loss-review-arena.json','build/engine-readiness-current.json','docs/ENGINE_READINESS_CURRENT.md','docs/PILOT_ORDINARY_ERRORS.md');paths=[ROOT/name for name in names];paths += [p for p in PilotLossReviewIndependentAudit.base.rglob('*') if p.is_file()]
    body=dict(schema='pilot-loss-review-independent-audit-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),covered_methods=methods,inputs=pins(paths),evidence=EVIDENCE,retained_directory=relative(PilotLossReviewIndependentAudit.base),duration_seconds=time.monotonic()-started,failure_details=[dict(method=str(t),trace=trace) for t,trace in result.failures+result.errors]);save(ROOT/'build/pilot-loss-review-audit.json',body);print(config.canonical(dict(event='pilot_loss_review_independent_audit',passed=body['passed'],tests=result.testsRun)),flush=True);return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
