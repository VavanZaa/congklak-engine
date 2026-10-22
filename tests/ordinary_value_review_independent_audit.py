"""Independent saved-data audit; no training, search, games or Torch."""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import closing
import argparse,bisect,copy,importlib,json,math,random,statistics,struct,sys,time,unittest,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config
from data_pilot_independent_audit import sha,load,save,relative,pins,dbopen,norm,position,quotient,mechanism,rows,run_child
from stage5_deeper_audit_acceptance import literal_features,value_and_policy
import reference as literal

BASELINE=ROOT/'build/ordinary-value-review-baseline.json'
WORK=ROOT/'build/ordinary-value-review-audit'
FREEZE=ROOT/'build/ordinary-value-review-audit-source-freeze.json'
OUTPUT=ROOT/'build/ordinary-value-review-audit.json'
CONTROL=ROOT/'data/quality-train-v1/model.json'
CANDIDATE=ROOT/'data/ordinary-value-pilot-v1/model.json'
JOURNAL=ROOT/'data/ordinary-value-benchmark-v1/work.sqlite3'
MODULES=[dict(module='congklak.ordinary_value_overlap_review',report='build/ordinary-value-overlap-review.json',cli='tools/review-ordinary-value-overlap.py'),
 dict(module='congklak.ordinary_value_loss_review',report='build/ordinary-value-loss-review.json',cli='tools/review-ordinary-value-losses.py'),
 dict(module='congklak.ordinary_value_validation_review',report='build/ordinary-value-validation-review.json',cli='tools/review-ordinary-value-validation.py',verify_args=['--verify']),
 dict(module='congklak.ordinary_value_review_conclusions',report='build/ordinary-value-review-conclusions.json',cli='tools/review-ordinary-value-conclusions.py')]
FROZEN={'layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias','policy.weight','policy.bias'}
EVIDENCE={}


def raw(p):return bytes([*p['pits'][0],p['stores'][0],*p['pits'][1],p['stores'][1],p['controller']]).hex()
def swap(p):return dict(pits=[p['pits'][1],p['pits'][0]],stores=p['stores'][::-1],controller=1-p['controller'])
def active(p):return sum(map(sum,p['pits']))
def band(p):
    n=active(p);return '0-4' if n<=4 else '5-24' if n<=24 else '25-49' if n<=49 else '50-74' if n<=74 else '75-98'
def flat(v):return [x for row in v for x in row] if isinstance(v[0],list) else v
def bits(v):
    v=flat(v);return struct.pack('<'+'f'*len(v),*v)
def signature(path):
    p=Path(path);return dict(sha256=sha(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)


def sampler(groups):
    """Use explicit integer draws, independent of producer choice/CDF helpers."""
    rng=random.Random(20261028);counts=Counter(r['stratum'] for r in groups['rehearsal']);cdf=[];mass=0.;byroot={}
    for r in groups['rehearsal']:
        mass+=.9/len(groups['rehearsal'])+.1/(len(counts)*counts[r['stratum']]);cdf.append(mass)
    for pool in ('high','low'):
        values=defaultdict(list)
        for r in groups[pool]:values[r['state_key']].append(r)
        byroot[pool]=[values[k] for k in sorted(values)]
    ledger=Counter()
    for _ in range(200):
        for pool in ('high','low'):
            for _ in range(16):
                chosen=byroot[pool][rng.randrange(len(byroot[pool]))]
                ledger[chosen[rng.randrange(len(chosen))]['id']]+=1
        for _ in range(32):ledger[groups['rehearsal'][bisect.bisect_right(cdf,rng.random()*cdf[-1])]['id']]+=1
    return dict(sorted(ledger.items())),rng.getstate(),cdf


def dense(p,model):
    x,d,r=literal_features(norm(p),127);h=list(x);w=model['weights']
    for prefix in ('layers.0','layers.2'):
        h=[max(0.,sum(a*b for a,b in zip(weight,h))+bias) for weight,bias in zip(w[prefix+'.weight'],w[prefix+'.bias'])]
    t=sum(a*b for a,b in zip(w['layers.4.weight'][0],h))+w['layers.4.bias'][0]
    logits=[sum(a*b for a,b in zip(row,h))+bias for row,bias in zip(w['policy.weight'],w['policy.bias'])]
    return dict(hidden=h,t=t,z=math.tanh(t),value=d+r*math.tanh(t),logits=logits,d=d,r=r,x=list(x))


def independent_index_states(plan):
    """Read only the consumed scope, filtering 9k root metadata first."""
    states=set();entries=0
    for group in load(ROOT/'build/value-pilot-rows.json')['groups'].values():
        for row in group:states.add(raw(row['state']));entries+=1
    for row in plan['cases']:states.add(raw(row['state']));entries+=1
    sources={'ordinary-data-pilot-v1':['labels'],'natural-study-v1':['h8_labels','h10_labels'],
        'horizon-finish-v1':['h8_references','h10_references','h12_references','prior_h14_labels','tier1_labels','tier2_labels'],
        'midgame-label-pilot-v1':['old_h8_references','h8_labels','h10_labels','h12_labels'],
        'ordinary-value-labels-v1':['h8_labels','h10_labels','h12_labels'],'deeper-9000-v2':['labels']}
    for directory,tables in sources.items():
        with closing(dbopen(ROOT/'data'/directory/'work.sqlite3')) as db:
            for key,text,digest in db.execute('SELECT state,body,hash FROM roots ORDER BY state'):
                row=json.loads(text);assert config.digest(row)==digest
                if directory=='deeper-9000-v2' and row['split'] not in ('train','development'):continue
                for table in tables:
                    # Existence suffices for coverage; unrelated label bodies remain unread.
                    if db.execute('SELECT 1 FROM '+table+' WHERE state=?',(key,)).fetchone():states.add(raw(row['state']));entries+=1
    return states,entries


class OrdinaryValueReviewIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True,exist_ok=True);cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()
        cls.reports=[load(ROOT/v['report']) for v in MODULES];cls.overlap,cls.loss,cls.validation,cls.conclusions=cls.reports
        cls.groups=load(ROOT/'build/ordinary-value-rows.json')['groups'];cls.control=load(CONTROL)['model'];cls.candidate=load(CANDIDATE)['model']
        cls.ledger,cls.rng,cls.cdf=sampler(cls.groups)
    def close(self,a,b,msg=None):self.assertAlmostEqual(a,b,delta=1e-10,msg=msg)
    def fresh(self,name):p=self.base/name;p.mkdir();return p
    def current(self,mapping):
        for name,digest in mapping.items():self.assertEqual(sha(ROOT/name),digest,name)

    def test_01_prior3779_fixed_models_frozen_policy_and_source_identity(self):
        b=load(BASELINE);self.assertEqual((b['inputs_count'],b['prior_accepted_inputs']),(3779,3776));self.assertEqual(len(b['inputs']),3779);self.current(b['inputs'])
        self.assertEqual(sha(CONTROL),'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558')
        self.assertEqual(self.control['identity']['rules_digest'],config.RULES_DIGEST);self.assertEqual(self.candidate['identity']['rules_digest'],config.RULES_DIGEST)
        w0=self.control['weights'];w1=self.candidate['weights'];self.assertEqual(set(w0),FROZEN|{'layers.4.weight','layers.4.bias'})
        for name in FROZEN:self.assertEqual(bits(w0[name]),bits(w1[name]),name)
        self.assertEqual({k for k in w0 if bits(w0[k])!=bits(w1[k])},{'layers.4.weight','layers.4.bias'})
        self.assertEqual(sum(len(flat(w0[k])) for k in FROZEN),12807);self.assertEqual(sum(len(flat(w0[k])) for k in set(w0)-FROZEN),65)
        for item,body in zip(MODULES,self.reports):
            self.assertTrue(body['passed']);self.current(body['inputs']);self.assertIn(item['cli'],body['inputs']);self.assertIn(item['module'].replace('.','/')+'.py',body['inputs'])
        old=load(ROOT/'build/ordinary-value-acceptance.json');self.assertTrue(old['passed']);self.assertEqual(len(old['inputs']),3776)
        self.assertEqual(self.overlap['summary']['original_allocation'],'hold');self.assertEqual(self.conclusions['decision']['current_allocation'],'hold')
        self.assertNotIn('torch',sys.modules)
        EVIDENCE['identity']=dict(all3779_prior_hashes_current=True,six_frozen_tensors_bit_exact=True,frozen_parameters=12807,changed_value_parameters=65,rules_unchanged=True,current_review_source_pins_verified=True)

    def test_02_independent_sampler_overlap_orientation_and_ancestral_exposure(self):
        self.assertEqual({k:len(v) for k,v in self.groups.items()},dict(high=24,low=24,rehearsal=7539));self.assertEqual((sum(self.ledger.values()),len(self.ledger)),(12800,4366))
        previous=load(ROOT/'build/ordinary-value-audit/model.json')['evidence']['production_optimizer_audit'];self.assertEqual(self.ledger,previous['draw_ledger'])
        self.assertEqual(config.digest(self.ledger),self.overlap['ledger_sha256']);self.assertEqual(self.overlap['draw_totals'],dict(high=3200,low=3200,rehearsal=6400))
        allrows=[r for group in self.groups.values() for r in group];byid={r['id']:r for r in allrows};indexes={}
        for pool,rr in (('admitted',allrows),('drawn',[r for r in allrows if self.ledger.get(r['id'],0)])):
            ix={relation:defaultdict(list) for relation in ('raw_full_state','swapped_full_state','quotient','family')}
            for r in rr:
                for relation,key in (('raw_full_state',raw(r['state'])),('swapped_full_state',min(raw(r['state']),raw(swap(r['state'])))),('quotient',quotient(r['state'])),('family',r['family'])):ix[relation][key].append(r['id'])
            indexes[pool]=ix
        counts={p:Counter({r:0 for r in ix}) for p,ix in indexes.items()};bands={p:defaultdict(lambda:Counter({r:0 for r in ix})) for p,ix in indexes.items()};collisions=[];observations=Counter();total=0
        with closing(dbopen(JOURNAL)) as db:
            for game,choice,text,digest in db.execute('SELECT game,choice,body,hash FROM decisions ORDER BY game,choice'):
                row=json.loads(text);self.assertEqual(config.digest(row),digest);p=row['state'];total+=1;observations[band(p)]+=1
                if active(p)<=4:continue
                keys=dict(raw_full_state=raw(p),swapped_full_state=min(raw(p),raw(swap(p))),quotient=quotient(p),family=load_game_family(db,game))
                matches={pool:{relation:ix[relation].get(keys[relation],[]) for relation in ix} for pool,ix in indexes.items()}
                for pool in indexes:
                    for relation,ids in matches[pool].items():counts[pool][relation]+=bool(ids);bands[pool][band(p)][relation]+=bool(ids)
                if any(ids for pp in matches.values() for ids in pp.values()):collisions.append((game,choice,row,matches))
        self.assertEqual(total,3268);self.assertEqual(self.overlap['observation_bands'],dict(observations));self.assertEqual(self.overlap['match_counts'],{p:dict(c) for p,c in counts.items()});self.assertEqual(self.overlap['by_active_band'],{p:{b:dict(c) for b,c in bb.items()} for p,bb in bands.items()})
        self.assertEqual([(g,c) for g,c,_,_ in collisions],[(0,31),(4,31)])
        for (g,c,row,matches),saved in zip(collisions,self.overlap['collisions']):
            self.assertEqual(saved['matches'],matches);self.assertEqual(saved['source_decision_sha256'],config.digest(row));self.assertEqual(saved['state'],row['state']);self.assertEqual(active(row['state']),5)
            a=saved['alignments'][0];r=byid[a['training_id']];aligned=swap(r['state']);self.assertEqual(a['aligned_training_state'],aligned);self.assertNotEqual(raw(aligned),raw(row['state']));self.assertEqual(aligned['pits'],row['state']['pits'])
            self.assertEqual(a['store_offsets_per_player'],[row['state']['stores'][i]-aligned['stores'][i] for i in range(2)]);self.assertEqual(a['store_offsets_per_player'],[5,-5]);self.assertEqual(a['player0_margin_offset'],10)
            x0,d0,r0=literal_features(norm(r['state']),127);x1,d1,r1=literal_features(norm(row['state']),127);self.assertEqual(x0,x1);self.assertEqual(config.digest(list(x0)),a['encoder_inputs_sha256']);self.assertEqual([d0,r0],a['source_offset_and_scale']);self.assertEqual([d1,r1],a['arena_offset_and_scale'])
            self.assertEqual(self.ledger.get(r['id'],0),0);self.assertEqual(a['current_pilot_draws'],0);self.assertFalse(a['target_transferred_to_arena']);self.assertIsNone(a['source_control_actual_draw_count']);self.assertTrue(a['source_control_positive_draw_exposure_from_sealed_all_roots_schedule'])
        quality=load(ROOT/'build/quality-data.json');review=load(ROOT/'build/sampling-review.json');qset=set(quality['train_state_keys']);obs=review['observed_accepted_run'];self.assertEqual((len(qset),obs['unique_roots_drawn'],obs['total_draws']),(7669,7669,384000))
        for _,_,_,matches in collisions:
            for rid in matches['admitted']['quotient']:self.assertIn(byid[rid]['source_id'],qset)
        self.assertTrue(self.overlap['ancestral_exposure']['matching_row_positive_prior_draw_exposure']);self.assertFalse(self.overlap['ancestral_exposure']['exact_old_per_row_counts_recomputed']);self.assertFalse(self.overlap['ancestral_exposure']['nonzero_per_row_gradient_or_game_causality_established'])
        registry=load(ROOT/'build/ordinary-value-registry.json');self.assertFalse({r['family'] for r in allrows}&set(registry['families']));self.assertFalse({quotient(r['state']) for r in allrows}&set(registry['quotients']))
        EVIDENCE['overlap']=dict(current_draws_independently_reconstructed=12800,unique_drawn_rows=4366,all_admitted_rows=7587,all_arena_observations=3268,admitted_quotient_observations=2,current_drawn_quotient_observations=0,raw_full_or_swapped_full_or_family_matches=0,positive_ancestral_draw_exposure_from_sealed_all7669_support=True,new384000_draw_replay=False,encoder_inputs_equal_with_store_offset=[5,-5],labels_transferred=False,causal_effect_claimed=False)

    def test_03_all96_literal_games_matched_margins_and_saved_divergences(self):
        plan=load(ROOT/'build/ordinary-value-benchmark-plan.json');games={};records=defaultdict(list)
        with closing(dbopen(JOURNAL)) as db:
            for idx,comp,pair,seat,text,digest in db.execute('SELECT idx,comparison,pair,seat,body,hash FROM games ORDER BY idx'):
                g=json.loads(text);self.assertEqual(config.digest(g),digest);self.assertEqual((g['comparison'],g['pair'],g['candidate_seat']),(comp,pair,seat));games[idx]=g
            for game,choice,text,digest in db.execute('SELECT game,choice,body,hash FROM decisions ORDER BY game,choice'):
                row=json.loads(text);self.assertEqual(config.digest(row),digest);self.assertEqual(choice,len(records[game]));records[game].append(row)
        self.assertEqual((len(games),sum(map(len,records.values()))),(96,3268));replays={r['game_index']:r for r in self.loss['game_replays']};total=0
        for idx,g in games.items():
            p=norm(g['start']);saved=replays[idx];self.assertEqual(g['start'],plan['arena_starts'][g['pair']]['state']);self.assertEqual(g['moves'],[r['action'] for r in records[idx]]);self.assertEqual(g['decisions'],[r['row'] for r in records[idx]])
            costs=defaultdict(lambda:dict(decisions=0,wall_ms=0.,overshoot_ms=0.,nodes=0,completed_simulations=0,physical_forwards=0,feature_simulations=0,feature_deposits=0))
            for choice,row in enumerate(records[idx]):
                self.assertEqual(row['state'],position(p));a=row['action'];self.assertIn(a,literal.legal_actions(p));t=literal.apply(p,a);after=position(t.position);self.assertEqual(row['after'],after);self.assertFalse(row['row']['fallback'])
                store=7+8*t.actor;before=p.counts[store]+sum(e['kind']=='deposit' and e['index']==store for e in t.events)
                captures=[dict(actor=t.actor,after=before+e['seeds'],before=before,count=e['seeds'],hand=0,opposite=e['opposite'],slot=e['index'],type='capture') for e in t.events if e['kind']=='capture']
                last=next(e for e in reversed(t.events) if e['kind']=='deposit')['index'];reason='own_store' if t.bonus else 'own_capture' if t.captured else 'own_empty' if last in literal.row_indices(7,t.actor) else 'opponent_empty'
                step=dict(choice_index=choice,choice=choice+1,before_state_key=raw(row['state']),after_state_key=raw(after),actor=t.actor,engine=row['row']['engine'],pit=a,display_pit=a+1,store_gain=[after['stores'][j]-row['state']['stores'][j] for j in (0,1)],capture_events=captures,capture_transfer=t.captured,bonus_choice=t.bonus and not t.terminal,forced_passes_after=[dict(zip(('from','to'),pair)) for pair in t.final_passes],terminal_after=t.terminal,reason=reason,relays=t.relay_pickups,deposits=t.deposits)
                self.assertEqual(step,saved['steps'][choice]);env=row['row']['decision']['decision'];telemetry=row['row']['decision']['evaluator'];report=env['report'];cost=costs[row['row']['engine']]
                for key,value in dict(decisions=1,wall_ms=env['wall_ms'],overshoot_ms=env['overshoot_ms'],nodes=env['nodes'],completed_simulations=env['completed_simulations'] or 0,physical_forwards=telemetry['physical_forward_executions'] if telemetry else report.get('nn_evaluations',0),feature_simulations=telemetry['counts']['feature_simulations'] if telemetry else report.get('feature_simulations',0),feature_deposits=telemetry['counts']['feature_deposits'] if telemetry else report.get('feature_deposits',0)).items():cost[key]+=value
                p=t.position;total+=1
            self.assertTrue(literal.normalize(p).terminal);self.assertEqual(position(p),g['final']);margin=literal.evaluate(p)*(1 if g['candidate_seat']==0 else -1);self.assertEqual(margin,g['candidate_final_margin']);self.assertEqual(margin,saved['margin_seeds']);self.assertEqual(dict(costs),saved['cost_by_engine']);self.assertEqual(config.digest(g),saved['game_sha256'])
        indexed,indexentries=independent_index_states(plan);self.assertEqual(indexentries,self.loss['label_index_scope']['indexed_entries']);self.assertEqual(len(indexed),self.loss['label_index_scope']['indexed_unique_full_states'])
        lookup={(g['comparison'],g['pair'],g['candidate_seat']):i for i,g in games.items()};seats={};divstates=set();divergences=0
        for saved in self.loss['seats']:
            base,pair,seat=saved['baseline'],saved['pair'],saved['candidate_seat'];ci=lookup[('ordinary_control_vs_'+base,pair,seat)];ti=lookup[('ordinary_value_vs_'+base,pair,seat)];c,t=games[ci],games[ti];cr,tr=records[ci],records[ti]
            self.assertEqual((saved['control_game_index'],saved['candidate_game_index']),(ci,ti));delta=t['candidate_final_margin']-c['candidate_final_margin'];self.assertEqual(saved['candidate_minus_control_margin'],delta);seats[(base,pair,seat)]=saved
            n=0
            while n<min(len(cr),len(tr)) and cr[n]['action']==tr[n]['action']:self.assertEqual(cr[n]['state'],tr[n]['state']);n+=1
            d=saved['first_divergence'];self.assertEqual(d['shared_choices'],n)
            if n==len(cr)==len(tr):self.assertTrue(d['identical_action_trajectory']);continue
            divergences+=1;state=cr[n]['state'];self.assertEqual(state,tr[n]['state']);self.assertEqual(d['state'],state);self.assertEqual(d['choice_index'],n);self.assertEqual(d['first_different_choice'],n+1);self.assertEqual(d['opponent_first'],state['controller']!=seat);self.assertEqual(d['divergence_stratum'],mechanism(state)[1]);self.assertNotIn(raw(state),indexed);self.assertEqual(d['existing_Q_comparison']['coverage'],'uncovered_in_index');self.assertEqual(d['existing_Q_comparison']['comparisons'],[])
            for name,rr in (('control',cr),('candidate',tr)):
                snap=d[name];self.assertEqual(snap['saved_decision'],rr[n]['row']);self.assertEqual(snap['source']['decision_sha256'],config.digest(rr[n]));env=rr[n]['row']['decision']['decision'];summary=snap['summary'];self.assertEqual(summary['completed_depth'],env['completed_depth']);self.assertEqual(summary['completed_simulations'],env['completed_simulations']);self.assertEqual(summary['wall_ms'],env['wall_ms']);self.assertEqual(summary['evaluator'],rr[n]['row']['decision']['evaluator']);self.assertEqual(d[name+'_nearby_observations'],replays[ci if name=='control' else ti]['steps'][max(0,n-2):n+4])
            if state['controller']==seat:self.assertEqual(d['control']['summary']['root_priors'],d['candidate']['summary']['root_priors']);self.assertTrue(d['candidate_priors_equal'])
            divstates.add(raw(state))
        self.assertEqual((len(seats),divergences,len(divstates)),(48,10,9));self.assertEqual(set(self.loss['literal_divergence_predictions']),divstates)
        for key,saved in self.loss['literal_divergence_predictions'].items():
            for name,model in (('control',self.control),('candidate',self.candidate)):
                v,u,l=value_and_policy(norm(saved['state']),model);self.close(v,saved[name]['value_player0']);self.assertEqual(u,saved[name]['quantized_value_units'])
                for actual,expected in zip(l,saved[name]['policy_logits']):self.close(actual,expected)
            self.assertFalse(saved['new_timing_measured']);self.assertEqual(saved['control']['policy_logits'],saved['candidate']['policy_logits'])
        for baseline in ('H_v1','H8_relay_CPU'):
            values=[v for (b,_,_),v in seats.items() if b==baseline];summary=self.loss['summary'][baseline];familydeltas=[]
            for pair in range(12):
                subset=[seats[(baseline,pair,s)] for s in (0,1)];mean=statistics.mean(v['candidate_minus_control_margin'] for v in subset);familydeltas.append(mean);f=next(x for x in self.loss['families'] if x['baseline']==baseline and x['pair']==pair);self.assertEqual(f['paired_mean_candidate_minus_control'],mean)
            self.close(summary['paired_mean_candidate_minus_control'],statistics.mean(familydeltas))
            for key,name in (('control_wdl','control_margin'),('candidate_wdl','candidate_margin')):self.assertEqual(summary[key],dict(wins=sum(v[name]>0 for v in values),draws=sum(v[name]==0 for v in values),losses=sum(v[name]<0 for v in values)))
            self.assertEqual(summary['identical_action_trajectories'],sum(v['first_divergence']['identical_action_trajectory'] for v in values));self.assertEqual(summary['negative_delta_seats'],sum(v['candidate_minus_control_margin']<0 for v in values))
            for row in summary['leave_one_family_out_means']:self.close(row['paired_mean_candidate_minus_control'],statistics.mean(x for i,x in enumerate(familydeltas) if i!=row['omitted_pair']))
        negatives=[(p,s,v['candidate_minus_control_margin']) for (b,p,s),v in seats.items() if b=='H_v1' and v['candidate_minus_control_margin']<0];self.assertEqual(negatives,[(4,1,-6),(6,1,-4),(7,1,-6)])
        expectedlosses=sorted((v for (b,_,_),v in seats.items() if b=='H_v1' and v['candidate_margin']<0),key=lambda v:(v['candidate_margin'],v['pair'],v['candidate_seat']));self.assertEqual(len(expectedlosses),13)
        self.assertEqual([(v['pair'],v['candidate_seat'],v['candidate_margin']) for v in expectedlosses],[(v['pair'],v['candidate_seat'],v['candidate_margin']) for v in self.loss['absolute_H_v1_losses']])
        EVIDENCE['arena']=dict(games_literally_replayed=96,literal_choices=total,matched_seats=48,matched_family_units=24,all_absolute_H_v1_losses=13,identical_H_v1_paths=19,identical_H8_paths=19,first_divergences=10,unique_divergence_states=9,literal_fixed_model_forwards=18,exact_state_index_covered_divergences=0,losses_with_identical_control_paths=12,H_v1_negative_seats=negatives,causal_blunder_attribution=False)

    def test_04_six_roots_errors_actor_bounds_head_algebra_and_interpolation(self):
        j=ROOT/'data/ordinary-value-labels-v1/work.sqlite3';rootmap=rows(j,'roots','state');labels={s:rows(j,s+'_labels','state') for s in ('h8','h10','h12')};self.assertEqual(set(rootmap),set(self.validation['validation_roots']));models={'control':self.control,'candidate':self.candidate};forward={};allerrors={s:[] for s in labels}
        for key,row in rootmap.items():
            saved=self.validation['validation_roots'][key];p=row['state'];a=dense(p,self.control);b=dense(p,self.candidate);forward[key]=(a,b);self.assertEqual(a['logits'],b['logits']);self.assertEqual(a['logits'],saved['control_policy_logits']);self.close(a['value'],saved['control_prediction_player0']);self.close(b['value'],saved['candidate_prediction_player0']);sign=1 if p['controller']==0 else -1
            self.assertEqual(saved['source_wrapper_sha256'],config.digest(row));self.assertEqual(saved['original_root_sha256'],config.digest(row['root']));self.assertEqual(saved['D_player0'],a['d']);self.assertEqual(saved['signed_active'],a['r']);self.assertEqual(saved['actor_sign'],sign)
            for stage in labels:
                item=labels[stage][key];v=saved['horizons'][stage];y=item['label']['target_player0'];self.assertEqual(item['root_sha256'],config.digest(row['root']));self.assertEqual(v['target_player0'],y);self.assertEqual(v['requested_depth'],int(stage[1:]));self.assertEqual(v['label_sha256'],config.digest(item['label']));self.assertEqual(v['report_sha256'],config.digest(item['report']));z=(y-a['d'])/a['r'];self.assertEqual(v['physically_bounded_residual_target'],abs(z)<=1);self.assertEqual(v['exact_finite_head_representable'],abs(z)<1);self.assertEqual(v['saturation_endpoint'],abs(z)==1)
                for name,f in (('control',a),('candidate',b)):
                    e=v[name];error=f['value']-y;self.close(e['target_actor'],sign*y);self.close(e['target_residual'],z);self.close(e['signed_error_player0'],error);self.close(e['signed_error_actor'],sign*error);self.close(e['signed_error_residual'],error/a['r']);self.close(e['absolute_error_seeds'],abs(error));self.close(e['absolute_error_residual'],abs(error/a['r']))
                delta=abs(b['value']-y)-abs(a['value']-y);self.close(v['absolute_error_delta'],delta)
                if stage=='h10':self.close(v['H10_aggregate_contribution'],delta/6)
                else:self.assertIsNone(v['H10_aggregate_contribution'])
                allerrors[stage].append((key,row,abs(a['value']-y),abs(b['value']-y)))
            bias=self.candidate['weights']['layers.4.bias'][0]-self.control['weights']['layers.4.bias'][0];part=sum((wb-wa)*h for wa,wb,h in zip(self.control['weights']['layers.4.weight'][0],self.candidate['weights']['layers.4.weight'][0],a['hidden']));shift=saved['head_shift'];self.close(shift['bias_contribution'],bias);self.close(shift['hidden_weight_contribution'],part);self.close(shift['actual_preactivation_change'],b['t']-a['t']);self.close(part+bias,b['t']-a['t']);self.assertEqual(shift['fixed_store_offset_change'],0)
            target=saved['horizons']['h10']['target_player0'];needed=math.atanh((target-a['d'])/a['r'])-a['t'];self.close(shift['H10_needed_preactivation_change'],needed);self.assertEqual(shift['H10_direction_aligned'],(b['t']-a['t'])*needed>=0);self.close(shift['control_seed_gain_per_preactivation'],a['r']*(1-a['z']**2));self.close(shift['candidate_seed_gain_per_preactivation'],b['r']*(1-b['z']**2))
        for stage,items in allerrors.items():
            for scope,subset in [('all',items)]+[(band,[v for v in items if v[1]['stratum'].startswith(band+'/')]) for band in ('25-49','50-74')]:
                report=self.validation['horizon_summary'][stage]['all'] if scope=='all' else self.validation['horizon_summary'][stage]['by_band'][scope];self.assertEqual(report['roots'],len(subset));self.close(report['control_mae_seeds'],statistics.mean(v[2] for v in subset));self.close(report['candidate_mae_seeds'],statistics.mean(v[3] for v in subset));self.close(report['mae_delta_seeds'],statistics.mean(v[3]-v[2] for v in subset))
        self.assertEqual(self.validation['head_interpolation']['coefficients'],[0.,.25,.5,.75,1.]);self.assertFalse(self.validation['head_interpolation']['model_or_checkpoint_selected'])
        for coefficient in (0.,.25,.5,.75,1.):
            saved=self.validation['head_interpolation']['values'][str(coefficient)];predictions={k:a['d']+a['r']*math.tanh((1-coefficient)*a['t']+coefficient*b['t']) for k,(a,b) in forward.items()}
            for k,v in predictions.items():self.close(saved['predictions_player0'][k],v)
            for stage in labels:self.close(saved['mae_seeds'][stage],statistics.mean(abs(v-labels[stage][k]['label']['target_player0']) for k,v in predictions.items()))
        h=self.validation['horizon_summary']['h10']['all'];self.assertEqual((h['improved'],h['worsened'],h['tied']),(3,3,0));self.assertTrue(all(r['state']['controller']==1 for r in rootmap.values() if r['stratum'].startswith('50-74/')))
        EVIDENCE['six_root_math']=dict(roots=6,requested_references=18,all_actor_player0_residual_error_signs_verified=True,policy_logits_equal=True,head_bias_and_hidden_algebra_checked=True,mathematical_interpolation_coefficients=[0.,.25,.5,.75,1.],H10_control_MAE=h['control_mae_seeds'],H10_candidate_MAE=h['candidate_mae_seeds'],improve=3,worsen=3,all_three_high_actor1_confounded=True,H12_not_game_truth=True,model_selected=False)

    def test_05_exposure_pair_noise_neighbors_protection_and_conclusions(self):
        v=self.validation;ex=v['exposure'];self.assertEqual(ex['final_sampler_rng_sha256'],config.digest(self.rng));self.assertEqual(ex['total_draws'],12800);self.assertEqual(len(ex['ledger']),7587);rootgroups=defaultdict(list);neighborstates=set();allstate=set()
        strata=Counter(r['stratum'] for r in self.groups['rehearsal']);qset=set(load(ROOT/'build/quality-data.json')['train_state_keys']);support=load(ROOT/'build/sampling-review.json')['observed_accepted_run'];self.assertEqual(support['unique_roots_drawn'],len(qset));self.assertEqual(len(qset),7669)
        for pool,rr in self.groups.items():
            states=Counter();families=Counter()
            for row in rr:
                saved=ex['ledger'][row['id']];n=self.ledger.get(row['id'],0);self.assertEqual(saved['actual_draws'],n);p=.25/24 if pool!='rehearsal' else .5*(.9/len(rr)+.1/(len(strata)*strata[row['stratum']]));self.close(saved['overall_sampling_probability'],p);self.close(saved['expected_draws'],12800*p);states[row['state_key']]+=n;families[row['family']]+=n
                if pool!='rehearsal':rootgroups[row['state_key']].append(row);allstate.add(raw(row['state']))
                else:self.assertIn(row['source_id'],qset)
            summary=ex['by_group'][pool];self.assertEqual(summary['draws'],sum(states.values()));self.assertEqual(summary['distinct_records_drawn'],sum(self.ledger.get(r['id'],0)>0 for r in rr));self.assertEqual(summary['distinct_states_drawn'],sum(n>0 for n in states.values()));self.close(summary['state_effective_support'],sum(states.values())**2/sum(n*n for n in states.values()))
        pairs={r['state_key']:r for r in v['ordinary_training_pairs']};gaps=[]
        for key,rr in rootgroups.items():
            saved=pairs[key];ys=[r['target_player0'] for r in rr];ns=[self.ledger[r['id']] for r in rr];gaps.append(abs(ys[1]-ys[0]));self.assertEqual(saved['record_ids'],[r['id'] for r in rr]);self.assertEqual(saved['terminal_returns_player0'],ys);self.assertEqual(saved['actual_record_draws'],ns);self.close(saved['draw_weighted_return_mean_player0'],sum(y*n for y,n in zip(ys,ns))/sum(ns));self.assertEqual(saved['absolute_pair_gap_seeds'],abs(ys[1]-ys[0]))
            for name,model in (('control',self.control),('candidate',self.candidate)):self.close(saved[name+'_prediction_player0'],dense(rr[0]['state'],model)['value'])
        self.assertEqual((len(pairs),sum(g>0 for g in gaps),statistics.median(gaps),max(gaps)),(24,22,9.,56))
        def relativepits(p):return p['pits'][p['controller']]+p['pits'][1-p['controller']]
        for key,query in v['validation_roots'].items():
            qp=relativepits(query['state']);ordered=sorted(self.groups['rehearsal'],key=lambda r:(sum(abs(a-b) for a,b in zip(qp,relativepits(r['state']))),r['id']))[:5];saved=v['coverage'][key];self.assertEqual([r['id'] for r in ordered],[r['row_id'] for r in saved['neighbors']]);self.assertEqual(saved['candidate_pool_records'],7539)
            for row,n in zip(ordered,saved['neighbors']):
                self.assertNotEqual(row['family'],query['family']);neighborstates.add(raw(row['state']));allstate.add(raw(row['state']));self.assertEqual(n['distance_seeds'],sum(abs(a-b) for a,b in zip(qp,relativepits(row['state']))));self.assertEqual(n['actual_draws'],self.ledger.get(row['id'],0));self.assertEqual(n['source_row_sha256'],row['source_row_sha256'])
                for name,model in (('control',self.control),('candidate',self.candidate)):self.close(n[name+'_prediction_player0'],dense(row['state'],model)['value'])
            allstate.add(raw(query['state']))
        self.assertEqual((len(allstate),len(neighborstates)),(60,30));self.assertEqual(v['inference_budget']['unique_fullstates_with_literal_trunk_forwards'],len(allstate));self.assertFalse(ex['ancestral_quality_draw_exposure']['nonzero_gradient_or_causal_effect_proven']);self.assertEqual(ex['ancestral_quality_draw_exposure']['current_zero_draw_rehearsal_rows_with_positive_ancestral_draws'],sum(not self.ledger.get(r['id'],0) for r in self.groups['rehearsal']))
        followup=load(ROOT/'build/ordinary-value-followup.json');roles=followup['identity']['roles'];traces=roles['latest96_trajectories'];self.assertEqual(len(traces),96);protectedq=set(roles['true_evaluation_quotients']);protectedf=set(roles['true_evaluation_families'])
        gamebodies=rows(JOURNAL,'games','idx')
        for trace in traces:
            g=gamebodies[trace['index']];self.assertEqual(trace['game_sha256'],config.digest(g));self.assertIn(g['family'],protectedf);p=norm(g['start']);expected=[]
            for action in g['moves']:expected.append(dict(state_key=raw(position(p)),quotient=quotient(position(p))));p=literal.apply(p,action).position
            expected.append(dict(state_key=raw(position(p)),quotient=quotient(position(p))));self.assertEqual(trace['states'],expected)
            for item in expected:self.assertIn(item['quotient'],protectedq)
        c=self.conclusions;facts=c['facts'];self.assertEqual(facts['H_v1_control_wdl'],self.loss['summary']['H_v1']['control_wdl']);self.assertEqual(facts['H_v1_candidate_wdl'],self.loss['summary']['H_v1']['candidate_wdl']);self.close(facts['H10_candidate_mae'],v['summary']['H10_candidate_mae']);self.close(facts['H10_control_mae'],v['summary']['H10_control_mae'])
        self.assertEqual(c['decision']['current_allocation'],'hold');self.assertFalse(c['decision']['further_training_authorized']);self.assertFalse(c['decision']['new_data_collection_authorized']);self.assertFalse(c['decision']['candidate_promoted']);self.assertFalse(c['interpretation']['value_head_isolated_as_game_loss_cause']);self.assertFalse(c['interpretation']['H12_is_exact_truth']);self.assertFalse(c['interpretation']['zero_current_draws_implies_no_ancestry'])
        for action in c['prospective_next_gates']:self.assertFalse(action['executed'])
        self.assertEqual(self.overlap['hypothetical_rule_change']['result']['branch'],'repeat_high');self.assertFalse(self.overlap['hypothetical_rule_change']['executed'])
        EVIDENCE['coverage_and_inference']=dict(independent_draw_schedule=[3200,3200,6400],individual_noisy_return_pairs=24,pairs_disagree=22,median_pair_gap=9,max_pair_gap=56,nearest_neighbors=30,nearest_selection_uses_targets=False,literal_forward_union=60,all96_future_protection_traces_verified=True,prior_positive_draw_exposure_for_all7539_rehearsal_roots=True,hypothetical_gate_not_executed=True,original_hold_preserved=True,no_new_training_or_collection_authorized=True)

    def test_06_rehashed_metrics_source_pins_and_immutable_publication_refused(self):
        probes=[]
        for item,original in zip(MODULES,self.reports):
            module=importlib.import_module(item['module']);metric=copy.deepcopy(original)
            if isinstance(metric.get('summary'),dict):
                key=next((k for k,v in metric['summary'].items() if isinstance(v,int) and not isinstance(v,bool)),None)
                if key:metric['summary'][key]+=1
                else:metric['summary']['injected_false_metric']=1
            else:metric['facts']['H10_candidate_mae']+=1
            if 'sha256' in metric:metric['sha256']=config.digest({k:v for k,v in metric.items() if k!='sha256'})
            with self.assertRaises((ValueError,AssertionError)):module.validate_report(metric)
            identity=copy.deepcopy(original);key=next(iter(identity['inputs']));identity['inputs'][key]='0'*64
            if 'sha256' in identity:identity['sha256']=config.digest({k:v for k,v in identity.items() if k!='sha256'})
            with self.assertRaises((ValueError,AssertionError)):module.validate_report(identity)
            target=self.base/(item['module'].split('.')[-1]+'-bad-existing.json');save(target,metric);before=signature(target)
            with self.assertRaises((ValueError,AssertionError)):module.run(target)
            self.assertEqual(signature(target),before);probes.append(dict(module=item['module'],rehashed_metric_refused=True,rehashed_source_identity_refused=True,immutable_wrong_existing_output_refused_without_mutation=True))
        EVIDENCE['corruption_refusal']=probes

    def test_07_completed_API_CLI_missing_output_rebuild_no_work_and_immutability(self):
        p=self.fresh('closed-no-work');spec=dict(reports=[v['report'] for v in MODULES],modules=MODULES,fixture_directory=str(self.fresh('missing-output-rebuild')),output=str(p/'proof.json'));save(p/'spec.json',spec)
        run_child([sys.executable,str(ROOT/'tools/ordinary-value-review-audit.py'),str(p/'spec.json')],p/'worker.log',timeout=1800)
        proof=load(p/'proof.json');self.assertTrue(proof['passed']);self.assertEqual(len(proof['modules']),4);self.assertTrue(proof['all_bytes_sizes_mtimes_unchanged']);self.assertTrue(proof['old3779_inputs_current']);self.assertFalse(proof['torch_imported']);self.assertEqual(proof['new_teacher_PUCT_searches'],0);self.assertEqual(proof['new_training_steps'],0);self.assertEqual(proof['new_strength_or_selfplay_games'],0);self.assertEqual(proof['old_producer_publications'],0);self.current(load(BASELINE)['inputs']);self.assertNotIn('torch',sys.modules)
        EVIDENCE['completed_no_work']=dict(path=relative(p/'proof.json'),**proof)


def load_game_family(db,index):return json.loads(db.execute('SELECT body FROM games WHERE idx=?',(index,)).fetchone()[0])['family']


METHODS=[name for name in OrdinaryValueReviewIndependentAudit.__dict__ if name.startswith('test_')]


def science_inputs():
    names={relative(Path(__file__)), 'tools/ordinary-value-review-audit.py',relative(BASELINE),'tests/data_pilot_independent_audit.py','tests/reference.py','tests/stage5_deeper_audit_acceptance.py','tests/stage5_audit_acceptance.py'}
    for item in MODULES:
        body=load(ROOT/item['report']);names.add(item['report']);names.update(body['inputs']);names.add(item['cli'])
    return {name:sha(ROOT/name) for name in sorted(names)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['freeze','run']);args=parser.parse_args()
    if args.operation=='freeze':
        assert not FREEZE.exists(),'Independent source already frozen';save(FREEZE,dict(schema='ordinary-value-review-independent-executed-source-v1',methods=METHODS,inputs=science_inputs()));return 0
    frozen=load(FREEZE);assert frozen['methods']==METHODS
    for name,digest in frozen['inputs'].items():assert sha(ROOT/name)==digest,'Frozen audit input changed: '+name
    assert not OUTPUT.exists(),'Immutable independent report already exists'
    started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(OrdinaryValueReviewIndependentAudit(name) for name in METHODS))
    files=[p for p in OrdinaryValueReviewIndependentAudit.base.rglob('*') if p.is_file()];inputs={**frozen['inputs'],relative(FREEZE):sha(FREEZE),**pins(files)}
    body=dict(schema='ordinary-value-review-independent-audit-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),covered_methods=METHODS,covered_checks=[OrdinaryValueReviewIndependentAudit(name).id() for name in METHODS],duration_seconds=time.monotonic()-started,retained_directory=relative(OrdinaryValueReviewIndependentAudit.base),inputs=inputs,evidence=EVIDENCE,failure_details=[dict(method=str(test),trace=trace) for test,trace in result.errors+result.failures]);save(OUTPUT,body)
    print(config.canonical(dict(event='ordinary_value_review_independent_audit',passed=body['passed'],tests=result.testsRun)),flush=True);return 0 if body['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
