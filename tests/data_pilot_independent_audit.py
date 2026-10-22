"""Independent ordinary-data ownership, label and live-recovery audit."""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import closing
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
import unittest
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config
from congklak.native import Native,flatten
import reference as literal
from quality_audit_acceptance import kill_barrier,run_child,snapshot

WORK=ROOT/'build/data-pilot-audit'
EVIDENCE={}
MODEL=ROOT/'data/quality-train-v1/model.json'
MODEL_SHA='bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
SOURCE_PATHS=['tests/data_pilot_independent_audit.py','congklak/data_pilot.py','tools/data-pilot.py','tools/review-sampling.py','tools/analyze-data-pilot.py']

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def pins(paths):return {relative(p):sha(p) for p in paths}
def dbopen(path):return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
def norm(p):
    q=literal.Position(7,tuple(flatten(p)),p['controller']);n=literal.normalize(q)
    return literal.Position(7,q.counts,n.controller)
def position(q):return dict(pits=[list(q.counts[:7]),list(q.counts[8:15])],stores=[q.counts[7],q.counts[15]],controller=q.controller)
def key_of(p):
    q=norm(p);c=list(q.counts);a=0 if literal.normalize(q).terminal else q.controller
    return min(bytes(c+[a]),bytes(c[8:]+c[:8]+[0 if literal.normalize(q).terminal else 1-a])).hex()
def quotient(p):
    q=norm(p);c=list(q.counts[:7]+q.counts[8:15]);a=q.controller if any(c) else 0
    return min(bytes(c+[a]),bytes(c[7:]+c[:7]+[1-a if any(c) else 0])).hex()
def from_key(key):
    c=bytes.fromhex(key) if isinstance(key,str) else bytes(key)
    return dict(pits=[list(c[:7]),list(c[8:15])],stores=[c[7],c[15]],controller=c[16])
def mechanism(p):
    q=norm(p);n=literal.normalize(q);active=sum(q.counts[:7]+q.counts[8:15])
    band='0-4' if active<=4 else '5-24' if active<=24 else '25-49' if active<=49 else '50-74' if active<=74 else '75-98'
    if n.terminal:return active,band+'/terminal',0
    steps=[literal.apply(q,a) for a in literal.legal_actions(q)]
    kind='pass' if any(t.initial_passes or t.final_passes for t in steps) else 'capture' if any(t.captured for t in steps) else 'bonus' if any(t.bonus for t in steps) else 'ordinary'
    return active,band+'/'+kind,sum(t.terminal for t in steps)
def rows(path,table,key):
    with closing(dbopen(path)) as db:
        result={}
        for k,text,digest in db.execute(f'SELECT {key},body,hash FROM {table} ORDER BY {key}'):
            body=json.loads(text)
            if config.digest(body)!=digest:raise AssertionError('Journal checksum differs '+table)
            result[k]=body
        return result
def semantic(value):
    if isinstance(value,dict):return {k:semantic(v) for k,v in value.items() if k not in {'elapsed_ms','peak_process_bytes','elapsed_seconds','wall_seconds','build_seconds','created_at','updated_at','work','label_sha256'}}
    if isinstance(value,list):return [semantic(v) for v in value]
    return value

def protected_reference():
    inherited=load(ROOT/'build/value-registry.json');protected=set(inherited['quotients'])
    with closing(dbopen(ROOT/'data/deeper-9000-v2/work.sqlite3')) as db:oldkeys={r[0] if isinstance(r[0],str) else r[0].hex() for r in db.execute('SELECT state FROM roots')}
    assert len(oldkeys)==9000
    protected.update(quotient(from_key(k)) for k in oldkeys)
    observations=[];games=choices=0
    with closing(dbopen(ROOT/'data/value-benchmark-v1/work.sqlite3')) as db:
        for idx,text in db.execute('SELECT idx,body FROM games ORDER BY idx'):
            g=json.loads(text);q=norm(g['start'])
            for step,(move,row) in enumerate(zip(g['moves'],g['decisions'])):
                assert move in literal.legal_actions(q) and row['actor']==q.controller
                p=position(q);protected.add(quotient(p));observations.append((key_of(p),idx,g['family'],row,step))
                q=literal.apply(q,move).position;choices+=1
            assert len(g['moves'])==len(g['decisions']) and position(q)==g['final']
            protected.add(quotient(position(q)));games+=1
    assert games==192
    return protected,oldkeys,observations,dict(games=games,literal_choices=choices,old_root_keys=len(oldkeys),protected_quotients=len(protected))

def fresh_candidate(p,family,cohort,provenance):
    k=key_of(p);s=from_key(k);active,stratum,terminal=mechanism(s)
    if active<25 or not stratum.endswith('/ordinary'):return None
    q=norm(p);swapped=bytes(q.counts+(q.controller,)).hex()!=k
    return dict(state_key=k,state=s,quotient=quotient(s),family=family,cohort=cohort,
        purpose='protected_diagnostic' if cohort=='diagnostic' else 'fresh_data_pilot',band=stratum.split('/')[0],
        stratum=stratum,active=active,legal_mask=sum(1<<a for a in literal.legal_actions(norm(s))),terminal_children=terminal,
        provenance={**provenance,'raw_state':p,'swapped':swapped})

def literal_family(index,settings,evaluator):
    family=f"ordinary-history-pilot/{settings['seed']}/{index}"
    rank=int.from_bytes(hashlib.sha256(('data-pilot-owner-v1/'+family).encode()).digest()[:8],'big')%100
    cohort='training_candidate' if rank<80 else 'validation';source='random_history' if index%2==0 else 'frozen_policy_epsilon_history'
    rng=random.Random(settings['seed']+104729*index);q=norm(config.position());moves=[];candidates=[];forwards=0
    for step in range(settings['max_choices']):
        p=position(q)
        if literal.normalize(q).terminal or sum(p['pits'][0]+p['pits'][1])<25:break
        provenance=dict(source=source,family_index=index,step=step,start=config.position(),moves_prefix=list(moves),source_game_idx=None,source_comparison=None,saved_puct_chosen=None)
        item=fresh_candidate(p,family,cohort,provenance)
        if item:candidates.append(item)
        legal=list(literal.legal_actions(q))
        if index%2==0 or step<settings['random_opening_choices'] or rng.random()<settings['epsilon']:action=rng.choice(legal)
        else:
            logits=evaluator.evaluate_many([p])[0]['policy_logits'];forwards+=1;action=min(legal,key=lambda a:(-logits[a],a))
        moves.append(action);q=literal.apply(q,action).position
    p=position(q);active=sum(p['pits'][0]+p['pits'][1])
    return dict(index=index,family=family,cohort=cohort,source=source,rng_seed=settings['seed']+104729*index,start=config.position(),moves=moves,final=p,choices=len(moves),candidates=candidates,policy_evaluations=forwards,
        stop_reason='terminal' if not active else 'below25' if active<25 else 'choice_cap')

def expected_selection(families,settings,registry):
    candidates=[r for f in families.values() for r in f['candidates']];owners=defaultdict(set)
    for r in candidates:owners[r['quotient']].add(r['cohort'])
    collided={q for q,v in owners.items() if len(v)>1};protected=set(registry['quotients']);oldfamilies=set(registry['families'])
    available=sorted((r for r in candidates if r['quotient'] not in protected and r['quotient'] not in collided and r['family'] not in oldfamilies),key=lambda r:(r['provenance']['family_index'],r['provenance']['step'],r['state_key']))
    selected={};counts=Counter();seenq=set();seenfb=set()
    for r in available:
        c,b=r['cohort'],r['band'];fb=(r['family'],b)
        if counts[c,b]<settings['caps'][c][b] and fb not in seenfb and r['quotient'] not in seenq:
            selected[r['state_key']]=r;counts[c,b]+=1;seenfb.add(fb);seenq.add(r['quotient'])
    for r in registry['diagnostic_pool'][:settings['diagnostic_limit']]:selected[r['state_key']]=r
    return dict(sorted(selected.items())),dict(ordinary_observations=len(candidates),ordinary_unique_quotients=len(owners),cross_cohort_candidate_collisions=len(collided),protected_candidate_observations=sum(r['quotient'] in protected for r in candidates),
        counts={c:{b:counts[c,b] for b in settings['caps'][c]} for c in settings['caps']},shortfall={c:{b:settings['caps'][c][b]-counts[c,b] for b in settings['caps'][c]} for c in settings['caps']},
        diagnostic_rows=min(settings['diagnostic_limit'],len(registry['diagnostic_pool'])),observed_legal_action_counts=dict(sorted(Counter(str(r['legal_mask'].bit_count()) for r in candidates).items())),selected_legal_action_counts=dict(sorted(Counter(str(r['legal_mask'].bit_count()) for r in selected.values()).items())),diagnostic_shortfall=max(0,settings['diagnostic_limit']-len(registry['diagnostic_pool'])))

class DataPilotIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True,exist_ok=True);cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()
    def fresh(self,name):
        p=self.base/name;p.mkdir();return p
    def command(self,stage,path,families=4,diagnostics=2):
        return [sys.executable,str(ROOT/'tools/data-pilot.py'),stage,'--run',str(path),'--fixture','--families',str(families),'--diagnostic-limit',str(diagnostics)]
    def backup(self,run,path):
        with closing(dbopen(run/'work.sqlite3')) as source,closing(sqlite3.connect(path)) as target:source.backup(target)
    def test_01_frozen_baseline_and_probability_formulas(self):
        baseline=load(ROOT/'build/data-pilot-baseline.json')
        self.assertEqual(len(baseline['inputs']),1834);self.assertEqual(sha(MODEL),MODEL_SHA)
        for path,digest in baseline['inputs'].items():self.assertEqual(sha(ROOT/path),digest,path)
        EVIDENCE['baseline']=dict(prior_inputs=1834,all_hashes_current=True,fixed_model_sha256=MODEL_SHA)
        review=load(ROOT/'build/sampling-review.json');admitted=load(ROOT/'build/quality-data.json')
        with closing(dbopen(ROOT/'data/deeper-9000-v2/work.sqlite3')) as db:
            metadata={k:(family,stratum) for k,family,stratum in db.execute("SELECT state,json_extract(body,'$.family'),json_extract(body,'$.stratum') FROM roots")}
        keys=admitted['train_state_keys'];data=[metadata[k] for k in keys];counts=Counter(s for f,s in data);n=len(data);ns=len(counts);den=math.fsum(math.sqrt(v) for v in counts.values())
        self.assertEqual((n,ns,len({f for f,s in data})),(7669,14,4576))
        support={}
        for name in ('uniform_root','uniform_stratum','sqrt_count','root90_stratum10'):
            probabilities=[1/n if name=='uniform_root' else 1/(ns*counts[s]) if name=='uniform_stratum' else 1/(math.sqrt(counts[s])*den) if name=='sqrt_count' else .9/n+.1/(ns*counts[s]) for f,s in data]
            self.assertAlmostEqual(math.fsum(probabilities),1.,places=12);self.assertTrue(all(v>0 for v in probabilities))
            family=defaultdict(list);strata=defaultdict(list)
            for (f,s),probability in zip(data,probabilities):family[f].append(probability);strata[s].append(probability)
            fp={f:math.fsum(v) for f,v in family.items()};r=review['distributions'][name]
            self.assertAlmostEqual(r['root_effective_support'],1/math.fsum(v*v for v in probabilities),places=9)
            self.assertAlmostEqual(r['family_effective_support'],1/math.fsum(v*v for v in fp.values()),places=9)
            for s,values in strata.items():
                row=r['strata'][s];self.assertAlmostEqual(row['probability'],math.fsum(values),places=12);self.assertAlmostEqual(row['per_root_probability'],values[0],places=14)
                self.assertAlmostEqual(row['expected_draws'],384000*math.fsum(values),places=7);self.assertEqual(row['roots'],counts[s])
            for f in r['top_families']:self.assertAlmostEqual(f['probability'],fp[f['family']],places=12)
            support[name]=dict(root=r['root_effective_support'],family=r['family_effective_support'])
        groups=defaultdict(list)
        for index,(f,s) in enumerate(data):groups[s].append(index)
        rng=random.Random(20261015);strata=sorted(groups);draws=[0]*n
        for _ in range(384000):draws[rng.choice(groups[rng.choice(strata)])]+=1
        observed=review['observed_accepted_run'];ledger=[[k,draws[i]] for i,k in enumerate(keys)]
        self.assertEqual(observed['root_draw_ledger_sha256'],config.digest(ledger));self.assertEqual(sum(draws[i] for i,(f,s) in enumerate(data) if counts[s]<=5),54873)
        self.assertEqual(observed['total_draws'],384000);self.assertFalse(review['training_performed']);self.assertFalse(review['model_changed'])
        self.assertFalse(review['recommendation']['sampling_weights_applied']);self.assertTrue(review['recommendation']['no_rows_pruned_or_deleted'])
        EVIDENCE['sampling_review']=dict(distributions=4,probability_mass_conserved=True,positive_support=True,root_and_family_concentration_verified=True,original_draws_literal_rng_replayed=384000,tiny_nine_root_draws=54873,effective_support=support,training_weights_applied=False)

    def test_02_literal_protected_registry_and_diagnostic_novelty(self):
        from congklak import data_pilot as p
        registry=load(p.REGISTRY);protected,oldkeys,observations,evidence=protected_reference()
        self.assertEqual(set(registry['quotients']),protected);self.assertEqual(set(registry['existing9000_state_keys']),oldkeys)
        oldq={quotient(from_key(k)) for k in oldkeys};expected={}
        with closing(dbopen(ROOT/'data/value-benchmark-v1/work.sqlite3')) as db:
            games={idx:json.loads(text) for idx,text in db.execute('SELECT idx,body FROM games')}
        for key,index,family,row,step in observations:
            g=games[index]
            if g['comparison'] not in ('policy_control_vs_H_v1','policy_control_vs_H8_relay_CPU') or row['engine']!='policy_control':continue
            q=norm(g['start'])
            for action in g['moves'][:step]:q=literal.apply(q,action).position
            root=position(q);provenance=dict(source='protected_control_arena',family_index=None,step=step,start=g['start'],moves_prefix=g['moves'][:step],source_game_idx=index,source_comparison=g['comparison'],saved_puct_chosen=g['moves'][step])
            item=fresh_candidate(root,family,'diagnostic',provenance)
            if item and item['state_key'] not in oldkeys and item['quotient'] not in oldq:expected.setdefault(item['state_key'],item)
        unique=[];seen=set()
        for key in sorted(expected):
            r=expected[key]
            if r['quotient'] not in seen:unique.append(r);seen.add(r['quotient'])
        self.assertEqual(registry['diagnostic_pool'],unique);self.assertEqual(registry['protected_value_games'],192)
        self.assertFalse(registry['sealed_target_bodies_read']);self.assertEqual(load(p.PLAN)['diagnostic_roots'],unique[:16])
        EVIDENCE['protected_registry']={**evidence,'diagnostic_pool':len(unique),'independently_literal_rebuilt':True,'old_sealed_target_bodies_read':False}

    def test_03_contract_ordinary_classifier_history_and_ownership(self):
        from congklak import data_pilot as p
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        s=p.CONTRACT
        self.assertEqual((s['families'],s['seed'],s['max_choices'],s['random_opening_choices'],s['epsilon']),(256,20261020,192,8,.25))
        self.assertEqual(s['caps'],dict(training_candidate={'25-49':32,'50-74':32,'75-98':8},validation={'25-49':8,'50-74':8,'75-98':2}))
        self.assertEqual((s['diagnostic_limit'],s['choice_depth'],s['node_budget'],s['time_ms'],s['tt_mb'],s['memory_mb'],s['use_pvs']),(16,8,2000000,0,16,512,True))
        self.assertFalse(s['training_performed']);self.assertFalse(s['sealed_final_test_consumed']);p.admit()
        generated={};native=Native()
        with CpuBatchEvaluator(MODEL,cache_capacity=0,max_batch=1) as evaluator:
            for index in range(8):
                expected=literal_family(index,s,evaluator);actual=p.generate_family(index,s,native,evaluator)
                self.assertEqual(actual,expected);generated[index]=expected
                for row in expected['candidates']:self.assertEqual(p.candidate(row['provenance']['raw_state'],native,row['family'],row['cohort'],{k:v for k,v in row['provenance'].items() if k not in {'raw_state','swapped'}}),row)
        expected,detail=expected_selection(generated,s,load(p.REGISTRY));self.assertEqual(p.select(generated,s,load(p.REGISTRY)),(expected,detail))
        # The same candidate quotient observed under opposite ownership must vanish on both sides.
        candidates=[r for f in generated.values() for r in f['candidates']]
        self.assertTrue(candidates)
        a=copy.deepcopy(candidates[0]);b=copy.deepcopy(a);b['cohort']='validation' if a['cohort']=='training_candidate' else 'training_candidate';b['family']='other-new-family';b['provenance']['family_index']=999
        selected,detail=p.select({0:{'candidates':[a]},1:{'candidates':[b]}},s,{'quotients':[],'families':[],'diagnostic_pool':[]})
        self.assertFalse(selected);self.assertEqual(detail['cross_cohort_candidate_collisions'],1)
        EVIDENCE['contract']=dict(full_pool_before_selection=True,strict_ordinary_literal_verified=True,fixture_families=8,ownership_before_sampling=True,cross_cohort_collision_excluded_both=True,maximum_attempts=106)

    def test_04_actual_collection_pre_postcommit_kills_and_stability(self):
        parent=self.fresh('collection');fresh=parent/'fresh'
        run_child(self.command('collect',fresh),parent/'fresh.log')
        expected=rows(fresh/'work.sqlite3','families','idx');expected_roots=rows(fresh/'work.sqlite3','roots','state')
        evidence=[]
        for event,prefix in (('data_pilot_family_precommit',1),('data_pilot_family_committed',2)):
            run=parent/event;probe=parent/(event+'-probe')
            marker=kill_barrier(self.command('collect',run),probe,event,env={'DATA_PILOT_BLOCK_EVENT':event,'DATA_PILOT_BLOCK_STEP':'1','DATA_PILOT_MARKER':str(probe/'declared-marker.json')},condition=lambda v:v['index']==1)
            before=rows(run/'work.sqlite3','families','idx');self.assertEqual(list(before),list(range(prefix)));self.backup(run,probe/'validated-prefix.sqlite3')
            run_child(self.command('collect',run),parent/(event+'-resume.log'))
            after=rows(run/'work.sqlite3','families','idx');self.assertEqual(after,expected);self.assertEqual({k:after[k] for k in before},before)
            self.assertEqual(rows(run/'work.sqlite3','roots','state'),expected_roots)
            stable=snapshot(run);run_child(self.command('collect',run),parent/(event+'-completed.log'));self.assertEqual(snapshot(run),stable)
            evidence.append(dict(kill_event=event,committed_prefix=prefix,families_after_resume=4,exact_fresh_histories_and_selection=True,prefix_unchanged=True,completed_bytes_mtimes_unchanged=True,marker=marker))
        EVIDENCE['live_collection_recovery']=dict(actual_process_kills=True,tests=evidence,raw_journal_backup_markers_retained=True)

    def test_05_actual_label_pre_postcommit_kills_and_stability(self):
        from congklak import data_pilot as p
        parent=self.fresh('labeling');fresh=parent/'fresh'
        run_child(self.command('all',fresh),parent/'fresh.log');roots,labels,attempts,manifest=p.load_rows(fresh)
        self.assertGreaterEqual(len(attempts),2);self.assertTrue(labels)
        evidence=[]
        for event,step,prefix in (('data_pilot_label_precommit',1,1),('data_pilot_label_committed',2,2)):
            run=parent/event;probe=parent/(event+'-probe')
            marker=kill_barrier(self.command('all',run),probe,event,env={'DATA_PILOT_BLOCK_EVENT':event,'DATA_PILOT_BLOCK_STEP':str(step),'DATA_PILOT_MARKER':str(probe/'declared-marker.json')},condition=lambda v:v['completed']==step)
            before=rows(run/'work.sqlite3','attempts','state');beforelabels=rows(run/'work.sqlite3','labels','state')
            self.assertEqual(list(before),sorted(roots)[:prefix]);self.backup(run,probe/'validated-prefix.sqlite3')
            self.assertFalse((run/'manifest.json').exists())
            run_child(self.command('all',run),parent/(event+'-resume.log'));r,l,a,m=p.load_rows(run)
            self.assertEqual(r,roots);self.assertEqual(semantic(l),semantic(labels));self.assertEqual(semantic(a),semantic(attempts))
            self.assertEqual({k:a[k] for k in before},before);self.assertEqual({k:l[k] for k in beforelabels},beforelabels)
            stable=snapshot(run);run_child(self.command('verify',run),parent/(event+'-completed.log'));self.assertEqual(snapshot(run),stable)
            evidence.append(dict(kill_event=event,committed_prefix=prefix,attempts_after_resume=len(a),labels_after_resume=len(l),fresh_complete_H8_semantics_and_counters_equal=True,prefix_exactly_retained=True,completed_bytes_mtimes_unchanged=True,marker=marker))
        EVIDENCE['live_label_recovery']=dict(actual_process_kills=True,tests=evidence,raw_journal_backup_markers_retained=True)

    def test_06_identity_publication_partial_and_work_corruption_refusals(self):
        from congklak import data_pilot as p
        parent=self.fresh('refusals');fresh=parent/'fresh';run_child(self.command('all',fresh),parent/'fresh.log')
        roots,labels,attempts,manifest=p.load_rows(fresh);native=Native();key=next(iter(labels));row=roots[key];item=labels[key]
        partial=native.analyze(flatten(row['state']),actor=row['state']['controller'],total=98,max_depth=8,time_ms=0,node_budget=1,tt_mb=16,use_pvs=True,memory_mb=512,keep_history=False)
        self.assertEqual(partial['stop_reason'],'node_budget')
        report=partial.get('report');depth=report['choice_depth'] if report else 0
        rejection=dict(root_sha256=config.digest(row),status='budget_rejected',stop_reason='node_budget',completed_depth=depth,label_sha256=None,work={k:partial[k] for k in ('nodes','deposits','elapsed_ms','peak_process_bytes')},partial_report=report)
        settings={**p.CONTRACT,'node_budget':1};p.validate_attempts({key:row},{key:rejection},{},settings,native)
        forged=copy.deepcopy(rejection);forged['completed_depth']=8
        with self.assertRaises(ValueError):p.validate_attempts({key:row},{key:forged},{},settings,native)
        with self.assertRaises(ValueError):p.validate_attempts({key:row},{key:rejection},{key:item},settings,native)
        forgeditem=copy.deepcopy(item);forgeditem['report']['completed_iteration']=False
        with self.assertRaises((ValueError,p.teacher.DeeperError)):p.validate_label_item(forgeditem,row,native)
        forgeditem=copy.deepcopy(item);forgeditem['label']['target_player0']+=1
        with self.assertRaises((ValueError,p.teacher.DeeperError)):p.validate_label_item(forgeditem,row,native)
        complete=copy.deepcopy(attempts[key]);complete['work']={k:0 for k in complete['work']}
        with self.assertRaises(ValueError):p.validate_attempts({key:row},{key:complete},{key:item},p.CONTRACT,native)
        for kind in ('identity','publication','family','attempt_prefix'):
            run=parent/kind;shutil.copytree(fresh,run)
            with closing(sqlite3.connect(run/'work.sqlite3')) as db:
                if kind in ('identity','publication'):
                    name='identity' if kind=='identity' else 'complete';value=json.loads(db.execute('SELECT body FROM meta WHERE key=?',(name,)).fetchone()[0])
                    if kind=='identity':value['model_file_sha256']='0'*64
                    else:value['labels']+=1
                    db.execute('UPDATE meta SET body=?,hash=? WHERE key=?',(config.canonical(value),config.digest(value),name))
                elif kind=='family':
                    value=json.loads(db.execute('SELECT body FROM families WHERE idx=0').fetchone()[0]);value['moves'][0]=(value['moves'][0]+1)%7
                    db.execute('UPDATE families SET body=?,hash=? WHERE idx=0',(config.canonical(value),config.digest(value)))
                else:db.execute('DELETE FROM attempts WHERE state=?',(sorted(attempts)[0],))
                db.commit()
            with self.assertRaises((ValueError,KeyError,AssertionError)):p.load_rows(run)
            self.assertNotEqual(run_child(self.command('verify',run),parent/(kind+'.log'),success=False),0)
        before=snapshot(fresh);export=(fresh/'manifest.json').read_bytes();(fresh/'manifest.json').unlink()
        with self.assertRaises(FileNotFoundError):p.load_rows(fresh)
        run_child(self.command('all',fresh),parent/'repair.log');self.assertEqual((fresh/'manifest.json').read_bytes(),export)
        after=snapshot(fresh);self.assertEqual({k:v for k,v in before.items() if k!='manifest.json'},{k:v for k,v in after.items() if k!='manifest.json'})
        EVIDENCE['refusals']=dict(real_node_budget_partial_generated=True,partial_H8_not_labeled=True,depth8_budget_rejection_refused=True,partial_as_label_refused=True,incomplete_report_refused=True,report_label_target_mismatch_refused=True,zero_work_complete_attempt_refused=True,rehashed_identity_publication_family_and_attempt_prefix_refused=True,missing_export_requires_repair=True,repair_preserves_journal_bytes_mtime=True)

    def test_07_exhaustive_production_histories_ownership_and_selection(self):
        from congklak import data_pilot as p
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        roots,labels,attempts,manifest=p.load_rows(p.DEFAULT);saved=rows(p.DEFAULT/'work.sqlite3','families','idx');self.assertEqual(list(saved),list(range(256)))
        expected={}
        with CpuBatchEvaluator(MODEL,cache_capacity=0,max_batch=1) as evaluator:
            for index in range(256):
                expected[index]=literal_family(index,p.CONTRACT,evaluator);self.assertEqual(saved[index],expected[index],f'Family {index}')
        registry=load(p.REGISTRY);selected,detail=expected_selection(expected,p.CONTRACT,registry)
        self.assertEqual(roots,selected);self.assertEqual(manifest['selection'],detail);self.assertEqual(manifest,load(p.DEFAULT/'manifest.json'));self.assertEqual(manifest,load(p.OUTPUT))
        protected=set(registry['quotients']);cohort_q=defaultdict(set);familybands=set()
        for k,r in roots.items():
            self.assertEqual(k,key_of(r['state']));self.assertEqual(mechanism(r['state'])[:2],(r['active'],r['stratum']))
            self.assertGreaterEqual(r['active'],25);self.assertTrue(r['stratum'].endswith('/ordinary'))
            if r['cohort']=='diagnostic':
                self.assertIn(r['quotient'],protected);self.assertEqual(r['purpose'],'protected_diagnostic')
            else:
                self.assertNotIn(r['quotient'],protected);fb=(r['family'],r['band']);self.assertNotIn(fb,familybands);familybands.add(fb);cohort_q[r['cohort']].add(r['quotient'])
        self.assertFalse(cohort_q['training_candidate']&cohort_q['validation']);self.assertLessEqual(len(roots),106)
        self.assertEqual(set(attempts),set(roots));self.assertEqual(manifest['labels'],len(labels));self.assertEqual(manifest['budget_rejected'],len(attempts)-len(labels))
        baseline=load(ROOT/'build/data-pilot-baseline.json')
        for path,digest in baseline['inputs'].items():self.assertEqual(sha(ROOT/path),digest,path)
        EVIDENCE['production_collection']=dict(families=256,source_families=dict(Counter(f['source'] for f in expected.values())),cohort_families=dict(Counter(f['cohort'] for f in expected.values())),literal_choices=sum(f['choices'] for f in expected.values()),policy_evaluations=sum(f['policy_evaluations'] for f in expected.values()),candidate_observations=detail['ordinary_observations'],selected_roots=len(roots),selection=detail,all_histories_replayed=True,all_rng_and_policy_choices_rederived=True,full_pool_collision_closure_rederived=True,selected_terminal_children=sum(r['terminal_children'] for r in roots.values()),prior_inputs_unchanged_after_collection=1834)

    def test_08_complete_H8_labels_metrics_and_completed_publication(self):
        from congklak import data_pilot as p,reports
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        from stage5_deeper_audit_acceptance import value_and_policy
        roots,labels,attempts,manifest=p.load_rows(p.DEFAULT);native=Native();pvs=choices=0
        for k,item in labels.items():
            r=roots[k];p.validate_label_item(item,r,native);report=item['report'];label=item['label'];self.assertEqual(report['choice_depth'],8)
            legal=list(literal.legal_actions(norm(r['state'])));scores={c['pit']:c['value_player0'] for c in label['action_scores']};self.assertEqual(list(scores),legal)
            sign=1 if r['state']['controller']==0 else -1;best=max(sign*v for v in scores.values());self.assertEqual(sign*label['target_player0'],best);self.assertEqual(label['best_moves'],[a for a in legal if sign*scores[a]==best])
            self.assertEqual(label['kind'],'finite_choice_horizon');self.assertIsNone(label['oracle_manifest_sha256'])
            for path,value in [(report['pv'],label['target_player0'])]+[(c['pv'],c['depth_result']['value_player0']) for c in report['candidates']]:
                q=norm(r['state'])
                for decision in path:
                    self.assertEqual(decision['actor'],q.controller);self.assertIn(decision['pit'],literal.legal_actions(q));t=literal.apply(q,decision['pit']);q=t.position
                    self.assertEqual(decision['terminal_after'],t.terminal);self.assertEqual(decision['bonus_choice'],t.bonus and not t.terminal)
                    self.assertEqual(decision['forced_passes_after'],[dict(zip(('from','to'),v)) for v in t.final_passes]);choices+=1
                self.assertTrue(literal.normalize(q).terminal or len(path)==8);self.assertEqual(literal.evaluate(q),value);pvs+=1
        p.validate_attempts(roots,attempts,labels,p.CONTRACT,native)
        fresh_checked=[]
        for k in sorted(labels):
            if len(fresh_checked)==2:break
            state=roots[k]['state'];result=native.analyze(flatten(state),actor=state['controller'],total=98,max_depth=8,time_ms=0,node_budget=2000000,memory_mb=512,tt_mb=16,use_pvs=True,keep_history=False)
            self.assertEqual(result['stop_reason'],'max_depth');self.assertEqual(semantic(result['report']),semantic(labels[k]['report']));fresh_checked.append(k)
        diagnosis=load(ROOT/'build/data-pilot-model-diagnosis.json');actual={r['state_key']:r for r in diagnosis['metrics']};self.assertEqual(set(actual),set(labels))
        with CpuBatchEvaluator(MODEL,cache_capacity=0,max_batch=1) as evaluator:
            for k in sorted(labels):
                state=roots[k]['state'];value,_,logits=value_and_policy(norm(state),evaluator.model.envelope['model']);nativepred=evaluator.evaluate_many([state])[0]
                self.assertAlmostEqual(value,nativepred['value_player0'],places=10);self.assertEqual(logits,nativepred['policy_logits'])
                a=actual[k];l=labels[k]['label'];sign=1 if state['controller']==0 else -1;q={x['pit']:x['value_player0'] for x in l['action_scores']};chosen=min(q,key=lambda pit:(-logits[pit],pit));saved=roots[k]['provenance']['saved_puct_chosen']
                self.assertEqual(a['legal_action_count'],len(q));self.assertEqual(a['policy_chosen'],chosen);self.assertEqual(a['saved_puct_chosen'],saved)
                self.assertAlmostEqual(a['value_absolute_error'],abs(value-l['target_player0']),places=10);self.assertAlmostEqual(a['signed_actor_error'],sign*(value-l['target_player0']),places=10)
                self.assertEqual(a['policy_teacher_regret'],sign*(l['target_player0']-q[chosen]));self.assertEqual(a['saved_puct_teacher_regret'],None if saved is None else sign*(l['target_player0']-q[saved]));self.assertGreaterEqual(a['policy_teacher_regret'],0)
        def check_summary(summary,values):
            self.assertEqual(summary['rows'],len(values));self.assertEqual(summary['families'],len({v['family'] for v in values}));self.assertEqual(summary['single_legal_action_rows'],sum(v['legal_action_count']==1 for v in values))
            for output,key in (('value_mae_seeds','value_absolute_error'),('signed_actor_error','signed_actor_error'),('policy_teacher_regret','policy_teacher_regret')):
                self.assertEqual(summary[output],statistics.mean(v[key] for v in values) if values else None)
            self.assertEqual(summary['policy_teacher_best_fraction'],statistics.mean(v['policy_teacher_regret']==0 for v in values) if values else None)
            saved=[v for v in values if v['saved_puct_teacher_regret'] is not None];self.assertEqual(summary['saved_puct_rows'],len(saved));self.assertEqual(summary['saved_puct_teacher_disagreements'],sum(v['saved_puct_teacher_regret']>0 for v in saved));self.assertEqual(summary['saved_puct_teacher_regret'],statistics.mean(v['saved_puct_teacher_regret'] for v in saved) if saved else None)
        for cohort in ('diagnostic','training_candidate','validation'):
            values=[r for r in actual.values() if r['cohort']==cohort];check_summary(diagnosis['cohorts'][cohort],values);check_summary(diagnosis['choice_bearing_cohorts'][cohort],[r for r in values if r['legal_action_count']>=2])
        for group,summary in diagnosis['strata'].items():check_summary(summary,[r for r in actual.values() if r['cohort']+'/'+r['stratum']==group])
        for group,summary in diagnosis['by_source'].items():check_summary(summary,[r for r in actual.values() if r['provenance_source']==group])
        for flag in ('training_performed','model_changed','new_search_performed','active_sampling_weights_changed','sealed_final_test_consumed','independent_unseen_release_test','strength_improvement_claimed'):self.assertFalse(diagnosis[flag])
        before=snapshot(p.DEFAULT);report_path=ROOT/'build/data-pilot-model-diagnosis.json';report_snapshot=(sha(report_path),report_path.stat().st_mtime_ns)
        run_child([sys.executable,str(ROOT/'tools/data-pilot.py'),'verify'],self.base/'production-completed.log');self.assertEqual(snapshot(p.DEFAULT),before)
        run_child([sys.executable,str(ROOT/'tools/analyze-data-pilot.py')],self.base/'diagnosis-completed.log');self.assertEqual((sha(report_path),report_path.stat().st_mtime_ns),report_snapshot)
        for path,digest in load(ROOT/'build/data-pilot-baseline.json')['inputs'].items():self.assertEqual(sha(ROOT/path),digest,path)
        EVIDENCE['production_labels']=dict(selected_roots=len(roots),attempts=len(attempts),complete_H8_labels=len(labels),budget_rejected=len(attempts)-len(labels),all_report_label_links_validated=True,literal_pv_paths=pvs,literal_pv_choices=choices,fresh_H8_reports_recomputed=len(fresh_checked),fixed_model_literal_forwards=len(labels),actor_signs_best_ties_and_metric_groups_rederived=True,single_action_and_choice_bearing_groups_distinguished=True,completed_journal_and_diagnosis_bytes_mtimes_unchanged=True,prior_inputs_unchanged_after_labeling=1834)

GROUPS={'preflight':[f'test_0{i}_'+s for i,s in [(1,'frozen_baseline_and_probability_formulas'),(2,'literal_protected_registry_and_diagnostic_novelty'),(3,'contract_ordinary_classifier_history_and_ownership'),(4,'actual_collection_pre_postcommit_kills_and_stability'),(5,'actual_label_pre_postcommit_kills_and_stability'),(6,'identity_publication_partial_and_work_corruption_refusals')]],
    'final':['test_07_exhaustive_production_histories_ownership_and_selection','test_08_complete_H8_labels_metrics_and_completed_publication']}

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('group',choices=list(GROUPS));args=parser.parse_args();started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(DataPilotIndependentAudit(name) for name in GROUPS[args.group]))
    paths=[ROOT/p for p in SOURCE_PATHS if (ROOT/p).is_file()]
    paths += [ROOT/'build'/name for name in ('data-pilot-baseline.json','data-pilot-registry.json','data-pilot-plan.json','data-pilot-implementation.json','sampling-review.json') if (ROOT/'build'/name).is_file()]
    paths += [p for p in DataPilotIndependentAudit.base.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))]
    if args.group=='final':
        paths += [ROOT/'data/ordinary-data-pilot-v1/work.sqlite3',ROOT/'data/ordinary-data-pilot-v1/manifest.json',ROOT/'build/data-pilot-result.json',ROOT/'build/data-pilot-model-diagnosis.json',WORK/'preflight.json']
        paths += [ROOT/p for p in load(WORK/'preflight.json')['inputs']]
    report=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),covered_methods=GROUPS[args.group],evidence=EVIDENCE,inputs=pins(paths),duration_seconds=time.monotonic()-started,retained_directory=relative(DataPilotIndependentAudit.base),failure_details=[dict(method=str(test),trace=trace) for test,trace in result.failures+result.errors])
    save(WORK/(args.group+'.json'),report);print(config.canonical(dict(event='data_pilot_independent_audit',group=args.group,passed=report['passed'],tests=result.testsRun)),flush=True)
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())

