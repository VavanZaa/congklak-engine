"""Four source-frozen synthetic checks; no producer, Native, model or Torch."""
import copy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import qstability_metrics as m

IMPLEMENTATION=ROOT/'build/qstability-metrics-implementation.json'
SOURCES=('congklak/qstability_metrics.py','tests/qstability_metrics_author.py','docs/QSTABILITY_METRICS.md','docs/QSTABILITY_SCOPE.md','congklak/qstability_common.py','build/qstability-baseline.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def item(row,depth,q):
    best=(max if row['actor']==0 else min)(q.values())
    return dict(root_sha256=m.digest(row),label=dict(choice_depth=depth,target_player0=best,action_scores=[dict(pit=a,value_player0=v) for a,v in sorted(q.items())],best_moves=[a for a,v in sorted(q.items()) if v==best]),report=dict(synthetic=True,no_engine_call=True))


def fixture(uniform=False,extreme=False):
    """Toy bands/full states; mechanism tags are metadata, not game assertions."""
    rows={};h8={};h10={};h12={}
    for index,cell in enumerate(m.CELLS):
        stratum,actor=cell.rsplit('/actor',1);actor=int(actor);active=60 if stratum.startswith('50-74') else 40
        for j in (0,1):
            state=dict(pits=[[1+j,active//2-1-j,0,0,0,0,0],[2+index%3,active//2-2-index%3,0,0,0,0,0]],stores=[index*2+j,98-active-index*2-j],controller=actor);key=m.rawkey(state)
            row=dict(state_key=key,state=state,actor=actor,stratum=stratum,cell=cell,family='synthetic-family/'+str(index)+'/'+str(j),component='synthetic-component/'+str(index)+'/'+str(j),group='assessment',legal_mask=3);rows[key]=row
            sign=1 if actor==0 else -1;base=state['stores'][0]-state['stores'][1];shift=0 if uniform or j==0 else 20 if extreme else 2
            q8={0:base,1:base-2*sign};q10={a:v+shift*sign for a,v in q8.items()};q12={a:v+4*j*sign for a,v in q10.items()}
            h8[key]=item(row,8,q8);h10[key]=item(row,10,q10);h12[key]=item(row,12,q12)
    return rows,h8,h10,h12


def constant_shift(rows,h10,adverse_cells=()):
    targets={}
    for cell in m.CELLS:
        for j,key in enumerate(k for k,r in rows.items() if r['cell']==cell):
            sign=1 if rows[key]['actor']==0 else -1;shift=4*(1-j if cell in adverse_cells else j)
            targets[key]=item(rows[key],12,{v['pit']:v['value_player0']+shift*sign for v in h10[key]['label']['action_scores']})
    return targets


class QStabilityMetricsAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        proof=json.loads(IMPLEMENTATION.read_text(encoding='utf-8'))
        expected=dict(schema='qstability-metrics-source-freeze-v1',inputs={name:sha(ROOT/name) for name in SOURCES},pure_helper=True,no_fixture_before_freeze=True)
        m.require(m.same(proof,expected),'Formal owner checks require exact current source freeze')
        cls.retained=ROOT/'build/qstability-metrics-author'/uuid.uuid4().hex;cls.retained.mkdir(parents=True)

    def test_01_worst_action_only_exact_mixture_and_concentration(self):
        rows,h8,h10,h12=fixture();projection=m.weight_projection(rows,h8,h10);m.validate_projection(rows,projection)
        for cell in m.CELLS:
            a,b=(projection['rows'][k] for k,r in rows.items() if r['cell']==cell)
            self.assertEqual(m.fraction(a['raw_rule']['raw_weight']),1);self.assertEqual(m.fraction(b['raw_rule']['raw_weight']),F(2,3))
            self.assertEqual((m.fraction(a['weighted_probability']),m.fraction(b['weighted_probability'])),(F(3,40),F(1,20)))
        for masses in projection['masses'].values():
            self.assertTrue(all(m.fraction(v)==F(1,8) for v in masses['cells'].values()));self.assertTrue(all(m.fraction(v)==F(1,4) for v in masses['strata'].values()));self.assertTrue(all(m.fraction(v)==F(1,2) for v in masses['actors'].values()));self.assertEqual(m.fraction(masses['ordinary']),F(1,2))
        result=m.evaluate(rows,projection,h12);scope=result['metrics']['overall']
        self.assertEqual(m.fraction(scope['weighted']['ESS']),F(200,13));self.assertEqual(m.fraction(scope['weighted']['action_Q_drift']),F(8,5));self.assertEqual(m.fraction(scope['uniform']['action_Q_drift']),2);self.assertTrue(result['screen']['passes'])
        self.assertTrue(all(m.fraction(v['weighted']['ESS'])==F(25,13) for v in result['metrics']['by_cell'].values()))
        key=next(iter(rows));row=rows[key];base=row['state']['stores'][0]-row['state']['stores'][1];p=item(row,8,{0:base+10,1:base-10})
        sparse=m.raw_rule(row,p,item(row,10,{0:base+10,1:base-6}));translated=m.raw_rule(row,p,item(row,10,{0:base+14,1:base-6}))
        self.assertEqual((sparse['root_shift_seeds'],translated['root_shift_seeds']),(0,4));self.assertEqual((m.fraction(sparse['mean_action_shift_seeds']),m.fraction(translated['mean_action_shift_seeds'])),(2,4))
        self.assertEqual(m.fraction(sparse['raw_weight']),F(1,2));self.assertEqual(sparse['raw_weight'],translated['raw_weight']);self.assertFalse(sparse['root_and_mean_are_rule_inputs']);self.assertEqual(sparse['max_shift_moves'],[1])
        for field in ('family','component'):
            bad=copy.deepcopy(rows);keys=list(bad);bad[keys[1]][field]=bad[keys[0]][field]
            with self.assertRaises(ValueError):m.weight_projection(bad,h8,h10)
        for change in ('missing','partial','float','bool'):
            bad=copy.deepcopy(h10)
            if change=='missing':del bad[key]
            elif change=='partial':bad[key]['label']['action_scores'].pop()
            else:bad[key]['label']['choice_depth']=10.0 if change=='float' else True
            with self.assertRaises(ValueError):m.weight_projection(rows,h8,bad)

    def test_02_full_tie_actor_signs_rank_changes_and_nonbest_accounting(self):
        rows,h8,h10,h12=fixture()
        for key,row in rows.items():
            b=h10[key]['label']['target_player0'];h10[key]=item(row,10,{0:b,1:b});h8[key]=item(row,8,{0:b,1:b})
            h12[key]=item(row,12,{0:b,1:b-3}) if row['actor']==0 else item(row,12,{0:b+3,1:b})
        projection=m.weight_projection(rows,h8,h10);result=m.evaluate(rows,projection,h12)
        for key,r in result['roots'].items():
            metrics=r['metrics'];self.assertEqual(r['H10_best_moves'],[0,1]);self.assertEqual(m.fraction(metrics['H10_best_H12_regret_mean']),F(3,2));self.assertEqual(m.fraction(metrics['H10_best_H12_regret_min']),0);self.assertEqual(m.fraction(metrics['H10_best_H12_regret_max']),3)
            self.assertEqual(m.fraction(metrics['H10_lexicographic_H12_regret']),0 if r['actor']==0 else 3);self.assertEqual(m.fraction(metrics['max_action_Q_drift']),3)
            self.assertEqual(r['ranking']['strict_reversals'],0);self.assertEqual(r['ranking']['tie_changes'],1);self.assertEqual(m.fraction(metrics['H10_best_action_Q_drift_contribution']),F(3,2));self.assertEqual(m.fraction(metrics['H10_nonbest_action_Q_drift_contribution']),0)
            for action in r['action_changes']:
                self.assertEqual(action['delta_actor'],(1 if r['actor']==0 else -1)*action['delta_player0']);self.assertEqual(action['H12_actor_gap'],next(v['regret_seeds'] for v in r['regrets_by_H10_best_move'] if v['pit']==action['pit']))
        key=next(iter(rows));bad=copy.deepcopy(h10);bad[key]['label']['best_moves']=[0]
        with self.assertRaises(ValueError):m.weight_projection(rows,h8,bad)
        bad=copy.deepcopy(h12);bad[key]['label']['choice_depth']=10
        with self.assertRaises(ValueError):m.evaluate(rows,projection,bad)

    def test_03_strict_primary_capture_ordinary_maximum_and_all_cell_guards(self):
        rows,h8,h10,h12=fixture();projection=m.weight_projection(rows,h8,h10)
        constant={k:item(rows[k],12,{q['pit']:q['value_player0']+3*(1 if rows[k]['actor']==0 else -1) for q in h10[k]['label']['action_scores']}) for k in rows}
        result=m.evaluate(rows,projection,constant);self.assertFalse(result['screen']['passes']);self.assertFalse(result['screen']['conditions']['primary_action_Q_drift_strictly_lower_overall']);self.assertTrue(result['screen']['conditions']['nontrivial_within_cell_rule'])
        ur,ua,ub,uc=fixture(uniform=True);result=m.evaluate(ur,m.weight_projection(ur,ua,ub),uc);self.assertFalse(result['screen']['conditions']['nontrivial_within_cell_rule']);self.assertFalse(result['screen']['passes'])
        er,ea,eb,ec=fixture(extreme=True);result=m.evaluate(er,m.weight_projection(er,ea,eb),ec);self.assertEqual(m.fraction(result['metrics']['overall']['weighted']['ESS']),F(200,17));self.assertFalse(result['screen']['conditions']['ESS_at_least_four_fifths_overall']);self.assertTrue(all(v is False for v in result['screen']['conditions']['ESS_at_least_four_fifths_all_actor_cells'].values()))
        saturated={k:item(rows[k],10,{q['pit']:q['value_player0']+(12+2*(i%2))*(1 if rows[k]['actor']==0 else -1) for q in h8[k]['label']['action_scores']}) for i,k in enumerate(rows)}
        sat=m.weight_projection(rows,h8,saturated);self.assertTrue(all(m.fraction(v['raw_rule']['raw_weight'])==F(1,4) and m.fraction(v['weighted_probability'])==F(1,16) for v in sat['rows'].values()))
        capture=[v for v in m.CELLS if '/capture/' in v];result=m.evaluate(rows,projection,constant_shift(rows,h10,capture));self.assertTrue(result['screen']['conditions']['primary_action_Q_drift_strictly_lower_overall']);self.assertFalse(result['screen']['conditions']['capture_stratum_action_Q_drift_nonincrease']);self.assertFalse(result['screen']['passes'])
        adverse={};target='25-49/ordinary/actor0'
        for cell in m.CELLS:
            for j,key in enumerate(k for k,r in rows.items() if r['cell']==cell):
                b=h10[key]['label']['target_player0'];sign=1 if rows[key]['actor']==0 else -1;regret=4*(1-j if cell==target else j);adverse[key]=item(rows[key],12,{0:b-regret*sign,1:b})
        result=m.evaluate(rows,projection,adverse);self.assertTrue(result['screen']['conditions']['primary_fulltie_regret_nonincrease_overall']);self.assertFalse(result['screen']['conditions']['primary_fulltie_regret_nonincrease_ordinary_actor_cells'][target]);self.assertFalse(result['screen']['conditions']['worst_tie_regret_nonincrease_ordinary_actor_cells'][target]);self.assertFalse(result['screen']['passes'])
        full=m.evaluate(rows,projection,h12);groups=dict(all=full['metrics']['overall'],ordinary=full['metrics']['ordinary'],strata=full['metrics']['by_stratum'],cells=full['metrics']['by_cell'],actors=full['metrics']['by_actor'])
        for metric,guard in (('root_drift','root_drift_nonincrease_overall'),('max_action_Q_drift','max_action_Q_drift_nonincrease_overall')):
            altered=copy.deepcopy(groups);scope=altered['all'];value=m.rational(m.fraction(scope['uniform'][metric])+1);scope['weighted'][metric]=value;scope['weighted']['complete_risk'][metric]=value
            screen=m.favorable_screen(altered,projection);self.assertFalse(screen['conditions'][guard]);self.assertFalse(screen['passes'])

    def test_04_missing_original_mass_rehashed_refusal_and_byte_stable_rebuild(self):
        rows,h8,h10,h12=fixture();projection=m.weight_projection(rows,h8,h10);key=next(iter(rows));partial={k:v for k,v in h12.items() if k!=key};result=m.evaluate(rows,projection,partial);scope=result['metrics']['overall']
        self.assertEqual((result['requested_roots'],result['complete_roots']),(16,15));self.assertEqual(m.fraction(scope['weighted']['original_missing_mass']),F(3,40));self.assertEqual(m.fraction(scope['uniform']['original_missing_mass']),F(1,16));self.assertTrue(all(scope['weighted'][v] is None for v in m.METRICS));self.assertEqual(result['screen']['status'],'incomplete_reference_hold');self.assertIsNone(result['screen']['conditions']['primary_action_Q_drift_strictly_lower_overall']);self.assertFalse(result['screen']['passes'])
        missing=m.evaluate(rows,projection,{});self.assertEqual(m.fraction(missing['metrics']['overall']['weighted']['original_missing_mass']),1);self.assertIsNone(missing['metrics']['overall']['weighted']['conditional_complete_diagnostic']['max_action_Q_drift'])
        full=m.evaluate(rows,projection,h12);file=self.retained/'synthetic-result.json';file.write_text(m.canonical(full)+'\n',encoding='utf-8',newline='\n');before=(sha(file),file.stat().st_size,file.stat().st_mtime_ns)
        self.assertEqual(file.read_bytes(),(m.canonical(m.evaluate(rows,projection,h12))+'\n').encode());m.validate_report(rows,projection,h12,json.loads(file.read_text(encoding='utf-8')));self.assertEqual(before,(sha(file),file.stat().st_size,file.stat().st_mtime_ns))
        for change in ('risk','guard','root','float','rank'):
            bad=copy.deepcopy(full)
            if change=='risk':bad['metrics']['overall']['weighted']['max_action_Q_drift']=m.rational(0)
            elif change=='guard':bad['screen']['conditions']['primary_action_Q_drift_strictly_lower_overall']=False
            elif change=='root':bad['roots'][key]['root_sha256']='0'*64
            elif change=='rank':bad['roots'][key]['ranking']['strict_reversals']=99
            else:bad['requested_roots']=16.0
            with self.assertRaises(ValueError):m.validate_report(rows,projection,h12,bad)
        bad=copy.deepcopy(projection);bad['rows'][key]['weighted_probability']['numerator']=True
        with self.assertRaises(ValueError):m.evaluate(rows,bad,h12)
        bad=copy.deepcopy(h12);bad[key]=None
        with self.assertRaises(ValueError):m.evaluate(rows,projection,bad)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('congklak.native',sys.modules)


if __name__=='__main__':
    started=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromTestCase(QStabilityMetricsAuthor);ids=[v.id() for v in suite];result=unittest.TextTestRunner(verbosity=2).run(suite)
    inputs={name:sha(ROOT/name) for name in SOURCES};inputs[IMPLEMENTATION.relative_to(ROOT).as_posix()]=sha(IMPLEMENTATION)
    retained=getattr(QStabilityMetricsAuthor,'retained',None)
    if retained is not None:inputs.update({p.relative_to(ROOT).as_posix():sha(p) for p in retained.iterdir() if p.is_file()})
    body=dict(schema='qstability-metrics-author-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),covered_methods=ids,methods=unittest.defaultTestLoader.getTestCaseNames(QStabilityMetricsAuthor),inputs=inputs,retained_directory=retained.relative_to(ROOT).as_posix() if retained is not None else None,duration_seconds=time.perf_counter()-started)
    path=ROOT/'build/qstability-metrics-author.json';path.write_text(m.canonical(body)+'\n',encoding='utf-8',newline='\n');raise SystemExit(not result.wasSuccessful())
