"""Four pure synthetic owner checks; no native/search/model/producer imports."""
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
from congklak import stratum_metrics as m

IMPLEMENTATION=ROOT/'build/stratum-metrics-implementation.json'
SOURCES=('congklak/stratum_metrics.py','tests/stratum_metrics_author.py','docs/STRATUM_RESEARCH.md','docs/STRATUM_METRICS.md','docs/STRATUM_SCOPE.md','congklak/stratum_common.py','build/stratum-baseline.json')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def item(row,depth,q):
    best=(max if row['actor']==0 else min)(q.values());return dict(root_sha256=m.digest(row),label=dict(choice_depth=depth,target_player0=best,action_scores=[dict(pit=a,value_player0=v) for a,v in sorted(q.items())],best_moves=[a for a,v in sorted(q.items()) if v==best]),report=dict(synthetic=True,no_engine_call=True))


def fixture(extreme=False,uniform=False):
    """Toy full states/bands; mechanism tags are metadata, not game assertions."""
    rows={};h8={};h10={};h12={}
    for cell_index,cell in enumerate(m.CELLS):
        stratum,actor=cell.rsplit('/actor',1);actor=int(actor);active=60 if stratum.startswith('50-74') else 40
        for j in (0,1):
            state=dict(pits=[[1+j,active//2-1-j,0,0,0,0,0],[2+cell_index%3,active//2-2-cell_index%3,0,0,0,0,0]],stores=[cell_index*2+j,98-active-cell_index*2-j],controller=actor);key=m.rawkey(state)
            row=dict(state_key=key,state=state,actor=actor,stratum=stratum,cell=cell,family='synthetic-family/'+str(cell_index)+'/'+str(j),component='synthetic-component/'+str(cell_index)+'/'+str(j),group='assessment',legal_mask=3);rows[key]=row
            sign=1 if actor==0 else -1;base=state['stores'][0]-state['stores'][1];shift=0 if uniform or j==0 else 6 if extreme else 2
            q8={0:base,1:base-2*sign};q10={a:v+shift*sign for a,v in q8.items()};q12={a:v+4*j*sign for a,v in q10.items()}
            h8[key]=item(row,8,q8);h10[key]=item(row,10,q10);h12[key]=item(row,12,q12)
    return rows,h8,h10,h12


class StratumMetricsAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        proof=json.loads(IMPLEMENTATION.read_text(encoding='utf-8'));expected=dict(schema='stratum-metrics-source-freeze-v1',inputs={name:sha(ROOT/name) for name in SOURCES},pure_helper=True,no_search_before_freeze=True)
        m.require(m.same(proof,expected),'Formal owner checks require exact current source freeze')
        cls.retained=ROOT/'build/stratum-metrics-author'/uuid.uuid4().hex;cls.retained.mkdir(parents=True)

    def test_01_exact_rule_fixed_cells_actors_and_concentration(self):
        rows,h8,h10,h12=fixture();projection=m.weight_projection(rows,h8,h10);self.assertEqual(projection,m.validate_projection(rows,projection));self.assertFalse(projection['H12_used']);self.assertFalse(projection['weights_applied'])
        for cell in m.CELLS:
            pair=[k for k,r in rows.items() if r['cell']==cell];a,b=(projection['rows'][k] for k in pair)
            self.assertEqual(m.fraction(a['raw_rule']['raw_weight']),1);self.assertEqual(m.fraction(b['raw_rule']['raw_weight']),F(1,2));self.assertEqual(m.fraction(a['weighted_probability']),F(1,12));self.assertEqual(m.fraction(b['weighted_probability']),F(1,24));self.assertEqual(m.fraction(a['uniform_probability']),F(1,16))
        for name in ('uniform','weighted'):
            masses=projection['masses'][name];self.assertTrue(all(m.fraction(v)==F(1,8) for v in masses['cells'].values()));self.assertTrue(all(m.fraction(v)==F(1,4) for v in masses['strata'].values()));self.assertTrue(all(m.fraction(v)==F(1,2) for v in masses['actors'].values()));self.assertEqual(m.fraction(masses['ordinary']),F(1,2))
        result=m.evaluate(rows,projection,h12);overall=result['metrics']['overall'];self.assertEqual(m.fraction(overall['weighted']['ESS']),F(72,5));self.assertEqual(m.fraction(overall['uniform']['ESS']),16);self.assertEqual(m.fraction(overall['weighted']['root_drift']),F(4,3));self.assertEqual(m.fraction(overall['uniform']['root_drift']),2);self.assertTrue(result['screen']['passes'])
        self.assertTrue(all(m.fraction(c['weighted']['ESS'])==F(9,5) for c in result['metrics']['by_cell'].values()));self.assertTrue(all(m.fraction(c['weighted']['max_conditional_probability'])==F(2,3) for c in result['metrics']['by_cell'].values()))
        for field in ('family','component'):
            bad=copy.deepcopy(rows);keys=list(bad);bad[keys[1]][field]=bad[keys[0]][field]
            with self.assertRaises(ValueError):m.weight_projection(bad,h8,h10)

    def test_02_full_tie_actor_correct_mean_max_and_lexicographic_regrets(self):
        rows,h8,h10,h12=fixture()
        for key,row in rows.items():
            b=h10[key]['label']['target_player0'];h10[key]=item(row,10,{0:b,1:b});h8[key]=item(row,8,{0:b,1:b})
            h12[key]=item(row,12,{0:b,1:b-3}) if row['actor']==0 else item(row,12,{0:b+3,1:b})
        projection=m.weight_projection(rows,h8,h10);result=m.evaluate(rows,projection,h12)
        for key,row in result['roots'].items():
            self.assertEqual(row['H10_best_moves'],[0,1]);q=row['metrics'];self.assertEqual(m.fraction(q['H10_best_H12_regret_mean']),F(3,2));self.assertEqual(m.fraction(q['H10_best_H12_regret_min']),0);self.assertEqual(m.fraction(q['H10_best_H12_regret_max']),3)
            self.assertEqual(m.fraction(q['H10_lexicographic_H12_regret']),0 if row['actor']==0 else 3);self.assertFalse(row['best_sets_disjoint'])
        key=next(iter(rows));bad=copy.deepcopy(h10);bad[key]['label']['best_moves']=[0]
        with self.assertRaises(ValueError):m.weight_projection(rows,h8,bad)
        bad=copy.deepcopy(h12);bad[key]['label']['choice_depth']=10
        with self.assertRaises(ValueError):m.evaluate(rows,projection,bad)
        for value in (True,12.0):
            bad=copy.deepcopy(h12);bad[key]['label']['choice_depth']=value
            with self.assertRaises(ValueError):m.evaluate(rows,projection,bad)

    def test_03_identity_concentration_and_hidden_ordinary_cell_harm_guards(self):
        rows,h8,h10,h12=fixture();projection=m.weight_projection(rows,h8,h10)
        constant={k:item(rows[k],12,{a:v+3*(1 if rows[k]['actor']==0 else -1) for a,v in {q['pit']:q['value_player0'] for q in h10[k]['label']['action_scores']}.items()}) for k in rows}
        result=m.evaluate(rows,projection,constant);self.assertFalse(result['screen']['passes']);self.assertFalse(result['screen']['conditions']['at_least_one_overall_risk_strictly_improves']);self.assertTrue(result['screen']['conditions']['nontrivial_within_cell_rule'])
        urows,uh8,uh10,uh12=fixture(uniform=True);uniform=m.evaluate(urows,m.weight_projection(urows,uh8,uh10),uh12);self.assertFalse(uniform['screen']['passes']);self.assertFalse(uniform['screen']['conditions']['nontrivial_within_cell_rule'])
        erows,eh8,eh10,eh12=fixture(extreme=True);extreme=m.evaluate(erows,m.weight_projection(erows,eh8,eh10),eh12);self.assertEqual(m.fraction(extreme['metrics']['overall']['weighted']['ESS']),F(200,17));self.assertFalse(extreme['screen']['conditions']['ESS_at_least_four_fifths_overall']);self.assertFalse(extreme['screen']['conditions']['ESS_at_least_four_fifths_ordinary_actor_cells']['25-49/ordinary/actor0'])
        adverse={};target='25-49/ordinary/actor0'
        for cell in m.CELLS:
            pair=[k for k,r in rows.items() if r['cell']==cell]
            for j,key in enumerate(pair):
                b=h10[key]['label']['target_player0'];sign=1 if rows[key]['actor']==0 else -1;regret=4*(1-j if cell==target else j);adverse[key]=item(rows[key],12,{0:b-regret*sign,1:b})
        result=m.evaluate(rows,projection,adverse);self.assertLess(m.fraction(result['metrics']['overall']['weighted']['H10_best_H12_regret_mean']),m.fraction(result['metrics']['overall']['uniform']['H10_best_H12_regret_mean']));self.assertTrue(result['screen']['conditions']['primary_fulltie_regret_nonincrease_overall']);self.assertFalse(result['screen']['conditions']['primary_fulltie_regret_nonincrease_ordinary_actor_cells'][target]);self.assertFalse(result['screen']['conditions']['worst_tie_regret_nonincrease_ordinary_actor_cells'][target]);self.assertFalse(result['screen']['passes'])

    def test_04_original_missing_mass_rehash_refusal_and_deterministic_restart(self):
        rows,h8,h10,h12=fixture();projection=m.weight_projection(rows,h8,h10);key=next(iter(rows));partial={k:v for k,v in h12.items() if k!=key};result=m.evaluate(rows,projection,partial);overall=result['metrics']['overall']
        self.assertEqual((result['requested_roots'],result['complete_roots']),(16,15));self.assertEqual(m.fraction(overall['weighted']['original_missing_mass']),F(1,12));self.assertEqual(m.fraction(overall['uniform']['original_missing_mass']),F(1,16));self.assertIsNone(overall['weighted']['H10_best_H12_regret_mean']);self.assertEqual(result['screen']['status'],'incomplete_reference_hold');self.assertIsNone(result['screen']['conditions']['primary_fulltie_regret_nonincrease_overall']);self.assertFalse(result['screen']['passes'])
        missing=m.evaluate(rows,projection,{});self.assertEqual(m.fraction(missing['metrics']['overall']['weighted']['original_missing_mass']),1);self.assertIsNone(missing['metrics']['overall']['weighted']['root_drift']);self.assertIsNone(missing['metrics']['overall']['weighted']['conditional_complete_diagnostic']['root_drift'])
        full=m.evaluate(rows,projection,h12);file=self.retained/'synthetic-result.json';file.write_text(m.canonical(full)+'\n',encoding='utf-8',newline='\n');before=(sha(file),file.stat().st_size,file.stat().st_mtime_ns);rebuilt=m.evaluate(rows,projection,h12);self.assertEqual(file.read_bytes(),(m.canonical(rebuilt)+'\n').encode());m.validate_report(rows,projection,h12,json.loads(file.read_text(encoding='utf-8')));self.assertEqual(before,(sha(file),file.stat().st_size,file.stat().st_mtime_ns))
        for change in ('risk','guard','raw_identity','float_alias'):
            bad=copy.deepcopy(full)
            if change=='risk':bad['metrics']['overall']['weighted']['root_drift']=m.rational(0)
            elif change=='guard':bad['screen']['conditions']['at_least_one_overall_risk_strictly_improves']=False
            elif change=='raw_identity':bad['roots'][key]['root_sha256']='0'*64
            else:bad['requested_roots']=16.0
            with self.assertRaises(ValueError):m.validate_report(rows,projection,h12,bad)
        bad=copy.deepcopy(projection);bad['rows'][key]['weighted_probability']['numerator']=True
        with self.assertRaises(ValueError):m.evaluate(rows,bad,h12)
        bad=copy.deepcopy(h12);bad[key]=None
        with self.assertRaises(ValueError):m.evaluate(rows,projection,bad)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('congklak.native',sys.modules)


if __name__=='__main__':
    started=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromTestCase(StratumMetricsAuthor);ids=[v.id() for v in suite];result=unittest.TextTestRunner(verbosity=2).run(suite)
    inputs={name:sha(ROOT/name) for name in SOURCES};inputs[IMPLEMENTATION.relative_to(ROOT).as_posix()]=sha(IMPLEMENTATION)
    retained=getattr(StratumMetricsAuthor,'retained',None)
    if retained is not None:inputs.update({p.relative_to(ROOT).as_posix():sha(p) for p in retained.iterdir() if p.is_file()})
    body=dict(schema='stratum-metrics-author-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),covered_methods=ids,methods=unittest.defaultTestLoader.getTestCaseNames(StratumMetricsAuthor),inputs=inputs,retained_directory=retained.relative_to(ROOT).as_posix() if retained is not None else None,duration_seconds=time.perf_counter()-started)
    path=ROOT/'build/stratum-metrics-author.json';path.write_text(m.canonical(body)+'\n',encoding='utf-8');raise SystemExit(not result.wasSuccessful())
