"""Four source-frozen literal/saved-data owner checks; two consumed cases only."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import unittest
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import regret_forensics as f
FREEZE=ROOT/'build/regret-forensics-source-freeze.json'
SOURCES=('congklak/regret_forensics.py','tests/regret_forensics_author.py','docs/REGRET_FORENSICS.md','docs/REGRET_SCOPE.md','tests/reference.py','congklak/capture_forensics.py','congklak/regret_common.py','build/regret-baseline.json')
SOURCE_DATA=('data/qstability-v1/work.sqlite3','data/qstability-v1/manifest.json','data/qstability-v1/references-prefix.sqlite3','data/qstability-v1/references-manifest.json','build/qstability-rule.json','build/qstability-result.json','build/qstability-registry.json')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fixture_inputs():
    out=[]
    with sqlite3.connect((ROOT/SOURCE_DATA[0]).resolve().as_uri()+'?mode=ro',uri=True) as db:
        for suffix in ('178','91'):
            selected=db.execute('SELECT state,body,hash FROM roots WHERE json_extract(body,\'$.family\')=?',('midgame-history-pilot/20261023/'+suffix,)).fetchall();f.require(len(selected)==1,'Expected exact consumed fixture family')
            key,text,checksum=selected[0];row=json.loads(text);f.require(f.digest(row)==checksum and row['state_key']==key,'Corrupt saved fixture root');items={};attempts={}
            for stage in f.STAGES:
                for ending,target in (('_labels',items),('_attempts',attempts)):
                    text,checksum=db.execute('SELECT body,hash FROM '+stage+ending+' WHERE state=?',(key,)).fetchone();body=json.loads(text);f.require(f.digest(body)==checksum,'Corrupt saved fixture source');target[stage]=body
            out.append((row,items,attempts))
    return out


class RegretForensicsAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        proof=json.loads(FREEZE.read_text(encoding='utf-8'));expected=dict(schema='regret-forensics-source-freeze-v1',inputs={n:sha(ROOT/n) for n in (*SOURCES,*SOURCE_DATA)},literal_only=True,no_formal_fixture_before_freeze=True)
        f.require(f.same(proof,expected),'Formal fixtures require exact current source freeze');cls.fixtures=fixture_inputs();cls.retained=ROOT/'build/regret-forensics-author'/uuid.uuid4().hex;cls.retained.mkdir(parents=True);cls.cases=[f.build_case(*args) for args in cls.fixtures]

    def test_01_all_histories_actions_sources_overlap_and_literal_reuse(self):
        for (row,items,attempts),case in zip(self.fixtures,self.cases):
            legal=[p for p in range(7) if row['legal_mask']&(1<<p)];self.assertEqual(case['legal_actions'],legal);self.assertEqual(case['counts']['stage_report_references'],30);self.assertEqual(case['counts']['endpoint_action_Q_values'],3*len(legal));self.assertEqual(case['counts']['PV_references'],30*(len(legal)+1));self.assertEqual(case['counts']['action_pairs'],len(legal)*(len(legal)-1)//2);self.assertEqual(case['counts']['action_pair_history_rows'],12*case['counts']['action_pairs'])
            self.assertEqual(case['overlap']['unique_actual_depths'],list(range(1,13)));self.assertTrue(case['overlap']['Q_best_signatures_equal']);self.assertEqual(case['literal_helper_source_sha256'],sha(ROOT/'congklak/capture_forensics.py'));self.assertEqual(case['source_root_sha256'],row['source_root_sha256']);self.assertEqual(case['root_sha256'],f.digest(row));self.assertNotEqual(case['root_sha256'],case['source_root_sha256'])
            refs=[r for line in case['lines'].values() for r in line['source_references']];self.assertEqual(len(refs),case['counts']['PV_references']);self.assertEqual(len({(r['stage'],r['actual_depth'],r['kind'],r['pit']) for r in refs}),len(refs));self.assertLessEqual(case['counts']['deduplicated_literal_lines'],len(refs))
            for stage,depth in f.DEPTHS.items():
                self.assertEqual([r['actual_depth'] for r in case['report_index'][stage]],list(range(1,depth+1)))
                for pit in legal:
                    score=case['actions'][str(pit)]['scores'][stage];source=next(v['value_player0'] for v in items[stage]['label']['action_scores'] if v['pit']==pit);self.assertEqual(score['value_player0'],source);self.assertEqual(score['value_actor'],source*(1 if row['actor']==0 else -1))
            self.assertIs(f.replay_line,f.saved.replay_line);self.assertIs(f.compare_lines,f.saved.compare_lines);self.assertIs(f.rank_diagnostics,f.saved.rank_diagnostics)

    def test_02_actor1_best_reversal_all_alternatives_and_gap_trajectory(self):
        case=self.cases[0];self.assertEqual(case['family'],'midgame-history-pilot/20261023/178');self.assertEqual(case['actor'],1);self.assertEqual(case['root_mechanism'],'bonus');self.assertEqual(len(case['actions']),7)
        regret=case['H10_best_H12_regret'];self.assertEqual((regret['before_best_moves'],regret['after_best_moves']),([5],[6]));self.assertTrue(regret['best_sets_disjoint']);self.assertEqual(regret['regret_mean'],f.rational(4));self.assertEqual(regret['regret_max'],f.rational(4));self.assertEqual(regret['after_regrets_by_before_best'],[dict(pit=5,regret_seeds=4)])
        pair=case['action_pair_trajectories']['5_6'];self.assertEqual([pair['endpoints'][s]['advantage_a_over_b_actor'] for s in f.STAGES],[2,1,-4]);self.assertEqual([(pair['endpoints'][s]['Q_a_player0'],pair['endpoints'][s]['Q_b_player0']) for s in f.STAGES],[(14,16),(13,14),(16,12)])
        self.assertEqual([v['actual_depth'] for v in pair['H12_job']],list(range(1,13)));self.assertEqual(len(pair['adjacent_changes']),11)
        for before,after,change in zip(pair['H12_job'],pair['H12_job'][1:],pair['adjacent_changes']):
            self.assertEqual(before['advantage_a_over_b_actor'],before['Q_b_player0']-before['Q_a_player0']);self.assertEqual(change['numeric_gap_change_actor'],after['advantage_a_over_b_actor']-before['advantage_a_over_b_actor']);self.assertEqual(change['strict_reversal'],before['advantage_a_over_b_actor']*after['advantage_a_over_b_actor']<0)
        synthetic=copy.deepcopy(case);report=synthetic['report_index']['h12'][0];report['signature']['action_scores']=[dict(pit=5,value_player0=4),dict(pit=6,value_player0=4)];report['signature']['best_moves']=[5,6]
        tied=f.pair_trajectory(synthetic,5,6);self.assertEqual(tied['H12_job'][0]['relation'],'tied');self.assertEqual(tied['H12_job'][0]['best_moves'],[5,6]);self.assertTrue(tied['H12_job'][0]['a_in_best_set'] and tied['H12_job'][0]['b_in_best_set'])
        with self.assertRaises(ValueError):f.pair_trajectory(case,6,5)

    def test_03_generic_mechanisms_prefix_tails_and_leaf_reconciliation(self):
        bonus,capture=self.cases;self.assertEqual(capture['root_mechanism'],'capture');self.assertTrue(any(v['immediate_capture'] for v in capture['actions'].values()));self.assertFalse(any(v['immediate_capture'] for v in bonus['actions'].values()));self.assertTrue(any(v['immediate_bonus'] for v in bonus['actions'].values()))
        for case in self.cases:
            for v in case['lines'].values():
                line=v['trace'];leaf=line['leaf'];totals=line['totals'];self.assertEqual(leaf['saved_score_minus_leaf_store_margin'],0);self.assertEqual(sum(leaf['stores'])+leaf['active_seeds'],98);self.assertEqual(totals['store_margin_change_player0'],totals['capture_margin_change_player0']+totals['sowing_store_margin_change_player0'])
            for a in case['actions'].values():
                comparison=a['PV_comparisons']['h10_h12'];prefix=comparison['common_choice_prefix'];difference=comparison['first_different_choice']
                if difference is not None:self.assertEqual(difference['choice'],prefix+1);self.assertEqual(difference['left_mechanics']['before'],difference['right_mechanics']['before'])
        terminal=dict(pits=[[0,0,0,0,0,0,1],[0]*7],stores=[48,49],controller=0);move=dict(actor=0,pit=6,bonus_choice=False,forced_passes_after=[],next_controller=None,terminal_after=True);line=f.replay_line(terminal,[move],1,0);self.assertTrue(line['leaf']['terminal']);self.assertEqual(line['leaf']['stores'],[49,49])
        passed=dict(pits=[[1,0,0,0,0,0,0],[0]*7],stores=[48,49],controller=0);move=dict(actor=0,pit=0,bonus_choice=False,forced_passes_after=[dict(zip(('from','to'),(1,0)))],next_controller=0,terminal_after=False);line=f.replay_line(passed,[move],1,-1);self.assertEqual(line['totals']['forced_passes'],1);self.assertFalse(line['steps'][0]['bonus_choice'])
        row,items,attempts=copy.deepcopy(self.fixtures[1]);row['stratum']='25-49/bonus'
        for attempt in attempts.values():attempt['study_row_sha256']=f.digest(row)
        with self.assertRaises(ValueError):f.build_case(row,items,attempts)

    def test_04_rehashed_facts_alias_source_refusal_and_byte_stable_rebuild(self):
        row,items,attempts=self.fixtures[0];case=self.cases[0];path=self.retained/'saved-case-rebuild.json';path.write_text(f.canonical(case)+'\n',encoding='utf-8',newline='\n');before=(sha(path),path.stat().st_size,path.stat().st_mtime_ns)
        self.assertEqual(path.read_bytes(),(f.canonical(f.build_case(row,items,attempts))+'\n').encode());f.validate_case(row,items,attempts,json.loads(path.read_text(encoding='utf-8')));self.assertEqual(before,(sha(path),path.stat().st_size,path.stat().st_mtime_ns))
        for field in ('pair','regret','mechanism','hash'):
            bad=copy.deepcopy(case)
            if field=='pair':bad['action_pair_trajectories']['5_6']['endpoints']['h12']['advantage_a_over_b_actor']=4
            elif field=='regret':bad['H10_best_H12_regret']['regret_max']=f.rational(0)
            elif field=='mechanism':bad['root_mechanism']='capture'
            else:bad['literal_helper_source_sha256']='0'*64
            with self.assertRaises(ValueError):f.validate_case(row,items,attempts,bad)
        bad=copy.deepcopy(attempts);bad['h10']['history'].pop(0)
        with self.assertRaises(ValueError):f.build_case(row,items,bad)
        bad=copy.deepcopy(items);bad['h8']['root_sha256']='0'*64
        with self.assertRaises(ValueError):f.build_case(row,bad,attempts)
        bad=copy.deepcopy(row);bad['actor']=True
        with self.assertRaises(ValueError):f.build_case(bad,items,attempts)
        source=items['h12']['report']['candidates'][0];pv=copy.deepcopy(source['pv']);pv[0]['terminal_after']=not pv[0]['terminal_after']
        with self.assertRaises(ValueError):f.replay_line(row['state'],pv,12)
        with self.assertRaises(ValueError):f.replay_line(row['state'],source['pv'],12,source['depth_result']['value_player0']+1)
        self.assertNotIn('congklak.native',sys.modules);self.assertNotIn('congklak.neural_relay',sys.modules);self.assertNotIn('torch',sys.modules)


if __name__=='__main__':
    started=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromTestCase(RegretForensicsAuthor);ids=[test.id() for test in suite];result=unittest.TextTestRunner(verbosity=2).run(suite)
    inputs={n:sha(ROOT/n) for n in (*SOURCES,*SOURCE_DATA)};inputs[FREEZE.relative_to(ROOT).as_posix()]=sha(FREEZE);retained=getattr(RegretForensicsAuthor,'retained',None)
    if retained is not None:inputs.update({p.relative_to(ROOT).as_posix():sha(p) for p in retained.iterdir() if p.is_file()})
    body=dict(schema='regret-forensics-author-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),covered_methods=ids,methods=unittest.defaultTestLoader.getTestCaseNames(RegretForensicsAuthor),inputs=inputs,retained_directory=retained.relative_to(ROOT).as_posix() if retained is not None else None,duration_seconds=time.perf_counter()-started)
    (ROOT/'build/regret-forensics-author.json').write_text(f.canonical(body)+'\n',encoding='utf-8',newline='\n');raise SystemExit(not result.wasSuccessful())
