"""Four source-frozen saved-data/literal-only owner checks, first2 consumed cases."""
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
from congklak import capture_forensics as f
FREEZE=ROOT/'build/capture-forensics-source-freeze.json'
SOURCES=('congklak/capture_forensics.py','tests/capture_forensics_author.py','docs/CAPTURE_FORENSICS.md','docs/CAPTURE_SCOPE.md','tests/reference.py','congklak/capture_common.py','build/capture-baseline.json')
SOURCE_DATA=('data/stratum-weighting-v1/work.sqlite3','data/stratum-weighting-v1/manifest.json','data/stratum-weighting-v1/references-prefix.sqlite3','data/stratum-weighting-v1/references-manifest.json','build/stratum-rule.json','build/stratum-result.json','build/stratum-registry.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fixture_inputs():
    with sqlite3.connect((ROOT/SOURCE_DATA[0]).resolve().as_uri()+'?mode=ro',uri=True) as db:
        selected=db.execute("SELECT state,body,hash FROM roots WHERE json_extract(body,'$.stratum')='25-49/capture' ORDER BY state LIMIT 2").fetchall();out=[]
        for key,text,checksum in selected:
            row=json.loads(text);f.require(f.digest(row)==checksum,'Corrupt saved root');items={};attempts={}
            for stage in f.STAGES:
                for suffix,target in (('_labels',items),('_attempts',attempts)):
                    text,checksum=db.execute('SELECT body,hash FROM '+stage+suffix+' WHERE state=?',(key,)).fetchone();body=json.loads(text);f.require(f.digest(body)==checksum,'Corrupt saved source');target[stage]=body
            out.append((row,items,attempts))
    return out


class CaptureForensicsAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        proof=json.loads(FREEZE.read_text(encoding='utf-8'));expected=dict(schema='capture-forensics-source-freeze-v1',inputs={n:sha(ROOT/n) for n in (*SOURCES,*SOURCE_DATA)},literal_only=True,no_formal_fixture_before_freeze=True)
        f.require(f.same(proof,expected),'Formal fixture requires current exact source freeze');cls.fixtures=fixture_inputs();cls.retained=ROOT/'build/capture-forensics-author'/uuid.uuid4().hex;cls.retained.mkdir(parents=True);cls.cases=[f.build_case(*args) for args in cls.fixtures]

    def test_01_all_saved_histories_full_actions_overlap_and_actor_signs(self):
        for (row,items,attempts),case in zip(self.fixtures,self.cases):
            legal=[p for p in range(7) if row['legal_mask']&(1<<p)];self.assertEqual(case['legal_actions'],legal);self.assertEqual(case['counts']['stage_report_references'],30);self.assertEqual(case['counts']['endpoint_action_Q_values'],len(legal)*3);self.assertEqual(case['counts']['PV_references'],(len(legal)+1)*30);self.assertLessEqual(case['counts']['deduplicated_literal_lines'],case['counts']['PV_references']);self.assertEqual(case['overlap']['unique_actual_depths'],list(range(1,13)))
            references=[link for v in case['lines'].values() for link in v['source_references']];self.assertEqual(len(references),case['counts']['PV_references']);self.assertEqual(len({(v['stage'],v['actual_depth'],v['kind'],v['pit']) for v in references}),len(references))
            for stage,depth in f.DEPTHS.items():
                self.assertEqual([v['actual_depth'] for v in case['report_index'][stage]],list(range(1,depth+1)))
                for pit in legal:
                    score=case['actions'][str(pit)]['scores'][stage];source=next(v['value_player0'] for v in items[stage]['label']['action_scores'] if v['pit']==pit);self.assertEqual(score['value_player0'],source);self.assertEqual(score['value_actor'],source*(1 if row['actor']==0 else -1))
            self.assertEqual([v['depth'] for v in case['H12_job_depth_trajectory']],list(range(1,13)));self.assertEqual(case['source_root_sha256'],row['source_root_sha256']);self.assertEqual(case['root_sha256'],f.digest(row));self.assertNotEqual(case['root_sha256'],case['source_root_sha256'])

    def test_02_literal_capture_relay_bonus_forced_pass_and_terminal(self):
        case=self.cases[0];self.assertEqual(case['family'],'midgame-history-pilot/20261023/283');a=case['actions'];self.assertEqual(a['1']['immediate']['captured_seeds'],4);self.assertFalse(a['1']['immediate_bonus']);self.assertTrue(a['3']['immediate_bonus']);self.assertFalse(a['3']['immediate_capture']);self.assertEqual(a['3']['immediate_relays'],1);self.assertEqual(a['3']['immediate']['deposits'],4);self.assertEqual(a['4']['immediate']['captured_seeds'],2)
        terminal=dict(pits=[[0,0,0,0,0,0,1],[0]*7],stores=[48,49],controller=0);move=dict(actor=0,pit=6,bonus_choice=False,forced_passes_after=[],next_controller=None,terminal_after=True);line=f.replay_line(terminal,[move],1,0);self.assertTrue(line['leaf']['terminal']);self.assertEqual(line['leaf']['active_seeds'],0);self.assertEqual(line['leaf']['stores'],[49,49]);self.assertEqual(line['totals']['store_margin_change_player0'],1)
        passed=dict(pits=[[1,0,0,0,0,0,0],[0]*7],stores=[48,49],controller=0);move=dict(actor=0,pit=0,bonus_choice=False,forced_passes_after=[dict(zip(('from','to'),(1,0)))],next_controller=0,terminal_after=False);line=f.replay_line(passed,[move],1,-1);self.assertFalse(line['steps'][0]['bonus_choice']);self.assertEqual(line['steps'][0]['after']['controller'],0);self.assertEqual(line['totals']['forced_passes'],1)
        for v in case['lines'].values():
            trace=v['trace'];self.assertEqual(trace['leaf']['saved_score_minus_leaf_store_margin'],0);totals=trace['totals'];self.assertEqual(totals['store_margin_change_player0'],totals['capture_margin_change_player0']+totals['sowing_store_margin_change_player0']);self.assertEqual(sum(trace['leaf']['stores'])+trace['leaf']['active_seeds'],98)

    def test_03_prefix_divergence_full_ties_and_actor_relative_gaps(self):
        q0={0:4,1:4,2:0};q1={0:3,1:1,2:2};rank=f.rank_diagnostics(q0,q1,0);self.assertEqual(rank['best_before'],[0,1]);self.assertEqual(rank['best_after'],[0]);self.assertEqual(rank['strict_reversals'],1);self.assertEqual(rank['tie_changes'],1);self.assertFalse(rank['best_sets_disjoint']);self.assertEqual(len(rank['before_best_anchor_gaps']),6)
        rank=f.rank_diagnostics(q0,q1,1);self.assertEqual(rank['best_before'],[2]);self.assertEqual(rank['best_after'],[1]);self.assertEqual(next(v['gap_change_actor'] for v in rank['before_best_anchor_gaps'] if v['pit']==0),-3)
        row,items,_=self.fixtures[0];candidate=items['h12']['report']['candidates'][0];short=f.replay_line(row['state'],candidate['pv'][:1],1);long=f.replay_line(row['state'],candidate['pv'][:2],2);comp=f.compare_lines(short,long);self.assertEqual(comp['common_choice_prefix'],1);self.assertIsNone(comp['first_different_choice']);self.assertEqual(comp['relationship'],'one_recorded_line_is_a_prefix')
        other=items['h12']['report']['candidates'][1];different=f.replay_line(row['state'],other['pv'][:1],1);comp=f.compare_lines(short,different);self.assertEqual(comp['common_choice_prefix'],0);self.assertEqual(comp['first_different_choice']['choice'],1);self.assertEqual(comp['first_different_choice']['state'],row['state'])
        bad=copy.deepcopy(candidate['pv'][:1]);bad[0]['actor']=1-row['actor']
        with self.assertRaises(ValueError):f.replay_line(row['state'],bad,1)

    def test_04_rehashed_facts_source_alias_refusal_and_byte_stable_rebuild(self):
        row,items,attempts=self.fixtures[0];case=self.cases[0];path=self.retained/'synthetic-rebuild.json';path.write_text(f.canonical(case)+'\n',encoding='utf-8',newline='\n');before=(sha(path),path.stat().st_size,path.stat().st_mtime_ns);rebuilt=f.build_case(row,items,attempts);self.assertEqual(path.read_bytes(),(f.canonical(rebuilt)+'\n').encode());f.validate_case(row,items,attempts,json.loads(path.read_text(encoding='utf-8')));self.assertEqual(before,(sha(path),path.stat().st_size,path.stat().st_mtime_ns))
        bad=copy.deepcopy(case);bad['actions']['1']['immediate']['captured_seeds']+=1
        with self.assertRaises(ValueError):f.validate_case(row,items,attempts,bad)
        bad=copy.deepcopy(attempts);bad['h10']['history'].pop(0)
        with self.assertRaises(ValueError):f.build_case(row,items,bad)
        bad=copy.deepcopy(items);bad['h8']['root_sha256']='0'*64
        with self.assertRaises(ValueError):f.build_case(row,bad,attempts)
        bad=copy.deepcopy(row);bad['actor']=False
        with self.assertRaises(ValueError):f.build_case(bad,items,attempts)
        saved=items['h12']['report']['candidates'][0];pv=copy.deepcopy(saved['pv']);pv[0]['terminal_after']=not pv[0]['terminal_after']
        with self.assertRaises(ValueError):f.replay_line(row['state'],pv,12)
        with self.assertRaises(ValueError):f.replay_line(row['state'],saved['pv'],12,saved['depth_result']['value_player0']+1)
        self.assertNotIn('congklak.native',sys.modules);self.assertNotIn('congklak.neural_relay',sys.modules);self.assertNotIn('torch',sys.modules)


if __name__=='__main__':
    started=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromTestCase(CaptureForensicsAuthor);ids=[test.id() for test in suite];result=unittest.TextTestRunner(verbosity=2).run(suite);inputs={n:sha(ROOT/n) for n in (*SOURCES,*SOURCE_DATA)};inputs[FREEZE.relative_to(ROOT).as_posix()]=sha(FREEZE);retained=getattr(CaptureForensicsAuthor,'retained',None)
    if retained is not None:inputs.update({p.relative_to(ROOT).as_posix():sha(p) for p in retained.iterdir() if p.is_file()})
    body=dict(schema='capture-forensics-author-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),covered_methods=ids,methods=unittest.defaultTestLoader.getTestCaseNames(CaptureForensicsAuthor),inputs=inputs,retained_directory=retained.relative_to(ROOT).as_posix() if retained is not None else None,duration_seconds=time.perf_counter()-started)
    (ROOT/'build/capture-forensics-author.json').write_text(f.canonical(body)+'\n',encoding='utf-8',newline='\n');raise SystemExit(not result.wasSuccessful())
