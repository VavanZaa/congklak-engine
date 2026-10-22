"""Four current saved-data checks; none execute new searches or optimizer work."""
from collections import defaultdict
from pathlib import Path
import json
import subprocess
import sys
import unittest
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config, ordinary_value_loss_review as review


class OrdinaryValueLossReviewAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=review.common.read(review.OUTPUT)
        cls.fixture=ROOT/'build/ordinary-value-loss-review-author'/uuid.uuid4().hex
        cls.fixture.mkdir(parents=True)

    def test_01_all_saved_games_matched_margins_and_absolute_losses(self):
        body=self.report;review.common.current(body['inputs'])
        self.assertEqual((body['counts']['games'],body['counts']['choices'],body['counts']['matched_seats']),(96,3268,48))
        with review.closing(review._ro(review.JOURNAL)) as db:
            games={}
            for text,digest in db.execute('SELECT body,hash FROM games'):
                row=json.loads(text);self.assertEqual(config.digest(row),digest);self.assertEqual(row['status'],'terminal');games[row['comparison'],row['pair'],row['candidate_seat']]=row
        sums=defaultdict(list);losses=[]
        for seat in body['seats']:
            base,pair,position=seat['baseline'],seat['pair'],seat['candidate_seat'];left=games['ordinary_control_vs_'+base,pair,position];right=games['ordinary_value_vs_'+base,pair,position]
            self.assertEqual(left['start'],right['start']);delta=right['candidate_final_margin']-left['candidate_final_margin'];self.assertEqual(delta,seat['candidate_minus_control_margin']);sums[base].append(delta)
            self.assertEqual(left['moves']==right['moves'],seat['first_divergence']['identical_action_trajectory'])
            if base=='H_v1' and right['candidate_final_margin']<0:losses.append((right['candidate_final_margin'],pair,position))
        self.assertEqual(sorted(losses),[(row['candidate_margin'],row['pair'],row['candidate_seat']) for row in body['absolute_H_v1_losses']])
        for base,values in sums.items():self.assertEqual(sum(values)/24,body['summary'][base]['paired_mean_candidate_minus_control'])
        self.assertEqual(body['summary']['H_v1']['absolute_losses_with_identical_trajectory'],12)
        self.assertEqual(body['summary']['H8_relay_CPU']['opponent_first_divergences'],2)

    def test_02_exact_state_Q_and_player1_regret_no_quotient_transfer(self):
        state=dict(pits=[[1,2,0,0,0,0,0],[0,0,0,0,2,1,0]],stores=[45,47],controller=1)
        label=dict(kind='finite_choice_depth',choice_depth=8,target_player0=-3,best_moves=[4],action_scores=[dict(pit=4,value_player0=-3),dict(pit=5,value_player0=2)])
        item=dict(state=state,label=label,source=dict(path='synthetic'));index={config.canonical(state):[item,{**item,'source':dict(path='second-provenance')}]}
        q=review.existing_Q(state,4,5,index);self.assertEqual(q['coverage'],'exact_full_state_stored_Q');self.assertEqual(len(q['comparisons']),1)
        row=q['comparisons'][0];self.assertEqual((row['control_actor_regret'],row['candidate_actor_regret']),(0,5));self.assertEqual(len(row['sources']),2)
        swapped=dict(pits=list(reversed(state['pits'])),stores=list(reversed(state['stores'])),controller=0)
        self.assertEqual(review.existing_Q(swapped,4,5,index)['coverage'],'uncovered_in_index')
        self.assertEqual(review.existing_Q({**state,'stores':[46,46]},4,5,index)['coverage'],'uncovered_in_index')

    def test_03_rehashed_wrong_metrics_and_source_pins_are_refused(self):
        for kind in ('metric','source'):
            body=json.loads(config.canonical(self.report))
            if kind=='metric':body['summary']['H_v1']['paired_mean_candidate_minus_control']+=1
            else:body['inputs']['congklak/ordinary_value_loss_review.py']='0'*64
            path=self.fixture/('corrupt-'+kind+'.json');path.write_text(config.canonical(body)+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'differs'):review.verify(path)

    def test_04_completed_rebuild_API_CLI_no_search_and_immutable_publication(self):
        before=(review.OUTPUT.read_bytes(),review.OUTPUT.stat().st_mtime_ns)
        with review.no_search():
            self.assertEqual(review.run(),self.report)
            fresh=self.fixture/'rebuilt.json';self.assertEqual(review.run(fresh),self.report)
            self.assertEqual(review.verify(fresh),self.report)
        process=subprocess.run([sys.executable,str(review.CLI),'verify'],cwd=ROOT,text=True,capture_output=True)
        (self.fixture/'completed-cli.stdout.log').write_text(process.stdout,encoding='utf-8');(self.fixture/'completed-cli.stderr.log').write_text(process.stderr,encoding='utf-8')
        self.assertEqual(process.returncode,0,process.stderr);self.assertEqual(before,(review.OUTPUT.read_bytes(),review.OUTPUT.stat().st_mtime_ns));self.assertNotIn('torch',sys.modules)
        self.assertEqual(self.report['work']['new_PUCT_searches'],0);self.assertFalse(self.report['work']['new_forward_timing_measured'])
        self.assertLessEqual(self.report['counts']['literal_model_forwards'],20)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(OrdinaryValueLossReviewAuthor);covered=[test.id() for test in suite]
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    paths=(Path(__file__),Path(review.__file__),review.CLI,review.common.BASELINE,review.common.SCOPE,review.OUTPUT)
    body=dict(schema='ordinary-value-loss-review-author-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),
        methods=unittest.defaultTestLoader.getTestCaseNames(OrdinaryValueLossReviewAuthor),covered_methods=covered,inputs={**OrdinaryValueLossReviewAuthor.report['inputs'],**review.common.pins(paths)},retained_directory=OrdinaryValueLossReviewAuthor.fixture.relative_to(ROOT).as_posix())
    from congklak.deeper_teacher_v2 import atomic_json
    atomic_json(ROOT/'build/ordinary-value-loss-review-author.json',body)
    raise SystemExit(not result.wasSuccessful())
