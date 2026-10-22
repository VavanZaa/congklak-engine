"""Four distinct read-only review checks; no games/search/optimizer work."""
from collections import defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,ordinary_selfplay_review as arena,ordinary_selfplay_sampling as sampling


class OrdinarySelfplayReviewAuthor(unittest.TestCase):
    def test_01_saved_matched_starts_counts_and_group_margins(self):
        report=arena.read(arena.ARENA_OUTPUT);self.assertEqual(report['counts']['games'],96);self.assertEqual(report['counts']['decisions'],2958)
        for name,digest in report['inputs'].items():self.assertEqual(arena.sha(ROOT/name),digest,name)
        with sqlite3.connect((ROOT/'data/selfplay-pilot-benchmark-v1/work.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
            games={}
            for text,digest in db.execute('SELECT body,hash FROM games'):
                game=json.loads(text);self.assertEqual(config.digest(game),digest);self.assertEqual(game['status'],'terminal');games[game['comparison'],game['pair'],game['candidate_seat']]=game
            self.assertEqual(db.execute('SELECT count(*) FROM decisions').fetchone()[0],2958)
        deltas=defaultdict(list)
        for seat in report['matched_seats']:
            pair,position=seat['pair'],seat['candidate_seat'];baseline=seat['baseline'];left=games['selfplay_control_vs_'+baseline,pair,position];right=games['selfplay_challenger_vs_'+baseline,pair,position]
            self.assertEqual(left['start'],right['start']);self.assertEqual(left['family'],right['family']);delta=right['candidate_final_margin']-left['candidate_final_margin'];self.assertEqual(delta,seat['challenger_minus_control_margin']);deltas[baseline].append(delta)
        self.assertEqual(len(report['matched_seats']),48)
        for name,values in deltas.items():self.assertAlmostEqual(sum(values)/len(values),report['aggregate_by_baseline'][name]['paired_mean_challenger_minus_control'])

    def test_02_exact_raw_state_orientation_and_actor_regret(self):
        state=dict(pits=[[1,2,0,0,0,0,0],[0,0,0,0,2,1,0]],stores=[45,47],controller=1)
        label=dict(kind='finite_choice_depth',choice_depth=8,target_player0=-3,best_moves=[4],action_scores=[dict(pit=4,value_player0=-3),dict(pit=5,value_player0=2)])
        index={config.canonical(state):[dict(state=state,label=label,source=dict(path='synthetic'))]}
        result=arena.existing_Q(state,4,5,index);self.assertEqual(result['comparisons'][0]['control_actor_regret'],0);self.assertEqual(result['comparisons'][0]['challenger_actor_regret'],5)
        swapped=dict(pits=list(reversed(state['pits'])),stores=list(reversed(state['stores'])),controller=0)
        self.assertEqual(arena.existing_Q(swapped,0,1,index)['coverage'],'uncovered')
        offset={**state,'stores':[46,46]};self.assertEqual(arena.existing_Q(offset,4,5,index)['coverage'],'uncovered')

    def test_03_probability_formulas_positive_support_and_unit_loss(self):
        rows=[dict(id=str(i),family=f,state_key=s,stratum=t,choice=0) for i,(f,s,t) in enumerate((('A','s1','low'),('A','s1','low'),('B','s2','high'),('B','s3','high'),('C','s4','high')))]
        expected={'observation_uniform':[Fraction(1,5)]*5,'family_uniform':[Fraction(1,6)]*4+[Fraction(1,3)],'raw_state_uniform':[Fraction(1,8)]*2+[Fraction(1,4)]*3,'stratum_uniform':[Fraction(1,4)]*2+[Fraction(1,6)]*3}
        expected['root90_stratum10']=[Fraction(9,10)*a+Fraction(1,10)*b for a,b in zip(expected['raw_state_uniform'],expected['stratum_uniform'])]
        for mode in sampling.MODES:
            actual=sampling.probabilities(rows,mode);self.assertAlmostEqual(math.fsum(actual),1)
            for a,b in zip(actual,expected[mode]):self.assertAlmostEqual(a,float(b))
            result=sampling.exposure(rows,mode);self.assertTrue(all(row['loss_weight']==1 for row in result['observation_exposures']));self.assertGreater(result['minimum_observation_probability'],0)
            self.assertAlmostEqual(result['observation_distribution_effective_support'],1/sum(float(v)**2 for v in expected[mode]))
        self.assertEqual(sampling.probabilities([],'family_uniform'),[]);self.assertIsNone(sampling.exposure([],'raw_state_uniform')['families']['distribution_effective_support'])
        with self.assertRaises(ValueError):sampling.probabilities(rows,'unregistered')

    def test_04_completed_arena_review_no_search_and_bytes_mtime_stable(self):
        path=arena.ARENA_OUTPUT;before=(path.read_bytes(),path.stat().st_mtime_ns)
        from congklak import quality_search
        from congklak.neural_relay import NeuralNative
        def prohibited(*_args,**_kwargs):raise AssertionError('Review attempted new search')
        with patch.object(arena.Native,'analyze',side_effect=prohibited),patch.object(NeuralNative,'analyze',side_effect=prohibited),patch.object(quality_search,'search',side_effect=prohibited),patch.object(arena.bench,'decide',side_effect=prohibited):
            body=arena.arena_main()
        self.assertEqual(body['new_searches'],0);self.assertEqual(body['new_games'],0);self.assertFalse(body['training_performed']);self.assertEqual(before,(path.read_bytes(),path.stat().st_mtime_ns));self.assertNotIn('torch',sys.modules)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OrdinarySelfplayReviewAuthor))
    paths=('congklak/ordinary_selfplay_review.py','tools/review-ordinary-selfplay-pilot.py','congklak/ordinary_selfplay_sampling.py','tools/review-ordinary-selfplay-sampling.py','tests/ordinary_selfplay_review_author.py','build/ordinary-selfplay-baseline.json','build/ordinary-selfplay-prior-arena-review.json')
    report=dict(schema='ordinary-selfplay-review-author-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),methods=unittest.defaultTestLoader.getTestCaseNames(OrdinarySelfplayReviewAuthor),inputs={name:arena.sha(ROOT/name) for name in paths})
    from congklak.deeper_teacher_v2 import atomic_json
    atomic_json(ROOT/'build/ordinary-selfplay-review-author.json',report)
    raise SystemExit(not result.wasSuccessful())
