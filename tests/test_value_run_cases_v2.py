"""Meaningful fixed-pilot schedule, objective and clock guard checks."""
import copy
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
from congklak import value_run_cases_v2 as c


class ValueRunCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.panel=c.panel()

    def test_frozen_exposure_and_full_action_ties(self):
        tasks=c._tasks(self.panel)
        self.assertEqual(len(tasks),650)
        timed=[r for r in tasks if r['mode']=='timed']
        self.assertEqual(len(timed),512)
        self.assertEqual(len({(r['root']['state_key'],r['engine'],r['ms']) for r in timed}),512)
        self.assertEqual(sum(r['mode']=='fixed128' for r in tasks),138)
        for r in self.panel['fit_cases']+self.panel['exact']:
            ref=r['reference'];q={a['pit']:a['value_player0'] for a in ref['action_scores']}
            actor=r['state']['controller'];target=(max if actor==0 else min)(q.values())
            self.assertEqual(target,ref['target_player0'])
            self.assertEqual([a for a in q if q[a]==target],ref['best_moves'])

    def test_margins_and_huber_are_normalized_by_signed_active(self):
        values=[dict(huber=.001,normalized_error=.1,seed_error=4.),dict(huber=.003,normalized_error=-.2,seed_error=-6.)]
        got=c._aggregate(values)
        self.assertEqual(got['roots'],2)
        self.assertEqual(got['seed_mae'],5.)
        self.assertEqual(got['seed_bias'],-1.)
        self.assertEqual(got['normalized_huber'],.002)
        self.assertAlmostEqual(got['seed_rmse'],(26.)**.5)

    def test_regret_respects_actor_and_optimal_ties(self):
        r={'task':{'root':{'state':{'controller':1},'reference':{
            'action_scores':[{'pit':0,'value_player0':-6},{'pit':1,'value_player0':-6},{'pit':2,'value_player0':2}],
            'best_moves':[0,1],'target_player0':-6}}},'answer':{'decision':{'chosen':1}}}
        self.assertEqual(c._score(r),dict(optimal=True,regret=0))
        r['answer']['decision']['chosen']=2
        self.assertEqual(c._score(r),dict(optimal=False,regret=8))
        r['answer']['decision']['chosen']=None
        self.assertEqual(c._score(r),dict(optimal=None,regret=None))

    def test_zero_never_enters_unlimited_timed_search(self):
        for bad in (0,-1,1,24,101,25.0):
            with self.assertRaises(RuntimeError):c.timed_decision('control',None,{},bad)

    def test_completed_anchor_requires_certificate_before_writer_or_engine(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            (directory/'COMPLETED.json').write_text('{}')
            with patch.object(c,'_catalog',return_value=({'task_count':6},{'count':6})), \
                 patch.object(c,'_writer',side_effect=AssertionError('writer forbidden')), \
                 patch.object(c,'open_engines',side_effect=AssertionError('engine forbidden')):
                for operation in (c.completed,c.status,c.run):
                    with self.assertRaises(FileNotFoundError):operation(directory)


if __name__=='__main__':unittest.main()
