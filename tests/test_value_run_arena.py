"""Arena contracts: full schedule, literal transitions, alias bridges, bootstrap."""
from collections import Counter
from pathlib import Path
import unittest
from unittest.mock import patch
from congklak import config,value_run_arena as arena,value_run_resources as resources


class ArenaContracts(unittest.TestCase):
    def test_all192_rotated_assignments(self):
        rows=arena.read(arena.RESERVATIONS/'selection.json')['arena'];order=arena.schedule(rows)
        self.assertEqual(len(order),192)
        for i in range(24):
            units=[u for u in order if u['family_index']==i]
            self.assertEqual(len(units),8)
            self.assertEqual({(u['profile'],u['baseline'],u['seat']) for u in units},
                {(p,b,s) for p in arena.PROFILES for b in arena.BASELINES for s in (0,1)})

    def test_literal_transitions_conserve_and_never_increase_active(self):
        ref=arena._literal();rows=arena.read(arena.RESERVATIONS/'selection.json')['arena']
        transitions=0
        for row in rows:
            normal=arena.normalize(row['state'],ref)
            for action in normal['legal']:
                t=arena.transition(normal['state'],action,ref);after=t['after']
                self.assertEqual(sum(map(sum,after['pits']))+sum(after['stores']),98)
                self.assertLessEqual(sum(map(sum,after['pits'])),sum(map(sum,row['state']['pits'])))
                self.assertEqual(arena.normalize(after,ref)['state'],after)
                transitions+=1
        self.assertGreater(transitions,48)

    def test_full_historical_component_bridge(self):
        rows=arena.read(arena.RESERVATIONS/'selection.json')['arena'][:2]
        p0,p1=rows[0]['state'],rows[1]['state'];q0=arena.metadata.quotient(p0);q1=arena.metadata.quotient(p1)
        self.assertNotEqual(q0,q1)
        inv=dict(components={'foreign-component':dict(quotients=[q0,q1],families=[]),
             'arena0':dict(quotients=[],families=['f0']),'arena1':dict(quotients=[],families=['f1'])})
        selection=dict(train=[],fit_check=[])
        identity=dict(schedule=[dict(index=i,family_index=i,component='arena'+str(i),family='f'+str(i),start=p) for i,p in enumerate((p0,p1))])
        games=[dict(index=0),dict(index=1)]
        with patch.object(arena,'read',side_effect=lambda p:inv if Path(p).name=='inventory.json' else selection):
            result=arena.ownership(games,[],identity)
        self.assertFalse(result['ownership_passed'])
        self.assertEqual(result['cross_arena_alias_component_hits'],[dict(historical_component='foreign-component',family_indices=[0,1])])
        self.assertFalse(result['cross_arena_component_hits'])

    def test_whole_cluster_bootstrap_known_constant(self):
        rows=arena.read(arena.RESERVATIONS/'selection.json')['arena'];schedule=arena.schedule(rows)
        games=[dict(index=u['index'],terminal=True,capped=False,fallbacks=0,model_margin=2 if u['profile']=='control' else 3) for u in schedule]
        prefix=dict(identity=dict(schedule=schedule,fixture=False),games=games,decisions=[])
        with patch.object(arena,'ownership',return_value=dict(ownership_passed=True)):
            result=arena.summarize(prefix)
        self.assertEqual(len(result['cluster_statistics']),6)
        for baseline in arena.BASELINES:
            self.assertEqual(result['cluster_statistics']['candidate-minus-control/'+baseline]['interval95'],[1.,1.])
            self.assertEqual(result['cluster_statistics']['candidate/'+baseline]['interval95'],[3.,3.])
        self.assertTrue(result['cluster_analysis_valid'])
        with patch.object(arena,'ownership',return_value=dict(ownership_passed=False)):
            held=arena.summarize(prefix)
        self.assertFalse(held['cluster_analysis_valid']);self.assertEqual(held['cluster_statistics'],{})

    def test_cumulative_resource_protocol_and_shared_namespace(self):
        self.assertEqual(resources.PROTOCOL['lease_ns'],300_000_000_000)
        self.assertEqual(resources.PROTOCOL['cumulative_ns'],5_400_000_000_000)
        self.assertEqual(resources.PROTOCOL['quota_bytes'],5*1024**3)
        paths=[arena.ROOT/p for p in ('data/value-training-v1','build/value-training-v1','data/value-run-v1','build/value-run-v1')]
        actual=sum(p.stat().st_size for root in paths if root.exists() for p in root.rglob('*') if p.is_file())
        self.assertEqual(resources.namespace_bytes(),actual)


if __name__=='__main__':unittest.main()
