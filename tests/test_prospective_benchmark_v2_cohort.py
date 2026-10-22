"""Hand-authored structural fixtures; no prospective population draws."""
import unittest
from congklak import prospective_benchmark_v2_cohort as cohort


def boundary():
    return dict(pits=[[0,0,0,0,0,5,7],[0,0,0,0,0,6,7]],stores=[36,37],controller=0)


class StructuralCohortTests(unittest.TestCase):
    def test_all_legal_actions_of_both_actors_enter_shared_low(self):
        certificate=cohort.certify(boundary())
        self.assertTrue(certificate['universal_legal_frontier'])
        self.assertEqual({row['actor'] for row in certificate['probes']},{0,1})
        self.assertEqual(len(certificate['probes']),4)
        self.assertTrue(all(row['child_active']<=24 for row in certificate['probes']))
        self.assertEqual(len(certificate['root_actor_states']),2)

    def test_nonboundary_root_cannot_borrow_boundary_proof(self):
        state=boundary();state['pits'][0][6]+=1;state['stores'][0]-=1
        with self.assertRaisesRegex(ValueError,'exactly25'):cohort.certify(state)

    def test_quotient_preserves_player_exchange_and_ignores_store_split(self):
        state=boundary();opposite=dict(pits=state['pits'][::-1],stores=state['stores'][::-1],controller=1)
        self.assertEqual(cohort.quotient(state),cohort.quotient(opposite))
        shifted=dict(state,stores=[35,38]);self.assertEqual(cohort.quotient(state),cohort.quotient(shifted))

    def test_optional_cap_is_a_failed_certificate(self):
        state=boundary();state['pits'][0][6]+=1;state['stores'][0]-=1
        result=cohort.certify_high_closure(state,lambda q:None,cap=1)
        self.assertFalse(result['complete']);self.assertEqual(result['reason'],'high-state cap')
        self.assertEqual(len(result['states']),1)

    def test_optional_protected_root_retains_conflict_and_cannot_pass(self):
        result=cohort.certify_high_closure(boundary(),lambda q:dict(component='protected'))
        self.assertFalse(result['complete']);self.assertEqual(result['reason'],'protected prior high state')
        self.assertEqual(len(result['states']),1);self.assertEqual(len(result['conflicts']),1)


if __name__=='__main__':unittest.main()
