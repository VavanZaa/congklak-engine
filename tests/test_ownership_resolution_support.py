"""Ownership constraints and sampler invariants; no positions are generated."""
import copy
from fractions import Fraction
import unittest
from unittest.mock import patch
from congklak import ownership_resolution_support as support


class SupportResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base,cls.pool,cls.giant=support._base()
        cls.safe=[r for r in cls.pool if r['eligible']]

    def component(self,row,blocked=False):
        return dict(families=[row['family']],origins=[],quotients=[row['quotient']],
            optimizer_blocked=blocked,fit_blocked=True,arena_blocked=True,
            ancestral_training=True,acquisition_exposed=False)

    def test_four_cell_rational_weights_have_exact_fixed_mass(self):
        result=support._weights(self.safe)
        self.assertTrue(result['available']);self.assertEqual(len(result['weights']),192)
        sums={cell:sum((Fraction(r['numerator'],r['denominator']) for r in result['weights'] if r['cell']==cell),Fraction())
              for cell in support.ORDINARY_CELLS}
        self.assertEqual(set(sums.values()),{Fraction(1,4)})
        self.assertFalse(result['component_independence_claim'])

    def test_missing_ordinary_cell_holds_without_renormalizing(self):
        rows=[r for r in self.safe if r['cell']!=support.ORDINARY_CELLS[0]]
        result=support._weights(rows)
        self.assertFalse(result['available']);self.assertEqual(result['weights'],[])
        self.assertEqual(result['missing_cells'],[support.ORDINARY_CELLS[0]])

    def test_declared_constraints_propagate_whole_quarantine_and_all_flags(self):
        a,b=self.safe[:2]
        overlay=dict(components={'protected':self.component(a,True),'safe':self.component(b)})
        selection=dict(components={'accepted-population-constraint':dict(
            families=[],quotients=[a['quotient'],b['quotient']])})
        graph=support._joint_graph(overlay,selection)
        cid=graph['family_components'][a['family']]
        self.assertEqual(cid,graph['family_components'][b['family']])
        self.assertEqual(graph['components'][cid]['known_components'],['protected','safe'])
        self.assertTrue(graph['components'][cid]['optimizer_blocked'])
        self.assertEqual(graph['components'][cid]['population_components'],['accepted-population-constraint'])

    def test_fit_protection_of_authorized_training_is_not_an_optimizer_ban(self):
        row=self.safe[0];graph=support._joint_graph(dict(components={'train':self.component(row)}),dict(components={}))
        c=graph['components'][graph['family_components'][row['family']]]
        self.assertTrue(c['fit_blocked']);self.assertTrue(c['arena_blocked']);self.assertFalse(c['optimizer_blocked'])

    def test_component_flag_integer_alias_is_refused(self):
        row=self.safe[0];c=self.component(row);c['optimizer_blocked']=0
        with self.assertRaisesRegex(ValueError,'must be boolean'):
            support._joint_graph(dict(components={'train':c}),dict(components={}))

    def test_actual_population_protection_ledger_preserves_count_types(self):
        selection=support.read(support.SELECTION);support._validate_population(selection)
        selection['target_values_read']=False
        with self.assertRaisesRegex(ValueError,'acquisition/protection ledger'):
            support._validate_population(selection)

    def test_low_state_cannot_be_used_as_a_component_connector(self):
        row=self.safe[0];overlay=dict(components={'train':self.component(row)})
        selection=dict(components={'bad-low-constraint':dict(families=[],quotients=['00'*15])})
        with self.assertRaisesRegex(ValueError,'Shared low key'):support._joint_graph(overlay,selection)

    def test_overlay_cannot_split_old_origin_or_erase_optimizer_marker(self):
        a,b=self.safe[:2];c=self.component(a,True);c['families'].append(b['family'])
        base=dict(components={'old':c})
        overlay=dict(ancestry_unknown=True,historical_provenance_complete=False,source_complete_for_declared_persisted=True,
            components={'new':copy.deepcopy(c)},family_components={a['family']:'new',b['family']:'new'},
            quotient_components={a['quotient']:'new'})
        support._validate_overlay(base,overlay)
        overlay['components']['new']['optimizer_blocked']=False
        with self.assertRaisesRegex(ValueError,'erased'):support._validate_overlay(base,overlay)
        overlay['components']['new']['optimizer_blocked']=True;overlay['family_components'][b['family']]='separate'
        with self.assertRaisesRegex(ValueError,'split/dropped'):support._validate_overlay(base,overlay)

    def test_actual_320_paths_and_independent_validator_do_not_call_producer(self):
        result=support.inventory();self.assertEqual(result['causal_paths_complete'],320)
        self.assertEqual(result['projected_before_overlay'],192);self.assertEqual(result['original_pool_roots'],512)
        self.assertFalse(result['original_balanced_recipe_available']);self.assertFalse(result['training_authorized'])
        for row in result['original_512_dispositions']:
            if row['old_eligible']:continue
            path=row['causal_path'];proofs=path['actual_incidence_witnesses']
            self.assertEqual([p['incidence']['family'] for p in proofs],[row['family'],row['family'],path['nodes'][-1][1]])
            self.assertEqual(proofs[0]['incidence']['raw_state_key'],row['state_key'])
            self.assertEqual([p['incidence']['quotient'] for p in proofs],[row['quotient'],support.OPENING,support.OPENING])
            self.assertTrue(path['protected_role_marker']['marker']['optimizer_blocked'])
        with patch.object(support,'inventory',side_effect=AssertionError('producer called by validator')):
            self.assertTrue(support.validate_inventory(result))
            bad=copy.deepcopy(result);bad['training_authorized']=0
            with self.assertRaisesRegex(ValueError,'differs from immutable'):
                support.validate_inventory(bad)
        # Returned dictionaries cannot poison the immutable in-process cache.
        result['original_512_dispositions'][0]['disposition']='forged'
        self.assertNotEqual(support.inventory()['original_512_dispositions'][0]['disposition'],'forged')


if __name__=='__main__':unittest.main()
