"""Matcher-only scratch fixtures: the population constructor is never called."""
import copy
import unittest
from unittest.mock import patch
from congklak import prospective_reservations as r


def fixtures():
    out=[]
    for cell_index,cell in enumerate(r.population.CELLS):
        actor=int(cell[-1]);active=50 if cell.startswith('50') else 25
        for index in range(20):
            pits=[[0]*7,[0]*7];first=cell_index%7;second=(first+1)%7
            pits[actor][first]=3+index%6;pits[actor][second]=2+index//6
            pits[1-actor][(first+3)%7]=active-sum(pits[actor])
            state=dict(pits=pits,stores=[0,98-active],controller=actor)
            out.append(dict(counter=len(out),cell=cell,family='scratch/'+str(cell_index)+'/'+str(index),
                state=state,state_key=r.common.rawkey(state),quotient=r.provenance.metadata.quotient(state),
                active_seeds=active,probes=[],metadata_eligible=True,rejection_reasons=[],
                native_action_probes=0,native_normalizations=0))
    return out


class ProspectiveMatcherAuthor(unittest.TestCase):
    def setUp(self):
        self.records=fixtures();self.prior=dict(quotient_components={},support_roots_with_genuine_exclusion=320)

    def test_01_exact_joint_roles_cases_and_deterministic_order_without_draw(self):
        with (patch.object(r.population,'candidate',side_effect=AssertionError('forbidden population draw')) as draw,
              patch.object(r,'Native',side_effect=AssertionError('forbidden native classifier')) as native):
            result=r.select(self.records,self.prior,'f'*64)
            reordered=r.select(list(reversed(self.records)),self.prior,'f'*64)
        draw.assert_not_called();native.assert_not_called()
        self.assertTrue(result['metadata_reservations_complete'])
        self.assertEqual(result['roles'],reordered['roles'])
        self.assertEqual([len(result['roles'][k]) for k in ('fit','arena','cases')],[128,24,32])
        self.assertEqual(len(result['selected_components']),152)
        for cell in r.population.CELLS:
            fit=[row for row in result['roles']['fit'] if row['cell']==cell]
            cases=[row for row in result['roles']['cases'] if row['cell']==cell]
            self.assertEqual(len(fit),16);self.assertEqual(len(cases),4)
            self.assertEqual({row['state_key'] for row in cases},
                             {row['state_key'] for row in sorted(fit,key=lambda row:r.rank('fit',row))[:4]})
        self.assertFalse(result['training_admission'])
        self.assertEqual(result['prior_support_roots_with_genuine_exclusion'],320)

    def test_02_same_high_component_shortage_gives_no_partial_roles(self):
        records=copy.deepcopy(self.records)
        # Capture actor1 needs all20 components: sharing two roots' quotient
        # leaves19 and must not be repaired by a partial or smaller fit panel.
        records[-1].update(state=records[-2]['state'],state_key=records[-2]['state_key'],
                           quotient=records[-2]['quotient'])
        result=r.select(records,self.prior,'f'*64)
        self.assertFalse(result['metadata_reservations_complete'])
        self.assertEqual(result['status'],'HOLD_quota_shortage')
        self.assertEqual(result['roles'],dict(arena=[],fit=[],cases=[]))

    def test_03_shared_low_probes_do_not_merge_high_components(self):
        low=dict(pits=[[2,0,0,0,0,0,0],[3,0,0,0,0,0,0]],stores=[0,93],controller=0)
        q=r.provenance.metadata.quotient(low)
        for row in self.records:row['probes']=[dict(quotient=q,active_seeds=5)]
        result=r.select(self.records,self.prior,'f'*64)
        self.assertTrue(result['metadata_reservations_complete'])
        self.assertEqual(result['shared_observation_incidences'],160)
        self.assertEqual(len(result['components']),160)

    def test_04_transitive_prior_component_bridge_is_retained_and_excluded(self):
        self.prior['quotient_components']={self.records[-1]['quotient']:'known',self.records[-2]['quotient']:'known'}
        result=r.select(self.records,self.prior,'f'*64)
        linked=[c for c in result['components'].values() if c['prior_components']]
        self.assertEqual(len(linked),1)
        self.assertEqual(len(linked[0]['families']),2)
        self.assertFalse(result['metadata_reservations_complete'])
        self.assertEqual(result['rejected_reason_counts']['known high-domain component'],2)


if __name__=='__main__':unittest.main()
