"""Pure metadata/ownership acceptance; no positions, labels or searches made."""
import copy,unittest
from collections import Counter
from unittest.mock import patch
from congklak import config,footprint_screen_data as data,footprint_screen_reservations as r

def q(active,location=0,actor=0):
    ring=bytearray(16);ring[location]=active;ring[7]=98-active
    return r.quotient(bytes(ring),actor).hex()

class FootprintScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool=data.expected_pool();cls.prior=r.prior_closure()

    def test_pool_is_fixed512_existing_components_and_four_support_cells(self):
        self.assertEqual(Counter(row['cell'] for row in self.pool['roots']),Counter({cell:64 for cell in data.CELLS}))
        self.assertEqual(len({row['source_component'] for row in self.pool['roots']}),512)
        self.assertEqual(self.pool['training_support_cells'],dict(zip(data.CELLS[:4],(39,34,58,61))))
        self.assertEqual(len(self.pool['metadata_dispositions']),16384)
        self.assertFalse(self.pool['training_authorized']);self.assertFalse(self.pool['quality_values_read'])
        self.assertTrue(all(row['family'] in self.prior['family_components'] for row in self.pool['roots']))

    def test_deterministic_rank_and_round_robin_no_root_quality_fields(self):
        for index,row in enumerate(self.pool['roots']):
            self.assertEqual(row['index'],index);self.assertEqual(row['cell'],data.CELLS[index%8])
            self.assertNotIn('target',row);self.assertNotIn('Q',row);self.assertNotIn('value',row)
        changed=copy.deepcopy(self.pool);changed['quality_values_read']=0
        with self.assertRaisesRegex(ValueError,'derivation differs'):data.validate_pool(changed)

    def test_cache_returns_independent_copy_and_validators_need_no_producer_writer(self):
        first=data.expected_pool();first['roots'][0]['cell']='tampered'
        self.assertEqual(data.expected_pool()['roots'][0]['cell'],data.CELLS[0])
        with patch.object(data.s,'atomic',side_effect=AssertionError('writer opened')):
            self.assertTrue(data.validate_pool(self.pool))

    def test_full_old_rejected6669_not_only_the_offending_alias(self):
        rows=self.prior['previous_actual_footprints']
        self.assertEqual([len(row['quotients']) for row in rows],( [16,7,6194,6,6669]))
        self.assertEqual(sum(row['rejected_attempt'] for row in rows),1)
        failed=rows[-1]
        for key in failed['quotients']:
            cid=self.prior['quotient_components'][key]
            self.assertTrue(self.prior['components'][cid]['evaluation_blocked'])
        self.assertIn('010002000103050101010007030000',failed['quotients'])

    def test_shared_low_is_never_a_connector_upper_domain_always_is(self):
        for active in (25,74,75,98):self.assertEqual(r.high_key(q(active)),q(active))
        with self.assertRaisesRegex(ValueError,'Shared/noncanonical'):r.high_key(q(24))
        ring=bytes([2,3,0,0,0,0,0,21,0,4,0,0,0,0,0,68])
        other=ring[8:]+ring[:8]
        self.assertEqual(r.quotient(ring,0),r.quotient(other,1))
        stores_changed=bytearray(ring);stores_changed[7]+=1;stores_changed[15]-=1
        self.assertEqual(r.quotient(ring,0),r.quotient(bytes(stores_changed),0))

    def test_capped_root_full_trace_bridges_another_root_to_protected_component(self):
        a,b,c=q(25),q(75),q(98)
        prior=dict(components={'safe':dict(evaluation_blocked=False),'blocked':dict(evaluation_blocked=True)},
            family_components={'rootA':'safe'},quotient_components={a:'safe',c:'blocked'})
        records=[(dict(family='rootA',origin='true/A'),dict(quotients=[a,b],teacher_complete=True)),
                 (dict(family='cappedB',origin='true/B'),dict(quotients=[b,c],teacher_complete=False))]
        closed=r._acquired_graph(prior,records)
        cid=closed['family_components']['rootA']
        self.assertEqual(cid,closed['family_components']['cappedB'])
        self.assertTrue(closed['components'][cid]['evaluation_blocked'])
        self.assertEqual(closed['components'][cid]['blocking_prior_components'],['blocked'])

    def test_acceptance_constraints_dont_invent_edges_between_unconnected_components(self):
        a,b=q(25),q(75)
        prior=dict(components={'A':dict(evaluation_blocked=False),'B':dict(evaluation_blocked=True)},
            family_components={'a':'A','b':'B'},quotient_components={a:'A',b:'B'})
        closed=r._acquired_graph(prior,[(dict(family='a',origin='source-a'),dict(quotients=[a]))])
        self.assertNotEqual(closed['family_components']['a'],closed['family_components']['b'])
        self.assertFalse(closed['components'][closed['family_components']['a']]['evaluation_blocked'])

    def test_low_key_hidden_inside_acquired_footprint_is_refused(self):
        prior=dict(components={},family_components={},quotient_components={})
        with self.assertRaisesRegex(ValueError,'Shared/noncanonical'):
            r._acquired_graph(prior,[(dict(family='a',origin='source-a'),dict(quotients=[q(24)]))])

    def test_global_matching_recovers_perfect_assignment_where_greedy_loses(self):
        eligible=[];components={}
        for cell,count in (('x',32),('y',16)):
            for i in range(count):
                key=f'{cell}/{i}';eligible.append(dict(id=key,cell=cell));components[key]=f'component/{i}'
        slots,assigned,choices=r._matching(eligible,components,('x','y'))
        greedy=set()
        for slot in slots:
            available=[cid for cid in choices[slot] if cid not in greedy]
            if available:greedy.add(available[0])
        self.assertLess(len(greedy),32);self.assertEqual(len(assigned),32)
        self.assertEqual(len(set(assigned.values())),32)
        self.assertEqual(r._matching_shortage_witness(slots,assigned,choices)['deficiency'],0)

    def test_shared_component_count_shortage_has_hall_witness(self):
        eligible=[dict(id=str(i),cell='ordinary/test') for i in range(21)]
        components={row['id']:row['id'] for row in eligible}
        slots,assigned,choices=r._matching(eligible,components,('ordinary/test',))
        # '/ordinary/' is the exact role-cell marker; use the real cell below.
        cell=data.CELLS[0];eligible=[dict(row,cell=cell) for row in eligible]
        slots,assigned,choices=r._matching(eligible,components,(cell,))
        witness=r._matching_shortage_witness(slots,assigned,choices)
        self.assertEqual(len(slots),22);self.assertEqual(len(assigned),21)
        self.assertEqual(witness['deficiency'],1);self.assertEqual(witness['unfilled_slots'],1)

    def test_publication_scope_cannot_turn_engineering_fixture_into_admission(self):
        path=r.FIXTURE_BASE/'before_catalog'
        self.assertEqual(r.scope(path,True),(path,True))
        with self.assertRaisesRegex(ValueError,'treatment changed'):r.scope(path,False)
        with self.assertRaisesRegex(ValueError,'Unregistered'):r.scope(r.FIXTURE_BASE/'arbitrary',True)

    def test_real_review_wrapper_shape_requires_catalog_and_exact_snapshot(self):
        from congklak import footprint_screen_review as review
        snapshot=dict(inputs={'exact-source':'same-sha'},roots=[],schema='unit envelope')
        report=dict(schema=r.s.SCHEMA+'-independent-raw-review-report',passed=True,
            independent_of_implementation=True,acquired_roots=512,production_query_count=1024,
            pool_sha256='unit-pool-sha',snapshot_sha256=config.digest(snapshot),source_inputs=snapshot['inputs'])
        receipt=dict(report=report,report_ref=dict(path='unit-report',sha256='r'),
            catalog_ref=dict(path='unit-catalog',sha256='c'),publication_gap=False)
        with patch.object(review,'completed',return_value=receipt),patch.object(r,'sha',return_value='unit-pool-sha'):
            self.assertTrue(r._audit(snapshot,receipt))
            with self.assertRaisesRegex(ValueError,'receipt differs'):r._audit(snapshot,report)
        gap=dict(receipt,publication_gap=True,catalog_ref=None)
        with patch.object(review,'completed',return_value=gap),patch.object(r,'sha',return_value='unit-pool-sha'):
            with self.assertRaisesRegex(ValueError,'complete current acquisition'):r._audit(snapshot,gap)

    def test_public_completed_run_shortcut_opens_no_resource_lease(self):
        from congklak import footprint_screen_resources as resources
        result=dict(report=dict(reservation_admission=False),publication_gap=False)
        with patch.object(r,'has_completion',return_value=True),patch.object(r,'publish_completed',return_value=result),\
             patch.object(resources,'bind_worker',side_effect=AssertionError('lease bound')):
            self.assertEqual(r.main(['run','--output',str(r.DEFAULT)]),result)

if __name__=='__main__':unittest.main()
