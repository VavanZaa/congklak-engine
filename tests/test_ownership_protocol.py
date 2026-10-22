import unittest
from congklak import ownership_protocol as protocol, neurodiag_data as metadata


def state(a, actor=0):
    pits = [[0]*7, [0]*7]
    if a >= 2: pits[actor][:2] = [a-1, 1]
    elif a: pits[actor][0] = a
    return dict(pits=pits, stores=[98-a, 0], controller=actor)


def edge(family, a, roles=()):
    return dict(family=family, quotient=metadata.quotient(state(a)), roles=list(roles), source='scratch')


class OwnershipProtocolTests(unittest.TestCase):
    def test_boundary_seat_dispatch_and_leaf_contracts(self):
        self.assertTrue(protocol.fixture_checks()['passed'])
        self.assertEqual(protocol.fixture_checks()['dispatch_leaf_cases'], 160)
        self.assertTrue(protocol.root_admissible(state(25)))
        self.assertTrue(protocol.root_admissible(state(74)))
        self.assertFalse(protocol.root_admissible(state(24)))
        self.assertFalse(protocol.root_admissible(state(75)))

    def test_terminal_event_and_absorbing_suffix(self):
        terminal = protocol.dispatch(state(0), 'candidate', 'H_v1', 0)
        self.assertTrue(terminal['boundary_event_required'])
        self.assertFalse(terminal['decision_required'])
        with self.assertRaisesRegex(ValueError, 'absorbing'):
            protocol.dispatch(state(25), 'candidate', 'H_v1', 0, suffix_started=True)
        prior = protocol.dispatch(state(24), 'candidate', 'H_v1', 0, suffix_started=True)
        self.assertFalse(prior['handoff_required'])
        self.assertEqual(prior['engine'], 'control')

    def test_cut_actual_low_edges_preserves_families_and_upper_connections(self):
        low = protocol.graph_cut([edge('a',5),edge('b',5)])
        self.assertEqual(len(low['components']), 2)
        high = protocol.graph_cut([edge('a',25),edge('a',75),edge('b',75)])
        self.assertEqual(len(high['components']), 1)
        self.assertEqual(high['owned_edges'], 3)
        self.assertFalse(high['production_admission'])

    def test_cross_role_origin_protection_is_not_root_only(self):
        rows = [edge('origin',5,['protected']),edge('origin',25),edge('eval',25,['arena'])]
        result = protocol.graph_cut(rows)
        self.assertTrue(result['conflicting_components'])
        self.assertFalse(result['contract_graph_consistent'])

    def test_flattening_and_noncanonical_encoding_refuse(self):
        with self.assertRaisesRegex(ValueError, 'incidence'):
            protocol.graph_cut([dict(families=['a','b'],quotients=['00'*15])])
        row = edge('a',31); row['quotient'] = row['quotient'].upper()
        self.assertNotEqual(row['quotient'], row['quotient'].lower())
        with self.assertRaisesRegex(ValueError, 'geometry'):
            protocol.graph_cut([row])

    def test_missing_coverage_and_pretended_certificate_cannot_admit(self):
        incomplete = protocol.graph_cut([edge('a',25)])
        self.assertFalse(incomplete['source_coverage_sufficient'])
        coverage = dict(raw_incidence=True,declared_sources=['scratch'],verified_sources=['scratch'],
                        missing_sources=[],unresolved_unions=[])
        complete = protocol.graph_cut([edge('a',25)],coverage)
        self.assertTrue(complete['contract_graph_consistent'])
        self.assertFalse(complete['actual_source_certificate'])
        self.assertFalse(complete['production_admission'])

    def test_same_role_arena_origins_hold_and_single_origin_is_deduplicated(self):
        rows = [edge('a',25,['arena']),edge('b',25,['arena'])]
        self.assertTrue(protocol.graph_cut(rows)['conflicting_components'])
        for row in rows: row['arena_origin']='one_reserved_origin'
        self.assertFalse(protocol.graph_cut(rows)['conflicting_components'])

    def test_spec_and_no_implementation_authorization(self):
        c = protocol.contract()
        spec = dict(schema='prospective-ownership-section-spec-v1', machine_contract=c,
                    training_authorized=False,execution_ready=False,old_arena_hold_preserved=True)
        self.assertTrue(protocol.validate_spec(spec))
        spec['machine_contract']['boundary']['owned_min'] = 24
        with self.assertRaisesRegex(ValueError, 'spec differs'):
            protocol.validate_spec(spec)


if __name__ == '__main__': unittest.main()
