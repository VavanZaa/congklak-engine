"""Pure graph tests; no engine, data writer, model, or teacher is constructed."""
import unittest
from congklak.prospective_eval_reservations import close_units


def component(blocked=False):
    return dict(evaluation_blocked=blocked,acquired_origins_optimizer_blocked=not blocked)


def q(index):return (bytes(7)+bytes([25+index])+bytes(6)+b'\x00').hex()


class ProspectiveClosureTests(unittest.TestCase):
    def test_bridge_annexes_entire_unused_component(self):
        result=close_units({'fit':component(),'unused':component(),'arena':component(),'support':component(True)},
            {'f':'fit','a':'arena','t':'support'},{'f':'fit','a':'arena','t':'optimizer_support'},
            [dict(origin='f',quotients=[q(0)])],{q(0):'unused',q(1):'unused'})
        mapping=result['prior_to_prospective'];self.assertEqual(mapping['fit'],mapping['unused'])
        self.assertNotEqual(mapping['fit'],mapping['arena']);self.assertNotEqual(mapping['fit'],mapping['support'])
        self.assertEqual(result['components'][mapping['fit']]['prior_components'],['fit','unused'])
        self.assertEqual(result['new_quotient_components'],{})

    def test_selected_units_cannot_merge_even_in_same_role(self):
        with self.assertRaisesRegex(ValueError,'separate selected'):
            close_units({'a':component(),'b':component()},{'a':'a','b':'b'},{'a':'fit','b':'fit'},
                [dict(origin='a',quotients=[q(0)])],{q(0):'b'})

    def test_fit_arena_bridge_holds(self):
        with self.assertRaisesRegex(ValueError,'separate selected'):
            close_units({'fit':component(),'arena':component()},{'f':'fit','a':'arena'},{'f':'fit','a':'arena'},
                [dict(origin='f',quotients=[q(0)])],{q(0):'arena'})

    def test_protected_unused_prior_component_holds(self):
        with self.assertRaisesRegex(ValueError,'protected prior'):
            close_units({'fit':component(),'quarantine':component(True)},{'f':'fit'},{'f':'fit'},
                [dict(origin='f',quotients=[q(0)])],{q(0):'quarantine'})

    def test_dynamic_intersection_closes_both_units_and_holds(self):
        with self.assertRaisesRegex(ValueError,'separate selected'):
            close_units({'a':component(),'b':component()},{'a':'a','b':'b'},{'a':'fit','b':'arena'},
                [dict(origin='a',quotients=[q(0)]),dict(origin='b',quotients=[q(0)])],{})

    def test_same_origin_multiple_attempts_and_unknown_keys_are_retained(self):
        result=close_units({'a':component()},{'a':'a'},{'a':'fit'},
            [dict(origin='a',quotients=[q(0)]),dict(origin='a',quotients=[q(0),q(1)])],{})
        cid=result['origin_components']['a']
        self.assertEqual(result['new_quotient_components'],{q(0):cid,q(1):cid})
        self.assertEqual(result['components'][cid]['new_quotients'],[q(0),q(1)])


if __name__=='__main__':unittest.main()
