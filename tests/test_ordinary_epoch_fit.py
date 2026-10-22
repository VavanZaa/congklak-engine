"""Numerical diagnostic invariants; synthetic numbers are not training data."""
import math,unittest
from unittest.mock import patch
from congklak import ordinary_epoch_fit as fit

class OrdinaryEpochFitTests(unittest.TestCase):
    def test_huber_definition_and_seed_units_are_separate(self):
        self.assertAlmostEqual(fit._huber(.1),.005)
        self.assertAlmostEqual(fit._huber(-.2),.015)
        rows=[dict(target_z=0.,target_player0=0,control=dict(z=.2,value_player0=10.))]
        m=fit._metrics(rows,'control')
        self.assertAlmostEqual(m['normalized_huber'],.015);self.assertEqual(m['seed_mae'],10.)
        self.assertEqual(m['seed_bias_player0'],10.);self.assertEqual(m['count'],1)

    def test_frozen_arithmetic_zero_branch_and_policy_unchanged(self):
        # This is a two-dimensional arithmetic fixture, never an engine input.
        w={'layers.0.weight':[[1.,0.],[0.,1.]],'layers.0.bias':[0.,0.],
           'layers.2.weight':[[1.,0.],[0.,1.]],'layers.2.bias':[0.,0.],
           'layers.4.weight':[[.25,.5]],'layers.4.bias':[.125],
           'policy.weight':[[1.,-1.]],'policy.bias':[.25]}
        u,p=fit._base([2.,3.],w);self.assertEqual(u,2.125);self.assertEqual(p,[-.75])
        b=dict(A=[[.5,.5]],b=[0.],v=[[0.]],c=[0.])
        self.assertEqual(fit._correction([2.,3.],b),0.)
        self.assertEqual(math.tanh(u),math.tanh(u+fit._correction([2.,3.],b)))

    def test_native_input_order_is_not_compensated_python_sum(self):
        self.assertEqual(fit._dot([1e16,1.,-1e16],[1.,1.,1.]),0.)

    def test_completed_route_is_first_and_creates_no_writer(self):
        with patch.object(fit,'completed',return_value={'passed':False}),\
             patch.object(fit,'_derive',side_effect=AssertionError('producer')),\
             patch.object(fit,'Writer',side_effect=AssertionError('writer')),\
             patch.object(fit.Path,'exists',return_value=True):
            self.assertEqual(fit.run(),{'passed':False})

if __name__=='__main__':unittest.main()
