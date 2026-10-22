"""Literal whole-component annex transactions without engine queries."""
import hashlib
from types import SimpleNamespace
import unittest
from congklak import prospective_eval_annex as annex, prospective_eval_guard as guard


def q(index):return bytes(7)+bytes([25+index])+bytes(6)+b'\x00'
def owned(index):
    key=q(index);cells=key[:7]+bytes([73-index])+key[7:14]+b'\x00'
    return guard.OWNED.pack(key,0,cells,0)


class Batch:
    pass


def fixture():
    records=b''.join(guard.STATIC.pack(q(index),ordinal) for index,ordinal in enumerate((1,1,2,3)))
    batch=Batch();batch.records=records;batch.component_ordinals={'aux':1,'selected':2,'protected':3}
    batch.eligible_components=frozenset({'aux'})
    batch.annex_policy=dict(eligible_components=['aux'],complete_mapping=dict(path='fixture',sha256=hashlib.sha256(records).hexdigest()))
    lookup=guard.Lookup(records,batch.component_ordinals)
    loaded=guard.LoadedGuard({},dict(origin_ids={'A':1,'B':2}),records,lookup,{'aux','selected'},set(),None,batch)
    return loaded


class AnnexTests(unittest.TestCase):
    def test_projection_literal_numpy_equal_and_protected_denied(self):
        loaded=fixture()
        native=annex.projected_records(loaded.records,{1,2})
        self.assertEqual(native,annex.projected_records(loaded.records,{1,2},numpy=False))
        self.assertEqual([flag for _,flag in guard.STATIC.iter_unpack(native)],[0,0,0,2])

    def test_entire_component_claimed_from_one_observed_key(self):
        loaded=fixture();claims=guard.PackedClaims(guard.STATIC.pack(q(0),1),loaded.origin_ids)
        result=annex._derive(owned(0),loaded,'A',claims)
        self.assertEqual(result['components'],['aux']);self.assertEqual(result['keys_count'],1)
        self.assertEqual(result['delta_records'],guard.STATIC.pack(q(1),1))
        self.assertEqual(result['whole_component_proof'][0]['keys'],2)
        after=claims.merge_delta(result['delta_records'])
        self.assertEqual(result['canonical_claims_after'],after.sha256)
        # A later unit hitting the other key sees a live-owner violation,
        # despite that key having never occurred in the original query.
        violation=guard.reference_owned(owned(1),loaded,'B',after)
        _,flags,owner=guard.VIOLATION.unpack(violation)
        self.assertEqual(flags,guard.OTHER_LIVE_ORIGIN);self.assertEqual(owner,1)

    def test_repeated_component_event_has_empty_delta(self):
        loaded=fixture();claims=guard.PackedClaims(b''.join(guard.STATIC.pack(q(i),1) for i in (0,1)),loaded.origin_ids)
        result=annex._derive(owned(0),loaded,'A',claims)
        self.assertEqual(result['keys_count'],0);self.assertEqual(result['delta_records'],b'')
        self.assertEqual(result['canonical_claims_before'],result['canonical_claims_after'])

    def test_whole_component_conflicting_unobserved_key_holds(self):
        loaded=fixture();claims=guard.PackedClaims(guard.STATIC.pack(q(0),1)+guard.STATIC.pack(q(1),2),loaded.origin_ids)
        with self.assertRaisesRegex(ValueError,'already belongs'):
            annex._derive(owned(0),loaded,'A',claims)

    def test_annex_requires_exact_hit_commit_first(self):
        loaded=fixture();claims=guard.PackedClaims(b'',loaded.origin_ids)
        with self.assertRaisesRegex(ValueError,'published first'):
            annex._derive(owned(0),loaded,'A',claims)

    def test_no_touched_auxiliary_still_emits_empty_barrier_proof(self):
        loaded=fixture();claims=guard.PackedClaims(guard.STATIC.pack(q(2),1),loaded.origin_ids)
        result=annex._derive(owned(2),loaded,'A',claims)
        self.assertEqual(result['components'],[]);self.assertEqual(result['keys_count'],0)
        self.assertEqual(result['canonical_claims_before'],result['canonical_claims_after'])

    def test_protected_static_component_cannot_annex(self):
        loaded=fixture();claims=guard.PackedClaims(guard.STATIC.pack(q(3),1),loaded.origin_ids)
        with self.assertRaisesRegex(ValueError,'protected'):
            annex._derive(owned(3),loaded,'A',claims)


if __name__=='__main__':unittest.main()
