"""Bounded pure admission/ownership checks; no game or model is constructed."""
import hashlib,json,os,struct,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
from congklak import config,guarded_value_pilot_v2_common as stage
from congklak import guarded_value_pilot_v2_data as data,guarded_value_pilot_v2_guard as guard

def q(seeds):return bytes([0]*13+[seeds,0])
def owned(key,index=0):
    cells=key[:7]+bytes([0])+key[7:14]+bytes([98-sum(key[:14])])
    return guard.OWNED.pack(key,index,cells,key[14])
def loaded():
    records=guard.STATIC.pack(q(30),1)+guard.STATIC.pack(q(31),2)
    return guard.LoadedGuard(dict(records_sha256='a'*64),dict(origin_ids={'a':1,'b':2},guard_stamp='b'*64),
        records,guard.Lookup(records,{'c1':1,'c2':2}),{'c1'},set(),{},None)

class OwnershipChecks(unittest.TestCase):
    def test_literal_full_component_lookup(self):
        g=loaded();self.assertEqual(g.graph_lookup.get(q(30).hex()),'c1')
        self.assertEqual(g.graph_lookup.get(q(31).hex()),'c2');self.assertIsNone(g.graph_lookup.get(q(32).hex()))
    def test_other_component_always_blocks(self):
        g=loaded();claims=guard.PackedClaims(b'',g.origin_ids)
        self.assertEqual(guard.reference_owned(owned(q(30)),g,'a',claims),b'')
        violation=guard.reference_owned(owned(q(31)),g,'a',claims)
        self.assertEqual(guard.VIOLATION.unpack(violation),(0,2,0))
    def test_unknown_key_is_claimable_only_by_same_live_origin(self):
        g=loaded();claims=guard.PackedClaims(guard.STATIC.pack(q(32),2),g.origin_ids)
        self.assertEqual(guard.VIOLATION.unpack(guard.reference_owned(owned(q(32)),g,'a',claims)),(0,4,2))
        self.assertEqual(guard.reference_owned(owned(q(32)),g,'b',claims),b'')
    def test_stage_hash_and_delta_match_literal_protocol(self):
        g=loaded();claims=guard.PackedClaims(b'',g.origin_ids);before=hashlib.sha256(b'').hexdigest()
        result=guard.stage_reference(owned(q(30)),g,'a',claims,before,1,0)
        delta=guard.STATIC.pack(q(30),1);delta_sha=hashlib.sha256(delta).digest()
        expected=hashlib.sha256(bytes.fromhex(before)+delta_sha+struct.pack('<QIQ',1,1,1)).hexdigest()
        self.assertEqual(result['delta_records'],delta);self.assertEqual(result['staged_claim_hash'],expected)
        self.assertEqual(result['guard_records_sha256'],'a'*64)
        self.assertEqual(claims.records,b'')
    def test_shared_low_cannot_be_owned_or_claimed(self):
        with self.assertRaises(ValueError):guard.PackedClaims(guard.STATIC.pack(q(24),1),{'a':1})
    def test_fabricated_batch_lacks_constructor_seal(self):
        batch=object.__new__(guard.AdmissionBatch)
        with self.assertRaises(ValueError):batch.assert_sealed()
    def test_durable_proof_cannot_point_at_old_namespace(self):
        with self.assertRaises(ValueError):guard._proof_path(str(stage.ROOT/'build/ordinary-epoch-pilot-v1/identity.json'))

class CacheChecks(unittest.TestCase):
    def test_source_view_cannot_be_fabricated(self):
        view=object.__new__(data._TrainSourceView)
        with self.assertRaises(ValueError):view.assert_sealed()

    def test_source_scope_uses_exact_bytes_and_detects_preserved_time_changes(self):
        stage.FOLDER.joinpath('fixtures').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='pure-source-view-',dir=stage.FOLDER/'fixtures') as folder:
            source=Path(folder)/'source.txt';cache=Path(folder)/'cache.txt'
            source.write_bytes(b'original');cache.write_bytes(b'cache')
            pins=data.common.pins([source,cache]);narrow={stage.relative(cache):pins[stage.relative(cache)]}
            rows=[dict(state_key=str(i),target_z=i/192) for i in range(192)];manifest=dict(literal_source=True)
            def verified_pins():data.common.current(pins);return pins,narrow
            data._TRAIN_VIEWS.clear()
            try:
                with patch.object(data,'_source_view_pins',side_effect=verified_pins),\
                     patch.object(data.prior_data,'load_train',return_value=(rows,manifest)) as derive,\
                     patch.object(data,'membership',return_value=dict(train=rows)):
                    first,_=data._train_rows();first[0]['target_z']=99
                    second,_=data._train_rows();self.assertEqual(second,rows);self.assertEqual(derive.call_count,1)
                    stamp=source.stat()
                    with self.assertRaises(ValueError):
                        with data.admission_scope():
                            source.write_bytes(b'modified');os.utime(source,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
                            # Full sources are scoped; narrow cache pins still check
                            # on each use. The mandatory closing full check refuses.
                            self.assertEqual(data._train_rows()[0],rows)
                    self.assertIsNone(data._ACTIVE_VIEW.get())
                    with self.assertRaises(ValueError):data._train_rows()
            finally:data._TRAIN_VIEWS.clear()

    def test_source_scope_checks_cache_on_every_reuse(self):
        stage.FOLDER.joinpath('fixtures').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='pure-source-cache-',dir=stage.FOLDER/'fixtures') as folder:
            cache=Path(folder)/'cache.txt';cache.write_bytes(b'original');pins=data.common.pins([cache])
            rows=[dict(state_key=str(i)) for i in range(192)]
            def verified_pins():data.common.current(pins);return pins,pins
            data._TRAIN_VIEWS.clear()
            try:
                with patch.object(data,'_source_view_pins',side_effect=verified_pins),\
                     patch.object(data.prior_data,'load_train',return_value=(rows,{})),\
                     patch.object(data,'membership',return_value=dict(train=rows)):
                    with self.assertRaises(ValueError):
                        with data.admission_scope():
                            stamp=cache.stat();cache.write_bytes(b'modified');os.utime(cache,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
                            with self.assertRaises(ValueError):data._train_rows()
            finally:data._TRAIN_VIEWS.clear()

    def test_train_payload_and_component_projection_stay_separate(self):
        stage.FOLDER.joinpath('fixtures').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='pure-data-',dir=stage.FOLDER/'fixtures') as folder:
            path=Path(folder);rows=[dict(state_key='raw',component='old',x=[.5],target_z=.25)]
            identity=dict(current_components={'raw':'new'},role='train')
            manifest=data._prepare(path,identity,rows,'train')
            self.assertEqual(manifest['current_components'],{'raw':'new'})
            self.assertEqual(data.read(path/'train.json')['rows'],rows)
            self.assertEqual(data._admit(path,identity,rows,'train'),manifest)
            (path/'train.json').write_text(json.dumps(dict(schema=data.SCHEMA+'-role',role='train',rows=[])))
            with self.assertRaises(ValueError):data._admit(path,identity,rows,'train')

if __name__=='__main__':unittest.main()
