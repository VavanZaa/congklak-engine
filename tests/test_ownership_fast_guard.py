"""Adversarial pure ownership guards; no engines, training or new games."""
import hashlib
import json
from pathlib import Path
import struct
import unittest
import uuid
from congklak import config,ownership_fast_guard as guard


def state(active=40,left=10,actor=0,store0=7):
    # Different stores are deliberately preserved in the raw trace.
    return bytes([left,0,0,0,0,0,0,store0,active-left,0,0,0,0,0,0,98-active-store0]),actor


def loaded():
    cells,actor=state();q=guard.raw_key(cells,actor)
    upper,_=state(98,41,1,0);uq=guard.raw_key(upper,1)
    bad,_=state(75,30,0,0);bq=guard.raw_key(bad,0)
    graph=dict(components={'allow':{},'deny':{}},quotient_components={q.hex():'allow',uq.hex():'deny'})
    settings=dict(graph_path='fixture',allowed_components=['allow'],forbidden_quotients=[bq.hex()],engineering_consumed_only=True)
    records,flags,forbidden,allowed,shared=guard._static(settings,graph)
    refs=dict(records_sha256=hashlib.sha256(records).hexdigest(),guard_stamp='fixture')
    manifest=dict(origin_ids={'first':1,'second':2},guard_stamp='fixture')
    return guard.LoadedGuard(refs,manifest,records,flags,graph['quotient_components'],allowed,forbidden)


def footprint(cells,actor,index=0):
    return guard.OWNED.pack(guard.raw_key(cells,actor),index,cells,actor)


class Function:
    def __init__(self,callback):self.callback=callback
    def __call__(self,*args):return self.callback(*args)


class FakeLibrary:
    def __init__(self):
        self.published=[];self.restored=[];self.certificate={}
        self.cc_fast_guard_create=Function(lambda *a:19)
        self.cc_fast_guard_destroy=Function(lambda *a:None)
        self.cc_fast_guard_restore=Function(lambda handle,rows,count:self.restored.append(count) or 1)
        self.cc_fast_guard_update=Function(lambda *a:1)
        self.cc_fast_guard_restore_generation=Function(lambda *a:1)
        self.cc_fast_guard_publish_stage=Function(lambda handle,generation,owner:self.published.append((generation,owner)) or 1)
        self.cc_fast_guard_discard_stage=Function(lambda *a:None)
        self.cc_fast_guard_stage_certificate=Function(lambda *a:json.dumps(self.certificate).encode())


class FastGuardTests(unittest.TestCase):
    def test_store_shift_actor_swap_and_upper_states(self):
        a,actor=state(40,10,0,7);b,_=state(40,10,0,20)
        swapped=a[8:15]+a[15:16]+a[:7]+a[7:8]
        self.assertEqual(guard.raw_key(a,actor),guard.raw_key(b,actor))
        self.assertEqual(guard.raw_key(a,actor),guard.raw_key(swapped,1))
        self.assertNotEqual(a,b)
        g=loaded();self.assertIn(98,{sum(q[:14]) for q in g.flags})
        self.assertIn(75,{sum(q[:14]) for q in g.flags})

    def test_static_all_codes_and_live_origin(self):
        g=loaded();a,actor=state();row=footprint(a,actor);q=guard.raw_key(a,actor)
        for flags in range(4):
            g.allowed={'allow'} if not flags&2 else set()
            g.forbidden={q} if flags&1 else set()
            for other in (False,True):
                live={q.hex():'second' if other else 'first'}
                expected=flags|(4 if other else 0)
                result=guard.reference_owned(row,g,'first',live)
                self.assertEqual(result,b'' if expected==0 else guard.VIOLATION.pack(0,expected,2 if other else 0))

    def test_shared_raw_kept_and_full_metadata_validated(self):
        g=loaded();low,actor=state(24,5,0,20);high,_=state(98,41,1,0)
        raw=guard.EVENT.pack(low,actor,1,1,0,-1,-1)+guard.EVENT.pack(high,1,2,2,0,0,0)
        result=guard.reference_raw(raw,g,'first',{})
        self.assertEqual(result['event_count'],2);self.assertEqual(result['low_complete_events_retained'],1)
        self.assertEqual(result['owned_unique_count'],1);self.assertFalse(result['passed'])
        self.assertEqual(result['raw_sha256'],hashlib.sha256(raw).hexdigest())
        for bad in (guard.EVENT.pack(low,actor,1,1,1,-1,-1),guard.EVENT.pack(low,2,1,1,0,-1,-1),raw[:-1]):
            with self.assertRaises(ValueError):guard.reference_raw(bad,g,'first',{})

    def test_claims_and_footprint_forgeries_fail_closed(self):
        g=loaded();cells,actor=state();q=guard.raw_key(cells,actor);row=footprint(cells,actor)
        for claims in ({q:'first',q.hex():'second'},{q:'unknown'},{guard.raw_key(*state(24,5,0,20)):'first'}):
            with self.assertRaises(ValueError):guard.reference_owned(row,g,'first',claims)
        for bad in (row+row,row[:-1],footprint(cells,actor,3)):
            with self.assertRaises(ValueError):guard.owned_rows(bad,event_count=1)
        cells_bad=bytes([99])+bytes(15)
        with self.assertRaises(ValueError):guard.raw_key(cells_bad,0)
        with self.assertRaises(ValueError):guard.key(bytes([1])*14+bytes([2]))

    def test_stage_hash_and_no_failed_stage(self):
        g=loaded();cells,actor=state();row=footprint(cells,actor);q=guard.raw_key(cells,actor)
        previous=hashlib.sha256(b'').hexdigest();result=guard.stage_reference(row,g,'first',{},previous,3,1)
        delta=struct.pack('<15sI',q,1);dh=hashlib.sha256(delta).digest()
        expected=hashlib.sha256(bytes.fromhex(previous)+dh+struct.pack('<QIQ',3,1,1)).hexdigest()
        self.assertEqual(result['staged_claim_hash'],expected);self.assertEqual(result['delta_records'],delta)
        self.assertEqual(guard.stage_reference(row,g,'first',{q.hex():'first'},previous,4,3)['stage_new_count'],0)
        for claimed in ({q.hex():'second'},):
            with self.assertRaises(ValueError):guard.stage_reference(row,g,'first',claimed,previous,4,3)
        g.forbidden.add(q)
        with self.assertRaises(ValueError):guard.stage_reference(row,g,'first',{},previous,4,3)

    def test_publication_requires_actual_committed_proof(self):
        g=loaded();lib=FakeLibrary()
        # This test exercises publication proof logic without constructing a DLL.
        native=object.__new__(guard.NativeGuard);native.lib=lib;native.loaded=g;native.closed=False
        native.handle=19;native.run_identity=None;native.last_commit_index=-1;native.published_generation=0
        folder=guard.DEFAULT/'author-fixtures'/uuid.uuid4().hex;identity=folder/'identity.json'
        guard._publish(identity,(config.canonical({'guard_refs':g.refs})+'\n').encode())
        native.restore({},generation=2);native.bind_run({'path':str(identity),'sha256':guard.sha(identity)},last_commit_index=4)
        event=dict(kind='ownership_check',result=dict(passed=True,origin='first',fast_guard_certificate=dict(
            generation=3,previous_generation=2,owner=1,guard_records_sha256=g.refs['records_sha256'],
            previous_claim_hash='a'*64,stage_delta_hash='b'*64,staged_claim_hash='c'*64,
            staged_total_claims=10,stage_new_count=2)))
        lib.certificate=dict(event['result']['fast_guard_certificate'])
        payload=(config.canonical(event)+'\n').encode();digest=hashlib.sha256(payload).hexdigest()
        ep=folder/'events'/f'000007-{digest}.json';cp=folder/'commits/000007.json'
        proof=dict(event_path=str(ep),event_sha256=digest,commit_path=str(cp),commit_sha256='0'*64,index=7)
        with self.assertRaises(ValueError):native.publish_stage(3,'first',proof)
        self.assertEqual(lib.published,[])
        guard._publish(ep,payload)
        previous=folder/'commits/000006.json';guard._publish(previous,b'{"prior_fixture":true}\n')
        commit=dict(index=7,identity_sha256=guard.sha(identity),event=f'events/000007-{digest}.json',event_sha256=digest,previous_sha256=guard.sha(previous))
        guard._publish(cp,(config.canonical(commit)+'\n').encode());proof['commit_sha256']=guard.sha(cp)
        for bad in ({**proof,'index':4},{**proof,'event_sha256':'0'*64}):
            with self.assertRaises(ValueError):native.publish_stage(3,'first',bad)
        with self.assertRaises(ValueError):native.publish_stage(3,'second',proof)
        lib.certificate['stage_delta_hash']='f'*64
        with self.assertRaises(ValueError):native.publish_stage(3,'first',proof)
        lib.certificate=dict(event['result']['fast_guard_certificate'])
        self.assertEqual(lib.published,[])
        native.publish_stage(3,'first',proof);self.assertEqual(lib.published,[(3,1)])
        with self.assertRaises(ValueError):native.publish_stage(3,'first',proof)
        native.close()

    def test_native_refuses_unverified_manual_guard(self):
        lib=FakeLibrary()
        with self.assertRaises(ValueError):guard.NativeGuard(lib,loaded())

    def test_compiled_authorization_cannot_be_resealed_to_different_spec(self):
        folder=guard.DEFAULT/'author-fixtures'/uuid.uuid4().hex;gp=folder/'graph.json';base=loaded()
        graph=dict(components={'allow':{},'deny':{}},quotient_components=base.graph_lookup)
        guard._publish(gp,(config.canonical(graph)+'\n').encode());relative=gp.relative_to(guard.ROOT).as_posix()
        spec=dict(guard=dict(graph_path=relative,allowed_components=['allow'],forbidden_quotients=[],
                            engineering_consumed_only=True),schedule=[{'origin':'first'},{'origin':'second'}],
                  inputs={relative:guard.sha(gp)})
        refs=guard.prepare(spec,folder/'compiled');verified=guard.load_pure(refs,spec)
        with verified.attach(FakeLibrary()):pass
        with self.assertRaises(ValueError):guard.load_pure(refs,{**spec,'schedule':[{'origin':'new'}]})
        original=json.loads((guard.ROOT/refs['manifest_path']).read_bytes())
        for i in range(4):
            body=json.loads(json.dumps(original));body.pop('guard_stamp')
            if i==0:body['guard']['allowed_components']=['allow','deny']
            if i==1:body['inputs'].pop(guard.SOURCES[0])
            if i==2:body['inputs'].pop(relative)
            if i==3:body['spec_guard_and_origins_sha256']='f'*64
            body['guard_stamp']=config.digest(body);dest=folder/f'forgery-{i}'
            guard._publish(dest/'records.bin',verified.records)
            guard._publish(dest/'manifest.json',(config.canonical(body)+'\n').encode())
            forged=dict(manifest_path=(dest/'manifest.json').relative_to(guard.ROOT).as_posix(),
                manifest_sha256=guard.sha(dest/'manifest.json'),records_path=(dest/'records.bin').relative_to(guard.ROOT).as_posix(),
                records_sha256=guard.sha(dest/'records.bin'),guard_stamp=body['guard_stamp'])
            with self.assertRaises(ValueError):guard.load_pure(forged,spec)


if __name__=='__main__':unittest.main()
