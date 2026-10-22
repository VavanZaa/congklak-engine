"""Pure causal prefix/epoch tests; synthetic metadata uses one consumed root.

No archive here represents new engine work.  Usable lifecycle-only packets are
limited to `_prefix` unit checks; full-run fixtures explicitly say no attempt
started and test recovery/failure publication with constructor traps.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
from congklak import config,tracing_speed_timing as timing


EMPTY=hashlib.sha256(b'').hexdigest()
INITIAL=dict(created=2,destroyed=0,retained_inference_controls=dict(control=1,candidate=2))


def write(path,body):
    """Only a fresh test namespace is deliberately mutable for forgeries."""
    path=Path(path);assert path.resolve().is_relative_to(timing.FOLDER/'timing/fixtures/pure-prefix')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((config.canonical(body)+'\n').encode())


class PrefixFixture:
    def __init__(self):
        self.path=timing.FOLDER/'timing/fixtures/pure-prefix'/uuid.uuid4().hex;self.path.mkdir(parents=True)
        source=timing.ROOT/'build/ownership-implementation-v1/handoff/engineering-spec.json'
        self.root=timing.read(source)['schedule'][4]['start']
        self.spec=dict(search_policy=dict(reserve_ms=8.0,minimum_search_ms=10,same_all_engines=True,calibrated=True),capacity_events=2000000)
        sp=self.path/'fixture-spec.json';write(sp,self.spec)
        self.identity=dict(spec_path=sp.relative_to(timing.ROOT).as_posix(),native=dict(
            schema='pure-prefix-provider-metadata-only',consumed_source=source.relative_to(timing.ROOT).as_posix(),
            consumed_source_sha256=timing.sha(source)),cpu_declared=dict(logical=0))
        write(self.path/'identity.json',self.identity)
        self.panel=dict(panel=[dict(state=self.root)],schedule=[dict(index=i,root=0,repetition=i//4,
            engine=timing.contract.ENGINES[i%4]) for i in range(240)])
        self.loaded=SimpleNamespace(guard_stamp='pure-prefix-not-admitted',claims_blob=lambda claimed:b'')
        self.rows=[];self.epoch=None;self.after=None

    def epoch_at(self,index):
        path=self.path/'epochs'/f'{index:06d}.json'
        epoch=dict(schema='trace-speed-provider-epoch-v1',start_index=index,
            committed_prefix_sha256=timing.sha(self.path/'commits'/f'{2*index-1:06d}.json') if index else None,
            initial_lifecycle=INITIAL,native_identity_sha256=config.digest(self.identity['native']))
        write(path,epoch);self.epoch=dict(path=path.relative_to(self.path).as_posix(),sha256=timing.sha(path));self.after=INITIAL

    def add(self,usable=False,cold=False,started=False):
        index=len(self.rows);scheduled=self.panel['schedule'][index]
        if self.epoch is None or cold:self.epoch_at(index)
        timing._commit(self.path,dict(kind='guard_restore_context',attempt=index,generation=0,
            canonical_claim_hash=EMPTY,guard_stamp=self.loaded.guard_stamp,
            prefix_sha256=timing.sha(self.path/'commits'/f'{2*index-1:06d}.json') if index else None))
        error=None if usable else 'pure fixture: engine attempt never started'
        row=dict(scheduled,critical_ms=2.0,usable_decision=usable,coverage_passed=usable,guard_passed=usable,
            event_count=0,owned_count=0,native_search_ms=None,archival_ms=0.0,error=error,
            archive_path=f'attempts/{index:06d}.json')
        reset_after=dict(created=self.after['created']+2,destroyed=self.after['destroyed']+2,
            retained_inference_controls={k:v+2 for k,v in self.after['retained_inference_controls'].items()}) if usable or started else self.after
        after=dict(reset_after,created=reset_after['created']+1,destroyed=reset_after['destroyed']+1) if usable or started else self.after
        cert=dict(valid=usable,guard_passed=usable,event_count=0,attempted_events=0,owned_count=0)
        if usable:cert.update(generation=1,owner=1,previous_generation=0,stage_new_count=0)
        answer=dict(fast_certificate=cert,fast_error=error,engine_attempt_started=started,
            fast_timing=dict(policy=self.spec['search_policy'],outer_elapsed_ms=1.0),
            archival_timing=dict(elapsed_ms=0.0,charged_to_lease=True),
            both_seat_reset=dict(before=self.after,after=reset_after),lifecycle=dict(before=reset_after,after=after))
        decision_id=f'timing/{index:06d}'
        write(self.path/'intents'/f'{index:06d}.json',dict(scheduled=scheduled,root=self.root,
            identity_sha256=timing.sha(self.path/'identity.json'),decision_id=decision_id,claims_sha256=EMPTY))
        saved=dict(scheduled=scheduled,root=self.root,decision_id=decision_id,identity_sha256=timing.sha(self.path/'identity.json'),
            answer=answer,transition=None,row=row,provider_epoch=self.epoch)
        ap=self.path/row['archive_path'];write(ap,saved);row=dict(row,archive_sha256=timing.sha(ap))
        check=dict(kind='ownership_check' if usable else 'attempt_hold',attempt=index,archive_sha256=row['archive_sha256'])
        if usable:check['result']=dict(passed=True,origin=timing.ORIGIN,fast_guard_certificate=cert)
        else:check.update(error=error,production_admission=False)
        timing._commit(self.path,check);write(self.path/'rows'/f'{index:06d}.json',row)
        self.rows.append(row);self.after=after;return saved

    def reseal_last_archive(self,mutate):
        index=len(self.rows)-1;path=self.path/'attempts'/f'{index:06d}.json';saved=timing.read(path);mutate(saved);write(path,saved)
        self.rows[index]['archive_sha256']=timing.sha(path);write(self.path/'rows'/f'{index:06d}.json',self.rows[index])
        cp=self.path/'commits'/f'{2*index+1:06d}.json';commit=timing.read(cp);event=timing.read(self.path/commit['event'])
        event['archive_sha256']=self.rows[index]['archive_sha256'];digest=hashlib.sha256((config.canonical(event)+'\n').encode()).hexdigest()
        relative=f'events/{2*index+1:06d}-{digest}.json';write(self.path/relative,event);commit.update(event=relative,event_sha256=digest);write(cp,commit)

    def prefix(self):return timing._prefix(self.path,self.rows,self.panel,self.loaded,{},self.identity)

    def run_pure(self):
        import congklak.native_trace_fast_v2 as provider
        with patch.object(timing,'prepare'),patch.object(timing,'_identity',return_value=(self.identity,self.spec,self.panel,self.loaded)),\
             patch.object(timing,'_pressure',return_value={}),\
             patch.object(provider,'open_providers',side_effect=AssertionError('provider constructed')) as opened,\
             patch.object(timing,'PerformanceCore',side_effect=AssertionError('CPU context constructed')) as cpu:
            try:return timing.run(self.path,self.path/'fixture-spec.json','pure-no-lease')
            finally:
                assert opened.call_count==0,'Pure recovery opened a provider'
                assert cpu.call_count==0,'Pure recovery reached engine CPU setup'


class PrefixEpochTests(unittest.TestCase):
    def test_valid_contiguous_prefix_and_explicit_cold_restart(self):
        f=PrefixFixture();f.add(usable=True);f.add(usable=True,cold=True)
        self.assertEqual(len(f.prefix()),4)

    def test_missing_or_forged_epoch_refused_before_provider(self):
        for mutation in ('missing','source','prefix','reference'):
            with self.subTest(mutation=mutation):
                f=PrefixFixture();f.add();ep=f.path/'epochs/000000.json'
                if mutation=='missing':ep.unlink()
                elif mutation in ('source','prefix'):
                    epoch=timing.read(ep);epoch['native_identity_sha256' if mutation=='source' else 'committed_prefix_sha256']='f'*64;write(ep,epoch)
                else:
                    ap=f.path/'attempts/000000.json';saved=timing.read(ap);saved['provider_epoch']['sha256']='f'*64;write(ap,saved)
                    f.rows[0]['archive_sha256']=timing.sha(ap);write(f.path/'rows/000000.json',f.rows[0])
                with self.assertRaisesRegex(ValueError,'epoch'):f.run_pure()

    def test_unlogged_controls_within_epoch_are_rejected(self):
        f=PrefixFixture();f.add(usable=True);f.add(usable=True)
        ap=f.path/'attempts/000001.json';saved=timing.read(ap)
        saved['answer']['both_seat_reset']['before']['created']+=1;write(ap,saved)
        with self.assertRaisesRegex(ValueError,'unlogged native controls'):f.prefix()

    def test_initial_epoch_cannot_hide_prior_native_control_work(self):
        f=PrefixFixture();f.add(usable=True);ep=f.path/'epochs/000000.json';epoch=timing.read(ep)
        epoch['initial_lifecycle']['created']=3;write(ep,epoch)
        with self.assertRaisesRegex(ValueError,'cold provider epoch'):f.prefix()

    def test_failed_started_attempt_requires_continuous_complete_lifecycle(self):
        for mutation in ('discontinuous','missing-life','missing-reset','ordering'):
            with self.subTest(mutation=mutation):
                f=PrefixFixture();f.add(usable=True);f.add(started=True)
                def mutate(saved):
                    answer=saved['answer']
                    if mutation=='discontinuous':answer['both_seat_reset']['before']['created']+=1
                    elif mutation=='missing-life':answer.pop('lifecycle')
                    elif mutation=='missing-reset':answer.pop('both_seat_reset')
                    else:answer['lifecycle']['before']['created']+=1
                f.reseal_last_archive(mutate)
                with self.assertRaisesRegex(ValueError,'unlogged native controls|incomplete lifecycle'):f.prefix()

    def test_partial_failed_prefix_cannot_start_another_provider(self):
        f=PrefixFixture();f.add()
        with self.assertRaisesRegex(ValueError,'Failed timing prefix'):f.run_pure()

    def test_row_gap_and_hashlinked_commit_corruption_stop_before_provider(self):
        for mutation in ('rowgap','commit'):
            with self.subTest(mutation=mutation):
                f=PrefixFixture();f.add()
                if mutation=='rowgap':(f.path/'rows/000000.json').rename(f.path/'rows/000001.json')
                else:
                    path=f.path/'commits/000001.json';commit=timing.read(path);commit['previous_sha256']='f'*64;write(path,commit)
                with self.assertRaises(ValueError):f.run_pure()

    def test_forged_check_certificate_refused_even_after_resealing(self):
        f=PrefixFixture();f.add(usable=True);cp=f.path/'commits/000001.json';commit=timing.read(cp)
        event=timing.read(f.path/commit['event']);event['result']['fast_guard_certificate']['owner']=99
        digest=hashlib.sha256((config.canonical(event)+'\n').encode()).hexdigest();relative=f'events/000001-{digest}.json'
        write(f.path/relative,event);commit.update(event=relative,event_sha256=digest);write(cp,commit)
        with self.assertRaisesRegex(ValueError,'authorization'):f.prefix()

    def test_intent_archive_content_coupling_is_checked_before_provider(self):
        for mutation in ('intent-id','intent-source','archive-root'):
            with self.subTest(mutation=mutation):
                f=PrefixFixture();f.add()
                if mutation.startswith('intent'):
                    p=f.path/'intents/000000.json';intent=timing.read(p)
                    intent['decision_id' if mutation=='intent-id' else 'identity_sha256']='forged';write(p,intent)
                else:
                    p=f.path/'attempts/000000.json';saved=timing.read(p);saved['root']={'forged_metadata':True};write(p,saved)
                    f.rows[0]['archive_sha256']=timing.sha(p);write(f.path/'rows/000000.json',f.rows[0])
                with self.assertRaisesRegex(ValueError,'intent|cursor'):f.run_pure()

    def test_all_240_saved_failed_attempts_finish_without_provider(self):
        f=PrefixFixture()
        for _ in range(240):f.add()
        result=f.run_pure()
        self.assertFalse(result['timing_gate']['passed']);self.assertEqual(result['timing_gate']['attempts'],240)
        self.assertTrue((f.path/'catalog.json').is_file())


if __name__=='__main__':unittest.main()
