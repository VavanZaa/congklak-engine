"""Journal/restore and interrupted-intent checks without engines or play."""
import hashlib
import uuid
import unittest
from unittest.mock import patch
from congklak import config,tracing_speed_worker_v5 as worker,ownership_fast_guard as guard
from tests.test_ownership_fast_guard import loaded


def folder():
    result=worker.stage.FOLDER/'jobs/fixtures'/'author'/uuid.uuid4().hex
    result.mkdir(parents=True);return result


def identity(start):
    row=dict(index=0,profile='candidate',baseline='H_v1',seat=0,origin='first',start=start,source='consumed')
    return dict(spec=dict(schedule=[row],mode='fixed128'),guard_refs={})


def write_identity(path,body):worker.atomic(path/'identity.json',body)


class JournalTests(unittest.TestCase):
    def completed_prefix(self):
        path=folder();source=worker.ROOT/'build/ownership-implementation-v1/handoff/engineering-spec.json'
        state=worker.read(source)['schedule'][8]['start'];ident=identity(state)
        ident['sources']={'pure_consumed_fixture_sha256':worker.sha(source)};write_identity(path,ident)
        events=[];g=loaded();worker.append(path,events,dict(kind='guard_restore_context',generation=0,
            canonical_claims_sha256=hashlib.sha256(b'').hexdigest(),committed_prefix_sha256=None,guard_stamp=g.guard_stamp))
        worker.append(path,events,dict(kind='game_start',game=0,start=state))
        worker.append(path,events,dict(kind='handoff',game=0,state=state,terminal=True,reset_fields=list(worker.protocol.RESET_FIELDS)))
        worker.append(path,events,dict(kind='game_end',game=0,final=state,choices=0,terminal_margin=state['stores'][0]-state['stores'][1]))
        data=worker.replay(path,ident,g);worker.atomic(path/'report.json',worker.summary(data,ident))
        return path,ident,g

    def assert_complete_publication_is_pure(self,point):
        path,ident,g=self.completed_prefix()
        retained=[path/'report.json',path/'identity.json',*sorted((path/'commits').glob('*.json')),*sorted((path/'events').glob('*.json'))]
        before={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in retained}
        class SimulatedKill(RuntimeError):pass
        def pause(_path,actual,requested,*args):
            if actual==requested:raise SimulatedKill(actual)
        def completed(_path):return {'catalog':worker.read(_path/'catalog.json'),'report':worker.read(_path/'report.json')}
        real_writer=worker.Writer
        with patch.object(worker,'prepare',return_value=ident),patch.object(guard,'load_pure',return_value=g),\
             patch.object(worker,'sources',return_value=ident['sources']),patch.object(worker,'_capacity'),\
             patch.object(worker,'_pause',side_effect=pause),patch.object(worker,'completed',side_effect=completed),\
             patch.object(worker,'Writer',wraps=real_writer) as writer,\
             patch.object(worker,'append',side_effect=AssertionError('completed prefix changed its journal')) as appended,\
             patch('congklak.native_trace_fast_v2.open_providers',side_effect=AssertionError('provider constructed')) as provider,\
             patch('congklak.timing_v2.PerformanceCore',side_effect=AssertionError('CPU context constructed')) as cpu,\
             patch.object(guard.NativeGuard,'restore',side_effect=AssertionError('claims restored')) as restore:
            with self.assertRaises(SimulatedKill):worker.run(path,path/'unused-spec.json','pure-unused-lease',point,path/'marker.json')
            self.assertEqual(writer.call_count,1)
            if point=='after_catalog':
                with patch.object(worker,'Writer',side_effect=AssertionError('completed route opened writer')):
                    result=worker.run(path,path/'unused-spec.json','pure-unused-lease')
            else:result=worker.run(path,path/'unused-spec.json','pure-unused-lease')
            self.assertEqual(provider.call_count,0);self.assertEqual(cpu.call_count,0);self.assertEqual(restore.call_count,0)
            self.assertEqual(appended.call_count,0);self.assertEqual(result['report']['restore_contexts'],1)
        self.assertEqual(before,{str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in retained})
        self.assertTrue((path/'catalog.json').is_file())

    def test_complete_prefix_before_catalog_recovery_preserves_report_and_journal(self):
        self.assert_complete_publication_is_pure('before_catalog')

    def test_complete_prefix_after_catalog_recovery_opens_no_writer(self):
        self.assert_complete_publication_is_pure('after_catalog')

    def test_terminal_boundary_and_cold_context_reset_are_literal(self):
        path=folder();state=dict(controller=0,pits=[[0]*7,[0]*7],stores=[62,36]);ident=identity(state)
        write_identity(path,ident);events=[];g=loaded();empty=hashlib.sha256(b'').hexdigest()
        worker.append(path,events,dict(kind='guard_restore_context',generation=0,canonical_claims_sha256=empty,
            committed_prefix_sha256=None,guard_stamp=g.guard_stamp))
        worker.append(path,events,dict(kind='game_start',game=0,start=state))
        worker.append(path,events,dict(kind='handoff',game=0,state=state,terminal=True,reset_fields=list(worker.protocol.RESET_FIELDS)))
        worker.append(path,events,dict(kind='game_end',game=0,final=state,choices=0,terminal_margin=26))
        data=worker.replay(path,ident,g);self.assertEqual(data['game'],1);self.assertEqual(data['generation'],0)
        worker.append(path,events,dict(kind='guard_restore_context',generation=0,canonical_claims_sha256=empty,
            committed_prefix_sha256=data['prefix_sha256'],guard_stamp=g.guard_stamp))
        data=worker.replay(path,ident,g);self.assertEqual(len(data['restores']),2);self.assertEqual(data['claim_hash'],empty)

    def test_resealed_wrong_restore_prefix_and_unlinked_commit_refused(self):
        path=folder();state=dict(controller=0,pits=[[0]*7,[0]*7],stores=[62,36]);ident=identity(state);write_identity(path,ident)
        events=[];g=loaded();worker.append(path,events,dict(kind='guard_restore_context',generation=0,
            canonical_claims_sha256=hashlib.sha256(b'').hexdigest(),committed_prefix_sha256='f'*64,guard_stamp=g.guard_stamp))
        with self.assertRaises(ValueError):worker.replay(path,ident,g)
        cp=path/'commits/000000.json';commit=worker.read(cp);commit['previous_sha256']='a'*64
        # Deliberately corrupt a NEW fixture, never any accepted source.
        cp.write_bytes((config.canonical(commit)+'\n').encode())
        with self.assertRaises(ValueError):worker._journal(path,ident)

    def test_intent_without_saved_archive_is_durable_hold_before_provider(self):
        path=folder();state=dict(controller=1,pits=[[0,16,1,4,1,0,2],[0,0,2,0,1,0,1]],stores=[15,55])
        ident=identity(state);write_identity(path,ident);events=[];g=loaded()
        worker.append(path,events,dict(kind='guard_restore_context',generation=0,
            canonical_claims_sha256=hashlib.sha256(b'').hexdigest(),committed_prefix_sha256=None,guard_stamp=g.guard_stamp))
        worker.append(path,events,dict(kind='game_start',game=0,start=state))
        engine=worker.protocol.dispatch(state,'candidate','H_v1',0,False)['engine']
        worker.append(path,events,dict(kind='decision_intent',game=0,choice=0,decision_id='000/000',engine=engine,
            before=state,archive='archives/000-000.json',attempt_sink='attempts/000-000.bin'))
        with patch.object(worker,'prepare',return_value=ident),patch.object(guard,'load_pure',return_value=g),\
             patch.object(worker,'_capacity'),patch('congklak.native_trace_fast_v2.open_providers',side_effect=AssertionError('engine loaded')) as provider:
            with self.assertRaisesRegex(ValueError,'Missing complete attempt archive'):worker.run(path,path/'unused-spec.json','unused-lease')
        self.assertEqual(provider.call_count,0)
        data=worker.replay(path,ident,g);self.assertTrue(data['hold']['terminal']);self.assertEqual(len(data['decisions']),0)

    def test_completed_route_never_acquires_writer(self):
        path=folder();worker.atomic(path/'catalog.json',{})
        worker.atomic(path/'identity.json',{'spec':{},'schema':'corrupt'})
        with patch.object(worker,'Writer',side_effect=AssertionError('writer opened')) as writer:
            with self.assertRaises(ValueError):worker.run(path)
        self.assertEqual(writer.call_count,0)


if __name__=='__main__':unittest.main()
