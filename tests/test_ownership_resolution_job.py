"""Journal-mechanics fixtures use two toy dictionaries, no scientific positions."""
from contextlib import ExitStack
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch
from congklak import config, ownership_resolution_job as job


class Interrupted(RuntimeError):pass


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.root=job.FOLDER/'fixtures/author-journal'/uuid.uuid4().hex
        self.path=self.root/'output';self.plan=self.root/'plan.json'
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.value=dict(schema='toy-metadata-only',activity=job.stage.ZERO,models_fixed=True,policy_fixed=True)
        def validate(value):
            job.need(config.canonical(value)==config.canonical(self.value),'Toy metadata changed')
        module=SimpleNamespace(SOURCES=(),input_pins=lambda:{},inventory=lambda:copy.deepcopy(self.value),validate_inventory=validate)
        epoch=SimpleNamespace(SOURCES=(),input_pins=lambda:{})
        self.stack.enter_context(patch.object(job,'modules',return_value=(module,module,epoch)))
        job.prepare_plan(self.plan,True)

    def stop(self,point):
        def pause(path,actual,requested,marker):
            if actual==point:raise Interrupted(actual)
        return patch.object(job,'_pause',side_effect=pause)

    def snap(self,paths):
        return {str(p):(p.read_bytes(),p.stat().st_size,p.stat().st_mtime_ns) for p in paths}

    def test_orphan_unit_is_adopted_without_repeating_producer(self):
        with self.stop('before_unit_commit'):
            with self.assertRaises(Interrupted):job.run(self.path,self.plan,'before_unit_commit',self.root/'marker.json')
        unit=self.path/'units/000000.json';before=self.snap([unit])
        real=job._compute
        with patch.object(job,'_compute',wraps=real) as compute:job.run(self.path,self.plan)
        self.assertEqual(compute.call_count,1);self.assertEqual(before,self.snap([unit]))

    def test_complete_prefix_reuses_report_without_new_unit_work(self):
        with self.stop('before_catalog'):
            with self.assertRaises(Interrupted):job.run(self.path,self.plan,'before_catalog',self.root/'marker.json')
        retained=[self.path/'report.json',*sorted((self.path/'units').glob('*.json')),*sorted((self.path/'commits').glob('*.json'))]
        before=self.snap(retained)
        with patch.object(job,'_compute',side_effect=AssertionError('Repeated completed unit')):job.run(self.path,self.plan)
        self.assertEqual(before,self.snap(retained))

    def test_completed_catalog_routes_before_writer_and_producer(self):
        job.run(self.path,self.plan)
        with patch.object(job,'Writer',side_effect=AssertionError('Writer opened')),\
             patch.object(job,'prepare',side_effect=AssertionError('Prepare called')),\
             patch.object(job,'_compute',side_effect=AssertionError('Producer called')):
            self.assertTrue(job.run(self.path,self.plan)['report']['passed'])
            self.assertTrue(job.completed(self.path)['report']['passed'])

    def test_resealed_false_zero_orphan_refuses_before_writer(self):
        with self.stop('before_unit_commit'):
            with self.assertRaises(Interrupted):job.run(self.path,self.plan,'before_unit_commit',self.root/'marker.json')
        unit=self.path/'units/000000.json';body=job.read(unit)
        body['result']['activity']['optimizer_steps']=False
        body['result_sha256']=config.digest(body['result'])
        unit.write_bytes((config.canonical(body)+'\n').encode())
        with patch.object(job,'Writer',side_effect=AssertionError('Corrupt orphan opened writer')):
            with self.assertRaisesRegex(ValueError,'zero-work'):job.run(self.path,self.plan)

    def test_resealed_boolean_commit_index_refuses(self):
        job.run(self.path,self.plan);cp=self.path/'commits/000000.json';body=job.read(cp)
        body['index']=False;cp.write_bytes((config.canonical(body)+'\n').encode())
        with self.assertRaisesRegex(ValueError,'commit chain'):job.completed(self.path)

    def test_nonadjacent_orphan_refuses_before_writer(self):
        job.prepare(self.path,self.plan)
        job.atomic(self.path/'units/000001.json',{})
        with patch.object(job,'Writer',side_effect=AssertionError('Malformed prefix opened writer')):
            with self.assertRaisesRegex(ValueError,'orphan'):job.run(self.path,self.plan)


if __name__=='__main__':unittest.main()
