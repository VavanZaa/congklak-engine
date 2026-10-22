"""Independent accounting examples and zero-work launcher boundary checks."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from congklak import config,footprint_screen_resources as r,footprint_screen_common as s

class ResourceChecks(unittest.TestCase):
    def account(self,events,leases):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);(folder/'events').mkdir();(folder/'leases').mkdir()
            identity=dict(schema=r.SCHEMA+'-identity',protocol=r.PROTOCOL,
                parent=dict(charged_ns=737420000000))
            (folder/'identity.json').write_text(config.canonical(identity),encoding='utf-8')
            previous=None
            for i,body in enumerate(events):
                event=dict(body,index=i,previous=previous)
                (folder/'events'/f'{i:06d}.json').write_text(config.canonical(event),encoding='utf-8')
                previous=config.digest(event)
            for lease in leases:
                p=folder/'leases'/lease['token'];p.mkdir()
                (p/'lease.json').write_text(config.canonical(lease),encoding='utf-8')
            (folder/'head.json').write_text(config.canonical(dict(schema=r.SCHEMA+'-head',count=len(events),last_sha256=previous)),encoding='utf-8')
            with patch.object(r,'_path',return_value=folder),patch.object(r,'_identity',return_value=identity):
                return r.status(folder)
    def test_parent_plus_clean_and_full_unclean_charge(self):
        a=dict(token='a',reserved_ns=300000000000);b=dict(token='b',reserved_ns=300000000000)
        events=[dict(kind='reserve',lease=a),dict(kind='finish',token='a',charged_ns=20000000000,
            clean=True,unknown_overrun=False,elapsed_ns=20000000000),dict(kind='reserve',lease=b),
            dict(kind='finish',token='b',charged_ns=300000000000,clean=False,unknown_overrun=False,elapsed_ns=10000000000)]
        report=self.account(events,[a,b])
        self.assertEqual(report['local_charged_ns'],320000000000)
        self.assertEqual(report['charged_ns'],1057420000000)
        self.assertEqual(report['remaining_ns'],9742580000000)
        self.assertIsNone(report['pending'])
    def test_unclean_refund_refused(self):
        lease=dict(token='a',reserved_ns=300000000000)
        with self.assertRaises(ValueError):self.account([dict(kind='reserve',lease=lease),
            dict(kind='finish',token='a',charged_ns=1,clean=False,unknown_overrun=False,elapsed_ns=1)],[lease])
    def test_overlapping_and_uncommitted_leases_refused(self):
        a=dict(token='a',reserved_ns=300000000000);b=dict(token='b',reserved_ns=300000000000)
        with self.assertRaises(ValueError):self.account([dict(kind='reserve',lease=a),dict(kind='reserve',lease=b)],[a,b])
        with self.assertRaises(ValueError):self.account([],[a])
    def test_pending_lease_charges_entire_reservation(self):
        lease=dict(token='a',reserved_ns=300000000000)
        report=self.account([dict(kind='reserve',lease=lease)],[lease])
        self.assertEqual(report['charged_ns'],1037420000000)
        self.assertEqual(report['pending'],lease)
    def test_completed_native_route_starts_no_work(self):
        done=dict(report=dict(state='complete'),publication_gap=False)
        reader=SimpleNamespace(has_completion=lambda _:True,completed=lambda _:done)
        with (patch.object(r,'_implementation',return_value=reader),patch.object(r,'reserve',side_effect=AssertionError('new lease')),
              patch.object(r.subprocess,'Popen',side_effect=AssertionError('new process'))):
            self.assertEqual(r.launch(s.FOLDER/'acquisition'),done)
    def test_catalog_gap_only_calls_publication_reader(self):
        calls=[];done=dict(publication_gap=True)
        reader=SimpleNamespace(has_completion=lambda _:True,completed=lambda _:done,
            publish_completed=lambda p:calls.append(p) or dict(publication_gap=False))
        with (patch.object(r,'_implementation',return_value=reader),patch.object(r,'reserve',side_effect=AssertionError('new lease')),
              patch.object(r.subprocess,'Popen',side_effect=AssertionError('new process'))):
            self.assertFalse(r.launch(s.FOLDER/'acquisition')['publication_gap'])
        self.assertEqual(calls,[s.FOLDER/'acquisition'])
    def test_unregistered_worker_and_quota_extension_refused(self):
        with self.assertRaises(ValueError):r._implementation(s.ROOT/'tools/train-ordinary-epoch.py')
        with patch.object(r,'namespace_bytes',return_value=r.PROTOCOL['quota_bytes']-10):
            self.assertEqual(r.check_namespace(10)['additional_bytes'],10)
            with self.assertRaises(ValueError):r.check_namespace(11)
            with self.assertRaises(ValueError):r.check_namespace(True)

if __name__=='__main__':unittest.main()
