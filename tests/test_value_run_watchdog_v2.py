"""Quota race regression: existing temps count; only vanished temps may disappear."""
from pathlib import Path
import os
import tempfile
import threading
import unittest
from unittest.mock import patch
from congklak import value_run_watchdog_v2 as adapter,value_run_resources as frozen


class QuotaRepair(unittest.TestCase):
    def test_missing_only_pending_allowed(self):
        with patch.object(Path,'stat',side_effect=FileNotFoundError('raced rename')):
            self.assertEqual(adapter._size(Path('atomic.json.pending')),0)
            with self.assertRaises(FileNotFoundError):adapter._size(Path('committed.json'))
        with patch.object(Path,'stat',side_effect=PermissionError('unrelated access failure')):
            with self.assertRaises(PermissionError):adapter._size(Path('atomic.json.pending'))

    def test_existing_pending_is_counted(self):
        with tempfile.TemporaryDirectory(dir=adapter.ROOT/'build/value-run-v1') as d:
            root=Path(d);(root/'record.json').write_bytes(b'a'*173);(root/'quota.json.pending').write_bytes(b'b'*291)
            with patch.object(adapter,'ROOTS',(root,)):
                self.assertEqual(adapter.namespace_bytes(),464)

    def test_actual_concurrent_atomic_rename_stress(self):
        with tempfile.TemporaryDirectory(dir=adapter.ROOT/'build/value-run-v1') as d:
            root=Path(d);target=root/'current.json';target.write_bytes(b'x'*512)
            failures=[];renames=[];readings=[]
            def writer():
                try:
                    for i in range(1000):
                        temp=root/('current-%06d.pending'%i);temp.write_bytes(b'x'*512);os.replace(temp,target);renames.append(i)
                except Exception as error:failures.append(repr(error))
            with patch.object(adapter,'ROOTS',(root,)):
                thread=threading.Thread(target=writer);thread.start()
                while thread.is_alive():readings.append(adapter.namespace_bytes())
                thread.join();readings.append(adapter.namespace_bytes())
            self.assertFalse(failures);self.assertEqual(len(renames),1000);self.assertGreater(len(readings),50)
            self.assertTrue(all(512<=n<=1024 for n in readings))

    def test_directory_permission_errors_fail_closed(self):
        with self.assertRaises(PermissionError):adapter._walk_error(PermissionError('directory denied'))
        with patch.object(adapter.os,'walk',side_effect=PermissionError('cannot enumerate directory')):
            with self.assertRaises(PermissionError):adapter.namespace_bytes()

    def test_resource_limits_are_unchanged(self):
        self.assertEqual(adapter.witness()['resource_protocol'],frozen.PROTOCOL)
        self.assertEqual(frozen.PROTOCOL['lease_ns'],300_000_000_000)
        self.assertEqual(frozen.PROTOCOL['cumulative_ns'],5_400_000_000_000)
        self.assertEqual(frozen.PROTOCOL['private_commit_bytes'],4*1024**3)
        self.assertEqual(frozen.PROTOCOL['quota_bytes'],5*1024**3)


if __name__=='__main__':unittest.main()
