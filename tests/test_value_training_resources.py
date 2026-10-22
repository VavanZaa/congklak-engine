from pathlib import Path
import subprocess
import sys
import time
import unittest
import uuid
from congklak import value_training_resources as resource


class ResourceTests(unittest.TestCase):
    def folder(self,name):
        return resource.ROOT/'build/value-training-v1'/('resources-owner-'+name+'-'+uuid.uuid4().hex[:8])

    def protocol(self):
        return dict(resource.PROTOCOL,lease_ns=3_000_000_000,cumulative_ns=9_000_000_000,
            minimum_free_commit_bytes=1,minimum_free_ram_bytes=1,minimum_disk_bytes=1,
            startup_ns=1_000_000_000,clean_stop_margin_ns=300_000_000)

    def test_private_commit_is_process_metric(self):
        k,h=resource._open(resource.os.getpid(),0x1000|0x0400|0x0010)
        try:self.assertGreater(resource.private_commit(h),0)
        finally:k.CloseHandle(h)

    def test_missing_identity_cannot_reset_ledger(self):
        path=self.folder('missing-identity');resource.initialize(path,True,self.protocol())
        (path/'identity.json').unlink()
        with self.assertRaisesRegex(ValueError,'cannot reset'):resource.initialize(path,True,self.protocol())

    def test_pending_lease_and_hold_cannot_restart(self):
        path=self.folder('pending');resource.reserve(path,path/'out',path/'cache',True,self.protocol())
        report=resource.status(path);self.assertEqual(report['charged_ns'],self.protocol()['lease_ns'])
        with self.assertRaisesRegex(ValueError,'unknown overrun'):resource.reserve(path,path/'out',path/'cache',True,self.protocol())
        resource._atomic(path/'hold.json',{'reason':'fixture'},True)
        with self.assertRaisesRegex(ValueError,'HOLD'):resource.reserve(path,path/'out',path/'cache',True,self.protocol())

    def test_watchdog_real_deadline_kill(self):
        path=self.folder('deadline');path.mkdir(parents=True)
        worker=path/'worker.py'
        worker.write_text("import sys,time\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parents[3]))\nfrom congklak import value_training_resources as r\nr.bind_worker(sys.argv[1])\ntime.sleep(10)\n",encoding='utf-8')
        # Ledger directory must begin empty, so keep the fixture worker in a sibling.
        ledger=path/'ledger';lease_path=resource.reserve(ledger,path/'out',path/'cache',True,self.protocol());lease=resource.read(lease_path)
        kwargs=dict(cwd=resource.ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        watchdog=subprocess.Popen([sys.executable,str(resource.ROOT/'tools/run-value-training.py'),'watchdog','--lease',str(lease_path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kwargs)
        process=subprocess.Popen([sys.executable,str(worker),str(lease_path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kwargs)
        resource._atomic(lease['launcher_file'],dict(token=lease['token'],process=resource.process_identity(process.pid),supervisor=lease['supervisor_identity']),True)
        out,err=watchdog.communicate(timeout=12);process.communicate(timeout=5)
        self.assertEqual(watchdog.returncode,0,(out,err));self.assertNotEqual(process.returncode,0)
        report=resource.status(ledger);self.assertIsNone(report['pending'])
        event=report['events'][-1];self.assertFalse(event['clean']);self.assertEqual(event['reason'],'deadline')
        self.assertEqual(event['charged_ns'],lease['reserved_ns']);self.assertTrue(Path(lease['permit_file']).is_file())


if __name__=='__main__':unittest.main()
