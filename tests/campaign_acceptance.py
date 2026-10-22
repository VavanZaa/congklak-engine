"""Depth-six lineage admission and actual process-termination recovery."""
import json
from pathlib import Path
import queue
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from congklak import campaign,pilot,config
from congklak.native import ROOT

WORK=ROOT/"build/campaign-acceptance"
WORK.mkdir(parents=True,exist_ok=True)

class CampaignAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=WORK); self.path=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def test_01_imports_prior_state_quotient_family_and_completed_corpus(self):
        result=campaign.verify()
        self.assertTrue(result["imported_registry_verified"])
        self.assertEqual(result["records"],10000)
        self.assertEqual(result["splits"],{"train":8000,"development":1000,"sealed_final_test":1000})
        manifest=pilot.load(campaign.DEFAULT)
        self.assertEqual(manifest["identity"]["teacher_limits"]["max_depth"],6)
        self.assertEqual(manifest["label_kinds"],{"finite_choice_horizon":10000})
        self.assertEqual(manifest["selection"]["quotas"]["development:50-74/ordinary"],73)
        self.assertGreater(manifest["selection"]["imported_exclusion_count"],10000)

    def test_03_every_prior_state_tombstone_and_pruned_family_is_imported(self):
        parent,reserved,families,_=campaign.imports()
        db=sqlite3.connect(pilot.DEFAULT_PATH/parent["directory"]/"dataset.sqlite3")
        try:
            for key,split in db.execute("SELECT state,split FROM exclusions WHERE kind='state'"):
                self.assertIn(("state",bytes(key)),reserved)
                self.assertIn(("quotient",pilot.quotient(pilot.from_key(bytes(key)))),reserved)
        finally: db.close()
        ledger=sqlite3.connect(pilot.DEFAULT_PATH/"work.sqlite3")
        try: ids={row[0] for row in ledger.execute("SELECT family FROM collection_units")}
        finally: ledger.close()
        self.assertEqual(set(families),ids)

    def test_02_actual_kills_after_label_and_before_manifest_resume_equal_fresh(self):
        fresh=campaign.build(self.path/"fresh",target=100,seed=73,families=128)
        for boundary in ("label","publish"):
            run=self.path/boundary
            worker='''import json,sys
from congklak import campaign
def event(e):
 print(json.dumps(e),flush=True)
 if e.get("stage")==sys.argv[2] and (e.get("labels")==100 if sys.argv[2]=="label" else e.get("rows_written")==100):
  sys.stdin.readline()
campaign.build(sys.argv[1],target=100,seed=73,families=128,on_event=event)
'''
            child=subprocess.Popen([sys.executable,"-c",worker,str(run),boundary],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            messages=queue.Queue()
            def reader():
                for line in child.stdout: messages.put(json.loads(line))
            thread=threading.Thread(target=reader,daemon=True); thread.start()
            try:
                while True:
                    e=messages.get(timeout=60)
                    if e.get("stage")==boundary and (e.get("labels")==100 if boundary=="label" else e.get("rows_written")==100): break
                child.kill(); child.wait(timeout=20)
            finally:
                if child.poll() is None: child.kill(); child.wait(timeout=20)
                thread.join(timeout=5); child.stdin.close(); child.stdout.close(); child.stderr.close()
            with self.assertRaises(pilot.PilotError): pilot.load(run)
            resumed=campaign.build(run,target=100,seed=73,families=128)
            self.assertEqual(resumed["semantic_sha256"],fresh["semantic_sha256"])
            self.assertEqual(resumed["splits"],fresh["splits"])
            self.assertTrue(campaign.verify(run)["verified"])
            with self.assertRaises(pilot.PilotError): campaign.build(run,target=100,seed=74,families=128)

if __name__=="__main__": unittest.main()
