"""Dashboard admission must follow complete sealed evidence even on shortage."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from congklak import footprint_screen_finish as f,footprint_screen_common as s,divergence_study_common as base

class FinalReceiptChecks(unittest.TestCase):
    def fixture(self):
        folder=Path(self.temporary.name);self.folder=folder
        self.patches=[patch.object(s,'ROOT',folder),patch.object(s,'FOLDER',folder/'build'),patch.object(base,'ROOT',folder)]
        for item in self.patches:item.start();self.addCleanup(item.stop)
        self.event=s.FOLDER/'events/000000.json';s.atomic(self.event,dict(state='shortage'))
        body=dict(live_inputs={},reservation_admission=False,engineering_execution_complete=True,
            metrics=dict(fit_roots=1),decision=dict(ready_for_future_training_recipe=False,training_authorized=False))
        s.atomic(s.FOLDER/'report.json',body);s.atomic(s.FOLDER/'audit/audit.json',dict(passed=True,reservation_admission=False))
        marker=dict(schema=s.SCHEMA+'-acceptance',passed=True,training_authorized=False,
            models_fixed=True,policy_fixed=True,report_ref=f._ref(s.FOLDER/'report.json'),
            audit_ref=f._ref(s.FOLDER/'audit/audit.json'),closure_ref=dict(path='build/closure.json',sha256='frozen'),
            latest_event_ref=f._ref(self.event),reservation_admission=False,engineering_execution_complete=True,
            metrics=body['metrics'],decision=body['decision'],model_forwards=0,optimizer_steps=0,
            historically_unseen=False,inherited_unknown_background=True)
        marker['compact_inputs']=s.common.pins([s.FOLDER/'report.json',s.FOLDER/'audit/audit.json',self.event])
        return marker
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.marker=self.fixture()
    def test_honest_shortage_is_valid_observation(self):self.assertTrue(f.valid_compact(self.marker))
    def test_admission_or_metrics_cannot_override_source(self):
        for key,value in [('reservation_admission',True),('metrics',{}),('decision',{}),('optimizer_steps',False)]:
            marker=copy.deepcopy(self.marker);marker[key]=value
            with self.assertRaises(ValueError):f.valid_compact(marker)
    def test_compact_inputs_cannot_omit_receipts(self):
        self.marker['compact_inputs']={}
        with self.assertRaises(ValueError):f.valid_compact(self.marker)

if __name__=='__main__':unittest.main()
