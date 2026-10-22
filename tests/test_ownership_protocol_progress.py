import unittest
from unittest.mock import patch
from congklak import ownership_protocol_progress as progress


class ProtocolProgressTests(unittest.TestCase):
    def test_definition_scope_and_no_mutable_worker_reads(self):
        original=progress.read;opened=[]
        def observed(path):
            opened.append(str(path).replace('\\','/'));return original(path)
        with patch.object(progress,'read',side_effect=observed):result=progress.snapshot()
        self.assertTrue(result['definition_only'] and result['models_fixed'] and result['policy_fixed'])
        self.assertFalse(result['execution_ready'] or result['training_authorized'] or result['live_suffix_worker_verified'])
        self.assertFalse(result['confidence_intervals_enabled'] or result['new_reservations_made'])
        self.assertTrue(all(v==0 for v in result['counters'].values()))
        self.assertFalse(any(p.endswith(('/head.json','/progress.json','/ledger.json')) for p in opened))

    def test_stale_loaded_source_holds(self):
        with patch.object(progress,'LOADED_SHA','0'*64):result=progress.snapshot()
        self.assertFalse(result['accepted']);self.assertEqual(result['current_work']['state'],'hold')

    def test_final_event_identity_is_bound(self):
        original=progress.read;real_is_file=progress.Path.is_file
        def observed(path):
            if path==progress.FOLDER/'acceptance.json':
                return dict(passed=True,live_inputs={'fixture':'digest'},progress_event='stale/event.json')
            return original(path)
        def exists(path):return True if path==progress.FOLDER/'acceptance.json' else real_is_file(path)
        with patch.object(progress,'read',side_effect=observed),patch.object(progress.Path,'is_file',exists),patch.object(progress,'current',return_value=True):
            result=progress.snapshot()
        self.assertFalse(result['accepted']);self.assertEqual(result['current_work']['state'],'hold')


if __name__=='__main__':unittest.main()
