"""Management routing, strict persisted permit closure and read-only completion."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import unittest
from unittest.mock import patch
from congklak import ownership_handoff_launcher_v2 as launcher
from congklak import native_ownership_trace as trace

ROOT=launcher.ROOT
OUTPUT=ROOT/'build/ownership-implementation-v1/handoff/engineering-timed-v2'

def forbidden(*args,**kwargs):raise AssertionError('Completed path attempted writer/worker/engine')

class LauncherTests(unittest.TestCase):
 def test_resources_management_precedes_default_completed_catalog(self):
  self.assertTrue((launcher.worker.DEFAULT/'catalog.json').exists())
  with patch.object(launcher.worker,'completed',side_effect=forbidden),patch.object(launcher.resources,'status',return_value={'management':True}) as read:
   with redirect_stdout(io.StringIO()):code=launcher.main(['resources'])
   self.assertEqual(code,0);read.assert_called_once()

 def test_watchdog_management_precedes_default_completed_catalog(self):
  self.assertTrue((launcher.worker.DEFAULT/'catalog.json').exists())
  with patch.object(launcher.worker,'completed',side_effect=forbidden),patch.object(launcher,'watchdog',return_value={'clean':True}) as watch:
   with redirect_stdout(io.StringIO()):code=launcher.main(['watchdog','--lease','actual-lease.json'])
   self.assertEqual(code,0);watch.assert_called_once_with(Path('actual-lease.json'))

 def test_completed_api_cli_no_writer_reservation_engine_or_byte_mutation(self):
  files=[p for p in OUTPUT.rglob('*') if p.is_file()]
  before={str(p):(p.read_bytes(),p.stat().st_size,p.stat().st_mtime_ns) for p in files}
  with patch.object(launcher.worker,'run',side_effect=forbidden),patch.object(launcher.worker,'prepare',side_effect=forbidden),\
       patch.object(launcher.resources,'reserve',side_effect=forbidden),patch.object(launcher.resources,'_atomic',side_effect=forbidden),\
       patch.object(trace,'_classes',side_effect=forbidden),patch.object(trace,'open_providers',side_effect=forbidden):
   result=launcher.completed(OUTPUT)
   self.assertTrue(result['launcher']['verified'])
   with redirect_stdout(io.StringIO()):code=launcher.main(['run','--output',str(OUTPUT)])
   self.assertEqual(code,0)
  after={str(p):(p.read_bytes(),p.stat().st_size,p.stat().st_mtime_ns) for p in files}
  self.assertEqual(before,after)

 def test_missing_actual_watchdog_sources_refused_without_mutating_permit(self):
  seal=launcher.read(OUTPUT/'launcher-completed-v2.json');lease=launcher.read(seal['lease'])
  permit_path=Path(lease['permit_file']).resolve();original=launcher.read
  def corrupted(path):
   result=original(path)
   if Path(path).resolve()==permit_path:
    result=copy.deepcopy(result);result['launcher_adapter']['actual_watchdog_sources']={}
   return result
  with patch.object(launcher,'read',side_effect=corrupted):
   with self.assertRaisesRegex(ValueError,'attestation differs'):launcher.verify_permit(seal['lease'])

 def test_launcher_failure_is_nonzero(self):
  with patch.object(launcher,'launch',side_effect=ValueError('actual nonzero worker/watchdog')):
   stream=io.StringIO()
   with redirect_stdout(stream):code=launcher.main(['run','--output',str(OUTPUT)])
   self.assertEqual(code,2);self.assertIn('hold',stream.getvalue())

if __name__=='__main__':unittest.main()
