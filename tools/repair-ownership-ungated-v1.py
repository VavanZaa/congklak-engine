"""One incident: identity-checked stop and conservative lease settlement."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import os,time
from congklak import config,neurodiag_common as common,ownership_handoff_resources as r
from congklak.valuepilot_reservations import Writer

TOKEN='5f950192f50c48c8801d40ae634a26ba'
LEDGER=ROOT/'build/ownership-implementation-v1/resources'
OUT=ROOT/'build/ownership-implementation-v1/launcher-repair-v1'
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'result.json').exists():
  report=r.status(LEDGER);result=common.read(OUT/'result.json')
  assert report['pending'] is None
  assert len([e for e in report['events'] if e['kind']=='finish' and e['token']==TOKEN])==1
  print(config.canonical(result));return
 report=r.status(LEDGER);lease=report['pending']
 common.require(lease and lease['token']==TOKEN,'Expected exact pending incident lease missing')
 worker=common.read(lease['worker_identity_file']);launcher=common.read(lease['launcher_file'])
 expected={'pid':7520,'creation_filetime':134366262164960538,
  'image':r'C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\python3.12.exe','parent_pid':32068}
 common.require(worker['process']==expected and r.process_identity(7520)==expected,'Actual worker identity changed; refuse termination')
 common.require(launcher['process']==expected and launcher['supervisor']==lease['supervisor_identity'] and
  r.process_identity(32068)==lease['supervisor_identity'],'Supervisor/launcher identity changed')
 common.require(worker['argv']==launcher['command'][1:] and worker['token']==launcher['token']==TOKEN and
  worker['cwd']==str(ROOT) and Path(worker['argv'][0])==ROOT/'tools/run-ownership-handoff.py' and
  '--lease' in worker['argv'] and worker['argv'][worker['argv'].index('--lease')+1]==lease['lease_file'],
  'Worker command/lineage/token differs')
 common.require(not Path(lease['permit_file']).exists() and not Path(lease['completion_file']).exists(),
  'Incident acquired permit/completion: refuse assumed-gated repair')
 common.require(time.monotonic_ns()<lease['deadline_monotonic_ns'],'Lease already expired; overrun requires HOLD')
 inputs=common.pins([__file__,ROOT/'congklak/ownership_handoff_resources.py',ROOT/'congklak/ownership_handoff_worker.py',
  ROOT/'tools/run-ownership-handoff.py',Path(lease['lease_file']),Path(lease['worker_identity_file']),Path(lease['launcher_file']),LEDGER/'identity.json'])
 intent={'schema':'ownership-ungated-repair-intent-v1','token':TOKEN,'worker':expected,'supervisor':lease['supervisor_identity'],
  'inputs':inputs,'permit_absent':True,'completion_absent':True,'deadline_monotonic_ns':lease['deadline_monotonic_ns'],
  'stop_started_monotonic_ns':time.monotonic_ns(),'resource_head_before':report['head'],'charged_before_ns':report['charged_ns'],
  'reason':'Old CLI default-completed shortcut preceded watchdog dispatch; actual worker remained at permit gate'}
 r._atomic(OUT/'intent.json',intent,True)
 k,h=r._open(7520,0x1000|0x00100000|0x0001)
 try:
  common.require(r.process_identity(7520)==expected,'Worker identity changed immediately before stop')
  common.require(k.WaitForSingleObject(h,0)==258,'Expected gated worker already exited')
  common.require(k.TerminateProcess(h,125),'Cannot stop exact owned gated worker')
  common.require(k.WaitForSingleObject(h,5000)==0,'Stopped worker did not exit')
 finally:k.CloseHandle(h)
 stopped=time.monotonic_ns()
 common.require(stopped<lease['deadline_monotonic_ns'],'Stop exceeded deadline: accounting remains HOLD')
 with Writer(LEDGER):
  current=r.status(LEDGER)
  common.require(current['pending']==lease and not Path(lease['permit_file']).exists() and
   not Path(lease['completion_file']).exists(),'Incident lease/permit/completion changed during stop')
  event=r._append(LEDGER,current,dict(kind='finish',token=TOKEN,clean=False,charged_ns=lease['reserved_ns'],
   elapsed_ns=stopped-(lease['deadline_monotonic_ns']-lease['reserved_ns']),
   reason='source-bound conservative repair: ungated worker stopped before deadline; entire reservation charged',
   peak_private_commit_bytes=None,actual_worker=expected,unknown_overrun=False,
   watchdog=None,repair_process=r.process_identity(),repair_source_inputs=inputs,repair_intent_sha256=common.sha(OUT/'intent.json')))
 after=r.status(LEDGER)
 common.require(after['pending'] is None and len([e for e in after['events'] if e['kind']=='finish' and e['token']==TOKEN])==1,
  'Resource repair did not settle exactly once')
 result={'schema':'ownership-ungated-repair-result-v1','passed':True,'token':TOKEN,'stopped_worker':expected,
  'stop_completed_monotonic_ns':stopped,'stop_before_deadline':True,'full_unclean_charge_ns':lease['reserved_ns'],
  'charged_after_ns':after['charged_ns'],'resource_event_sha256':config.digest(event),'resource_head_after':after['head'],
  'pending':None,'accounting_reset':False,'optimizer_steps':0,'new_games':0,'searches':0,'source_inputs':inputs}
 r._atomic(OUT/'result.json',result,True);print(config.canonical(result))
if __name__=='__main__':main()
