"""Verify completed review dashboard via backend and listener, without browser/HTTP."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, allocation_progress as observer, quality_search
from congklak.native import Native
from congklak.neural_relay import NeuralNative
from congklak.allocation_common import publish


def signature(path):
    with path.open('rb') as stream: checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(sha256=checksum, size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)


def verify_seal_replay(accepted, replay, signatures):
    assert accepted['schema'] == 'allocation-acceptance-v1' and accepted['passed'] is True
    assert replay['schema'] == 'allocation-acceptance-rerun-v1' and replay['passed'] is True
    assert replay['acceptance_bytes_size_mtime_unchanged'] is True
    assert replay['before'] == replay['after'] == signatures['build/allocation-acceptance.json']
    assert replay['all_inputs_current'] == len(accepted['inputs']) and replay['tests_run'] == accepted['tests_run']
    assert replay['recorder_sha256'] == signatures['tools/record-allocation.py']['sha256']
    assert replay['verifier_sha256'] == signatures['tools/verify-allocation-acceptance.py']['sha256']
    assert replay['torch_imported'] is False and replay['new_training_or_games'] is False


def main():
    accepted = json.loads((ROOT / 'build/allocation-acceptance.json').read_text())
    names = set(accepted['inputs']) | {'build/allocation-acceptance.json','build/allocation-acceptance-rerun.json'}
    def record(): return {name: signature(ROOT / name) for name in sorted(names)}
    before = record()
    assert all(before[name]['sha256'] == digest for name, digest in accepted['inputs'].items())
    replay = json.loads((ROOT / 'build/allocation-acceptance-rerun.json').read_text())
    verify_seal_replay(accepted, replay, before)
    def forbidden(*args, **kwargs): raise AssertionError('Observer attempted new search')
    with patch.object(Native,'analyze',forbidden), patch.object(NeuralNative,'analyze',forbidden), patch.object(quality_search,'search',forbidden):
        value = observer.snapshot()
    cards = value['tools'][:8]
    assert {row['id']:row['state'] for row in cards} == {'allocation-'+key:'complete' for key in
        ('baseline','data','recovery','trace','trace-audit','references','weighting','audit')}
    assert not value['active_tools'] and value['current_work']['state'] == 'complete'
    script = "$connections=Get-NetTCPConnection -LocalPort 18766 -State Listen; @($connections | ForEach-Object { $process=Get-CimInstance Win32_Process -Filter ('ProcessId='+$_.OwningProcess); [pscustomobject]@{address=$_.LocalAddress;port=$_.LocalPort;pid=$_.OwningProcess;command=$process.CommandLine} }) | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell','-NoProfile','-Command',script],check=True,capture_output=True,text=True,timeout=30)
    listeners = json.loads(result.stdout)
    if isinstance(listeners,dict): listeners=[listeners]
    assert len(listeners)==1 and listeners[0]['address']=='127.0.0.1'
    assert '-m congklak.allocation_progress serve --port 18766' in listeners[0]['command']
    after=record()
    assert before==after and 'torch' not in sys.modules
    proof=dict(schema='allocation-dashboard-final-v1',passed=True,before=before,after=after,
               all_bytes_sizes_mtimes_unchanged=True,seal_replay_current_and_passed=True,cards=cards,active_tools=0,current_work=value['current_work'],listener=listeners[0],
               browser_or_HTTP_inspection=False,torch_imported=False,new_searches=0,new_training_updates=0,
               sources={name:signature(ROOT/name)['sha256'] for name in ('congklak/allocation_progress.py',
                       'tools/allocation-dashboard.html','tools/verify-allocation-dashboard.py')})
    publish(ROOT/'build/allocation-dashboard-final.json',proof)
    print(config.canonical(dict(event='allocation_dashboard_verified',passed=True,completed_cards=8,active_tools=0)),flush=True)


if __name__ == '__main__': main()
