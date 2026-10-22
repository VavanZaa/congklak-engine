"""Verify final value-pilot cards and the local listener, without browser/HTTP."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, ordinary_value_progress as observer, ordinary_value_pilot as learning
from congklak import ordinary_value_benchmark as benchmark, ordinary_value_followup as followup, quality_search
from congklak.native import Native
from congklak.neural_relay import NeuralNative
from congklak.deeper_teacher_v2 import atomic_json


def signature(path):
    with path.open('rb') as stream: checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(sha256=checksum, size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)


def main():
    accepted = json.loads((ROOT / 'build/ordinary-value-acceptance.json').read_text())
    files = set(accepted['inputs']) | {'build/ordinary-value-acceptance.json', 'build/ordinary-value-acceptance-rerun.json'}
    def record(): return {name: signature(ROOT / name) for name in sorted(files)}
    before = record()
    assert all(before[name]['sha256'] == digest for name, digest in accepted['inputs'].items())
    def prohibited(*args, **kwargs): raise AssertionError('Observer attempted production work')
    with patch.object(Native, 'analyze', prohibited), patch.object(NeuralNative, 'analyze', prohibited), \
         patch.object(quality_search, 'search', prohibited), patch.object(learning, 'run_labels', prohibited), \
         patch.object(learning, 'train', prohibited), patch.object(learning, 'runtime', prohibited), \
         patch.object(benchmark, 'run', prohibited), patch.object(followup, 'run', prohibited):
        value = observer.snapshot()
    cards = value['tools'][:8]
    assert {card['id']: card['state'] for card in cards} == {
        'ordinary-value-' + key: 'complete'
        for key in ('reservation', 'labels', 'data', 'training', 'model', 'benchmark', 'allocation', 'audit')}
    assert not value['active_tools'] and value['current_work']['state'] == 'complete'
    script = "$c=Get-NetTCPConnection -LocalPort 18766 -State Listen; @($c | ForEach-Object { $p=Get-CimInstance Win32_Process -Filter ('ProcessId='+$_.OwningProcess); [pscustomobject]@{address=$_.LocalAddress;port=$_.LocalPort;pid=$_.OwningProcess;command=$p.CommandLine} }) | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell', '-NoProfile', '-Command', script], check=True,
                            capture_output=True, text=True, timeout=30)
    listeners = json.loads(result.stdout)
    if isinstance(listeners, dict): listeners = [listeners]
    assert len(listeners) == 1 and listeners[0]['address'] == '127.0.0.1'
    assert '-m congklak.ordinary_value_progress serve --port 18766' in listeners[0]['command']
    after = record()
    assert before == after and 'torch' not in sys.modules
    sources = ('congklak/ordinary_value_progress.py', 'tools/ordinary-value-dashboard.html',
               'tools/verify-ordinary-value-dashboard.py')
    proof = dict(schema='ordinary-value-dashboard-final-v1', passed=True,
                 source_inputs={name: signature(ROOT / name)['sha256'] for name in sources},
                 before=before, after=after, all_bytes_sizes_mtimes_unchanged=True,
                 cards=cards, active_tools=0, current_work=value['current_work'], listener=listeners[0],
                 browser_or_HTTP_inspection=False, new_searches=0, new_optimizer_or_runtime_calls=0,
                 torch_imported=False, derivative_evidence_not_an_additional_test_method=True)
    output = ROOT / 'build/ordinary-value-dashboard-final.json'
    if output.exists(): assert json.loads(output.read_text()) == proof
    else: atomic_json(output, proof)
    print(config.canonical(dict(event='ordinary_value_dashboard_verified', passed=True, completed_cards=8, active_tools=0)), flush=True)


if __name__ == '__main__': main()
