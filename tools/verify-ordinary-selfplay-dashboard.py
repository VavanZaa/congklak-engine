"""Verify the completed observer through its backend and listener, without HTTP."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, ordinary_selfplay_pilot as producer
from congklak import ordinary_selfplay_progress as observer, quality_search
from congklak import selfplay_training_pilot as learning
from congklak.native import Native
from congklak.neural_relay import NeuralNative
from congklak.deeper_teacher_v2 import atomic_json


def signature(path):
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)


def main():
    accepted = json.loads((ROOT / 'build/ordinary-selfplay-acceptance.json').read_text())
    files = set(accepted['inputs']) | {'build/ordinary-selfplay-acceptance.json',
                                      'build/ordinary-selfplay-acceptance-rerun.json'}
    def record():
        return {name: signature(ROOT / name) for name in sorted(files)}
    before = record()
    assert all(before[name]['sha256'] == digest for name, digest in accepted['inputs'].items())
    def prohibited(*args, **kwargs):
        raise AssertionError('Observer attempted search, generation or training')
    with patch.object(Native, 'analyze', prohibited), patch.object(NeuralNative, 'analyze', prohibited), \
         patch.object(quality_search, 'search', prohibited), patch.object(producer, 'collect', prohibited), \
         patch.object(producer, 'generate', prohibited), patch.object(learning, 'train', prohibited), \
         patch.object(learning, 'runtime', prohibited):
        value = observer.snapshot()
    cards = value['tools'][:8]
    assert len({card['id'] for card in cards}) == 8
    assert all(card['id'].startswith('ordinary-selfplay-') for card in cards)
    assert {card['id']: card['state'] for card in cards} == {
        'ordinary-selfplay-' + name: ('held' if name == 'training' else 'complete')
        for name in ('diagnosis', 'protection', 'collection', 'rollouts', 'admission', 'sampling', 'training', 'audit')}
    assert not value['active_tools'] and value['current_work']['state'] == 'complete'
    summary = json.loads(producer.OUTPUT.read_text())['summary']
    by_id = {card['id']: card for card in cards}
    assert str(summary['committed_decisions']) + ' durable choices' in by_id['ordinary-selfplay-rollouts']['result']
    assert str(summary['retained_rows']) + ' retained records' in by_id['ordinary-selfplay-admission']['result']
    assert str(accepted['tests_run']) + ' current unique methods passed' in by_id['ordinary-selfplay-audit']['result']
    script = "$c=Get-NetTCPConnection -LocalPort 18766 -State Listen; @($c | ForEach-Object { $p=Get-CimInstance Win32_Process -Filter ('ProcessId='+$_.OwningProcess); [pscustomobject]@{address=$_.LocalAddress;port=$_.LocalPort;pid=$_.OwningProcess;command=$p.CommandLine} }) | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell', '-NoProfile', '-Command', script], check=True,
                            capture_output=True, text=True, timeout=30)
    listeners = json.loads(result.stdout)
    if isinstance(listeners, dict):
        listeners = [listeners]
    assert len(listeners) == 1 and listeners[0]['address'] == '127.0.0.1'
    assert '-m congklak.ordinary_selfplay_progress serve --port 18766' in listeners[0]['command']
    after = record()
    assert before == after and 'torch' not in sys.modules
    proof = dict(schema='ordinary-selfplay-dashboard-final-v1', passed=True,
                 source_inputs={name: signature(ROOT / name)['sha256'] for name in
                                ('congklak/ordinary_selfplay_progress.py', 'tools/ordinary-selfplay-dashboard.html',
                                 'tools/verify-ordinary-selfplay-dashboard.py')},
                 before=before, after=after, all_bytes_sizes_mtimes_unchanged=True,
                 cards=cards, active_tools=0, current_work=value['current_work'],
                 listener=listeners[0], browser_or_HTTP_inspection=False,
                 actual_replay_summary=summary, new_searches=0, new_optimizer_or_runtime_calls=0,
                 torch_imported=False, derivative_evidence_not_an_additional_test_method=True)
    output = ROOT / 'build/ordinary-selfplay-dashboard-final.json'
    if output.exists():
        assert json.loads(output.read_text()) == proof
    else:
        atomic_json(output, proof)
    print(config.canonical(dict(event='ordinary_selfplay_dashboard_verified', passed=True,
                               completed_cards=7, training_held=True, active_tools=0)), flush=True)


if __name__ == '__main__':
    main()
