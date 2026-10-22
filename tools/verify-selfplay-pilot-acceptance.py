"""Rebuild and verify the immutable seal without changing its bytes or mtime."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config
from congklak.deeper_teacher_v2 import atomic_json


def signature(path):
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)


def main():
    target = ROOT / 'build/selfplay-pilot-acceptance.json'
    before = signature(target)
    spec = importlib.util.spec_from_file_location('selfplay_pilot_recorder', ROOT / 'tools/record-selfplay-pilot.py')
    recorder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        recorder.build()
        recorder.verify()
    after = signature(target)
    assert before == after and 'torch' not in sys.modules
    accepted = json.loads(target.read_text(encoding='utf-8'))
    assert all(signature(ROOT / name)['sha256'] == digest for name, digest in accepted['inputs'].items())
    proof = dict(schema='selfplay-pilot-acceptance-idempotence-v1', passed=True,
                 before=before, after=after, acceptance_bytes_size_mtime_unchanged=True,
                 all_accepted_inputs_current=len(accepted['inputs']), tests_run=accepted['tests_run'],
                 torch_imported=False, no_new_training_or_strength_games=True,
                 recorder_sha256=signature(ROOT / 'tools/record-selfplay-pilot.py')['sha256'],
                 verifier_sha256=signature(Path(__file__))['sha256'])
    output = ROOT / 'build/selfplay-pilot-acceptance-rerun.json'
    if output.exists():
        assert json.loads(output.read_text(encoding='utf-8')) == proof
    else:
        atomic_json(output, proof)
    print(config.canonical(dict(event='selfplay_pilot_seal_rerun_verified', passed=True,
                               inputs=len(accepted['inputs']), tests_run=accepted['tests_run'])), flush=True)


if __name__ == '__main__':
    main()
