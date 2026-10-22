"""Verify that rebuilding review acceptance leaves its bytes and mtime intact."""
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
from congklak.qstability_common import publish


def signature(path):
    with path.open('rb') as stream: checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(sha256=checksum, size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)


def main():
    target = ROOT / 'build/qstability-acceptance.json'
    before = signature(target)
    spec = importlib.util.spec_from_file_location('qstability_recorder', ROOT / 'tools/record-qstability.py')
    recorder = importlib.util.module_from_spec(spec); spec.loader.exec_module(recorder)
    with contextlib.redirect_stdout(io.StringIO()): recorder.build(); body = recorder.verify()
    after = signature(target)
    assert before == after and 'torch' not in sys.modules
    proof = dict(schema='qstability-acceptance-rerun-v1', passed=True, before=before, after=after,
                 acceptance_bytes_size_mtime_unchanged=True, all_inputs_current=len(body['inputs']), tests_run=body['tests_run'],
                 recorder_sha256=signature(ROOT / 'tools/record-qstability.py')['sha256'],
                 verifier_sha256=signature(Path(__file__))['sha256'], torch_imported=False, new_training_or_games=False)
    publish(ROOT / 'build/qstability-acceptance-rerun.json', proof)
    print(config.canonical(dict(event='qstability_seal_rerun_verified', passed=True,
                               inputs=len(body['inputs']), tests_run=body['tests_run'])), flush=True)


if __name__ == '__main__': main()
