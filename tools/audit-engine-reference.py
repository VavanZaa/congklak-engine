"""Focused local regression audit alongside pinned Stockfish/OpenSpiel source review."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FOLDER = ROOT/'build/algorithm-reference-audit-20261015'
OUTPUT = FOLDER/'local-checks.json'
METHODS = (
    'tests.test_optimized_search.OptimizedSearchTests.test_03_product_bonus_pass_and_cache_collision_histories',
    'tests.test_optimized_search.OptimizedSearchTests.test_04_exact_remaining_depth_and_fresh_resume',
    'tests.test_optimized_search.OptimizedSearchTests.test_05_cancellation_rejects_partial_iteration',
    'tests.test_optimized_search.OptimizedSearchTests.test_06_options_guards_and_bounded_allocation',
    'tests.quality_search_acceptance.Tests.test_03_both_coefficients_literal_bonus_pass_and_player1',
    'tests.quality_search_acceptance.Tests.test_04_forest_is_independent_and_noise_deterministic',
    'tests.quality_search_acceptance.Tests.test_05_parent_fpu_store_offset_and_declared_scale',
    'tests.quality_search_acceptance.Tests.test_07_initial_and_post_forward_cancellation_no_backup')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    from congklak import config
    accepted = ROOT/'build/midgame-pilot-acceptance.json'
    rerun = ROOT/'build/midgame-pilot-acceptance-rerun.json'
    prior = json.loads(accepted.read_text())
    assert prior['passed'] and len(prior['inputs']) == 2818
    inputs = {**prior['inputs'], **{p.relative_to(ROOT).as_posix():sha(p) for p in (accepted, rerun)}}
    assert len(inputs) == 2820
    for name, digest in inputs.items():
        assert sha(ROOT/name) == digest, 'Prior evidence changed: '+name
    if OUTPUT.exists():
        body = json.loads(OUTPUT.read_text())
        assert body['passed'] and body['covered_checks'] == list(METHODS)
        assert all(sha(ROOT/n) == d for n, d in body['inputs'].items())
        print(config.canonical(dict(event='algorithm_reference_audit_already_complete', passed=True,
              tests_run=body['tests_run'], prior_inputs_preserved=2820)), flush=True)
        return
    for manifest in ('sources.json', 'supplemental-sources.json'):
        path = FOLDER/manifest
        value = json.loads(path.read_text())
        entries = value['files'] if manifest.startswith('supplemental') else [entry for repo in value.values() for entry in repo['files']]
        for entry in entries:
            if 'local' in entry:
                assert sha(ROOT/entry['local']) == entry['sha256']
                inputs[entry['local']] = entry['sha256']
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ('tools/audit-engine-reference.py', 'native/search.cpp', 'native/rules.cpp',
                 'native/neural_relay.cpp', 'native/neural_puct_cpu.cpp', 'congklak/quality_search.py',
                 'congklak/phase6_selfplay.py', 'congklak/phase6_training_v2.py',
                 'congklak/value_pilot_learning.py', 'tests/test_optimized_search.py',
                 'tests/quality_search_acceptance.py'):
        inputs[name] = sha(ROOT/name)
    started = time.monotonic()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in METHODS)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    log = FOLDER/'local-checks.log'
    log.write_text(stream.getvalue(), encoding='utf-8')
    assert all(sha(ROOT/n) == d for n, d in inputs.items()), 'Audit modified accepted or reviewed inputs'
    assert 'torch' not in sys.modules
    body = dict(schema='source-reference-focused-local-audit-v1', passed=result.wasSuccessful(),
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skips=len(result.skipped),
        covered_checks=list(METHODS), prior_inputs_preserved=2820, new_training=False,
        new_strength_games=False, source_modified=False, torch_imported=False,
        external_sources_executed=False, original_AlphaZero_production_code_audited=False,
        scope='Pinned primary-source architecture review and selected existing CPU regressions; no strength comparison or exhaustive equivalence proof.',
        duration_seconds=time.monotonic()-started, log_sha256=sha(log), inputs=dict(sorted(inputs.items())),
        failure_details=[dict(method=str(test), trace=trace) for test, trace in result.failures+result.errors])
    OUTPUT.write_text(json.dumps(body, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    print(config.canonical(dict(event='algorithm_reference_audit_completed', passed=body['passed'],
        tests_run=body['tests_run'], failures=body['failures'], errors=body['errors'],
        skips=body['skips'], prior_inputs_preserved=2820)), flush=True)
    raise SystemExit(not body['passed'])


if __name__ == '__main__':
    main()
