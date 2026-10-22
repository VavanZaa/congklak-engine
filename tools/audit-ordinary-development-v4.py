"""Repair the completed-worker dispatch only, then seal executed audit groups."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import unittest

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


audit = load('ordinary_completed_dispatch_v4', 'tests/test_ordinary_development_audit_v4.py')
original = load('ordinary_original_audit_tool_for_pins', 'tools/audit-ordinary-development.py')
FOLDER = ROOT / 'build/ordinary-development-v1'


def clone_complete_generation(source, target):
    """Rebind only location witnesses; retain every literal/identity input byte."""
    from congklak import ordinary_development_v3 as generation
    before = audit.snapshot(source); shutil.copytree(source, target)
    cert_path = target / 'certificate.pending.json'; cert = audit.read(cert_path)
    prefix = audit.rel(source) + '/'
    assert all(name.startswith(prefix) for name in cert['files'])
    cert['files'] = {audit.rel(target / name[len(prefix):]): digest for name, digest in cert['files'].items()}
    cert_path.write_text(json.dumps(cert, sort_keys=True, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')
    catalog_path = target / 'catalog.json'; catalog = audit.read(catalog_path)
    catalog['certificate_sha256'] = audit.sha(cert_path)
    catalog_path.write_text(json.dumps(catalog, sort_keys=True, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')
    generation.verify(target)
    assert audit.snapshot(source) == before
    for name, (digest, size, _mtime) in before.items():
        if name not in ('certificate.pending.json', 'catalog.json'):
            assert audit.sha(target / name) == digest and (target / name).stat().st_size == size


def seal(work, proof):
    first = audit.read(FOLDER / 'audit-v1.json')
    initial = audit.read(FOLDER / 'independent-v1/methods.json')
    repair4 = audit.read(FOLDER / 'audit-method4-repair-v2.json')
    latest = audit.read(FOLDER / 'audit-recovery-v3-executed.json')
    assert repair4['passed'] is True and proof['passed'] is True
    assert latest['passed'] is False and latest['tests_run'] == 2 and latest['failures'] == 1 and latest['errors'] == 0
    method5 = 'test_05_real_generation_reservation_export_kills_and_prefixes'
    method6 = 'test_06_completed_failclosed_history_and_dashboard'
    method4 = 'test_04_low_active_search_identity_fp32_and_stepzero'
    assert method6 in latest['failure_details'][0]['method']
    assert "KeyError: 'generation_v2'" in latest['failure_details'][0]['trace']
    assert latest['evidence']['publication_recovery']['real_process_kills'] == 11
    assert len(latest['process_kill_cases']) == 11 and len(initial['process_kill_cases']) == 10
    merged = dict(initial['evidence']); merged.update(repair4['evidence']); merged.update(latest['evidence']); merged.update(proof['evidence'])
    fp32 = merged['strict_identity_and_fp32']['torch']['maximum_errors']
    native = merged['native_conformance']; origin = merged['synthetic_provenance']; completed = merged['completed_failclosed_and_history']
    from collections import Counter
    from congklak import ordinary_development as generation, valuepilot_reservations_v2 as data
    selection = audit.read(data.DEFAULT / 'selection.json')
    ordinary_cells = Counter(r['cell'] for r in selection['arena'] if r['stratum'].endswith('/ordinary'))
    actual_cell_counts = {cell: ordinary_cells[cell] for cell in generation.PROTOCOL['cells']}
    assert list(actual_cell_counts.values()) == [3, 3, 0, 2]
    document = ROOT / 'docs/ORDINARY_DEVELOPMENT_AUDIT.md'; assert not document.exists()
    document.write_text(
        '# Ordinary development admission audit\n\n'
        'Independent pre-training acceptance passed. Eight ordinary arena components were selected from 32 protected synthetic development origins. Joint reservation admitted 512 training roots, 128 fit-check roots and 24 arena roots in 664 globally distinct full family/quotient components. All 32 generated origins, including the 24 unused ordinary starts, are excluded from optimizer support. The 32 fit cases remain a subset of the 128 fit-check roots; prior exact guards, forensic cases, trajectory ownership and earlier reservations remain protected.\n\n'
        'The new roots are legal custom boards with exactly two actor choices, balanced stores and 25–74 active seeds. They are not claimed reachable from the standard opening, representative of all ordinary play, or unseen release validation. The pool has eight accepted roots in each band/actor cell. Hash selection chose 3, 3, 0 and 2 ordinary arena roots in 25–49/actor0, 25–49/actor1, 50–74/actor0 and 50–74/actor1 respectively; the eight-root ordinary arena quota is not a per-cell balance guarantee. Fit checks retain historical ancestral training exposure.\n\n'
        'Every accepted origin includes its literal root and all normalized legal-action children used for classification. Observations above four active seeds enter full family/quotient protection, and each accepted origin is disjoint from the full old graph and every other accepted origin. Rejected construction attempts remain a reproducible metadata filter journal and confer no admitted origin role. No target, model prediction, teacher result or game outcome selected a root.\n\n'
        f'Accepted generation used {origin["attempted_counters"]} fixed counter attempts, {origin["metadata_classified_roots"]} classified roots and {origin["generated_classification_transitions"]} literal transition probes; independent audit separately replayed all {origin["independently_replayed_classification_transitions"]} saved probes. These are simulator calls, not games or teacher searches. Logical committed probe counts exclude bounded interruption repeats. The initial dense construction that failed its quota remains retained with source and HOLD evidence and did not publish production memberships.\n\n'
        f'Independent conformance checked {native["corpus_rows"]} row occurrences: all 664 admitted roots, 32 reused exact guards, five forensic roots and 32 boundary/pass fixtures. Scalar and AVX2 policy logits were bit exact for both zero and independent synthetic nonzero branches. The zero branch reproduced control values exactly. Native versus independent float64 algebra stayed within 1e-10 seeds/logits with exact scale-256 scores; single-row and batch results agreed for every occurrence. CPU FP32 maximum errors were {fp32["seeds"]:.12g} seeds, {fp32["logits"]:.12g} logits and {fp32["score_units"]} score units, within unchanged tolerances. All eight original tensors remain unchanged; the separate production step-zero branch has deterministic A/b and zero v/c. No CUDA context was initialized.\n\n'
        'Sixteen independent fixed-128-backup improved-PUCT searches on eight low-active roots preserved actions, visits, backed values and physical work exactly, excluding timings. Owner conformance also checks all 32 reused exact guards. These checks verify the bypass domain and interface, not equal-time playing strength.\n\n'
        'Two incomplete-generation weaknesses were found and retained as scratch reproductions. V1 anchored only the latest chunk, allowing valid-JSON corruption of an earlier child controller; v2 anchored the full committed prefix but could promote an internally consistent corrupted orphan. The actual completed v1 pool remains valid because its final certificate binds every chunk. Its production data, source, reservations, conformance and model were preserved. The separate v3 producer binds the complete ordered prefix and identity and reconstructs every literal record in the single uncommitted orphan before promotion. It never redraws committed facts, refuses earlier/latest committed corruption or missing progress, and refuses legacy identities without fallback. No future v3 production pool is admitted here; its recovery fixtures replay the same counter recipe and family identities and are not additional distinct development data.\n\n'
        f'The accepted recovery run passed eleven real process kills: before/after generation chunk commit, before cursor commit after chunk fsync, before/after generation catalog commit, reservation catalog commit, model commit and certificate commit. The harness verified actual Windows worker identity/lineage before terminating a stable process handle. Immutable committed prefixes preserved bytes, sizes and modification times; only the mutable cursor advanced. Valid orphan recovery rechecked {completed["valid_orphan_literal_attempts_rechecked"]} counter attempts with a bound of {completed["valid_orphan_transition_probe_bound"]} transition probes, then aligned reads used no native work. Corrupted orphan bytes and cursor remained unchanged after refusal. Missing all committed chunks could not reset the budget. Already committed model bytes resumed with the default interpreter without Torch.\n\n'
        'Completed API and CLI checks trapped writers, native constructors, generator transition work, encoders, forwards, search and Torch imports. Pure metadata counter reconstruction remains allowed. Missing/corrupt required evidence refused without recomputation. All 5,950 previously accepted inputs stayed unchanged. The dashboard backend passed read-only and stale-source checks with zero training/game counters; no browser or HTTP inspection was used. Root final acceptance seals the completed dashboard after this audit.\n\n'
        'Six substantive method groups passed across ten method executions. Three numerical/ownership methods are accepted from the first run, method four from an exact legacy-loader assertion repair, method five from v3 recovery, and method six from the matching completed-worker dispatch. Two harness failures remain retained: the old v4 loader correctly raised KeyError(identity) outside an incomplete assertion allowlist; a copied completed-worker call used a legacy specification key and refused. Each repair changed only the relevant assertion or worker dispatch and ran only its affected method. No production code, model, tolerance or reservation changed for either harness repair. Eleven latest recovery kills are accepted; ten earlier kills remain retained, giving 21 actual protocol kills across all audit attempts. Superseded protocol reviews and both corruption reproductions remain pinned.\n\n'
        'Training remains off. No optimizer steps or draws, new teacher labels or arena games ran, and H_v1 remains the default. Next is the fixed value-training pilot with optimizer/RNG checkpoints, resource limits and interruption tests, followed by equal-time H_v1/H8 relay comparisons and the endgame regression guard. This admission does not establish strength or authorize promotion.\n\n'
        'Evidence: [independent audit](../build/ordinary-development-v1/audit.json), [original methods](../build/ordinary-development-v1/independent-v1/methods.json), [method-four repair](../build/ordinary-development-v1/independent-repair-v2/methods.json), [v3 recovery](../build/ordinary-development-v1/independent-recovery-v3/methods.json), [completed-worker repair](../build/ordinary-development-v1/independent-completed-v4/methods.json), and [actual-corpus conformance](../build/ordinary-development-v1/conformance-owner.json).\n',
        encoding='utf-8', newline='\n')
    mapping = dict(first['inputs']); mapping.update(latest['inputs']); mapping.update(original.inputs(work))
    for path in audit.files(FOLDER / 'independent-repair-v2'): mapping[audit.rel(path)] = audit.sha(path)
    for name in ('tools/audit-ordinary-development-v2.py', 'tools/audit-ordinary-development-v3.py', 'tools/audit-ordinary-development-v4.py',
                 'tests/test_ordinary_development_audit_v2.py', 'tests/test_ordinary_development_audit_v3.py', 'tests/test_ordinary_development_audit_v4.py',
                 'congklak/ordinary_development_v3.py', 'tools/generate-ordinary-development-v3.py', 'tests/test_ordinary_development_v3.py',
                 'docs/ORDINARY_DEVELOPMENT_AUDIT.md', 'build/ordinary-development-v1/audit-v1.json',
                 'build/ordinary-development-v1/audit-method4-repair-v2.json', 'build/ordinary-development-v1/audit-recovery-v3-executed.json',
                 'build/ordinary-development-v1/root-publication-checks.json', 'build/ordinary-development-v1/protocol-review-v3.json',
                 'build/ordinary-development-v1/audit-orphan-corruption-proof.json'):
        mapping[name] = audit.sha(ROOT / name)
    for path in audit.files(FOLDER / 'audit-orphan-corruption-probe'): mapping[audit.rel(path)] = audit.sha(path)
    for name, digest in mapping.items(): assert audit.sha(ROOT / name) == digest, name
    methods123 = [name for name in initial['automated_test_methods'] if name.startswith(('test_01_', 'test_02_', 'test_03_'))]
    assert len(methods123) == 3
    record = dict(schema='ordinary-development-independent-audit-v4', passed=True, independent_of_implementation=True,
        automated_test_methods=methods123 + [method4, method5, method6], automated_test_methods_count=6,
        tests_run=10, accepted_methods=6, retained_failed_harness_methods=2, process_kill_cases=latest['process_kill_cases'],
        process_kill_cases_count=11, all_attempted_process_kill_cases=initial['process_kill_cases'] + latest['process_kill_cases'],
        actual_process_kills_all_attempts=21, unresolved_findings=[], evidence=merged, inputs=dict(sorted(mapping.items())),
        attempts=initial['attempts'] + repair4['attempts'] + latest['attempts'] + proof['attempts'],
        counters=dict(training_steps=0, optimizer_draws=0, new_games=0, new_teacher_labels=0), data_and_inference_ready=True,
        ordinary_arena_cell_counts=actual_cell_counts, latest_safe_generation_module='congklak/ordinary_development_v3.py',
        fresh_v3_production_pool_admitted=False, actual_completed_v1_pool_unchanged=True,
        executed_groups=[dict(source='tests/test_ordinary_development_audit.py', proof='build/ordinary-development-v1/independent-v1/methods.json', accepted_methods=methods123),
                         dict(source='tests/test_ordinary_development_audit_v2.py', proof='build/ordinary-development-v1/independent-repair-v2/methods.json', accepted_methods=[method4]),
                         dict(source='tests/test_ordinary_development_audit_v3.py', proof='build/ordinary-development-v1/independent-recovery-v3/methods.json', accepted_methods=[method5]),
                         dict(source='tests/test_ordinary_development_audit_v4.py', proof='build/ordinary-development-v1/independent-completed-v4/methods.json', accepted_methods=[method6])],
        retained_failed_harness_proofs=['build/ordinary-development-v1/independent-v1/methods.json', 'build/ordinary-development-v1/independent-recovery-v3/methods.json'],
        retained_corruption_findings=['build/ordinary-development-v1/audit-prefix-corruption-proof.json', 'build/ordinary-development-v1/audit-orphan-corruption-proof.json'],
        closed_findings=['Positive cursor missing chunks cannot reset progress.', 'Full-prefix v3 hashes reject earlier committed corruption.',
            'Bounded literal v3 orphan reconstruction rejects wrong child controllers before promotion.',
            'Two exact harness repairs accepted after isolated method re-execution.'],
        limitations=first['limitations'], training_started=False, promotion=False)
    audit.save(FOLDER / 'audit-v4-executed.json', record); audit.save(FOLDER / 'audit.json', record)
    print(json.dumps(dict(event='ordinary_development_independent_audit_closed_v4', passed=True,
        accepted_methods=6, method_executions=10, latest_accepted_kills=11, all_actual_kills=21), sort_keys=True), flush=True)


def run():
    report = audit.read(FOLDER / 'audit-recovery-v3-executed.json')
    assert report['passed'] is False and report['tests_run'] == 2 and len(report['process_kill_cases']) == 11
    for name, digest in report['inputs'].items(): assert audit.sha(ROOT / name) == digest, name
    old = (ROOT / 'tests/test_ordinary_development_audit_v3.py').read_text(encoding='utf-8')
    match = "str(ROOT / 'tools/audit-ordinary-development.py'), 'closed-worker'"
    replacement = "str(ROOT / 'tools/audit-ordinary-development-v4.py'), 'closed-worker'"
    assert old.count(match) == 1 and old.replace(match, replacement) == (ROOT / 'tests/test_ordinary_development_audit_v4.py').read_text(encoding='utf-8')
    work = FOLDER / 'independent-completed-v4'; work.mkdir(parents=True, exist_ok=False)
    clone_complete_generation(FOLDER / 'independent-recovery-v3/generation-fresh', work / 'generation-fresh')
    source_folder = work / 'execution-sources'; source_folder.mkdir()
    sources = (Path(__file__), ROOT / 'tests/test_ordinary_development_audit_v4.py')
    source_pins = {audit.rel(path): audit.sha(path) for path in sources}
    for path in sources: shutil.copy2(path, source_folder / path.name)
    audit.WORK = work; method = 'test_06_completed_failclosed_history_and_dashboard'
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([audit.OrdinaryDevelopmentIndependentAudit(method)]))
    proof = dict(schema='ordinary-development-independent-completed-repair-v4', passed=result.wasSuccessful(),
        automated_test_methods=[method], tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        failure_details=[dict(method=str(test), trace=trace) for test, trace in result.failures + result.errors],
        elapsed_seconds=time.monotonic() - started, execution_sources=source_pins, evidence=audit.EVIDENCE,
        attempts=audit.ATTEMPTS, process_kill_cases=audit.KILLS,
        repair='Only route completed-worker to the matching v3-generation audit specification; all other test code identical.',
        generation_fixture='Literal v3 recovery fixture copied; only certificate/catalog relative location witnesses rebound.')
    audit.save(work / 'methods.json', proof)
    for name, digest in source_pins.items(): assert audit.sha(ROOT / name) == digest
    if not result.wasSuccessful(): audit.save(FOLDER / 'audit-completed-v4-failed.json', proof); return 1
    assert result.testsRun == 1 and not audit.KILLS
    seal(work, proof); return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'closed-worker':
        assert len(sys.argv) == 4 and sys.argv[2] == '--spec'
        audit.closed_worker(audit.read(sys.argv[3])); raise SystemExit(0)
    raise SystemExit(run())
