"""Repair one independent legacy-loader assertion; retain all v1 evidence."""
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


original = load('ordinary_audit_original_tool', 'tools/audit-ordinary-development.py')
audit = load('ordinary_development_repaired_test', 'tests/test_ordinary_development_audit_v2.py')
FOLDER = ROOT / 'build/ordinary-development-v1'


def main():
    old = audit.read(FOLDER / 'audit-v1.json'); methods = audit.read(FOLDER / 'independent-v1/methods.json')
    method = 'test_04_low_active_search_identity_fp32_and_stepzero'
    assert old['passed'] is False and methods['errors'] == 1 and methods['failures'] == 0
    assert len(methods['failure_details']) == 1
    failure = methods['failure_details'][0]
    assert method in failure['method'] and "KeyError: 'identity'" in failure['trace']
    assert len(methods['process_kill_cases']) == 10
    for name, digest in old['inputs'].items(): assert audit.sha(ROOT / name) == digest, name
    # Exact source-diff assertion: no implementation, tolerance or other test
    # changed to repair the old v4 loader's deliberate missing-identity refusal.
    before = (ROOT / 'tests/test_ordinary_development_audit.py').read_text(encoding='utf-8')
    replacement = "        with self.assertRaises((NativeError, ValueError)): OldModel(self.initial / 'model.json')"
    after = "        with self.assertRaises(KeyError) as refusal: OldModel(self.initial / 'model.json')\n        self.assertEqual(refusal.exception.args, ('identity',))"
    assert before.count(replacement) == 1
    assert before.replace(replacement, after) == (ROOT / 'tests/test_ordinary_development_audit_v2.py').read_text(encoding='utf-8')
    work = FOLDER / 'independent-repair-v2'; work.mkdir(parents=True, exist_ok=False)
    source_folder = work / 'execution-sources'; source_folder.mkdir()
    for path in (Path(__file__), ROOT / 'tests/test_ordinary_development_audit_v2.py'):
        shutil.copy2(path, source_folder / path.name)
    source_pins = {audit.rel(p): audit.sha(p) for p in (Path(__file__), ROOT / 'tests/test_ordinary_development_audit_v2.py')}
    audit.WORK = work
    suite = unittest.TestSuite([audit.OrdinaryDevelopmentIndependentAudit(method)])
    started = time.monotonic(); result = unittest.TextTestRunner(verbosity=2).run(suite)
    proof = dict(schema='ordinary-development-independent-repair-method-v2', passed=result.wasSuccessful(),
        automated_test_methods=[method], tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        failure_details=[dict(method=str(test), trace=trace) for test, trace in result.failures + result.errors],
        elapsed_seconds=time.monotonic() - started, execution_sources=source_pins,
        evidence=audit.EVIDENCE, attempts=audit.ATTEMPTS, process_kill_cases=audit.KILLS,
        repair='Accept only the actual KeyError with args(identity) from the frozen old v4 loader; all other audit code and tolerances identical.')
    audit.save(work / 'methods.json', proof)
    for name, digest in source_pins.items(): assert audit.sha(ROOT / name) == digest
    if not result.wasSuccessful(): audit.save(FOLDER / 'audit-repair-v2-failed.json', proof); return 1
    assert result.testsRun == 1 and not audit.KILLS
    if '--method-only' in sys.argv:
        audit.save(FOLDER / 'audit-method4-repair-v2.json', proof)
        print(json.dumps(dict(event='ordinary_development_method4_repair_v2', passed=True,
            methods=1, canonical_audit_published=False), sort_keys=True), flush=True)
        return 0
    merged = dict(methods['evidence']); merged.update(audit.EVIDENCE)
    # The narrative is written only after actual independent execution passes.
    provenance = merged['synthetic_provenance']; numeric = merged['native_conformance']
    fp32 = merged['strict_identity_and_fp32']['torch']['maximum_errors']
    document = ROOT / 'docs/ORDINARY_DEVELOPMENT_AUDIT.md'; assert not document.exists()
    document.write_text(
        '# Ordinary development admission audit\n\n'
        'Independent pre-training acceptance passed. Eight ordinary arena components were selected from 32 protected synthetic development origins. Joint reservation admitted 512 training roots, 128 fit-check roots and 24 arena roots in 664 globally distinct full family/quotient components. All 32 generated origins, including the 24 unused ordinary starts, are excluded from this optimizer support. The 32 fit cases remain a subset of the 128 fit-check roots, and earlier exact guards and forensic positions retain their protected roles.\n\n'
        'The new roots are legal custom boards with exactly two legal choices for the actor, balanced stores and 25–74 active seeds. They are not claimed reachable from the standard opening, representative of all ordinary play, or unseen release validation. There are eight accepted roots per active-band/actor cell. Every accepted origin includes its root and every normalized legal-action child used for classification; ownership uses observations above four active seeds. Their complete eligible quotient sets are disjoint from the full prior graph and every other accepted origin. Rejected construction attempts remain a metadata filter journal and confer no admitted development role. No target, prediction, teacher-search result or game outcome selected a root.\n\n'
        f'Accepted generation used {provenance["attempted_counters"]} deterministic counter attempts and {provenance["generated_classification_transitions"]} literal classification transitions. Independent audit replayed all {provenance["independently_replayed_classification_transitions"]} saved transitions separately. These are simulator calls, not self-play games or teacher searches. Logical committed probe counts exclude bounded repeats of an interrupted uncommitted chunk. The initial dense construction failed its quota and remains retained with its source and HOLD evidence; it published no production memberships.\n\n'
        f'Independent conformance checked {numeric["corpus_rows"]} row occurrences: all 664 admitted roots, 32 reused exact guards, five forensic roots and 32 boundary/pass fixtures. Scalar and AVX2 native backends preserved policy logits bit for bit for both the zero branch and a separate synthetic nonzero branch. The zero branch reproduced base values exactly. Native versus independent float64 algebra stayed within the fixed 1e-10 seed/logit tolerance, with exact scale-256 scores. CPU FP32 maximum errors were {fp32["seeds"]:.12g} seeds, {fp32["logits"]:.12g} logits and {fp32["score_units"]} score units, within the unchanged limits. Every row passed single-row versus batch parity. All eight original tensors remain unchanged. The separately identified production step-zero branch has deterministic A/b and zero v/c; no CUDA context was initialized.\n\n'
        'Sixteen independent fixed-128-backup improved-PUCT searches on eight low-active roots preserved actions, visits, backed values and physical work exactly, excluding timings. The owner conformance additionally checks 32 reused exact guards and the boundary set. These are domain-bypass checks, not equal-time strength evidence.\n\n'
        'Independent review found an incomplete-v1 cursor bound only its latest chunk. A retained scratch reproduction changed an earlier accepted child controller in valid JSON; metadata replay incorrectly accepted it. The accepted completed v1 pool remains valid because its final certificate binds every chunk. Its source/data, actual reservations and conformance remain unchanged. A separate v2 producer binds the full ordered committed-prefix name/SHA256 list and digest plus identity. It rejects earlier/latest chunk changes, positive-cursor chunk loss, missing anchors and legacy v1 namespaces without unsafe fallback. Future fresh generation must use v2 and recheck current ownership; no v2 production pool is admitted here. Its recovery fixtures replay the exact same counter recipe and family names and are not additional distinct development data.\n\n'
        'Ten real process-kill cases passed: before/after v2 generation chunk commit, v2 generation catalog commit, reservation catalog commit, model commit and initialization-certificate commit. The harness verified the actual Windows worker script and parent lineage before terminating its process handle. All committed immutable prefixes preserved bytes, sizes and modification times; the mutable identity-bound generation cursor was allowed to advance. Missing all committed chunks beneath a positive cursor and valid-JSON corruption of an earlier committed child refused before new generation, without resetting the counter budget. Model bytes already committed resumed with the default interpreter without Torch. Completed API and CLI routes passed with writers, native constructors, generator transition work, encoders, forwards, search and Torch imports trapped; pure metadata counter reconstruction remains allowed. Missing/corrupt required evidence refused without recomputation.\n\n'
        'Six substantive methods passed across seven method executions. The original run passed five methods and all ten kills; method four stopped at a harness allowlist that omitted the old v4 loader’s expected KeyError for its missing identity field. The failed report and exact executed sources remain retained. A separate v2 harness changes only that assertion, accepts precisely KeyError with args (identity), and reruns only method four. It passed the CPU FP32, initialization/identity and fixed-search checks. No implementation, tolerance, model or source binding changed for this repair.\n\n'
        'All 5,950 previously accepted inputs remain unchanged. The dashboard backend passed read-only/stale-source checks and shows zero training, optimizer draws, teacher labels and games. No browser or HTTP inspection was used; root final acceptance seals dashboard completion after this audit.\n\n'
        'Training remains off and H_v1 remains the default. Next is the fixed value-training pilot with optimizer/RNG checkpoints, resource limits and process-interruption checks, followed by the planned equal-time H_v1/H8 relay comparisons and endgame regression guard. This admission does not establish playing strength or authorize promotion.\n\n'
        'Evidence: [independent audit](../build/ordinary-development-v1/audit.json), [original executed methods and retained error](../build/ordinary-development-v1/independent-v1/methods.json), [repaired method](../build/ordinary-development-v1/independent-repair-v2/methods.json), and [owner actual-corpus conformance](../build/ordinary-development-v1/conformance-owner.json).\n',
        encoding='utf-8', newline='\n')
    mapping = dict(old['inputs']); mapping.update(original.inputs(work)); mapping.update(source_pins)
    mapping.update({audit.rel(p): audit.sha(p) for p in (FOLDER / 'audit-v1.json', FOLDER / 'root-publication-checks.json', document)})
    accepted_v1 = [name for name in methods['automated_test_methods'] if name != method]
    assert len(accepted_v1) == 5
    record = dict(schema='ordinary-development-independent-audit-v2', passed=True, independent_of_implementation=True,
        automated_test_methods=accepted_v1 + [method], automated_test_methods_count=6, tests_run=7,
        accepted_methods=6, retained_failed_harness_methods=1, process_kill_cases=methods['process_kill_cases'],
        process_kill_cases_count=10, inputs=dict(sorted(mapping.items())), unresolved_findings=[], evidence=merged,
        attempts=methods['attempts'] + audit.ATTEMPTS, counters=dict(training_steps=0, optimizer_draws=0, new_games=0, new_teacher_labels=0),
        data_and_inference_ready=True, new_distinct_generation_components=32, new_distinct_arena_ordinary_components=8,
        generation_v2_production_pool_admitted=False, old_completed_production_inputs_unchanged=True,
        executed_groups=[dict(source='tests/test_ordinary_development_audit.py', proof='build/ordinary-development-v1/independent-v1/methods.json', accepted_methods=accepted_v1),
                         dict(source='tests/test_ordinary_development_audit_v2.py', proof='build/ordinary-development-v1/independent-repair-v2/methods.json', accepted_methods=[method])],
        closed_findings=['Incomplete generation high-water loss fixed by identity-bound cursor.',
            'Incomplete-v1 earlier-chunk corruption fixed in separate full-prefix v2 producer; certified completed v1 pool retained.',
            'Harness legacy identity-refusal allowlist repaired exactly in v2; original error retained.'],
        limitations=old['limitations'], repair=proof['repair'], training_started=False, promotion=False)
    audit.save(FOLDER / 'audit-v2-executed.json', record); audit.save(FOLDER / 'audit.json', record)
    print(json.dumps(dict(event='ordinary_development_independent_audit_v2', passed=True,
        accepted_methods=6, method_executions=7, actual_process_kills=10), sort_keys=True), flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
