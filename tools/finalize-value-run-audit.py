"""Seal independent methods and retained failed attempts; never optimize/search."""
from __future__ import annotations
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import neurodiag_common as common

FOLDER = ROOT / 'build/value-run-v1'
CAMPAIGNS = {
    'audit-training-v1': 3,
    'audit-required-v1': 1,
    'audit-arena-recovery-v1': 1,
    'audit-cases-recovery-v1': 1,
    'audit-classifier-v1': 1,
    'audit-final-cases-v1': 1,
    'audit-dashboard-v1': 1,
    'audit-final-v1': 4,
}

def relative(path): return Path(path).resolve().relative_to(ROOT).as_posix()

def main():
    baseline = common.read(FOLDER / 'baseline.json')
    common.require(baseline['passed'] and len(baseline['inputs']) == baseline['inputs_count'] == 8898, 'Prior baseline differs')
    common.current(baseline['inputs'])
    evidence = {}; methods = []; witnesses = []; reports = []
    for folder, count in CAMPAIGNS.items():
        path = FOLDER / folder / 'methods.json'; body = common.read(path)
        common.require(body['passed'] and body['source_unchanged'] and body['independent_of_implementation'] and
                       body['tests_run'] == count and body['failed'] == body['errors'] == 0, 'Independent campaign failed: ' + folder)
        names = re.findall(r'^(test_\w+) \(', (path.parent / 'methods.log').read_text(), re.MULTILINE)
        common.require(len(names) == count, 'Method log count differs: ' + folder)
        reports.append(dict(path=relative(path), sha256=common.sha(path), methods=names))
        for name in names: methods.append(dict(method=name, proof=relative(path), proof_sha256=common.sha(path)))
        for name, checksum in body['inputs'].items():
            source = ROOT / name
            if source.is_file() and common.sha(source) == checksum:
                actual = source
            else:
                matches = [p for p in FOLDER.glob('audit-source-v*/' + name) if p.is_file() and common.sha(p) == checksum]
                common.require(bool(matches), 'Executed source witness is missing: ' + name)
                actual = matches[0]
            witnesses.append(dict(campaign=folder, original_path=name, executed_sha256=checksum,
                                  retained_exact_path=relative(actual)))
        for key, value in body['evidence'].items():
            common.require(key not in evidence, 'Independent evidence key repeated: ' + key)
            evidence[key] = value
    common.require(len(methods) == 13 and len({r['method'] for r in methods}) == 13, 'Expected thirteen independent methods')
    decision = common.read(FOLDER / 'decision.json')
    common.current(decision['inputs'])
    common.require(decision['evidence_complete'] and decision['production_steps'] == 400 and
                   decision['production_draws'] == 51200 and decision['arena_games'] == 192 and not decision['promotion'],
                   'Final decision coverage or policy differs')
    common.require(decision['pilot_promising'] == all(decision['gates'].values()), 'Decision gates differ')
    kills = evidence['actual_case_recovery']['prefix_proofs'] + evidence['actual_arena_recovery']['prefix_proofs']
    kills = [r['killed'] for r in kills]
    common.require(len(kills) == 6 and all(r['actual_process_kill'] for r in kills), 'Actual process kills incomplete')
    owner = common.read(FOLDER / 'arena-owner-acceptance.json')['all_owner_engineering_totals']
    common.require(owner['decisions'] == 107 and owner['requested_fixed_backups'] == 13696 and owner['terminal_games'] == 4, 'Owner retained work differs')
    independent_searches = (evidence['actual_case_recovery']['fresh_and_resumed_physical_searches'] +
                            evidence['actual_arena_recovery']['fresh_and_resumed_physical_searches'])
    engineering = dict(optimizer_steps=0, optimizer_draws=0, new_teacher_labels=0,
        owner_case_smoke_fixed_searches=4, owner_case_smoke_backups=512,
        owner_arena_fixed_searches=107, owner_arena_fixed_backups=13696,
        independent_case_recovery_searches=evidence['actual_case_recovery']['fresh_and_resumed_physical_searches'],
        independent_arena_recovery_searches=evidence['actual_arena_recovery']['fresh_and_resumed_physical_searches'],
        independent_fixed_searches=independent_searches, independent_fixed_backups=independent_searches * 128,
        total_engineering_fixed_searches=111 + independent_searches,
        total_engineering_fixed_backups=14208 + independent_searches * 128,
        owner_terminal_fixture_games=4, independent_terminal_fixture_games=10, total_terminal_fixture_games=14,
        independent_actual_process_kills=6, production_training_cooperative_stops=1,
        production_arena_unclean_invocations=evidence['all_arena_games']['unclean_full_reserved_charges'],
        production_arena_unreported_inflight_work_possible=True,
        independent_FP32_corpus_occurrences=733, independent_native_policy_comparisons=1466,
        watchdog_zero_search_managed_workers=3, watchdog_owner_atomic_renames=3000,
        independent_test_method_executions=13, independent_failed_method_executions=0)
    common.require(engineering['total_engineering_fixed_searches'] == 391 and
                   engineering['total_engineering_fixed_backups'] == 50048, 'Recorded engineering work differs')
    fit = common.read(FOLDER / 'audit-training-v1/independent-fit.json')
    all_fit = fit['metrics']['all']; capture = fit['metrics']['capture']
    lines = [
        '# Independent full value-pilot audit', '',
        'The complete experiment is accepted as trustworthy evidence. Experimental success is separate from this integrity result. '
        'The final candidate remains step 400, the original network and policy stayed frozen, and H_v1 remains the default. No automatic retraining, data expansion or promotion follows this run.', '',
        'The production run executed 400 AdamW updates and 51,200 seeded replacement draws among 512 admitted roots. '
        'All 21 initialization/scheduled checkpoints, optimizer moments, sample prefixes and Python/global Torch/dedicated sampler RNG states were checked independently without executing another optimizer update. '
        'A real stop at step 20 resumed the remaining 380 updates with the committed checkpoint bytes, sizes and modification times unchanged. All eight original tensors (12,872 parameters) remained exact; only the 2,065 branch parameters changed. CPU FP32 used no CUDA context.', '',
        'Independent 128-root fit diagnostics failed the predeclared gate. Normalized Huber was '
        f"{all_fit['candidate']['normalized_huber']:.11f} versus control {all_fit['control']['normalized_huber']:.11f}; "
        f"overall seed MAE was {all_fit['candidate']['seed_mae']:.8f} versus {all_fit['control']['seed_mae']:.8f}; "
        f"capture MAE was {capture['candidate']['seed_mae']:.8f} versus {capture['control']['seed_mae']:.8f}. "
        'These roots were excluded from this update but have historical ancestral exposure; they are not unseen validation.', '',
        'The learned export passed 733 occurrence scalar/AVX2/literal-float64 and independent Torch FP32 checks, with 1,466 bit-exact native policy comparisons and no added feature simulations per evaluation. '
        'FP32 maximum seed error was 0.0000140843, maximum logit error 0.000000618382 and score-unit error zero. All 512 timed cases and 138 fixed searches (17,664 completed backups) were independently checked. '
        'The candidate chose an optimal action on all 32 reused exact roots at both clocks with no fallback. Sixteen exact roots are forced moves. Every tested low-active control/candidate fixed-work pair matched recommendations, visits, Q values and physical work exactly. '
        'The saved H8 mean regret comparisons were candidate/control 0.625/0.75 at 25 ms, 0.71875/0.71875 at 100 ms and 0.65625/0.90625 at fixed 128 backups. Overshoots remain in the raw records.', '',
        f"The full arena contains {decision['arena_games']} games, {decision['arena_terminal_games']} terminal results, "
        f"{decision['arena_fallbacks']} fallbacks and {decision['arena_capped_games']} caps. "
        'Both profiles faced H_v1 and H8 relay CPU in both seats on all 24 frozen families at 25 ms whole decisions. '
        'Full actual trajectories were checked against all 512 optimizer components, 128 fit components and their complete historical aliases, plus bridges among arena families. '
        f"Ownership passed: {decision['ownership']['ownership_passed']}. Strength interpretation allowed: {decision['strength_interpretation_allowed']}. "
        f"The retained ownership result has {len(decision['ownership']['support_or_fit_hits'])} support/fit observations, "
        f"{len(decision['ownership']['cross_arena_component_hits'])} cross-family quotient bridges and "
        f"{len(decision['ownership']['cross_arena_alias_component_hits'])} historical-component alias bridges. "
        'When ownership/terminal/fallback requirements permit it, six intervals are recomputed from the same 4,096 draws of 24 whole-family clusters, seed 20261019. Invalid ownership suppresses that independence-based cluster analysis; raw descriptive results are retained without filtering, replacing families or merging clusters after outcomes.', '',
        'Six actual Windows worker kills covered before/after decision commits and before/after game commits. '
        'Fresh and resumed fixed-work actions, visits, Q values, physical work and terminal trajectories matched exactly, and all committed raw byte/size/mtime prefixes were preserved. '
        'These recovery fixtures executed 280 native searches and 35,840 backups; with retained owner fixtures, engineering work totals 391 fixed searches, 50,048 backups and 14 terminal fixture games. No engineering optimizer updates occurred.', '',
        'Two production infrastructure failures and all their artifacts remain retained: an initially uncaptured worker error after a derived progress write (the Windows open-target replacement problem was separately reproduced), and a captured watchdog namespace-scan race when an atomic pending file disappeared. '
        'Each unclean invocation was charged its full durable five-minute allowance. V3 observation avoids opening mutable arena files and labels directory counts provisional; immutable completion evidence supplies exact final counts. '
        'A versioned enforcement-only adapter ignores only vanished pending-file samples and propagates missing committed paths and file/directory permission errors. It retains the original worker/model/search, deadlines, four-GiB private-commit limit, five-GiB namespace quota and 90-minute cumulative ledger. '
        'Actual adapter source/lease/token/watchdog attestations were fsynced and bound to permits before worker release. Both interrupted production prefixes were preserved. Unreported in-flight native work is possible on unclean exits and is covered conservatively by the full reservation charges.', '',
        'Completed production training, cases, arena, CLI and dashboard readers ran under forbidden Torch/writer/engine/search traps and preserved bytes, sizes and modification times. '
        'Missing required checkpoint/certificate/head/prefix evidence refused before those traps. The actual dashboard listener and startup source hash were verified through Windows process/listener metadata and a direct backend snapshot; no browser or HTTP inspection was used.', '',
        f"Final failed research gates: {', '.join(decision['failed_gates']) or 'none'}. "
        f"Candidate quarantined: {decision['candidate_quarantined']}. Promotion: false. "
        'Ordinary arena starts are synthetic custom positions with no proven opening reachability. Fit ancestry, known endgames and reused forensic positions limit generalization claims.', '',
        'All earlier accepted 8,898 inputs were checked unchanged. Thirteen independent automated methods have exact source-to-method witnesses; older executed audit sources, failed reader probes, original supervisors and enforcement helper versions are retained rather than overwritten.',
    ]
    doc = ROOT / 'docs/VALUE_RUN_AUDIT.md'
    document = '\n'.join(lines) + '\n'
    if doc.exists(): common.require(doc.read_text(encoding='utf-8') == document, 'Audit document already differs')
    else: doc.write_text(document, encoding='utf-8', newline='\n')
    inputs = dict(baseline['inputs'])
    for path in (FOLDER, ROOT / 'data/value-run-v1', ROOT / 'data/value-training-v1/training', ROOT / 'data/value-training-v1/resources'):
        for file in path.rglob('*'):
            if (file.is_file() and file not in (FOLDER / 'audit.json', FOLDER / 'acceptance.json', FOLDER / 'stage-status.json') and
                    file != FOLDER / 'stage-status.json.pending' and
                    file.name != 'writer.lock' and '__pycache__' not in file.parts and file.suffix != '.pyc'):
                inputs.update(common.pins([file]))
    for pattern in ('congklak/value_run*.py', 'tools/*value-run*.py', 'tools/*value-arena*.py', 'tools/value-run-dashboard*.html', 'tests/test_value_run*.py'):
        inputs.update(common.pins(ROOT.glob(pattern)))
    inputs.update(common.pins([doc, ROOT / 'docs/VALUE_RUN_RESULTS.md']))
    repair = common.read(FOLDER / 'dashboard-read-repair.json')
    for name, checksum in repair['inputs'].items():
        common.require(name not in inputs or inputs[name] == checksum, 'Repair witness changed: ' + name)
        inputs[name] = checksum
    common.current(inputs)
    body = dict(schema='value-run-independent-audit-v1', passed=True, independent_of_implementation=True,
        unresolved_findings=[], production_counters=dict(training_steps=400, optimizer_draws=51200, new_games=192, new_teacher_labels=0),
        engineering_counters=engineering, automated_test_methods=methods, process_kill_cases=kills,
        method_reports=reports, executed_source_witnesses=witnesses, evidence=evidence,
        production_policy_and_models_fixed=True, independent_optimizer_steps=0,
        previous_accepted_inputs_preserved=8898, experimental_gates_passed=decision['pilot_promising'],
        failed_experimental_gates=decision['failed_gates'], candidate_quarantined=decision['candidate_quarantined'],
        strength_interpretation_allowed=decision['strength_interpretation_allowed'], promotion=False,
        retained_infrastructure_failures=True, unknown_inflight_work_fully_budget_charged=True,
        browser_or_HTTP_inspection=False, inputs=dict(sorted(inputs.items())))
    common.publish(FOLDER / 'audit.json', body)
    print(json.dumps(dict(passed=True, independent_methods=len(methods), actual_worker_kills=len(kills),
        pilot_promising=decision['pilot_promising'], inputs_count=len(inputs)), sort_keys=True))

if __name__ == '__main__': main()
