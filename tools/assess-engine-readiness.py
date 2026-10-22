"""Artifact-grounded readiness assessment; does not run engines or training."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config
from congklak.deeper_teacher_v2 import atomic_json

OUTPUT = ROOT / 'build/engine-readiness-current.json'
SOURCES = ('README.md', 'congklak/__main__.py', 'congklak/config.py', 'congklak/oracle.py',
    'congklak/storage.py', 'congklak/quality_analysis.py', 'congklak/value_analysis.py',
    'tools/setup-training.ps1', 'tools/build.ps1', 'congklak.cmd',
    'docs/CONGKLAK_ENGINE_PLAN.md', 'docs/PHASE6_SCOPE_AND_ENGINE_READINESS.md',
    'docs/DATA_PILOT_RESULTS.md', 'docs/NATURAL_STUDY_RESULTS.md', 'docs/K4_ORACLE.md',
    'docs/VALUE_PILOT_RESULTS.md', 'build/value-pilot-acceptance.json',
    'build/value-pilot-data.json', 'build/value-pilot-model-witness.json',
    'build/value-pilot-benchmark-posttraining.json', 'rules/id-betawi-relay7-v0.json',
    'build/pilot-loss-review-baseline.json', 'tools/assess-engine-readiness.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline = json.loads((ROOT/'build/pilot-loss-review-baseline.json').read_text())
    assert baseline['passed'] is True and len(baseline['inputs']) == 2600
    assert all(sha(ROOT/p) == digest for p, digest in baseline['inputs'].items())
    inputs = {name:sha(ROOT/name) for name in SOURCES}
    accepted = json.loads((ROOT/'build/value-pilot-acceptance.json').read_text())
    data = json.loads((ROOT/'build/value-pilot-data.json').read_text())
    benchmark = json.loads((ROOT/'build/value-pilot-benchmark-posttraining.json').read_text())
    witness = json.loads((ROOT/'build/value-pilot-model-witness.json').read_text())
    assert accepted['passed'] is True and accepted['tests_run'] == 24
    assert witness['all_frozen_tensors_bit_exact'] is witness['all_dataset_policy_logits_bit_exact'] is True
    assert benchmark['both_baseline_primary_gates']['value_pilot'] is False
    request_source = (ROOT/'congklak/config.py').read_text()
    assert '"model"' not in request_source.split('allowed=')[1].split('\n')[0]
    assert '"backend"' not in request_source.split('allowed=')[1].split('\n')[0]
    rules = json.loads((ROOT/'rules/id-betawi-relay7-v0.json').read_text())
    assert rules['payoff'] == 'final_store0_minus_store1' and rules['pits_per_side'] == 7
    gates = [
        dict(id='rules_simulator', status='implemented_audited',
             evidence=['rules/id-betawi-relay7-v0.json', 'docs/IMPLEMENTATION_AUDIT.md'],
             meaning='Declared Betawi seven-pit single-round seed-margin contract; unsupported variants remain separate.'),
        dict(id='cpu_analyzer', status='usable_internal_prototype',
             evidence=['README.md', 'congklak/__main__.py', 'congklak/storage.py'],
             meaning='H_v1 CPU search with TT/PVS, legal rankings, trace explanations and completed-depth resume is usable now.'),
        dict(id='data_training_recovery', status='ready_for_bounded_expansion',
             evidence=['build/value-pilot-acceptance.json', 'build/value-pilot-data.json'],
             meaning='Versioned targets, ownership exclusions, explicit weights, checkpoints, exact optimizer recovery and game audit exist.'),
        dict(id='natural_ordinary_coverage', status='insufficient_for_unqualified_scale',
             evidence=['docs/DATA_PILOT_RESULTS.md', 'docs/NATURAL_STUDY_RESULTS.md'],
             meaning='Only61 recent ordinary optimizer roots from59 families; recent strict ordinary75-98 pool produced zero roots; fresh50-74 additions remain small.'),
        dict(id='target_reliability', status='partial',
             evidence=['docs/HORIZON_FINISH_RESULTS.md', 'build/value-pilot-data.json'],
             meaning='Requested H14 completed on six selected disagreements, only five training roots; mixed H8/H10/H14 finite estimates and uncalibrated exposure remain.'),
        dict(id='learned_strength', status='gate_open',
             evidence=['build/value-pilot-benchmark-posttraining.json', 'docs/VALUE_PILOT_RESULTS.md'],
             meaning='Value pilot does not clear both baseline gates; ordinary validation worsened slightly; larger training benefit is unproved.'),
        dict(id='exact_retrieval', status='implemented_separate_interface',
             evidence=['congklak/oracle.py', 'congklak/__main__.py', 'docs/K4_ORACLE.md'],
             meaning='Verified K<=4 lookup exists as explicit operation. A unified product dispatcher still needs an explicit fact/estimate policy; interior lookup is optional separate search work.'),
        dict(id='product_integration', status='partial',
             evidence=['congklak/config.py', 'congklak/quality_analysis.py', 'congklak/value_analysis.py', 'README.md'],
             meaning='Main NDJSON analyzer and experimental learned analyzers are separate; model/backend selection, consistent ranks/explanations and supported mode policy need consolidation.'),
        dict(id='fresh_release_validation', status='not_completed_for_candidate',
             evidence=['build/value-pilot-benchmark-posttraining.json', 'docs/PHASE6_SCOPE_AND_ENGINE_READINESS.md'],
             meaning='Latest strength evaluation reuses12 consumed development families. Freeze a release candidate, validate on fresh protected families, and use sealed release results only to characterize that frozen release.'),
        dict(id='portable_release', status='not_demonstrated',
             evidence=['tools/setup-training.ps1', 'tools/build.ps1', 'congklak.cmd', 'README.md'],
             meaning='Documented local Windows build and machine-specific training environment exist; clean-machine packaging, pinned installation, limits, rollback and release acceptance remain.'),
        dict(id='gui_client', status='optional_scope_decision',
             evidence=['docs/CONGKLAK_ENGINE_PLAN.md', 'README.md'],
             meaning='A board editor/play client is a product choice; it is not required to validate the core CLI or prove search correctness.'),
        dict(id='full_game_proof', status='future_goal_not_first_product_gate',
             evidence=['docs/CONGKLAK_ENGINE_PLAN.md', 'docs/PHASE6_SCOPE_AND_ENGINE_READINESS.md'],
             meaning='Exact root certificates, larger tables and full-start solving remain later stages7/8; a useful strong analyzer does not require solving the full game.')]
    body = dict(schema='congklak-current-readiness-v1', passed=True, assessment_scope='Read-only assessment of local evidence, not a new release or benchmark',
        larger_data=dict(engineering='ready_for_bounded_versioned_collection',
            learning_evidence='not_ready_for_blind_100k_or_long_training',
            next_gate='Use reviewed failure categories to predeclare a modest new natural-history data slice, target budgets, ownership and a fixed learning/strength experiment before scaling.',
            volume_is_not_readiness=True),
        product=dict(internal_analyzer='usable_now_with_declared_H_v1_limits',
            finished_strong_analyzer='not_release_ready', default='H_v1',
            blocks=['ordinary_decision_quality', 'fresh_strength_validation', 'unified_analysis_and_exact_lookup_modes', 'portable_release_and_rollback']),
        gates=gates, no_completion_percentage=True, no_date_or_training_budget_promised=True,
        existing_data_counts=data['groups'], fixed_policy_verified=True, both_baseline_strength_gate_passed=False,
        new_training=False, new_games=False, new_teacher_labels=False, default_promotion=False,
        recommendations_are_unexecuted=True, inputs=inputs)
    text = config.canonical(body) + '\n'
    if OUTPUT.exists():
        assert OUTPUT.read_text(encoding='utf-8') == text, 'Completed readiness report differs'
    else:
        atomic_json(OUTPUT, body)
    assert all(sha(ROOT/p) == digest for p, digest in inputs.items())
    print(config.canonical(dict(event='engine_readiness_assessed', passed=True,
        gates=len(gates), larger_data=body['larger_data']['engineering'],
        product=body['product']['finished_strong_analyzer'])), flush=True)


if __name__ == '__main__':
    main()
