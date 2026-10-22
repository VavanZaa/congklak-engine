"""Protect the preceding accepted study before within-cell weighting diagnostics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config
from congklak.divergence_study_common import read,sha,publish,current


def main():
    previous=read(ROOT/'build/allocation-acceptance.json')
    assert previous['passed'] is previous['policy_fixed'] is previous['all_model_weights_fixed'] is True
    assert previous['training_performed'] is previous['promotion'] is False
    inputs=dict(previous['inputs']);assert len(inputs)==4543
    for name in ('build/allocation-acceptance.json','build/allocation-acceptance-rerun.json','build/allocation-dashboard-final.json'):
        inputs[name]=sha(ROOT/name)
    assert len(inputs)==4546;current(inputs)
    body=dict(schema='stratum-prior-immutability-v1',passed=True,inputs_count=4546,prior_accepted_inputs=4543,
        inputs=dict(sorted(inputs.items())),models_fixed=True,policy_fixed=True,previous_allocation='hold',
        within_cell_weighting_assessment_authorized=True,bounded_saved_root_labels_authorized=True,
        training_authorized=False,strength_games_authorized=False,selfplay_authorized=False,
        source_pruning_authorized=False,promotion=False)
    publish(ROOT/'build/stratum-baseline.json',body)
    print(config.canonical(dict(event='stratum_baseline_prepared',passed=True,prior_inputs=4546)),flush=True)


if __name__=='__main__':main()
