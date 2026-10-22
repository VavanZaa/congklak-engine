"""Protect the preceding accepted study before prospective Q-stability assessment."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config
from congklak.divergence_study_common import read,sha,publish,current


def main():
    previous=read(ROOT/'build/capture-acceptance.json')
    assert previous['passed'] is previous['policy_fixed'] is previous['all_model_weights_fixed'] is True
    assert previous['training_performed'] is previous['promotion'] is False
    inputs=dict(previous['inputs']);assert len(inputs)==5047
    for name in ('build/capture-acceptance.json','build/capture-acceptance-rerun.json','build/capture-dashboard-final.json'):
        inputs[name]=sha(ROOT/name)
    assert len(inputs)==5050;current(inputs)
    body=dict(schema='qstability-prior-immutability-v1',passed=True,inputs_count=5050,prior_accepted_inputs=5047,
        inputs=dict(sorted(inputs.items())),models_fixed=True,policy_fixed=True,previous_allocation='hold',
        prospective_qstability_assessment_authorized=True,new_labels_authorized=True,new_searches_authorized=True,
        training_authorized=False,strength_games_authorized=False,selfplay_authorized=False,
        source_pruning_authorized=False,promotion=False)
    publish(ROOT/'build/qstability-baseline.json',body)
    print(config.canonical(dict(event='qstability_baseline_prepared',passed=True,prior_inputs=5050)),flush=True)


if __name__=='__main__':main()
