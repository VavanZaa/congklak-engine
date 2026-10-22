"""Protect the preceding accepted study before regret drift diagnostics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config
from congklak.divergence_study_common import read,sha,publish,current


def main():
    previous=read(ROOT/'build/qstability-acceptance.json')
    assert previous['passed'] is previous['policy_fixed'] is previous['all_model_weights_fixed'] is True
    assert previous['training_performed'] is previous['promotion'] is False
    inputs=dict(previous['inputs']);assert len(inputs)==5249
    for name in ('build/qstability-acceptance.json','build/qstability-acceptance-rerun.json','build/qstability-dashboard-final.json'):
        inputs[name]=sha(ROOT/name)
    assert len(inputs)==5252;current(inputs)
    body=dict(schema='regret-prior-immutability-v1',passed=True,inputs_count=5252,prior_accepted_inputs=5249,
        inputs=dict(sorted(inputs.items())),models_fixed=True,policy_fixed=True,previous_allocation='hold',
        existing_regret_diagnostics_authorized=True,new_labels_authorized=False,new_searches_authorized=False,
        training_authorized=False,strength_games_authorized=False,selfplay_authorized=False,
        source_pruning_authorized=False,promotion=False)
    publish(ROOT/'build/regret-baseline.json',body)
    print(config.canonical(dict(event='regret_baseline_prepared',passed=True,prior_inputs=5252)),flush=True)


if __name__=='__main__':main()
