"""Protect the preceding accepted study before neurodiag drift diagnostics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config
from congklak.divergence_study_common import read,sha,publish,current


def main():
    previous=read(ROOT/'build/regret-acceptance.json')
    assert previous['passed'] is previous['policy_fixed'] is previous['all_model_weights_fixed'] is True
    assert previous['training_performed'] is previous['promotion'] is False
    inputs=dict(previous['inputs']);assert len(inputs)==5374
    for name in ('build/regret-acceptance.json','build/regret-acceptance-rerun.json','build/regret-dashboard-final.json'):
        inputs[name]=sha(ROOT/name)
    assert len(inputs)==5377;current(inputs)
    body=dict(schema='neurodiag-prior-immutability-v1',passed=True,inputs_count=5377,prior_accepted_inputs=5374,
        inputs=dict(sorted(inputs.items())),models_fixed=True,policy_fixed=True,previous_allocation='hold',
        existing_neural_diagnostics_authorized=True,frozen_model_forwards_authorized=True,analytic_sensitivities_authorized=True,new_labels_authorized=False,new_searches_authorized=False,
        training_authorized=False,strength_games_authorized=False,selfplay_authorized=False,
        source_pruning_authorized=False,promotion=False)
    publish(ROOT/'build/neurodiag-baseline.json',body)
    print(config.canonical(dict(event='neurodiag_baseline_prepared',passed=True,prior_inputs=5377)),flush=True)


if __name__=='__main__':main()
