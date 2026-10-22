"""Protect the preceding accepted study before capture drift diagnostics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config
from congklak.divergence_study_common import read,sha,publish,current


def main():
    previous=read(ROOT/'build/stratum-acceptance.json')
    assert previous['passed'] is previous['policy_fixed'] is previous['all_model_weights_fixed'] is True
    assert previous['training_performed'] is previous['promotion'] is False
    inputs=dict(previous['inputs']);assert len(inputs)==4920
    for name in ('build/stratum-acceptance.json','build/stratum-acceptance-rerun.json','build/stratum-dashboard-final.json'):
        inputs[name]=sha(ROOT/name)
    assert len(inputs)==4923;current(inputs)
    body=dict(schema='capture-prior-immutability-v1',passed=True,inputs_count=4923,prior_accepted_inputs=4920,
        inputs=dict(sorted(inputs.items())),models_fixed=True,policy_fixed=True,previous_allocation='hold',
        existing_capture_diagnostics_authorized=True,new_labels_authorized=False,new_searches_authorized=False,
        training_authorized=False,strength_games_authorized=False,selfplay_authorized=False,
        source_pruning_authorized=False,promotion=False)
    publish(ROOT/'build/capture-baseline.json',body)
    print(config.canonical(dict(event='capture_baseline_prepared',passed=True,prior_inputs=4923)),flush=True)


if __name__=='__main__':main()
