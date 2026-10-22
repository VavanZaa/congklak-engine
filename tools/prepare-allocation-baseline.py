"""Protect the preceding accepted study before new allocation/weighting diagnostics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config
from congklak.divergence_study_common import read,sha,publish,current


def main():
    previous=read(ROOT/'build/divergence-acceptance.json')
    assert previous['passed'] is previous['policy_fixed'] is previous['all_model_weights_fixed'] is True
    assert previous['training_performed'] is previous['promotion'] is False
    inputs=dict(previous['inputs']);assert len(inputs)==4252
    for name in ('build/divergence-acceptance.json','build/divergence-acceptance-rerun.json','build/divergence-dashboard-final.json'):
        inputs[name]=sha(ROOT/name)
    assert len(inputs)==4255;current(inputs)
    body=dict(schema='allocation-prior-immutability-v1',passed=True,inputs_count=4255,prior_accepted_inputs=4252,
        inputs=dict(sorted(inputs.items())),models_fixed=True,policy_fixed=True,previous_allocation='hold',
        bounded_trace_authorized=True,separate_weighting_assessment_authorized=True,
        training_authorized=False,strength_games_authorized=False,selfplay_authorized=False,
        source_pruning_authorized=False,promotion=False)
    publish(ROOT/'build/allocation-baseline.json',body)
    print(config.canonical(dict(event='allocation_baseline_prepared',passed=True,prior_inputs=4255)),flush=True)


if __name__=='__main__':main()
