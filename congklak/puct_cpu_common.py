"""Immutable CPU PUCT experiment protocol; prior accepted sources are reused."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from . import config,phase6_common as prior
from .native import ROOT

PLAN=ROOT/'build/puct-cpu-plan.json'
IMPLEMENTATION=ROOT/'build/puct-cpu-implementation.json'
WARM=prior.WARM
UPDATE=prior.UPDATE
PROFILES=('warm_legacy','updated_legacy','warm_single','updated_single','warm_cache','updated_cache')
BASELINES=('H_v1','H8_relay_CPU')
ROLES={'benchmark':'tools/benchmark-puct-cpu.py','inference':'congklak/puct_cpu_inference.py',
       'common':'congklak/puct_cpu_common.py','analysis':'congklak/puct_cpu_analysis.py','speed':'tools/puct-cpu-speed.py',
       'resource_wrapper':'tools/puct-cpu-resource-run.py','progress':'congklak/puct_cpu_progress.py',
       'native':'native/neural_puct_cpu.cpp','build_script':'tools/build-puct-cpu.ps1','build_cmd':'tools/build-puct-cpu.cmd',
       'dashboard':'tools/puct-cpu-dashboard.html'}
REQUIRED_SCIENCE=('build/phase6-acceptance.json','build/phase6-plan.json','build/phase6-implementation-v2.json',
                  'build/puct-cpu-plan.json','build/puct-cpu-build.json','build/congklak_puct_cpu.dll','build/congklak_native.dll','build/congklak_neural_relay.dll',
                  'congklak/config.py','congklak/native.py','congklak/neural_relay.py','congklak/features_relay.py',
                  'congklak/features_v3.py','congklak/puct.py','congklak/phase6_store.py','congklak/timing_v2.py',
                  'data/training-phase6-warm-v1/model.json','data/training-phase6-update-v1/model.json')

def expected_plan():
    previous=json.loads((ROOT/'build/phase6-plan.json').read_text(encoding='utf-8'))
    acceptance=json.loads((ROOT/'build/phase6-acceptance.json').read_text(encoding='utf-8'))
    prior.require(acceptance['passed'] and acceptance['engineering_passed'] and acceptance['tests_run']==42,'Prior engineering acceptance incomplete')
    result=copy.deepcopy(previous)
    result.update(schema='puct-cpu-plan-v1',prior_plan_file_sha256=prior.sha(ROOT/'build/phase6-plan.json'),
                  prior_acceptance_file_sha256=prior.sha(ROOT/'build/phase6-acceptance.json'),prior_accepted_input_count=len(acceptance['inputs']),
                  engines=list(BASELINES+PROFILES),training='No training or target generation in this experiment',
                  warmup='One batch8 inference per PUCT evaluator before timed work; startup separately recorded; cache reset inside every decision clock',
                  cache_capacity=4096,cache_reset='Before every decision inside whole-decision clock',
                  cache_policy='Per-model-evaluator bounded LRU; immutable copies; full input16cells+inputcontroller key; no quotient or store-offset reuse',
                  clock='Whole decision including cache reset; floor remaining positive integer milliseconds for frozen search; <1ms returns unusable fallback, never time_ms=0',
                  telemetry='Frozen PUCT neural_evaluations means logical leaf requests; separate evaluator counters record actual feature work, native forwards and cache hits/misses',
                  arena_order='Family outer; rotate comparison IDs by pair modulo12; seats0,1 on even pair and1,0 on odd pair',
                  release='Development reuse only, including prior arena families; no automatic promotion; sealed test untouched; all variants reported without selection')
    result['profiles']={name:dict(model='warm' if name.startswith('warm_') else 'updated',
                                    evaluator='legacy' if name.endswith('legacy') else 'single_pass',
                                    cache_capacity=4096 if name.endswith('cache') else 0,
                                    physical_feature_passes_per_miss=2 if name.endswith('legacy') else 1) for name in PROFILES}
    result['models']={'warm':str(WARM.relative_to(ROOT)).replace('\\','/'),'updated':str(UPDATE.relative_to(ROOT)).replace('\\','/')}
    result['model_files_sha256']={key:prior.sha(ROOT/path) for key,path in result['models'].items()}
    result['arena_comparisons']=[dict(candidate=name,opponent=baseline,id=name+'_vs_'+baseline,pairs=24) for name in PROFILES for baseline in BASELINES]
    comparisons=[row['id'] for row in result['arena_comparisons']];order=[]
    for pair in range(24):
        rotation=pair%len(comparisons)
        for comparison in comparisons[rotation:]+comparisons[:rotation]:
            for seat in ((0,1) if pair%2==0 else (1,0)):order.append(dict(comparison=comparison,pair=pair,seat=seat))
    result['arena_work_order']=order
    result['speed']=dict(batch_sizes=[1,2,4,8,16,32],blocks=7,repetitions=64,
                         roots='All48 current_H8_development cases, in predeclared case order',
                         micro_workloads=['cold','hot'],cold='Reset before each measured call inside clock; no prefilled-cache credit',
                         hot='Reset then populate all48 roots outside call clock; all calls reuse those roots; retrieval ceiling only, not a tree throughput claim',
                         tree_blocks=3,tree_simulations=64,tree_max_nodes=2048,
                         tree='Every one of48 H8 roots, all6 profiles, three rotated blocks; reset inside each decision; fixed seed20261015; exact semantic comparison to legacy',
                         unit_order='Block outer, rotate profile/batch order; cold and hot separately declared; retain all units')
    return result

def freeze_plan():
    value=expected_plan();prior.publish(PLAN,value);return value

def read_plan():
    value=json.loads(PLAN.read_text(encoding='utf-8'))
    prior.require(value==expected_plan(),'CPU PUCT protocol or prior evidence differs')
    return value

def speed_roots(plan):return [row['state'] for row in plan['cases'] if row['cohort']=='current_H8_development']

def admit():
    """Check immutable sources/science and previous acceptance before DLL use."""
    plan=read_plan();manifest=json.loads(IMPLEMENTATION.read_text(encoding='utf-8'))
    prior.require(manifest['schema']=='puct-cpu-implementation-v1' and manifest['sha256']==config.digest({k:v for k,v in manifest.items() if k!='sha256'}),'CPU implementation manifest corrupt')
    prior.require(all(manifest.get('roles',{}).get(role)==path for role,path in ROLES.items()),'Incomplete CPU producer roles')
    prior.require(set(ROLES.values())<=set(manifest.get('sources',{})) and set(REQUIRED_SCIENCE)<=set(manifest.get('science_inputs',{})),'Mandatory CPU source/science pin omitted')
    for group in ('sources','science_inputs'):
        prior.require(isinstance(manifest[group],dict) and manifest[group],'Missing implementation pins')
        for name,expected in manifest[group].items():
            path=(ROOT/name).resolve()
            prior.require(path.is_relative_to(ROOT) and path.is_file() and prior.sha(path)==expected,'CPU implementation input differs: '+name)
    acceptance=json.loads((ROOT/'build/phase6-acceptance.json').read_text(encoding='utf-8'))
    for name,expected in acceptance['inputs'].items():
        path=(ROOT/name).resolve()
        prior.require(path.is_relative_to(ROOT) and path.is_file() and prior.sha(path)==expected,'Prior accepted input differs: '+name)
    return plan,manifest

if __name__=='__main__':
    body=freeze_plan();print(config.canonical(dict(event='puct_cpu_plan_frozen',cases=len(body['cases']),engines=len(body['engines']),games=len(body['arena_work_order']),training=False)),flush=True)
