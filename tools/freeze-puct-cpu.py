"""Freeze producer byte identities before any CPU PUCT timing or arena outcomes."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,phase6_common as prior,puct_cpu_common as common

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--refresh-fixture',action='store_true');args=parser.parse_args()
    common.read_plan();old=json.loads((ROOT/'build/phase6-implementation-v2.json').read_text(encoding='utf-8'))
    sources={path:prior.sha(ROOT/path) for path in common.ROLES.values()}
    sources['tools/freeze-puct-cpu.py']=prior.sha(__file__)
    science={path:prior.sha(ROOT/path) for path in set(common.REQUIRED_SCIENCE)|set(old['science_inputs'])|set(old['sources'])|{'build/puct-cpu-cost-gate.json','docs/PUCT_CPU_SCOPE.md'}}
    body=dict(schema='puct-cpu-implementation-v1',roles=common.ROLES,sources=sources,science_inputs=science,
              rules_digest=config.RULES_DIGEST,
              artifacts=dict(benchmark_run='data/puct-cpu-benchmark-v1',benchmark_output='build/puct-cpu-benchmark.json',
                             speed_run='data/puct-cpu-speed-v1',speed_output='build/puct-cpu-speed.json',
                             benchmark_producer_resources='build/puct-cpu-benchmark-resources.json',
                             analysis_demo_run='data/puct-cpu-analysis-demo'),
              scope='Frozen compact models and search semantics; exact CPU inference/caching experiment; no training, sealed targets, automatic promotion or energy claim')
    body['sha256']=config.digest(body)
    if common.IMPLEMENTATION.exists() and json.loads(common.IMPLEMENTATION.read_text(encoding='utf-8'))!=body:
        prior.require(args.refresh_fixture,'Sources changed; explicitly refresh preproduction fixture freeze')
        prior.require(not any((ROOT/body['artifacts'][name]/'work.sqlite3').exists() for name in ('benchmark_run','speed_run','analysis_demo_run')),'Cannot refresh source identity after production journal creation')
        index=0
        while (ROOT/f'build/puct-cpu-implementation-prefreeze-v{index}.json').exists():index+=1
        (ROOT/f'build/puct-cpu-implementation-prefreeze-v{index}.json').write_bytes(common.IMPLEMENTATION.read_bytes())
        prior.teacher.atomic_json(common.IMPLEMENTATION,body)
    else:prior.publish(common.IMPLEMENTATION,body)
    common.admit();print(config.canonical(dict(event='puct_cpu_implementation_frozen',sha256=body['sha256'],sources=len(sources),science_inputs=len(science))),flush=True)

if __name__=='__main__':main()
