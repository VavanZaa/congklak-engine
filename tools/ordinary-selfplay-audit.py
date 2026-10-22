"""Separate no-Torch process for fixed-model ordinary-pilot closed-run checks."""
from __future__ import annotations
from contextlib import closing
import json
from pathlib import Path
import runpy
import sys
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config,quality_search
from congklak.native import Native
from data_pilot_independent_audit import sha,load,save,pins,snapshot


def closed(spec):
    from congklak import ordinary_selfplay_pilot as producer
    path=Path(spec['run']);before=snapshot(path);events=[]
    with mock.patch.object(quality_search,'search',side_effect=AssertionError('Closed pilot searched PUCT')) as puct,\
         mock.patch.object(quality_search,'forest',side_effect=AssertionError('Closed pilot searched forest')) as forest,\
         mock.patch.object(Native,'analyze',side_effect=AssertionError('Closed pilot searched teacher')) as teacher:
        if spec['stage']=='collection':
            value=producer.collect(path,fixture=spec.get('fixture',False),on_event=events.append)
            producer.load_rows(path)
        else:
            value=producer.generate(path,fixture=spec.get('fixture',False),on_event=events.append)
            producer.load_replay(path)
        previous_argv=sys.argv
        sys.argv=[str(ROOT/'tools/ordinary-selfplay-pilot.py'),
            'collect' if spec['stage']=='collection' else 'generate','--run',str(path)]
        if spec.get('fixture',False):sys.argv.append('--fixture')
        try:
            try:runpy.run_path(str(ROOT/'tools/ordinary-selfplay-pilot.py'),run_name='__main__')
            except SystemExit as exc:assert exc.code in (None,0)
        finally:sys.argv=previous_argv
    assert value['status']=='complete' and before==snapshot(path)
    assert puct.call_count==forest.call_count==teacher.call_count==0 and 'torch' not in sys.modules
    assert not any(v['event'].endswith(('_precommit','_committed')) for v in events)
    save(spec['output'],dict(schema='ordinary-selfplay-independent-closed-worker-v1',passed=True,
        stage=spec['stage'],all_durable_bytes_mtimes_unchanged=True,new_PUCT_searches=0,
        new_teacher_searches=0,torch_imported=False,completed_API_and_CLI_verified=True,committed_events=events,
        input_snapshots={name:dict(sha256=value[0],mtime_ns=value[1]) for name,value in before.items()},
        inputs=pins([Path(__file__),Path(spec['spec_path'])])))


def main():
    spec=load(Path(sys.argv[1]));spec['spec_path']=str(Path(sys.argv[1]));output=Path(spec['output']).resolve()
    assert output.is_relative_to(ROOT/'build')
    {'closed':closed}[spec['kind']](spec)
    print(config.canonical(dict(event='ordinary_selfplay_independent_worker',passed=True,kind=spec['kind'])),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
