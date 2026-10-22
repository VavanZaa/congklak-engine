"""Pure prospective schedule, recovery and source-boundary acceptance checks."""
import ast
from collections import Counter
import hashlib,json
import uuid
import base64
from pathlib import Path
from unittest.mock import patch
import pytest
from congklak import config, prospective_eval_common as s, prospective_eval_execute as e
from congklak import prospective_eval_runtime as r, prospective_eval_recovery as recovery

@pytest.fixture
def workspace_tmp():
    path=(s.FOLDER/'fixtures/pure-tests'/uuid.uuid4().hex).resolve()
    assert path.is_relative_to(s.ROOT)
    path.mkdir(parents=True,exist_ok=False)
    return path

def test_complete_finite_workloads_and_paired_order():
    end=e.schedule('endgame');arena=e.schedule('arena')
    assert len(end)==128 and len(arena)==192
    grouped=Counter((x['meta']['state_key'],x['mode'],x['engine']) for x in end)
    assert len(grouped)==128 and set(grouped.values())=={1}
    assert len({x['meta']['state_key'] for x in end})==32
    cells=Counter((x['meta']['origin'],x['baseline'],x['seat'],x['profile']) for x in arena)
    assert len(cells)==192 and set(cells.values())=={1}
    for index in range(0,len(arena),2):
        a,b=arena[index:index+2]
        assert (a['meta']['origin'],a['baseline'],a['seat'])==(b['meta']['origin'],b['baseline'],b['seat'])
        assert {a['profile'],b['profile']}=={'control','candidate'}
    assert not any(x.get('kind')=='teacher' for x in end+arena)

def test_predecessor_order_and_only_two_registered_kills():
    assert e.predecessor(e.DEFAULT,'endgame') is None
    before,after=(e.FIXTURE_BASE/x for x in e.HANDOFF_KILL_HOOKS)
    assert e.predecessor(before,'arena',True)['path']==s.relative(e.DEFAULT)
    assert e.predecessor(after,'arena',True)['path']==s.relative(before)
    assert e.predecessor(s.FOLDER/'jobs/arena','arena')['path']==s.relative(after)
    with pytest.raises(ValueError):e.classify(e.FIXTURE_BASE/'before_query','arena',True)
    with pytest.raises(ValueError):e.schedule('endgame',True)

def test_exact_final400_and_carried_resource_constants():
    assert e._candidate('endgame',False,None)['sha256']==s.CANDIDATE_SHA
    assert r.PROTOCOL['lease_ns']==300_000_000_000
    assert r.PROTOCOL['cumulative_ns']==14_400_000_000_000
    assert r.PROTOCOL['unclean_charge']=='entire durable reservation'
    text=Path(recovery.__file__).read_text(encoding='utf-8')
    assert 'timeout=660' in text and 'time.monotonic() + 660' in text
    assert 'TerminateProcess(handle, 137)' in text
    assert 'worker[\'argv\'] == marker[\'argv\'] == expected_argv' in text

def test_unknown_return_is_held_before_any_provider_construction(tmp_path):
    identity=dict(fixture=False)
    data=dict(hold=None,intent=dict(archive='archives/missing.json',decision_id='query/000000'),events=[])
    fake_lease=tmp_path/'lease.json'
    with patch.object(s,'path_of',return_value=tmp_path),patch.object(s,'admit_plan'),\
         patch.object(e,'_identity',return_value=identity),patch.object(e,'_replay',return_value=data),\
         patch.object(e,'_check_orphans'),patch.object(e,'_append',side_effect=lambda p,events,event:events.append(event)),\
         patch.object(e,'_finalize',side_effect=lambda p,i,d,pause:d['hold']),\
         patch.object(r,'assert_worker_lease'),patch('congklak.guarded_value_pilot_v2_native.open_providers',side_effect=AssertionError('unexpected native query')):
        result=e.run(tmp_path,s.DATA/'cache',fake_lease,'endgame')
    assert result['reason']=='Unknown unreturned query; no retry'

def test_orphan_attempt_forbids_replay(workspace_tmp):
    tmp_path=workspace_tmp
    (tmp_path/'attempts').mkdir();(tmp_path/'attempts/extra.bin').write_bytes(b'complete-but-unreferenced')
    with pytest.raises(ValueError,match='Unreferenced raw attempt'):
        e._check_orphans(tmp_path,dict(results=[],intent=None,events=[]))

def test_full_source_constructor_and_large_data_close_check(workspace_tmp):
    tmp_path=workspace_tmp
    source=tmp_path/'source.bin';source.write_bytes(b'fixed')
    plan=tmp_path/'plan.json';review=tmp_path/'preflight.json'
    body=dict(source_inputs={str(source):s.sha(source)})
    plan.write_text(config.canonical(body)+'\n',encoding='utf-8')
    review.write_text(config.canonical(dict(schema=s.SCHEMA+'-independent-preflight',passed=True,
      independent_of_implementation=True,plan_ref=dict(path=str(plan),sha256=s.sha(plan)),source_inputs=body['source_inputs']))+'\n',encoding='utf-8')
    with patch.object(s,'PLAN',plan),patch.object(s,'PREFLIGHT',review),\
         patch.object(s,'ref',side_effect=lambda p:dict(path=str(p),sha256=s.sha(p))):
        with pytest.raises(ValueError,match='constructor body'):
            s._SourceView(dict(source_inputs={}))
        view=s._SourceView(body);token=s._ACTIVE.set(view)
        try:
            view.verify(body)
            # Tampering cannot be hidden by unchanged size or any mtime.
            source.write_bytes(b'other')
            with pytest.raises(ValueError):view.close()
        finally:s._ACTIVE.reset(token)

def test_projection_outputs_match_saved_actual_sources():
    record=s.read(s.FOLDER/'source-projection.json')
    s.common.current(record['source_inputs'])
    for name in ('executor-projection.json','runtime-projection.json','recovery-projection.json'):
        body=s.read(s.FOLDER/name)
        out=body.get('output_ref') or dict(path=body['path'],sha256=body['sha256'])
        assert s.sha(s.ROOT/out['path'])==out['sha256']

@pytest.mark.parametrize('nonempty',[False,True])
def test_saved_exact_stage_annex_barrier_precedes_accept_and_next_query(workspace_tmp,nonempty):
    from congklak import prospective_eval_guard as guard
    from types import SimpleNamespace
    path=workspace_tmp;origin='fixture-origin';ids={origin:1}
    registry=e.prepare_registry();q=(s.ROOT/registry['seed_claims']['path']).read_bytes()[:15]
    delta=guard.STATIC.pack(q,1) if nonempty else b''
    body=dict(decision_id='query/000000',origin=origin,engine='control',query_kind='decision',state={},
      answer=dict(trace={},fast_owned_high=base64.b64encode(b'').decode(),lifecycle=dict(after=dict(created=2))))
    (path/'archives').mkdir();ap=path/'archives/query-000000.json'
    ap.write_text(config.canonical(body)+'\n',encoding='utf-8')
    pending=dict(body=body,result={},delta=b'',archive_ref=s.ref(ap),ownership_committed=True)
    data=dict(intent=dict(body,archive='archives/query-000000.json'),pending=pending,claimed=guard.PackedClaims(b'',ids),
      claim_hash='a'*64,generation=354,results=[],events=[],last_lifecycle=None)
    order=[]
    def append(p,events,event):
        assert not data['results']
        order.append(event['kind']);events.append(event)
        return dict(event_path=str(path/'event.json'),event_sha256='b'*64)
    def restore(claimed,generation):
        assert order==['component_annex'] and not data['results'] and generation==354
        order.append('restore')
    proof=dict(components=['fixture-cid'] if nonempty else [],delta_records=delta)
    # A recovered ownership stage can append only the missing deterministic
    # annex. It cannot issue another native query or exact ownership commit.
    with patch.object(guard,'derive_annex',return_value=proof),patch.object(e,'_append',side_effect=append),\
         patch('congklak.guarded_value_pilot_v2_native.open_providers',side_effect=AssertionError('repeated query')):
        e._adopt(path,dict(operation='endgame'),data,live_guard=SimpleNamespace(restore=restore),compiled=object())
    assert order==(['component_annex','restore'] if nonempty else ['component_annex'])
    assert len(data['results'])==1 and data['intent'] is None and data['pending'] is None
    assert data['claim_hash']==(data['claimed'].sha256 if nonempty else 'a'*64)
