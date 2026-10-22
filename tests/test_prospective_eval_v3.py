"""Effective-capacity prefix replay preserves old bounds and all charges."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from congklak import config, prospective_eval_common as s, prospective_eval_runtime_v3 as r

def write(path,body):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(config.canonical(body)+'\n',encoding='utf-8')

def fixture_ledger(tmp_path,bad_old_duration=False,bad_future_binding=False):
    original=dict(r.base.PROTOCOL);effective=dict(original,cumulative_ns=18_000_000_000_000,quota_bytes=32*1024**3,lease_ns=1_800_000_000_000)
    identity=dict(schema=r.SCHEMA+'-identity',protocol=original,fixture=False,inherited_charged_ns=10_845_432_000_000,old_ledger={})
    write(tmp_path/'identity.json',identity);events=[];binding=dict(reviewed='v3')
    def append(event):
        event=dict(event,index=len(events),previous=config.digest(events[-1]) if events else None)
        write(tmp_path/'events'/f'{len(events):06d}.json',event);events.append(event)
    def lease(index,p,duration,capacity=False):
        row=dict(token=str(index),reserved_ns=duration,protocol=p,worker_entry=str(r.WORKER),worker_entry_sha256=s.sha(r.WORKER))
        if capacity:row['capacity_binding']=dict(wrong=True) if bad_future_binding else binding
        write(tmp_path/'leases'/str(index)/'lease.json',row);append(dict(kind='reserve',lease=row))
        append(dict(kind='finish',token=str(index),charged_ns=100_000_000_000 if index==0 else 1_000_000_000_000,
                    elapsed_ns=100_000_000_000 if index==0 else 1_000_000_000_000,unknown_overrun=False,clean=True))
    lease(0,original,1_800_000_000_000 if bad_old_duration else 300_000_000_000)
    carried=dict(schema=r.SCHEMA+'-head',count=2,last_sha256=config.digest(events[-1]))
    for index in range(1,5):lease(index,effective,1_800_000_000_000,True)
    write(tmp_path/'head.json',dict(schema=r.SCHEMA+'-head',count=len(events),last_sha256=config.digest(events[-1])))
    capacity=dict(carried_head=carried,carried_charged_ns=10_945_432_000_000,effective_protocol=effective)
    return identity,capacity,binding

def call_status(tmp_path,identity,capacity,binding):
    with patch.object(r,'admit',return_value=capacity),patch.object(r,'_path',return_value=tmp_path),\
         patch.object(r,'_identity',return_value=identity),patch.object(r,'execution_binding',return_value=binding):
        return r.status(tmp_path)

def test_effective_status_crosses_old_cap_without_resetting_charge(tmp_path):
    identity,capacity,binding=fixture_ledger(tmp_path)
    report=call_status(tmp_path,identity,capacity,binding)
    assert report['charged_ns']==14_945_432_000_000>identity['protocol']['cumulative_ns']
    assert report['remaining_ns']==3_054_568_000_000
    assert report['original_protocol']['lease_ns']==300_000_000_000
    assert report['effective_protocol']['lease_ns']==1_800_000_000_000
    assert report['pending'] is None and len(report['events'])==10

def test_historical_prefix_retains_original_300_second_bound(tmp_path):
    values=fixture_ledger(tmp_path,bad_old_duration=True)
    with pytest.raises(ValueError,match='Invalid durable reservation'):call_status(tmp_path,*values)

def test_future_lease_requires_exact_reviewed_capacity_binding(tmp_path):
    values=fixture_ledger(tmp_path,bad_future_binding=True)
    with pytest.raises(ValueError,match='Prospective lease'):call_status(tmp_path,*values)

def test_physical_executor_changes_only_runtime_imports():
    old=(s.ROOT/'congklak/prospective_eval_execute_v2.py').read_text(encoding='utf-8')
    new=(s.ROOT/'congklak/prospective_eval_execute_v3.py').read_text(encoding='utf-8')
    assert new==old.replace('prospective_eval_runtime_v2','prospective_eval_runtime_v3')

def test_actual_419_prefix_regression_and_exact_effective_capacity():
    receipt=s.read(r.CAPACITY.parent/'reader-regression.json');body=s.read(r.CAPACITY)
    assert receipt['passed'] and receipt['arena_queries']==419 and receipt['native_queries']==receipt['process_launches']==0
    assert receipt['saved_inventory_unchanged'] and not receipt['pending_intent'] and not receipt['pending_annex']
    old=s.read(r.DEFAULT/'identity.json')['protocol'];new=body['effective_protocol']
    assert {k for k in old if old[k]!=new[k]}=={'cumulative_ns','quota_bytes','lease_ns'}
    assert (new['cumulative_ns'],new['quota_bytes'],new['lease_ns'])==(18_000_000_000_000,32*1024**3,1_800_000_000_000)
