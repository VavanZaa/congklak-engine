"""Pure acceptance for the versioned reader and actual entry binding."""
from pathlib import Path
import ast
import pytest
from congklak import prospective_eval_common as s, prospective_eval_execute_v2 as e
from congklak import prospective_eval_runtime_v2 as r

def test_versioned_projection_actual_bytes():
    projection=s.read(r.PROJECTION)
    for row in projection['pairs']:
        assert s.sha(s.ROOT/row['original'])==row['original_sha256']
        assert s.sha(s.ROOT/row['generated'])==row['generated_sha256']
    assert projection['runtime_adapter_ref']==s.ref(Path(r.__file__))
    assert not projection['journal_and_arithmetic_changed']
    assert not projection['ledger_identity_changed']

def test_fresh_guard_batch_closes_before_return():
    node=next(n for n in ast.parse(Path(e.__file__).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='_guard_batch')
    assert ast.unparse(node).splitlines()[-2:]==['    batch.verify_current()', '    return batch']

def test_legacy_identity_exception_is_only_sealed_endgame():
    identity=s.read(s.FOLDER/'jobs/endgame/identity.json')
    assert r.verify_execution_identity(identity,s.FOLDER/'jobs/endgame')
    with pytest.raises(ValueError):r.verify_execution_identity(identity,s.FOLDER/'jobs/arena')
    with pytest.raises(ValueError):r.verify_execution_identity(dict(identity,fixture=True),s.FOLDER/'jobs/endgame')

def test_actual_v2_entry_and_same_ledger_interface():
    assert r.DEFAULT==r.base.DEFAULT
    assert r.status is r.base.status
    assert r.WORKER.name=='prospective_eval_worker_v2.py'
    assert r.SUPERVISOR.name=='prospective_eval_cli_v2.py'
    text=Path(r.WORKER).read_text(encoding='utf-8')
    assert 'prospective_eval_execute_v2 import main' in text and 'runpy' not in text
    recovery=s.read if False else (s.ROOT/'congklak/prospective_eval_recovery_v2.py').read_text(encoding='utf-8')
    assert "audit/native-recovery-v2'" in recovery and 'timeout=660' in recovery
