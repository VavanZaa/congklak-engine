"""Predeclared whole unused-development component claims after each query.

This adds no positions or engine work. Eligibility is frozen before queries.
A query's normal durable native claim publication must precede this transaction;
its complete component delta must be durably accepted before another query.
"""
from collections import defaultdict
import hashlib
import struct
import weakref
from . import config, neurodiag_common as common
from . import prospective_eval_reservations as reservations
from . import ordinary_epoch_guard as old
from .value_diagnostics import atomic

ROOT=common.ROOT
DEFAULT=reservations.FOLDER/'annex'
POLICY=DEFAULT/'policy-v2.json'
SCHEMA='prospective-eval-v1-whole-component-annex'
need,read,sha=common.require,common.read,common.sha
_COMPONENT_CACHE=weakref.WeakKeyDictionary()


def eligibility(report):
    return sorted(cid for cid,body in report['components'].items()
        if body['acquired_origins_optimizer_blocked'] is True and
        body['evaluation_blocked'] is False and body['origins']==[])


def prepare():
    report=reservations.verify()
    body=dict(schema=SCHEMA+'-policy',eligible_components=eligibility(report),
        base_certificate=reservations.ref(reservations.DEFAULT/'certificate.json'),
        complete_mapping=report['shared_records_ref'],eligibility='acquired_origins_optimizer_blocked=true AND evaluation_blocked=false AND registered_origins=[]',
        support_fit_arena_quarantine_eligible=False,whole_component_transaction=True,
        order='normal native ownership_check publication, mandatory component_annex commit, then next query or score acceptance; nonempty delta uses canonical-after hash and same-generation full native claims restore; empty delta preserves staged hash and performs no restore',
        every_query_annex_event=True,empty_annex_events_retained=True,
        native_static='own complete component OR prospectively eligible unused development component => flag0; all other known components => flag2',
        live_owner_conflicts='native flag4; never reassigned, exempted or retried',
        complete_component_keys='claim every canonical key in immutable complete mapping, not only observed hits',
        timing='projection and whole-component annex outside decision clock; unchanged native exact-hit guard/hash/stage remains inside whole decision clock',
        no_outcome_selection=True,query_retry_authorized=False,historically_unseen=False,
        policy_fixed=True,models_fixed=True,training_authorized=False,
        source_inputs=common.pins([__file__]))
    atomic(POLICY,body);return verify()


def verify():
    body=read(POLICY);report=read(reservations.DEFAULT/'report.json')
    need(body['schema']==SCHEMA+'-policy' and body['eligible_components']==eligibility(report) and
         body['base_certificate']==reservations.ref(reservations.DEFAULT/'certificate.json') and
         body['complete_mapping']==report['shared_records_ref'] and body['whole_component_transaction'] is True and
         body['query_retry_authorized'] is False,'Prospective annex policy/eligibility changed')
    common.current(body['source_inputs']);return body


def projected_records(records,allowed_ordinals,numpy=True):
    """Pure literal q15+flag32 projection; independent of the engine DLL."""
    need(type(records) is bytes and len(records)%19==0,'Annex projection record width differs')
    allowed=set(allowed_ordinals)
    if numpy:
        try:import numpy as np
        except ImportError:pass
        else:
            dtype=np.dtype([('q','V15'),('ordinal','<u4')]);source=np.frombuffer(records,dtype=dtype)
            out=source.copy();maximum=int(source['ordinal'].max(initial=0));lut=np.full(maximum+1,2,dtype='<u4')
            for ordinal in allowed:
                if 0<=ordinal<=maximum:lut[ordinal]=0
            out['ordinal']=lut[source['ordinal']];return out.tobytes()
    out=bytearray(records)
    for offset in range(15,len(out),19):
        ordinal=struct.unpack_from('<I',out,offset)[0]
        struct.pack_into('<I',out,offset,0 if ordinal in allowed else 2)
    return bytes(out)


def derive(owned,loaded,origin,claimed):
    """Deterministic all-key delta after the query's ordinary native stage.

    Selected/protected bridges remain errors. Any earlier component annex is
    represented by its complete key claims, so a different key in that component
    is already protected by the unchanged native global-owner check.
    """
    from . import prospective_eval_guard as guard
    need(type(loaded.admission_batch) is guard.AdmissionBatch,'Annex requires genuine constructor admission')
    loaded.admission_batch.assert_sealed(require_closed=True)
    return _derive(owned,loaded,origin,claimed)


def _derive(owned,loaded,origin,claimed):
    """Pure graph transaction, separately available to literal fixture tests."""
    policy=loaded.admission_batch.annex_policy;eligible=loaded.admission_batch.eligible_components
    need(origin in loaded.origin_ids,'Annex owner is unregistered')
    rows=old.owned_rows(owned)
    touched=sorted({loaded.graph_lookup.get(row['quotient']) for row in rows} & eligible)
    owner=loaded.owner_id(origin);before=hashlib.sha256(loaded.claims_blob(claimed)).hexdigest()
    need(not old.reference_owned(owned,loaded,origin,claimed),'Annex cannot accept a protected or foreign-owner bridge')
    need(all(claimed.get(row['quotient'])==origin for row in rows),
         'Whole-component annex requires the exact-hit native stage to be published first')
    wanted={loaded.admission_batch.component_ordinals[c]:c for c in touched};proofs=defaultdict(lambda:dict(count=0,hash=hashlib.sha256()))
    cache=_COMPONENT_CACHE.setdefault(loaded.admission_batch,{})
    missing={ordinal for ordinal in wanted if ordinal not in cache}
    if missing:
        try:import numpy as np
        except ImportError:
            buffers={ordinal:bytearray() for ordinal in missing}
            for offset in range(0,len(loaded.records),19):
                ordinal=struct.unpack_from('<I',loaded.records,offset+15)[0]
                if ordinal in buffers:buffers[ordinal].extend(loaded.records[offset:offset+19])
            cache.update({k:bytes(v) for k,v in buffers.items()})
        else:
            source=np.frombuffer(loaded.records,dtype=np.dtype([('q','V15'),('ordinal','<u4')]))
            for ordinal in missing:cache[ordinal]=source[source['ordinal']==ordinal].tobytes()
    delta=[]
    keys=sorted(row for ordinal in wanted for row in old.STATIC.iter_unpack(cache[ordinal]))
    for q,ordinal in keys:
        cid=wanted[ordinal];entry=proofs[cid];entry['count']+=1;entry['hash'].update(q)
        previous=claimed.get(q)
        need(previous in (None,origin),'Whole unused-development component already belongs to another live origin')
        if previous is None:delta.append(old.STATIC.pack(q,owner))
    packed=b''.join(delta);need(len(claimed)+len(delta)<=8000000,'Whole-component annex exceeds global claim capacity')
    after=claimed.merge_delta(packed)
    return dict(schema=SCHEMA+'-transaction',components=touched,delta_records=packed,keys_count=len(delta),
        delta_sha256=hashlib.sha256(packed).hexdigest(),canonical_claims_before=before,canonical_claims_after=after.sha256,
        policy_ref=reservations.ref(POLICY),whole_component_proof=[dict(component=c,ordinal=loaded.admission_batch.component_ordinals[c],
            keys=proofs[c]['count'],sorted_key_sha256=proofs[c]['hash'].hexdigest()) for c in touched],
        complete_mapping_ref=policy['complete_mapping'],origin=origin,owner=owner)
