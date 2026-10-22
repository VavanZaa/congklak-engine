"""Frozen definitions for trace-speed clocks, panel gates and audit coverage."""
import math
from . import config,tracing_speed_common as stage

METHODS=('source_integrity','native_coverage','guard_semantics','fixed_work_conformance',
         'timing_panel','worker_recovery','resource_accounting','completed_readers','observer')
KILL_POINTS=('before_handoff_commit','after_handoff_commit','before_decision_commit','after_decision_commit',
             'before_ownership_check_commit','after_ownership_check_commit','before_catalog','after_catalog')
ENGINES=('H_v1','H8_relay_CPU','control','candidate')
CELLS=tuple(band+'/actor'+str(actor) for band in ('50-74/ordinary','25-49/ordinary','25-49/bonus','25-49/capture')
            for actor in (0,1))+('75-98/ordinary/actor0','75-98/ordinary/actor1',
                                '5-24/ordinary/actor0','5-24/ordinary/actor1')
SCHEMA='trace-speed-clock-contract-v1'

def definition():
    return dict(schema=SCHEMA,decision_ms=25,p95_headroom_ms=22.5,engines=list(ENGINES),panel_cells=list(CELLS),
        repeats=5,expected_attempts=240,first_timed_attempt_included=True,
        setup='Frozen models, raw storage capacity, immutable guard tables and predeclared diagnostic pressure claims may be preloaded; no root-derived features or per-decision probes.',
        critical='Starts before both-seat reset; includes search/features/forward, complete raw capture, immutable owned copies, SHA256, validity and physical work partitions, all high-state checks and private staged-claim insertion/allocation, chosen literal validation and CPU verification.',
        archival='Only immutable already-hashed/validated/guarded bytes may be compressed/base64/JSON serialized and fsynced after the critical clock. Finish before decision/check publication; all elapsed time charged to the cumulative lease.',
        claims='All per-key work staged inside critical clock. Durable check publication precedes constant-time native stage publication. Cold rebuild of durable claims is setup, before engines.',
        failure='Retain every attempt/overshoot/overflow/unusable answer and any recoverable prefix. Incomplete missing raw capture holds the run; never silently retry or report zero work.',
        timing_gate='All240 declared attempts usable and fully covered/guarded, every critical elapsed<=25ms and nearest-rank p95<=22.5ms. Any failed attempt/overshoot gives HOLD. Empirical sample only; no hard-real-time guarantee.',
        source_hold='Existing320 support conflicts and missing historical footprints remain; no training or production strength admission.',
        recovery_points=list(KILL_POINTS),models_fixed=True,policy_fixed=True,training_counters=stage.ZERO)

def timing_gate(rows,protocol_sha256):
    need=stage.need
    need(type(rows) is list,'Timing rows must be an ordered list')
    elapsed=[];failures=[]
    for index,row in enumerate(rows):
        value=row.get('critical_ms')
        valid=type(value) in (int,float) and math.isfinite(value) and value>0
        if valid:elapsed.append(value)
        if not valid or row.get('usable_decision') is not True or row.get('coverage_passed') is not True or row.get('guard_passed') is not True or value>25:
            failures.append(index)
    ordered=sorted(elapsed);p95=ordered[math.ceil(.95*len(ordered))-1] if ordered else None
    passed=len(rows)==240 and not failures and p95 is not None and p95<=22.5
    return dict(schema='trace-speed-empirical-timing-gate-v1',protocol_sha256=protocol_sha256,
        clock_contract_sha256=config.digest(definition()),expected_attempts=240,attempts=len(rows),failed_attempt_indices=failures,
        median_ms=ordered[len(ordered)//2] if ordered else None,p95_ms=p95,maximum_ms=max(ordered) if ordered else None,
        passed=passed,state='empirical_pass' if passed else 'HOLD',hard_real_time_guarantee=False,
        training_admitted=False,production_evaluation_admitted=False)
