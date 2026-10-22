"""Freeze all timing inputs and the uniform budget before the 240 attempts."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,tracing_speed_common as stage,tracing_speed_contract as contract
from congklak import tracing_speed_launcher_v4 as launcher,tracing_speed_timing as timing
from congklak import ownership_fast_guard as guards,native_trace_fast_v2 as fast

def main():
    calibration=stage.FOLDER/'native/calibration-v4/report.json'
    report=stage.read(calibration);stage.need(report['policy_frozen'] is True,'Native calibration has not frozen a budget')
    oldspec=ROOT/'build/ownership-implementation-v1/handoff/engineering-spec.json'
    supplement=ROOT/'build/ownership-implementation-v1/provenance/engineering-supplement-v1/report.json'
    panel=stage.FOLDER/'timing/panel.json';source=stage.read(oldspec);known=stage.read(supplement)
    keys=sorted(known['combined_known_high_quotient_components'])[:4096]
    stage.need(len(keys)==4096 and all(sum(bytes.fromhex(k)[:14])>=25 for k in keys),'Pressure key domain differs')
    pressure=stage.FOLDER/'timing/pressure.json'
    stage.atomic(pressure,dict(schema='trace-speed-pressure-fixture-v1',keys=keys,origin=timing.ORIGIN,
        diagnostic_only=True,source_path=supplement.relative_to(ROOT).as_posix(),source_sha256=stage.sha(supplement),
        selection='first4096 lexicographic canonical high keys; no results used'))
    inputs=dict(source['inputs']);inputs.update(stage.common.pins(list(timing.SOURCES)+list(guards.SOURCES)+
        [__file__,oldspec,supplement,pressure,panel,calibration,'congklak/native_trace_fast_v2.py']))
    spec=dict(schema='trace-speed-timing-protocol-v1',kind='timing',guard=source['guard'],inputs=inputs,
        schedule=[dict(origin=timing.ORIGIN)],launcher_identity=launcher.identity(),
        panel_path=panel.relative_to(ROOT).as_posix(),panel_sha256=stage.sha(panel),
        pressure_path=pressure.relative_to(ROOT).as_posix(),pressure_sha256=stage.sha(pressure),
        clock_contract=contract.definition(),clock_contract_sha256=config.digest(contract.definition()),
        capacity_events=2000000,search_policy=report['search_policy'],native_identity=fast.source_identity(),
        models_fixed=True,policy_fixed=True,training_counters=stage.ZERO,production_admission=False,
        first_timed_attempt_included=True,budget_frozen_before_panel=True)
    spec['guard_refs']=guards.prepare(spec,stage.FOLDER/'guard/timing')
    guards.load_pure(spec['guard_refs'],spec)
    path=stage.FOLDER/'timing/protocol.json';stage.atomic(path,spec)
    print(config.canonical(dict(prepared=True,protocol_sha256=stage.sha(path),search_policy=spec['search_policy'])),flush=True)

if __name__=='__main__':main()
