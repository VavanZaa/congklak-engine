"""Bind the completed deeper/relay experiment to its independent evidence."""
from contextlib import closing
import ast
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config
from congklak.progress import atomic_json,current_evidence,read_json,sha,utc


def require(value,message):
    if not value: raise ValueError(message)


def main():
    audit=read_json(ROOT/"build/stage5-deeper-audit/audit.json")
    require(audit and audit["passed"] and audit["benchmark_audit_complete"],"Independent full-ledger audit incomplete")
    require(not current_evidence(ROOT,audit),"Independent audit inputs changed")
    for name,digest in audit["frozen_prior_inputs"].items():
        require(sha(ROOT/name)==digest,"Frozen prior science changed: "+name)
    independent=read_json(ROOT/"build/stage5-deeper-audit/tests.json")
    require(independent["passed"] and independent["tests_run"]==13 and
            independent["failures"]==independent["errors"]==independent["skips"]==0,"Independent acceptance incomplete")
    require(independent["source_sha256"]==sha(ROOT/"tests/stage5_deeper_audit_acceptance.py"),"Independent test source changed")
    tree=ast.parse((ROOT/"tests/stage5_deeper_audit_acceptance.py").read_text(encoding="utf-8"))
    suite=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="IndependentDeeperAcceptance")
    expected_methods={node.name for node in suite.body if isinstance(node,ast.FunctionDef) and node.name.startswith("test_")}
    covered=independent["covered_methods"]
    require(len(covered)==len(set(covered))==13 and set(covered)==expected_methods,"Independent method coverage incomplete")
    group_methods=[]
    for group in independent["groups"]:
        path=(ROOT/group["path"]).resolve()
        require(path.is_relative_to(ROOT) and sha(path)==group["sha256"],"Independent group evidence changed")
        body=read_json(path)
        require(body["passed"] and body["source_sha256"]==independent["source_sha256"] and
                body["covered_methods"]==group["methods"] and
                body["tests_run"]==group["tests_run"]==len(group["methods"]) and
                body["failures"]==body["errors"]==body["skips"]==0,"Independent group incomplete")
        group_methods.extend(group["methods"])
    require(len(group_methods)==13 and len(set(group_methods))==13 and set(group_methods)==expected_methods,"Independent group coverage duplicates or omits tests")
    training=read_json(ROOT/"build/stage5-deeper-training-acceptance.json")
    require(training["passed"] and training["tests"]==5 and
            training["failures"]==training["errors"]==training["skipped"]==0,"Training acceptance incomplete")
    for name,key in (("tests/stage5_deeper_training_acceptance.py","source_sha256"),
                     ("congklak/training_v7.py","trainer_sha256"),
                     ("congklak/deeper_teacher_v2.py","deeper_teacher_sha256")):
        require(sha(ROOT/name)==training[key],"Training source changed: "+name)
    native=read_json(ROOT/"build/stage5-relay-native-acceptance.json")
    require(native["passed"] and native["tests"]==6 and native["failures"]==native["errors"]==0,"Native acceptance incomplete")
    require(not current_evidence(ROOT,native),"Native acceptance inputs changed")
    verification=read_json(ROOT/"build/stage5-deeper-training-verification.json")
    require(verification["all_checks_passed"] and verification["checkpoint_files_verified"]==640
            and verification["matched_sampler_sequences_equal_by_seed"] and verification["dummy_policy_head_frozen"],"Checkpoint/sampling verification failed")
    require(verification["source_sha256"]==sha(ROOT/"tools/stage5-deeper-training-v3.py"),"Verification producer changed")
    progress_runs=sorted((ROOT/"build/progress").glob("run-*.json"),key=lambda p:p.stat().st_mtime_ns,reverse=True)
    progress=next((read_json(p) for p in progress_runs if read_json(p).get("tool")=="audit-progress"),None)
    require(progress and progress["state"]=="complete" and progress["exit_code"]==0,"Progress acceptance missing")
    logfile=Path(progress["log"]);log=logfile.read_text(encoding="utf-8")
    require(re.search(r"Ran 10 tests in [\d.]+s",log) and re.search(r"\nOK\s*$",log),"Progress checks incomplete")
    benchmark=read_json(ROOT/"build/stage5-deeper-benchmark.json")
    plan=read_json(ROOT/"build/stage5-deeper-benchmark-plan.json")
    require(config.digest(plan)=="dfd98b0580cb1f69a16a8a408d0d2ea425bf19dc48b289c113331265451fe6f1", "Predeclared benchmark geometry changed")
    require(benchmark["identity"]["source_sha256"]==sha(ROOT/"tools/benchmark-stage5-deeper.py"),"Benchmark source changed")
    for name,digest in benchmark["identity"]["bindings"].items():
        require(sha(ROOT/name)==digest,"Benchmark runtime changed: "+name)
    with closing(sqlite3.connect((ROOT/"data/stage5-deeper-benchmark/work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN")
        body,digest=db.execute("SELECT body,hash FROM meta WHERE key='deeper_complete'").fetchone()
        marker=json.loads(body)
        require(config.digest(marker)==digest and marker==dict(cases=391,games=288,report_sha256=config.digest(benchmark)),"Benchmark completion marker/report differ")
        require(db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]==391 and
                db.execute("SELECT COUNT(*) FROM games").fetchone()[0]==288,"Benchmark incomplete")
    speed=read_json(ROOT/"build/stage5-relay-speed-results.json")
    selection=read_json(ROOT/"build/stage5-deeper-training-results.json")
    selected=selection["selected"]
    require(speed["complete"] and set(speed["models"])=={row["model"] for row in selected.values()},"Four selected speed measurements incomplete")
    for name,key in (("tools/stage5-relay-speed.py","source_sha256"),("congklak/neural_relay.py","binding_sha256"),
                     ("congklak/features_relay.py","features_sha256"),("congklak/features_v3.py","base_features_sha256"),
                     ("congklak/timing_v2.py","timing_sha256"),("build/congklak_native.dll","native_h_sha256"),
                     ("build/congklak_neural_relay.dll","native_relay_sha256")):
        require(sha(ROOT/name)==speed["identity"][key],"Speed runtime changed: "+name)
    require(speed["identity"]["cpu"]==benchmark["identity"]["cpu"],"Speed/strength CPU declarations differ")
    for name,model in speed["identity"]["models"].items():
        require(sha(ROOT/name)==model["file_sha256"] and speed["models"][name]["cases"]==12 and speed["models"][name]["repetitions"]==3,"Speed model or geometry differs")
    speed_ledger=(ROOT/speed["ledger"]).resolve()
    require(speed_ledger.is_relative_to((ROOT/"data/stage5-relay-speed").resolve()),"Speed ledger path escapes experiment")
    with closing(sqlite3.connect((speed_ledger/"work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN")
        body,digest=db.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone()
        require(json.loads(body)==speed["identity"] and config.digest(json.loads(body))==digest,"Speed catalog identity differs")
        units=[]
        for kind,model,unit,body,digest in db.execute("SELECT kind,model,unit,body,hash FROM units ORDER BY kind,model,unit"):
            require(config.digest(json.loads(body))==digest,"Damaged speed unit")
            units.append((kind,model,unit,digest))
        expected={(kind,model,unit) for model in speed["models"] for kind,count in (("parity",1),("micro",7),("depth",36)) for unit in range(count)}
        require(len(units)==176 and {(k,m,u) for k,m,u,_ in units}==expected,"Speed unit coverage incomplete")
    require(independent["analyzer_recovery"]["actual_process_kill"] and
            independent["teacher_recovery"]["actual_precommit_process_kill"] and
            independent["cuda_recovery"]["actual_cuda_precommit_kill"] and
            independent["benchmark_recovery"]["actual_process_kill"] and
            independent["arena_recovery"]["actual_arena_hard_kill"] and
            independent["cpu"]["actual_normal_and_exception_restore"] and
            independent["final_checkpoint_recovery"]["actual_final_commit_kill"] and
            independent["cuda_recovery"]["completed_model_result_catalog_checkpoint_bytes_and_mtime_unchanged"],"Actual recovery evidence missing")
    require(audit["migration"]["all_original_reports_targets_action_q_unchanged"] and
            audit["migration"]["independently_compared"]==dict(roots=9000,features=3473,labels=3472),"Exhaustive unchanged-target migration evidence missing")
    states=audit["evaluation"]["independent_checkpoint_state_audit"]
    require(states["checkpoint_states_read"]==640 and states["seeds_replayed"]==2 and
            states["full_sampler_draws_replayed_per_seed"]==4096000,"Independent checkpoint/sampling audit incomplete")
    idempotence=read_json(ROOT/"build/stage5-deeper-training-idempotence.json")
    require(idempotence and idempotence["passed"] and idempotence["checkpoints_verified"]==640 and
            idempotence["bytes_unchanged"] and idempotence["mtimes_unchanged"] and
            idempotence["whole_orchestrator_train_and_verify_rerun"],"Completed production rerun check missing")
    require(idempotence["source_sha256"]==sha(ROOT/"tools/stage5-deeper-training-idempotence.py") and
            idempotence["orchestrator_sha256"]==sha(ROOT/"tools/stage5-deeper-training-v3.py"),"Completed rerun producer changed")
    for name,record in idempotence["files"].items():
        path=(ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and sha(path)==record["sha256"] and path.stat().st_mtime_ns==record["mtime_ns"],"Completed rerun artifact changed: "+name)
    require(audit["benchmark"]["cases_independently_recomputed"]==391 and audit["benchmark"]["literal_games_replayed"]==288,"Independent benchmark coverage incomplete")
    opening=read_json(ROOT/"build/stage5-deeper-opening-recovery.json")
    require(opening["passed"] and opening["fresh_final_value_best_moves_and_candidate_PVs_equal"] and
            opening["committed_depths"]==list(range(1,9)) and opening["invocations"][0]["last_depth"]==1 and
            opening["invocations"][1]["limits"]["start_depth"]==2 and opening["invocations"][1]["last_depth"]==8 and
            sha(ROOT/opening["model"])==opening["model_file_sha256"],"Production-model opening recovery evidence missing")
    ordinary={key:row["ordinary_gate_passed"] for key,row in selected.items()}
    primary={key:arena["primary_strength_gate_passed"] for key,arena in benchmark["arenas"].items()}
    combined={"h8-f43":ordinary["h8-f43"] and primary["deep_counts_vs_H"],
              "h8-f127":ordinary["h8-f127"] and primary["deep_relay_vs_H"]}
    files=list(audit["inputs"])+list(audit["frozen_prior_inputs"])+list(native["inputs"])+list(idempotence["files"])
    files += ["tools/record-stage5-deeper.py","congklak/progress.py","tests/progress_acceptance.py","tools/progress-dashboard.html",
              "README.md","docs/PROGRESS_REPORTING.md","docs/STAGE5_DEEPER_RESULTS.md","docs/STAGE5_DEEPER_INDEPENDENT_AUDIT.md",
              "build/stage5-deeper-training-acceptance.json","build/stage5-deeper-training-verification.json","build/stage5-deeper-training-results.json",
              "build/stage5-relay-native-acceptance.json","build/stage5-relay-speed-results.json","build/stage5-relay-resume-native-identity.json",
              "build/stage5-deeper-training-idempotence.json","tools/stage5-deeper-training-idempotence.py",
              "build/stage5-deeper-opening-recovery.json","build/opening-position.json",
              "data/deeper-9000-v1/work.sqlite3","data/deeper-9000-v2/work.sqlite3",
              "build/stage5-deeper-audit/audit.json","build/stage5-deeper-audit/tests.json","build/stage5-deeper-benchmark-plan.json",
              "build/stage5-deeper-benchmark.json","build/stage5-deeper-prior-snapshot/progress.py",str(logfile.relative_to(ROOT))]
    files += ["build/stage5-deeper-resume-native-diagnostic.json","build/stage5-deeper-work-link-diagnostic.json",
              "build/stage5-deeper-audit/teacher-independent.json",
              "build/progress/run-35632151e3e84df78da6def5db904504.json","build/progress/run-35632151e3e84df78da6def5db904504.log",
              "build/progress/run-86b63b9a1e5142078ca7beb8fe1a90d6.json","build/progress/run-86b63b9a1e5142078ca7beb8fe1a90d6.log"]
    for row in selection["rows"]:
        files += [row["model"],str(Path(row["model"]).parent/"result.json")]
    for record in progress_runs:
        value=read_json(record)
        if value.get("state") in ("complete","failed") and (value.get("tool") in ("stage5-deeper-training","stage5-deeper-benchmark","stage5-relay-speed","audit-deeper") or
                (value.get("tool")=="neural-relay" and "stage5-deeper-demo" in str(value.get("command",[])))):
            files += [str(record.relative_to(ROOT)),str(Path(value["log"]).relative_to(ROOT))]
    report=dict(schema="stage5-deeper-acceptance-v1",recorded_at_utc=utc(),passed=True,tests_run=34,failures=0,errors=0,skips=0,
                groups={"training":5,"native":6,"independent_auditor":13,"progress":10},independent_audit_complete=True,
                teacher_roots=9000,training_runs=8,checkpoint_files_verified=640,position_cases=391,arena_games=288,
                speed_units=176,speed_units_sha256=config.digest(units),
                ordinary_H8_gates=ordinary,primary_arena_gates=primary,combined_deep_model_gates=combined,
                recovery_checks={"native_actual_hard_kill":True,"teacher_actual_precommit_hard_kill":True,
                    "cuda_actual_precommit_hard_kill":True,"cuda_final_checkpoint_pre_export_kill":True,
                    "completed_artifacts_byte_and_mtime_stable":True,"benchmark_actual_hard_kill":True,"arena_actual_hard_kill":True,
                    "checksum_valid_corruption_refused":True,"affinity_normal_and_exception_restored":True},
                migrated_prior_targets_unchanged=True,original_unknown_native_stop_cause_preserved=True,
                promotion="H_v1 remains default; no automatic promotion from this development experiment",
                sealed_target_bodies_read=False,energy_measured=False,selected=selected,arenas=benchmark["arenas"],
                inputs={name.replace("\\","/"):sha(ROOT/name) for name in sorted(set(files))})
    atomic_json(ROOT/"build/stage5-deeper-acceptance.json",report)
    print(config.canonical(dict(event="stage5_deeper_acceptance_recorded",passed=True,tests_run=34,cases=391,games=288,
                               ordinary_H8_gates=ordinary,primary_gates=primary,promotion=report["promotion"])),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
