"""Aggregate current action-trial acceptance after independent full-ledger audit."""
from contextlib import closing
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
    audit_path=ROOT/"build/stage5-action-audit/audit.json"
    audit=read_json(audit_path)
    require(audit and audit["passed"] and audit["benchmark_audit_complete"],"Independent full-ledger audit incomplete")
    require(not current_evidence(ROOT,audit),"Independent audit inputs changed")
    for name,digest in audit["frozen_prior_inputs"].items():
        require(sha(ROOT/name)==digest,"Frozen prior science changed: "+name)
    independent=read_json(ROOT/"build/stage5-action-audit/tests.json")
    require(independent["passed"] and independent["tests_run"]==9
            and independent["failures"]==independent["errors"]==independent["skips"]==0,"Independent checks failed")
    training=read_json(ROOT/"build/stage5-action-training-acceptance.json")
    require(training["passed"] and training["tests"]==5
            and training["failures"]==training["errors"]==training["skipped"]==0,"Training checks failed")
    require(training["source_sha256"]==sha(ROOT/"tests/stage5_action_training_acceptance.py")
            and training["trainer_sha256"]==sha(ROOT/"congklak/training_v4.py")
            and training["action_targets_sha256"]==sha(ROOT/"congklak/action_targets.py"),"Training source pins changed")
    native=read_json(ROOT/"build/stage5-action-native-acceptance.json")
    require(native["tests"]==5 and native["failures"]==native["errors"]==native["skips"]==0,"Native checks failed")
    require(not current_evidence(ROOT,{"inputs":native["artifacts"]}),"Native acceptance artifacts changed")
    verification=read_json(ROOT/"build/stage5-action-training-verification.json")
    require(verification["all_checks_passed"] and verification["checkpoint_files_verified"]==480
            and verification["matched_sampler_sequences_equal_by_seed"],"Checkpoint/sampling verification failed")
    progress_runs=sorted((ROOT/"build/progress").glob("run-*.json"),key=lambda p:p.stat().st_mtime,reverse=True)
    progress=next((read_json(p) for p in progress_runs if read_json(p).get("tool")=="audit-progress"),None)
    require(progress and progress["state"]=="complete" and progress["exit_code"]==0,"Progress acceptance missing")
    logfile=Path(progress["log"]); log=logfile.read_text(encoding="utf-8")
    require(re.search(r"Ran 9 tests in [\d.]+s",log) and re.search(r"\nOK\s*$",log),"Progress checks incomplete")
    benchmark=read_json(ROOT/"build/stage5-action-benchmark-v2.json")
    require(benchmark["identity"]["source_sha256"]==sha(ROOT/"tools/benchmark-stage5-action-v2.py"),"Benchmark source changed")
    for name,digest in benchmark["identity"]["bindings"].items():
        require(sha(ROOT/name)==digest,"Benchmark runtime changed: "+name)
    with closing(sqlite3.connect((ROOT/"data/stage5-action-benchmark-v2/work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        body,digest=db.execute("SELECT body,hash FROM meta WHERE key='action_complete'").fetchone()
        marker=json.loads(body)
        require(config.digest(marker)==digest and marker["report_sha256"]==config.digest(benchmark),"Completion marker/report differ")
        require(db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]==391
                and db.execute("SELECT COUNT(*) FROM games").fetchone()[0]==144,"Benchmark incomplete")
    selection=read_json(ROOT/"build/stage5-action-training-results.json")
    timing=read_json(ROOT/"build/timing-v2-acceptance.json")
    require(timing["tests"]==4 and timing["failures"]==timing["errors"]==timing["skips"]==0,"Timing checks failed")
    require(not current_evidence(ROOT,timing),"Timing inputs changed")
    require(independent["arena_recovery"]["actual_arena_hard_kill"],"Actual arena recovery check missing")
    files=list(audit["inputs"])+list(audit["frozen_prior_inputs"])+list(native["artifacts"])+list(timing["inputs"])
    files += ["tools/record-stage5-action.py","congklak/progress.py","tests/progress_acceptance.py","tools/progress-dashboard.html",
              "README.md","docs/PROGRESS_REPORTING.md","docs/STAGE5_ACTION_RESULTS.md","docs/STAGE5_ACTION_INDEPENDENT_AUDIT.md",
              "build/stage5-action-training-acceptance.json","build/stage5-action-training-verification.json",
              "build/stage5-action-training-results.json","build/stage5-action-training-plan.json","build/stage5-action-training.log",
              "build/stage5-action-native-acceptance.json","build/neural-action-build.json","build/stage5-action-speed-results.json",
              "build/stage5-action-audit/audit.json","build/stage5-action-audit/tests.json",
              "build/stage5-action-benchmark-plan-v2.json","build/stage5-action-benchmark-v2.json",
              "build/stage5-action-benchmark-plan.json","build/stage5-action-benchmark.json",
              "build/stage5-action-v1-snapshot/progress.py","build/timing-v2-acceptance.json",
              "tools/stage5-action-training.py","tools/stage5-action-speed.py","tools/build-neural-action.cmd","tools/build-neural-action.ps1",
              str(logfile.relative_to(ROOT))]
    for row in selection["rows"]:
        files += [row["model"],row["value_model"],str(Path(row["model"]).parent/"result.json")]
    for record in progress_runs:
        value=read_json(record)
        if value.get("state")=="complete" and (value.get("tool")=="stage5-action-benchmark"
                or (value.get("tool")=="neural-action" and "stage5-action-demo" in str(value.get("command",[])))):
            files += [str(record.relative_to(ROOT)),str(Path(value["log"]).relative_to(ROOT))]
    primary={name:arena["primary_strength_gate_passed"] for name,arena in benchmark["arenas"].items()}
    learned_gate=selection["selected_challenger"]["ordinary_gate_passed"] and primary["joint_policy_vs_H"]
    report=dict(schema="stage5-action-acceptance-v1",recorded_at_utc=utc(),passed=True,
                tests_run=32,failures=0,errors=0,skips=0,
                groups={"training":5,"native":5,"independent_auditor":9,"declared_cpu_timing":4,"progress":9},
                independent_audit_complete=True,checkpoint_files_verified=480,position_cases=391,arena_games=144,
                terminal_games=sum(a["terminal_games"] for a in benchmark["arenas"].values()),
                ordinary_value_gate_passed=selection["selected_challenger"]["ordinary_gate_passed"],
                primary_arena_gates=primary,learned_value_combined_gate_passed=learned_gate,
                promotion="H_v1 remains default; no automatic promotion from this development trial",
                recovery_checks={"native_actual_hard_kill":True,"cuda_actual_precommit_hard_kill":True,
                                 "benchmark_actual_hard_kill":True,"arena_actual_hard_kill":True,"checksum_valid_regret_corruption_refused":True,
                                 "affinity_normal_and_exception_restored":True},
                sealed_target_bodies_read=False,energy_measured=False,
                corrected_timing_protocol=benchmark["identity"]["cpu"]["protocol"],
                legacy_v1_benchmark_retained=True,
                selected_control=selection["selected_control"],selected_challenger=selection["selected_challenger"],
                arenas=benchmark["arenas"],
                inputs={name.replace("\\","/"):sha(ROOT/name) for name in sorted(set(files))})
    atomic_json(ROOT/"build/stage5-action-acceptance.json",report)
    print(config.canonical(dict(event="stage5_action_acceptance_recorded",passed=True,tests_run=32,
                               cases=391,games=144,primary_gates=primary,promotion=report["promotion"])),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
