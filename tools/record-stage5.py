"""Record Stage 5 acceptance with independent audit and current input hashes."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config
from congklak.progress import atomic_json,current_evidence,read_json,sha,utc


def require(condition,message):
    if not condition: raise ValueError(message)


def main():
    audit=read_json(ROOT/"build/stage5-audit/audit.json")
    require(audit["passed"] and audit["benchmark_audit_complete"],"Independent audit incomplete")
    require(not current_evidence(ROOT,audit),"Audited inputs changed")
    independent=audit["independent_acceptance"]
    require(independent["passed"] and independent["tests_run"]==9 and independent["failures"]==independent["errors"]==independent["skips"]==0,"Independent checks failed")
    for name,digest in independent["frozen_inputs"].items():
        require(sha(ROOT/name)==digest,"Frozen prior core changed: "+name)
    feature=read_json(ROOT/"build/stage5-evaluation-acceptance.json")
    require(feature["passed"] and feature["tests"]==4 and feature["failures"]==feature["errors"]==feature["skipped"]==0,"Feature acceptance failed")
    require(feature["trainer_sha256"]==sha(ROOT/"congklak/training_v3.py") and feature["features_sha256"]==sha(ROOT/"congklak/features_v3.py")
            and feature["source_sha256"]==sha(ROOT/"tests/stage5_evaluation_acceptance.py"),"Feature acceptance inputs changed")
    speed=read_json(ROOT/"build/stage5-speed-acceptance.json")
    require(speed["tests"]==5 and speed["failures"]==speed["errors"]==speed["skips"]==0,"Speed acceptance failed")
    require(not current_evidence(ROOT,{"inputs":speed["artifacts"]}),"Speed acceptance inputs changed")
    verified=read_json(ROOT/"build/stage5-evaluation-verification.json")
    require(verified["all_checks_passed"] and verified["checkpoint_files_verified"]==640 and verified["matched_sampler_sequences_equal_by_seed"],"Matched checkpoint/sampler verification failed")
    runs=sorted((ROOT/"build/progress").glob("run-*.json"),key=lambda p:p.stat().st_mtime,reverse=True)
    progress=next((read_json(p) for p in runs if read_json(p).get("tool")=="audit-progress"),None)
    require(progress and progress["state"]=="complete" and progress["exit_code"]==0,"No successful progress acceptance")
    logfile=Path(progress["log"]);log=logfile.read_text();require(re.search(r"Ran 8 tests in [\d.]+s",log) and re.search(r"\nOK\s*$",log),"Incomplete progress checks")
    benchmark=read_json(ROOT/"build/stage5-benchmark.json")
    with sqlite3.connect((ROOT/"data/stage5-benchmark/work.sqlite3").as_uri()+"?mode=ro",uri=True) as db:
        body,digest=db.execute("SELECT body,hash FROM meta WHERE key='stage5_complete'").fetchone()
        marker=json.loads(body);require(config.digest(marker)==digest and marker["report_sha256"]==config.digest(benchmark),"Benchmark commit/report differ")
        require(db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]==407 and db.execute("SELECT COUNT(*) FROM games").fetchone()[0]==72,"Benchmark incomplete")
    files=list(audit["inputs"])+list(independent["frozen_inputs"])+list(speed["artifacts"])
    files += ["tools/record-stage5.py","congklak/progress.py","tests/progress_acceptance.py","tools/progress-dashboard.html",
              "README.md","docs/STAGE5_RESULTS.md","docs/STAGE5_INDEPENDENT_AUDIT.md","docs/PROGRESS_REPORTING.md",
              "build/stage5-audit/audit.json","build/stage5-audit/tests.json","build/stage5-audit/legacy-exact-aliases.json",
              "build/stage5-audit/arena-margin-secondary.json","build/stage5-evaluation-acceptance.json",
              "build/stage5-evaluation-verification.json","build/stage5-speed-results.json","build/neural-fast-build.json",
              "tests/stage5_evaluation_acceptance.py","tests/stage5_speed_acceptance.py","tools/stage5-speed.py",
              "tools/build-neural-fast.cmd","tools/build-neural-fast.ps1",str(logfile.relative_to(ROOT)).replace("\\","/")]
    evaluation=read_json(ROOT/"build/stage5-evaluation-results.json")
    for row in evaluation["rows"]:
        path=Path(row["model"]);files += [str(path),str(path.parent/"result.json")]
    report=dict(schema="stage5-acceptance-v1",recorded_at_utc=utc(),passed=True,tests_run=26,failures=0,errors=0,skips=0,
                groups={"feature_training":4,"native_speed":5,"independent_auditor":9,"progress_and_recovery":8},
                independent_audit_complete=True,checkpoint_files_verified=640,position_cases=407,terminal_games=72,
                recovery_checks={"native_actual_hard_kill":True,"cuda_actual_hard_kill_before_and_after_commit":True,
                                 "benchmark_actual_hard_kill":True,"production_reporting_failure_resumed_at_case":189,
                                 "transient_windows_sharing_retry_tested":True,"checksum_valid_saved_regret_corruption_refused":True},
                selected_models={f:{k:row[k] for k in ("model","weights_sha256","width","input_count","ordinary_gate_passed")}
                                 for f,row in evaluation["selected"].items()},
                old_exact_diagnostics_sealed_quotient_aliases=16,sealed_target_bodies_read=False,
                promotion="held; both ordinary gates fail; H_v1 remains default",arenas=benchmark["arenas"],
                inputs={name.replace("\\","/"):sha(ROOT/name) for name in sorted(set(files))})
    atomic_json(ROOT/"build/stage5-acceptance.json",report)
    print(json.dumps(dict(event="stage5_acceptance_recorded",passed=True,tests_run=26,cases=407,games=72,promotion=report["promotion"])),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
