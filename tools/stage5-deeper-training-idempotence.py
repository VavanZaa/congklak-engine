"""Verify whole completed experiment replay preserves every saved artifact."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,deeper_teacher_v2 as dt


def snapshot():
    plan=json.loads((ROOT/"build/stage5-deeper-training-plan-v3.json").read_text(encoding="utf-8"))
    files=[ROOT/"build"/name for name in ("stage5-deeper-training-plan-v3.json","stage5-deeper-training-results.json","stage5-deeper-training-verification.json")]
    for h in plan["teacher_depths"]:
        for f in plan["input_counts"]:
            for seed in plan["seeds"]:
                run=ROOT/f"data/training-stage5-deeper-h{h}-f{f}-seed{seed}"
                files.extend(p for p in run.iterdir() if p.is_file() and not p.name.endswith(("-wal","-shm")))
    return {str(p.relative_to(ROOT)):{"sha256":dt.sha(p),"mtime_ns":p.stat().st_mtime_ns,"bytes":p.stat().st_size} for p in sorted(files)}


def main():
    before=snapshot()
    dt.require(len([p for p in before if p.endswith(".pt")])==640,"Eight scheduled checkpoint histories are incomplete")
    for operation in ("train","verify"):
        subprocess.run(["py","-3.12","-m","congklak.progress","run","stage5-deeper-training",operation],cwd=ROOT,check=True)
    after=snapshot()
    dt.require(before==after,"Completed experiment replay rewrote a checkpoint/catalog/export/summary/verification")
    value={"schema":"stage5-deeper-completed-experiment-idempotence-v1","passed":True,"files_verified":len(before),"checkpoints_verified":640,
           "bytes_unchanged":True,"mtimes_unchanged":True,"whole_orchestrator_train_and_verify_rerun":True,
           "source_sha256":dt.sha(__file__),"orchestrator_sha256":dt.sha(ROOT/"tools/stage5-deeper-training-v3.py"),"files":before,"sealed_final_test_consumed":False}
    output=ROOT/"build/stage5-deeper-training-idempotence.json"
    if not output.exists() or json.loads(output.read_text(encoding="utf-8"))!=value: dt.atomic_json(output,value)
    print(config.canonical({k:v for k,v in value.items() if k!="files"}),flush=True)


if __name__=="__main__":main()
