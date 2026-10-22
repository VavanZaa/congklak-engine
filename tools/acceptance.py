"""Run all acceptance checks and save an evidence manifest beside the log."""
from __future__ import annotations
import hashlib
import argparse
import json
import platform
from datetime import datetime,timezone
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from congklak.config import RULES_DIGEST,EVALUATOR,POLICY
from congklak.native import Native

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement",help="Run a newly added test and append its evidence to the previous full run")
    args=parser.parse_args()
    out=ROOT/"build"/"acceptance"
    out.mkdir(parents=True,exist_ok=True)
    if args.supplement:
        sys.path.insert(0,str(ROOT/"tests"))
        file=out/"acceptance.json"
        manifest=json.loads(file.read_text(encoding="utf-8"))
        if manifest["native_build_sha256"]!=Native().build_id:
            raise RuntimeError("Supplement requires the same tested native build")
        if args.supplement in manifest.get("supplemental_tests",[]):
            raise RuntimeError("Supplement already recorded")
        suite=unittest.defaultTestLoader.loadTestsFromName(args.supplement)
        begin=time.monotonic()
        with (out/"acceptance.log").open("a",encoding="utf-8") as log:
            log.write("\nSupplemental independent gate: "+args.supplement+"\n")
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        manifest.setdefault("runs",[{"kind":"full_discovery","tests_run":manifest["tests_run"],"seconds":manifest["duration_seconds"]}]).append(
            {"kind":"supplemental_independent_gate","test":args.supplement,"tests_run":result.testsRun,"seconds":round(time.monotonic()-begin,3)})
        manifest.setdefault("supplemental_tests",[]).append(args.supplement)
        manifest["tests_run"]+=result.testsRun; manifest["failures"]+=len(result.failures); manifest["errors"]+=len(result.errors); manifest["skips"]+=len(result.skipped)
        manifest["passed"]=manifest["passed"] and result.wasSuccessful() and not result.skipped
        manifest["executed_at_utc"]=datetime.now(timezone.utc).isoformat()
        for name in ("tools/acceptance.py","tests/test_phase2_audit.py"):
            manifest["inputs"][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        file.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({k:manifest[k] for k in ("passed","tests_run","failures","errors","skips","runs")}))
        return 0 if manifest["passed"] else 1
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    start=time.monotonic()
    with (out/"acceptance.log").open("w",encoding="utf-8") as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    core=sys.modules["test_native_acceptance"].NativeAcceptance
    inputs=[]
    for directory,patterns in (("congklak",("*.py",)),("native",("*.hpp","*.cpp")),("tests",("*.py",)),("rules",("*.json",)),("tools",("*.ps1","*.cmd","*.py"))):
        for pattern in patterns:
            inputs.extend((ROOT/directory).glob(pattern))
    manifest={"passed":result.wasSuccessful() and not result.skipped,"tests_run":result.testsRun,"failures":len(result.failures),
              "executed_at_utc":datetime.now(timezone.utc).isoformat(),
              "errors":len(result.errors),"skips":len(result.skipped),"duration_seconds":round(time.monotonic()-start,3),
              "native_build_sha256":Native().build_id,"rules_digest":RULES_DIGEST,"evaluator":EVALUATOR,"policy":POLICY,
              "python":sys.version,"platform":platform.platform(),
              "corpus":{"reduced_states":8274,"reduced_state_horizons":41370,"seeded_98_seed_histories":core.histories,
                        "seed_ids":"0..1023","checked_transitions":core.transitions,"checked_horizon_reports":core.horizons},
              "mid_sowing_cancellation":{"completed_action_deposits":489,"cancelled_after_deposits":core.mid_sow_cancel_deposits,
                                          "partial_successor_published":False},
              "inputs":{str(p.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(inputs))}}
    build=ROOT/"build"/"native-build.json"
    if build.is_file(): manifest["compiler"]=json.loads(build.read_text(encoding="utf-8-sig"))
    if (ROOT/"data"/"pilot-10000"/"work.sqlite3").is_file():
        from congklak.pilot import load
        from congklak.oracle import Oracle
        with Oracle() as oracle:
            manifest["oracle"]={"manifest_sha256":oracle.manifest_sha256,**oracle.manifest}
        manifest["pilot"]=load(ROOT/"data"/"pilot-10000")
        recovery=ROOT/"build"/"phase2-recovery.json"
        if recovery.is_file(): manifest["pilot_campaign_recovery"]=json.loads(recovery.read_text(encoding="utf-8"))
    (out/"acceptance.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:manifest[k] for k in ("passed","tests_run","failures","errors","skips","duration_seconds","corpus")}))
    return 0 if result.wasSuccessful() and not result.skipped else 1

if __name__=="__main__": raise SystemExit(main())
