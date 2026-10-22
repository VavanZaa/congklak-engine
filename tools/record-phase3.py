"""Bind completed phase-three acceptance logs to current immutable evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from congklak import campaign, config, pilot
from congklak.progress import atomic_json, current_evidence, read_json, sha, utc


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    original = read_json(ROOT / "build/acceptance/acceptance.json")
    training = read_json(ROOT / "build/training-acceptance.json")
    require(original["passed"] and training["passed"], "Prior acceptance did not pass")
    require(not current_evidence(ROOT, original), "Original engine acceptance inputs changed")
    require(not current_evidence(ROOT, training), "Original training acceptance inputs changed")
    require(sha(ROOT / "build/congklak_native.dll") == original["native_build_sha256"], "Frozen DLL changed")
    progress_runs = sorted((ROOT / "build/progress").glob("run-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    recent = next((read_json(p) for p in progress_runs if read_json(p).get("tool") == "audit-progress"), None)
    require(recent and recent["state"] == "complete" and recent["exit_code"] == 0, "No successful progress acceptance run")
    groups = [("native_neural", ROOT / "build/neural-acceptance.log", 6),
              ("corrected_corpus", ROOT / "build/campaign-acceptance.log", 3),
              ("current_cuda_training", ROOT / "build/curriculum-acceptance.log", 1),
              ("progress_observer", Path(recent["log"]), 5)]
    test_results = []
    for name, path, expected in groups:
        log = path.read_text(encoding="utf-8-sig")
        match = re.search(r"Ran (\d+) tests? in ([\d.]+)s", log)
        require(match and int(match[1]) == expected and re.search(r"\nOK\s*$", log), "Incomplete/failed test log: " + name)
        test_results.append(dict(group=name, tests_run=expected, passed=True, duration_seconds=float(match[2]),
                                 log=str(path.relative_to(ROOT)).replace("\\", "/"), log_sha256=sha(path),
                                 completed_at_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()))
    manifest = pilot.load(campaign.DEFAULT)
    _, reserved, families, fingerprint = campaign.imports()
    require(fingerprint == manifest["identity"]["imported_registry_sha256"], "Registry changed")
    registry = read_json(ROOT / "build/full-lineage-registry.json")
    require(registry["verified"] and registry["fingerprint"] == fingerprint, "Registry evidence does not match")
    db_path = campaign.DEFAULT / manifest["directory"] / "dataset.sqlite3"
    checked = overlaps = 0
    db = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        # Metadata projections only; sealed target bodies do not enter Python.
        for key, family in db.execute("SELECT state,json_extract(body,'$.family') FROM examples"):
            key = bytes(key)
            overlaps += (("state", key) in reserved or ("quotient", pilot.quotient(pilot.from_key(key))) in reserved or family in families)
            checked += 1
    finally:
        db.close()
    require(checked == 10000 and overlaps == 0, "Cross-generation overlap")
    parity = read_json(ROOT / "build/native-export-parity.json")
    benchmark = read_json(ROOT / "build/neural-benchmark-v3.json")
    selection = read_json(ROOT / "build/curriculum-selection.json")
    require(parity["passed"] and not benchmark["sealed_final_test_consumed"], "Invalid parity/benchmark")
    require(sha(ROOT / "congklak/neural.py") == benchmark["identity"]["binding_sha256"], "Benchmark binding changed")
    require(sha(ROOT / "tools/benchmark-neural-v3.py") == benchmark["identity"]["source_sha256"], "Benchmark source changed")
    require(sha(ROOT / "build/congklak_neural.dll") == benchmark["identity"]["native_nn_sha256"], "Neural build changed")
    for width in (64, 256):
        chosen = selection["selected"][str(width)]
        require(benchmark["identity"]["models"]["NN" + str(width)] == chosen["weights_sha256"], "Benchmark selection changed")
        require(any(m["weights_sha256"] == chosen["weights_sha256"] and m["quantized_units_equal"] for m in parity["models"]), "Parity selection changed")
    quarantine_paths = [ROOT / "data/stronger-10000"] + sorted((ROOT / "data").glob("training-stronger-w*"))
    require(len(quarantine_paths) == 5 and all(read_json(p / "quarantine.json")["status"] == "quarantined" for p in quarantine_paths), "Quarantine evidence missing")
    files = [ROOT / name for name in ["native/neural.cpp", "congklak/neural.py", "congklak/campaign.py",
             "congklak/training_v2.py", "congklak/progress.py", "tools/build-neural.cmd", "tools/build-neural.ps1",
             "tools/benchmark-neural-v3.py", "tools/progress-dashboard.html", "tools/record-phase3.py",
             "tests/neural_acceptance.py", "tests/campaign_acceptance.py", "tests/curriculum_training_acceptance.py",
             "tests/progress_acceptance.py", "build/congklak_native.dll", "build/congklak_neural.dll",
             "build/native-export-parity.json", "build/full-lineage-registry.json", "build/neural-benchmark-v3.json",
             "build/curriculum-selection.json", "build/acceptance/acceptance.json", "build/training-acceptance.json",
             "README.md", "docs/PHASE3_RESULTS.md", "docs/PROGRESS_REPORTING.md", "docs/DATA_AND_TRAINING_PLAN.md"]]
    files += [p for _, p, _ in groups]
    files += [campaign.DEFAULT / manifest["directory"] / f["name"] for f in manifest["files"]]
    files += [p / "quarantine.json" for p in quarantine_paths]
    for path in sorted((ROOT / "data").glob("training-stronger2-*")):
        result = read_json(path / "result.json")
        require(result["status"] == "complete" and result["steps_completed"] == 2000 and result["identity"]["dataset_semantic_sha256"] == manifest["semantic_sha256"], "Current training incomplete")
        files += [path / "model.json", path / "result.json"]
    report = dict(schema="phase3-acceptance-v1", recorded_at_utc=utc(), passed=True, tests_run=15,
                  failures=0, errors=0, skips=0, test_results=test_results,
                  prior_acceptance_inputs_unchanged={"engine_tests": original["tests_run"], "training_tests": training["tests_run"]},
                  corpus_semantic_sha256=manifest["semantic_sha256"],
                  lineage_check={"records_checked": checked, "overlaps": overlaps, "reserved_keys": len(reserved),
                                 "prior_family_ids": len(families), "registry_sha256": fingerprint},
                  recovery_checks={"native_analysis_actual_hard_kill": True, "corpus_actual_hard_kills": True,
                                   "current_cuda_bitwise_restore": True, "prior_cuda_hard_kill": True,
                                   "final_benchmark_hard_kill_tested": False},
                  sealed_final_test_consumed=False, promotion="held; H_v1 remains default", arena=benchmark["arena"],
                  inputs={str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files})
    atomic_json(ROOT / "build/phase3-acceptance.json", report)
    print(json.dumps(dict(event="phase3_evidence_recorded", passed=True, tests_run=15,
                          lineage_overlap=overlaps, promotion=report["promotion"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
