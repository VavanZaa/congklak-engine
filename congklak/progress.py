"""Read-only flow status, local dashboard and streamed tool progress.

SQLite catalogs describe committed work. Operational run logs never authorize
resumption, model promotion, or a recommendation. No training rows are loaded.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import uuid
from . import config

ROOT = Path(__file__).resolve().parent.parent
HASH_CACHE = {}


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = HASH_CACHE.get(str(path))
    if not cached or cached[0] != key:
        with path.open("rb") as stream:
            value = hashlib.file_digest(stream, "sha256").hexdigest()
        # One cached version per file; watch mode has bounded memory.
        HASH_CACHE[str(path)] = (key, value)
    cached = HASH_CACHE[str(path)]
    if cached[0] == key:
        return cached[1]
    raise RuntimeError("Hash cache changed unexpectedly")


@contextmanager
def database(path):
    path = Path(path).resolve()
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=.25)
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")  # Counts and metadata from one committed snapshot.
        yield db
    finally:
        db.close()


def metadata(db, key, table="meta"):
    if table == "metadata":
        row = db.execute("SELECT body,checksum FROM metadata WHERE name=?", (key,)).fetchone()
    else:
        row = db.execute("SELECT body,hash FROM meta WHERE key=?", (key,)).fetchone()
    if row is None:
        return None
    value = json.loads(row[0])
    if config.digest(value) != row[1]:
        raise ValueError("Metadata checksum mismatch: " + key)
    return value


def inside(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact escapes its managed directory")
    return path


def card(key, title, state, result, evidence, improvement, counters=()):
    return dict(id=key, title=title, state=state, result=result,
                evidence=evidence, improvement=improvement, counters=list(counters))


def count(label, completed, total=None):
    return dict(label=label, completed=completed, total=total)


def corpus(path, key, title):
    if not (path / "work.sqlite3").exists():
        return card(key, title, "pending", "No committed campaign yet.", str(path), "Generate and verify the corpus.")
    with database(path / "work.sqlite3") as db:
        identity = metadata(db, "identity")
        progress = metadata(db, "progress")
        current = metadata(db, "current")
        units = db.execute("SELECT COUNT(*) FROM collection_units").fetchone()[0]
        labels = db.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
        candidates = db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        complete = False
        if current:
            row = db.execute("SELECT status,hash FROM artifacts WHERE id=?", (current["generation"],)).fetchone()
            complete = bool(row and row[0] == "complete" and row[1] == config.digest(current)
                            and progress["stage"] == "complete"
                            and current["records"] == identity["target"]
                            and labels == identity["target"] and units == identity["family_count"])
            directory = inside(path, current["directory"])
            for entry in current["files"]:
                file = inside(directory, entry["name"])
                if not file.is_file() or file.stat().st_size != entry["size"]:
                    raise ValueError("Committed artifact missing or wrong size: " + entry["name"])
        counters = [count("families collected", units, identity["family_count"]),
                    count("labels committed", labels, identity["target"])]
        result = "Stage: " + progress["stage"]
        if complete:
            splits = current["splits"]
            result = (f"{current['records']:,} published; {splits['train']:,} train / "
                      f"{splits['development']:,} development / {splits['sealed_final_test']:,} sealed.")
        improvement = ("Improve ordinary-move targets and reachable examples; keep final-test data sealed."
                       if key == "campaign" else "Keep the original lineage frozen for reproducible comparisons.")
        out = card(key, title, "complete" if complete else "partial", result,
                   str(path / "work.sqlite3"), improvement, counters)
        out.update(stage=progress["stage"], semantic_sha256=current["semantic_sha256"] if current else None,
                   candidates_retained=candidates, manifest=current)
        return out


def training(path):
    name = path.name
    if (path / "quarantine.json").exists():
        return card(name, name, "quarantined", read_json(path / "quarantine.json")["reason"],
                    str(path / "quarantine.json"), "Excluded from selection and promotion; retain audit evidence.")
    with database(path / "catalog.sqlite3") as db:
        body, digest = db.execute("SELECT body,hash FROM config").fetchone()
        identity = json.loads(body)
        if config.digest(identity) != digest:
            raise ValueError("Training identity checksum mismatch")
        row = db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()
        step, metrics = 0, None
        if row:
            meta = json.loads(row[3])
            checkpoint = inside(path, row[1])
            if config.digest(meta) != row[4] or not checkpoint.is_file() or sha(checkpoint) != row[2]:
                raise ValueError("Latest committed checkpoint missing or invalid")
            step, metrics = row[0], meta.get("metrics")
        result = read_json(path / "result.json")
        complete = bool(result and result.get("status") == "complete"
                        and step == identity["steps"]
                        and result.get("identity") == identity
                        and result.get("steps_completed") == step
                        and (path / "model.json").is_file())
        if identity.get("variant") is not None:
            complete = complete and (path / "policy-model.json").is_file()
        inputs=identity.get("input_count",15)
        summary = f"Inputs {inputs}; width {identity['width']}; committed step {step:,}; {identity['device']}."
        if complete:
            finite = result["selected_development"]["finite_choice_horizon"]
            summary += (f" Selected step {result['best_step']:,}; dev MAE {finite['mae_seeds']:.3f} seeds; "
                        f"stratum MAE {finite['uniform_stratum_mae_seeds']:.3f}.")
        variant=f" · objective {identity['variant']}" if identity.get("variant") else ""
        if identity.get("teacher_depth") is not None: variant+=f" · teacher H{identity['teacher_depth']}"
        return card(name, f"{'Stage 5 ' if name.startswith('training-stage5-') else ''}Training F{inputs}/W{identity['width']}{variant} · seed {identity['seed']}",
                    "complete" if complete else "partial", summary, str(path / "catalog.sqlite3"),
                    "Reduce ordinary-move error and inference cost before another size increase.",
                    [count("optimizer steps committed", step, identity["steps"])])


def process_identity(pid):
    """Liveness plus process birth prevents a reused PID looking like our tool."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False, None
        try:
            code = wintypes.DWORD()
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
                return False, None
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                return None, None
            return True, (times[0].dwHighDateTime << 32) + times[0].dwLowDateTime
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True, Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19]
    except (OSError, IndexError):
        return False, None


def runs(root):
    result = []
    for path in sorted((root / "build/progress").glob("run-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
        try:
            value = read_json(path)
            if value["state"] == "running":
                alive, birth = process_identity(value["pid"])
                if not alive or birth != value.get("process_birth"):
                    value["state"] = "interrupted_or_exited"
                    value["last_event"] = "Process is absent; inspect its durable checkpoint and rerun the same command."
            result.append(value)
        except (OSError, ValueError, KeyError):
            continue
    return result


def current_evidence(root, report):
    changed = []
    for name, expected in report.get("inputs", {}).items():
        path = inside(root, name)
        if not path.is_file() or sha(path) != expected:
            changed.append(name)
    return changed


def snapshot(root=ROOT):
    root = Path(root).resolve()
    cards = []
    def add(key, title, function):
        try:
            value = function()
            if value:
                cards.append(value)
        except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
            cards.append(card(key, title, "read_error", str(exc), "Read-only observation",
                              "Inspect the named evidence; this observer does not repair or mutate it."))
    def build():
        a = read_json(root / "build/acceptance/acceptance.json")
        if not a:
            return card("build", "Simulator · TT/PVS", "pending", "No acceptance artifact.", "build/acceptance", "Build and run acceptance checks.")
        changed = current_evidence(root, a)
        dll = root / "build/congklak_native.dll"
        valid = a.get("passed") and not changed and dll.is_file() and sha(dll) == a["native_build_sha256"]
        return card("build", "Simulator · TT/PVS", "complete" if valid else "stale",
                    f"{a['tests_run']} baseline checks passed." if valid else "Baseline evidence changed or build missing.",
                    "build/acceptance/acceptance.json", "Preserve exact rule semantics while optimizing the next evaluator.")
    add("build", "Simulator · TT/PVS", build)
    def oracle():
        path = root / "data/oracle-k4.sqlite3"
        if not path.exists():
            return card("oracle", "K=4 oracle", "pending", "No oracle catalog.", str(path), "Generate and independently verify coverage.")
        with database(path) as db:
            stage = metadata(db, "stage", "metadata")
            manifest = metadata(db, "manifest", "metadata")
            total, solved = db.execute("SELECT COUNT(*),COALESCE(SUM(solved),0) FROM states").fetchone()
        complete = stage == "complete" and manifest and manifest.get("status") == "complete" and solved == total == manifest["covered_normalized_states"]
        return card("oracle", "K=4 oracle", "complete" if complete else "partial",
                    f"Stage: {stage}; {solved:,} solved normalized states. Exact only inside declared coverage.",
                    str(path), "Larger K needs a separate measured storage and solve-cost experiment.", [count("states solved", solved, total)])
    add("oracle", "K=4 oracle", oracle)
    add("pilot", "Original pilot database", lambda: corpus(root / "data/pilot-10000", "pilot", "Original pilot database"))
    add("campaign", "Depth-six database generation", lambda: corpus(root / "data/stronger-10000-v2", "campaign", "Depth-six database generation"))
    def data_quality():
        campaign = next((c for c in cards if c["id"] == "campaign"), {})
        m = campaign.get("manifest")
        registry = read_json(root / "build/full-lineage-registry.json")
        valid = bool(m and campaign["state"] == "complete" and m.get("q_equals_P") and registry
                     and registry.get("verified") and registry["fingerprint"] == m["identity"]["imported_registry_sha256"])
        return card("curation", "Split isolation · weights · pruning", "complete" if valid else "pending",
                    (f"{registry['reserved_keys']:,} imported state/quotient keys; {registry['prior_family_ids']:,} prior family IDs. "
                     "P=q, unit loss weights; disposable unselected payloads pruned after publication."
                     if valid else "Corrected corpus and registry evidence required."),
                    "build/full-lineage-registry.json; campaign catalog",
                    "The first depth-six corpus and its four models are quarantined after a two-row alias leak; use stronger-10000-v2.")
    add("curation", "Data curation", data_quality)
    def capacity():
        acceptance = read_json(root / "build/model-size-acceptance.json")
        path = root / "build/model-size-results.json"
        valid = bool(acceptance and acceptance.get("passed") and path.is_file()
                     and sha(path) == acceptance["report_sha256"]
                     and sha(root / "tools/compare-model-sizes.py") == acceptance["comparison_source_sha256"])
        return card("capacity", "Model-size experiment · W64 / W128 / W256", "complete" if valid else "pending",
                    "Six matched pilot runs complete; W256 selected for native testing. Full W256 hard-kill restore matched uninterrupted training."
                    if valid else "Matched model-size evidence unavailable or changed.",
                    "build/model-size-acceptance.json; docs/MODEL_SIZE_EXPERIMENT.md",
                    "Native equal-time results do not justify another capacity increase; improve features and inference first.")
    add("capacity", "Model-size experiment", capacity)
    for path in sorted((root / "data").glob("training-stronger2-*/catalog.sqlite3")):
        add(path.parent.name, "Training", lambda p=path.parent: training(p))
    for path in sorted((root / "data").glob("training-stage5-*/catalog.sqlite3")):
        add(path.parent.name, "Stage 5 training", lambda p=path.parent: training(p))
    def parity():
        p = read_json(root / "build/native-export-parity.json")
        return card("neural", "Native neural inference · search", "complete" if p and p.get("passed") else "pending",
                    "C++ evaluator and resumable search; selected W64/W256 checked on 1,000 development roots each."
                    if p and p.get("passed") else "Build and validate exported evaluator parity.",
                    "build/native-export-parity.json", "Optimize evaluator features and arithmetic; equal-time search currently loses depth.")
    add("neural", "Native neural search", parity)
    def analysis():
        path = root / "data/neural-demo/work.sqlite3"
        if not path.exists():
            return card("analyze", "Position analysis · recovery", "pending", "No demonstration job.", str(path), "Analyze a position; rerun the same command to resume.")
        with database(path) as db:
            row = db.execute("SELECT depth,body,hash FROM iterations ORDER BY depth DESC LIMIT 1").fetchone()
            calls = db.execute("SELECT COUNT(*) FROM invocations").fetchone()[0]
            last = db.execute("SELECT body,hash FROM invocations ORDER BY id DESC LIMIT 1").fetchone()
            depth = row[0] if row else 0
            if row and config.digest(json.loads(row[1])) != row[2]:
                raise ValueError("Analysis iteration checksum mismatch")
            last_body = json.loads(last[0]) if last else None
            if last and config.digest(last_body) != last[1]:
                raise ValueError("Analysis invocation checksum mismatch")
        target = last_body["limits"]["max_depth"] if last_body else None
        return card("analyze", "Position analysis · recovery", "complete" if target and depth >= target else "partial",
                    f"Opening demo: depth {depth} committed across {calls} invocations. Finite-horizon estimate; no full-game proof.",
                    str(path), "Use a new run directory for another position/model; rerun the same job after interruption.", [count("choice depth committed", depth, target)])
    add("analyze", "Position analysis", analysis)
    def benchmark():
        path = root / "data/neural-benchmark-v3/work.sqlite3"
        if not path.exists():
            return card("benchmark", "Equal-time comparison · arena", "pending", "No benchmark ledger.", str(path), "Measure speed and strength with paired independent starts.")
        with database(path) as db:
            identity = metadata(db, "identity")
            cases = db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
            games = db.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        report = read_json(root / "build/neural-benchmark-v3.json")
        complete = bool(report and report["identity"] == identity and cases == report["cases"] and games == identity["arena_pairs"] * 2)
        result = f"{cases} position cases / {games} committed games."
        if complete:
            arena = report["arena"]
            result += (f" W256: {arena['candidate_wins']} wins, {arena['candidate_losses']} losses, "
                       f"{arena['candidate_draws']} draw; score {arena['paired_mean_score']:.2%}.")
        return card("benchmark", "Equal-time comparison · arena", "complete" if complete else "partial", result,
                    str(path), "Candidate not promoted. Fix ordinary-move error and slower evaluation, then repeat paired games.",
                    [count("position cases committed", cases, report["cases"] if report else None),
                     count("games committed", games, identity["arena_pairs"] * 2)])
    add("benchmark", "Benchmark", benchmark)
    def audit():
        report = read_json(root / "build/phase3-acceptance.json")
        if not report:
            return card("audit", "Acceptance · interruption audit", "pending", "Phase-three checks ran; final evidence report is being assembled.",
                        "build/*-acceptance.log", "Bind the results to current files and complete progress-view checks.")
        changed = current_evidence(root, report)
        reporting_only = changed and all(p in ("congklak/progress.py", "tests/progress_acceptance.py", "tools/progress-dashboard.html", "docs/PROGRESS_REPORTING.md", "README.md") for p in changed)
        valid = report.get("passed") and not changed
        return card("audit", "Phase-three acceptance · recovery", "complete" if valid else "historical" if reporting_only else "stale",
                    f"{report['tests_run']} phase-three checks passed; native/data hard-kill and CUDA restore verified."
                    if valid else "15 checks passed on the recorded phase-three version. Reporting has been updated for Stage 5; its current audit is shown separately."
                    if reporting_only else "Acceptance evidence changed; inspect or rerun affected checks.",
                    "build/phase3-acceptance.json", "Keep checkpoint compatibility checks when implementing another evaluator.")
    add("audit", "Acceptance audit", audit)
    def stage5_tasks():
        report = read_json(root / "build/stage5-progress.json")
        if not report:
            return
        if report.get("schema") != "stage5-work-v1":
            raise ValueError("Unknown Stage 5 progress schema")
        for task in report["tasks"]:
            task=dict(task)
            if task["id"]=="stage5-audit" and (root/"build/stage5-acceptance.json").exists():
                accepted=read_json(root/"build/stage5-acceptance.json")
                valid=accepted.get("passed") and not current_evidence(root,accepted)
                changed=current_evidence(root,accepted)
                reporting_only=changed and all(p in ("congklak/progress.py","tools/progress-dashboard.html","tests/progress_acceptance.py","README.md","docs/PROGRESS_REPORTING.md") for p in changed)
                task["state"]="complete" if valid else "historical" if reporting_only else "stale"
                task["result"]=(f"{accepted['tests_run']} current checks pass; independent auditor checked407 cases and literally replayed72 games. Strength promotion remains held."
                                if valid else "26 checks passed on the recorded feature/speed experiment. Reporting has been updated for the next action-aware trial."
                                if reporting_only else "Stage 5 acceptance inputs changed; inspect or rerun affected checks.")
                task["evidence"]="build/stage5-acceptance.json; docs/STAGE5_INDEPENDENT_AUDIT.md"
            if task["id"]=="stage5-strength" and (root/"data/stage5-benchmark/work.sqlite3").exists():
                with database(root/"data/stage5-benchmark/work.sqlite3") as db:
                    cases=db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
                    games=db.execute("SELECT COUNT(*) FROM games").fetchone()[0]
                    complete=metadata(db,"stage5_complete")
                final=read_json(root/"build/stage5-benchmark.json")
                verified=complete and final and config.digest(final)==complete.get("report_sha256")
                task["counters"]=[count("position cases committed",cases,407),count("paired games committed",games,72)]
                task["state"]="complete" if verified and cases==407 and games==72 else "partial"
                task["result"]=f"{cases}/407 position cases and {games}/72 paired games committed. Original/exact and current ordinary cohorts stay separate."
            cards.append(card(task["id"], task["title"], task["state"], task["result"],
                              task.get("evidence", "build/stage5-progress.json"), task["improvement"], task.get("counters", ())))
    add("stage5", "Stage 5 work", stage5_tasks)
    def action_tasks():
        report=read_json(root/"build/stage5-action-progress.json")
        if not report: return
        if report.get("schema")!="stage5-action-work-v1": raise ValueError("Unknown action-work progress schema")
        for task in report["tasks"]:
            task=dict(task)
            ledger=inside(root,report.get("benchmark_run","data/stage5-action-benchmark"))/"work.sqlite3"
            output=inside(root,report.get("benchmark_output","build/stage5-action-benchmark.json"))
            if task["id"]=="stage5-action-strength" and ledger.exists():
                with database(ledger) as db:
                    case_count=db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
                    game_count=db.execute("SELECT COUNT(*) FROM games").fetchone()[0]
                    complete=metadata(db,"action_complete")
                final=read_json(output)
                valid=complete and final and config.digest(final)==complete.get("report_sha256")
                task["state"]="complete" if valid and case_count==391 and game_count==144 else "partial"
                task["counters"]=[count("position cases committed",case_count,391),count("paired games committed",game_count,144)]
                task["result"]=f"{case_count}/391 position cases and {game_count}/144 paired games committed; seed-margin objective, WDL secondary."
            if task["id"]=="stage5-action-audit" and (root/"build/stage5-action-acceptance.json").exists():
                accepted=read_json(root/"build/stage5-action-acceptance.json")
                valid=accepted.get("passed") and not current_evidence(root,accepted)
                changed=current_evidence(root,accepted)
                reporting_only=changed and all(p in ("congklak/progress.py","tools/progress-dashboard.html","tests/progress_acceptance.py","README.md","docs/PROGRESS_REPORTING.md") for p in changed)
                task["state"]="complete" if valid else "historical" if reporting_only else "stale"
                task["result"]=(f"{accepted['tests_run']} current checks pass; independent audit completed. Model promotion is a separate gate."
                                if valid else "32 checks passed on the recorded action experiment. Reporting is updated for deeper targets and relay features."
                                if reporting_only else "Action experiment acceptance inputs changed; inspect the evidence.")
                task["evidence"]="build/stage5-action-acceptance.json"
            cards.append(card(task["id"],task["title"],task["state"],task["result"],task.get("evidence","build/stage5-action-progress.json"),
                              task["improvement"],task.get("counters",())))
    add("stage5-action", "Action-aware work", action_tasks)
    def deeper_tasks():
        report=read_json(root/"build/stage5-deeper-progress.json")
        if not report:return
        if report.get("schema")!="stage5-deeper-work-v1":raise ValueError("Unknown deeper-work schema")
        for task in report["tasks"]:
            task=dict(task)
            teacher_root=inside(root,report.get("teacher_run","data/deeper-9000-v2"))
            if task["id"]=="stage5-deeper-data" and (teacher_root/"work.sqlite3").exists():
                with database(teacher_root/"work.sqlite3") as db:
                    labels=db.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
                    features=db.execute("SELECT COUNT(*) FROM features").fetchone()[0]
                    complete=metadata(db,"complete")
                manifest=read_json(teacher_root/"manifest.json")
                valid=complete and manifest and complete==manifest and labels==features==9000
                task["state"]="complete" if valid else "partial"
                task["counters"]=[count("H8 root labels committed",labels,9000),count("relay feature rows committed",features,9000)]
                task["result"]=f"{labels}/9000 complete H8 labels and {features}/9000 exact relay encodings; fixed training/development ownership."
            if task["id"]=="stage5-deeper-strength" and (root/"data/stage5-deeper-benchmark/work.sqlite3").exists():
                with database(root/"data/stage5-deeper-benchmark/work.sqlite3") as db:
                    cases=db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
                    games=db.execute("SELECT COUNT(*) FROM games").fetchone()[0]
                    complete=metadata(db,"deeper_complete")
                final=read_json(root/"build/stage5-deeper-benchmark.json")
                valid=complete and final and config.digest(final)==complete.get("report_sha256") and cases==391 and games==288
                task["state"]="complete" if valid else "partial"
                task["counters"]=[count("position cases committed",cases,391),count("paired games committed",games,288)]
                task["result"]=f"{cases}/391 cases and {games}/288 games committed; new arena families, paired seed margin primary."
            if task["id"]=="stage5-deeper-audit" and (root/"build/stage5-deeper-acceptance.json").exists():
                accepted=read_json(root/"build/stage5-deeper-acceptance.json")
                valid=accepted.get("passed") and not current_evidence(root,accepted)
                task["state"]="complete" if valid else "stale"
                task["result"]=f"{accepted.get('tests_run')} current checks pass; independent deeper/relay audit complete." if valid else "Deeper acceptance inputs changed."
                task["evidence"]="build/stage5-deeper-acceptance.json"
            cards.append(card(task["id"],task["title"],task["state"],task["result"],task.get("evidence","build/stage5-deeper-progress.json"),task["improvement"],task.get("counters",())))
    add("stage5-deeper","Deeper targets and relay features",deeper_tasks)
    cards.append(card("promotion", "Default-engine decision", "held", "H_v1 remains the default. Neural models remain experimental; each trial reports its measured gates.",
                      "build/stage5-deeper-training-results.json; build/stage5-deeper-benchmark.json",
                      "Use ordinary H8 error and paired seed-margin evidence to choose the next controlled experiment."))
    cards.sort(key=lambda c:0 if c["id"].startswith("stage5-deeper-") else 1 if c["id"].startswith("stage5-action-") else 2 if c["id"] in ("stage5-features","stage5-speed","stage5-strength","stage5-audit") else 3)
    for c in cards:
        c.pop("manifest", None)
        c["needs_improvement"] = (c["state"] != "complete" or c["id"].startswith("training-")
                                  or c["id"] in ("campaign", "capacity", "neural", "benchmark", "stage5-features", "stage5-speed", "stage5-strength")
                                  or c["id"] in ("stage5-action-training","stage5-action-native","stage5-action-strength"))
        if c["id"].startswith("stage5-deeper-") and c["id"]!="stage5-deeper-audit":c["needs_improvement"]=True
    invocations = runs(root)
    active = [r for r in invocations if r["state"] == "running"]
    activity = read_json(root / "build/progress/activity.json")
    if active:
        activity = {"state": "running", "message": "Running: " + " · ".join(
            r["tool"] + " — " + r["last_event"] for r in active)}
    return dict(schema="congklak-flow-progress-v1", observed_at=utc(),
                current_work=activity or {"state": "idle", "message": "No tracked build command is running."},
                active_tools=active, recent_runs=invocations[:12], tools=cards,
                scope="Committed work counters, not a percentage of game-solving or playing strength. A snapshot is not a new acceptance run.")


def describe(event):
    name = event.get("event", "progress")
    error = event.get("error")
    if error:
        return name + ": " + (str(error.get("message", error)) if isinstance(error, dict) else str(error))
    bits = [name]
    for key in ("stage", "cursor", "labels", "rows_written", "candidate_count", "states", "generated", "solved",
                "completed", "total", "step", "choice_depth", "job_id", "pair", "seat", "status"):
        if key in event:
            bits.append(f"{key}={event[key]}")
    identity = event.get("identity")
    if isinstance(identity, dict):
        for key in ("width", "seed", "steps", "device"):
            if key in identity:
                bits.append(f"{key}={identity[key]}")
    report = event.get("report")
    if isinstance(report, dict) and "choice_depth" in report:
        bits.append(f"committed_depth={report['choice_depth']}")
    finite = (event.get("metrics") or {}).get("finite_choice_horizon")
    if finite:
        bits.append(f"dev_MAE={finite['mae_seeds']:.3f}; stratum_MAE={finite['uniform_stratum_mae_seeds']:.3f}")
    result = event.get("result")
    if isinstance(result, dict):
        for key in ("records", "steps_completed", "best_step", "stop_reason", "verified"):
            if key in result:
                bits.append(f"{key}={result[key]}")
        if result.get("summary"):
            bits.append("depth=" + str(result["summary"].get("completed_choice_depth")))
    return " | ".join(bits)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        for attempt in range(12):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt==11:
                    raise
                # Windows readers/indexers may briefly deny delete sharing.
                # Retry publication; readers retain the previous whole file.
                time.sleep(min(.01*(attempt+1),.1))
    finally:
        temporary.unlink(missing_ok=True)


def command(tool, extra, root=ROOT):
    python = sys.executable
    if tool in ("training", "training-v2", "training-v3", "training-v4", "training-v5", "training-v7", "stage5-evaluation", "stage5-action-training", "stage5-deeper-training", "audit-training", "audit-curriculum", "audit-stage5-evaluation", "audit-action-training"):
        runtime = read_json(root / "build/training-runtime.json")
        if not runtime or not Path(runtime["python"]).is_file():
            raise ValueError("Run tools/setup-training.ps1 to select the existing CUDA environment.")
        python = runtime["python"]
    fixed = {
        "build": ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(root / "tools/build.ps1")],
        "build-neural": ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(root / "tools/build-neural.ps1")],
        "analyzer": [python, "-u", "-m", "congklak"],
        "oracle": [python, "-u", "-m", "congklak.oracle"],
        "pilot": [python, "-u", "-m", "congklak.pilot"],
        "campaign": [python, "-u", "-m", "congklak.campaign"],
        "training": [python, "-u", "-m", "congklak.training"],
        "training-v2": [python, "-u", "-m", "congklak.training_v2"],
        "training-v3": [python, "-u", "-m", "congklak.training_v3"],
        "training-v4": [python, "-u", "-m", "congklak.training_v4"],
        "training-v5": [python, "-u", "-m", "congklak.training_v5"],
        "training-v7": [python, "-u", "-m", "congklak.training_v7"],
        "deeper-teacher": [python, "-u", "-m", "congklak.deeper_teacher_v2"],
        "stage5-deeper-training": [python, "-u", str(root/"tools/stage5-deeper-training-v3.py")],
        "build-neural-relay": ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(root/"tools/build-neural-relay.ps1")],
        "neural-relay": [python, "-u", "-m", "congklak.neural_relay"],
        "stage5-relay-speed": [python, "-u", str(root/"tools/stage5-relay-speed.py")],
        "stage5-deeper-benchmark": [python, "-u", str(root/"tools/benchmark-stage5-deeper.py")],
        "audit-deeper": [python, "-u", str(root/"tools/stage5-deeper-audit.py")],
        "stage5-action-training": [python, "-u", str(root/"tools/stage5-action-training.py")],
        "build-neural-action": ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(root/"tools/build-neural-action.ps1")],
        "neural-action": [python, "-u", "-m", "congklak.neural_action"],
        "stage5-action-speed": [python, "-u", str(root/"tools/stage5-action-speed.py")],
        "stage5-action-benchmark": [python, "-u", str(root/"tools/benchmark-stage5-action-v2.py")],
        "audit-action": [python, "-u", str(root/"tools/stage5-action-audit.py")],
        "audit-action-training": [python, "-u", "-m", "unittest", "discover", "-s", "tests", "-p", "stage5_action_training_acceptance.py", "-v"],
        "stage5-evaluation": [python, "-u", str(root / "tools/stage5-evaluation.py")],
        "build-neural-fast": ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(root / "tools/build-neural-fast.ps1")],
        "neural-fast": [python, "-u", "-m", "congklak.neural_fast"],
        "stage5-speed": [python, "-u", str(root / "tools/stage5-speed.py")],
        "stage5-benchmark": [python, "-u", str(root / "tools/benchmark-stage5.py")],
        "audit-stage5": [python, "-u", str(root / "tools/stage5-audit.py")],
        "audit-stage5-evaluation": [python, "-u", "-m", "unittest", "discover", "-s", "tests", "-p", "stage5_evaluation_acceptance.py", "-v"],
        "neural": [python, "-u", "-m", "congklak.neural"],
        "benchmark": [python, "-u", str(root / "tools/benchmark-neural-v3.py")],
        "model-sizes": [python, "-u", str(root / "tools/compare-model-sizes.py")],
        "audit": [python, "-u", str(root / "tools/acceptance.py")],
        "audit-neural": [python, "-u", "-m", "unittest", "discover", "-s", "tests", "-p", "neural_acceptance.py", "-v"],
        "audit-campaign": [python, "-u", "-m", "unittest", "discover", "-s", "tests", "-p", "campaign_acceptance.py", "-v"],
        "audit-curriculum": [python, "-u", "-m", "unittest", "discover", "-s", "tests", "-p", "curriculum_training_acceptance.py", "-v"],
        "audit-training": [python, "-u", "-m", "unittest", "discover", "-s", "tests", "-p", "training_acceptance.py", "-v"],
        "audit-progress": [python, "-u", "-m", "unittest", "discover", "-s", "tests", "-p", "progress_acceptance.py", "-v"],
    }
    if tool not in fixed:
        raise ValueError("Unknown flow tool: " + tool + ". See docs/PROGRESS_REPORTING.md.")
    return fixed[tool] + extra


def run_command(tool, cmd, root=ROOT, raw=False):
    """Keep raw output and exit code; summarize every emitted progress event."""
    directory = Path(root) / "build/progress"
    directory.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    path = directory / ("run-" + run_id + ".json")
    log_path = directory / ("run-" + run_id + ".log")
    # Operational logs become durable evidence when acceptance pins them.
    # Keep those records as well as the newest 100 ordinary invocation logs.
    protected=set()
    for name in ("phase3-acceptance.json","stage5-acceptance.json","stage5-action-acceptance.json","stage5-deeper-acceptance.json","training-acceptance.json","model-size-acceptance.json"):
        manifest=read_json(Path(root)/"build"/name)
        if manifest:
            for entry in manifest.get("inputs",{}):
                evidence_path=(Path(root)/entry).resolve()
                if evidence_path.parent==directory.resolve():
                    protected.add(evidence_path)
    older = sorted(directory.glob("run-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[99:]
    for old in older:
        if old.resolve() in protected or old.with_suffix(".log").resolve() in protected:
            continue
        record = read_json(old)
        if record and record.get("state") != "running":
            old.with_suffix(".log").unlink(missing_ok=True)
            old.unlink()
    child = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
    state = dict(id=run_id, tool=tool, pid=child.pid, process_birth=process_identity(child.pid)[1],
                 started_at=utc(), updated_at=utc(), state="running", last_event="started", log=str(log_path), command=cmd)
    atomic_json(path, state)
    print(f"[{tool}] started · PID {child.pid} · {log_path}", flush=True)
    interrupted = False
    try:
        with log_path.open("w", encoding="utf-8") as log:
            for line in child.stdout:
                if log.tell() < 4 * 1024 * 1024:
                    log.write(line)
                    log.flush()
                try:
                    message = json.loads(line)
                    summary = describe(message) if isinstance(message, dict) else line.strip()
                except (ValueError, KeyError, TypeError):
                    summary = line.strip()
                if summary:
                    state.update(last_event=summary[:2000], updated_at=utc())
                    atomic_json(path, state)
                    print(line.rstrip() if raw else f"[{tool}] {summary}", flush=True)
        code = child.wait()
    except KeyboardInterrupt:
        interrupted = True
        print(f"[{tool}] interruption requested; waiting for the child checkpoint boundary.", flush=True)
        # Console Ctrl+C also reaches the child. Preserve its normal handler;
        # if it cannot finish promptly, its durable ledger still owns recovery.
        try:
            code = child.wait(timeout=10)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            child.terminate()
            code = child.wait(timeout=10)
    finally:
        child.stdout.close()
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        state.update(state="interrupted" if interrupted else ("complete" if child.returncode == 0 else "failed"),
                     exit_code=child.returncode, updated_at=utc(), finished_at=utc())
        atomic_json(path, state)
        print(f"[{tool}] {state['state']} · exit={child.returncode} · durable catalogs retain committed work", flush=True)
    return 130 if interrupted else code


def text_report(value):
    print("Congklak flow · " + value["observed_at"])
    print("Current work: " + value["current_work"]["message"])
    for tool in value["tools"]:
        print(f"\n[{tool['state'].upper()}] {tool['title']}: {tool['result']}")
        for counter in tool["counters"]:
            total = "" if counter["total"] is None else f"/{counter['total']:,}"
            print(f"  {counter['label']}: {counter['completed']:,}{total}")
        print("  Next: " + tool["improvement"])
    for run in value["active_tools"]:
        print(f"\n[RUNNING] {run['tool']}: {run['last_event']}")
    print("\n" + value["scope"])


def serve(root=ROOT, port=18765):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # The server exposes exactly two read-only resources; no files,
            # job-start endpoints, or data bodies are served.
            if self.path not in ("/", "/api/status"):
                self.send_error(404)
                return
            try:
                if self.path == "/api/status":
                    body = json.dumps(snapshot(root), ensure_ascii=True).encode()
                    mime = "application/json; charset=utf-8"
                else:
                    body = (Path(root) / "tools/progress-dashboard.html").read_bytes()
                    mime = "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError, KeyError) as exc:
                self.send_error(503, "Progress observer unavailable")
                print(str(exc), file=sys.stderr, flush=True)
        def log_message(self, *_):
            pass
    class Server(ThreadingHTTPServer):
        allow_reuse_address = False
        def server_bind(self):
            if os.name == "nt":
                import socket
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()
    server = Server(("127.0.0.1", port), Handler)
    print(f"Progress dashboard: http://127.0.0.1:{server.server_port}/ · read-only · Ctrl+C to close", flush=True)
    try:
        server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for name in ("status", "watch", "report"):
        p = sub.add_parser(name)
        p.add_argument("--json", action="store_true")
        if name == "watch":
            p.add_argument("--interval", type=float, default=2)
        if name == "report":
            p.add_argument("--output", default=str(ROOT / "build/progress/latest.json"))
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=18765)
    p = sub.add_parser("run")
    p.add_argument("--raw", action="store_true")
    p.add_argument("tool")
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.operation == "serve":
            if not 1 <= args.port <= 65535:
                parser.error("Port must be 1..65535")
            serve(port=args.port)
        elif args.operation == "run":
            extra = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
            return run_command(args.tool, command(args.tool, extra), raw=args.raw)
        else:
            if args.operation == "watch" and not .5 <= args.interval <= 60:
                parser.error("Watch interval must be 0.5..60 seconds")
            previous = None
            while True:
                value = snapshot()
                comparable = {k: v for k, v in value.items() if k != "observed_at"}
                if comparable != previous:
                    if args.operation == "report":
                        atomic_json(Path(args.output), value)
                    if args.json:
                        print(json.dumps(value), flush=True)
                    else:
                        text_report(value)
                    previous = comparable
                if args.operation != "watch":
                    break
                time.sleep(args.interval)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, sqlite3.Error) as exc:
        print("Progress tool: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
