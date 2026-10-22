"""Bounded natural 50–74 ordinary roots and H8/H10 stability study, no training."""
from collections import Counter, defaultdict
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from . import config, data_pilot as prior, deeper_teacher_v2 as teacher, reports
from .native import ROOT, Native, Control, flatten
from .neural_relay import Model
from .puct_cpu_inference import CpuBatchEvaluator

SCHEMA = "natural-ordinary-midgame-H8-H10-study-v1"
DEFAULT = ROOT / "data/natural-study-v1"
BASELINE = ROOT / "build/natural-study-baseline.json"
REGISTRY = ROOT / "build/natural-study-registry.json"
PLAN = ROOT / "build/natural-study-plan.json"
IMPLEMENTATION = ROOT / "build/natural-study-implementation.json"
OUTPUT = ROOT / "build/natural-study-result.json"
MODEL = prior.MODEL
MODEL_SHA = prior.MODEL_SHA
SOURCES = ("congklak/natural_study.py", "tools/natural-study.py", "tests/natural_study_author.py", "docs/NATURAL_STUDY_SCOPE.md")
CONTRACT = {"families": 512, "seed": 20261021, "max_choices": 192, "random_opening_choices": 8, "epsilon": .25,
            "generation": "sealed data_pilot.generate_family: even random, odd fixed-policy epsilon0.25",
            "ownership": "sealed family hash before generation; selected-root bridge, shared prefixes possible",
            "fresh_band": "50-74", "fresh_caps": {"training_candidate": 24, "validation": 8},
            "followups": 15, "one_fresh_root_per_family": True,
            "h8": {"depth": 8, "node_budget": 2000000}, "h10": {"depth": 10, "node_budget": 5000000},
            "time_ms": 0, "tt_mb": 16, "memory_mb": 512, "use_pvs": True,
            "h10_denominator": "Every selected fresh and old followup root, irrespective of H8 completion",
            "training_performed": False, "sealed_final_test_consumed": False, "automatic_expansion": False}
MANIFEST_FIELDS = {"schema", "identity", "status", "families", "selected_fresh", "followups", "h8_new_labels",
                   "h8_inherited_labels", "h8_rejected", "h10_labels", "h10_rejected", "paired_complete", "selection",
                   "table_sha256", "training_performed", "torch_imported", "sealed_final_test_consumed", "elapsed_seconds_initial_run"}
IDENTITY_FIELDS = {"schema", "settings", "fixture", "rules_digest", "fixed_model_sha256", "fixed_model_weights_sha256",
                   "source_sha256", "tool_sha256", "registry_sha256", "baseline_sha256", "native_build_sha256",
                   "plan_sha256", "implementation_sha256", "reserve_bytes", "quota_bytes"}
require, sha, read, publish, table, stopped = prior.require, prior.sha, prior.read, prior.publish, prior.table, prior.stopped


def checked_baseline():
    value = read(BASELINE)
    require(value["passed"] is True and value["inputs_count"] == len(value["inputs"]) == 2060 and value["prior_accepted_inputs"] == 2058 and value["fixed_model_sha256"] == sha(MODEL) == MODEL_SHA, "Study baseline/model differs")
    for name, checksum in value["inputs"].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == checksum, "Sealed input changed: " + name)
    return value


def make_registry():
    checked_baseline()
    roots, labels, _, manifest = prior.load_rows()
    require(len(roots) == len(labels) == 62, "Prior completed pilot changed")
    protected = read(prior.REGISTRY)
    quotients, families = set(protected["quotients"]), set(protected["families"])
    quotients.update(row["quotient"] for row in roots.values())
    with closing(sqlite3.connect((prior.DEFAULT / "work.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        pool = table(db, "families", "idx")
    require(len(pool) == 256, "Prior complete family pool changed")
    families.update(row["family"] for row in pool.values())
    followups = {key: {"root": row, "h8": labels[key]} for key, row in roots.items() if row["band"] == "50-74"}
    require(len(followups) == 15, "All15 old midgame roots must be followed up")
    value = {"schema": "natural-study-prior-ownership-v1", "rules_digest": config.RULES_DIGEST,
             "quotients": sorted(quotients), "families": sorted(families), "followups": followups,
             "prior_pilot_root_count": len(roots), "prior_pilot_family_count": len(pool),
             "prior_manifest_sha256": sha(prior.DEFAULT / "manifest.json"), "prior_journal_sha256": sha(prior.DEFAULT / "work.sqlite3"),
             "prior_registry_sha256": sha(prior.REGISTRY), "baseline_sha256": sha(BASELINE),
             "sealed_target_bodies_read": False, "no_prefix_disjointness_claim": True}
    value["sha256"] = config.digest(value); publish(REGISTRY, value)
    return value


def plan():
    registry = make_registry()
    value = {"schema": SCHEMA, "contract": CONTRACT, "rules_digest": config.RULES_DIGEST,
             "fixed_model_sha256": MODEL_SHA, "baseline_sha256": sha(BASELINE), "registry_sha256": sha(REGISTRY),
             "followup_state_keys": list(registry["followups"]), "old_h8_reused_unchanged": True,
             "h10_is_deeper_estimate_not_game_truth": True, "paired_analysis": "Complete H8/H10 intersection; all unpaired counts explicit"}
    publish(PLAN, value); return value


def freeze():
    plan(); baseline = checked_baseline()
    value = {"schema": "natural-study-implementation-v1", "plan_sha256": sha(PLAN), "registry_sha256": sha(REGISTRY),
             "baseline_sha256": sha(BASELINE), "sources": {name: sha(ROOT / name) for name in SOURCES}, "prior_inputs": baseline["inputs"]}
    publish(IMPLEMENTATION, value); return value


def admit():
    value, implementation, baseline = read(PLAN), read(IMPLEMENTATION), checked_baseline()
    require(value["schema"] == SCHEMA and value["contract"] == CONTRACT and value["fixed_model_sha256"] == MODEL_SHA and value["rules_digest"] == config.RULES_DIGEST, "Study fixed plan differs")
    require(implementation["schema"] == "natural-study-implementation-v1" and implementation["plan_sha256"] == sha(PLAN) and implementation["registry_sha256"] == sha(REGISTRY) and implementation["baseline_sha256"] == sha(BASELINE), "Study source freeze differs")
    require(implementation["sources"] == {name: sha(ROOT / name) for name in SOURCES} and implementation["prior_inputs"] == baseline["inputs"], "Study producer/prior input changed")
    return value, implementation


def settings_for(fixture=False, families=None, followups=None):
    settings = json.loads(config.canonical(CONTRACT))
    if fixture:
        settings["families"] = 8 if families is None else config.integer(families, "fixture_families", 1, 32)
        settings["followups"] = 2 if followups is None else config.integer(followups, "fixture_followups", 1, 15)
        settings["fresh_caps"] = {"training_candidate": 1, "validation": 1}
    else:
        require(families is None and followups is None, "Production budget must equal fixed study")
    return settings


def wrap(row, role, old_label=None):
    fields = ("state_key", "state", "legal_mask", "family", "cohort", "band", "stratum", "quotient", "active")
    return {**{name: row[name] for name in fields}, "study_role": role, "root": row,
            "source_root_sha256": config.digest(row), "prior_h8_item_sha256": config.digest(old_label) if old_label is not None else None}


def select(families, settings, registry):
    observations = [row for family in families.values() for row in family["candidates"] if row["band"] == "50-74"]
    cohorts = defaultdict(set)
    for row in observations:
        cohorts[row["quotient"]].add(row["cohort"])
    collisions = {key for key, owners in cohorts.items() if len(owners) > 1}
    protected, prior_families = set(registry["quotients"]), set(registry["families"])
    available = [r for r in observations if r["quotient"] not in protected | collisions and r["family"] not in prior_families]
    rows, used_q, used_family, counts = {}, set(), set(), Counter()
    for row in sorted(available, key=lambda r: (r["provenance"]["family_index"], r["provenance"]["step"], r["state_key"])):
        cohort = row["cohort"]
        if counts[cohort] >= settings["fresh_caps"][cohort] or row["quotient"] in used_q or row["family"] in used_family:
            continue
        rows[row["state_key"]] = wrap(row, "fresh"); used_q.add(row["quotient"]); used_family.add(row["family"]); counts[cohort] += 1
    inherited = {}
    for key in sorted(registry["followups"])[:settings["followups"]]:
        reference = registry["followups"][key]
        require(key not in rows, "Protected followup leaked to fresh cohort")
        rows[key] = wrap(reference["root"], "followup", reference["h8"]); inherited[key] = reference["h8"]
    evidence = {"ordinary_50_74_observations": len(observations), "unique_candidate_quotients": len(cohorts),
                "cross_cohort_collisions": len(collisions), "protected_observations": sum(r["quotient"] in protected for r in observations),
                "fresh_counts": {c: counts[c] for c in settings["fresh_caps"]},
                "shortfall": {c: settings["fresh_caps"][c] - counts[c] for c in settings["fresh_caps"]},
                "followups": len(inherited), "observed_legal_action_counts": dict(sorted(Counter(str(r["legal_mask"].bit_count()) for r in observations).items()))}
    return dict(sorted(rows.items())), dict(sorted(inherited.items())), evidence


def validate_label(item, row, depth, native):
    root = row["root"]
    require(set(item) == {"root_sha256", "label", "report"} and item["root_sha256"] == config.digest(root) == row["source_root_sha256"], "Target original-root linkage differs")
    label, report = item["label"], item["report"]
    teacher.validate_label(label, root["state"], root["legal_mask"], depth, native)
    reports.validate(report, root["state"], config.identity(native.build_id), native)
    require(report["search_options"]["tt_mb"] == 16 and report["search_options"]["use_pvs"] is True, "Stage target teacher TT/PVS protocol differs")
    require(report["choice_depth"] == depth and label["target_player0"] == report["depth_result"]["value_player0"] and label["best_moves"] == report["best_moves"] and label["action_scores"] == [{"pit": c["pit"], "value_player0": c["depth_result"]["value_player0"]} for c in report["candidates"]], "Stage target/Q/best report linkage differs")
    require(label["coverage"] == report["coverage"] and label["work"] == {name: report[name] for name in ("nodes", "deposits", "elapsed_ms", "peak_process_bytes")}, "Stage target work/coverage linkage differs")


def attempt_for(item, row, stage, result=None, inherited=False):
    return {"study_row_sha256": config.digest(row), "source_root_sha256": row["source_root_sha256"], "stage": stage,
            "status": "inherited_complete" if inherited else "complete" if item is not None else "budget_rejected",
            "stop_reason": "max_depth" if inherited else result["stop_reason"],
            "completed_depth": 8 if inherited else result["report"]["choice_depth"] if result.get("report") else 0,
            "work": dict(item["label"]["work"]) if inherited else {name: result[name] for name in ("nodes", "deposits", "elapsed_ms", "peak_process_bytes")},
            "partial_report": None if item is not None else result.get("report"), "label_sha256": config.digest(item) if item is not None else None}


def validate_attempts(rows, attempts, labels, stage, settings, registry, native):
    depth, budget = settings[stage]["depth"], settings[stage]["node_budget"]
    inherited_keys = {key for key, row in rows.items() if row["study_role"] == "followup"} if stage == "h8" else set()
    targets = [key for key in sorted(rows) if key not in inherited_keys]
    actual_new = sorted(set(attempts) - inherited_keys)
    require(set(attempts) <= set(rows) and actual_new == targets[:len(actual_new)] and inherited_keys <= set(attempts) and set(labels) <= set(attempts), "Teacher attempt ownership/prefix differs")
    for key, attempt in attempts.items():
        row = rows[key]
        require(set(attempt) == {"study_row_sha256", "source_root_sha256", "stage", "status", "stop_reason", "completed_depth", "work", "partial_report", "label_sha256"} and attempt["study_row_sha256"] == config.digest(row) and attempt["source_root_sha256"] == row["source_root_sha256"] and attempt["stage"] == stage, "Teacher attempt fields/root differ")
        require(type(attempt["completed_depth"]) is int and 0 <= attempt["completed_depth"] <= depth and set(attempt["work"]) == {"nodes", "deposits", "elapsed_ms", "peak_process_bytes"} and all(type(v) is int and v >= 0 for v in attempt["work"].values()), "Teacher attempt work/depth differs")
        if key in inherited_keys:
            reference = registry["followups"][key]["h8"]
            require(key in labels and labels[key] == reference and attempt == attempt_for(reference, row, "h8", inherited=True), "Old H8 reference was rewritten or rerun")
            validate_label(labels[key], row, 8, native)
        elif attempt["status"] == "complete":
            require(key in labels and attempt["stop_reason"] == "max_depth" and attempt["completed_depth"] == depth and attempt["partial_report"] is None and attempt["label_sha256"] == config.digest(labels[key]), "Complete target ownership differs")
            validate_label(labels[key], row, depth, native)
            require(0 < attempt["work"]["nodes"] <= budget and all(attempt["work"][name] >= labels[key]["report"][name] for name in attempt["work"]), "Complete target physical work differs")
        else:
            require(attempt["status"] == "budget_rejected" and key not in labels and attempt["stop_reason"] == "node_budget" and attempt["completed_depth"] < depth and attempt["work"]["nodes"] == budget and attempt["label_sha256"] is None, "Partial target admitted or budget evidence differs")
            partial = attempt["partial_report"]
            require((partial is None) == (attempt["completed_depth"] == 0), "Partial target report/depth differs")
            if partial:
                reports.validate(partial, row["state"], config.identity(native.build_id), native)
                require(partial["search_options"]["tt_mb"] == 16 and partial["search_options"]["use_pvs"] is True, "Partial teacher TT/PVS protocol differs")
                require(partial["choice_depth"] == attempt["completed_depth"] and all(partial[name] <= attempt["work"][name] for name in attempt["work"]), "Partial target work coverage differs")


def execute_target(row, stage, settings, native, cancel):
    depth, budget = settings[stage]["depth"], settings[stage]["node_budget"]
    p = row["state"]
    with Control(native, max_depth=depth, time_ms=0, node_budget=budget, memory_mb=512, tt_mb=16, use_pvs=True) as control:
        if cancel is not None and hasattr(cancel, "_register"):
            cancel._register(control)
        try:
            result = native.analyze(flatten(p), actor=p["controller"], total=98, max_depth=depth, time_ms=0, node_budget=budget,
                                    memory_mb=512, tt_mb=16, use_pvs=True, keep_history=False, control=control)
        finally:
            if cancel is not None and hasattr(cancel, "_remove"):
                cancel._remove(control)
    if stopped(cancel):
        return None, None
    report = result.get("report"); item = None
    if result["stop_reason"] == "max_depth" and report and report["choice_depth"] == depth:
        label = {"kind": "finite_choice_horizon", "choice_depth": depth, "oracle_manifest_sha256": None,
                 "target_player0": report["depth_result"]["value_player0"], "best_moves": report["best_moves"],
                 "action_scores": [{"pit": c["pit"], "value_player0": c["depth_result"]["value_player0"]} for c in report["candidates"]],
                 "coverage": report["coverage"], "work": {name: report[name] for name in ("nodes", "deposits", "elapsed_ms", "peak_process_bytes")}}
        item = {"root_sha256": row["source_root_sha256"], "label": label, "report": report}; validate_label(item, row, depth, native)
    else:
        require(result["stop_reason"] == "node_budget", "Unexpected teacher stop is not a coverage shortfall")
    return item, attempt_for(item, row, stage, result)


def counts(rows, h8, h10, a8, a10):
    return {"selected_fresh": sum(row["study_role"] == "fresh" for row in rows.values()),
            "followups": sum(row["study_role"] == "followup" for row in rows.values()),
            "h8_new_labels": sum(rows[key]["study_role"] == "fresh" for key in h8),
            "h8_inherited_labels": sum(rows[key]["study_role"] == "followup" for key in h8),
            "h8_rejected": sum(a["status"] == "budget_rejected" for a in a8.values()),
            "h10_labels": len(h10), "h10_rejected": sum(a["status"] == "budget_rejected" for a in a10.values()),
            "paired_complete": len(set(h8) & set(h10))}


def validate_complete_marker(manifest, identity, pool, rows, stores, selection):
    """Side-effect-free guard used before exports and by the closed reader."""
    require(set(identity) == IDENTITY_FIELDS and identity["schema"] == SCHEMA and identity["reserve_bytes"] == 2*1024**3 and identity["quota_bytes"] == 50*1024**3, "Closed study identity fields/schema/reserve differ")
    require(set(manifest) == MANIFEST_FIELDS and manifest["schema"] == SCHEMA and manifest["identity"] == identity and manifest["status"] == "complete", "Closed study fields/identity differ")
    require(len(pool) == identity["settings"]["families"] and len(stores["h8_attempts"]) == len(stores["h10_attempts"]) == len(rows) and selection is not None, "Complete marker has unfinished collection/targets")
    actual = counts(rows,stores["h8_labels"],stores["h10_labels"],stores["h8_attempts"],stores["h10_attempts"])
    require(type(manifest["families"]) is int and manifest["families"] == len(pool) and all(type(manifest[name]) is int and manifest[name] == value for name,value in actual.items()), "Closed study counters differ")
    require(manifest["selection"] == selection and manifest["table_sha256"] == {"families":config.digest(pool),"roots":config.digest(rows),**{name:config.digest(value) for name,value in stores.items()}}, "Closed study selection/table digests differ")
    require(manifest["training_performed"] is False and manifest["torch_imported"] is False and manifest["sealed_final_test_consumed"] is False and type(manifest["elapsed_seconds_initial_run"]) in (int,float) and math.isfinite(manifest["elapsed_seconds_initial_run"]) and manifest["elapsed_seconds_initial_run"] > 0, "Closed study scope/clock differs")


def run(path=DEFAULT, stage="all", fixture=False, families=None, followups=None, on_event=None, cancel=None):
    path = Path(path).resolve(); require(stage in ("collect", "h8", "h10", "all"), "Unknown study stage")
    settings = settings_for(fixture, families, followups)
    if fixture:
        require(path.is_relative_to(ROOT / "build"), "Fixture must stay inside build")
    else:
        require(path == DEFAULT.resolve(), "Production study path differs"); admit()
    registry = read(REGISTRY); require(registry["sha256"] == config.digest({k: v for k, v in registry.items() if k != "sha256"}), "Study registry checksum differs")
    native = Native(); identity = {"schema": SCHEMA, "settings": settings, "fixture": fixture, "rules_digest": config.RULES_DIGEST,
                                  "fixed_model_sha256": sha(MODEL), "fixed_model_weights_sha256": Model(MODEL).id,
                                  "source_sha256": sha(__file__), "tool_sha256": sha(ROOT / "tools/natural-study.py"),
                                  "registry_sha256": sha(REGISTRY), "baseline_sha256": sha(BASELINE), "native_build_sha256": native.build_id,
                                  "plan_sha256": None if fixture else sha(PLAN), "implementation_sha256": None if fixture else sha(IMPLEMENTATION),
                                  "reserve_bytes": 2*1024**3, "quota_bytes": 50*1024**3}
    require(identity["fixed_model_sha256"] == MODEL_SHA and "torch" not in sys.modules, "Fixed CPU-only model/runtime differs")
    work = teacher.Work(path, identity); started = time.monotonic()
    def event(name, **values):
        if on_event:
            on_event({"event": name, **values})
    try:
        work.db.executescript("CREATE TABLE IF NOT EXISTS families(idx INTEGER PRIMARY KEY,body TEXT,hash TEXT);" + "".join("CREATE TABLE IF NOT EXISTS " + name + "(state TEXT PRIMARY KEY,body TEXT,hash TEXT);" for name in ("h8_attempts", "h8_labels", "h10_attempts", "h10_labels")))
        pool = table(work.db, "families", "idx"); require(list(pool) == list(range(len(pool))) and len(pool) <= settings["families"], "Study family prefix differs")
        completed = work.get("complete")
        with CpuBatchEvaluator(MODEL, cache_capacity=0, max_batch=1) as evaluator:
            for index, value in pool.items():
                require(value == prior.generate_family(index, settings, native, evaluator), "Study literal family replay differs")
            if completed is None:
                for index in range(len(pool), settings["families"]):
                    if stopped(cancel):
                        break
                    value = prior.generate_family(index, settings, native, evaluator)
                    event("natural_study_family_precommit", index=index, completed=len(pool), families=settings["families"])
                    with work.db:
                        work.db.execute("INSERT INTO families VALUES(?,?,?)", (index, config.canonical(value), config.digest(value)))
                    pool[index] = value; event("natural_study_family_committed", index=index, completed=len(pool), families=settings["families"])
        rows = table(work.db, "roots"); stores = {name: table(work.db, name) for name in ("h8_attempts", "h8_labels", "h10_attempts", "h10_labels")}; selection = None
        if len(pool) == settings["families"]:
            expected, inherited, selection = select(pool, settings, registry)
            require(not rows or rows == expected, "Study root ownership selection differs")
            if not rows:
                with work.db:
                    for key, row in expected.items():
                        work.db.execute("INSERT INTO roots VALUES(?,?,?)", (key, config.canonical(row), config.digest(row)))
                    for key, label in inherited.items():
                        attempt = attempt_for(label, expected[key], "h8", inherited=True)
                        work.db.execute("INSERT INTO h8_labels VALUES(?,?,?)", (key, config.canonical(label), config.digest(label)))
                        work.db.execute("INSERT INTO h8_attempts VALUES(?,?,?)", (key, config.canonical(attempt), config.digest(attempt)))
                        stores["h8_labels"][key] = label; stores["h8_attempts"][key] = attempt
                    work.put("selection", selection)
                rows = expected; event("natural_study_selected", **selection)
            require(work.get("selection") == selection, "Study selection marker differs")
            for target in ("h8", "h10"):
                validate_attempts(rows, stores[target+"_attempts"], stores[target+"_labels"], target, settings, registry, native)
            require(not stores["h10_attempts"] or len(stores["h8_attempts"]) == len(rows), "H10 exists before all H8 attempts")
        else:
            require(not rows and not any(stores.values()), "Study labels precede full family pool")
        if completed is None and len(pool) == settings["families"] and stage != "collect":
            for target in ("h8", "h10"):
                if target == "h10" and stage == "h8":
                    break
                attempts, labels = stores[target+"_attempts"], stores[target+"_labels"]
                for key in sorted(set(rows) - set(attempts)):
                    if stopped(cancel):
                        break
                    item, attempt = execute_target(rows[key], target, settings, native, cancel)
                    if attempt is None:
                        break
                    target_count = len(rows) if target == "h10" else sum(r["study_role"] == "fresh" for r in rows.values())
                    attempt_count = len(attempts) if target == "h10" else sum(rows[k]["study_role"] == "fresh" for k in attempts)
                    label_count = len(labels) if target == "h10" else sum(rows[k]["study_role"] == "fresh" for k in labels)
                    event("natural_study_"+target+"_precommit", state_key=key, completed=attempt_count, labels=label_count, target=target_count)
                    work.admit(1024**2)
                    with work.db:
                        work.db.execute("INSERT INTO "+target+"_attempts VALUES(?,?,?)", (key, config.canonical(attempt), config.digest(attempt)))
                        if item is not None:
                            work.db.execute("INSERT INTO "+target+"_labels VALUES(?,?,?)", (key, config.canonical(item), config.digest(item)))
                    attempts[key] = attempt
                    if item is not None:
                        labels[key] = item
                    attempt_count = len(attempts) if target == "h10" else sum(rows[k]["study_role"] == "fresh" for k in attempts)
                    label_count = len(labels) if target == "h10" else sum(rows[k]["study_role"] == "fresh" for k in labels)
                    event("natural_study_"+target+"_committed", state_key=key, completed=attempt_count, labels=label_count, target=target_count, rejected=sum(a["status"] == "budget_rejected" for a in attempts.values()))
                if len(attempts) != len(rows):
                    break
        a8, h8, a10, h10 = (stores[name] for name in ("h8_attempts", "h8_labels", "h10_attempts", "h10_labels"))
        if completed is None and len(pool) == settings["families"] and len(a8) == len(a10) == len(rows):
            completed = {"schema": SCHEMA, "identity": identity, "status": "complete", "families": len(pool), **counts(rows, h8, h10, a8, a10),
                         "selection": selection, "table_sha256": {"families": config.digest(pool), "roots": config.digest(rows), **{name: config.digest(value) for name, value in stores.items()}},
                         "training_performed": False, "torch_imported": False, "sealed_final_test_consumed": False, "elapsed_seconds_initial_run": time.monotonic()-started}
            validate_complete_marker(completed, identity, pool, rows, stores, selection)
            with work.db:
                work.put("complete", completed)
        if completed is not None:
            validate_complete_marker(completed, identity, pool, rows, stores, selection)
            publish(path / "manifest.json", completed)
            if not fixture:
                publish(OUTPUT, completed)
            load_rows(path)
            event("natural_study_complete", **counts(rows, h8, h10, a8, a10), families=len(pool)); return completed
        return {"status": "collected" if len(pool) == settings["families"] else "interrupted", "families": len(pool), **counts(rows, h8, h10, a8, a10), "selection": selection}
    finally:
        work.close()


def load_rows(path=DEFAULT):
    path = Path(path).resolve()
    with closing(sqlite3.connect((path / "work.sqlite3").as_uri()+"?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        def meta(key):
            row = db.execute("SELECT body,hash FROM meta WHERE key=?", (key,)).fetchone()
            require(row is not None, "Study missing committed " + key)
            value = json.loads(row[0]); require(config.digest(value) == row[1], "Corrupt study metadata " + key); return value
        manifest, identity = meta("complete"), meta("identity")
        require(set(identity) == IDENTITY_FIELDS and identity["schema"] == SCHEMA and identity["reserve_bytes"] == 2*1024**3 and identity["quota_bytes"] == 50*1024**3, "Closed study identity fields/schema/reserve differ")
        require(set(manifest) == MANIFEST_FIELDS and manifest["schema"] == SCHEMA and manifest["identity"] == identity and manifest["status"] == "complete", "Closed study fields/identity differ")
        require(identity["source_sha256"] == sha(__file__) and identity["tool_sha256"] == sha(ROOT/"tools/natural-study.py") and identity["fixed_model_sha256"] == sha(MODEL) == MODEL_SHA and identity["fixed_model_weights_sha256"] == Model(MODEL).id and identity["rules_digest"] == config.RULES_DIGEST and identity["registry_sha256"] == sha(REGISTRY) and identity["baseline_sha256"] == sha(BASELINE), "Closed study source/model/protection differs")
        settings = identity["settings"]
        if identity["fixture"] is False:
            admit(); require(settings == CONTRACT and identity["plan_sha256"] == sha(PLAN) and identity["implementation_sha256"] == sha(IMPLEMENTATION), "Closed production plan differs")
        else:
            require(identity["fixture"] is True and path.is_relative_to(ROOT/"build") and settings == settings_for(True, settings["families"], settings["followups"]) and identity["plan_sha256"] is None and identity["implementation_sha256"] is None, "Closed fixture settings differ")
        native = Native(); require(identity["native_build_sha256"] == native.build_id, "Closed teacher native build differs")
        registry = read(REGISTRY); require(registry["sha256"] == config.digest({k:v for k,v in registry.items() if k != "sha256"}), "Closed registry checksum differs")
        pool = table(db,"families","idx"); require(list(pool) == list(range(settings["families"])), "Closed family pool incomplete")
        with CpuBatchEvaluator(MODEL, cache_capacity=0, max_batch=1) as evaluator:
            for index, family in pool.items():
                require(family == prior.generate_family(index, settings, native, evaluator), "Closed literal study history differs")
        rows = table(db,"roots"); expected, _, selection = select(pool,settings,registry)
        require(rows == expected and meta("selection") == selection == manifest["selection"], "Closed study ownership/selection differs")
        stores = {name: table(db,name) for name in ("h8_attempts","h8_labels","h10_attempts","h10_labels")}
        for target in ("h8","h10"):
            require(len(stores[target+"_attempts"]) == len(rows), "Closed teacher target coverage incomplete")
            validate_attempts(rows,stores[target+"_attempts"],stores[target+"_labels"],target,settings,registry,native)
        validate_complete_marker(manifest,identity,pool,rows,stores,selection)
        require((path/"manifest.json").read_text(encoding="utf-8") == config.canonical(manifest)+"\n", "Closed file/commit publication differs")
        return rows, stores["h8_labels"], stores["h10_labels"], stores["h8_attempts"], stores["h10_attempts"], manifest
