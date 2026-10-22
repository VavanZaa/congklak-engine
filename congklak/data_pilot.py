"""Bounded ordinary-root collection and full-H8 labeling; no optimizer execution."""
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import time
from . import config, pilot, deeper_teacher_v2 as teacher, reports
from .native import ROOT, Native, Control, flatten
from .puct_cpu_inference import CpuBatchEvaluator
from .neural_relay import Model

SCHEMA = "ordinary-history-H8-data-pilot-v1"
DEFAULT = ROOT / "data/ordinary-data-pilot-v1"
MODEL = ROOT / "data/quality-train-v1/model.json"
MODEL_SHA = "bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558"
BASELINE = ROOT / "build/data-pilot-baseline.json"
REGISTRY = ROOT / "build/data-pilot-registry.json"
PLAN = ROOT / "build/data-pilot-plan.json"
IMPLEMENTATION = ROOT / "build/data-pilot-implementation.json"
OUTPUT = ROOT / "build/data-pilot-result.json"
SOURCES = ("congklak/data_pilot.py", "tools/data-pilot.py", "tests/data_pilot_author.py", "docs/DATA_PILOT_SCOPE.md")
CONTRACT = {"families": 256, "seed": 20261020, "max_choices": 192, "random_opening_choices": 8,
            "sources": "even-index random; odd-index frozen-policy epsilon0.25", "epsilon": .25,
            "ownership": "sha256(data-pilot-owner-v1/+family) first8bytes mod100: train<80, validation>=80",
            "mechanism": "pilot._mechanism across every legal root action; strict /ordinary",
            "caps": {"training_candidate": {"25-49": 32, "50-74": 32, "75-98": 8},
                     "validation": {"25-49": 8, "50-74": 8, "75-98": 2}},
            "diagnostic_limit": 16, "choice_depth": 8, "node_budget": 2000000, "time_ms": 0,
            "tt_mb": 16, "memory_mb": 512, "use_pvs": True,
            "training_performed": False, "sealed_final_test_consumed": False,
            "ownership_scope": "selected ordinary-root quotient bridge; shared history prefixes possible",
            "selection": "whole pool before closure; cross-cohort candidate quotient collisions excluded from both; one root/family/band"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stopped(cancel):
    return bool(cancel() if callable(cancel) else cancel.is_set()) if cancel is not None else False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def publish(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, "Immutable publication differs: " + str(path))
    else:
        teacher.atomic_json(path, value)


def table(db, name, key="state"):
    values = {}
    for item, body, checksum in db.execute("SELECT " + key + ",body,hash FROM " + name + " ORDER BY " + key):
        value = json.loads(body)
        require(config.digest(value) == checksum, "Corrupt journal row: " + name)
        values[item] = value
    return values


def checked_baseline():
    value = read(BASELINE)
    require(value["passed"] is True and value["inputs_count"] == len(value["inputs"]) and value["fixed_model_sha256"] == sha(MODEL) == MODEL_SHA,
            "Baseline/model identity differs")
    for name, checksum in value["inputs"].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == checksum, "Prior accepted input changed: " + name)
    return value


def candidate(p, native, family, cohort, provenance):
    key, state, swapped = pilot.canonicalize(p, native)
    active, stratum = pilot._mechanism(state, native)
    if active < 25 or not stratum.endswith("/ordinary"):
        return None
    normal = native.normalize(flatten(state), actor=state["controller"], total=98)
    terminal_children = sum(native.apply(flatten(state), actor=state["controller"], action=a, total=98, trace=False)["terminal"] for a in normal["legal_actions"])
    return {"state_key": key.hex(), "state": state, "quotient": pilot.quotient(state).hex(), "family": family,
            "cohort": cohort, "purpose": "protected_diagnostic" if cohort == "diagnostic" else "fresh_data_pilot",
            "band": stratum.split("/")[0], "stratum": stratum, "active": active,
            "legal_mask": sum(1 << a for a in normal["legal_actions"]), "terminal_children": terminal_children,
            "provenance": {**provenance, "raw_state": p, "swapped": swapped}}


def make_registry():
    """Metadata roots and literal trajectories only; no sealed target access."""
    baseline = checked_baseline()
    old = read(ROOT / "build/value-registry.json")
    quotients, families = set(old["quotients"]), set(old["families"])
    root_keys, root_meta = set(), []
    with closing(sqlite3.connect((ROOT / "data/deeper-9000-v2/work.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        for key, state, family in db.execute("SELECT state,json_extract(body,'$.state'),json_extract(body,'$.family') FROM roots ORDER BY state"):
            p = config.position(json.loads(state)); root_keys.add(key); families.add(family); quotients.add(pilot.quotient(p).hex())
            root_meta.append([key, p, family])
    require(len(root_keys) == 9000, "Existing root metadata count differs")
    root_q = {pilot.quotient(row[1]).hex() for row in root_meta}
    native = Native(); diagnostics = {}; projection = []; game_count = 0
    with closing(sqlite3.connect((ROOT / "data/value-benchmark-v1/work.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        for index, body, checksum in db.execute("SELECT idx,body,hash FROM games ORDER BY idx"):
            game = json.loads(body); require(config.digest(game) == checksum, "Old value arena journal corrupt")
            families.add(game["family"]); p = game["start"]; game_count += 1
            projection.append([index, game["family"], game["start"], game["moves"]])
            for step, action in enumerate(game["moves"]):
                p = native.normalize(flatten(p), actor=p["controller"], total=98)["position"]
                quotients.add(pilot.quotient(p).hex())
                if game["comparison"] in ("policy_control_vs_H_v1", "policy_control_vs_H8_relay_CPU") and game["decisions"][step]["engine"] == "policy_control":
                    provenance = {"source": "protected_control_arena", "family_index": None, "step": step,
                                  "start": game["start"], "moves_prefix": game["moves"][:step],
                                  "source_game_idx": index, "source_comparison": game["comparison"], "saved_puct_chosen": action}
                    item = candidate(p, native, game["family"], "diagnostic", provenance)
                    if item and item["state_key"] not in root_keys and item["quotient"] not in root_q:
                        diagnostics.setdefault(item["state_key"], item)
                transition = native.apply(flatten(p), actor=p["controller"], action=action, total=98, trace=False)
                require(transition.get("ok"), "Old arena contains illegal move")
                p = transition["position"]
            quotients.add(pilot.quotient(p).hex())
    require(game_count == 192, "Value arena protection must cover all192 games")
    unique, seen = [], set()
    for item in (diagnostics[key] for key in sorted(diagnostics)):
        if item["quotient"] not in seen:
            unique.append(item); seen.add(item["quotient"])
    value = {"schema": "ordinary-pilot-root-protection-v1", "rules_digest": config.RULES_DIGEST,
             "quotients": sorted(quotients), "families": sorted(families), "existing9000_state_keys": sorted(root_keys),
             "diagnostic_pool": unique, "protected_value_games": game_count,
             "old_registry_sha256": sha(ROOT / "build/value-registry.json"), "root_metadata_sha256": config.digest(root_meta),
             "arena_projection_sha256": config.digest(projection), "baseline_sha256": sha(BASELINE),
             "sealed_target_bodies_read": False, "training_performed": False}
    value["sha256"] = config.digest(value)
    publish(REGISTRY, value)
    return value


def plan():
    registry = make_registry()
    value = {"schema": SCHEMA, "contract": CONTRACT, "rules_digest": config.RULES_DIGEST,
             "fixed_model_sha256": MODEL_SHA, "baseline_sha256": sha(BASELINE), "registry_sha256": sha(REGISTRY),
             "diagnostic_roots": registry["diagnostic_pool"][:CONTRACT["diagnostic_limit"]],
             "no_independent_unseen_claim": True, "shared_history_prefixes_possible": True}
    publish(PLAN, value)
    return value


def freeze():
    value = plan(); baseline = checked_baseline()
    manifest = {"schema": "ordinary-pilot-implementation-v1", "plan_sha256": sha(PLAN),
                "registry_sha256": sha(REGISTRY), "sources": {name: sha(ROOT / name) for name in SOURCES},
                "prior_inputs": baseline["inputs"], "baseline_sha256": sha(BASELINE)}
    publish(IMPLEMENTATION, manifest)
    return manifest


def admit():
    value = read(PLAN); manifest = read(IMPLEMENTATION); checked_baseline()
    require(value["contract"] == CONTRACT and value["rules_digest"] == config.RULES_DIGEST and value["fixed_model_sha256"] == MODEL_SHA, "Frozen pilot plan differs")
    require(manifest["schema"] == "ordinary-pilot-implementation-v1" and manifest["plan_sha256"] == sha(PLAN) and manifest["registry_sha256"] == sha(REGISTRY) and manifest["baseline_sha256"] == sha(BASELINE), "Pilot freeze witness differs")
    require(manifest["sources"] == {name: sha(ROOT / name) for name in SOURCES} and manifest["prior_inputs"] == read(BASELINE)["inputs"], "Frozen pilot producer/prior source changed")
    return value, manifest


def generate_family(index, settings, native, evaluator):
    family = "ordinary-history-pilot/" + str(settings["seed"]) + "/" + str(index)
    rank = int.from_bytes(hashlib.sha256(("data-pilot-owner-v1/" + family).encode()).digest()[:8], "big") % 100
    cohort = "training_candidate" if rank < 80 else "validation"
    source = "random_history" if index % 2 == 0 else "frozen_policy_epsilon_history"
    rng = random.Random(settings["seed"] + index * 104729); p = config.position(); moves = []; candidates = []; forwards = 0
    for step in range(settings["max_choices"]):
        normal = native.normalize(flatten(p), actor=p["controller"], total=98)
        p = normal["position"]
        if normal["terminal"] or sum(map(sum, p["pits"])) < 25:
            break
        provenance = {"source": source, "family_index": index, "step": step, "start": config.position(), "moves_prefix": list(moves),
                      "source_game_idx": None, "source_comparison": None, "saved_puct_chosen": None}
        item = candidate(p, native, family, cohort, provenance)
        if item is not None:
            candidates.append(item)
        legal = normal["legal_actions"]
        if index % 2 == 0 or step < settings["random_opening_choices"] or rng.random() < settings["epsilon"]:
            action = rng.choice(legal)
        else:
            prediction = evaluator.evaluate_many([p])[0]; forwards += 1
            action = max(legal, key=lambda a: (prediction["policy_logits"][a], -a))
        transition = native.apply(flatten(p), actor=p["controller"], action=action, total=98, trace=False)
        require(transition.get("ok"), "Fresh history move is illegal")
        moves.append(action); p = transition["position"]
    return {"index": index, "family": family, "cohort": cohort, "source": source,
            "rng_seed": settings["seed"] + index * 104729, "start": config.position(), "moves": moves,
            "final": p, "choices": len(moves), "candidates": candidates, "policy_evaluations": forwards,
            "stop_reason": "terminal" if not sum(map(sum, p["pits"])) else "below25" if sum(map(sum, p["pits"])) < 25 else "choice_cap"}


def select(families, settings, registry):
    observations = [row for family in families.values() for row in family["candidates"]]
    cohorts = defaultdict(set)
    for row in observations:
        cohorts[row["quotient"]].add(row["cohort"])
    collisions = {key for key, owners in cohorts.items() if len(owners) > 1}
    protected = set(registry["quotients"]); oldfamilies = set(registry["families"])
    available = [r for r in observations if r["quotient"] not in protected | collisions and r["family"] not in oldfamilies]
    selected = {}; used_q = set(); used_family_band = set(); counts = Counter()
    for row in sorted(available, key=lambda r: (r["provenance"]["family_index"], r["provenance"]["step"], r["state_key"])):
        cohort, band = row["cohort"], row["band"]
        if counts[cohort, band] >= settings["caps"][cohort][band] or row["quotient"] in used_q or (row["family"], band) in used_family_band:
            continue
        selected[row["state_key"]] = row; used_q.add(row["quotient"]); used_family_band.add((row["family"], band)); counts[cohort, band] += 1
    for row in registry["diagnostic_pool"][:settings["diagnostic_limit"]]:
        require(row["state_key"] not in selected and row["quotient"] in protected, "Diagnostic ownership bridge differs")
        selected[row["state_key"]] = row
    detail = {"ordinary_observations": len(observations), "ordinary_unique_quotients": len(cohorts),
              "cross_cohort_candidate_collisions": len(collisions), "protected_candidate_observations": sum(r["quotient"] in protected for r in observations),
              "counts": {c: {b: counts[c, b] for b in settings["caps"][c]} for c in settings["caps"]},
              "shortfall": {c: {b: settings["caps"][c][b] - counts[c, b] for b in settings["caps"][c]} for c in settings["caps"]},
              "diagnostic_rows": sum(r["cohort"] == "diagnostic" for r in selected.values()),
              "observed_legal_action_counts": dict(sorted(Counter(str(r["legal_mask"].bit_count()) for r in observations).items())),
              "selected_legal_action_counts": dict(sorted(Counter(str(r["legal_mask"].bit_count()) for r in selected.values()).items())),
              "diagnostic_shortfall": settings["diagnostic_limit"] - min(settings["diagnostic_limit"], len(registry["diagnostic_pool"]))}
    return dict(sorted(selected.items())), detail


def validate_label_item(item, row, native):
    require(set(item) == {"root_sha256", "label", "report"} and item["root_sha256"] == config.digest(row), "Label root/fields differ")
    teacher.validate_label(item["label"], row["state"], row["legal_mask"], 8, native)
    teacher.validate_link(item, row, native)


def validate_attempts(roots, attempts, labels, settings, native):
    order = sorted(roots)
    require(list(attempts) == order[:len(attempts)] and set(labels) <= set(attempts), "Label attempt journal is not a declared root prefix")
    for key, attempt in attempts.items():
        require(set(attempt) == {"root_sha256", "status", "stop_reason", "completed_depth", "label_sha256", "work", "partial_report"}, "Attempt fields differ")
        require(attempt["root_sha256"] == config.digest(roots[key]) and attempt["status"] in ("complete", "budget_rejected"), "Attempt root/status differs")
        require(type(attempt["completed_depth"]) is int and 0 <= attempt["completed_depth"] <= 8 and set(attempt["work"]) == {"nodes", "deposits", "elapsed_ms", "peak_process_bytes"} and all(type(v) is int and v >= 0 for v in attempt["work"].values()), "Attempt work/depth differs")
        if attempt["status"] == "complete":
            require(key in labels and attempt["label_sha256"] == config.digest(labels[key]) and attempt["completed_depth"] == 8 and attempt["stop_reason"] == "max_depth" and attempt["partial_report"] is None, "Complete attempt/label differs")
            validate_label_item(labels[key], roots[key], native)
            require(0 < attempt["work"]["nodes"] <= settings["node_budget"] and all(attempt["work"][name] >= labels[key]["report"][name] for name in attempt["work"]), "Complete attempt physical work is below its report or over budget")
        else:
            require(key not in labels and attempt["stop_reason"] == "node_budget" and attempt["label_sha256"] is None and attempt["completed_depth"] < 8 and attempt["work"]["nodes"] == settings["node_budget"], "Partial label was admitted")
            partial = attempt["partial_report"]
            require((partial is None) == (attempt["completed_depth"] == 0), "Partial report/depth ownership differs")
            if partial is not None:
                reports.validate(partial, roots[key]["state"], config.identity(native.build_id), native)
                require(partial["choice_depth"] == attempt["completed_depth"] and all(partial[name] <= attempt["work"][name] for name in attempt["work"]), "Partial report/work coverage differs")


def run(path=DEFAULT, stage="all", fixture=False, families=None, diagnostic_limit=None, on_event=None, cancel=None, output=None):
    require(stage in ("collect", "label", "all"), "Unknown pilot stage")
    path = Path(path).resolve(); settings = json.loads(config.canonical(CONTRACT))
    if fixture:
        require(path.is_relative_to(ROOT / "build"), "Fixture must be inside build")
        settings["families"] = 8 if families is None else config.integer(families, "fixture_families", 1, 32)
        settings["diagnostic_limit"] = 2 if diagnostic_limit is None else config.integer(diagnostic_limit, "fixture_diagnostics", 0, 16)
        settings["caps"] = {c: {b: 1 for b in bands} for c, bands in settings["caps"].items()}
        registry = read(REGISTRY); plan_sha = implementation_sha = None
    else:
        require(path == DEFAULT.resolve() and families is None and diagnostic_limit is None, "Production path/options must equal fixed contract")
        admit(); registry = read(REGISTRY); plan_sha = sha(PLAN); implementation_sha = sha(IMPLEMENTATION)
    require(registry["sha256"] == config.digest({k: v for k, v in registry.items() if k != "sha256"}), "Registry checksum differs")
    native = Native()
    identity = {"schema": SCHEMA, "settings": settings, "fixture": fixture, "rules_digest": config.RULES_DIGEST,
                "model_file_sha256": sha(MODEL), "model_weights_sha256": Model(MODEL).id,
                "source_sha256": sha(__file__), "tool_sha256": sha(ROOT / "tools/data-pilot.py"),
                "native_build_sha256": native.build_id, "registry_sha256": sha(REGISTRY),
                "baseline_sha256": sha(BASELINE), "plan_sha256": plan_sha, "implementation_sha256": implementation_sha,
                "reserve_bytes": 2 * 1024**3, "quota_bytes": 50 * 1024**3}
    require(identity["model_file_sha256"] == MODEL_SHA, "Fixed policy model changed")
    work = teacher.Work(path, identity); started = time.monotonic()
    def event(name, **values):
        if on_event:
            on_event({"event": name, **values})
    try:
        work.db.executescript("CREATE TABLE IF NOT EXISTS families(idx INTEGER PRIMARY KEY,body TEXT,hash TEXT);CREATE TABLE IF NOT EXISTS attempts(state TEXT PRIMARY KEY,body TEXT,hash TEXT);")
        saved = table(work.db, "families", "idx")
        require(list(saved) == list(range(len(saved))) and len(saved) <= settings["families"], "Collection journal is not a declared prefix")
        with CpuBatchEvaluator(MODEL, cache_capacity=0, max_batch=1) as evaluator:
            for index, value in saved.items():
                require(value == generate_family(index, settings, native, evaluator), "Saved literal history/generator differs")
            completed = work.get("complete")
            if completed is None:
                for index in range(len(saved), settings["families"]):
                    if stopped(cancel):
                        break
                    value = generate_family(index, settings, native, evaluator)
                    event("data_pilot_family_precommit", index=index, completed=len(saved), families=settings["families"])
                    with work.db:
                        work.db.execute("INSERT INTO families VALUES(?,?,?)", (index, config.canonical(value), config.digest(value)))
                    saved[index] = value
                    event("data_pilot_family_committed", index=index, completed=len(saved), families=settings["families"])
        roots, attempts, labels = table(work.db, "roots"), table(work.db, "attempts"), table(work.db, "labels")
        selection = None
        if len(saved) == settings["families"]:
            expected, selection = select(saved, settings, registry)
            require(not roots or roots == expected, "Saved root ownership/quota selection differs")
            if not roots:
                with work.db:
                    for key, value in expected.items():
                        work.db.execute("INSERT INTO roots VALUES(?,?,?)", (key, config.canonical(value), config.digest(value)))
                    work.put("selection", selection)
                roots = expected
                event("data_pilot_selected", **selection)
            require(work.get("selection") == selection, "Selection metadata differs")
        else:
            require(not roots and not attempts and not labels, "Labels/roots exist before full collection pool")
        order = sorted(roots)
        validate_attempts(roots, attempts, labels, settings, native)
        if stage != "collect" and completed is None and len(saved) == settings["families"]:
            for key in order[len(attempts):]:
                if stopped(cancel):
                    break
                row = roots[key]; p = row["state"]
                with Control(native, max_depth=8, time_ms=0, node_budget=settings["node_budget"], memory_mb=512, tt_mb=16, use_pvs=True) as control:
                    if cancel is not None and hasattr(cancel, "_register"):
                        cancel._register(control)
                    try:
                        result = native.analyze(flatten(p), actor=p["controller"], total=98, max_depth=8, time_ms=0,
                                                node_budget=settings["node_budget"], memory_mb=512, tt_mb=16, use_pvs=True,
                                                keep_history=False, control=control)
                    finally:
                        if cancel is not None and hasattr(cancel, "_remove"):
                            cancel._remove(control)
                if stopped(cancel):
                    break
                report = result.get("report"); complete = result["stop_reason"] == "max_depth" and report is not None and report["choice_depth"] == 8
                item = None
                if complete:
                    reports.validate(report, p, config.identity(native.build_id), native)
                    label = {"kind": "finite_choice_horizon", "choice_depth": 8, "oracle_manifest_sha256": None,
                             "target_player0": report["depth_result"]["value_player0"], "best_moves": report["best_moves"],
                             "action_scores": [{"pit": c["pit"], "value_player0": c["depth_result"]["value_player0"]} for c in report["candidates"]],
                             "coverage": report["coverage"], "work": {k: report[k] for k in ("nodes", "deposits", "elapsed_ms", "peak_process_bytes")}}
                    item = {"root_sha256": config.digest(row), "label": label, "report": report}; validate_label_item(item, row, native)
                else:
                    require(result["stop_reason"] == "node_budget", "Unexpected teacher failure is not a budget shortfall")
                attempt = {"root_sha256": config.digest(row), "status": "complete" if complete else "budget_rejected",
                           "stop_reason": result["stop_reason"], "completed_depth": report["choice_depth"] if report else 0,
                           "work": {name: result[name] for name in ("nodes", "deposits", "elapsed_ms", "peak_process_bytes")},
                           "partial_report": None if complete else report,
                           "label_sha256": config.digest(item) if item else None}
                event("data_pilot_label_precommit", state_key=key, completed=len(attempts), labels=len(labels), target=len(roots))
                work.admit(1024**2)
                with work.db:
                    work.db.execute("INSERT INTO attempts VALUES(?,?,?)", (key, config.canonical(attempt), config.digest(attempt)))
                    if item:
                        work.db.execute("INSERT INTO labels VALUES(?,?,?)", (key, config.canonical(item), config.digest(item)))
                attempts[key] = attempt
                if item:
                    labels[key] = item
                event("data_pilot_label_committed", state_key=key, completed=len(attempts), labels=len(labels), rejected=len(attempts)-len(labels), target=len(roots))
        if completed is not None:
            require(completed["identity"] == identity and completed["family_semantic_sha256"] == config.digest(saved) and completed["roots_semantic_sha256"] == config.digest(roots) and completed["attempts_sha256"] == config.digest(attempts) and completed["labels_sha256"] == config.digest(labels), "Completed pilot publication differs")
            publish(path / "manifest.json", completed)
            if not fixture:
                publish(OUTPUT if output is None else output, completed)
            load_rows(path)
            event("data_pilot_complete", families=len(saved), roots=len(roots), labels=len(labels), files_rewritten=False)
            return completed
        if len(saved) == settings["families"] and len(attempts) == len(roots) and stage != "collect":
            counts = Counter(roots[key]["cohort"] + "/" + roots[key]["band"] for key in labels)
            completed = {"schema": SCHEMA, "identity": identity, "status": "complete", "families": len(saved),
                         "selected_roots": len(roots), "labels": len(labels), "budget_rejected": len(attempts)-len(labels),
                         "labeled_counts": dict(sorted(counts.items())), "selection": selection,
                         "family_semantic_sha256": config.digest(saved), "roots_semantic_sha256": config.digest(roots),
                         "attempts_sha256": config.digest(attempts), "labels_sha256": config.digest(labels),
                         "teacher_scope": "Approximate completed full-root H8; H7 successor action values; no exact-game claims",
                         "training_performed": False, "torch_imported": "torch" in __import__("sys").modules,
                         "sealed_final_test_consumed": False, "elapsed_seconds_initial_run": time.monotonic()-started}
            with work.db:
                work.put("complete", completed)
            publish(path / "manifest.json", completed)
            if not fixture:
                publish(OUTPUT if output is None else output, completed)
            load_rows(path)
            event("data_pilot_complete", families=len(saved), roots=len(roots), labels=len(labels), rejected=len(attempts)-len(labels))
            return completed
        return {"status": "collected" if len(saved) == settings["families"] else "interrupted", "families": len(saved), "roots": len(roots), "labels": len(labels), "attempts": len(attempts), "selection": selection}
    finally:
        work.close()


def load_rows(path=DEFAULT):
    path = Path(path).resolve()
    with closing(sqlite3.connect((path / "work.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        row = db.execute("SELECT body,hash FROM meta WHERE key='complete'").fetchone()
        require(row is not None, "Pilot has no committed complete publication")
        manifest = json.loads(row[0]); require(config.digest(manifest) == row[1], "Corrupt complete marker")
        require(set(manifest) == {"schema", "identity", "status", "families", "selected_roots", "labels", "budget_rejected", "labeled_counts", "selection", "family_semantic_sha256", "roots_semantic_sha256", "attempts_sha256", "labels_sha256", "teacher_scope", "training_performed", "torch_imported", "sealed_final_test_consumed", "elapsed_seconds_initial_run"}, "Closed publication fields differ")
        require(manifest["schema"] == SCHEMA and manifest["training_performed"] is False and manifest["torch_imported"] is False and manifest["sealed_final_test_consumed"] is False and type(manifest["elapsed_seconds_initial_run"]) in (int, float) and math.isfinite(manifest["elapsed_seconds_initial_run"]) and manifest["elapsed_seconds_initial_run"] > 0, "Closed scope/resources differ")
        require(all(type(manifest[name]) is int for name in ("families", "selected_roots", "labels", "budget_rejected")), "Closed count types differ")
        identity_row = db.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone()
        identity = json.loads(identity_row[0]); require(config.digest(identity) == identity_row[1] and manifest["identity"] == identity, "Complete identity differs")
        native = Native(); settings = identity["settings"]; registry = read(REGISTRY)
        require(identity["schema"] == SCHEMA and identity["source_sha256"] == sha(__file__) and identity["tool_sha256"] == sha(ROOT / "tools/data-pilot.py") and identity["model_file_sha256"] == sha(MODEL) == MODEL_SHA and identity["model_weights_sha256"] == Model(MODEL).id and identity["rules_digest"] == config.RULES_DIGEST and identity["native_build_sha256"] == native.build_id and identity["registry_sha256"] == sha(REGISTRY) and identity["baseline_sha256"] == sha(BASELINE), "Closed pilot source/model/protection differs")
        if identity["fixture"] is False:
            admit(); require(settings == CONTRACT and identity["plan_sha256"] == sha(PLAN) and identity["implementation_sha256"] == sha(IMPLEMENTATION), "Closed production plan differs")
        else:
            require(identity["fixture"] is True and path.is_relative_to(ROOT / "build") and 1 <= settings["families"] <= 32 and 0 <= settings["diagnostic_limit"] <= 16, "Closed fixture options differ")
            expected_settings = json.loads(config.canonical(CONTRACT)); expected_settings["families"] = settings["families"]; expected_settings["diagnostic_limit"] = settings["diagnostic_limit"]
            expected_settings["caps"] = {c: {b: 1 for b in bands} for c, bands in expected_settings["caps"].items()}
            require(settings == expected_settings and identity["plan_sha256"] is None and identity["implementation_sha256"] is None, "Closed fixture protocol differs")
        families = table(db, "families", "idx"); require(list(families) == list(range(settings["families"])), "Closed family pool incomplete")
        with CpuBatchEvaluator(MODEL, cache_capacity=0, max_batch=1) as evaluator:
            for index, family in families.items():
                require(family == generate_family(index, settings, native, evaluator), "Closed literal history differs")
        roots, labels, attempts = table(db, "roots"), table(db, "labels"), table(db, "attempts")
        expected_roots, selection = select(families, settings, registry)
        require(roots == expected_roots, "Closed selection ownership differs")
        validate_attempts(roots, attempts, labels, settings, native)
        require(len(attempts) == len(roots), "Closed attempt coverage incomplete")
        selection_row = db.execute("SELECT body,hash FROM meta WHERE key='selection'").fetchone()
        require(selection_row is not None and json.loads(selection_row[0]) == selection and config.digest(selection) == selection_row[1] and manifest["selection"] == selection, "Closed selection marker differs")
        require(manifest["status"] == "complete" and manifest["families"] == len(families) and manifest["selected_roots"] == len(roots) and manifest["labels"] == len(labels) and manifest["budget_rejected"] == len(attempts)-len(labels), "Closed publication counters differ")
        require(manifest["labeled_counts"] == dict(sorted(Counter(roots[key]["cohort"] + "/" + roots[key]["band"] for key in labels).items())), "Closed labeled cohort counters differ")
        require(manifest["family_semantic_sha256"] == config.digest(families) and manifest["roots_semantic_sha256"] == config.digest(roots) and manifest["attempts_sha256"] == config.digest(attempts) and manifest["labels_sha256"] == config.digest(labels), "Closed table digests differ")
        require((path / "manifest.json").read_text(encoding="utf-8") == config.canonical(manifest) + "\n", "Closed file/commit publication differs")
        return roots, labels, attempts, manifest
