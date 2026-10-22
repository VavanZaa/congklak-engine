"""One bounded value-head refit with exact frozen-policy checkpoint recovery.

Corpus admission is CPU-only. H8 root margins are approximate finite-choice
targets. Action values enter descriptive metrics only; policy is frozen.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sqlite3
import struct
import time
from . import config, deeper_teacher_v2 as teacher, features_relay, pilot, quality_learning as ancestor
from .native import ROOT, flatten
from .neural_relay import Model, flatten_weights

SCHEMA = "frozen-policy-seed-space-value-head-refit-v1"
SOURCE = ROOT / "data/quality-train-v1/model.json"
SOURCE_SHA = "bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558"
SOURCE_WEIGHTS = "1efd42cf39e64183752ad3bcbf3ba1b82e7e5fdc4c273426c763d4e20d746649"
TRAINABLE = ("layers.4.weight", "layers.4.bias")
FROZEN_WEIGHTS = "e20ef839b5354ed3c384f68cd043b405b26c0d4b0d4f353ae7f008652710a82a"
DATA_SHA = "f2878aac7ef0bee1a1358e36d981656dfb242683c31e245205790beae393b6c4"
DEFAULT = ROOT / "data/value-train-v1"
REGISTRY = ROOT / "build/value-registry.json"
CONTRACT = {
    "steps": 1500, "batch_size": 256, "lr": .001, "seed": 20261018,
    "checkpoint_every": 50, "input_count": 127, "width": 64,
    "parameters": 12872, "trainable_parameters": 65, "frozen_parameters": 12807,
    "teacher_horizon": 8, "policy_temperature_seeds": 2.,
    "value_huber_delta": 4., "policy_loss_weight": 0.,
    "objective": "Huber(D_player0+signed_active*tanh_head,H8_root_player0_SEEDS,delta4)",
    "optimizer": "AdamW/train65only/fp32/weight_decay0/grad_clip1/foreachFalse",
    "sampler": "Uniform admitted teacher strata with replacement; unit row weights",
    "selection": "Fixed final optimizer step; development metrics descriptive only",
    "selfplay_rows": 0, "oracle_training": False, "sealed_final_test_consumed": False,
    "amp": False, "loader_workers": 0,
}
STATE_FIELDS = {"identity", "step", "draws", "model", "optimizer", "sampler_rng",
                "python_rng", "torch_rng", "cuda_rng", "scheduler", "amp_scaler", "pending_batches"}
PROMOTION = "Experimental value-head-only refit; fixed policy; independent equal-time gates required"
RESULT_FIELDS = {"schema", "status", "identity", "steps_completed", "selected_step", "draws", "weights_sha256",
                 "train_rows", "excluded_training_rows", "development", "initial_development_this_invocation", "recovery",
                 "elapsed_seconds_this_invocation", "peak_host_working_set_bytes", "peak_gpu_allocated_bytes",
                 "peak_gpu_reserved_bytes", "sealed_final_test_consumed", "promotion", "frozen_weights_sha256", "frozen_policy_unchanged"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    path = Path(path)
    if path.exists():
        require(json.loads(path.read_text(encoding="utf-8")) == value, "Existing immutable publication differs: " + str(path))
        return
    teacher.atomic_json(path, value)


def read_registry(path=REGISTRY):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(value.get("rules_digest") == config.RULES_DIGEST and value.get("sealed_target_bodies_read") is False,
            "Protected registry rules or metadata-only access differs")
    require(isinstance(value.get("quotients"), list) and value["quotients"] == sorted(set(value["quotients"])), "Invalid protected quotient list")
    require(isinstance(value.get("families"), list) and value["families"] == sorted(set(value["families"])), "Invalid protected family list")
    require(all(isinstance(x, str) and len(bytes.fromhex(x)) == 15 for x in value["quotients"]), "Invalid protected quotient encoding")
    require(all(isinstance(x, str) and x for x in value["families"]), "Invalid protected family encoding")
    if "sha256" in value:
        require(value["sha256"] == config.digest({k: v for k, v in value.items() if k != "sha256"}), "Protected registry checksum differs")
    return value


def target(row, temperature=2.):
    require(type(temperature) in (int, float) and math.isfinite(temperature) and temperature > 0, "Invalid policy temperature")
    p = config.position(row["state"])
    native = features_relay._native()
    teacher.validate_label(row["label"], p, row["legal_mask"], 8, native)
    normal = native.normalize(flatten(p), actor=p["controller"], total=98)
    require(not normal["terminal"] and normal["normalized_controller"] == p["controller"], "Training root must be normalized nonterminal")
    x = row["cached_features"]
    d, r = x["d"], x["r"]
    active = sum(map(sum, p["pits"]))
    sign = 1 if p["controller"] == 0 else -1
    require(type(d) is int and d == p["stores"][0] - p["stores"][1] and type(r) is int and r == sign * active and active > 4,
            "Value residual offset/scale differs")
    y = row["label"]["target_player0"]
    z = (y - d) / r
    require(math.isfinite(z) and -1 <= z <= 1 and abs(d + r * z - y) < 1e-12, "Unbounded/inconsistent signed-active root target")
    scores = row["label"]["action_scores"]
    optimum = max(sign * action["value_player0"] for action in scores)
    weights = [math.exp((sign * action["value_player0"] - optimum) / temperature) for action in scores]
    total = sum(weights)
    probability, legal, q = [0.] * 7, [False] * 7, [0.] * 7
    for action, weight in zip(scores, weights):
        pit = action["pit"]
        probability[pit], legal[pit], q[pit] = weight / total, True, sign * action["value_player0"]
    require([i for i, valid in enumerate(legal) if valid] == normal["legal_actions"] and abs(sum(probability) - 1) < 1e-12,
            "Teacher policy mass or legal coverage differs")
    return {"probability": probability, "legal": legal, "q_actor": q, "residual": z, "y_player0": y, "d": d, "r": r}


def prepare(registry_path=REGISTRY, output=None):
    """Return validated rows and a metadata-only admitted-set manifest, before Torch."""
    registry_path = Path(registry_path).resolve()
    registry = read_registry(registry_path)
    inherited = ancestor.witness(SOURCE.parent)
    require(inherited["model_file_sha256"] == SOURCE_SHA and inherited["model_weights_sha256"] == SOURCE_WEIGHTS, "Refined ancestor model witness differs")
    manifest, data, source_fingerprints = teacher.load(teacher_depth=8)
    require(manifest["semantic_sha256"] == DATA_SHA and len(data["train"]) == 8000 and len(data["development"]) == 1000, "Original H8 corpus changed")
    protected_q, protected_f = set(registry["quotients"]), set(registry["families"])
    quotient_families = {row["family"] for row in data["train"] if pilot.quotient(row["state"]).hex() in protected_q}
    admitted, excluded = [], []
    for row in data["train"]:
        reasons = []
        if pilot.quotient(row["state"]).hex() in protected_q:
            reasons.append("protected_active_quotient")
        if row["family"] in protected_f:
            reasons.append("protected_family")
        if row["family"] in quotient_families:
            reasons.append("protected_family_via_quotient")
        if reasons:
            excluded.append({"state_key": row["state_key"], "family": row["family"], "reasons": reasons})
        else:
            admitted.append(row)
    require(admitted, "Protected exclusions removed all training rows")
    ancestor_data_path = ROOT / "build/quality-data.json"
    ancestor_data = json.loads(ancestor_data_path.read_text(encoding="utf-8"))
    require(ancestor_data["sha256"] == config.digest({k: v for k, v in ancestor_data.items() if k != "sha256"}), "Ancestor admitted-set checksum differs")
    ancestor_keys = set(ancestor_data["train_state_keys"])
    require({r["state_key"] for r in admitted} <= ancestor_keys, "Value refit must not expand beyond ancestral admitted training roots")
    require(not ({r["family"] for r in admitted} & {r["family"] for r in data["development"]}), "Development family leaked into optimizer")
    require(not ({pilot.quotient(r["state"]) for r in admitted} & {pilot.quotient(r["state"]) for r in data["development"]}), "Development quotient leaked into optimizer")
    for rows in (admitted, data["development"]):
        for row in rows:
            target(row)
    fingerprints = {"train": config.digest([r["state_key"] for r in admitted]), "development": source_fingerprints["development"]}
    value = {"schema": "value-admitted-teacher-set-v1", "rules_digest": config.RULES_DIGEST,
             "source_dataset_semantic_sha256": DATA_SHA, "source_manifest_sha256": config.digest(manifest),
             "source_split_fingerprints": source_fingerprints, "split_fingerprints": fingerprints,
             "ancestor_admitted_manifest_file_sha256": sha(ancestor_data_path),
             "ancestor_model_file_sha256": SOURCE_SHA, "ancestor_model_weights_sha256": SOURCE_WEIGHTS,
             "ancestral_exposure": "Frozen shared trunk/policy retain ancestral 7669-root exposure; exclusions prevent new updates, not unlearning",
             "registry_path": str(registry_path.relative_to(ROOT)), "registry_file_sha256": sha(registry_path),
             "registry_semantic_sha256": config.digest(registry), "original_train_rows": 8000,
             "quotient_excluded_families": sorted(quotient_families),
             "train_rows": len(admitted), "development_rows": 1000, "excluded_rows": excluded,
             "train_state_keys": [r["state_key"] for r in admitted],
             "train_strata": dict(sorted(Counter(r["stratum"] for r in admitted).items())),
             "train_families": sorted({r["family"] for r in admitted}),
             "development_families": sorted({r["family"] for r in data["development"]}),
             "labels": "Approximate H8 root margins only; action targets retained solely for unchanged-policy descriptive metrics",
             "training_replay_rows": 0, "sealed_target_bodies_read": False, "oracle_training": False}
    value["sha256"] = config.digest(value)
    if output is not None:
        publish(output, value)
    return value, {"train": admitted, "development": data["development"]}


def runtime():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from . import training_v3 as prior
    return torch, prior


def evaluator():
    torch, _ = runtime()
    class Evaluator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(torch.nn.Linear(127, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                                             torch.nn.ReLU(), torch.nn.Linear(64, 1), torch.nn.Tanh())
            self.policy = torch.nn.Linear(64, 7)
        def both(self, x):
            h = self.layers[1](self.layers[0](x))
            h = self.layers[3](self.layers[2](h))
            return self.layers[5](self.layers[4](h)).squeeze(-1), self.policy(h)
    model = Evaluator()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in TRAINABLE)
    return model


def tensors(rows, device="cpu"):
    torch, _ = runtime()
    targets = [target(row) for row in rows]
    result = {name: torch.tensor([t[name] for t in targets], dtype=torch.bool if name == "legal" else torch.float32, device=device)
              for name in ("probability", "legal", "q_actor", "residual", "y_player0", "d", "r")}
    result["x"] = torch.tensor([r["cached_features"]["x"] for r in rows], dtype=torch.float32, device=device)
    return result


def policy_losses(logits, probability, legal):
    torch, _ = runtime()
    require(bool(legal.any(dim=1).all()), "All-masked policy target")
    require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(probability).all()) and bool((probability >= 0).all()), "Invalid policy tensors")
    require(bool((probability.masked_select(~legal) == 0).all()) and bool(((probability.sum(dim=1) - 1).abs() < 1e-6).all()), "Illegal/unscaled policy mass")
    logp = torch.nn.functional.log_softmax(logits.masked_fill(~legal, -math.inf), dim=1).masked_fill(~legal, 0.)
    return -(probability * logp).sum(dim=1)


def metrics(model, rows, data):
    torch, _ = runtime()
    model.eval()
    with torch.no_grad():
        residual, logits = model.both(data["x"])
        prediction = data["d"] + data["r"] * residual
        require(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(logits).all()), "Nonfinite descriptive metrics")
        error = (prediction - data["y_player0"]).abs().cpu().tolist()
        huber = torch.nn.functional.huber_loss(prediction, data["y_player0"], delta=4., reduction="none").cpu().tolist()
        residual_error = (residual - data["residual"]).abs().cpu().tolist()
        ce = policy_losses(logits, data["probability"], data["legal"]).cpu().tolist()
        chosen = logits.masked_fill(~data["legal"], -math.inf).argmax(dim=1)
        regret = (data["q_actor"].masked_fill(~data["legal"], -math.inf).max(dim=1).values - data["q_actor"].gather(1, chosen[:, None]).squeeze(1)).cpu().tolist()
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["stratum"]].append(i)
    strata = {name: {"rows": len(ids), "mae_seeds": sum(error[i] for i in ids) / len(ids),
                    "seed_huber_delta4": sum(huber[i] for i in ids) / len(ids),
                    "residual_mae": sum(residual_error[i] for i in ids) / len(ids),
                    "policy_cross_entropy": sum(ce[i] for i in ids) / len(ids),
                    "policy_H8_action_regret_seeds": sum(regret[i] for i in ids) / len(ids),
                    "policy_H8_optimal_set_accuracy": sum(regret[i] == 0 for i in ids) / len(ids)}
              for name, ids in sorted(groups.items())}
    return {"rows": len(rows), "strata": strata,
            **{"uniform_stratum_" + name: sum(s[name] for s in strata.values()) / len(strata)
               for name in ("mae_seeds", "seed_huber_delta4", "residual_mae", "policy_cross_entropy", "policy_H8_action_regret_seeds", "policy_H8_optimal_set_accuracy")},
            "scope": "Descriptive reused development H8 finite-choice targets; frozen policy; no checkpoint selection"}


def export_payload(state, identity, step):
    _, prior = runtime()
    return {"schema": "congklak-mlp-export-v4", "encoder": features_relay.ENCODER127,
            "identity": identity, "step": step, "status": "experimental_value_head_refit_candidate",
            "weights": {k: v.detach().cpu().tolist() for k, v in state.items()}, "weights_sha256": prior.weight_digest(state),
            "terminal": "store0-minus-store1", "output": "D_player0 + signed_active*tanh(current-actor-residual)", "uses_oracle": False,
            "policy_output": "7 current-root-actor logits; legal actions masked by rules; terminal policy empty",
            "policy_training": "Unchanged refined policy; six shared-trunk/policy tensors frozen bit-exact",
            "value_training": "Seed-space Huber delta4 on approximate H8 root finite-choice margins; only value-head65 trained"}


def frozen_digest(state):
    _, prior = runtime()
    return prior.weight_digest({name: value for name, value in state.items() if name not in TRAINABLE})


def model_frozen_digest(model):
    digest = hashlib.sha256()
    for name in sorted(set(model.weights) - set(TRAINABLE)):
        numbers = list(flatten_weights(model.weights[name]))
        digest.update(name.encode())
        digest.update(struct.pack("<" + "f" * len(numbers), *numbers))
    return digest.hexdigest()


def validate_state(state, identity, model):
    torch, prior = runtime()
    require(isinstance(state, dict) and set(state) == STATE_FIELDS and state["identity"] == identity, "Checkpoint state fields/identity differ")
    step = state["step"]
    require(type(step) is int and 0 < step <= identity["steps"] and type(state["draws"]) is int and state["draws"] == step * identity["batch_size"], "Checkpoint step/draw conservation differs")
    expected = model.state_dict()
    require(isinstance(state["model"], dict) and state["model"].keys() == expected.keys(), "Checkpoint parameter ownership differs")
    for name, value in state["model"].items():
        require(torch.is_tensor(value) and value.shape == expected[name].shape and value.dtype == torch.float32 and bool(torch.isfinite(value).all()), "Invalid checkpoint parameter: " + name)
    require(frozen_digest(state["model"]) == FROZEN_WEIGHTS, "Frozen shared-trunk/policy tensor bytes changed")
    opt = state["optimizer"]
    require(isinstance(opt, dict) and set(opt) == {"state", "param_groups"} and len(opt["param_groups"]) == 1, "Checkpoint optimizer groups differ")
    group = opt["param_groups"][0]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    reference = torch.optim.AdamW(trainable, lr=identity["lr"], weight_decay=0., foreach=False).state_dict()["param_groups"][0]
    require(group == reference and group["params"] == [0, 1] and set(opt["state"]) == {0, 1}, "Checkpoint optimizer ownership/hyperparameters differ")
    for i, parameter in enumerate(trainable):
        values = opt["state"][i]
        require(set(values) == {"step", "exp_avg", "exp_avg_sq"}, "Checkpoint optimizer tensor ownership differs")
        require(torch.is_tensor(values["step"]) and values["step"].dtype == torch.float32 and values["step"].ndim == 0 and values["step"].item() == step, "Checkpoint optimizer step differs")
        for name in ("exp_avg", "exp_avg_sq"):
            value = values[name]
            require(torch.is_tensor(value) and value.shape == parameter.shape and value.dtype == torch.float32 and bool(torch.isfinite(value).all()), "Invalid checkpoint optimizer moment")
        require(bool((values["exp_avg_sq"] >= 0).all()), "Negative checkpoint squared moment")
    random.Random().setstate(state["sampler_rng"])
    random.Random().setstate(state["python_rng"])
    require(torch.is_tensor(state["torch_rng"]) and state["torch_rng"].dtype == torch.uint8 and state["torch_rng"].shape == torch.get_rng_state().shape, "Invalid Torch RNG state")
    require(isinstance(state["cuda_rng"], list) and (len(state["cuda_rng"]) == torch.cuda.device_count() if identity["device"] == "cuda" else not state["cuda_rng"]), "Invalid CUDA RNG ownership")
    expected_cuda = torch.cuda.get_rng_state_all() if identity["device"] == "cuda" else []
    for rng, expected_rng in zip(state["cuda_rng"], expected_cuda):
        require(torch.is_tensor(rng) and rng.dtype == torch.uint8 and rng.shape == expected_rng.shape, "Invalid CUDA RNG state")
    require(state["scheduler"] is None and state["amp_scaler"] is None and state["pending_batches"] == [], "Unsupported pending optimizer state")
    return prior.weight_digest(state["model"])


def validate_result(result, identity, admitted):
    require(isinstance(result, dict) and set(result) == RESULT_FIELDS and result["schema"] == SCHEMA and result["status"] == "complete" and result["identity"] == identity,
            "Completed training witness fields/identity differ")
    for name, expected in (("steps_completed", identity["steps"]), ("selected_step", identity["steps"]),
                           ("draws", identity["steps"] * identity["batch_size"]), ("train_rows", admitted["train_rows"]),
                           ("excluded_training_rows", len(admitted["excluded_rows"]))):
        require(type(result[name]) is int and result[name] == expected, "Completed training count differs: " + name)
    require(result["sealed_final_test_consumed"] is False and result["promotion"] == PROMOTION, "Completed ownership/promotion differs")
    require(result["frozen_weights_sha256"] == FROZEN_WEIGHTS and result["frozen_policy_unchanged"] is True, "Completed frozen policy ownership differs")
    require(type(result["elapsed_seconds_this_invocation"]) in (int, float) and math.isfinite(result["elapsed_seconds_this_invocation"]) and result["elapsed_seconds_this_invocation"] > 0,
            "Completed elapsed clock differs")
    for name in ("peak_host_working_set_bytes", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes"):
        require(type(result[name]) is int and 0 <= result[name] < 12 * 1024**3, "Completed resource counter differs")
    require(result["peak_host_working_set_bytes"] > 0, "Missing completed host observation")
    if identity["device"] == "cpu":
        require(result["peak_gpu_allocated_bytes"] == result["peak_gpu_reserved_bytes"] == 0, "CPU result claims GPU work")
    else:
        require(0 < result["peak_gpu_allocated_bytes"] <= result["peak_gpu_reserved_bytes"], "CUDA resource conservation differs")
    recovery = result["recovery"]
    require(isinstance(recovery, dict) and set(recovery) == {"invalid_generations", "restored_step"} and isinstance(recovery["invalid_generations"], list), "Completed recovery witness differs")
    require(type(recovery["restored_step"]) is int and 0 <= recovery["restored_step"] <= identity["steps"], "Completed recovery step differs")
    return result


def verify_run(path=DEFAULT, output=None, fixture=False):
    """Check every committed tensor payload and publish a CPU-readable witness."""
    path = Path(path).resolve()
    require(fixture or path == DEFAULT.resolve(), "Unknown production training path")
    model = Model(path / "model.json")
    require(model_frozen_digest(model) == FROZEN_WEIGHTS, "Export frozen shared-trunk/policy bytes differ")
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    identity = model.identity
    require(identity["schema"] == SCHEMA and identity["fixture"] is fixture and identity["source_sha256"] == sha(__file__) and identity["tool_sha256"] == sha(ROOT / "tools/value-train.py"), "Training verifier source/fixture differs")
    require(identity["initial_model_file_sha256"] == sha(SOURCE) == SOURCE_SHA and identity["initial_weights_sha256"] == SOURCE_WEIGHTS, "Training initial lineage differs")
    require(identity["initial_frozen_weights_sha256"] == FROZEN_WEIGHTS, "Training frozen ancestor identity differs")
    for name, checksum in identity["source_dependencies"].items():
        require(sha(ROOT / name) == checksum, "Training dependency changed: " + name)
    if not fixture:
        from . import value_common
        value_common.admit()
        require(identity["device"] == "cuda", "Production model uses an unapproved training runtime")
        for name, value in CONTRACT.items():
            require(type(identity[name]) is type(value) and identity[name] == value, "Training fixed contract differs: " + name)
        require(identity["plan_file_sha256"] == sha(ROOT / "build/value-plan.json") and identity["implementation_file_sha256"] == sha(ROOT / "build/value-implementation.json"), "Training plan/implementation changed")
    admitted, rows = prepare(REGISTRY)
    require(identity["admitted_manifest_sha256"] == config.digest(admitted) and identity["registry_file_sha256"] == sha(REGISTRY) and identity["split_fingerprints"] == admitted["split_fingerprints"], "Training admitted set changed")
    validate_result(result, identity, admitted)
    torch, prior = runtime()
    prior.deterministic(identity["device"])
    check_model = evaluator().to(identity["device"])
    files = [path / "model.json", path / "result.json", path / "catalog.sqlite3", SOURCE,
             ROOT / "congklak/value_learning.py", ROOT / "tools/value-train.py", REGISTRY,
             ROOT / "build/quality-data.json", ROOT / "build/quality-model-witness.json"]
    if not fixture:
        files += [ROOT / "build/value-plan.json", ROOT / "build/value-implementation.json", ROOT / "build/value-data.json"]
    files += [ROOT / name for name in identity["source_dependencies"]]
    db = sqlite3.connect((path / "catalog.sqlite3").as_uri() + "?mode=ro", uri=True)
    checkpoints = []
    try:
        db.execute("BEGIN")
        configs = db.execute("SELECT body,hash FROM config").fetchall()
        require(len(configs) == 1 and config.digest(json.loads(configs[0][0])) == configs[0][1] and json.loads(configs[0][0]) == identity, "Training catalog identity differs")
        entries = db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id").fetchall()
        seen_steps = [item[0] for item in entries]
        required_steps = list(range(identity["checkpoint_every"], identity["steps"] + 1, identity["checkpoint_every"]))
        if identity["steps"] not in required_steps:
            required_steps.append(identity["steps"])
        require(seen_steps == sorted(set(seen_steps)) and set(required_steps) <= set(seen_steps) and seen_steps[-1] == identity["steps"], "Checkpoint cadence/final step differs")
        sampler = random.Random(identity["seed"])
        groups = defaultdict(list)
        for i, row in enumerate(rows["train"]):
            groups[row["stratum"]].append(i)
        strata = sorted(groups)
        previous_draws = 0
        for step, name, checksum, body, digest in entries:
            file = (path / name).resolve()
            require(file.parent == path and file.is_file() and sha(file) == checksum, "Checkpoint file missing/damaged")
            meta = json.loads(body)
            require(config.digest(meta) == digest and set(meta) == {"step", "identity_sha256", "weights_sha256", "metrics"} and meta["step"] == step and meta["identity_sha256"] == config.digest(identity), "Checkpoint catalog metadata differs")
            state = torch.load(file, map_location="cpu", weights_only=True)
            require(validate_state(state, identity, check_model) == meta["weights_sha256"] and state["step"] == step, "Checkpoint tensor payload differs")
            for _ in range(state["draws"] - previous_draws):
                sampler.choice(groups[sampler.choice(strata)])
            previous_draws = state["draws"]
            require(sampler.getstate() == state["sampler_rng"], "Checkpoint sampler RNG is inconsistent with all committed draws")
            history = db.execute("SELECT body,hash FROM history WHERE step=?", (step,)).fetchall()
            require(len(history) == 1 and config.digest(json.loads(history[0][0])) == history[0][1] and json.loads(history[0][0]) == meta["metrics"], "Checkpoint history differs")
            checkpoints.append({"step": step, "name": name, "sha256": checksum, "weights_sha256": meta["weights_sha256"]})
            files.append(file)
        require(result["weights_sha256"] == model.id == prior.weight_digest(state["model"]), "Final model/result/checkpoint differs")
        require(model.envelope["model"] == export_payload(state["model"], identity, identity["steps"]), "Final export tensor mapping differs")
        check_model.load_state_dict(state["model"])
        cpu_dev = tensors(rows["development"], identity["device"])
        final_metrics = metrics(check_model, rows["development"], cpu_dev)
        require(result["development"] == final_metrics == meta["metrics"], "Final checkpoint/export descriptive metrics differ")
        initial = Model(SOURCE)
        check_model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in initial.weights.items()})
        require(result["initial_development_this_invocation"] == metrics(check_model, rows["development"], cpu_dev), "Initial warm descriptive metrics differ")
    finally:
        db.close()
    value = {"schema": "value-final-model-witness-v1", "passed": True, "identity": identity,
             "model_file_sha256": model.file_sha256, "model_weights_sha256": model.id,
             "selected_step": identity["steps"], "checkpoints_verified": len(checkpoints),
             "scheduled_checkpoints_verified": len(required_steps), "checkpoints": checkpoints,
             "all_optimizer_sampler_rng_states_verified": True, "final_export_equals_checkpoint_tensors": True,
             "all_frozen_tensors_bit_exact": True, "frozen_weights_sha256": FROZEN_WEIGHTS,
             "frozen_parameter_count": 12807, "trainable_parameter_count": 65,
             "train_rows": admitted["train_rows"], "excluded_training_rows": len(admitted["excluded_rows"]),
             "development": result["development"], "initial_development": result["initial_development_this_invocation"],
             "sealed_final_test_consumed": False, "oracle_training": False,
             "inputs": {str(file.relative_to(ROOT)).replace("\\", "/"): sha(file) for file in files}}
    value["sha256"] = config.digest(value)
    if output is not None:
        publish(output, value)
    return value


def witness(path=DEFAULT, proof=None):
    """CPU-only admission of a previously tensor-verified immutable model proof."""
    proof = ROOT / "build/value-model-witness.json" if proof is None else Path(proof)
    value = json.loads(proof.read_text(encoding="utf-8"))
    require(value["schema"] == "value-final-model-witness-v1" and value["passed"] is True and value["sha256"] == config.digest({k: v for k, v in value.items() if k != "sha256"}), "Model witness checksum differs")
    for name, checksum in value["inputs"].items():
        file = (ROOT / name).resolve()
        require(file.is_relative_to(ROOT) and sha(file) == checksum, "Model witness input changed: " + name)
    model = Model(Path(path) / "model.json")
    require(model_frozen_digest(model) == FROZEN_WEIGHTS, "Witness frozen shared-trunk/policy bytes differ")
    require(model.id == value["model_weights_sha256"] and model.file_sha256 == value["model_file_sha256"] and model.identity == value["identity"] and value["selected_step"] == CONTRACT["steps"] and value["all_optimizer_sampler_rng_states_verified"] is True and value["final_export_equals_checkpoint_tensors"] is True, "Final model witness differs")
    require(value["all_frozen_tensors_bit_exact"] is True and value["frozen_weights_sha256"] == FROZEN_WEIGHTS and value["frozen_parameter_count"] == 12807 and value["trainable_parameter_count"] == 65, "Frozen model witness ownership differs")
    identity = model.identity
    require(identity["source_sha256"] == sha(__file__) and identity["tool_sha256"] == sha(ROOT / "tools/value-train.py") and identity["fixture"] is False, "Model witness producer differs")
    require(identity["device"] == "cuda", "Production model witness runtime differs")
    for name, checksum in identity["source_dependencies"].items():
        require(sha(ROOT / name) == checksum, "Model witness dependency changed: " + name)
    for name, expected in CONTRACT.items():
        require(type(identity[name]) is type(expected) and identity[name] == expected, "Model witness fixed contract differs: " + name)
    admitted = json.loads((ROOT / "build/value-data.json").read_text(encoding="utf-8"))
    require(config.digest(admitted) == identity["admitted_manifest_sha256"] and sha(REGISTRY) == identity["registry_file_sha256"], "Model witness admitted set differs")
    result = json.loads((Path(path) / "result.json").read_text(encoding="utf-8"))
    validate_result(result, identity, admitted)
    require(result["weights_sha256"] == model.id and result["development"] == value["development"] and result["initial_development_this_invocation"] == value["initial_development"], "Model witness result differs")
    db = sqlite3.connect((Path(path).resolve() / "catalog.sqlite3").as_uri() + "?mode=ro", uri=True)
    try:
        db.execute("BEGIN")
        configs = db.execute("SELECT body,hash FROM config").fetchall()
        require(len(configs) == 1 and json.loads(configs[0][0]) == identity and config.digest(identity) == configs[0][1], "Model witness catalog identity differs")
        actual = []
        for step, name, checksum, body, digest in db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id"):
            meta = json.loads(body)
            require(config.digest(meta) == digest and meta["step"] == step and meta["identity_sha256"] == config.digest(identity), "Model witness checkpoint metadata differs")
            file = (Path(path) / name).resolve()
            require(file.parent == Path(path).resolve() and sha(file) == checksum, "Model witness checkpoint changed")
            actual.append({"step": step, "name": name, "sha256": checksum, "weights_sha256": meta["weights_sha256"]})
        require(actual == value["checkpoints"] and len(actual) == value["checkpoints_verified"] and actual[-1]["step"] == CONTRACT["steps"] and actual[-1]["weights_sha256"] == model.id, "Model witness checkpoint catalog differs")
        require(meta["metrics"] == result["development"], "Model witness final catalog metrics differ")
    finally:
        db.close()
    return value


def train(path=DEFAULT, device="cuda", on_event=None, cancel=None, fixture=False, steps=None,
          checkpoint_every=None, batch_size=None, registry_path=REGISTRY):
    path = Path(path).resolve()
    settings = dict(CONTRACT)
    if fixture:
        require(path.is_relative_to((ROOT / "build").resolve()), "Training fixture must be inside build")
        settings.update(steps=6 if steps is None else steps, checkpoint_every=2 if checkpoint_every is None else checkpoint_every,
                        batch_size=32 if batch_size is None else batch_size)
        config.integer(settings["steps"], "fixture_steps", 1, 12)
        config.integer(settings["checkpoint_every"], "fixture_checkpoint", 1, 12)
        config.integer(settings["batch_size"], "fixture_batch", 2, 256)
    else:
        require(steps is None and checkpoint_every is None and batch_size is None and path == DEFAULT.resolve(), "Production budget/path must equal frozen contract")
        require(device == "cuda", "Production training requires the fixed CUDA runtime")
        require(Path(registry_path).resolve() == REGISTRY.resolve(), "Production training requires the frozen protected registry")
        from . import value_common
        value_common.admit()
    require(device in ("cpu", "cuda"), "Unknown training device")
    started = time.monotonic()
    admitted, rows = prepare(registry_path)
    if not fixture:
        frozen_data = json.loads((ROOT / "build/value-data.json").read_text(encoding="utf-8"))
        require(admitted == frozen_data, "Re-derived production admitted set differs from frozen metadata")
    initial = Model(SOURCE)
    require(sha(SOURCE) == SOURCE_SHA and initial.id == SOURCE_WEIGHTS and initial.width == 64 and initial.input_count == 127, "Initial refined model differs")
    if on_event:
        on_event({"event": "value_training_cpu_admitted", "train_rows": admitted["train_rows"], "excluded_rows": len(admitted["excluded_rows"]), "development_rows": 1000, "torch_imported": "torch" in __import__("sys").modules})
    torch, prior = runtime()
    require(device != "cuda" or not torch.cuda.is_initialized(), "CUDA training requires a fresh worker after CPU admission")
    prior.deterministic("cpu")
    cpu = {split: tensors(value) for split, value in rows.items()}
    prior.deterministic(device)
    identity = {"schema": SCHEMA, **settings, "rules_digest": config.RULES_DIGEST, "encoder": features_relay.ENCODER127,
                "device": device, "fixture": fixture, "source_sha256": sha(__file__),
                "tool_sha256": sha(ROOT / "tools/value-train.py"), "initial_model_file_sha256": SOURCE_SHA,
                "initial_weights_sha256": SOURCE_WEIGHTS, "initial_frozen_weights_sha256": FROZEN_WEIGHTS,
                "admitted_manifest_sha256": config.digest(admitted),
                "registry_file_sha256": admitted["registry_file_sha256"], "split_fingerprints": admitted["split_fingerprints"],
                "plan_file_sha256": sha(ROOT / "build/value-plan.json") if not fixture else None,
                "implementation_file_sha256": sha(ROOT / "build/value-implementation.json") if not fixture else None,
                "source_dependencies": {name: sha(ROOT / name) for name in ("congklak/quality_learning.py", "congklak/training_v3.py", "congklak/deeper_teacher_v2.py", "congklak/features_relay.py", "congklak/neural_relay.py", "congklak/config.py")},
                "torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda, "python_version": platform.python_version(),
                "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    rng = random.Random(settings["seed"])
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    model = evaluator().to(device)
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in initial.weights.items()})
    trainable = [p for p in model.parameters() if p.requires_grad]
    require(sum(p.numel() for p in trainable) == 65 and frozen_digest(model.state_dict()) == FROZEN_WEIGHTS, "Value-head/frozen parameter ownership differs")
    optimizer = torch.optim.AdamW(trainable, lr=settings["lr"], weight_decay=0., foreach=False)
    data = {split: {k: v.to(device) for k, v in value.items()} for split, value in cpu.items()}
    initial_metrics = metrics(model, rows["development"], data["development"])
    groups = defaultdict(list)
    for i, row in enumerate(rows["train"]):
        groups[row["stratum"]].append(i)
    strata = sorted(groups)
    store = prior.Checkpoints(path, identity)
    step, last_saved, draws = 0, 0, 0
    try:
        saved, recovery = store.restore()
        if saved is not None:
            validate_state(saved, identity, model)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            for values in optimizer.state.values():
                for name, value in values.items():
                    if torch.is_tensor(value):
                        values[name] = value.cpu() if name == "step" else value.to(device)
            rng.setstate(saved["sampler_rng"])
            random.setstate(saved["python_rng"])
            torch.set_rng_state(saved["torch_rng"])
            if device == "cuda":
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            step, draws = saved["step"], saved["draws"]
        last_saved = step
        result_path = path / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            validate_result(result, identity, admitted)
            require(saved is not None and step == settings["steps"], "Completed final checkpoint missing")
            payload = export_payload(saved["model"], identity, step)
            envelope = json.loads((path / "model.json").read_text(encoding="utf-8"))
            require(envelope == {"model": payload, "sha256": config.digest(payload)} and result["weights_sha256"] == payload["weights_sha256"], "Completed model differs")
            require(result["development"] == metrics(model, rows["development"], data["development"]), "Completed descriptive metrics differ")
            require(result["initial_development_this_invocation"] == initial_metrics, "Completed initial metrics differ")
            if on_event:
                on_event({"event": "value_training_completed_verified", "step": step, "files_rewritten": False})
            return result
        if on_event:
            on_event({"event": "value_training_started", "identity": identity, "recovery": recovery, "train_rows": len(rows["train"])})
        def commit():
            nonlocal last_saved
            if device == "cuda":
                torch.cuda.synchronize()
            prior.host_memory()
            state = {"identity": identity, "step": step, "draws": draws,
                     "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                     "optimizer": optimizer.state_dict(), "sampler_rng": rng.getstate(), "python_rng": random.getstate(),
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                     "scheduler": None, "amp_scaler": None, "pending_batches": []}
            validate_state(state, identity, model)
            store.save(state, metrics(model, rows["development"], data["development"]), on_event)
            last_saved = step
        while step < settings["steps"]:
            if cancel and cancel():
                break
            ids = torch.tensor([rng.choice(groups[rng.choice(strata)]) for _ in range(settings["batch_size"])], dtype=torch.int64, device=device)
            t = data["train"]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            residual, logits = model.both(t["x"][ids])
            prediction = t["d"][ids] + t["r"][ids] * residual
            loss = torch.nn.functional.huber_loss(prediction, t["y_player0"][ids], delta=4.)
            require(bool(torch.isfinite(loss)), "Nonfinite seed-space value-head loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            step += 1
            draws += settings["batch_size"]
            if device == "cuda":
                require(torch.cuda.max_memory_reserved() < 12 * prior.GIB, "12GiB GPU reserve exceeded")
            if step % settings["checkpoint_every"] == 0 or step == settings["steps"] or (cancel and cancel()):
                commit()
        if step > last_saved:
            commit()
        if step < settings["steps"]:
            return {"status": "interrupted", "steps_completed": step, "model_available": False, "recovery": recovery}
        if on_event:
            on_event({"event": "value_training_final_checkpoint_ready", "step": step})
        state, _ = store.restore()
        validate_state(state, identity, model)
        payload = export_payload(state["model"], identity, step)
        publish(path / "model.json", {"model": payload, "sha256": config.digest(payload)})
        result = {"schema": SCHEMA, "status": "complete", "identity": identity, "steps_completed": step, "selected_step": step,
                  "draws": draws, "weights_sha256": payload["weights_sha256"], "train_rows": len(rows["train"]),
                  "frozen_weights_sha256": FROZEN_WEIGHTS, "frozen_policy_unchanged": True,
                  "excluded_training_rows": len(admitted["excluded_rows"]), "development": metrics(model, rows["development"], data["development"]),
                  "initial_development_this_invocation": initial_metrics, "recovery": recovery,
                  "elapsed_seconds_this_invocation": time.monotonic() - started,
                  "peak_host_working_set_bytes": prior.host_memory(),
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
                  "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else 0,
                  "sealed_final_test_consumed": False, "promotion": PROMOTION}
        validate_result(result, identity, admitted)
        publish(result_path, result)
        return result
    finally:
        store.close()
