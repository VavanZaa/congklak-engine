"""Root choice policy targets from complete H6 action scores; no child value rows."""
import math
from collections import Counter
from . import config

SCHEMA="root-h6-action-preference-v1"
TEMPERATURE_SEEDS=4.0
HORIZON=6

def target(row,temperature=TEMPERATURE_SEEDS):
    if type(temperature) not in (int,float) or not math.isfinite(temperature) or temperature<=0: raise ValueError("Invalid policy temperature")
    p=config.position(row["state"]); active=sum(map(sum,p["pits"])); actor=p["controller"]
    if active and not sum(p["pits"][actor]): actor=1-actor
    legal=[pit for pit,count in enumerate(p["pits"][actor]) if count]
    if not active or not legal: raise ValueError("No terminal/all-masked training policy target")
    label=row["label"]
    if label["kind"]!="finite_choice_horizon" or label["choice_depth"]!=HORIZON or label["oracle_manifest_sha256"] is not None:
        raise ValueError("Policy requires oracle-free root H6 labels")
    actions=label["action_scores"]
    if [item["pit"] for item in actions]!=legal or row["legal_mask"]!=sum(1<<pit for pit in legal):
        raise ValueError("Root action Q labels must cover every legal action exactly once")
    values=[item["value_player0"] for item in actions]
    offset=p["stores"][0]-p["stores"][1]
    if any(type(value) not in (int,float) or not math.isfinite(value) or not offset-active<=value<=offset+active for value in values):
        raise ValueError("Invalid root choice Q")
    if type(label["target_player0"]) not in (int,float) or not math.isfinite(label["target_player0"]):
        raise ValueError("Invalid root value")
    sign=1 if actor==0 else -1; best=max(sign*value for value in values)
    if sign*label["target_player0"]!=best or label["best_moves"]!=[pit for pit,value in zip(legal,values) if sign*value==best]:
        raise ValueError("Root value or best-action set differs from action Q labels")
    q=[0.0]*7; probability=[0.0]*7; mask=[False]*7
    unnormalized=[math.exp((sign*value-best)/temperature) for value in values]
    total=sum(unnormalized)
    for pit,value,weight in zip(legal,values,unnormalized):
        q[pit]=sign*value; probability[pit]=weight/total; mask[pit]=True
    return probability,mask,q,sign

def ordinary_calibration(train_rows):
    """Eligibility uses train counts only; coefficients are predeclared constants."""
    counts=Counter(row["stratum"] for row in train_rows)
    return {"schema":"train-ordinary-stratum-eligibility-v1","minimum_train_rows":20,
            "eligible_strata":[name for name,count in sorted(counts.items()) if name.endswith("/ordinary") and count>=20],
            "eligible_train_counts":{name:count for name,count in sorted(counts.items()) if name.endswith("/ordinary") and count>=20},
            "batch_missing_strata":"omit from mean; if none present return graph-connected zero",
            "coefficient":.01,"unit":"squared player0 seeds after current-root-actor sign",
            "development_calibration":False}
