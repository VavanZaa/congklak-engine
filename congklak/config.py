"""Frozen product identity and strict stable-position validation."""
from __future__ import annotations
import hashlib
import json
from .native import ROOT

PROFILE = "id-betawi-relay7-v0"
RULES_DIGEST = "c521cc9fea211aeb63d5942d80cc084ea095da74effec13c041325e80345414e"
ENCODING = "stable-counts-v1"
EVALUATOR = "H_v1"
POLICY = "terminal-first/no-oracle/full-root-window-v2"
DEFAULT_LIMITS = {"max_depth":16,"time_ms":1000,"node_budget":1000000,"memory_mb":512}
SEARCH_DEFAULTS={"tt_mb":16,"use_pvs":True}

class InputError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code=code

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",",":"), ensure_ascii=True, allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

def verify_rules():
    rules=json.loads((ROOT/"rules"/(PROFILE+".json")).read_text(encoding="utf-8"))
    if digest(rules) != RULES_DIGEST:
        raise InputError("rules_contract_mismatch", "Rule file differs from this compiled engineering contract")

def integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise InputError("invalid_input", f"{name} must be an integer in {lower}..{upper}")
    return value

def position(value=None):
    if value is None:
        value={"pits":[[7]*7,[7]*7],"stores":[0,0],"controller":0}
    if not isinstance(value,dict) or set(value)-{"pits","stores","controller"}:
        raise InputError("unsupported_position", "Position requires only pits, stores and controller; import a stable exact-count position")
    if set(value)!={"pits","stores","controller"}:
        raise InputError("invalid_position", "Position requires pits, stores and controller")
    rows=value["pits"]
    if not isinstance(rows,list) or len(rows)!=2 or any(not isinstance(r,list) or len(r)!=7 for r in rows):
        raise InputError("invalid_position", "pits must be two arrays of seven counts")
    stores=value["stores"]
    if not isinstance(stores,list) or len(stores)!=2:
        raise InputError("invalid_position", "stores must contain two counts")
    counts=[integer(x,"seed count",0,98) for r in rows for x in r]
    counts += [integer(x,"store count",0,98) for x in stores]
    if sum(counts)!=98:
        raise InputError("seed_total_mismatch", "Stable product positions must contain exactly 98 seeds")
    return {"pits":[list(r) for r in rows],"stores":list(stores),
            "controller":integer(value["controller"],"controller",0,1)}

def limits(value=None, base=None):
    if value is None: value={}
    if not isinstance(value,dict) or set(value)-(set(DEFAULT_LIMITS)|set(SEARCH_DEFAULTS)):
        raise InputError("invalid_limits", "limits allows max_depth, time_ms, node_budget, memory_mb, tt_mb and use_pvs")
    result=dict(base or {**DEFAULT_LIMITS,**SEARCH_DEFAULTS})
    result.update(value)
    ranges={"max_depth":(0,128),"time_ms":(0,86400000),"node_budget":(0,2**63-1),"memory_mb":(16,24576)}
    checked={k:integer(result[k],k,*bounds) for k,bounds in ranges.items()}
    maximum=min(1024,checked["memory_mb"]-16)
    tt=result.get("tt_mb",16)
    if "tt_mb" not in value and base is None: tt=min(tt,maximum)
    checked["tt_mb"]=integer(tt,"tt_mb",0,maximum)
    if type(result.get("use_pvs",True)) is not bool:
        raise InputError("invalid_limits","use_pvs must be boolean")
    checked["use_pvs"]=result.get("use_pvs",True)
    return checked

def request(value):
    if not isinstance(value,dict):
        raise InputError("invalid_request", "Request must be a JSON object")
    allowed={"op","id","profile","rules_digest","state_encoding","position","action","limits","job_id","trace"}
    if set(value)-allowed:
        raise InputError("unsupported_request", "Unsupported request fields: "+", ".join(sorted(set(value)-allowed)))
    if value.get("profile",PROFILE)!=PROFILE:
        raise InputError("unsupported_profile", "Only experimental "+PROFILE+" is implemented")
    if value.get("rules_digest",RULES_DIGEST)!=RULES_DIGEST or value.get("state_encoding",ENCODING)!=ENCODING:
        raise InputError("identity_mismatch", "Rules digest or state encoding is incompatible")
    if value.get("op") not in {"validate","apply","analyze","stop","load","resume","oracle"}:
        raise InputError("unknown_operation", "Use validate, apply, analyze, stop, load, resume or oracle")
    if "id" in value and (type(value["id"]) not in (str,int) or len(str(value["id"]))>128):
        raise InputError("invalid_request_id", "id must be a string or integer of at most 128 characters")
    return value

def identity(build_id):
    return {"profile":PROFILE,"rules_digest":RULES_DIGEST,"state_encoding":ENCODING,
            "evaluator":EVALUATOR,"policy":POLICY,"native_build_sha256":build_id,"schema":"analysis-job-v2"}
