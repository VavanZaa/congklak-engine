"""Deterministic pilot collection, isolated splits and resumable label publication.

Work is transactional SQLite. Published generations are immutable SQLite indexes
and packed 52-byte row shards, referenced only by a committed catalog pointer.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import struct
import sys
import time
import uuid
from . import config
from .native import Native,ROOT,flatten,Control
from .oracle import Oracle
from .reports import validate as validate_report

SCHEMA="congklak-pilot-v1"
DEFAULT_PATH=ROOT/"data"/"pilot-10000"
SPLITS=("train","development","sealed_final_test")
SOURCES=("random_history","heuristic_history","search_history")
MIB=1024*1024
ROW=struct.Struct("<17shBBBB7hf dHB".replace(" ",""))

class PilotError(RuntimeError):
    def __init__(self,message,code="pilot_error"):
        super().__init__(message); self.code=code

class Interrupted(PilotError):
    def __init__(self): super().__init__("Pilot stopped at a committed boundary","interrupted")

def canonicalize(root,native):
    normal=native.normalize(flatten(config.position(root)),actor=root["controller"],total=98)
    p=normal["position"]
    actor=0 if normal["terminal"] else p["controller"]
    a=bytes(flatten(p)+[actor])
    swapped={"pits":[p["pits"][1],p["pits"][0]],"stores":p["stores"][::-1],
             "controller":0 if normal["terminal"] else 1-actor}
    b=bytes(flatten(swapped)+[swapped["controller"]])
    if b<a: return b,swapped,True
    p={**p,"controller":actor}
    return a,p,False

def from_key(key):
    return {"pits":[list(key[:7]),list(key[8:15])],"stores":[key[7],key[15]],"controller":key[16]}

def quotient(root):
    counts=tuple(root["pits"][0]+root["pits"][1]); actor=root["controller"]
    if not any(counts): actor=0
    a=bytes(counts)+bytes([actor])
    b=bytes(counts[7:]+counts[:7])+bytes([0 if not any(counts) else 1-actor])
    return min(a,b)

def family_split(family):
    rank=int.from_bytes(hashlib.sha256(("split-v1/"+family).encode()).digest()[:8],"big")%100
    return SPLITS[0 if rank<80 else 1 if rank<90 else 2]

def _mechanism(root,native):
    normal=native.normalize(flatten(root),actor=root["controller"],total=98)
    active=sum(sum(r) for r in root["pits"])
    band="0-4" if active<=4 else "5-24" if active<=24 else "25-49" if active<=49 else "50-74" if active<=74 else "75-98"
    if normal["terminal"]: kind="terminal"
    else:
        facts=[native.apply(flatten(root),actor=root["controller"],action=a,total=98,trace=False) for a in normal["legal_actions"]]
        kind=("pass" if any(t["passes"] for t in facts) else
              "capture" if any(t["reason"]=="own_capture" for t in facts) else
              "bonus" if any(t["reason"]=="own_store" for t in facts) else "ordinary")
    return active,band+"/"+kind

class Work:
    def __init__(self,path,identity,quota_mb=51200,min_free_mb=2048):
        self.path=Path(path).resolve(); self.identity=identity
        self.quota=quota_mb*MIB; self.reserve=min_free_mb*MIB
        self.db=None; self.lock=None
        self.path.mkdir(parents=True,exist_ok=True)
        try:
            self.lock=(self.path/"writer.lock").open("a+b")
            if self.lock.seek(0,2)==0: self.lock.write(b"0"); self.lock.flush()
            self.lock.seek(0)
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.admit(8*MIB)
            self.db=sqlite3.connect(self.path/"work.sqlite3")
            self.db.row_factory=sqlite3.Row
            version=self.db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0,2): raise PilotError("Incompatible pilot work schema","incompatible_saved_data")
            if self.db.execute("PRAGMA quick_check").fetchone()[0]!="ok": raise PilotError("Corrupt pilot SQLite","corrupt_saved_data")
            self.db.execute("PRAGMA journal_mode=WAL"); self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS families(id TEXT PRIMARY KEY,split TEXT NOT NULL,source TEXT NOT NULL,seed INTEGER NOT NULL,body TEXT NOT NULL,hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS collection_units(idx INTEGER PRIMARY KEY,family TEXT NOT NULL,archive_hash TEXT NOT NULL,hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS candidates(state BLOB PRIMARY KEY,position TEXT NOT NULL,stratum TEXT NOT NULL,active INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS origins(state BLOB NOT NULL,family TEXT NOT NULL,step INTEGER NOT NULL,swapped INTEGER NOT NULL,PRIMARY KEY(state,family));
            CREATE TABLE IF NOT EXISTS exclusions(kind TEXT NOT NULL,state BLOB NOT NULL,split TEXT NOT NULL,PRIMARY KEY(kind,state));
            CREATE TABLE IF NOT EXISTS selected(state BLOB PRIMARY KEY,family TEXT NOT NULL,step INTEGER NOT NULL,swapped INTEGER NOT NULL,split TEXT NOT NULL,stratum TEXT NOT NULL,inclusion REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS labels(state BLOB PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS artifacts(id TEXT PRIMARY KEY,name TEXT NOT NULL,status TEXT NOT NULL,owner_pid INTEGER NOT NULL,body TEXT,hash TEXT);
            PRAGMA user_version=2;
            """)
            prior=self.get("identity")
            if prior is not None and prior!=identity: raise PilotError("Pilot identities/settings differ; use another directory","incompatible_saved_data")
            if prior is None:
                with self.db:
                    self.put("identity",identity)
                    self.put("progress",{"stage":"collect","cursor":0,"duplicates":0,"observations":0,"elapsed_seconds":0})
        except Exception:
            self.close(); raise

    def admit(self,estimate):
        # Parent is the managed data root; count sibling oracle/library/WAL once.
        physical=sum(f.stat().st_size for f in self.path.parent.rglob("*") if f.is_file())
        if physical+estimate+self.reserve>self.quota:
            raise PilotError("Pilot write/recovery reserve exceeds managed physical quota","storage_error")
        if shutil.disk_usage(self.path).free<estimate+self.reserve:
            raise PilotError("Insufficient disk plus 2 GiB bulk recovery headroom","storage_error")

    def get(self,key):
        row=self.db.execute("SELECT body,hash FROM meta WHERE key=?",(key,)).fetchone()
        if row is None: return None
        try:
            value=json.loads(row[0])
            if config.digest(value)!=row[1]: raise ValueError("checksum")
            return value
        except (ValueError,TypeError,RecursionError) as exc: raise PilotError("Corrupt pilot metadata "+key,"corrupt_saved_data") from exc

    def put(self,key,value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?,?)",(key,config.canonical(value),config.digest(value)))

    def close(self):
        if self.db: self.db.close(); self.db=None
        if self.lock: self.lock.close(); self.lock=None

def _family(index,seed,native):
    if index%4==3:
        rng=random.Random(seed+index*104729)
        active=rng.randrange(1,5); counts=[0]*14
        for _ in range(active): counts[rng.randrange(14)]+=1
        actor=rng.randrange(2)
        if not any(counts[actor*7:actor*7+7]): actor=1-actor
        bank=rng.randrange(99-active)
        p={"pits":[counts[:7],counts[7:]],"stores":[bank,98-active-bank],"controller":actor}
        q=quotient(p)
        statekey,_,_=canonicalize(p,native)
        family="constructed-k4/"+q.hex()+"/"+statekey.hex()
        # Only benchmarks receive oracle labels; no exact supervision leaks into train.
        split="development" if hashlib.sha256(q).digest()[0]<128 else "sealed_final_test"
        return family,split,"constructed_exact",seed+index*104729,p,[],"counterfactual_valid_stable",[p]
    source=SOURCES[index%3]
    family=f"{source}/{seed}/{index}"
    rng=random.Random(seed+index*104729)
    prefix=rng.randrange(4,16)
    p=config.position(); moves=[]; states=[p]
    for step in range(256):
        n=native.normalize(flatten(p),actor=p["controller"],total=98)
        if n["terminal"]: break
        actions=n["legal_actions"]
        if step<prefix or source=="random_history" or rng.random()<.25:
            action=rng.choice(actions)
        elif source=="search_history":
            r=native.analyze(flatten(p),actor=p["controller"],total=98,max_depth=2,time_ms=0,node_budget=0,tt_mb=1,keep_history=False)
            if r["report"] is None: raise PilotError("Generator search did not complete")
            action=rng.choice(r["report"]["best_moves"])
        else:
            scored=[]
            for a in actions:
                t=native.apply(flatten(p),actor=p["controller"],action=a,total=98,trace=False)
                d=t["position"]["stores"][0]-t["position"]["stores"][1]
                scored.append((d if p["controller"]==0 else -d,a))
            best=max(v for v,a in scored); action=rng.choice([a for v,a in scored if v==best])
        moves.append(action)
        p=native.apply(flatten(p),actor=p["controller"],action=action,total=98,trace=False)["position"]
        states.append(p)
    status="terminal" if not any(sum(r) for r in p["pits"]) else "truncated_prefix"
    # Thin neighboring decisions before admission; still retain tactical variety.
    selected=[(i,s) for i,s in enumerate(states) if i%3==0 or i==len(states)-1]
    return family,family_split(family),source,seed+index*104729,states[0],moves,"observed_legal_history",selected,status

def _need(condition,message):
    if not condition: raise PilotError(message,"corrupt_saved_data")

def _int(value,lo=0,hi=2**63-1):
    _need(type(value) is int and lo<=value<=hi,"Invalid pilot integer")

def _validate_progress(work):
    p=work.get("progress")
    _need(isinstance(p,dict) and {"stage","cursor","duplicates","observations","elapsed_seconds"}<=set(p),"Malformed pilot progression")
    _need(p["stage"] in {"collect","select","label","publish","complete"},"Unknown pilot stage")
    _int(p["cursor"],0,work.identity["family_count"]); _int(p["duplicates"]); _int(p["observations"])
    units=work.db.execute("SELECT * FROM collection_units ORDER BY idx").fetchall()
    _need(len(units)==p["cursor"],"Collection cursor is not bound to committed units")
    for expected,u in enumerate(units):
        _need(type(u["idx"]) is int and u["idx"]==expected,"Collection units are not contiguous")
        _need(config.digest({"index":u["idx"],"family":u["family"],"archive_hash":u["archive_hash"]})==u["hash"],"Corrupt collection unit")
        family=work.db.execute("SELECT hash FROM families WHERE id=?",(u["family"],)).fetchone()
        if family is not None: _need(family[0]==u["archive_hash"],"Collection unit/archive mismatch")
        else: _need(p["stage"]=="complete","Unfinished collection is missing its archive")
    _need(type(p["elapsed_seconds"]) in (float,int) and p["elapsed_seconds"]>=0,"Invalid progression time")
    if p["stage"]!="collect": _need(p["cursor"]==work.identity["family_count"],"Incomplete collection cannot advance")
    if p["stage"] in {"label","publish","complete"}:
        _need(work.db.execute("SELECT COUNT(*) FROM selected").fetchone()[0]==work.identity["target"],"Selection is incomplete")
    if p["stage"] in {"publish","complete"}:
        _need(work.db.execute("SELECT COUNT(*) FROM labels").fetchone()[0]==work.identity["target"],"Labels are incomplete")
    return p

def _validate_archive(row):
    archive=json.loads(row["body"])
    _need(config.digest(archive)==row["hash"],"Family checksum mismatch")
    _need(isinstance(archive,dict) and set(archive)=={"start","moves","reachability","status","generator","source","seed"},"Malformed family archive")
    _need(isinstance(archive["start"],dict),"Missing explicit archive position")
    config.position(archive["start"])
    _need(isinstance(archive["moves"],list) and len(archive["moves"])<=256,"Unbounded family moves")
    for a in archive["moves"]: _int(a,0,6)
    _need(archive["source"]==row["source"] and archive["seed"]==row["seed"] and archive["generator"]=="legal-prefix-v1","Family identity mismatch")
    _need(row["split"] in SPLITS and archive["status"] in {"terminal","truncated_prefix","counterfactual"},"Malformed family status")
    _need(archive["reachability"] in {"counterfactual_valid_stable","observed_legal_history"},"Malformed reachability")
    return archive

def _memory_check(native):
    with Control(native,max_depth=0,time_ms=0,node_budget=0,memory_mb=512,tt_mb=0,use_pvs=False) as control:
        result=native.apply(flatten(config.position()),action=0,total=98,trace=False,control=control)
        if not result["ok"]: raise PilotError("Pilot process memory monitor stopped work","memory_budget")

def _check_pending(work,native,oracle):
    expected_count=work.db.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    rows=work.db.execute("SELECT l.state,l.body,l.hash,s.family,s.step,s.swapped,s.split,s.stratum,s.inclusion,c.position FROM labels l JOIN selected s ON s.state=l.state JOIN candidates c ON c.state=s.state ORDER BY l.state").fetchall()
    _need(len(rows)==expected_count,"Unselected/orphan pending label")
    counts=Counter((r["split"],r["stratum"]) for r in work.db.execute("SELECT split,stratum FROM selected"))
    support={sp:len({s for split,s in counts if split==sp}) for sp in SPLITS}
    cached={}
    for row in rows:
        record=json.loads(row["body"])
        _need(config.digest(record)==row["hash"],"Bad pending checksum")
        validate_record(record,work.identity,native,oracle,relabel=True)
        _need(record["state_key"]==bytes(row["state"]).hex() and record["state"]==json.loads(row["position"]),"Pending body/index mismatch")
        _need(record["family"]==row["family"] and record["step"]==row["step"] and record["source_swapped"]==bool(row["swapped"]) and record["split"]==row["split"] and record["stratum"]==row["stratum"],"Pending selection provenance mismatch")
        _need(record["selection_inclusion_probability"]==row["inclusion"] and record["draw_probability"]==1/(support[row["split"]]*counts[(row["split"],row["stratum"])]),"Pending selection probability mismatch")
        if row["family"] not in cached:
            family=work.db.execute("SELECT * FROM families WHERE id=?",(row["family"],)).fetchone()
            _need(family is not None and family["split"]==row["split"],"Pending family missing/ownership mismatch")
            archive=_validate_archive(family); states=[archive["start"]]
            for action in archive["moves"]:
                p=states[-1]; step=native.apply(flatten(p),actor=p["controller"],action=action,total=98,trace=False)
                _need(step["ok"] is True,"Illegal archived choice"); states.append(step["position"])
            cached[row["family"]]=states
        states=cached[row["family"]]
        _need(row["step"]<len(states),"Pending provenance step exceeds archive")
        key,_,swapped=canonicalize(states[row["step"]],native)
        _need(key==row["state"] and swapped==bool(row["swapped"]),"Pending archived state mismatch")
    _memory_check(native)

def validate_record(record,identity,native,oracle,relabel=False):
    fields={"state","state_key","rules_digest","state_encoding","objective","legal_mask","label","family","step","source_swapped","split","stratum",
            "teacher_revision","label_weight","sampling_priority","relabel_priority","retention_reason","selection_inclusion_probability","draw_probability","importance_ratio"}
    _need(isinstance(record,dict) and set(record)==fields,"Malformed pilot record fields")
    _need(isinstance(record["state"],dict),"Missing explicit record position")
    p=config.position(record["state"]); key,canonical,_=canonicalize(p,native)
    _need(record["state_key"]==key.hex() and canonical==p,"Noncanonical pilot state")
    _need(record["rules_digest"]==identity["rules_digest"] and record["state_encoding"]==identity["state_encoding"] and
          record["objective"]=="final-store0-minus-store1" and record["teacher_revision"]==identity["teacher_revision"],"Pilot record identity mismatch")
    _need(record["split"] in SPLITS and isinstance(record["family"],str) and 1<=len(record["family"])<=256,"Malformed record family/split")
    _int(record["step"],0,256); _need(type(record["source_swapped"]) is bool,"Malformed swap flag")
    active,stratum=_mechanism(p,native)
    _need(record["stratum"]==stratum and record["retention_reason"]=="validated-uniform-stratum-pilot","Invalid record stratum/retention")
    for k in ("label_weight","sampling_priority","importance_ratio"):
        _need(type(record[k]) in (float,int) and record[k]==1,"Non-unit pilot weight")
    _need(type(record["relabel_priority"]) in (float,int) and record["relabel_priority"]==0,"Undeclared relabel priority")
    for k in ("selection_inclusion_probability","draw_probability"):
        _need(type(record[k]) in (float,int) and 0<record[k]<=1,"Invalid sampling probability")
    n=native.normalize(flatten(p),actor=p["controller"],total=98); legal=n["legal_actions"]
    _int(record["legal_mask"],0,127); _need(record["legal_mask"]==sum(1<<a for a in legal),"Wrong legal mask")
    l=record["label"]
    _need(isinstance(l,dict) and set(l)=={"kind","target_player0","best_moves","action_scores","choice_depth","oracle_manifest_sha256","coverage","work"},"Malformed label fields")
    _int(l["target_player0"],-98,98); d=p["stores"][0]-p["stores"][1]
    _need(d-active<=l["target_player0"]<=d+active,"Target outside physical range")
    _need(isinstance(l["best_moves"],list) and isinstance(l["action_scores"],list),"Malformed action targets")
    for a in l["best_moves"]: _int(a,0,6)
    scores={}
    for a in l["action_scores"]:
        _need(isinstance(a,dict) and set(a)=={"pit","value_player0"},"Malformed action score")
        _int(a["pit"],0,6); _int(a["value_player0"],-98,98)
        _need(a["pit"] not in scores,"Duplicate scored action"); scores[a["pit"]]=a["value_player0"]
    _need(list(scores)==legal,"Incomplete/misordered scored actions")
    if legal:
        value=(max if p["controller"]==0 else min)(scores.values())
        _need(l["target_player0"]==value and l["best_moves"]==[a for a in legal if scores[a]==value],"Invalid complete tied ranking")
    else: _need(l["best_moves"]==[] and l["target_player0"]==d,"Invalid terminal label")
    if active<=4:
        _need(record["split"]!="train" and l["kind"]=="exact_optimal_margin" and l["choice_depth"] is None,"Oracle benchmark target kind/ownership mismatch")
        _need(l["oracle_manifest_sha256"]==identity["oracle_manifest_sha256"]==oracle.manifest_sha256 and l["target_player0"]%2==0,"Wrong exact proof identity/parity")
        exact=oracle.lookup(p)
        _need(exact["covered"] and exact["value_player0"]==l["target_player0"] and exact["best_moves"]==l["best_moves"],"Wrong exact oracle label")
        for a,v in scores.items():
            child=native.apply(flatten(p),actor=p["controller"],action=a,total=98,trace=False)["position"]
            _need(oracle.lookup(child)["value_player0"]==v,"Wrong exact action score")
        _need(l["coverage"]=={"domain":"verified-K4"} and l["work"]=={},"Malformed exact coverage")
    else:
        _need(l["kind"]=="finite_choice_horizon" and l["oracle_manifest_sha256"] is None,"Invalid teacher kind/oracle ancestry")
        _int(l["choice_depth"],1,128); _need(l["choice_depth"]==identity["teacher_limits"]["max_depth"],"Wrong teacher horizon")
        _need(isinstance(l["coverage"],dict) and isinstance(l["work"],dict) and set(l["work"])=={"nodes","deposits","elapsed_ms","peak_process_bytes"},"Malformed teacher coverage/work")
        from .reports import coverage
        try: coverage(l["coverage"],l["choice_depth"])
        except (ValueError,TypeError,KeyError) as exc: raise PilotError("Malformed teacher coverage","corrupt_saved_data") from exc
        for value in l["work"].values(): _int(value)
        if relabel:
            result=native.analyze(flatten(p),actor=p["controller"],total=98,**identity["teacher_limits"],keep_history=False)
            report=result["report"]
            _need(result["stop_reason"]=="max_depth" and report is not None,"Re-labeling did not complete")
            expected={c["pit"]:c["depth_result"]["value_player0"] for c in report["candidates"]}
            _need(report["depth_result"]["value_player0"]==l["target_player0"] and report["best_moves"]==l["best_moves"] and scores==expected,"Teacher action relabel mismatch")
    return record

def _collect(work,native,on_progress,cancel):
    progress=work.get("progress"); identity=work.identity
    target=identity["target"]; seed=identity["seed"]
    # Fixed family budget is declared before generation, not adapted by test results.
    family_count=identity["family_count"]
    for index in range(progress["cursor"],family_count):
        if cancel and cancel(): raise Interrupted()
        if index%10==0: _memory_check(native)
        before=time.monotonic()
        result=_family(index,seed,native)
        if index%4==3:
            family,split,source,gseed,start,moves,reachable,positions=result
            states=[(0,p) for p in positions]; status="counterfactual"
        else:
            family,split,source,gseed,start,moves,reachable,states,status=result
        archive={"start":start,"moves":moves,"reachability":reachable,"status":status,"generator":"legal-prefix-v1","source":source,"seed":gseed}
        observations=[]
        for step,p in states:
            key,canonical,swapped=canonicalize(p,native)
            active,stratum=_mechanism(canonical,native)
            observations.append((key,canonical,swapped,step,active,stratum))
        work.admit(MIB)
        with work.db:
            work.db.execute("INSERT OR IGNORE INTO families VALUES(?,?,?,?,?,?)",(family,split,source,gseed,config.canonical(archive),config.digest(archive)))
            archive_hash=work.db.execute("SELECT hash FROM families WHERE id=?",(family,)).fetchone()[0]
            unit={"index":index,"family":family,"archive_hash":archive_hash}
            work.db.execute("INSERT INTO collection_units VALUES(?,?,?,?)",(index,family,archive_hash,config.digest(unit)))
            for key,p,swapped,step,active,stratum in observations:
                old=work.db.execute("INSERT OR IGNORE INTO candidates VALUES(?,?,?,?)",(key,config.canonical(p),stratum,active)).rowcount
                if not old: progress["duplicates"]+=1
                work.db.execute("INSERT OR IGNORE INTO origins VALUES(?,?,?,?)",(key,family,step,int(swapped)))
            progress["cursor"]=index+1; progress["observations"]+=len(observations)
            _need(work.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]<=100000,"Candidate buffer limit reached")
            progress["elapsed_seconds"]+=time.monotonic()-before
            work.put("progress",progress)
        if on_progress and (index%10==0 or index+1==family_count): on_progress({**progress,"candidate_count":work.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]})
    with work.db:
        progress["stage"]="select"; work.put("progress",progress)

def _select(work,native,on_progress,cancel):
    if cancel and cancel(): raise Interrupted()
    ranks={s:i for i,s in enumerate(SPLITS)}
    buckets=defaultdict(list); withheld=0
    # All families now exist. Holdout exclusions are frozen before ANY labeling.
    rows=work.db.execute("SELECT * FROM candidates ORDER BY state").fetchall()
    _need(len(rows)<=100000,"Candidate memory cap exceeded")
    owner={}
    claims=defaultdict(list)
    for row in work.db.execute("SELECT o.*,f.split FROM origins o JOIN families f ON o.family=f.id ORDER BY o.family"):
        claims[bytes(row["state"])].append(dict(row))
    qclaims={}
    for row in rows:
        p=json.loads(row["position"]); key=bytes(row["state"])
        split=max((c["split"] for c in claims[key]),key=ranks.get)
        owner[key]=split
        if split!="train":
            q=quotient(p)
            if row["active"]<=4: qclaims[q]=max(split,qclaims.get(q,"train"),key=ranks.get)
    work.admit(8*MIB)
    with work.db:
        work.db.execute("DELETE FROM selected"); work.db.execute("DELETE FROM exclusions")
        for row in rows:
            key=bytes(row["state"]); p=json.loads(row["position"]); split=owner[key]
            if split!="train": work.db.execute("INSERT INTO exclusions VALUES('state',?,?)",(key,split))
            q=quotient(p)
            if q in qclaims: work.db.execute("INSERT OR IGNORE INTO exclusions VALUES('quotient',?,?)",(q,qclaims[q]))
            if split=="train" and row["active"]<=4:
                withheld+=1; continue
            origin=min((c for c in claims[key] if c["split"]==split),key=lambda c:(c["family"],c["step"]))
            score=hashlib.sha256(str(work.identity["seed"]).encode()+key).digest()
            buckets[row["stratum"]].append((score,key,origin,split))
        target=work.identity["target"]
        if sum(map(len,buckets.values()))<target: raise PilotError("Insufficient unique eligible candidates; increase declared family_count in a new campaign")
        for b in buckets.values(): b.sort(key=lambda x:x[0])
        quotas={k:0 for k in sorted(buckets)}
        # Balanced quotas with explicit without-replacement inclusion probabilities.
        for i in range(target):
            available=[k for k in quotas if quotas[k]<len(buckets[k])]
            k=min(available,key=lambda k:(quotas[k],k)); quotas[k]+=1
        for stratum,count in quotas.items():
            for score,key,o,split in buckets[stratum][:count]:
                work.db.execute("INSERT INTO selected VALUES(?,?,?,?,?,?,?)",(key,o["family"],o["step"],o["swapped"],split,stratum,count/len(buckets[stratum])))
        progress=work.get("progress"); progress.update(stage="label",withheld_train_oracle_domain=withheld)
        work.put("progress",progress)
        work.put("selection",{"quotas":quotas,"eligible_counts":{k:len(v) for k,v in buckets.items()},"method":"seeded-uniform-priority/without-replacement/balanced-strata-v1"})
    if on_progress: on_progress(progress)

def _label(work,native,oracle,on_progress,cancel):
    rows=work.db.execute("SELECT s.*,c.position,c.active FROM selected s JOIN candidates c ON c.state=s.state WHERE NOT EXISTS(SELECT 1 FROM labels l WHERE l.state=s.state) ORDER BY s.state").fetchall()
    counts=Counter((r["split"],r["stratum"]) for r in work.db.execute("SELECT split,stratum FROM selected"))
    strata={split:len({s for sp,s in counts if sp==split}) for split in SPLITS}
    settings=work.identity["teacher_limits"]
    for row in rows:
        if cancel and cancel(): raise Interrupted()
        before=time.monotonic(); p=json.loads(row["position"])
        normal=native.normalize(flatten(p),actor=p["controller"],total=98)
        legal=normal["legal_actions"]; d=p["stores"][0]-p["stores"][1]
        if row["active"]<=4:
            if row["split"]=="train": raise PilotError("Oracle ancestry cannot enter training")
            lookup=oracle.lookup(p)
            if not lookup["covered"]: raise PilotError("Oracle did not cover admitted endgame")
            value=lookup["value_player0"]; best=lookup["best_moves"]
            action_scores=[]
            for a in legal:
                step=native.apply(flatten(p),actor=p["controller"],action=a,total=98,trace=False)
                action_scores.append({"pit":a,"value_player0":oracle.lookup(step["position"])["value_player0"]})
            label={"kind":"exact_optimal_margin","target_player0":value,"best_moves":best,
                   "action_scores":action_scores,"choice_depth":None,"oracle_manifest_sha256":lookup["oracle_manifest_sha256"],"coverage":{"domain":"verified-K4"},"work":{}}
        else:
            result=native.analyze(flatten(p),actor=p["controller"],total=98,**settings,keep_history=False)
            report=result["report"]
            if result["stop_reason"]!="max_depth" or report is None: raise PilotError("Incomplete teacher target is not admitted")
            validate_report(report,p,config.identity(native.build_id),native)
            value=report["depth_result"]["value_player0"]; best=report["best_moves"]
            label={"kind":"finite_choice_horizon","target_player0":value,"best_moves":best,
                   "action_scores":[{"pit":c["pit"],"value_player0":c["depth_result"]["value_player0"]} for c in report["candidates"]],
                   "choice_depth":settings["max_depth"],"oracle_manifest_sha256":None,"coverage":report["coverage"],
                   "work":{k:result[k] for k in ("nodes","deposits","elapsed_ms","peak_process_bytes")}}
        if not d-row["active"]<=value<=d+row["active"]: raise PilotError("Target violates physical margin interval")
        if label["kind"]=="exact_optimal_margin" and (value-98)%2: raise PilotError("Exact target violates conservation parity")
        record={"state":p,"state_key":bytes(row["state"]).hex(),"rules_digest":config.RULES_DIGEST,"state_encoding":"canonical-player-swap/normalized/terminal-actor0-v1",
                "objective":"final-store0-minus-store1","legal_mask":sum(1<<a for a in legal),"label":label,
                "family":row["family"],"step":row["step"],"source_swapped":bool(row["swapped"]),"split":row["split"],"stratum":row["stratum"],
                "teacher_revision":work.identity["teacher_revision"],"label_weight":1.0,"sampling_priority":1.0,"relabel_priority":0.0,
                "retention_reason":"validated-uniform-stratum-pilot","selection_inclusion_probability":row["inclusion"],
                "draw_probability":1/(strata[row["split"]]*counts[(row["split"],row["stratum"])]),"importance_ratio":1.0}
        validate_record(record,work.identity,native,oracle)
        work.admit(MIB)
        with work.db:
            work.db.execute("INSERT INTO labels VALUES(?,?,?)",(row["state"],config.canonical(record),config.digest(record)))
            progress=work.get("progress"); progress["elapsed_seconds"]+=time.monotonic()-before
            progress["labels"]=work.db.execute("SELECT COUNT(*) FROM labels").fetchone()[0]; work.put("progress",progress)
        if on_progress and (progress["labels"]%100==0 or progress["labels"]==work.identity["target"]): on_progress(progress)
    with work.db:
        progress=work.get("progress"); progress["stage"]="publish"; work.put("progress",progress)

def _safe(path,name):
    target=(path/name).resolve()
    if not target.is_relative_to(path.resolve()): raise PilotError("Artifact escapes managed path","corrupt_saved_data")
    return target

def _sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(MIB),b""): h.update(chunk)
    return h.hexdigest()

def _packed(record,stratum_id):
    label=record["label"]; scores={a["pit"]:a["value_player0"] for a in label["action_scores"]}
    return ROW.pack(bytes.fromhex(record["state_key"]),label["target_player0"],1 if label["kind"]=="exact_optimal_margin" else 2,
                    sum(1<<a for a in label["best_moves"]),record["legal_mask"],label["choice_depth"] or 0,
                    *(scores.get(a,32767) for a in range(7)),record["label_weight"],record["draw_probability"],stratum_id,SPLITS.index(record["split"]))

def _publish(work,native,oracle,on_progress,cancel):
    if cancel and cancel(): raise Interrupted()
    _check_pending(work,native,oracle)
    generation=uuid.uuid4().hex
    directory="generation-"+generation
    with work.db: work.db.execute("INSERT INTO artifacts VALUES(?,?,'staging',?,NULL,NULL)",(generation,directory,os.getpid()))
    root=_safe(work.path,directory); root.mkdir()
    work.admit(64*MIB)
    database=root/"dataset.sqlite3"
    out=sqlite3.connect(database)
    out.execute("PRAGMA synchronous=FULL")
    out.executescript("""CREATE TABLE examples(state BLOB PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL,shard TEXT NOT NULL,offset INTEGER NOT NULL);
    CREATE TABLE families(id TEXT PRIMARY KEY,split TEXT,source TEXT,seed INTEGER,body TEXT,hash TEXT);
    CREATE TABLE exclusions(kind TEXT,state BLOB,split TEXT,PRIMARY KEY(kind,state));
    CREATE TABLE meta(key TEXT PRIMARY KEY,body TEXT,hash TEXT);""")
    rows=work.db.execute("SELECT * FROM labels ORDER BY state").fetchall()
    strata=sorted({json.loads(r["body"])["stratum"] for r in rows})
    files=[]; semantic=hashlib.sha256(); split_counts=Counter(); source_counts=Counter(); kind_counts=Counter()
    try:
        for batch in range(0,len(rows),1000):
            if cancel and cancel(): raise Interrupted()
            filename=f"rows-{batch//1000:04d}.cgp"; file=root/filename
            with file.open("xb") as stream:
                for index,row in enumerate(rows[batch:batch+1000]):
                    record=json.loads(row["body"])
                    if config.digest(record)!=row["hash"]: raise PilotError("Corrupt pending label","corrupt_saved_data")
                    validate_record(record,work.identity,native,oracle)
                    stream.write(_packed(record,strata.index(record["stratum"])))
                    out.execute("INSERT INTO examples VALUES(?,?,?,?,?)",(row["state"],row["body"],row["hash"],filename,index*ROW.size))
                    stable={**record,"label":{**record["label"],"work":{}}}
                    semantic.update((config.canonical(stable)+"\n").encode())
                    split_counts[record["split"]]+=1; kind_counts[record["label"]["kind"]]+=1
                    source=work.db.execute("SELECT source FROM families WHERE id=?",(record["family"],)).fetchone()[0]; source_counts[source]+=1
                stream.flush(); os.fsync(stream.fileno())
            files.append({"name":filename,"size":file.stat().st_size,"sha256":_sha(file)})
            if on_progress: on_progress({"stage":"publish","rows_written":min(batch+1000,len(rows))})
        with out:
            for row in work.db.execute("SELECT * FROM families WHERE id IN (SELECT family FROM selected)"):
                out.execute("INSERT INTO families VALUES(?,?,?,?,?,?)",tuple(row))
            for row in work.db.execute("SELECT * FROM exclusions"): out.execute("INSERT INTO exclusions VALUES(?,?,?)",tuple(row))
            for key,value in (("identity",work.identity),("selection",work.get("selection")),("strata",strata)):
                out.execute("INSERT INTO meta VALUES(?,?,?)",(key,config.canonical(value),config.digest(value)))
        if out.execute("PRAGMA integrity_check").fetchone()[0]!="ok": raise PilotError("Published index integrity failed")
    finally: out.close()
    # Windows _commit requires a write-capable descriptor even after SQLite closed.
    with database.open("r+b") as stream: os.fsync(stream.fileno())
    files.append({"name":database.name,"size":database.stat().st_size,"sha256":_sha(database)})
    progress=work.get("progress")
    manifest={"schema":SCHEMA,"identity":work.identity,"generation":generation,"directory":directory,"records":len(rows),"record_bytes":ROW.size,
              "semantic_sha256":semantic.hexdigest(),"files":files,"splits":dict(split_counts),"sources":dict(source_counts),"label_kinds":dict(kind_counts),
              "strata":strata,"selection":work.get("selection"),"generation_statistics":progress,
              "oracle_isolation":"active-seed<=4 benchmark-only; train teacher has no oracle leaves",
              "sampler":"uniform-primary-stratum/independent-with-replacement/v1","q_equals_P":True,"protected":True}
    if len(rows)!=work.identity["target"]: raise PilotError("Incomplete dataset cannot be published")
    # Verify every file before the catalog commit. Staging paths never imply completion.
    for entry in files:
        f=_safe(root,entry["name"])
        if _sha(f)!=entry["sha256"]: raise PilotError("Artifact checksum failed before commit")
    if cancel and cancel(): raise Interrupted()
    work.admit(MIB)
    with work.db:
        work.db.execute("UPDATE artifacts SET status='complete',body=?,hash=? WHERE id=?",(config.canonical(manifest),config.digest(manifest),generation))
        work.put("current",manifest)
        progress["stage"]="complete"; work.put("progress",progress)
        # Discard disposable unselected payloads only after a full immutable generation exists.
        work.db.execute("DELETE FROM origins WHERE state NOT IN (SELECT state FROM selected)")
        work.db.execute("DELETE FROM candidates WHERE state NOT IN (SELECT state FROM selected)")
        work.db.execute("DELETE FROM families WHERE id NOT IN (SELECT family FROM selected)")
    work.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    if on_progress: on_progress({"stage":"complete","records":len(rows),"semantic_sha256":manifest["semantic_sha256"]})
    return manifest

def build(path=DEFAULT_PATH,oracle_path=None,target=10000,seed=20261014,family_count=None,native=None,
          on_progress=None,cancel=None,quota_mb=51200,min_free_mb=2048):
    config.verify_rules(); native=native or Native()
    target=config.integer(target,"target",1,10000); seed=config.integer(seed,"seed",0,2**31-1)
    family_count=family_count or max(128,(target+4)//5)
    family_count=config.integer(family_count,"family_count",8,10000)
    oracle=Oracle(oracle_path or ROOT/"data"/"oracle-k4.sqlite3",native=native)
    settings=config.limits({"max_depth":4,"time_ms":0,"node_budget":0,"tt_mb":16,"use_pvs":True})
    teacher={"identity":config.identity(native.build_id),"limits":settings,"oracle_leaf_policy":"disabled","lineage":"untrained-H_v1"}
    identity={"schema":SCHEMA,"rules_digest":config.RULES_DIGEST,"state_encoding":"canonical-player-swap/normalized/terminal-actor0-v1",
              "native_build_sha256":native.build_id,"target":target,"seed":seed,"family_count":family_count,
              "teacher_limits":settings,"teacher_revision":config.digest(teacher),"teacher":teacher,
              "oracle_manifest_sha256":oracle.manifest_sha256,
              "oracle_path":str(Path(oracle_path or ROOT/"data"/"oracle-k4.sqlite3").resolve()),
              "implementation_sha256":config.digest({name:hashlib.sha256((ROOT/"congklak"/name).read_bytes()).hexdigest() for name in ("pilot.py","config.py","native.py","reports.py","oracle.py")})}
    work=None
    try:
        work=Work(path,identity,quota_mb,min_free_mb)
        stage=_validate_progress(work)["stage"]
        for row in work.db.execute("SELECT * FROM families"): _validate_archive(row)
        _check_pending(work,native,oracle)
        if stage=="complete": return load(path)
        if stage=="collect": _collect(work,native,on_progress,cancel)
        if work.get("progress")["stage"]=="select": _select(work,native,on_progress,cancel)
        if work.get("progress")["stage"]=="label": _label(work,native,oracle,on_progress,cancel)
        return _publish(work,native,oracle,on_progress,cancel)
    except (sqlite3.Error,OSError) as exc:
        raise PilotError("Pilot persistence failed: "+str(exc),"storage_error") from exc
    finally:
        if work: work.close()
        oracle.close()

def load(path=DEFAULT_PATH):
    path=Path(path).resolve()
    db=sqlite3.connect((path/"work.sqlite3").as_uri()+"?mode=ro",uri=True)
    try:
        row=db.execute("SELECT body,hash FROM meta WHERE key='current'").fetchone()
        if row is None: raise PilotError("Pilot has no complete committed generation","incomplete_dataset")
        manifest=json.loads(row[0])
        if config.digest(manifest)!=row[1]: raise PilotError("Corrupt pilot catalog pointer","corrupt_saved_data")
        row=db.execute("SELECT status,hash FROM artifacts WHERE id=?",(manifest["generation"],)).fetchone()
        if row is None or row[0]!="complete" or row[1]!=config.digest(manifest): raise PilotError("Uncommitted pilot generation","corrupt_saved_data")
        root=_safe(path,manifest["directory"])
        for entry in manifest["files"]:
            f=_safe(root,entry["name"])
            if not f.is_file() or f.stat().st_size!=entry["size"] or _sha(f)!=entry["sha256"]: raise PilotError("Missing/damaged pilot artifact "+entry["name"],"corrupt_saved_data")
        return manifest
    except (sqlite3.Error,ValueError,KeyError,TypeError) as exc: raise PilotError("Malformed pilot catalog","corrupt_saved_data") from exc
    finally: db.close()

def records(path=DEFAULT_PATH,split=None):
    manifest=load(path); root=_safe(Path(path).resolve(),manifest["directory"])
    db=sqlite3.connect((root/"dataset.sqlite3").as_uri()+"?mode=ro",uri=True)
    try:
        for body,checksum in db.execute("SELECT body,hash FROM examples ORDER BY state"):
            record=json.loads(body)
            if config.digest(record)!=checksum: raise PilotError("Corrupt immutable record","corrupt_saved_data")
            if split is None or record["split"]==split: yield record
    finally: db.close()

def verify(path=DEFAULT_PATH,native=None,relabel=False):
    native=native or Native(); manifest=load(path)
    if manifest["identity"]["native_build_sha256"]!=native.build_id: raise PilotError("Pilot native build differs","incompatible_saved_data")
    root=_safe(Path(path).resolve(),manifest["directory"])
    db=sqlite3.connect((root/"dataset.sqlite3").as_uri()+"?mode=ro",uri=True); db.row_factory=sqlite3.Row
    oracle=Oracle(manifest["identity"]["oracle_path"],native=native)
    semantic=hashlib.sha256(); splits=Counter(); strata=Counter(); families={}; seen=set()
    try:
        for row in db.execute("SELECT * FROM families"):
            archive=_validate_archive(row)
            states=[archive["start"]]; p=states[0]
            for a in archive["moves"]:
                p=native.apply(flatten(p),actor=p["controller"],action=a,total=98,trace=False)["position"]; states.append(p)
            families[row["id"]]=(row["split"],archive,states)
        all_rows=list(db.execute("SELECT * FROM examples ORDER BY state"))
        counts=Counter((json.loads(r["body"])["split"],json.loads(r["body"])["stratum"]) for r in all_rows)
        support={split:len({s for sp,s in counts if sp==split}) for split in SPLITS}
        for row in all_rows:
            r=json.loads(row["body"])
            if config.digest(r)!=row["hash"]: raise PilotError("Bad row checksum")
            validate_record(r,manifest["identity"],native,oracle,relabel=relabel)
            p=config.position(r["state"]); key,canonical,_=canonicalize(p,native)
            if key!=row["state"] or key.hex()!=r["state_key"] or canonical!=p or key in seen: raise PilotError("Noncanonical/duplicate state")
            seen.add(key); family_split_value,archive,states=families[r["family"]]
            if r["split"]!=family_split_value: raise PilotError("Family split leakage")
            original=states[r["step"]]
            k,_,swapped=canonicalize(original,native)
            if k!=key or swapped!=r["source_swapped"]: raise PilotError("Game provenance mismatch")
            active,stratum=_mechanism(p,native)
            if stratum!=r["stratum"]: raise PilotError("Stratum mismatch")
            if r["split"]=="train":
                if active<=4 or r["label"]["kind"]!="finite_choice_horizon": raise PilotError("Oracle ancestry leakage")
                if db.execute("SELECT 1 FROM exclusions WHERE kind='state' AND state=?",(key,)).fetchone(): raise PilotError("Holdout state leakage")
                if db.execute("SELECT 1 FROM exclusions WHERE kind='quotient' AND state=?",(quotient(p),)).fetchone(): raise PilotError("Holdout quotient leakage")
            expectedP=1/(support[r["split"]]*counts[(r["split"],stratum)])
            if r["draw_probability"]!=expectedP or r["label_weight"]!=1 or r["importance_ratio"]!=1: raise PilotError("Incorrect unit weighting/P=q")
            if not 0<r["selection_inclusion_probability"]<=1: raise PilotError("Invalid pilot inclusion probability")
            normal=native.normalize(flatten(p),actor=p["controller"],total=98); legal=normal["legal_actions"]
            if r["legal_mask"]!=sum(1<<a for a in legal): raise PilotError("Illegal action mask")
            l=r["label"]; value=l["target_player0"]; d=p["stores"][0]-p["stores"][1]
            if type(value) is not int or not d-active<=value<=d+active: raise PilotError("Invalid target")
            scores={a["pit"]:a["value_player0"] for a in l["action_scores"]}
            if set(scores)!=set(legal): raise PilotError("Incomplete action scores")
            if legal:
                best=(max if p["controller"]==0 else min)(scores.values())
                if value!=best or l["best_moves"]!=[a for a in legal if scores[a]==best]: raise PilotError("Wrong tied optimal set")
            elif l["best_moves"] or value!=d: raise PilotError("Wrong terminal target")
            if l["kind"]=="exact_optimal_margin" and value%2: raise PilotError("Wrong exact parity")
            file=_safe(root,row["shard"])
            with file.open("rb") as stream:
                stream.seek(row["offset"]); packed=stream.read(ROW.size)
            if packed!=_packed(r,manifest["strata"].index(stratum)): raise PilotError("Packed/index disagreement")
            stable={**r,"label":{**l,"work":{}}}; semantic.update((config.canonical(stable)+"\n").encode())
            splits[r["split"]]+=1; strata[stratum]+=1
        if len(seen)!=manifest["records"] or len(seen)!=manifest["identity"]["target"] or semantic.hexdigest()!=manifest["semantic_sha256"]: raise PilotError("Incomplete/content digest mismatch")
        if dict(splits)!=manifest["splits"]: raise PilotError("Split summary mismatch")
        return {"verified":True,"records":len(seen),"semantic_sha256":semantic.hexdigest(),"splits":dict(splits),"strata":dict(strata),"teacher_relabelled":relabel}
    finally: oracle.close(); db.close()

def sample(path=DEFAULT_PATH,count=32,seed=0,split="train"):
    if split not in SPLITS: raise PilotError("Unknown sample split")
    rows=list(records(path,split)); groups=defaultdict(list)
    for row in rows: groups[row["stratum"]].append(row)
    if not groups: raise PilotError("Requested split is empty")
    rng=random.Random(seed); strata=sorted(groups)
    for _ in range(count):
        s=rng.choice(strata); r=rng.choice(groups[s]); yield r

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("build","verify","status","sample")); p.add_argument("--path",default=str(DEFAULT_PATH))
    p.add_argument("--oracle",default=str(ROOT/"data"/"oracle-k4.sqlite3")); p.add_argument("--target",type=int,default=10000)
    p.add_argument("--seed",type=int,default=20261014); p.add_argument("--families",type=int); p.add_argument("--relabel",action="store_true")
    args=p.parse_args(); stopped=False
    import signal
    def stop(*_):
        nonlocal stopped; stopped=True
    signal.signal(signal.SIGINT,stop)
    if hasattr(signal,"SIGTERM"): signal.signal(signal.SIGTERM,stop)
    def progress(event): print(config.canonical({"event":"pilot_progress",**event}),flush=True)
    try:
        if args.command=="build": result=build(args.path,args.oracle,args.target,args.seed,args.families,on_progress=progress,cancel=lambda:stopped)
        elif args.command=="verify": result=verify(args.path,relabel=args.relabel)
        elif args.command=="status": result=load(args.path)
        else: result={"samples":[{"state_key":r["state_key"],"P":r["draw_probability"],"label_weight":r["label_weight"]} for r in sample(args.path,seed=args.seed)]}
        print(config.canonical({"event":"pilot_done","ok":True,"result":result}),flush=True); return 0
    except Exception as exc:
        print(config.canonical({"event":"pilot_error","ok":False,"error":{"code":getattr(exc,"code","pilot_error"),"message":str(exc)}}),flush=True); return 2

if __name__=="__main__": raise SystemExit(main())
