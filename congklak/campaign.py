"""Depth-six corpus with imported lineage exclusions and targeted ordinary states."""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import random
import signal
import sqlite3
import time
from . import config,pilot
from .native import ROOT,Native,flatten
from .oracle import Oracle

DEFAULT=ROOT/"data/stronger-10000-v2"

def imports(parent=pilot.DEFAULT_PATH):
    manifest=pilot.load(parent); root=Path(parent).resolve()/manifest["directory"]
    db=sqlite3.connect((root/"dataset.sqlite3").as_uri()+"?mode=ro",uri=True)
    reserved={}; families={}
    rank={s:i for i,s in enumerate(pilot.SPLITS)}
    def reserve(kind,key,split):
        old=reserved.get((kind,key),"train")
        reserved[(kind,key)]=max(old,split,key=rank.get)
    try:
        # Metadata only: old sealed targets are never returned to this generator.
        for key,split in db.execute("SELECT state,json_extract(body,'$.split') FROM examples ORDER BY state"):
            key=bytes(key); reserve("state",key,split)
            q=pilot.quotient(pilot.from_key(key)); reserve("quotient",q,split)
        for kind,key,split in db.execute("SELECT kind,state,split FROM exclusions"):
            key=bytes(key); reserve(kind,key,split)
            if kind=="state": reserve("quotient",pilot.quotient(config.position(pilot.from_key(key))),split)
        for family,split in db.execute("SELECT id,split FROM families"): families[family]=split
    finally: db.close()
    # Collection IDs remain permanent even when unselected archive payloads
    # have been pruned after publication.
    ledger=sqlite3.connect((Path(parent).resolve()/"work.sqlite3").as_uri()+"?mode=ro",uri=True)
    try:
        units=list(ledger.execute("SELECT idx,family,archive_hash,hash FROM collection_units ORDER BY idx"))
        pilot._need(len(units)==manifest["identity"]["family_count"],"Prior family ledger is incomplete")
        for expected,(index,family,archive_hash,digest) in enumerate(units):
            pilot._need(index==expected and config.digest({"index":index,"family":family,"archive_hash":archive_hash})==digest,"Prior family ledger is corrupt")
            if family.startswith("constructed-k4/"):
                q=bytes.fromhex(family.split("/")[1]); split="development" if hashlib.sha256(q).digest()[0]<128 else "sealed_final_test"
            else: split=pilot.family_split(family)
            if family in families: pilot._need(families[family]==split,"Prior family ownership changed")
            families[family]=split
    finally: ledger.close()
    fingerprint=config.digest({"manifest":config.digest(manifest),"reserved":[[k,s.hex(),v] for (k,s),v in sorted(reserved.items())],"families":families})
    return manifest,reserved,families,fingerprint

def ordinary(index,seed,native):
    rng=random.Random(seed+index*104729); lo,hi=((5,24),(25,49),(50,74))[(index//4)%3]
    for attempt in range(8192):
        active=rng.randint(lo,hi); bars=sorted(rng.sample(range(active+13),13)); counts=[]; previous=-1
        for b in bars+[active+13]: counts.append(b-previous-1); previous=b
        bank=rng.randrange(99-active)
        p={"pits":[counts[:7],counts[7:]],"stores":[bank,98-active-bank],"controller":rng.randrange(2)}
        _,p,_=pilot.canonicalize(p,native)
        if pilot._mechanism(p,native)[1].endswith("/ordinary"):
            family=f"constructed-ordinary/{seed}/{index}"
            return family,pilot.family_split(family),"constructed_ordinary",seed+index*104729,p,[],"counterfactual_valid_stable",[(0,p)],"counterfactual"
    raise pilot.PilotError("Declared ordinary-state generation work bound exhausted")

def collect(work,native,reserved,old_families,event,cancel):
    progress=work.get("progress")
    for index in range(progress["cursor"],work.identity["family_count"]):
        if cancel and cancel(): raise pilot.Interrupted()
        before=time.monotonic()
        result=ordinary(index,work.identity["seed"],native) if index%4==3 else pilot._family(index,work.identity["seed"],native)
        family,split,source,seed,start,moves,reachable,states,status=result
        archive={"start":start,"moves":moves,"reachability":reachable,"status":status,"generator":"legal-prefix-v1","source":source,"seed":seed}
        observations=[]; withheld=0
        if family not in old_families:
            for step,p in states:
                key,p,swapped=pilot.canonicalize(p,native); active,stratum=pilot._mechanism(p,native)
                if active<=4 or ("state",key) in reserved or ("quotient",pilot.quotient(p)) in reserved:
                    withheld+=1; continue
                observations.append((key,p,swapped,step,active,stratum))
        else: withheld=len(states)
        work.admit(pilot.MIB)
        with work.db:
            body=config.canonical(archive); digest=config.digest(archive)
            work.db.execute("INSERT INTO families VALUES(?,?,?,?,?,?)",(family,split,source,seed,body,digest))
            unit={"index":index,"family":family,"archive_hash":digest}
            work.db.execute("INSERT INTO collection_units VALUES(?,?,?,?)",(index,family,digest,config.digest(unit)))
            for key,p,swapped,step,active,stratum in observations:
                added=work.db.execute("INSERT OR IGNORE INTO candidates VALUES(?,?,?,?)",(key,config.canonical(p),stratum,active)).rowcount
                if not added: progress["duplicates"]+=1
                work.db.execute("INSERT OR IGNORE INTO origins VALUES(?,?,?,?)",(key,family,step,int(swapped)))
            progress["cursor"]=index+1; progress["observations"]+=len(states)
            progress["prior_lineage_or_endgame_withheld"]=progress.get("prior_lineage_or_endgame_withheld",0)+withheld
            progress["elapsed_seconds"]+=time.monotonic()-before
            pilot._need(work.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]<=100000,"Candidate capacity exceeded")
            work.put("progress",progress)
        if event and index%100==0: event({**progress,"candidates":work.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]})
    with work.db: progress["stage"]="select"; work.put("progress",progress)

def select(work,native,reserved,event,cancel):
    if cancel and cancel(): raise pilot.Interrupted()
    rank={s:i for i,s in enumerate(pilot.SPLITS)}; claims=defaultdict(list); buckets=defaultdict(list)
    for row in work.db.execute("SELECT o.*,f.split FROM origins o JOIN families f ON f.id=o.family ORDER BY o.family"):
        claims[bytes(row["state"])].append(dict(row))
    rows=work.db.execute("SELECT * FROM candidates ORDER BY state").fetchall(); qowners={}; owners={}
    for row in rows:
        key=bytes(row["state"]); split=max((o["split"] for o in claims[key]),key=rank.get); owners[key]=split
        q=pilot.quotient(json.loads(row["position"])); qowners[q]=max(split,qowners.get(q,"train"),key=rank.get)
    with work.db:
        work.db.execute("DELETE FROM selected"); work.db.execute("DELETE FROM exclusions")
        for (kind,key),split in reserved.items(): work.db.execute("INSERT INTO exclusions VALUES(?,?,?)",(kind,key,split))
        for row in rows:
            key=bytes(row["state"]); p=json.loads(row["position"]); q=pilot.quotient(p); split=owners[key]
            if split!="train": work.db.execute("INSERT OR REPLACE INTO exclusions VALUES('state',?,?)",(key,split))
            if qowners[q]!="train": work.db.execute("INSERT OR REPLACE INTO exclusions VALUES('quotient',?,?)",(q,qowners[q]))
            # A new holdout alias cannot enter train, or exposed development
            # when its equivalent belongs to the new sealed split.
            if rank[split]<rank[qowners[q]] or ("state",key) in reserved or ("quotient",q) in reserved: continue
            origin=min((o for o in claims[key] if o["split"]==split),key=lambda o:(o["family"],o["step"]))
            priority=hashlib.sha256(str(work.identity["seed"]).encode()+key).digest()
            buckets[(split,row["stratum"])].append((priority,key,origin))
        for values in buckets.values(): values.sort(key=lambda v:v[0])
        target=work.identity["target"]; targets={"train":target*8//10,"development":target//10,"sealed_final_test":target-target*8//10-target//10}
        quotas={k:0 for k in sorted(buckets)}
        for split,count in targets.items():
            available=[k for k in quotas if k[0]==split]
            pilot._need(sum(len(buckets[k]) for k in available)>=count,"Insufficient split candidates; a new campaign needs more families")
            for _ in range(count):
                key=min((k for k in available if quotas[k]<len(buckets[k])),key=lambda k:(quotas[k],k)); quotas[key]+=1
        for (split,stratum),count in quotas.items():
            values=buckets[(split,stratum)]
            for _,key,o in values[:count]:
                work.db.execute("INSERT INTO selected VALUES(?,?,?,?,?,?,?)",(key,o["family"],o["step"],o["swapped"],split,stratum,count/len(values)))
        work.put("selection",{"method":"balanced-per-split-stratum/seeded-uniform-without-replacement-v2",
                             "quotas":{sp+":"+st:v for (sp,st),v in quotas.items()},
                             "eligible_counts":{sp+":"+st:len(v) for (sp,st),v in buckets.items()},"imported_exclusion_count":len(reserved)})
        progress=work.get("progress"); progress["stage"]="label"; work.put("progress",progress)
    if event: event(progress)

def build(path=DEFAULT,parent=pilot.DEFAULT_PATH,target=10000,seed=20261015,families=8000,on_event=None,cancel=None):
    config.integer(target,"target",100,10000); config.integer(families,"families",128,10000)
    native=Native(); previous,reserved,old_families,fingerprint=imports(parent)
    oracle=Oracle(native=native); work=None
    settings=config.limits({"max_depth":6,"time_ms":0,"node_budget":0,"tt_mb":16,"use_pvs":True})
    teacher={"identity":config.identity(native.build_id),"limits":settings,"oracle_leaf_policy":"disabled","lineage":"untrained-H_v1"}
    identity={"schema":"stronger-corpus-v2","rules_digest":config.RULES_DIGEST,"state_encoding":"canonical-player-swap/normalized/terminal-actor0-v1",
              "native_build_sha256":native.build_id,"target":target,"seed":seed,"family_count":families,"teacher_limits":settings,
              "teacher":teacher,"teacher_revision":config.digest(teacher),"oracle_manifest_sha256":oracle.manifest_sha256,
              "oracle_path":str(ROOT/"data/oracle-k4.sqlite3"),"parent_manifest_sha256":config.digest(previous),
              "parent_path":str(Path(parent).resolve()),"imported_registry_sha256":fingerprint,
              "implementation_sha256":config.digest({name:pilot._sha(ROOT/"congklak"/name) for name in ("campaign.py","pilot.py","config.py","native.py","reports.py")})}
    try:
        work=pilot.Work(path,identity); stage=pilot._validate_progress(work)["stage"]
        for row in work.db.execute("SELECT * FROM families"): pilot._validate_archive(row)
        for row in work.db.execute("SELECT s.state,c.position FROM selected s JOIN candidates c ON c.state=s.state"):
            p=json.loads(row["position"])
            pilot._need(("state",bytes(row["state"])) not in reserved and ("quotient",pilot.quotient(p)) not in reserved,"Imported lineage leaked")
        if stage=="complete": return pilot.load(path)
        pilot._check_pending(work,native,oracle)
        if stage=="collect": collect(work,native,reserved,old_families,on_event,cancel)
        if work.get("progress")["stage"]=="select": select(work,native,reserved,on_event,cancel)
        if work.get("progress")["stage"]=="label": pilot._label(work,native,oracle,on_event,cancel)
        return pilot._publish(work,native,oracle,on_event,cancel)
    finally:
        if work: work.close()
        oracle.close()

def verify(path=DEFAULT,parent=pilot.DEFAULT_PATH,relabel=False):
    result=pilot.verify(path,relabel=relabel); manifest=pilot.load(path)
    _,reserved,families,fingerprint=imports(parent)
    pilot._need(manifest["identity"]["imported_registry_sha256"]==fingerprint,"Imported registry changed")
    root=Path(path)/manifest["directory"]; db=sqlite3.connect((root/"dataset.sqlite3").resolve().as_uri()+"?mode=ro",uri=True)
    try:
        for key,split,family in db.execute("SELECT state,json_extract(body,'$.split'),json_extract(body,'$.family') FROM examples"):
            p=pilot.from_key(bytes(key))
            pilot._need(("state",bytes(key)) not in reserved and ("quotient",pilot.quotient(p)) not in reserved and family not in families,"Cross-generation ancestry overlap")
    finally: db.close()
    return {**result,"imported_registry_verified":True,"prior_sealed_targets_read":False}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("operation",choices=("build","verify","status"))
    p.add_argument("--path",default=str(DEFAULT)); p.add_argument("--parent",default=str(pilot.DEFAULT_PATH))
    p.add_argument("--target",type=int,default=10000); p.add_argument("--seed",type=int,default=20261015); p.add_argument("--families",type=int,default=8000)
    p.add_argument("--relabel",action="store_true"); a=p.parse_args(); stopped=False
    def stop(*_):
        nonlocal stopped; stopped=True
    signal.signal(signal.SIGINT,stop)
    if hasattr(signal,"SIGTERM"): signal.signal(signal.SIGTERM,stop)
    def event(e): print(config.canonical({"event":"campaign_progress",**e}),flush=True)
    try:
        if a.operation=="build": r=build(a.path,a.parent,a.target,a.seed,a.families,event,lambda:stopped)
        elif a.operation=="verify": r=verify(a.path,a.parent,a.relabel)
        else: r=pilot.load(a.path)
        print(config.canonical({"event":"campaign_done","ok":True,"result":r}),flush=True); return 0
    except Exception as exc:
        print(config.canonical({"event":"campaign_error","ok":False,"error":str(exc)}),flush=True); return 2
if __name__=="__main__": raise SystemExit(main())
