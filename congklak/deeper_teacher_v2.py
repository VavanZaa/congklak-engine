"""Root-preserving H8 relabelling and relay features with FULL/WAL recovery.

Only corrected training/development bodies are fetched. H6 references and H8
labels remain distinct; no child or sealed target enters this experiment.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import time
from . import config,pilot,campaign,features_v3,reports,native as native_binding,deeper_teacher as legacy
from .native import ROOT,Native,flatten

SCHEMA="root-preserving-H8-relay-corpus-v2"
DEFAULT=ROOT/"data/deeper-9000-v2"
GIB=1024**3

class DeeperError(RuntimeError): pass

def require(ok,message):
    if not ok: raise DeeperError(message)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path,value):
    import uuid
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("x",encoding="utf-8") as stream:
            stream.write(config.canonical(value)+"\n"); stream.flush(); os.fsync(stream.fileno())
        for attempt in range(12):
            try: os.replace(temporary,path); break
            except PermissionError:
                if attempt==11: raise
                time.sleep(.05*(attempt+1))
    finally:
        if temporary.exists(): temporary.unlink()

def roots(source=campaign.DEFAULT):
    """Validate exact corrected lineage without returning sealed bodies."""
    source=Path(source).resolve(); manifest=pilot.load(source)
    require(not (source/"quarantine.json").exists() and manifest["identity"]["schema"]=="stronger-corpus-v2","Wrong/quarantined parent corpus")
    native=Native(); require(manifest["identity"]["native_build_sha256"]==native.build_id,"Original teacher DLL changed")
    _,reserved,old_families,registry=campaign.imports(manifest["identity"]["parent_path"])
    require(registry==manifest["identity"]["imported_registry_sha256"],"Imported lineage changed")
    database=source/manifest["directory"]/"dataset.sqlite3"
    db=sqlite3.connect(database.as_uri()+"?mode=ro",uri=True); db.execute("BEGIN"); groups={}
    try:
        for split,expected in (("train",8000),("development",1000)):
            rows=[]
            for key,body,digest in db.execute("SELECT state,body,hash FROM examples WHERE json_extract(body,'$.split')=? ORDER BY state",(split,)):
                row=json.loads(body); require(config.digest(row)==digest,"Damaged parent row")
                require(row["split"]==split and row["state_key"]==key.hex(),"Parent ownership/key mismatch")
                require(row["rules_digest"]==config.RULES_DIGEST and row["teacher_revision"]==manifest["identity"]["teacher_revision"],"Parent teacher identity mismatch")
                require(row["label_weight"]==1 and row["importance_ratio"]==1,"Unexpected weighting")
                p=config.position(row["state"]); canonical_key,canonical,_=pilot.canonicalize(p,native)
                require(canonical_key==key and canonical==p,"Noncanonical parent root")
                require(sum(map(sum,p["pits"]))>4,"Endgame/oracle ancestry")
                require(("state",key) not in reserved and ("quotient",pilot.quotient(p)) not in reserved and row["family"] not in old_families,"Prior lineage leaked")
                validate_label(row["label"],p,row["legal_mask"],6,native)
                if split=="train":
                    for kind,value in (("state",key),("quotient",pilot.quotient(p))):
                        require(db.execute("SELECT 1 FROM exclusions WHERE kind=? AND state=?",(kind,value)).fetchone() is None,"Holdout leaked")
                rows.append(row)
            require(len(rows)==expected,"Parent root count changed"); groups[split]=rows
    finally: db.close()
    counts=Counter(row["stratum"] for row in groups["train"])
    for row in groups["train"]: require(row["draw_probability"]==1/(len(counts)*counts[row["stratum"]]),"Sampling P != q")
    require(not ({pilot.quotient(r["state"]) for r in groups["train"]}&{pilot.quotient(r["state"]) for r in groups["development"]}),"Current holdout quotient leak")
    return manifest,groups,{s:config.digest([r["state_key"] for r in rows]) for s,rows in groups.items()}

def validate_label(label,p,mask,depth,native):
    require(isinstance(label,dict) and set(label)=={"kind","target_player0","best_moves","action_scores","choice_depth","oracle_manifest_sha256","coverage","work"},"Wrong teacher label fields")
    require(label["kind"]=="finite_choice_horizon" and label["choice_depth"]==depth and type(label["choice_depth"]) is int and label["oracle_manifest_sha256"] is None,"Wrong/incomplete teacher label")
    normal=native.normalize(flatten(p),actor=p["controller"],total=98)
    legal=normal["legal_actions"]; require(not normal["terminal"] and type(mask) is int and mask==sum(1<<a for a in legal),"Label legal coverage mismatch")
    scores=label["action_scores"]; require(isinstance(scores,list) and [r["pit"] for r in scores]==legal,"Action-Q coverage/order mismatch")
    require(isinstance(label["best_moves"],list) and all(type(a) is int for a in label["best_moves"]),"Invalid optimum action types")
    reports.coverage(label["coverage"],depth)
    require(isinstance(label["work"],dict) and set(label["work"])=={"nodes","deposits","elapsed_ms","peak_process_bytes"} and all(type(x) is int and x>=0 for x in label["work"].values()),"Invalid teacher work fields")
    d=p["stores"][0]-p["stores"][1]; active=sum(map(sum,p["pits"]))
    for item in scores:
        require(set(item)=={"pit","value_player0"} and type(item["pit"]) is int and type(item["value_player0"]) is int and d-active<=item["value_player0"]<=d+active,"Invalid action Q")
    best=(max if normal["normalized_controller"]==0 else min)(r["value_player0"] for r in scores)
    require(type(label["target_player0"]) is int and label["target_player0"]==best and label["best_moves"]==[r["pit"] for r in scores if r["value_player0"]==best],"Invalid root label ranking")

def identity(source=campaign.DEFAULT):
    from . import features_relay
    manifest,groups,fingerprints=roots(source); native=Native()
    modules=(config,pilot,campaign,features_v3,reports,native_binding,features_relay)
    value={"schema":SCHEMA,"rules_digest":config.RULES_DIGEST,"parent_path":str(Path(source).resolve()),"parent_semantic_sha256":manifest["semantic_sha256"],
           "parent_manifest_sha256":config.digest(manifest),"split_fingerprints":fingerprints,"root_rows_sha256":config.digest(groups),
           "teacher":"H_v1/untrained/oracle-free/exact-completed-root-depth8/full-root-Q", "choice_depth":8,"action_successor_horizon":7,
           "native_build_sha256":native.build_id,"teacher_source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"native/engine.hpp",ROOT/"native/rules.cpp",ROOT/"native/search.cpp")},
           "source_sha256":sha(__file__),"source_dependencies_sha256":{m.__name__:sha(m.__file__) for m in modules},
           "splits":{"train":8000,"development":1000},"sealed_final_test_consumed":False,"new_successor_rows":0,
           "search":{"max_depth":8,"time_ms":0,"node_budget":0,"memory_mb":512,"tt_mb":16,"use_pvs":True},
           "feature_count":127,"feature_encoder":features_relay.ENCODER127,"quota_bytes":50*GIB,"reserve_bytes":2*GIB,
           "work_snapshot":"Exact completed-depth report; outer analyze-return timing excluded",
           "migration_source":migration_snapshot()}
    return value,groups

class Work:
    def __init__(self,path,ident):
        self.path=Path(path).resolve(); self.path.mkdir(parents=True,exist_ok=True); self.identity=ident; self.db=None; self.lock=None
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
            self.admit(8*1024**2); self.db=sqlite3.connect(self.path/"work.sqlite3")
            require(self.db.execute("PRAGMA quick_check").fetchone()[0]=="ok","Corrupt deeper work database")
            self.db.execute("PRAGMA journal_mode=WAL"); self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,body TEXT,hash TEXT); CREATE TABLE IF NOT EXISTS roots(state TEXT PRIMARY KEY,body TEXT,hash TEXT); CREATE TABLE IF NOT EXISTS labels(state TEXT PRIMARY KEY,body TEXT,hash TEXT); CREATE TABLE IF NOT EXISTS features(state TEXT PRIMARY KEY,body TEXT,hash TEXT);")
            saved=self.get("identity"); require(saved is None or saved==ident,"Deeper identity differs; use a versioned path")
            if saved is None:
                with self.db: self.put("identity",ident)
        except Exception: self.close(); raise
    def admit(self,estimate):
        physical=sum(f.stat().st_size for f in (ROOT/"data").rglob("*") if f.is_file())
        if not self.path.is_relative_to(ROOT/"data"): physical+=sum(f.stat().st_size for f in self.path.rglob("*") if f.is_file())
        require(physical+estimate+self.identity["reserve_bytes"]<self.identity["quota_bytes"],"Managed quota/recovery reserve exceeded")
        require(shutil.disk_usage(self.path).free>estimate+self.identity["reserve_bytes"],"Free-space/recovery reserve exceeded")
    def get(self,key):
        row=self.db.execute("SELECT body,hash FROM meta WHERE key=?",(key,)).fetchone()
        if not row: return None
        value=json.loads(row[0]); require(config.digest(value)==row[1],"Corrupt deeper metadata "+key); return value
    def put(self,key,value): self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?,?)",(key,config.canonical(value),config.digest(value)))
    def close(self):
        if self.db is not None: self.db.close(); self.db=None
        if self.lock is not None: self.lock.close(); self.lock=None

def checked_table(db,name):
    rows={}
    for key,body,digest in db.execute("SELECT state,body,hash FROM "+name+" ORDER BY state"):
        value=json.loads(body); require(config.digest(value)==digest,"Corrupt "+name+" row"); rows[key]=value
    return rows

def checked_labels(db,expected,native):
    """Replay one report at a time; retain only compact supervision."""
    values={}
    for key,body,digest in db.execute("SELECT state,body,hash FROM labels ORDER BY state"):
        item=json.loads(body); require(config.digest(item)==digest and key in expected,"Corrupt/unknown H8 label")
        row=expected[key]; require(item["root_sha256"]==config.digest(row),"H8 label root mismatch")
        validate_label(item["label"],row["state"],row["legal_mask"],8,native); validate_link(item,row,native)
        values[key]={"root_sha256":item["root_sha256"],"label":item["label"]}
    return values

def generate(path=DEFAULT,source=campaign.DEFAULT,on_event=None,cancel=None,label_limit=None):
    from . import features_relay
    require(label_limit is None or type(label_limit) is int and label_limit>0,"Label limit must be positive integer")
    ident,groups=identity(source); all_rows=sorted(groups["train"]+groups["development"],key=lambda r:r["state_key"]); expected={r["state_key"]:r for r in all_rows}
    work=Work(path,ident); native=Native(); started=time.monotonic()
    try:
        existing=checked_table(work.db,"roots")
        require(not (set(existing)-set(expected)) and all(expected[k]==v for k,v in existing.items()),"Saved roots changed")
        with work.db:
            for key,row in expected.items(): work.db.execute("INSERT OR IGNORE INTO roots VALUES(?,?,?)",(key,config.canonical(row),config.digest(row)))
        saved_labels=checked_labels(work.db,expected,native); saved_features=checked_table(work.db,"features")
        require(set(saved_labels)<=set(expected) and set(saved_features)<=set(expected),"Saved labels/features escape roots")
        for key,item in saved_features.items(): validate_features(item,expected[key],ident)
        if on_event: on_event({"event":"deeper_started","roots":len(expected),"restored_labels":len(saved_labels),"restored_features":len(saved_features),"identity":ident})
        added=0
        for row in all_rows:
            if cancel and cancel(): break
            key=row["state_key"]; p=row["state"]
            if key not in saved_features:
                before=time.monotonic(); x,d,r=features_relay.encode(p,127)
                item={"root_sha256":config.digest(row),"encoder":ident["feature_encoder"],"x":x,"d":d,"r":r,"build_seconds":time.monotonic()-before}
                validate_features(item,row,ident); work.admit(65536)
                with work.db: work.db.execute("INSERT INTO features VALUES(?,?,?)",(key,config.canonical(item),config.digest(item)))
                saved_features[key]=item
            if key not in saved_labels:
                result=native.analyze(flatten(p),actor=p["controller"],total=98,keep_history=False,**ident["search"])
                report=result.get("report")
                require(result["stop_reason"]=="max_depth" and report is not None and report["choice_depth"]==8,"Incomplete H8 target is not admitted")
                reports.validate(report,p,config.identity(native.build_id),native)
                label={"kind":"finite_choice_horizon","choice_depth":8,"oracle_manifest_sha256":None,"target_player0":report["depth_result"]["value_player0"],"best_moves":report["best_moves"],
                       "action_scores":[{"pit":c["pit"],"value_player0":c["depth_result"]["value_player0"]} for c in report["candidates"]],"coverage":report["coverage"],"work":{k:report[k] for k in ("nodes","deposits","elapsed_ms","peak_process_bytes")}}
                validate_label(label,p,row["legal_mask"],8,native)
                item={"root_sha256":config.digest(row),"label":label,"report":report}; work.admit(1024**2)
                if on_event: on_event({"event":"deeper_label_ready","state_key":key,"committed_labels":len(saved_labels)})
                with work.db: work.db.execute("INSERT INTO labels VALUES(?,?,?)",(key,config.canonical(item),config.digest(item)))
                saved_labels[key]={"root_sha256":item["root_sha256"],"label":item["label"]}; added+=1
                if on_event: on_event({"event":"deeper_label_committed","labels":len(saved_labels),"features":len(saved_features),"target":len(expected),"state_key":key,"elapsed_seconds_this_invocation":time.monotonic()-started})
            if label_limit is not None and added>=label_limit: break
        if len(saved_labels)==len(expected) and len(saved_features)==len(expected):
            semantic=config.digest([{ "root":row,"H8":{**saved_labels[row["state_key"]]["label"],"work":{}},"features":{k:v for k,v in saved_features[row["state_key"]].items() if k!="build_seconds"}} for row in all_rows])
            manifest={"schema":SCHEMA,"identity":ident,"rows":len(expected),"semantic_sha256":semantic,"feature_build_seconds":sum(x["build_seconds"] for x in saved_features.values()),"teacher_elapsed_ms":sum(x["label"]["work"]["elapsed_ms"] for x in saved_labels.values()),"sealed_final_test_consumed":False}
            with work.db: work.put("complete",manifest)
            atomic_json(Path(path)/"manifest.json",manifest)
        else: manifest=None
        return {"status":"complete" if manifest else "interrupted","labels":len(saved_labels),"features":len(saved_features),"target":len(expected),"manifest":manifest,"restored_labels":len(saved_labels)-added}
    finally: work.close()

@lru_cache(maxsize=9000)
def expected_features(canonical_position):
    from . import features_relay
    x,d,r=features_relay.encode(json.loads(canonical_position),127)
    return tuple(x),d,r

def validate_features(item,row,ident):
    require(set(item)=={"root_sha256","encoder","x","d","r","build_seconds"},"Invalid feature fields")
    require(item["root_sha256"]==config.digest(row) and item["encoder"]==ident["feature_encoder"],"Feature root/source mismatch")
    require(isinstance(item["x"],list) and len(item["x"])==127 and all(type(x) in (int,float) and math.isfinite(x) for x in item["x"]),"Invalid relay feature tensor")
    x,d,r=features_v3.encode(row["state"],43)
    require(item["x"][:43]==x and type(item["d"]) is int and type(item["r"]) is int and item["d"]==d and item["r"]==r,"Feature prefix/offset/scale differs")
    full_x,full_d,full_r=expected_features(config.canonical(row["state"]))
    require(tuple(item["x"])==full_x and item["d"]==full_d and item["r"]==full_r,"Relay feature derivation differs")
    require(type(item["build_seconds"]) in (int,float) and math.isfinite(item["build_seconds"]) and item["build_seconds"]>=0,"Invalid feature timing")

def validate_link(item,row,native):
    report=item["report"]; label=item["label"]
    reports.validate(report,row["state"],config.identity(native.build_id),native)
    require(report["choice_depth"]==8 and label["target_player0"]==report["depth_result"]["value_player0"] and label["best_moves"]==report["best_moves"] and label["action_scores"]==[{"pit":c["pit"],"value_player0":c["depth_result"]["value_player0"]} for c in report["candidates"]],"Saved report/label differs")
    require(label["coverage"]==report["coverage"] and label["work"]=={k:report[k] for k in ("nodes","deposits","elapsed_ms","peak_process_bytes")},"Label work/coverage differs from report")

def load(path=DEFAULT,source=campaign.DEFAULT,teacher_depth=8):
    require(type(teacher_depth) is int and teacher_depth in (6,8),"Unknown teacher")
    ident,groups=identity(source); path=Path(path).resolve(); manifest=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    require(manifest["identity"]==ident and manifest["schema"]==SCHEMA,"Deeper manifest identity changed")
    db=sqlite3.connect((path/"work.sqlite3").as_uri()+"?mode=ro",uri=True); db.execute("BEGIN")
    try:
        expected={r["state_key"]:r for rows in groups.values() for r in rows}
        roots_saved=checked_table(db,"roots"); labels=checked_labels(db,expected,Native()); features=checked_table(db,"features")
        body,digest=db.execute("SELECT body,hash FROM meta WHERE key='complete'").fetchone(); require(config.digest(json.loads(body))==digest and json.loads(body)==manifest,"Uncommitted/damaged manifest")
    finally: db.close()
    expected={r["state_key"]:r for rows in groups.values() for r in rows}
    require(roots_saved==expected and set(labels)==set(expected)==set(features),"Incomplete/changed root labels/features")
    native=Native(); result={}; semantic_rows=[]
    for split,rows in groups.items():
        result[split]=[]
        for row in rows:
            key=row["state_key"]; item=labels[key]; require(item["root_sha256"]==config.digest(row),"H8 root mismatch")
            validate_label(item["label"],row["state"],row["legal_mask"],8,native); validate_features(features[key],row,ident)
            result[split].append({**row,"H6_reference":row["label"],"H8_reference":item["label"],"label":row["label"] if teacher_depth==6 else item["label"],"cached_features":features[key]})
    for row in sorted(expected.values(),key=lambda r:r["state_key"]):
        key=row["state_key"]; semantic_rows.append({"root":row,"H8":{**labels[key]["label"],"work":{}},"features":{k:v for k,v in features[key].items() if k!="build_seconds"}})
    require(config.digest(semantic_rows)==manifest["semantic_sha256"] and manifest["rows"]==9000,"Deeper semantic mismatch")
    return manifest,result,ident["split_fingerprints"]


def migration_snapshot(source=legacy.DEFAULT):
    """Pin the immutable failed-v1 ledger by committed row keys and digests."""
    source=Path(source).resolve()
    db=sqlite3.connect((source/"work.sqlite3").as_uri()+"?mode=ro",uri=True); db.execute("BEGIN")
    try:
        body,digest=db.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone()
        value=json.loads(body); require(config.digest(value)==digest,"Legacy identity checksum differs")
        tables={}
        for table in ("roots","labels","features"):
            rows=list(db.execute("SELECT state,hash FROM "+table+" ORDER BY state"))
            tables[table]={"rows":len(rows),"key_hash_sha256":config.digest(rows)}
        return {"path":str(source),"identity_sha256":digest,"tables":tables,"legacy_source_sha256":sha(legacy.__file__)}
    finally: db.close()


def migrate(source=legacy.DEFAULT,path=DEFAULT,on_event=None):
    """Copy checked v1 supervision without changing roots, reports or targets.

    V1 measured label timing after serializing its report, but required exact
    equality with the earlier report timestamp on resume. Only that metadata
    field is rebased to its original report. The complete original ledger and
    before/after row digests remain available for independent verification.
    """
    source=Path(source).resolve(); path=Path(path).resolve()
    require(source!=path and source==legacy.DEFAULT.resolve(),"Migration only accepts the preserved original ledger")
    ident,groups=identity(); expected={r["state_key"]:r for rows in groups.values() for r in rows}
    snapshot=migration_snapshot(source); require(snapshot==ident["migration_source"],"Legacy source changed before migration")
    old_ident,_=legacy.identity(); old=sqlite3.connect((source/"work.sqlite3").as_uri()+"?mode=ro",uri=True); old.execute("BEGIN")
    work=Work(path,ident); native=Native(); changes=[]; totals={}
    try:
        body,digest=old.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone()
        require(config.digest(json.loads(body))==digest and json.loads(body)==old_ident,"Legacy frozen generation identity differs")
        root_keys={r[0] for r in old.execute("SELECT state FROM roots")}
        feature_keys={r[0] for r in old.execute("SELECT state FROM features")}
        label_keys={r[0] for r in old.execute("SELECT state FROM labels")}
        require(root_keys==set(expected) and label_keys<=feature_keys<=root_keys,"Legacy incomplete root set or orphan supervision")
        copy_bytes=sum(old.execute("SELECT COALESCE(SUM(length(CAST(body AS BLOB))),0) FROM "+table).fetchone()[0] for table in ("roots","features","labels"))
        work.admit(copy_bytes*2+32*1024**2)
        for table in ("roots","features","labels"):
            count=0
            for key,body,digest in old.execute("SELECT state,body,hash FROM "+table+" ORDER BY state"):
                value=json.loads(body); require(config.digest(value)==digest and key in expected,"Corrupt/unknown legacy "+table)
                row=expected[key]
                if table=="roots": require(value==row,"Legacy root differs")
                elif table=="features": validate_features(value,row,ident)
                else:
                    require(set(value)=={"root_sha256","label","report"} and value["root_sha256"]==config.digest(row),"Legacy label root/fields differ")
                    validate_label(value["label"],row["state"],row["legal_mask"],8,native)
                    report=value["report"]; reports.validate(report,row["state"],config.identity(native.build_id),native)
                    require(report["choice_depth"]==8,"Legacy incomplete horizon")
                    original_work=value["label"]["work"]
                    report_work={k:report[k] for k in ("nodes","deposits","elapsed_ms","peak_process_bytes")}
                    require(all(original_work[k]==report_work[k] for k in original_work if k!="elapsed_ms") and original_work["elapsed_ms"]>=report_work["elapsed_ms"],"Legacy drift escapes return-time timestamp")
                    if original_work!=report_work:
                        changes.append({"state_key":key,"original_row_sha256":digest,"return_elapsed_ms":original_work["elapsed_ms"],"report_elapsed_ms":report_work["elapsed_ms"]})
                    value["label"]["work"]=report_work; validate_link(value,row,native)
                    if original_work!=report_work: changes[-1]["migrated_row_sha256"]=config.digest(value)
                saved=work.db.execute("SELECT body,hash FROM "+table+" WHERE state=?",(key,)).fetchone()
                require(saved is None or (json.loads(saved[0])==value and config.digest(value)==saved[1]),"Previously migrated row differs")
                if saved is None:
                    with work.db: work.db.execute("INSERT INTO "+table+" VALUES(?,?,?)",(key,config.canonical(value),config.digest(value)))
                count+=1
            totals[table]=count
        require(migration_snapshot(source)==snapshot,"Legacy source changed during migration")
        evidence={"schema":"deeper-v1-report-timing-migration-v1","source":snapshot,"destination":str(path),"copied":totals,"adjusted_timing_rows":len(changes),"changes":changes,"roots_reports_targets_features_unchanged":True,"legacy_source_preserved":True}
        previous=work.get("migration"); require(previous is None or previous==evidence,"Migration evidence differs")
        if previous is None:
            with work.db: work.put("migration",evidence)
        evidence_path=path/"migration.json"
        if not evidence_path.exists(): atomic_json(evidence_path,evidence)
        else: require(json.loads(evidence_path.read_text(encoding="utf-8"))==evidence,"Published migration differs")
        if on_event: on_event({"event":"deeper_migration_verified","copied":totals,"adjusted_timing_rows":len(changes),"source":snapshot})
        return evidence
    finally: old.close(); work.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("operation",choices=("migrate","generate","verify")); parser.add_argument("--run",default=str(DEFAULT)); parser.add_argument("--source",default=str(campaign.DEFAULT)); parser.add_argument("--label-limit",type=int)
    args=parser.parse_args(); stopped=False
    def stop(*_):
        nonlocal stopped; stopped=True
    signal.signal(signal.SIGINT,stop)
    if hasattr(signal,"SIGTERM"): signal.signal(signal.SIGTERM,stop)
    def event(value): print(config.canonical(value),flush=True)
    try:
        if args.operation=="migrate": result=migrate(path=args.run,on_event=event)
        elif args.operation=="generate": result=generate(args.run,args.source,event,lambda:stopped,args.label_limit)
        else:
            manifest,groups,_=load(args.run,args.source); result={"status":"verified","rows":{s:len(r) for s,r in groups.items()},"semantic_sha256":manifest["semantic_sha256"]}
        event({"event":"deeper_done",**result}); return 0
    except Exception as exc: event({"event":"deeper_error","error":str(exc)}); return 2

if __name__=="__main__": raise SystemExit(main())
