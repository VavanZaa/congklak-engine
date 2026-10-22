"""Concurrent NDJSON coordinator: one native worker, committed depth checkpoints."""
from __future__ import annotations
import argparse
import json
import signal
import sqlite3
import sys
import threading
from . import config
from .native import ROOT, Native, NativeError, Control, flatten
from .storage import Store, StorageError
from .explanations import explain

MAX_LINE=65536

class Coordinator:
    def __init__(self,native,store):
        self.native=native
        self.store=store
        self.lock=threading.RLock()
        self.output_lock=threading.Lock()
        self.active=None
        self.worker=None
        self.oracle=None

    def emit(self,event,request_id=None,**values):
        message={"event":event,**values}
        if request_id is not None: message["id"]=request_id
        with self.output_lock:
            sys.stdout.write(config.canonical(message)+"\n")
            sys.stdout.flush()

    def start(self,job_id,root,settings,start_depth,request_id):
        with self.lock:
            if self.active is not None:
                raise config.InputError("busy","One analysis worker is already active; stop it or wait for analysis_done")
            root=config.position(root)
            settings=config.limits(settings)
            normal=self.native.normalize(flatten(root),actor=root["controller"],total=98)
            control=Control(self.native,**settings)
            try: run_id=self.store.begin(job_id,settings)
            except Exception:
                control.close()
                raise
            self.active={"job_id":job_id,"control":control}
            self.emit("job_started",request_id,ok=True,job_id=job_id,run_id=run_id,
                      start_depth=start_depth,limits=settings,initial=normal,identity=self.store.identity,
                      candidates=[{"actor":normal["normalized_controller"],"pit":p,"depth_result":None}
                                  for p in normal["legal_actions"]])
            self.worker=threading.Thread(target=self.run,
                args=(job_id,root,settings,start_depth,request_id,run_id,control),name="analysis-worker")
            self.worker.start()

    def run(self,job_id,root,settings,start_depth,request_id,run_id,control):
        problem=None
        result={}
        try:
            def publish(report):
                self.store.commit_report(job_id,report)
                self.emit("iteration",request_id,ok=True,job_id=job_id,report=report,explanations=explain(report))
                return True
            result=self.native.analyze(flatten(root),actor=root["controller"],total=98,
                **settings,start_depth=start_depth,on_report=publish,control=control,keep_history=False)
            reason=result.get("stop_reason","native_error")
            if not result.get("ok"):
                problem=result.get("error",{"message":"Native analysis failed"})
            if "callback_error" in result:
                reason="storage_or_publication_error"
                problem={"code":reason,"message":result["callback_error"]}
            complete=reason=="max_depth" and result.get("report") is not None
            self.store.finish(job_id,run_id,reason,complete=complete,error=bool(problem))
        except Exception as exc:
            reason=getattr(exc,"code","worker_error")
            problem={"code":reason,"message":str(exc)}
            try: self.store.finish(job_id,run_id,reason,error=True)
            except Exception: pass  # A failed final write remains recoverable as running.
        try:
            saved=self.store.load(job_id)
            report=saved["report"]
            depth=saved["last_completed_depth"]
        except Exception as exc:
            report=None; depth=-1
            problem={"code":getattr(exc,"code","storage_error"),"message":str(exc)}
        with self.lock:
            self.active=None
            control.close()
            self.emit("analysis_done",request_id,ok=problem is None,job_id=job_id,
                      stop_reason=reason,error=problem,last_completed_depth=depth,report=report,
                      explanations=explain(report),
                      work={k:result.get(k) for k in ("nodes","deposits","elapsed_ms","peak_process_bytes")})

    def handle(self,req):
        req=config.request(req)
        op=req["op"]
        rid=req.get("id")
        if op in {"validate","apply","analyze","oracle"}:
            if "position" in req and req["position"] is None:
                raise config.InputError("invalid_position","position must be a stable count object; omit it to use the opening")
            root=config.position(req.get("position"))
            normal=self.native.normalize(flatten(root),actor=root["controller"],total=98)
            if op=="oracle":
                from .oracle import Oracle
                if self.oracle is None: self.oracle=Oracle(self.store.path/"oracle-k4.sqlite3",native=self.native)
                self.emit("oracle_result",rid,ok=True,result=self.oracle.lookup(root))
            elif op=="validate":
                self.emit("validated",rid,**normal,identity=self.store.identity)
            elif op=="apply":
                if "action" not in req: raise config.InputError("invalid_action","apply requires action: pit index 0..6 or 'pass'")
                action=req["action"]
                trace=req.get("trace",True)
                if type(trace) is not bool: raise config.InputError("invalid_trace","trace must be boolean")
                if action=="pass":
                    if not normal["passes"]:
                        raise config.InputError("illegal_pass","Pass is legal only for an empty-row controller in a nonterminal position")
                    self.emit("applied",rid,**normal,action="pass",events=[{"type":"pass","actor":p["from"],"to":p["to"]} for p in normal["passes"]])
                else:
                    action=config.integer(action,"action",0,6)
                    result=self.native.apply(flatten(root),actor=root["controller"],action=action,total=98,trace=trace)
                    self.emit("applied",rid,**result)
            else:
                settings=config.limits(req.get("limits"))
                with self.lock:
                    if self.active is not None: raise config.InputError("busy","One analysis worker is already active")
                    job_id=self.store.create(root,settings)
                    self.start(job_id,root,settings,0 if settings["max_depth"]==0 else 1,rid)
        elif op=="stop":
            with self.lock:
                if self.active is None:
                    self.emit("stop_requested",rid,ok=True,active=False)
                else:
                    if req.get("job_id",self.active["job_id"])!=self.active["job_id"]:
                        raise config.InputError("job_not_active","The requested job is not the active analysis")
                    self.active["control"].cancel()
                    self.emit("stop_requested",rid,ok=True,active=True,job_id=self.active["job_id"])
        else:
            saved=self.store.load(req.get("job_id"),compatible=op=="resume")
            if op=="load":
                self.emit("loaded",rid,ok=True,job=saved,explanations=explain(saved["report"]))
            else:
                if "position" in req: raise config.InputError("invalid_resume","Resume uses the saved position; start analyze for a new position")
                base=saved["runs"][-1]["settings"] if saved["runs"] else saved["payload"]["initial_limits"]
                settings=config.limits(req.get("limits"),base)
                if saved["last_completed_depth"]>settings["max_depth"] and not saved["report"]["terminal"]:
                    raise config.InputError("invalid_resume_depth","Requested depth is below the saved horizon; start a new analyze job for that horizon")
                if saved["last_completed_depth"]>=settings["max_depth"] or (saved["report"] and saved["report"]["terminal"]):
                    self.emit("analysis_done",rid,ok=True,job_id=saved["id"],stop_reason="already_complete",
                              last_completed_depth=saved["last_completed_depth"],report=saved["report"],error=None,explanations=explain(saved["report"]))
                else:
                    self.start(saved["id"],saved["payload"]["position"],settings,
                               max(1,saved["last_completed_depth"]+1) if settings["max_depth"] else 0,rid)

    def cancel(self,*_):
        with self.lock:
            if self.active: self.active["control"].cancel()

    def join(self):
        if self.worker: self.worker.join()

def main():
    parser=argparse.ArgumentParser(description="Experimental Congklak NDJSON analyzer; one request per input line")
    parser.add_argument("--data-dir",default=str(ROOT/"data"))
    parser.add_argument("--storage-quota-mb",type=int,default=51200)
    parser.add_argument("--min-free-mb",type=int,default=256)
    args=parser.parse_args()
    store=None
    coord=None
    try:
        config.verify_rules()
        config.integer(args.storage_quota_mb,"storage-quota-mb",1,51200)
        config.integer(args.min_free_mb,"min-free-mb",256,2**40)
        native=Native()
        store=Store(args.data_dir,config.identity(native.build_id),args.storage_quota_mb,args.min_free_mb,native)
        coord=Coordinator(native,store)
        signal.signal(signal.SIGINT,coord.cancel)
        if hasattr(signal,"SIGTERM"): signal.signal(signal.SIGTERM,coord.cancel)
        coord.emit("ready",ok=True,identity=store.identity,data_dir=str(store.path),defaults=config.DEFAULT_LIMITS,
                   search_defaults=config.SEARCH_DEFAULTS,
                   protocol="congklak-ndjson-v1",profile_status="experimental_engineering_contract")
        while True:
            line=sys.stdin.buffer.readline(MAX_LINE+1)
            if not line: break
            if len(line)>MAX_LINE:
                # Bound input memory even for an oversized line with no newline.
                while line and not line.endswith(b"\n"): line=sys.stdin.buffer.readline(MAX_LINE+1)
                coord.emit("error",ok=False,error={"code":"request_too_large","message":"Request exceeds 64 KiB"})
                continue
            rid=None
            try:
                req=json.loads(line.decode("utf-8"),parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))
                rid=req.get("id") if isinstance(req,dict) else None
                if type(rid) not in (str,int) or len(str(rid))>128: rid=None
                coord.handle(req)
            except (config.InputError,StorageError,RuntimeError,ValueError,TypeError,KeyError,RecursionError,sqlite3.Error) as exc:
                coord.emit("error",rid,ok=False,error={"code":getattr(exc,"code","invalid_request"),"message":str(exc)})
        coord.join()
        return 0
    except Exception as exc:
        if coord:
            coord.cancel(); coord.join()
        message={"event":"error","ok":False,"error":{"code":getattr(exc,"code","startup_error"),"message":str(exc)}}
        sys.stdout.write(config.canonical(message)+"\n"); sys.stdout.flush()
        return 2
    finally:
        if coord and coord.oracle: coord.oracle.close()
        if store: store.close()

if __name__=="__main__":
    raise SystemExit(main())
