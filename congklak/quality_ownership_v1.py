"""Bounded PUCT exploration-scale ablation with fixed-player0 backups.

Only selection changes value perspective, using each normalized node actor.
Root expansion is separate from completed edge-backup simulations. An unfinished
simulation charges its transitions/inference but never changes visits or means.
No transpositions, oracle, asynchronous tree work, or proof claims are used.
The only search intervention is the predeclared exploration coefficient.
The c=1.5 control preserves frozen traversal semantics; c=0.25 is a hypothesis,
not a claim that the frozen coefficient is intrinsically incorrect.
"""
from __future__ import annotations
from dataclasses import dataclass,field
import math
import random
import threading
import time
from . import config
from .native import Native,NativeError,Control,flatten
from .phase6_inference import InferenceStopped,validate_prediction

SCHEMA="quality-puct-report-v1"
PROTOCOL="quality/fixed-player0/parent-bootstrap-fpu/legal-softmax/declared-cpuct-v1"
EXPLORATION=(1.5,0.25)

# Share cancellation token identity so frozen callers also cancel native transitions.
from .puct import CancellationToken

@dataclass
class Node:
    position:dict
    actor:int
    legal:tuple
    terminal:bool
    expanded:bool=False
    bootstrap:float=0.0
    priors:list=field(default_factory=lambda:[0.0]*7)
    visits:list=field(default_factory=lambda:[0]*7)
    sums:list=field(default_factory=lambda:[0.0]*7)
    children:dict=field(default_factory=dict)

@dataclass
class Tree:
    root:Node
    control:object
    nodes:int=1
    completed:int=0
    stop:str=""
    rng:object=None
    work:dict=field(default_factory=lambda:{"transitions":0,"deposits":0,"feature_simulations":0,"feature_deposits":0,"neural_evaluations":0,"neural_calls":0,"inference_ms":0.0,"root_expansions":0})

def legal_priors(logits,legal):
    if len(logits)!=7 or any(type(x) not in (int,float) or not math.isfinite(x) for x in logits):raise NativeError("Policy must contain seven finite logits")
    if not legal:return [0.0]*7
    top=max(logits[a] for a in legal);values={a:math.exp(logits[a]-top) for a in legal};total=sum(values.values())
    return [values.get(a,0.0)/total for a in range(7)]

def _node(native,p):
    n=native.normalize(flatten(p),actor=p["controller"],total=98)
    if not n.get("ok"):raise NativeError(str(n))
    q=n["position"];native.trace_mark(q,"puct_node");return Node(q,q["controller"],tuple(n["legal_actions"]),n["terminal"],bootstrap=(q["stores"][0]-q["stores"][1])/98)

def _poll(cancel,deadline):
    stopped=cancel() if callable(cancel) else cancel.is_set() if cancel is not None else False
    if stopped:raise InferenceStopped("cancelled")
    if deadline is not None and time.perf_counter()>=deadline:raise InferenceStopped("time_budget")

def _select(node,c_puct):
    total=sum(node.visits);sign=1 if node.actor==0 else -1
    def score(a):
        q=node.sums[a]/node.visits[a] if node.visits[a] else node.bootstrap
        return sign*q+c_puct*node.priors[a]*math.sqrt(max(1,total))/(1+node.visits[a])
    return max(node.legal,key=lambda a:(score(a),-a))

def _expand(node,prediction):
    validate_prediction(prediction,node.position);node.bootstrap=prediction["value_player0"]/98
    node.priors=legal_priors(prediction["policy_logits"],node.legal);node.expanded=True

def _noise(node,root_noise,rng):
    if root_noise is None:return
    if not isinstance(root_noise,dict) or set(root_noise)!={"alpha","fraction"}:raise ValueError("Root noise requires alpha and fraction")
    alpha=root_noise["alpha"];fraction=root_noise["fraction"]
    if type(alpha) not in (int,float) or not math.isfinite(alpha) or alpha<=0 or type(fraction) not in (int,float) or not math.isfinite(fraction) or not 0<=fraction<=1:raise ValueError("Invalid Dirichlet noise")
    draws=[rng.gammavariate(alpha,1) for _ in node.legal];total=sum(draws)
    if total<=0:raise ValueError("Dirichlet draw underflow")
    for a,z in zip(node.legal,draws):node.priors[a]=(1-fraction)*node.priors[a]+fraction*z/total

def _backup(tree,path,value):
    for node,action in path:node.visits[action]+=1;node.sums[action]+=value
    tree.completed+=1

def _charge(tree,row):
    for key in ("feature_simulations","feature_deposits"):tree.work[key]+=row[key]
    tree.work["neural_evaluations"]+=1;tree.work["neural_calls"]+=1;tree.work["inference_ms"]+=row.get("inference_ms",0.0)

def forest(positions,evaluator,simulations=64,time_ms=0,max_nodes=2048,rng=None,root_noise=None,cancel=None,c_puct=1.5,max_trees=8,native=None,memory_mb=8192):
    started=time.perf_counter();config.integer(simulations,"simulations",1,1000000);config.integer(time_ms,"time_ms",0,43200000)
    config.integer(max_nodes,"max_nodes",1,1000000);config.integer(max_trees,"max_trees",1,32);config.integer(len(positions),"forest_size",1,max_trees)
    config.integer(memory_mb,"memory_mb",512,8192)
    if type(c_puct) not in (int,float) or not math.isfinite(c_puct) or c_puct not in EXPLORATION:raise ValueError("Quality protocol requires declared c_puct 1.5 or 0.25")
    if len(positions)>getattr(evaluator,"max_batch",max_trees):raise ValueError("Forest exceeds evaluator batch bound")
    # Validate the noise contract before any work, even for terminal roots.
    if root_noise is not None:
        dummy=Node({},0,(0,),False,priors=[1.0]+[0.0]*6);_noise(dummy,root_noise,random.Random(0))
    if isinstance(rng,list):
        if len(rng)!=len(positions) or any(not isinstance(item,random.Random) for item in rng):raise ValueError("Per-root rng list must contain exactly one random.Random per position")
        rngs=list(rng)
    else:
        rng=random.Random(0) if rng is None else random.Random(rng) if type(rng) is int else rng
        if not isinstance(rng,random.Random):raise ValueError("rng must be an integer seed, random.Random, or per-root list")
        rngs=[rng]*len(positions)
    deadline=started+time_ms/1000 if time_ms else None;n=native or getattr(evaluator,"native",None) or Native();trees=[];batch_calls=0;pending=[];reports=[]
    def evaluate(pending):
        nonlocal batch_calls
        _poll(cancel,deadline);batch_calls+=1
        inputs=[leaf.position for _,leaf,_ in pending]
        if getattr(evaluator,"supports_cancel",False):rows=evaluator.evaluate_many(inputs,cancel=cancel,deadline=deadline)
        else:rows=evaluator.evaluate_many(inputs)
        if not isinstance(rows,list) or len(rows)!=len(pending):raise NativeError("Evaluator returned wrong batch length")
        # Atomic per-batch validation: a corrupt row must not publish backups.
        for row,(_,leaf,_) in zip(rows,pending):validate_prediction(row,leaf.position)
        for row,(tree,leaf,path) in zip(rows,pending):_charge(tree,row)
        _poll(cancel,deadline)
        for row,(tree,leaf,path) in zip(rows,pending):
            n.trace_mark(leaf.position,"puct_expand")
            _expand(leaf,row)
            if path:_backup(tree,path,leaf.bootstrap)
            else:tree.work["root_expansions"]+=1;_noise(leaf,root_noise,tree.rng)
    try:
        for index,p in enumerate(positions):
            root=_node(n,config.position(p));remaining=0 if deadline is None else max(1,math.ceil((deadline-time.perf_counter())*1000))
            control=Control(n,max_depth=128,time_ms=remaining,node_budget=0,memory_mb=memory_mb,tt_mb=0,use_pvs=False)
            if isinstance(cancel,CancellationToken):cancel._register(control)
            tree=Tree(root,control,rng=rngs[index]);trees.append(tree)
            if root.terminal:tree.stop="terminal"
        try:
            pending=[(t,t.root,[]) for t in trees if not t.stop]
            if pending:evaluate(pending)
            while any(not t.stop and t.completed<simulations for t in trees):
                pending=[]
                for tree in trees:
                    if tree.stop or tree.completed>=simulations:continue
                    _poll(cancel,deadline);node=tree.root;path=[]
                    while node.expanded and not node.terminal:
                        n.trace_mark(node.position,"puct_select")
                        _poll(cancel,deadline);action=_select(node,c_puct);path.append((node,action))
                        if action not in node.children:
                            if tree.nodes>=max_nodes:tree.stop="node_budget";break
                            t=n.apply(flatten(node.position),actor=node.actor,action=action,total=98,trace=False,control=tree.control)
                            if not t.get("ok"):
                                if t.get("error",{}).get("code")=="cancelled":
                                    # An interrupted native move reports cumulative control work.
                                    tree.work["deposits"]=max(tree.work["deposits"],t.get("deposits",0));raise InferenceStopped(t["error"]["message"])
                                raise NativeError(str(t))
                            tree.work["transitions"]+=1;tree.work["deposits"]+=t["drops"]
                            child=_node(n,t["position"]);node.children[action]=child;tree.nodes+=1
                        node=node.children[action]
                    if tree.stop:continue
                    if node.terminal:_backup(tree,path,node.bootstrap)
                    else:pending.append((tree,node,path))
                if pending:evaluate(pending)
        except InferenceStopped as exc:
            reason=exc.reason if exc.reason in ("cancelled","time_budget","memory_budget") else "cancelled"
            # Charge completed feature extraction from a cancelled batch. Its
            # output is deliberately not expanded/backed up.
            if exc.work and pending:
                partial=exc.work.get("per_position",[])
                for i,(tree,_,_) in enumerate(pending):
                    if i<len(partial):
                        for key in ("feature_simulations","feature_deposits"):tree.work[key]+=int(partial[i].get(key,0))
                        tree.work["neural_evaluations"]+=int(partial[i].get("evaluated",False))
                    tree.work["neural_calls"]+=1;tree.work["inference_ms"]+=exc.work.get("whole_call_ms",0.0)/len(pending)
            for tree in trees:
                if not tree.stop and tree.completed<simulations:tree.stop=reason
        elapsed=(time.perf_counter()-started)*1000
        for tree in trees:
            root=tree.root;count=sum(root.visits)
            policy=[v/count for v in root.visits] if count else list(root.priors) if root.expanded else [1/len(root.legal) if a in root.legal else 0.0 for a in range(7)]
            chosen=max(root.legal,key=lambda a:(root.visits[a],root.priors[a] if not count else 0,-a)) if root.legal else None
            estimate=sum(root.sums)/count if count else root.bootstrap
            reports.append({"schema":SCHEMA,"protocol":PROTOCOL,"rules_digest":config.RULES_DIGEST,"position":root.position,
                            "chosen":chosen,"visits":list(root.visits),"policy":policy,"root_priors":list(root.priors),"estimated_value_player0":estimate*98,
                            "action_mean_value_player0":[root.sums[a]/root.visits[a]*98 if root.visits[a] else None for a in range(7)],
                            "root_expanded":root.expanded,"completed_simulations":tree.completed,"requested_simulations":simulations,"nodes":tree.nodes,"max_nodes":max_nodes,
                            "usable_decision":bool(root.terminal or tree.completed>0),"memory_mb":memory_mb,
                            "stop_reason":tree.stop or "simulations","proof_status":"terminal_outcome" if root.terminal else "monte_carlo_estimate",
                            "work":{**tree.work,"total_deposits":tree.work["deposits"]+tree.work["feature_deposits"]},
                            "timing":{"elapsed_ms":elapsed,"requested_time_ms":time_ms,"overshoot_ms":max(0.0,elapsed-time_ms) if time_ms else 0.0,"whole_decision_includes_inference":True},
                            "forest_batch_calls":batch_calls,"forest_size":len(trees),"value_perspective":"fixed_player0_seeds","first_play_urgency":"parent_neural_bootstrap","unvisited_root_q_player0":root.bootstrap*98,"c_puct":c_puct})
    finally:
        for tree in trees:
            if isinstance(cancel,CancellationToken):cancel._remove(tree.control)
            tree.control.close()
    elapsed=(time.perf_counter()-started)*1000
    for report in reports:
        report["timing"]["elapsed_ms"]=elapsed;report["timing"]["overshoot_ms"]=max(0.0,elapsed-time_ms) if time_ms else 0.0
    return reports

def search(position,evaluator,**kwargs):
    return forest([position],evaluator,**kwargs)[0]


def validate_report(report,position,requested,max_nodes=2048,c_puct=1.5,
                    native=None,allow_interrupted=True):
    """Validate completed or interrupted recommendations without trusting hashes.

    This does not prove a neural estimate or independently repeat a timed search.
    Deterministic reference traversals are acceptance evidence, not admission work.
    """
    def require(value,message):
        if not value:raise NativeError(message)
    def number(value):return type(value) in (int,float) and math.isfinite(value)
    def integer(value):return type(value) is int and value>=0
    expected={'schema','protocol','rules_digest','position','chosen','visits','policy',
              'root_priors','estimated_value_player0','action_mean_value_player0',
              'root_expanded','completed_simulations','requested_simulations','nodes',
              'max_nodes','usable_decision','memory_mb','stop_reason','proof_status',
              'work','timing','forest_batch_calls','forest_size','value_perspective',
              'first_play_urgency','unvisited_root_q_player0','c_puct'}
    require(isinstance(report,dict) and set(report)==expected,'Unknown quality search fields')
    require(c_puct in EXPLORATION and type(c_puct) in (int,float),'Undeclared exploration coefficient')
    native=native or Native();root=_node(native,config.position(position));p=root.position
    require(report['schema']==SCHEMA and report['protocol']==PROTOCOL and report['rules_digest']==config.RULES_DIGEST,'Quality search protocol differs')
    require(report['position']==p and report['value_perspective']=='fixed_player0_seeds' and report['first_play_urgency']=='parent_neural_bootstrap','Quality search perspective differs')
    require(type(report['requested_simulations']) is int and report['requested_simulations']==requested and type(report['max_nodes']) is int and report['max_nodes']==max_nodes and type(report['c_puct']) in (int,float) and report['c_puct']==c_puct,'Quality search limits differ')
    require(type(report['memory_mb']) is int and 512<=report['memory_mb']<=8192,'Invalid search memory bound')
    visits=report['visits'];priors=report['root_priors'];policy=report['policy'];means=report['action_mean_value_player0'];completed=report['completed_simulations']
    require(all(isinstance(seq,list) and len(seq)==7 for seq in (visits,priors,policy,means)),'Malformed quality root arrays')
    require(all(integer(v) and (a in root.legal or v==0) for a,v in enumerate(visits)),'Illegal quality edge visits')
    require(integer(completed) and sum(visits)==completed and completed<=(0 if root.terminal else requested),'Invalid completed backups')
    require(type(report['nodes']) is int and 1<=report['nodes']<=max_nodes and report['nodes']<=completed+2,'Invalid node bound')
    require(all(number(v) and v>=0 and (a in root.legal or v==0) for seq in (priors,policy) for a,v in enumerate(seq)),'Invalid legal probabilities')
    d=p['stores'][0]-p['stores'][1];r=sum(map(sum,p['pits']))
    bounded=lambda value:number(value) and d-r-1e-4<=value<=d+r+1e-4
    require(bounded(report['estimated_value_player0']) and bounded(report['unvisited_root_q_player0']),'Quality estimate violates seed bounds')
    require(type(report['root_expanded']) is bool and type(report['usable_decision']) is bool,'Invalid search flags')
    require(report['usable_decision']==bool(root.terminal or completed),'Unusable search labeled usable')
    require(report['stop_reason'] in ('terminal','simulations','node_budget','cancelled','time_budget','memory_budget'),'Unknown quality stop reason')
    require(report['stop_reason']!='node_budget' or report['nodes']==max_nodes,'Node budget stop below declared limit')
    require(report['root_expanded'] or (report['nodes']==1 and completed==0),'Unexpanded root contains tree work')
    require(allow_interrupted or report['stop_reason'] in ('terminal','simulations'),'Interrupted report not admitted')
    if root.terminal:
        require(report['chosen'] is None and report['proof_status']=='terminal_outcome' and report['stop_reason']=='terminal' and report['estimated_value_player0']==d and not report['root_expanded'] and report['nodes']==1,'Fabricated terminal result')
        require(not any(priors+policy) and all(v is None for v in means),'Terminal policy should be empty')
    else:
        require(report['proof_status']=='monte_carlo_estimate' and report['stop_reason']!='terminal','Fabricated nonterminal proof')
        require((report['stop_reason']=='simulations')==(completed==requested),'Search completion reason differs')
        require(abs(sum(policy)-1)<=1e-12,'Quality policy mass differs')
        require((abs(sum(priors)-1)<=1e-12) if report['root_expanded'] else not any(priors),'Quality prior mass differs')
        require(all((v is None if visits[a]==0 else bounded(v)) for a,v in enumerate(means)),'Invalid edge value')
        if completed:
            require(report['root_expanded'] and policy==[v/completed for v in visits],'Quality visit policy differs')
            require(abs(sum(visits[a]*means[a] for a in root.legal if visits[a])/completed-report['estimated_value_player0'])<=1e-10,'Quality root mean differs')
        else:
            fallback=list(priors) if report['root_expanded'] else [1/len(root.legal) if a in root.legal else 0.0 for a in range(7)]
            require(policy==fallback and report['estimated_value_player0']==report['unvisited_root_q_player0'],'Quality fallback differs')
        require(report['chosen']==max(root.legal,key=lambda a:(visits[a],priors[a] if not completed else 0,-a)),'Quality chosen action differs')
    work=report['work'];fields={'transitions','deposits','feature_simulations','feature_deposits','neural_evaluations','neural_calls','inference_ms','root_expansions','total_deposits'}
    require(isinstance(work,dict) and set(work)==fields and all(integer(v) for key,v in work.items() if key!='inference_ms'),'Malformed quality work counters')
    require(number(work['inference_ms']) and work['inference_ms']>=0 and work['transitions']==report['nodes']-1 and work['total_deposits']==work['deposits']+work['feature_deposits'],'Invalid quality transition accounting')
    require(work['root_expansions']==int(report['root_expanded']) and work['neural_evaluations']<=work['neural_calls']<=report['nodes'],'Invalid quality neural accounting')
    require(not root.terminal or not any(work.values()),'Terminal root fabricated work')
    timing=report['timing'];require(isinstance(timing,dict) and set(timing)=={'elapsed_ms','requested_time_ms','overshoot_ms','whole_decision_includes_inference'},'Unknown quality timing fields')
    require(number(timing['elapsed_ms']) and timing['elapsed_ms']>=0 and type(timing['requested_time_ms']) is int and 0<=timing['requested_time_ms']<=43200000 and timing['whole_decision_includes_inference'] is True,'Invalid quality timing')
    require(timing['overshoot_ms']==(max(0.,timing['elapsed_ms']-timing['requested_time_ms']) if timing['requested_time_ms'] else 0.),'Quality overshoot differs')
    require(type(report['forest_size']) is int and 1<=report['forest_size']<=32 and integer(report['forest_batch_calls']),'Invalid quality forest accounting')
    return report
