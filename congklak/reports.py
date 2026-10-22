"""Validate durable report shape and replay evidence before trusting a checkpoint.

Replay checks stored facts and horizon-leaf PV values, not the search's minimax
proof. Exhaustive differential acceptance is the separate search verifier.
"""
from . import config
from .native import flatten

def require(condition,message):
    if not condition: raise ValueError(message)

def exact_int(value,lower=0,upper=2**64-1):
    require(type(value) is int and lower<=value<=upper,"invalid report integer")

def passes(items):
    require(isinstance(items,list),"invalid pass list")
    for p in items:
        require(isinstance(p,dict) and set(p)=={"from","to"},"invalid pass")
        exact_int(p["from"],0,1); exact_int(p["to"],0,1)
        require(p["from"]!=p["to"],"invalid pass actors")

def coverage(item,depth):
    fields={"nodes","transitions","terminal_leaves","horizon_leaves","leaves","alpha_beta_cutoffs",
            "controller_changes","completed_turns","terminal_transitions","opponent_choices","deepest_choice_depth","scope",
            "tt_probes","tt_hits","tt_exact_hits","tt_bound_cutoffs","pvs_probes","pvs_researches","pv_reconstruction_nodes"}
    require(isinstance(item,dict) and set(item)==fields,"invalid coverage fields")
    for key in fields-{"scope"}: exact_int(item[key])
    require(item["scope"]=="visited_search_tree" and item["deepest_choice_depth"]<=depth,"invalid coverage scope/depth")
    require(item["leaves"]==item["terminal_leaves"]+item["horizon_leaves"],"invalid coverage leaves")

def physical(item,root,actor):
    d=root["stores"][0]-root["stores"][1]
    active=sum(sum(r) for r in root["pits"])
    lo,hi=d-active,d+active
    expected={"lower_player0":lo,"upper_player0":hi,"score_player":actor,
              "lower":lo if actor==0 else -hi,"upper":hi if actor==0 else -lo,
              "scope":"eventual_game_physical_seed_interval","used_for_pruning":False}
    require(isinstance(item,dict) and set(item)==set(expected),"invalid physical interval fields")
    for k in ("lower_player0","upper_player0","lower","upper","score_player"): exact_int(item[k],-98,98)
    require(item["used_for_pruning"] is False and item==expected,"invalid physical interval")

def score(item,actor,depth,ident,terminal=False):
    fields={"value_player0","value_actor","value","score_player","choice_depth","bound","scope","evaluator","policy","terminal_root"}
    require(isinstance(item,dict) and set(item)==fields,"invalid depth_result fields")
    for k in ("value_player0","value_actor","value"): exact_int(item[k],-98,98)
    exact_int(item["score_player"],0,1); exact_int(item["choice_depth"],0,128)
    require(item["score_player"]==actor and item["choice_depth"]==depth,"invalid result actor/depth")
    require(item["value"]==item["value_actor"]==(item["value_player0"] if actor==0 else -item["value_player0"]),"invalid result perspective")
    require(item["bound"]=="exact" and item["scope"]=="finite_choice_horizon","invalid result claim")
    require(item["evaluator"]==ident["evaluator"] and item["policy"]==ident["policy"],"invalid result identity")
    require(type(item["terminal_root"]) is bool and item["terminal_root"]==terminal,"invalid terminal result")

def pv(items,truncated,root,depth,native,value=None,first=None):
    require(isinstance(items,list) and len(items)<=depth and type(truncated) is bool,"invalid PV shape")
    current=root
    for i,choice in enumerate(items):
        fields={"actor","pit","next_controller","bonus_choice","forced_passes_after","terminal_after"}
        require(isinstance(choice,dict) and set(choice)==fields,"invalid PV choice fields")
        exact_int(choice["actor"],0,1); exact_int(choice["pit"],0,6)
        require(type(choice["bonus_choice"]) is bool and type(choice["terminal_after"]) is bool,"invalid PV flags")
        normal=native.normalize(flatten(current),actor=current["controller"],total=98)
        require(choice["actor"]==normal["normalized_controller"] and choice["pit"] in normal["legal_actions"],"illegal PV choice")
        if first is not None and i==0: require(choice["pit"]==first,"invalid candidate PV origin")
        current=normal["position"]
        step=native.apply(flatten(current),actor=current["controller"],action=choice["pit"],total=98,trace=False)
        require(step["ok"] is True,"invalid PV transition")
        if not step["terminal"]: exact_int(choice["next_controller"],0,1)
        require(choice["next_controller"]==(None if step["terminal"] else step["position"]["controller"]),"invalid PV next actor")
        require(choice["terminal_after"]==step["terminal"] and choice["bonus_choice"]==(step["reason"]=="own_store" and not step["terminal"]),"invalid PV explanation")
        passes(choice["forced_passes_after"])
        require(choice["forced_passes_after"]==step["passes"],"invalid PV pass evidence")
        current=step["position"]
    terminal=not any(sum(r) for r in current["pits"])
    require(truncated is (not terminal),"invalid PV truncation")
    if not terminal: require(len(items)==depth,"short nonterminal PV")
    if first is not None: require(bool(items),"missing candidate PV")
    if value is not None:
        require(current["stores"][0]-current["stores"][1]==value,"PV leaf differs from claimed horizon value")

def validate(report,root,ident,native):
    fields={"completed_iteration","choice_depth","input_controller","normalized_controller","score_player","forced_passes",
            "terminal","position","legal_actions","physical_interval","evaluator","policy","engine_version","depth_result",
            "best_moves","candidates","pv","pv_truncated","proof_status","coverage","nodes","deposits","elapsed_ms","peak_process_bytes","search_options"}
    require(isinstance(report,dict) and set(report)==fields,"invalid report fields")
    require(report["completed_iteration"] is True,"incomplete checkpoint")
    depth=report["choice_depth"]; exact_int(depth,0,128)
    options=report["search_options"]
    require(isinstance(options,dict) and set(options)=={"tt_mb","use_pvs","tt_allocated_bytes","tt_capacity","tt_key","pv_policy"},"invalid search options")
    exact_int(options["tt_mb"],0,1024); exact_int(options["tt_allocated_bytes"]); exact_int(options["tt_capacity"])
    require(type(options["use_pvs"]) is bool and options["tt_allocated_bytes"]<=options["tt_mb"]*1024*1024,"invalid TT/PVS allocation")
    require(options["tt_key"]=="full-state/normalized-controller/exact-remaining-depth/H_v1/policy-v2" and options["pv_policy"]=="lexicographic-optimal/full-window-reconstruction","invalid optimization identity")
    normal=native.normalize(flatten(root),actor=root["controller"],total=98)
    terminal=normal["terminal"]
    actor=root["controller"] if terminal else normal["normalized_controller"]
    for k in ("input_controller","score_player"): exact_int(report[k],0,1)
    require(type(report["terminal"]) is bool and report["terminal"]==terminal,"invalid root terminal flag")
    require(report["input_controller"]==root["controller"] and report["score_player"]==actor,"invalid report perspective")
    if not terminal: exact_int(report["normalized_controller"],0,1)
    require(report["normalized_controller"]==(None if terminal else actor),"invalid report controller")
    config.position(report["position"])
    require(report["position"]==normal["position"],"invalid report root")
    passes(report["forced_passes"])
    require(report["forced_passes"]==normal["passes"],"invalid root passes")
    actions=normal["legal_actions"]
    require(isinstance(report["legal_actions"],list),"invalid legal_actions")
    for action in report["legal_actions"]: exact_int(action,0,6)
    require(report["legal_actions"]==actions,"invalid legal set")
    require(report["evaluator"]==ident["evaluator"] and report["policy"]==ident["policy"] and report["engine_version"]==native.abi,"invalid report identity")
    require(report["proof_status"]==("terminal_outcome" if terminal else "finite_horizon_result"),"invalid proof label")
    require(not terminal or depth==0,"terminal checkpoint depth must be zero")
    score(report["depth_result"],actor,depth,ident,terminal)
    physical(report["physical_interval"],root,actor)
    coverage(report["coverage"],depth)
    for k in ("nodes","deposits","elapsed_ms","peak_process_bytes"): exact_int(report[k])
    require(isinstance(report["candidates"],list) and len(report["candidates"])==len(actions),"invalid candidates")
    values=[]
    for expected_pit,c in zip(actions,report["candidates"]):
        cf={"actor","pit","depth_result","analysis_complete","transition","physical_interval","pv","pv_truncated","coverage"}
        require(isinstance(c,dict) and set(c)==cf,"invalid candidate fields")
        exact_int(c["actor"],0,1); exact_int(c["pit"],0,6)
        require(c["actor"]==actor and c["pit"]==expected_pit,"invalid candidate actor/pit")
        step=native.apply(flatten(normal["position"]),actor=actor,action=c["pit"],total=98,trace=True)
        require(config.canonical(c["transition"])==config.canonical(step),"candidate transition differs from replay")
        physical(c["physical_interval"],step["position"],actor)
        coverage(c["coverage"],depth)
        require(type(c["analysis_complete"]) is bool and c["analysis_complete"]==(depth>0),"invalid candidate completion flag")
        if depth==0:
            require(c["depth_result"] is None and c["pv"]==[] and c["pv_truncated"] is True,"depth-zero candidate invents ranking")
        else:
            score(c["depth_result"],actor,depth,ident)
            value=c["depth_result"]["value_player0"]
            values.append(value)
            pv(c["pv"],c["pv_truncated"],root,depth,native,value,c["pit"])
    require(isinstance(report["best_moves"],list),"invalid best_moves")
    for action in report["best_moves"]: exact_int(action,0,6)
    if values:
        best=(max if actor==0 else min)(values)
        require(report["depth_result"]["value_player0"]==best and report["best_moves"]==[p for p,v in zip(actions,values) if v==best],"invalid root ranking")
    else:
        require(report["best_moves"]==[] and report["depth_result"]["value_player0"]==root["stores"][0]-root["stores"][1],"invalid unranked result")
    pv(report["pv"],report["pv_truncated"],root,depth,native,report["depth_result"]["value_player0"])
