"""Human-readable statements derived only from replay-checked report facts."""
def label(actor,pit):
    return ("A" if actor==0 else "B")+str(pit)

def explain(report):
    if report is None:
        return {"summary":"No complete ranking is available. Resume to recompute unfinished work.","candidates":[]}
    actor=report["score_player"]
    player="A" if actor==0 else "B"
    value=report["depth_result"]["value_actor"]
    depth=report["choice_depth"]
    if report["terminal"]:
        outcome="wins" if value>0 else "loses" if value<0 else "draws"
        return {"summary":f"Both rows are empty. {player} {outcome} with final store margin {value:+d} seeds.","candidates":[]}
    prefix=" ".join(f"{'A' if p['from']==0 else 'B'} must pass to {'A' if p['to']==0 else 'B'}." for p in report["forced_passes"])
    if depth==0:
        return {"summary":prefix+" Depth zero evaluates current store margin only; no moves have been ranked.","candidates":[]}
    best=", ".join(label(actor,p) for p in report["best_moves"])
    summary=f"{prefix} {best} ranks best for {player} at choice depth {depth}, with H_v1 value {value:+d} seeds for {player}. This is a finite-horizon recommendation, not a full-game solution.".strip()
    candidates=[]
    for c in report["candidates"]:
        t=c["transition"]
        facts=[f"Adds {t['store_delta'][actor]} seeds to {player}'s store in {t['drops']} deposits and {t['relays']} relay pickups."]
        capture=next((e["count"] for e in t["events"] if e["type"]=="capture"),None)
        if capture is not None: facts.append(f"Captures {capture} seeds, including the landing seed.")
        if t["terminal"]: facts.append("Both rows are then empty; the game ends.")
        elif t["reason"]=="own_store":
            facts.append(f"Ends in {player}'s store." if t["passes"] else f"Ends in {player}'s store and grants another choice.")
        elif t["reason"]=="own_empty": facts.append("The last seed remains in an own pit because its opposite is empty; the turn ends.")
        else: facts.append("The turn ends after the landing/capture rule.")
        for p in t["passes"]: facts.append(f"{'A' if p['from']==0 else 'B'} has an empty row and must pass to {'A' if p['to']==0 else 'B'}.")
        continuation=" → ".join(label(m["actor"],m["pit"]) for m in c["pv"])
        continuation += ". Stops at the horizon; later choices are unsearched." if c["pv_truncated"] else ". This conditional line reaches a terminal position."
        candidates.append({"actor":actor,"pit":c["pit"],"move":label(actor,c["pit"]),
                           "immediate":" ".join(facts),
                           "evaluation":f"H_v1 at choice depth {depth}: {c['depth_result']['value_actor']:+d} seeds for {player}.",
                           "conditional_continuation":continuation})
    return {"summary":summary,"candidates":candidates}
