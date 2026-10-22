"""Role-aware metadata selection and existing-target packaging; no engine calls."""
from collections import Counter,defaultdict
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
from . import config,neurodiag_common as common

ROOT=common.ROOT
SCHEMA="neurodiag-existing-training-source-data-v1"
INVENTORY=ROOT/"build/neurodiag-inventory.json"
PANEL=ROOT/"build/neurodiag-panel.json"
INPUTS=ROOT/"build/neurodiag-inputs.json"
STRATA=("50-74/ordinary","25-49/ordinary","25-49/bonus","25-49/capture")
CELLS=tuple(s+"/actor"+str(a) for s in STRATA for a in (0,1))
SALT="neurodiag-training-source-panel/20261102/v1/"
HISTORIES=("ordinary-data-pilot-v1","natural-study-v1","midgame-label-pilot-v1")
JOURNALS=("deeper-9000-v2","ordinary-selfplay-pilot-v1","selfplay-training-pilot-v1",*HISTORIES)
SOURCE_FILES=tuple("data/"+n+"/work.sqlite3" for n in JOURNALS)+("build/ordinary-value-rows.json","build/value-pilot-rows.json","build/stratum-inventory.json","build/allocation-weighting-inventory.json","build/allocation-weighting-reservations.json","build/stratum-reservations.json","build/qstability-reservations.json","build/ordinary-value-registry.json","build/selfplay-training-pilot-registry.json","build/ordinary-selfplay-registry.json")
require,read,sha,publish=common.require,common.read,common.sha,common.publish
equal=common.same


def ro(name):return sqlite3.connect((ROOT/"data"/name/"work.sqlite3").resolve().as_uri()+"?mode=ro",uri=True)


def quotient(state):
    require(set(state)=={"pits","stores","controller"} and type(state["controller"]) is int and state["controller"] in (0,1),"State/controller differs")
    c=state["pits"][0]+state["pits"][1];require(len(c)==14 and all(type(v) is int and v>=0 for v in c+state["stores"]) and sum(c+state["stores"])==98,"State seed geometry differs")
    actor=state["controller"] if any(c) else 0
    return min(bytes(c+[actor]),bytes(c[7:]+c[:7]+[1-actor if any(c) else 0])).hex()


def metadata_rows(name,table,fields,key="state"):
    with closing(ro(name)) as db:
        query="SELECT "+key+",hash,"+",".join("json_extract(body,'$."+f+"')" for f in fields)+" FROM "+table+" ORDER BY "+key
        out=[]
        for ident,digest,*values in db.execute(query):
            row={k:json.loads(v) if k=="state" else v for k,v in zip(fields,values)};row.update(source_key=ident,source_sha256=digest);out.append(row)
        return out


def key_hashes(name,table):
    with closing(ro(name)) as db:return dict(db.execute("SELECT state,hash FROM "+table+" ORDER BY state"))


def caches():
    """Return only row identities/vector dimensions; never extract targets/vectors."""
    out=defaultdict(list)
    for filename in ("build/ordinary-value-rows.json","build/value-pilot-rows.json"):
        text=(ROOT/filename).read_text(encoding="utf-8")
        with closing(sqlite3.connect(":memory:")) as db:
            groups=[r[0] for r in db.execute("SELECT key FROM json_each(?,'$.groups')",(text,))]
            for group in groups:
                if group not in (("rehearsal","high","low") if "ordinary-value" in filename else ("train",)):continue
                for index,key,dimension in db.execute("SELECT key,json_extract(value,'$.state_key'),json_array_length(value,'$.cached_features.x') FROM json_each(?,'$.groups."+group+"')",(text,)):
                    if dimension==127:out[key].append(dict(path=filename,group=group,index=index))
    return out


def metadata_inventory():
    require("torch" not in __import__("sys").modules,"Diagnostic forbids Torch");baseline=common.baseline();require(set(SOURCE_FILES)<=set(baseline["inputs"]),"Missing accepted sources");common.current({p:baseline["inputs"][p] for p in SOURCE_FILES})
    old=read(ROOT/"build/stratum-inventory.json");protection=read(ROOT/"build/allocation-weighting-inventory.json");pq=set(protection["protected_quotients"])|set(protection["prior_H12_H14_exposed_quotients"]);pf=set(protection["protected_families"])|set(protection["prior_H12_H14_exposed_families"])
    for filename in ("ordinary-value-registry.json","selfplay-training-pilot-registry.json","ordinary-selfplay-registry.json"):
        ledger=read(ROOT/"build"/filename);pq.update(ledger["true_evaluation_quotients"] if "true_evaluation_quotients" in ledger else ledger["quotients"]);pf.update(ledger["true_evaluation_families"] if "true_evaluation_families" in ledger else ledger["families"])
    for filename in ("allocation-weighting-reservations.json","stratum-reservations.json","qstability-reservations.json"):
        overlay=read(ROOT/"build"/filename)
        for role in (overlay["roles"].values() if "roles" in overlay else [overlay]):pf.update(role["origin_families"]);pq.update(role["eligible_quotients"])
    parent={};node_reasons=defaultdict(set);ancestry=defaultdict(set)
    def find(n):
        parent.setdefault(n,n)
        if parent[n]!=n:parent[n]=find(parent[n])
        return parent[n]
    def join(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    def add(row,reason=None):
        f=("family",row["family"]);join(f,("quotient",quotient(row["state"])))
        if reason:node_reasons[f].add(reason)
    for cid,c in old["components"].items():
        anchor=("family",c["families"][0]);ancestry[anchor].add(cid)
        for f in c["families"]:join(anchor,("family",f))
        for q in c["quotients"]:join(anchor,("quotient",q))
        if c["blocked_reasons"]:node_reasons[anchor].update("ancestral/"+r for r in c["blocked_reasons"])
    base=metadata_rows("deeper-9000-v2","roots",["state_key","state","family","stratum","legal_mask","split"])
    for row in base:add(row,"original_development" if row["split"]!="train" else None)
    with closing(ro("ordinary-selfplay-pilot-v1")) as db:
        for family,cohort,text in db.execute("SELECT json_extract(body,'$.family'),json_extract(body,'$.cohort'),json_extract(body,'$.candidates') FROM families ORDER BY idx"):
            for row in json.loads(text):
                if row["stratum"] in STRATA[:2] and row["legal_mask"].bit_count()>=2:add(dict(row,family=family),"ordinary_validation_family" if cohort!="training_candidate" else None)
    sp={}
    for name in ("selfplay-training-pilot-v1","ordinary-selfplay-pilot-v1"):
        sp[name]=metadata_rows(name,"sp_rows",["id","state_key","state","family","stratum","legal_mask","choice"],"id")
        for row in sp[name]:add(row)
    for n in list(parent):
        if n[0]=="family" and n[1] in pf:node_reasons[n].add("protected_or_reserved_family")
        if n[0]=="quotient" and n[1] in pq:node_reasons[n].add("protected_or_reserved_quotient")
    nodes=defaultdict(list);reasons=defaultdict(set);old_components=defaultdict(set)
    for n in parent:nodes[find(n)].append(n)
    for n,why in node_reasons.items():reasons[find(n)].update(why)
    for n,ids in ancestry.items():old_components[find(n)].update(ids)
    components={};byfamily={}
    for root,values in nodes.items():
        body=dict(families=sorted(v for k,v in values if k=="family"),quotients=sorted(v for k,v in values if k=="quotient"),blocked_reasons=sorted(reasons[root]),ancestral_components=sorted(old_components[root]));cid=config.digest(body);components[cid]=body
        for f in body["families"]:byfamily[f]=cid
    def clean(row):
        require(row["family"] in byfamily,"Source origin family is missing from the full metadata graph")
        return not components[byfamily[row["family"]]]["blocked_reasons"]
    cache=caches();rehearsal={k for k,refs in cache.items() if any(v["path"]=="build/ordinary-value-rows.json" and v["group"]=="rehearsal" for v in refs)};h8=key_hashes("deeper-9000-v2","labels")
    def describe(row):
        require(row["state_key"]==common.rawkey(row["state"]),"Raw metadata key differs");return dict(row,actor=row["state"]["controller"],cell=row["stratum"]+"/actor"+str(row["state"]["controller"]),component=byfamily[row["family"]],quotient=quotient(row["state"]),historical_role=row.get("cohort",row.get("split","ordinary_selfplay_training_candidate")))
    primary=[]
    for row in base:
        if row["state_key"] not in rehearsal or not clean(row):continue
        require(row["split"]=="train" and row["state_key"] in h8,"Admitted base training/source label missing")
        r=describe(row);r.update(cohort="base_H8",target_refs=[dict(journal="deeper-9000-v2",table="labels",key=row["state_key"],sha256=h8[row["state_key"]],depth=8)],feature_ref=next(v for v in cache[row["state_key"]] if v["path"]=="build/ordinary-value-rows.json" and v["group"]=="rehearsal"));primary.append(r)
    pairs=[]
    for name in HISTORIES:
        roots=metadata_rows(name,"roots",["state_key","state","family","stratum","legal_mask","cohort"])
        with closing(ro(name)) as db:
            tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        h8table="h8_labels" if "h8_labels" in tables else "labels";a=key_hashes(name,h8table);b=key_hashes(name,"h10_labels") if "h10_labels" in tables else {}
        for row in roots:
            key=row["state_key"]
            if row["cohort"]!="training_candidate" or not clean(row) or key not in a or key not in b:continue
            r=describe(row);r.update(cohort="paired_H8_H10",source_journal=name,target_refs=[dict(journal=name,table=table,key=key,sha256=store[key],depth=depth) for table,store,depth in ((h8table,a,8),("h10_labels",b,10))],feature_ref=cache[key][0] if key in cache else None);pairs.append(r)
    require(len({r["state_key"] for r in pairs})==len(pairs),"Paired sources need explicit duplicate ownership")
    mc=[]
    for row in sp["ordinary-selfplay-pilot-v1"]:
        if row["choice"]==0 and clean(row):r=describe(row);r.update(cohort="ordinary_MC",target_refs=[dict(journal="ordinary-selfplay-pilot-v1",table="sp_rows",key=row["id"],sha256=row["source_sha256"],depth=None)],feature_ref=dict(journal="ordinary-selfplay-pilot-v1",table="sp_rows",key=row["id"],field="features"));mc.append(r)
    def counts(values):return dict(records=len(values),rawstates=len({r["state_key"] for r in values}),families=len({r["family"] for r in values}),components=len({byfamily[r["family"]] for r in values}),cells=dict(sorted(Counter(r["stratum"]+"/actor"+str(r["state"]["controller"]) for r in values).items())))
    return dict(schema=SCHEMA,primary_pool=sorted(primary,key=lambda r:r["state_key"]),paired_pool=sorted(pairs,key=lambda r:r["state_key"]),MC_pool=sorted(mc,key=lambda r:r["id"]),components=components,counts=dict(primary=counts(primary),paired=counts(pairs),MC=counts(mc),unused_joint_SP=counts([r for r in sp["selfplay-training-pilot-v1"] if clean(r)])),protected_quotients=sorted(pq),protected_families=sorted(pf),paired_feature_cache=dict(cached=sum(r["feature_ref"] is not None for r in pairs),literal_required=sum(r["feature_ref"] is None for r in pairs)),source_inputs=common.pins(SOURCE_FILES),target_values_extracted_for_selection=False,model_forwards=0,no_new_optimizer_reservations=True)


def select(inv):
    eligible={c:sorted((r for r in inv["primary_pool"] if r["cell"]==c and r["legal_mask"].bit_count()>=2),key=lambda r:(config.digest(SALT+r["state_key"]),r["state_key"])) for c in CELLS};assigned={};chosen={};slots=[(c,n) for c in CELLS for n in range(32)]
    def place(slot,seen):
        for row in eligible[slot[0]]:
            cid=row["component"]
            if cid in seen:continue
            seen.add(cid)
            if cid not in assigned or place(assigned[cid],seen):assigned[cid]=slot;chosen[slot]=row;return True
        return False
    for slot in slots:require(place(slot,set()),"256-root component matching shortage; no replacement/filler")
    primary=sorted(chosen.values(),key=lambda r:r["state_key"]);require(len(primary)==len({r["family"] for r in primary})==len({r["component"] for r in primary})==256,"Primary support repeated origin")
    sources=[*primary,*inv["paired_pool"],*inv["MC_pool"]]
    return dict(schema=SCHEMA+"-panel",primary=primary,paired=inv["paired_pool"],MC=inv["MC_pool"],source_records=len(sources),target_records=len(primary)+2*len(inv["paired_pool"])+len(inv["MC_pool"]),unique_rawstates=len({r["state_key"] for r in sources}),selection_salt=SALT,primary_cells={c:32 for c in CELLS},historical_roles_preserved=True,training_source_diagnostic=True,new_assessment_or_holdout_claim=False,no_new_optimizer_reservations=True,source_inputs=inv["source_inputs"])


def prepare():
    inv=metadata_inventory();panel=select(inv);publish(INVENTORY,inv);publish(PANEL,panel);return panel


def frozen():
    plan=read(ROOT/"build/neurodiag-plan.json");implementation=read(ROOT/"build/neurodiag-implementation.json")
    require(plan["panel_sha256"]==sha(PANEL) and implementation["plan_sha256"]==sha(ROOT/"build/neurodiag-plan.json") and implementation["sources"].get("congklak/neurodiag_data.py")==sha(__file__),"Input preparation requires frozen panel/source")
    common.current(implementation["sources"]);return plan,implementation


def source_item(ref,key_column="state"):
    with closing(ro(ref["journal"])) as db:row=db.execute("SELECT body,hash FROM "+ref["table"]+" WHERE "+key_column+"=?",(ref["key"],)).fetchone()
    require(row is not None,"Selected source record missing");body=json.loads(row[0]);require(config.digest(body)==row[1] and ("sha256" not in ref or ref["sha256"]==row[1]),"Selected target source/hash changed");return body,row[1]


def cache_features(ref,metadata):
    text=(ROOT/ref["path"]).read_text(encoding="utf-8");base="$.groups."+ref["group"]+"["+str(ref["index"])+"]"
    with closing(sqlite3.connect(":memory:")) as db:
        value=db.execute("SELECT json_extract(?,?),json_extract(?,?),json_extract(?,?),json_extract(?,?)",(text,base+".cached_features",text,base+".state",text,base+".state_key",text,base+".family")).fetchone()
    require(value[0] is not None and equal(json.loads(value[1]),metadata["state"]) and value[2]==metadata["state_key"] and value[3]==metadata["family"],"Cached features are not the exact admitted source state/origin")
    full=json.loads(value[0]);features={k:full[k] for k in ("x","d","r")};origin=dict(kind="cached",reference=ref,file_sha256=sha(ROOT/ref["path"]),cached_features_sha256=config.digest(full),features_sha256=config.digest(features));return features,origin


def validate_features(features,state):
    require(set(features)=={"x","d","r"} and isinstance(features["x"],list) and len(features["x"])==127 and all(type(v) in (int,float) and math.isfinite(v) for v in features["x"]),"Invalid frozen127 features")
    active=sum(map(sum,state["pits"]));require(type(features["d"]) is int and features["d"]==state["stores"][0]-state["stores"][1] and type(features["r"]) is int and features["r"]==(1 if state["controller"]==0 else -1)*active and active>4,"Feature residual orientation differs")


def teacher_target(meta,ref):
    item,digest=source_item(ref);label=item["label"];state=meta["state"];actor=meta["actor"]
    with closing(ro(ref["journal"])) as db:root_sha,nested,wrapper_sha=db.execute("SELECT json_extract(body,'$.source_root_sha256'),json_extract(body,'$.root'),hash FROM roots WHERE state=?",(ref["key"],)).fetchone()
    expected=root_sha or (config.digest(json.loads(nested)) if nested is not None else wrapper_sha)
    require(item["root_sha256"]==expected and type(label["choice_depth"]) is int and label["choice_depth"]==ref["depth"],"Requested teacher source/root/depth differs")
    scores=label["action_scores"];q={r["pit"]:r["value_player0"] for r in scores};legal=[a for a,n in enumerate(state["pits"][actor]) if n];require(len(q)==len(scores) and set(q)==set(legal) and all(type(v) is int for v in q.values()),"Teacher full-action support differs")
    value=(max if actor==0 else min)(q.values());require(label["target_player0"]==value and equal(label["best_moves"],[a for a in legal if q[a]==value]),"Teacher best value/full ties differ")
    return value,dict(reference=ref,item_sha256=digest,root_sha256=expected,requested_depth=ref["depth"],best_moves=label["best_moves"],label_sha256=config.digest(label),target_is_solved_truth=False)


def resolve_inputs(panel=None):
    """Only after source freeze: selected admitted targets/caches and28 literal checks."""
    frozen();panel=read(PANEL) if panel is None else panel;require(equal(panel,read(PANEL)),"Metadata panel changed");common.current(panel["source_inputs"])
    from . import neurodiag_metrics as helper
    records=[];generated=[];compared=[];encoder_sha=sha(helper.__file__)
    for meta in [*panel["primary"],*panel["paired"],*panel["MC"]]:
        if meta["cohort"]=="ordinary_MC":
            source,digest=source_item(meta["target_refs"][0],"id");features={k:source["features"][k] for k in ("x","d","r")};origin=dict(kind="cached",reference=meta["feature_ref"],source_sha256=digest,features_sha256=config.digest(features))
            game,gsha=source_item(dict(journal="ordinary-selfplay-pilot-v1",table="sp_games",key=source["gameid"]),"id");final=game["position"];value=source["terminal_return_player0"]
            require(gsha==source["terminal_game_sha256"] and game["status"]=="terminal" and not sum(map(sum,final["pits"])) and game["family"]==meta["family"] and value==game["terminal_margin"]==final["stores"][0]-final["stores"][1] and type(value) is int and value%2==0,"Individual observed terminal target provenance differs")
            with closing(ro("ordinary-selfplay-pilot-v1")) as db:decision_sha=db.execute("SELECT hash FROM sp_decisions WHERE gameid=? AND choice=?",(source["gameid"],source["choice"])).fetchone()[0]
            require(decision_sha==source["source_decision_sha256"] and equal(source["state"],meta["state"]) and source["choice"]==0,"Initial terminal-return source decision differs")
            targets=[(value,"observed_control_selfplay_terminal_margin",dict(reference=meta["target_refs"][0],item_sha256=digest,terminal_game_sha256=gsha,source_decision_sha256=decision_sha,gameid=source["gameid"],individual_policy_return_not_minimax_truth=True))]
        else:
            if meta["feature_ref"] is not None:features,origin=cache_features(meta["feature_ref"],meta)
            else:features=None
            if meta["cohort"]=="paired_H8_H10":
                literal=helper.literal_features(meta["state"]);validate_features(literal,meta["state"])
                if features is None:features=literal;generated.append(meta["state_key"]);origin=dict(kind="literal_generated",encoder_sha256=encoder_sha,features_sha256=config.digest(features))
                else:
                    maximum=max(abs(a-b) for a,b in zip(features["x"],literal["x"]));require(features["d"]==literal["d"] and features["r"]==literal["r"] and maximum<=1e-12,"Literal relay encoder differs from accepted paired cache");compared.append(dict(state_key=meta["state_key"],maximum_absolute_difference=maximum,features_sha256=config.digest(features),literal_features_sha256=config.digest(literal)))
            targets=[]
            for ref in meta["target_refs"]:
                value,evidence=teacher_target(meta,ref);targets.append((value,"approximate_H"+str(ref["depth"])+"_root_margin",evidence))
        validate_features(features,meta["state"])
        for value,kind,evidence in targets:
            require(type(value) is int and -1<=(value-features["d"])/features["r"]<=1,"Typed target lies outside the root residual bounds")
            prefix="ordinary_MC/"+meta["id"] if meta["cohort"]=="ordinary_MC" else ("base_H8/"+meta["state_key"] if meta["cohort"]=="base_H8" else "paired_H"+str(evidence["requested_depth"])+"/"+meta["source_journal"]+"/"+meta["state_key"])
            records.append(dict(id=prefix,target_player0=value,target_type=kind,cohort=meta["cohort"],stratum=meta["stratum"],actor=meta["actor"],family=meta["family"],historical_role=meta["historical_role"],component=meta["component"],state_key=meta["state_key"],state=meta["state"],features=features,feature_origin=origin,target_source=evidence,panel_row_sha256=config.digest(meta)))
    units={}
    for record in sorted(records,key=lambda r:r["id"]):
        key=config.digest(dict(state_key=record["state_key"],features=record["features"]))
        if key not in units:units[key]=dict(state_key=record["state_key"],state=record["state"],features=record["features"],records=[],feature_origin=dict(sources=[],generated=False))
        unit=units[key];unit["records"].append({k:v for k,v in record.items() if k not in ("state","features")})
        if not any(equal(v,record["feature_origin"]) for v in unit["feature_origin"]["sources"]):unit["feature_origin"]["sources"].append(record["feature_origin"])
        unit["feature_origin"]["generated"]|=record["feature_origin"]["kind"]=="literal_generated"
    body=dict(schema=SCHEMA+"-inputs",panel_sha256=sha(PANEL),units=dict(sorted(units.items())),counts=dict(source_records=panel["source_records"],target_records=len(records),forward_units=len(units),unique_rawstates=len({u["state_key"] for u in units.values()}),models=2,literal_generated_missing_vectors=len(generated),literal_cache_comparisons=len(compared),literal_encoder_calls=len(generated)+len(compared)),literal_generated_state_keys=sorted(generated),literal_cache_comparisons=sorted(compared,key=lambda r:r["state_key"]),feature_encoder_sha256=encoder_sha,source_inputs=panel["source_inputs"],model_files=common.pins([common.CONTROL,common.CANDIDATE]),unit_sha256={k:config.digest(v) for k,v in sorted(units.items())},training_source_diagnostic=True,new_optimizer_reservations=False)
    validate_inputs(body,panel);return body


def validate_inputs(pack,panel=None):
    """No source-target loading, encoder, forward, derivative or SVD work."""
    frozen();panel=read(PANEL) if panel is None else panel;require(pack["schema"]==SCHEMA+"-inputs" and pack["panel_sha256"]==sha(PANEL) and equal(pack["source_inputs"],panel["source_inputs"]),"Frozen input source/panel differs");common.current(pack["source_inputs"]);require(equal(pack["model_files"],common.pins([common.CONTROL,common.CANDIDATE])),"Fixed model source set differs")
    require(pack["feature_encoder_sha256"]==sha(ROOT/"congklak/neurodiag_metrics.py"),"Feature encoder source changed");units=pack["units"];require(list(units)==sorted(units) and equal(pack["unit_sha256"],{k:config.digest(v) for k,v in units.items()}),"Input unit order/hash differs")
    metadata={config.digest(row):row for name in ("primary","paired","MC") for row in panel[name]};expected_ids=set()
    for row in metadata.values():
        for ref in row["target_refs"]:expected_ids.add("ordinary_MC/"+row["id"] if row["cohort"]=="ordinary_MC" else "base_H8/"+row["state_key"] if row["cohort"]=="base_H8" else "paired_H"+str(ref["depth"])+"/"+row["source_journal"]+"/"+row["state_key"])
    ids=set();counts=Counter();raw=set()
    for key,unit in units.items():
        require(key==config.digest(dict(state_key=unit["state_key"],features=unit["features"])) and unit["state_key"]==common.rawkey(unit["state"]),"Raw unit identity differs");validate_features(unit["features"],unit["state"])
        require(unit["state_key"] not in raw,"One raw state has conflicting feature units");raw.add(unit["state_key"])
        require(unit["records"] and [r["id"] for r in unit["records"]]==sorted(r["id"] for r in unit["records"]),"Empty/unsorted target records")
        for row in unit["records"]:
            require(row["id"] not in ids and row["state_key"]==unit["state_key"] and row["actor"]==unit["state"]["controller"] and type(row["target_player0"]) is int and -1<=(row["target_player0"]-unit["features"]["d"])/unit["features"]["r"]<=1,"Duplicate/invalid oriented target record");ids.add(row["id"]);counts[row["cohort"]]+=1
            require(row["panel_row_sha256"] in metadata,"Target record has no admitted metadata origin");meta=metadata[row["panel_row_sha256"]];require(all(equal(row[k],meta[k]) for k in ("state_key","cohort","stratum","actor","family","historical_role","component")),"Target origin/role changed")
            ref=row["target_source"]["reference"];require(any(equal(ref,v) for v in meta["target_refs"]) and row["target_source"]["item_sha256"]==ref["sha256"],"Target reference is not its admitted source")
            kind="observed_control_selfplay_terminal_margin" if meta["cohort"]=="ordinary_MC" else "approximate_H"+str(ref["depth"])+"_root_margin";require(row["target_type"]==kind,"Target estimand/depth changed")
            origin=row["feature_origin"];require(origin["features_sha256"]==config.digest(unit["features"]),"Feature origin/value ownership differs")
            if meta["feature_ref"] is None:require(meta["cohort"]=="paired_H8_H10" and origin["kind"]=="literal_generated" and origin["encoder_sha256"]==pack["feature_encoder_sha256"],"Undeclared feature generation")
            else:require(origin["kind"]=="cached" and equal(origin["reference"],meta["feature_ref"]),"Cached feature origin changed")
    require(ids==expected_ids and counts==dict(base_H8=len(panel["primary"]),paired_H8_H10=2*len(panel["paired"]),ordinary_MC=len(panel["MC"])) and pack["counts"]["target_records"]==len(ids)==panel["target_records"] and pack["counts"]["forward_units"]==len(units)==len(raw)==panel["unique_rawstates"] and pack["counts"]["unique_rawstates"]==len(raw),"Frozen cohort/unit counts differ")
    require(pack["counts"]["literal_generated_missing_vectors"]==len(pack["literal_generated_state_keys"])==13 and pack["counts"]["literal_cache_comparisons"]==len(pack["literal_cache_comparisons"])==15,"Declared existing/literal feature coverage changed");return pack


def load_inputs():return validate_inputs(read(INPUTS))
