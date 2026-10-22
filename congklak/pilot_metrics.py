"""Development-only teacher stability and measured pilot coverage/throughput."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from . import config
from .native import Native,flatten
from . import pilot

def measure(path=pilot.DEFAULT_PATH,sample_size=64):
    native=Native(); manifest=pilot.load(path)
    start=time.monotonic(); rows=list(pilot.records(path)); read_seconds=time.monotonic()-start
    selected=sorted((r for r in rows if r["split"]=="development" and r["label"]["kind"]=="finite_choice_horizon"),
                    key=lambda r:hashlib.sha256(("dev-stability-v1/"+r["state_key"]).encode()).digest())[:sample_size]
    comparisons=[]; root_changed=best_changed=action_changed=0; before=time.monotonic()
    settings={**manifest["identity"]["teacher_limits"],"max_depth":6}
    for r in selected:
        p=r["state"]; result=native.analyze(flatten(p),actor=p["controller"],total=98,**settings,keep_history=False)
        if result["stop_reason"]!="max_depth" or not result["report"]: raise pilot.PilotError("Incomplete stability comparison")
        report=result["report"]; old=r["label"]
        changed=old["target_player0"]!=report["depth_result"]["value_player0"]
        policy=old["best_moves"]!=report["best_moves"]
        scores=[{"pit":c["pit"],"value_player0":c["depth_result"]["value_player0"]} for c in report["candidates"]]
        action=old["action_scores"]!=scores
        root_changed+=changed; best_changed+=policy; action_changed+=action
        comparisons.append({"state_key":r["state_key"],"depth4":old["target_player0"],"depth6":report["depth_result"]["value_player0"],
                            "best4":old["best_moves"],"best6":report["best_moves"],"action_scores4":old["action_scores"],"action_scores6":scores})
    comparison_seconds=time.monotonic()-before
    root=Path(path).resolve()/manifest["directory"]
    bytes_total=sum(f["size"] for f in manifest["files"])
    stats=manifest["generation_statistics"]
    return {"dataset_semantic_sha256":manifest["semantic_sha256"],"records":len(rows),"bytes_total":bytes_total,"bytes_per_record":bytes_total/len(rows),
            "packed_bytes_per_record":pilot.ROW.size,"read_seconds":read_seconds,"read_records_per_second":len(rows)/max(read_seconds,1e-9),
            "committed_collection_and_label_seconds":stats["elapsed_seconds"],"committed_collection_and_label_records_per_second":len(rows)/max(stats["elapsed_seconds"],1e-9),
            "rate_excludes":"resume revalidation, prepublication relabel/export, verification and lost in-flight work",
            "observations":stats["observations"],"duplicates":stats["duplicates"],"duplicate_observation_fraction":stats["duplicates"]/max(stats["observations"],1),
            "source_coverage":manifest["sources"],"split_counts":manifest["splits"],"label_kinds":manifest["label_kinds"],
            "active_seeds":dict(sorted(Counter(sum(sum(x) for x in r["state"]["pits"]) for r in rows).items())),
            "store_difference":dict(sorted(Counter(r["state"]["stores"][0]-r["state"]["stores"][1] for r in rows).items())),
            "legal_action_count":dict(sorted(Counter(r["legal_mask"].bit_count() for r in rows).items())),
            "primary_strata":dict(sorted(Counter(r["stratum"] for r in rows).items())),
            "development_stability":{"selection":f"sha256(dev-stability-v1/state_key), first{sample_size} eligible dev horizon rows","sample_size":len(selected),
                "teacher_revision":manifest["identity"]["teacher_revision"],"comparison_limits":settings,"root_value_changed":root_changed,"best_set_changed":best_changed,
                "any_action_score_changed":action_changed,"seconds":comparison_seconds,"comparisons":comparisons},
            "sealed_final_test_used_for_selection_or_tuning":False,"energy_measured":False,"strength_conclusion":"none; depth4 is a declared pilot teacher, not a promoted evaluator"}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--path",default=str(pilot.DEFAULT_PATH)); p.add_argument("--output")
    args=p.parse_args(); result=measure(args.path)
    recovery=pilot.ROOT/"build"/"phase2-recovery.json"
    if recovery.is_file():
        campaign=json.loads(recovery.read_text(encoding="utf-8"))
        if Path(campaign["data_directory"]).resolve()==Path(args.path).resolve():
            result["observed_campaign_wall_seconds"]=campaign["elapsed_seconds"]
            result["observed_campaign_records_per_second"]=result["records"]/campaign["elapsed_seconds"]
            result["campaign_recovery"]=campaign
    output=Path(args.output) if args.output else Path(args.path)/"measurements.json"
    output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(config.canonical({"records":result["records"],"bytes_total":result["bytes_total"],"committed_rate":result["committed_collection_and_label_records_per_second"],
                            "observed_campaign_rate":result.get("observed_campaign_records_per_second"),
                            "dev_root_changed":result["development_stability"]["root_value_changed"],"dev_best_changed":result["development_stability"]["best_set_changed"],"output":str(output.resolve())}))

if __name__=="__main__": main()
