"""Independent source-bound facts for the next-update ownership epoch."""
import hashlib,json,os,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'build/ownership-resolution-v1/audit'
PRIOR=ROOT/'build/tracing-speed-v1/acceptance.json'

def need(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def publish(path,body):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write((json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())

def facts():
    prior=read(PRIOR);need(sha(PRIOR)=='9b7da69df92fe6b7ca8bf0d126b3cd89cb6ff717c341b89db1ad0b573a0c717c' and prior['passed'] is True and len(prior['inputs'])==78403,'Accepted preservation boundary differs')
    need(prior['training_authorized'] is False and prior['execution_ready'] is False and prior['open_holds']==['320 prior support conflicts','missing historical engineering footprints'],'Prior holds changed')
    reportpath=ROOT/'build/ownership-implementation-v1/provenance/certified-v2/report.json';report=read(reportpath)
    support=report['candidate_pools']['support_reuse'];excluded=[r for r in support if not r['eligible']];eligible=[r for r in support if r['eligible']]
    need(len(support)==512 and len(excluded)==320 and len(eligible)==192,'Support root counts differ')
    badcomponents=Counter(r['new_domain_component'] for r in excluded);goodcomponents={r['new_domain_component'] for r in eligible}
    need(len(badcomponents)==1 and len(goodcomponents)==192 and not goodcomponents.intersection(badcomponents),'Whole-component support boundary differs')
    component=next(iter(badcomponents));need(component=='4a4fd6fd52b37c833f14044d30c890763f09110de74090273517f0ccda66c2a8','Giant blocker component changed')
    markers=[report['marker_ledger'][i] for i in report['components'][component]['marker_indices']]
    blockers=Counter(m['reason'] for m in markers if m['optimizer_blocked'])
    need(any('current frozen fit_check' in reason for reason in blockers) and any('current frozen arena' in reason for reason in blockers) and blockers['previous recorded internal search trace']>0,'Genuine cross-role/internal protections missing')
    supplementpath=ROOT/'build/ownership-implementation-v1/provenance/engineering-supplement-v1/report.json';supplement=read(supplementpath)
    need(supplement['complete_engineering_source_absence_claim'] is False and supplement['historical_unknown_leaves_still_unknown'] is True and supplement['missing_material_engineering_footprints_hold'] is True,'Unknown historical source scope changed')
    need(len(supplement['missing_attempt_ledger'])==2 and all(x['missing_high_footprints_possible'] is True for x in supplement['missing_attempt_ledger']),'Known missing-attempt definitions changed')
    qualitypath=ROOT/'build/quality-benchmark.json';quality=read(qualitypath)
    q1=quality['arenas']['refined_search_vs_H_v1'];q8=quality['arenas']['refined_search_vs_H8_relay_CPU']
    need(q1['complete_pairs']==q8['complete_pairs']==24 and not quality['both_baseline_primary_gates']['refined_search'],'Historical quality gate changed')
    valuepath=ROOT/'build/value-run-v1/decision.json';value=read(valuepath)
    need(value['candidate_quarantined'] is True and value['default_engine']=='H_v1' and value['strength_interpretation_allowed'] is False and value['cluster_statistics']=={},'Current candidate/strength scope changed')
    sourcepaths=[Path(__file__),PRIOR,reportpath,supplementpath,qualitypath,valuepath,ROOT/'docs/VALUE_RUN_RESULTS.md',ROOT/'docs/QUALITY_RESULTS.md',ROOT/'build/selfplay-pilot-benchmark.json',ROOT/'build/ownership-implementation-v1/reservations/pre-draw-binding-v4.json']
    source_inputs={p.relative_to(ROOT).as_posix():sha(p) for p in sourcepaths}
    need('torch' not in sys.modules,'Facts imported Torch')
    publish(OUT/'facts.json',dict(schema='ownership-resolution-independent-facts-v1',passed=True,source_inputs=source_inputs,
      preserved_prior_boundary_files=78404,prior_input_full_rehash_claimed=False,old_support_roots=512,genuine_excluded_roots=320,genuine_excluded_components=1,
      excluded_component=component,recorded_eligible_support_roots=192,recorded_eligible_components=192,blocking_marker_reasons=dict(blockers),
      missing_attempt_ledger=supplement['missing_attempt_ledger'],supplement_missing_ledger_not_claimed_exhaustive=True,
      current_strength=dict(scope='consumed development only; no certified superiority or release strength',control_vs_H_v1_mean_seeds=q1['primary_paired_mean_final_margin_seeds'],control_vs_H8_mean_seeds=q8['primary_paired_mean_final_margin_seeds'],both_baseline_gate=False,
        value_pilot_failed_gates=value['failed_gates'],value_raw_descriptive=value['descriptive_raw_comparisons'],confidence_intervals_withheld=True,default_engine='H_v1',candidate_quarantined=True),
      models_fixed=True,policy_fixed=True,new_searches=0,new_games=0,new_labels=0,optimizer_steps=0,no_historical_absence_claim=True))
    print(json.dumps(dict(passed=True,excluded_roots=320,excluded_components=1,recorded_eligible_roots=192,certified_strength=False)),flush=True)

if __name__=='__main__':facts()
