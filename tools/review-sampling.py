"""Offline root/family exposure review; no optimizer, pruning or model changes."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from contextlib import closing
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,phase6_common as common

OUTPUT=ROOT/'build/sampling-review.json'
SOURCE=ROOT/'data/deeper-9000-v2/work.sqlite3'
MODEL=ROOT/'data/quality-train-v1/model.json'
DRAWS=384000
SEED=20261015
MIXTURE_STRATUM_FRACTION=.1
INPUTS={}
require=common.require


def pin(path):
    path=Path(path).resolve();require(path.is_relative_to(ROOT) and path.is_file(),'Missing sampling input '+str(path))
    name=path.relative_to(ROOT).as_posix();INPUTS[name]=common.sha(path);return INPUTS[name]


def load(path):pin(path);return json.loads(Path(path).read_text(encoding='utf-8'))


def history_type(family):
    prefix=family.split('/')[0]
    return 'constructed' if prefix.startswith('constructed') else 'recorded_history' if prefix.endswith('_history') else 'unknown'


def quantiles(values):
    values=sorted(values);n=len(values)
    return dict(minimum=values[0],median=statistics.median(values),p95=values[math.ceil(.95*n)-1],maximum=values[-1])


def effective_support(probabilities):
    return 1/math.fsum(p*p for p in probabilities)


def exposure(rows,probabilities,draws):
    require(len(rows)==len(probabilities) and all(math.isfinite(p) and p>0 for p in probabilities) and abs(math.fsum(probabilities)-1)<1e-12,'Invalid sampling probability distribution')
    strata=defaultdict(list);families=defaultdict(list);types=defaultdict(list)
    for index,row in enumerate(rows):strata[row['stratum']].append(index);families[row['family']].append(index);types[history_type(row['family'])].append(index)
    n=len(rows);family_prob={f:math.fsum(probabilities[i] for i in indices) for f,indices in families.items()}
    require(abs(math.fsum(family_prob.values())-1)<1e-12,'Family mass conservation differs')
    by_stratum={}
    for name,indices in sorted(strata.items()):
        p=probabilities[indices[0]];require(all(probabilities[i]==p for i in indices),'Within-stratum root weights are not equal')
        by_stratum[name]=dict(roots=len(indices),families=len({rows[i]['family'] for i in indices}),probability=math.fsum(probabilities[i] for i in indices),per_root_probability=p,
            expected_draws=draws*math.fsum(probabilities[i] for i in indices),expected_draws_per_root=draws*p)
    top_family=sorted(family_prob,key=lambda f:(-family_prob[f],f))[:20]
    family_rows=[dict(family=f,history_type=history_type(f),roots=len(families[f]),strata=dict(sorted(Counter(rows[i]['stratum'] for i in families[f]).items())),
        probability=family_prob[f],expected_draws=draws*family_prob[f],mass_ratio_to_uniform_root=family_prob[f]/(len(families[f])/n),expected_draws_per_family_root=draws*family_prob[f]/len(families[f])) for f in top_family]
    root_top=sorted(range(n),key=lambda i:(-probabilities[i],rows[i]['state_key']))[:20]
    source_mass={}
    for name,indices in sorted(types.items()):
        mass=math.fsum(probabilities[i] for i in indices);owned={rows[i]['family'] for i in indices}
        source_mass[name]=dict(roots=len(indices),families=len(owned),probability=mass,expected_draws=draws*mass,
            conditional_root_effective_support=effective_support([probabilities[i]/mass for i in indices]),conditional_family_effective_support=effective_support([family_prob[f]/mass for f in owned]))
    size_buckets=defaultdict(list)
    for f,indices in families.items():
        count=len(indices);bucket='1' if count==1 else '2' if count==2 else '3-4' if count<=4 else '5-8' if count<=8 else '9+'
        size_buckets[bucket].append(f)
    sizes={name:dict(families=len(values),roots=sum(len(families[f]) for f in values),probability=math.fsum(family_prob[f] for f in values),expected_draws=draws*math.fsum(family_prob[f] for f in values)) for name,values in sorted(size_buckets.items())}
    rare=[i for i,r in enumerate(rows) if len(strata[r['stratum']])<=5];endgame=[i for i,r in enumerate(rows) if r['stratum'].startswith('5-24/')]
    return dict(root_effective_support=effective_support(probabilities),family_effective_support=effective_support(list(family_prob.values())),minimum_root_probability=min(probabilities),minimum_stratum_probability=min(v['probability'] for v in by_stratum.values()),
        expected_root_draws=quantiles([draws*p for p in probabilities]),expected_family_draws=quantiles([draws*p for p in family_prob.values()]),
        top10_root_probability=math.fsum(sorted(probabilities,reverse=True)[:10]),top10_family_probability=math.fsum(sorted(family_prob.values(),reverse=True)[:10]),
        expected_distinct_roots=math.fsum(-math.expm1(draws*math.log1p(-p)) for p in probabilities),expected_distinct_families=math.fsum(-math.expm1(draws*math.log1p(-p)) for p in family_prob.values()),
        rare_le5=dict(roots=len(rare),families=len({rows[i]['family'] for i in rare}),probability=math.fsum(probabilities[i] for i in rare),expected_draws=draws*math.fsum(probabilities[i] for i in rare)),
        low_active_5_24=dict(roots=len(endgame),probability=math.fsum(probabilities[i] for i in endgame),expected_draws=draws*math.fsum(probabilities[i] for i in endgame)),
        strata=by_stratum,history_types=source_mass,family_size_buckets=sizes,largest_family_root_count=max(map(len,families.values())),
        largest_rootcount_family=dict(family=max(families,key=lambda f:(len(families[f]),f)),roots=max(map(len,families.values()))),
        top_families=family_rows,top_roots=[dict(**rows[i],probability=probabilities[i],expected_draws=draws*probabilities[i]) for i in root_top])


def build():
    INPUTS.clear();pin(Path(__file__));baseline=load(ROOT/'build/data-pilot-baseline.json')
    require(baseline['passed'] is True and baseline['inputs_count']==len(baseline['inputs'])==1834,'Prior diagnosis baseline incomplete')
    for name,digest in baseline['inputs'].items():require(common.sha(ROOT/name)==digest,'Prior accepted input changed '+name)
    diagnosis=load(ROOT/'build/control-diagnosis.json');admitted=load(ROOT/'build/quality-data.json');model=load(MODEL);pin(SOURCE)
    require(pin(MODEL)==baseline['fixed_model_sha256'] and model['model']['weights_sha256']=='1efd42cf39e64183752ad3bcbf3ba1b82e7e5fdc4c273426c763d4e20d746649','Fixed retained control differs')
    with closing(sqlite3.connect(SOURCE.as_uri()+'?mode=ro',uri=True)) as db:
        require(db.execute('PRAGMA quick_check').fetchone()[0]=='ok','Corrupt metadata source')
        metadata={key:dict(state_key=key,family=family,stratum=stratum) for key,family,stratum in db.execute("SELECT json_extract(body,'$.state_key'),json_extract(body,'$.family'),json_extract(body,'$.stratum') FROM roots WHERE json_extract(body,'$.split')='train'")}
    keys=admitted['train_state_keys'];rows=[metadata[key] for key in keys];counts=Counter(r['stratum'] for r in rows);groups=defaultdict(list)
    require(len(rows)==len(set(keys))==admitted['train_rows']==7669 and len({r['family'] for r in rows})==4576 and dict(sorted(counts.items()))==admitted['train_strata'] and len(counts)==14,'Original control root/family/stratum metadata differs')
    require(not any(history_type(r['family'])=='unknown' for r in rows),'Unknown history provenance')
    for i,row in enumerate(rows):groups[row['stratum']].append(i)
    strata=sorted(groups);rng=random.Random(SEED);observed=[0]*len(rows)
    for _ in range(DRAWS):observed[rng.choice(groups[rng.choice(strata)])]+=1
    actual_by_stratum={s:sum(observed[i] for i in groups[s]) for s in strata}
    prior=diagnosis['training_coverage']['quality']
    require(prior['rows']==7669 and prior['families']==4576 and prior['total_draws']==DRAWS and prior['rare_le5_draws']==54873 and actual_by_stratum=={s:v['actual_draws'] for s,v in prior['strata'].items()},'Accepted original draw replay differs')
    actual_family=defaultdict(int)
    for row,count in zip(rows,observed):actual_family[row['family']]+=count
    rare_observed=sum(observed[i] for s in strata if counts[s]<=5 for i in groups[s]);require(rare_observed==54873,'Tiny-stratum actual count differs')
    total_sqrt=math.fsum(math.sqrt(n) for n in counts.values());n=len(rows);s=len(counts);lam=MIXTURE_STRATUM_FRACTION
    probabilities={'uniform_root':[1/n for _ in rows], 'uniform_stratum':[1/(s*counts[r['stratum']]) for r in rows],
        'sqrt_count':[1/(math.sqrt(counts[r['stratum']])*total_sqrt) for r in rows],
        'root90_stratum10':[(1-lam)/n+lam/(s*counts[r['stratum']]) for r in rows]}
    summaries={name:exposure(rows,p,DRAWS) for name,p in probabilities.items()}
    for name,p in probabilities.items():require(all(p[i]>0 for i in range(n)) and abs(sum(v['expected_draws'] for v in summaries[name]['strata'].values())-DRAWS)<1e-7,'Proposal loses coverage or mass')
    empirical=dict(root_effective_support=effective_support([v/DRAWS for v in observed]),family_effective_support=effective_support([v/DRAWS for v in actual_family.values()]),
        unique_roots_drawn=sum(v>0 for v in observed),total_draws=DRAWS,rare_le5_roots=9,rare_le5_draws=rare_observed,rare_le5_fraction=rare_observed/DRAWS,
        maximum_root_draws=max(observed),maximum_family_draws=max(actual_family.values()),strata={name:dict(rows=counts[name],actual_draws=actual_by_stratum[name],maximum_root_draws=max(observed[i] for i in groups[name])) for name in strata},
        root_draw_ledger_sha256=config.digest([[row['state_key'],observed[i]] for i,row in enumerate(rows)]),family_draw_ledger_sha256=config.digest(sorted(actual_family.items())))
    for name,digest in baseline['inputs'].items():require(common.sha(ROOT/name)==digest,'Prior accepted input changed during offline review '+name)
    require('torch' not in sys.modules,'Offline review imported Torch')
    return dict(schema='data-pilot-offline-sampling-review-v1',passed=True,fixed_model=dict(path=MODEL.relative_to(ROOT).as_posix(),file_sha256=pin(MODEL),weights_sha256=model['model']['weights_sha256']),
        corpus=dict(rows=n,families=4576,strata=s,assigned_family_ids_are_not_independence_proof=True,admitted_manifest_sha256=pin(ROOT/'build/quality-data.json'),raw_metadata_only=True),
        expected_draw_budget=DRAWS,original_draw_seed=SEED,observed_accepted_run=empirical,distributions=summaries,
        definitions=dict(uniform_root='p_i=1/N',uniform_stratum='p_i=1/(S*n_stratum)',sqrt_count='stratum mass proportional sqrt(n); p_i=1/(sqrt(n_stratum)*sum_s sqrt(n_s))',root90_stratum10='p_i=0.9/N+0.1/(S*n_stratum)',mixture_stratum_fraction=lam,
            mixture_reason='Retain 10% of intentional uniform-stratum mass while reducing repeated exposure; all existing roots/strata keep positive mass',effective_support='1/sum(probability_i^2), separately at root and assigned-family level; distribution concentration only, not gradient ESS or independent game count'),
        objective_scope=dict(accepted_run='Uniform-stratum draws with unit loss weights optimize stratum-balanced empirical risk, not uniform-root risk',
            proposals='Replacing draw distribution q while retaining unit loss weights changes the empirical loss objective to E_q[loss]; this is not merely a variance reduction',
            tradeoff='Lower tiny-stratum mass reduces repetition and increases broad support while reducing deliberate rare-stratum emphasis; corpus root counts are not estimates of natural game-state frequency',
            importance_correction_applied=False,common_target_for_importance_correction=None,importance_weight_ess_computed=False),
        recommendation=dict(status='Offline proposals only; decide a future bounded learning ablation separately',active_training_sampler=None,optimizer_running=False,sampling_weights_applied=False,pilot_training_weight_applied=False,
            preserve_rare_and_low_active=True,no_rows_pruned_or_deleted=True,strength_improvement_claimed=False,
            caution='All four positive-support proposals can revisit the same finite examples; expected draws and larger effective support do not add distinct game experience or prove stronger play. Do not assign large weight to the tiny new pilot.'),
        prior_inputs_preserved=dict(inputs=1834,checked_before_and_after=True,all_hashes_match=True),training_performed=False,model_changed=False,sealed_final_test_consumed=False,default_engine='H_v1',inputs=dict(sorted(INPUTS.items())))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-only',action='store_true');args=parser.parse_args();body=build()
    if not args.check_only:common.publish(OUTPUT,body)
    print(config.canonical(dict(event='sampling_review_checked' if args.check_only else 'sampling_review_complete',rows=7669,families=4576,distributions=4,actual_tiny_root_draws=54873,training_performed=False)),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:print(config.canonical(dict(event='sampling_review_refused',error=str(exc))),flush=True);raise SystemExit(2)
