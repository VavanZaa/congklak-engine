"""Backend-only read-only overlay proof; no browser or HTTP inspection."""
from pathlib import Path
import json
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,selfplay_pilot_benchmark as benchmark,quality_search,selfplay_training_pilot as learning,selfplay_pilot_live_progress as observer
from congklak.native import Native
from congklak.neural_relay import NeuralNative,sha


def main():
    implementation=benchmark.load(benchmark.IMPLEMENTATION)
    files=set(implementation['sources'])|set(implementation['science_inputs'])|{
        'congklak/selfplay_pilot_live_progress.py','tools/verify-selfplay-pilot-observer.py','congklak/selfplay_training_pilot.py',
        'build/selfplay-training-pilot-plan-v2.json','data/selfplay-training-pilot-v1/work.sqlite3','data/selfplay-training-pilot-v1/selfplay-manifest.json',
        'data/selfplay-training-pilot-v1/catalog.sqlite3','data/selfplay-training-pilot-v1/model.json','data/selfplay-training-pilot-v1/result.json',
        'data/quality-train-v1/model.json','build/selfplay-training-pilot-model-witness.json','build/selfplay-training-pilot-selfplay.json',
        'data/selfplay-pilot-benchmark-v1/work.sqlite3','build/selfplay-pilot-benchmark.json','build/selfplay-pilot-benchmark-resources.json'}
    def record():return {name:dict(sha256=sha(ROOT/name),size=(ROOT/name).stat().st_size,mtime_ns=(ROOT/name).stat().st_mtime_ns) for name in sorted(files)}
    before=record()
    def prohibited(*args,**kwargs):raise AssertionError('Observer attempted search, training or runtime activation')
    with patch.object(Native,'analyze',prohibited),patch.object(NeuralNative,'analyze',prohibited),patch.object(quality_search,'search',prohibited),patch.object(benchmark,'decide',prohibited),patch.object(benchmark,'puct_search',prohibited),patch.object(learning,'generate',prohibited),patch.object(learning,'train',prohibited),patch.object(learning,'runtime',prohibited):
        value=observer.snapshot();html=observer.html_bytes()
    after=record();assert before==after and 'torch' not in sys.modules
    ids=[card['id'] for card in value['tools'][:8]];assert len(ids)==len(set(ids))==8 and all(name.startswith('selfplay-pilot-') for name in ids)
    cards={card['id']:card for card in value['tools'][:8]};sp=json.loads((ROOT/'build/selfplay-training-pilot-selfplay.json').read_text());training=json.loads((ROOT/'data/selfplay-training-pilot-v1/result.json').read_text());plan=json.loads((ROOT/'build/selfplay-training-pilot-plan-v2.json').read_text());result=json.loads((ROOT/'build/selfplay-pilot-benchmark.json').read_text())
    summary=sp['summary'];batch=plan['contract']['training']['batch_size'];assert value['observer_corrections']['training_batch_size']==batch
    assert str(summary['terminal_games'])+'/32 terminal' in cards['selfplay-pilot-generation']['result'] and str(summary['committed_decisions'])+' durable decisions' in cards['selfplay-pilot-generation']['result']
    assert str(summary['admitted_families'])+'/16 admitted families' in cards['selfplay-pilot-replay']['result'] and str(summary['retained_rows'])+' retained observations' in cards['selfplay-pilot-replay']['result']
    assert 'batch '+str(batch) in cards['selfplay-pilot-training']['result'] and str(training['draws'])+' replacement draws' in cards['selfplay-pilot-training']['result']
    assert str(result['games'])+'/96 full game traces committed' in cards['selfplay-pilot-strength']['result'] and str(result['decisions'])+' durable decisions' in cards['selfplay-pilot-strength']['result']
    assert b'congklak.selfplay_pilot_live_progress serve' in html
    body=dict(schema='selfplay-pilot-observer-readonly-proof-v1',passed=True,source_inputs={name:sha(ROOT/name) for name in ('congklak/selfplay_pilot_live_progress.py','tools/verify-selfplay-pilot-observer.py')},
        frozen_producer_sources=implementation['sources'],before=before,after=after,all_bytes_sizes_mtimes_unchanged=True,eight_current_card_ids=ids,
        actual_selfplay_counts={name:summary[name] for name in ('games','terminal_games','capped_games','committed_decisions','admitted_families','retained_rows','rawstate_unique')},
        actual_training_batch=batch,actual_training_draws=training['draws'],actual_benchmark_games=result['games'],actual_benchmark_decisions=result['decisions'],actual_benchmark_measurements=result['scalar_measurements'],
        initialized_and_terminal_counts_separated=True,nested_summary_used=True,torch_imported=False,new_searches=0,new_optimizer_or_runtime_calls=0,
        browser_or_HTTP_inspection=False,derived_HTML_changes_only_observer_command_module=True,derivative_evidence_not_an_additional_test_method=True)
    benchmark.common.publish(ROOT/'build/selfplay-pilot-observer-proof.json',body)
    print(config.canonical(dict(event='selfplay_pilot_observer_verified',passed=True,durable_files=len(before),cards=8,batch=batch,draws=training['draws'])),flush=True)


if __name__=='__main__':main()
