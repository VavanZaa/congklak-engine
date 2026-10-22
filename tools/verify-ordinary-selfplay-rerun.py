"""Verify a closed ordinary-data pilot without new transactions, searches or Torch."""
import contextlib
import hashlib
import io
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, quality_search, ordinary_selfplay_pilot as producer
from congklak import ordinary_selfplay_review as review, ordinary_selfplay_sampling as sampling
from congklak.native import Native
from congklak.neural_relay import NeuralNative


def signature(path):
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)


def paths():
    result = [producer.BASELINE, producer.REGISTRY, producer.PLAN, producer.IMPLEMENTATION,
              producer.COLLECTION, producer.PROTECTION, producer.OUTPUT,
              review.ARENA_OUTPUT, sampling.OUTPUT,
              ROOT / 'data/quality-train-v1/model.json',
              ROOT / 'data/selfplay-training-pilot-v1/model.json']
    result += [p for p in producer.DEFAULT.rglob('*') if p.is_file() and not p.name.endswith(('-wal', '-shm'))]
    return sorted(set(result))


def support_comparison(rows, old_rows):
    old = [row for row in old_rows if row['stratum'].endswith('/ordinary')]
    def compare(new_values, old_values):
        new_states = {config.canonical(row['state']) for row in new_values}
        old_states = {config.canonical(row['state']) for row in old_values}
        return dict(new_observations=len(new_values), old_observations=len(old_values),
                    unique_within_new=len(new_states), old_unique=len(old_states),
                    literal_raw_state_overlap=len(new_states & old_states),
                    novel_vs_old=len(new_states - old_states),
                    combined_unique=len(new_states | old_states))
    return dict(total=compare(rows, old), by_stratum={kind: compare(
        [row for row in rows if row['stratum'] == kind],
        [row for row in old if row['stratum'] == kind])
        for kind in producer.STRATA},
        interpretation='Literal full-state overlap with the prior admitted ordinary replay only; no claim of novelty against every historically generated or ancestral position.')


def main():
    watched = paths()
    assert all(path.is_file() for path in watched), 'Ordinary pilot is not closed'
    before = {p.relative_to(ROOT).as_posix(): signature(p) for p in watched}
    capture = io.StringIO()
    events = []
    with (patch.object(quality_search, 'search', side_effect=AssertionError('No new PUCT decisions')) as searches,
          patch.object(quality_search, 'forest', side_effect=AssertionError('No new PUCT forests')) as forests,
          patch.object(Native, 'analyze', side_effect=AssertionError('No teacher/baseline search')) as native,
          patch.object(NeuralNative, 'analyze', side_effect=AssertionError('No neural alpha-beta search')) as neural):
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            producer.collect(on_event=events.append)
            producer.generate(on_event=events.append)
            producer.load_rows()
            _, rows = producer.load_replay()
            from congklak import selfplay_training_pilot as old
            _, old_rows = old.load_replay()
            support = support_comparison(rows, old_rows)
            review.arena_main()
            sampling.main()
    assert searches.call_count == forests.call_count == native.call_count == neural.call_count == 0
    assert not any(row['event'].endswith(('_precommit', '_committed')) for row in events), 'Closed API committed work'
    assert 'torch' not in sys.modules, 'Closed readers imported Torch'
    assert paths() == watched, 'Closed readers added durable files'
    after = {p.relative_to(ROOT).as_posix(): signature(p) for p in watched}
    assert before == after, 'Closed readers changed bytes or modification times'
    baseline = producer.read(producer.BASELINE)
    assert all(producer.sha(ROOT / name) == digest for name, digest in baseline['inputs'].items())
    log = ROOT / 'build/ordinary-selfplay-rerun.log'
    if log.exists():
        assert log.read_bytes() == capture.getvalue().encode('utf-8')
    else:
        log.write_bytes(capture.getvalue().encode('utf-8'))
    proof = dict(schema='ordinary-selfplay-completed-no-work-v1', passed=True,
                 prior_inputs_preserved=len(baseline['inputs']), before=before, after=after,
                 completed_bytes_mtimes_unchanged=True, new_collection_commits=0,
                 new_selfplay_decisions=0, new_teacher_searches=0, new_optimizer_steps=0,
                 torch_imported=False, saved_history_revalidation_allowed=True,
                 ordinary_support_comparison=support,
                 sources={name: producer.sha(ROOT / name) for name in (
                     'tools/verify-ordinary-selfplay-rerun.py', 'congklak/ordinary_selfplay_pilot.py',
                     'congklak/ordinary_selfplay_review.py', 'congklak/ordinary_selfplay_sampling.py')})
    producer.publish(ROOT / 'build/ordinary-selfplay-root-verification.json', proof)
    print(config.canonical(dict(event='ordinary_selfplay_completed_rerun_verified', passed=True,
                               durable_files=len(watched), prior_inputs_preserved=len(baseline['inputs']),
                               new_searches=0, new_optimizer_steps=0, torch_imported=False)), flush=True)


if __name__ == '__main__':
    main()
