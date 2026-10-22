"""Small engineering check of the sequential opening; never training or proof data."""
from pathlib import Path
import hashlib
import json
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from congklak import config
from congklak.native import Native, ROOT, flatten

OUT = ROOT / 'build' / 'product-evaluation-v1' / 'first-player.json'
SETTINGS = dict(max_depth=8, time_ms=0, node_budget=30000, memory_mb=512, tt_mb=16, use_pvs=True)


def main():
    native = Native()
    identity = dict(native_sha256=native.build_id, rules_sha256=config.RULES_DIGEST,
                    settings=SETTINGS, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    value = json.loads(OUT.read_text()) if OUT.exists() else dict(identity=identity, games=[])
    assert value['identity'] == identity
    started = time.monotonic()
    for seed in (11, 29, 47, 83):
        for first in (0, 1):
            if any(g['seed'] == seed and g['first'] == first for g in value['games']):
                continue
            rng = random.Random(seed)
            p = config.position(); p['controller'] = first
            moves = []
            for _ in range(400):
                assert time.monotonic() - started < 90, 'Bounded diagnostic time limit reached'
                normal = native.normalize(flatten(p), actor=p['controller'], total=98)
                assert normal['ok']
                if normal['terminal']:
                    break
                p = normal['position']
                result = native.analyze(flatten(p), actor=p['controller'], total=98,
                                        keep_history=False, **SETTINGS)
                report = result.get('report')
                assert result['ok'] and report and report['completed_iteration'] and report['best_moves']
                action = rng.choice(sorted(report['best_moves']))
                step = native.apply(flatten(p), actor=p['controller'], action=action, total=98, trace=False)
                assert step['ok']
                moves.append(dict(actor=p['controller'], pit=action, depth=report['choice_depth']))
                p = step['position']
            else:
                raise AssertionError('Diagnostic game exceeded 400 choices')
            winner = None if p['stores'][0] == p['stores'][1] else (0 if p['stores'][0] > p['stores'][1] else 1)
            game = dict(seed=seed, first=first, winner=winner, stores=p['stores'], moves=moves)
            value['games'].append(game)
            value.update(complete=len(value['games']) == 8, research_training_eligible=False,
                         scope='Small sampled fixed-node H_v1 mirror check; not optimal-play proof or calibrated win rates.')
            temporary = OUT.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2)+'\n')
            temporary.replace(OUT)
            print(json.dumps(dict(seed=seed, first=first, winner=winner, stores=p['stores'], choices=len(moves))), flush=True)
    games=value['games']
    print(json.dumps(dict(games=len(games), starter_wins=sum(g['winner']==g['first'] for g in games),
                          second_wins=sum(g['winner'] is not None and g['winner']!=g['first'] for g in games),
                          draws=sum(g['winner'] is None for g in games))))


if __name__ == '__main__':
    main()
