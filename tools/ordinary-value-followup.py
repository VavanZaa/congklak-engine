"""Freeze, run or verify the single conditional ordinary-value followup batch."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, ordinary_value_followup as p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('plan', 'freeze', 'run', 'verify'))
    parser.add_argument('--run', default=str(p.DEFAULT)); parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--decision-limit', type=int)
    args = parser.parse_args(); stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, 'SIGTERM'): signal.signal(signal.SIGTERM, stop)
    def event(value):
        print(config.canonical(value), flush=True)
        if value['event'] == os.environ.get('ORDINARY_VALUE_FOLLOWUP_BLOCK_EVENT') and str(value.get('committed_decisions', value.get('completed'))) == os.environ.get('ORDINARY_VALUE_FOLLOWUP_BLOCK_STEP'):
            from congklak.deeper_teacher_v2 import atomic_json
            atomic_json(Path(os.environ['ORDINARY_VALUE_FOLLOWUP_MARKER']), value)
            while True: time.sleep(.05)
    if args.operation == 'plan': p.plan()
    elif args.operation == 'freeze': p.freeze()
    elif args.operation == 'verify':
        body, rows = p.load_run(args.run, args.fixture)
        event(dict(event='ordinary_value_followup_verified', status=body['status'], rows=len(rows), torch_imported='torch' in sys.modules))
    else: p.run(args.run, args.fixture, event, lambda: stopped, args.decision_limit)
    return 0


if __name__ == '__main__': raise SystemExit(main())
