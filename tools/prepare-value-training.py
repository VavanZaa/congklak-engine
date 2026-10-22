"""Prepare or verify role-separated H8 value-training data; never train."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from congklak import value_training_data as data
p = argparse.ArgumentParser(); p.add_argument('command', choices=('prepare', 'verify'))
p.add_argument('--output', type=Path, default=data.DEFAULT)
p.add_argument('--pause-at', choices=data.HOOKS); p.add_argument('--pause-marker', type=Path)
a = p.parse_args()
r = data.prepare(a.output, a.pause_at, a.pause_marker) if a.command == 'prepare' else data.admit(a.output)
print(json.dumps(dict(schema=r['schema'], counts={k:v['count'] for k,v in r['role_files'].items()}, optimizer_steps=r['optimizer_steps'], policy_fixed=r['policy_fixed']), sort_keys=True))
