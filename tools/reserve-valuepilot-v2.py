"""Reserve the joint fixed-policy pilot with clean ordinary development starts."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, valuepilot_reservations_v2 as module, ordinary_development


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'verify', 'status'))
    parser.add_argument('--out', type=Path, default=module.DEFAULT)
    parser.add_argument('--development', type=Path, default=ordinary_development.DEFAULT)
    parser.add_argument('--pause-at', choices=('before_catalog_commit', 'after_catalog_commit'))
    parser.add_argument('--pause-marker', type=Path)
    args = parser.parse_args()
    if args.command == 'run': result = module.run(args.out, args.pause_at, args.pause_marker)
    else:
        if args.pause_at or args.pause_marker: parser.error('Pause hooks require run')
        result = module.prepare(args.out, args.development) if args.command == 'prepare' else getattr(module, args.command)(args.out)
    print(config.canonical(dict(event='valuepilot_reservations_v2_' + args.command, status=result.get('status', 'prepared'), counts=result.get('counts'), optimizer_steps=0, training_started=False)), flush=True)


if __name__ == '__main__': main()
