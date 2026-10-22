"""Reserve fixed-policy value pilot metadata; never trains or loads targets."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, valuepilot_reservations as module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'verify', 'status'))
    parser.add_argument('--out', type=Path, default=module.DEFAULT)
    parser.add_argument('--pause-at', choices=('before_catalog_commit', 'after_catalog_commit'))
    parser.add_argument('--pause-marker', type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        result = module.run(args.out, args.pause_at, args.pause_marker)
    else:
        if args.pause_at or args.pause_marker: parser.error('Pause hooks require run')
        result = getattr(module, args.command)(args.out)
    print(config.canonical(dict(event='valuepilot_reservations_' + args.command,
                                status=result.get('status', 'complete' if args.command in ('run', 'verify') else 'prepared'),
                                counts=result.get('counts'), optimizer_steps=0, training_started=False)), flush=True)


if __name__ == '__main__': main()
