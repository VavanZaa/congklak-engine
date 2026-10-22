"""Publish or verify the saved-data value review's next gates."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, ordinary_value_review_conclusions as review

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('run', 'verify'))
    parser.add_argument('--output', type=Path, default=review.OUTPUT)
    args = parser.parse_args()
    body = review.run(args.output) if args.operation == 'run' else review.verify(args.output)
    print(config.canonical(dict(event='ordinary_value_review_conclusions_' + args.operation,
                               passed=True, summary=body['summary'])), flush=True)
