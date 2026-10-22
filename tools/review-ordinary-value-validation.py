"""Rebuild or verify the immutable saved ordinary-value validation review."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,ordinary_value_validation_review as p


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",default=str(p.OUTPUT));parser.add_argument("--verify",action="store_true");args=parser.parse_args();body=p.verify(args.output) if args.verify else p.run(args.output);print(config.canonical(dict(event="ordinary_value_validation_review_verified" if args.verify else "ordinary_value_validation_review_complete",summary=body["summary"],sha256=body["sha256"])));return 0


if __name__=="__main__":raise SystemExit(main())
