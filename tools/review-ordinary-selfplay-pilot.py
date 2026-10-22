"""Read-only existing-arena and subsequent offline sampling review."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak import ordinary_selfplay_review as review

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('arena',));args=parser.parse_args()
    review.arena_main()
