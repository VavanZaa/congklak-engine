"""Freeze or publish the fixed-work/action-Q comparison without search."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak import config,divergence_comparison as study
p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('freeze','run','verify'))
a=p.parse_args();r=getattr(study,a.operation)()
print(config.canonical(dict(event='divergence_comparison_'+a.operation,passed=True)),flush=True)
