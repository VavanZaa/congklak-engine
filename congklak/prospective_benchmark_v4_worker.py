"""Actual contained fixed-model benchmark worker entry."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
from congklak.prospective_benchmark_v4_execute import main
if __name__=='__main__':main()
