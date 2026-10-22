"""Direct contained entry: the durable argv[0] is the actual worker script."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
from congklak.prospective_eval_execute_v2 import main
if __name__=='__main__':main()
