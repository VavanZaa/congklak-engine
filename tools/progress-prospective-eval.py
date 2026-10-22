"""Source-bound local observer entry for the prospective evaluation campaign."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak.prospective_eval_progress import main
if __name__=='__main__':main()
