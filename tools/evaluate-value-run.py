"""Execute the fixed-final value pilot's recoverable diagnostics."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak.value_run_cases import main
if __name__=='__main__':raise SystemExit(main())
