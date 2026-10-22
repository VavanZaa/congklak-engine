"""Versioned ownership engineering launcher with unconditional management routing."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak.ownership_handoff_launcher_v2 import main
if __name__=='__main__':raise SystemExit(main())
