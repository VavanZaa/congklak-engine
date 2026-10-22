"""Initialize or verify admitted v2 value sidecar; no optimizer exists."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from congklak.value_sidecar_v2 import main
if __name__=='__main__':raise SystemExit(main())
