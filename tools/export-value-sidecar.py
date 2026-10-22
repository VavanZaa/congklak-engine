"""Initialize or verify the versioned value sidecar; no optimizer is present."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from congklak.value_sidecar import main
if __name__=='__main__':raise SystemExit(main())
