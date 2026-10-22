"""Resource-supervised ordinary-only value worker entry."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak.ordinary_epoch_training import main
if __name__=='__main__':raise SystemExit(main())
