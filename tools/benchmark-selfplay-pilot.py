"""Run the frozen self-play pilot equal-time benchmark."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak.selfplay_pilot_benchmark import main
if __name__=="__main__":raise SystemExit(main())
