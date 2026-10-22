"""Offline probability/return review; never starts an optimizer or new games."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak.ordinary_selfplay_sampling import main
if __name__=='__main__':main()
