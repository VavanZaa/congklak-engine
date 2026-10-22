"""Existing-data capture/ownership diagnostics, with immutable resume commits."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from congklak.value_diagnostics import main
if __name__ == '__main__': main()
