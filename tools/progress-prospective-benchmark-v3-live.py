"""Source-bound compact observer for the protected fixed-model benchmark."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from congklak.prospective_benchmark_v3_progress_live import main
if __name__ == '__main__':
    main()
