import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.dont_write_bytecode = True
from congklak.benchmark_loss_review_v1_progress import main
if __name__ == '__main__':
    main()
