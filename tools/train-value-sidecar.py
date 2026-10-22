"""Fixed value-sidecar worker. Use the independent resource supervisor to run."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from congklak.value_training import main
raise SystemExit(main())
