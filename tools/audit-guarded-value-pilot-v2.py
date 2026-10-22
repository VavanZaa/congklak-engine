from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import runpy
runpy.run_path(str(ROOT/'build/guarded-value-pilot-v2/audit/resource_replay_worker.py'),run_name='__main__')
