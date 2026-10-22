"""Run the versioned quota-safe wrapper around frozen arena enforcement."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,value_run_watchdog_v2 as watchdog
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lease',required=True)
a=p.parse_args()
print(config.canonical(watchdog.watchdog(a.lease)),flush=True)
