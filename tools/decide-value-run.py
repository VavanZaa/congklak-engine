"""Read or publish the predeclared full value-pilot decision."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from congklak.value_run_decision import run

result = run()
print(json.dumps({key: result[key] for key in ('evidence_complete', 'pilot_promising',
    'failed_gates', 'candidate_quarantined', 'strength_interpretation_allowed', 'promotion')}, sort_keys=True))
