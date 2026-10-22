"""Review/prepare the bounded value pilot without teacher or optimizer work."""
from pathlib import Path
import sys
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, value_pilot_data
from congklak.native import Native

if __name__ == '__main__':
    with patch.object(Native, 'analyze', side_effect=AssertionError('Data review cannot run teacher searches')) as calls:
        manifest, groups = value_pilot_data.build()
        assert calls.call_count == 0
    print(config.canonical(dict(event='value_pilot_data_reviewed', passed=True,
        groups={k:len(v) for k,v in groups.items()}, base_rows=manifest['base_rows'],
        ordinary_train_rows=manifest['new_ordinary_train_rows'], no_teacher_searches=True)), flush=True)
