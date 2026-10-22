"""Source-bound diagnostic optimizer replay, never a model producer."""
from pathlib import Path
import importlib.util
path = Path(__file__).resolve().parents[1] / 'tests/ordinary_epoch_training_replay.py'
spec = importlib.util.spec_from_file_location('ordinary_epoch_independent_optimizer_replay', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if __name__ == '__main__':
    module.main()
