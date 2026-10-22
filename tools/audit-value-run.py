"""Independent verification of the single fixed-final value-training pilot."""
from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
spec = importlib.util.spec_from_file_location('independent_value_run_audit', ROOT / 'tests/test_value_run_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=('training', 'fp32', 'closed'))
    parser.add_argument('--spec')
    parser.add_argument('--output')
    parser.add_argument('--methods', nargs='+')
    args = parser.parse_args()
    if args.worker:
        getattr(audit, args.worker + '_worker')(audit.read(args.spec))
        return 0
    result = audit.run(args.output, args.methods)
    print(audit.canonical({k: result[k] for k in ('passed', 'tests_run', 'failed', 'errors', 'source_unchanged')}))
    return 0 if result['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
