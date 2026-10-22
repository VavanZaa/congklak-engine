"""One explicit full prior-byte inspection; never opens a native provider."""
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, footprint_screen_common as s


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    destination = s.FOLDER/'audit/prior-preservation-v1.json'
    s.need(not destination.exists(), 'Full preservation inspection already recorded')
    started = time.perf_counter()
    cpu = time.process_time()
    baseline = s.read(s.FOLDER/'baseline.json')
    s.need(file_hash(s.PRIOR) == s.PRIOR_SHA, 'Original held-run marker changed')
    old = s.read(s.PRIOR)
    expected = dict(old['inputs'])
    expected[s.relative(s.PRIOR)] = s.PRIOR_SHA
    s.need(config.canonical(baseline['inputs']) == config.canonical(expected) and
           baseline['inputs_count'] == len(expected) == 81047,
           'Baseline omits or rebases original expected digests')
    failures = []
    total_bytes = 0
    for index, (name, digest) in enumerate(sorted(expected.items())):
        path = ROOT/name
        try:
            actual = file_hash(path)
            total_bytes += path.stat().st_size
            if actual != digest:
                failures.append(dict(path=name, expected=digest, actual=actual))
        except OSError as error:
            failures.append(dict(path=name, error=str(error)))
        if (index+1) % 10000 == 0:
            print(config.canonical(dict(event='independent_prior_bytes', files=index+1, failures=len(failures))), flush=True)
    models = {name: file_hash(ROOT/name) for name in s.MODELS}
    fixed = all(expected.get(name) == digest for name, digest in models.items())
    result = dict(schema=s.SCHEMA+'-independent-prior-preservation', passed=not failures and fixed,
                  independent_of_implementation=True, prior_sha256=s.PRIOR_SHA,
                  expected_manifest_sha256=config.digest(expected), inputs_verified=len(expected),
                  bytes_verified=total_bytes, failures=failures, models=models, models_fixed=fixed,
                  cache_used=False, elapsed_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu,
                  cost_scope='explicit read-only full-byte inspection outside native worker clocks',
                  new_queries=0, model_forwards=0, optimizer_steps=0,
                  source_inputs=s.common.pins([Path(__file__), s.FOLDER/'baseline.json', s.PRIOR]))
    s.atomic(destination, result)
    s.need(result['passed'], 'Prior expected bytes changed; retained independent failure')
    print(config.canonical(dict(passed=True, files=len(expected), bytes_verified=total_bytes,
                               elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
