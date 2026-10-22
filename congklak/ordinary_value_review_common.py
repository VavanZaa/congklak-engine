"""Identity and immutable publication helpers for the saved-data value review."""
from pathlib import Path
import hashlib
import json
from . import config
from .native import ROOT
from .deeper_teacher_v2 import atomic_json

BASELINE = ROOT / 'build/ordinary-value-review-baseline.json'
SCOPE = ROOT / 'docs/ORDINARY_VALUE_REVIEW_SCOPE.md'


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pins(names):
    return {str(Path(name).relative_to(ROOT) if Path(name).is_absolute() else Path(name)).replace('\\', '/'):
            sha(Path(name) if Path(name).is_absolute() else ROOT / name) for name in names}


def current(mapping):
    for name, digest in mapping.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, 'Input changed: ' + name)


def baseline(verify_all=False):
    body = read(BASELINE)
    require(body['schema'] == 'ordinary-value-review-prior-immutability-v1' and body['passed'] is True
            and len(body['inputs']) == body['inputs_count'] == 3779, 'Wrong review baseline')
    if verify_all: current(body['inputs'])
    return body


def publish(path, body):
    path = Path(path)
    if path.exists():
        require(path.read_text(encoding='utf-8') == config.canonical(body) + '\n', 'Immutable review output differs: ' + str(path))
    else: atomic_json(path, body)
    return body
