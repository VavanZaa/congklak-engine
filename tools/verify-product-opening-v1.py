"""Verify a constructive win for the starter in the declared sequential profile."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tests'))
import reference as literal
from congklak import config
from congklak.native import Native, flatten

# One-based labels: 1,7,4,1,6,6,3,4,7,2,4,1,7,6.
LINE = [0, 6, 3, 0, 5, 5, 2, 3, 6, 1, 3, 0, 6, 5]


def main():
    config.verify_rules()
    native = Native(); certificates = []
    for first in (0, 1):
        p = config.position(); p['controller'] = first
        start = p; ref = literal.Position(7, tuple(flatten(p)), first)
        steps = []
        for number, action in enumerate(LINE, 1):
            assert ref.controller == p['controller'] == first
            expected = literal.apply(ref, action)
            actual = native.apply(flatten(p), actor=first, action=action, total=98, trace=True)
            assert actual['ok']
            assert tuple(flatten(actual['position'])) == expected.position.counts
            assert actual['position']['controller'] == expected.position.controller == first
            assert expected.bonus and actual['reason'] == 'own_store'
            assert not expected.initial_passes and not expected.final_passes and not actual['passes']
            assert not expected.terminal and not actual['terminal']
            assert sum(expected.position.counts) == 98
            assert actual['position']['stores'][1-first] == 0
            p, ref = actual['position'], expected.position
            steps.append(dict(choice=number, pit=action, after=p, bonus=True))
        assert p['stores'][first] == 51 and p['stores'][1-first] == 0
        certificates.append(dict(first=first, start=start, steps=steps, secured=51,
                                 opponent_choices_before_win=0, remaining_seeds=47))
    receipt = dict(passed=True, schema='product-sequential-opening-certificate-v1',
                   profile=config.PROFILE, rules_digest=config.RULES_DIGEST,
                   native_sha256=native.build_id, one_based_line=[a+1 for a in LINE],
                   certificates=certificates, independent_literal_transitions=28,
                   conclusion='The starter has a constructive winning strategy in this exact sequential single-round profile: bank 51 of 98 seeds before the opponent can choose a move. This does not mean every opening move wins, or apply to other rule profiles.',
                   sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                            ['tools/verify-product-opening-v1.py','tests/reference.py','rules/id-betawi-relay7-v0.json']},
                   research_training_eligible=False, native_searches=0)
    path=ROOT/'build/product-evaluation-v1/opening-certificate.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(passed=True,independent_literal_transitions=28,secured_before_opponent_moves=51,
                          line_one_based=receipt['one_based_line'])))


if __name__ == '__main__':
    main()
