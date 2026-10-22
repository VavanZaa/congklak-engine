"""Independent saved-byte and recovery-route checks; no engine execution."""
import base64
import copy
import ctypes
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from congklak import config, footprint_screen_review as r

OLD = r.s.ROOT/'build/ordinary-epoch-pilot-v1/jobs/labels'


class IndependentReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identity = r.read(OLD/'identity.json')

    def test_saved_complete_teacher_feature_and_rejected_footprints(self):
        for index in (0, 1, 4):
            path = OLD/('failed-attempts' if index == 4 else 'archives')/f'query-{index:06d}.json'
            body = r.read(path)
            kind = 'teacher' if index % 2 == 0 else 'features'
            with patch.object(ctypes, 'CDLL', side_effect=AssertionError('DLL')), \
                 patch.object(ctypes, 'WinDLL', side_effect=AssertionError('Windows DLL')):
                proof = r.raw_proof(body['answer'], self.identity['schedule'][index]['meta'], kind)
            self.assertEqual(proof['events'], body['answer']['trace']['event_count'])
            self.assertEqual(proof['owned_count'], body['answer']['fast_certificate']['owned_count'])
            if index == 4:
                self.assertEqual(proof['owned_count'], 6669)
                self.assertIn('010002000103050101010007030000', proof['quotients'])

    def test_incomplete_compression_and_wrong_owned_bytes_refuse(self):
        original = r.read(OLD/'archives/query-000000.json')['answer']
        for change in ('raw_suffix', 'owned', 'root', 'forward'):
            meta = self.identity['schedule'][0]['meta']
            answer = copy.deepcopy(original)
            if change == 'raw_suffix':
                compressed = base64.b64decode(answer['trace']['data']) + b'x'
                answer['trace']['data'] = base64.b64encode(compressed).decode()
                answer['trace']['compressed_bytes'] = len(compressed)
            elif change == 'owned':
                owned = bytearray(base64.b64decode(answer['fast_owned_high']))
                owned[-1] ^= 1
                answer['fast_owned_high'] = base64.b64encode(owned).decode()
            elif change == 'root':
                meta = dict(meta, state=dict(meta['state'], controller=1-meta['state']['controller']))
            else:
                answer['trace']['operation_counts']['model_done'] = 1
            with self.subTest(change=change), self.assertRaises(Exception):
                r.raw_proof(answer, meta, 'teacher')

    def test_declared_high_domain_includes_upper_states_and_excludes_low(self):
        # A finite raw-only fixture keeps full98 stores, aliases and actor.
        cells = bytes([7]*7+[0]+[7]*7+[0])
        low = bytes([1]*7+[42]+[1]*7+[42])
        raw = r.PACK.pack(cells, 0, 31, 1, 0, -1, -1)+r.PACK.pack(low, 0, 32, 1, 0, -1, -1)
        q = min(cells[:7]+cells[8:15]+b'\0', cells[8:15]+cells[:7]+b'\1')
        owned = r.OWNED.pack(q, 0, cells, 0)
        import zlib
        compressed = zlib.compress(raw)
        operations = {name: 0 for name in r.OP_NAMES}
        operations.update(played_input=1, played_output=1)
        digest = hashlib.sha256(raw).hexdigest()
        cert = dict(event_sha256=digest, event_count=2, attempted_events=2, owned_count=1,
                    owned_sha256=hashlib.sha256(owned).hexdigest(), owned_occurrences=1,
                    operation_counts=[0]+[operations[n] for n in r.OP_NAMES]+[0]*26)
        answer = dict(trace=dict(event_count=2, event_bytes=24, encoding='base64/zlib/packed24/little-endian',
                      capacity_events=16000000, data=base64.b64encode(compressed).decode(),
                      compressed_bytes=len(compressed), event_sha256=digest, overflow=False, invalid=False,
                      truncated=False, attempted_events=2, operation_counts=operations),
                      fast_certificate=cert, fast_owned_high=base64.b64encode(owned).decode())
        meta = dict(state=dict(pits=[[7]*7,[7]*7], stores=[0,0], controller=0))
        proof = r.raw_proof(answer, meta, 'teacher')
        self.assertEqual((proof['owned_count'], proof['upper_owned_events'], proof['low_shared_events']), (1,1,1))

    def test_completed_route_precedes_lease_writer_and_replay(self):
        done = {'report': {'passed': True}}
        with patch.object(r, 'has_completion', return_value=True), patch.object(r, 'publish_completed', return_value=done), \
             patch.object(r, 'Writer', side_effect=AssertionError('Writer')), \
             patch.object(r, 'compute', side_effect=AssertionError('repeated raw review')):
            self.assertIs(r.run(lease=None), done)

    def test_orphan_unit_is_adopted_without_recomputation(self):
        parent = r.s.FOLDER/'pure-audit-tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            path = Path(temp)
            unit = dict(index=0, proof='fixed saved bytes', events=0, owned_count=0)
            identity = dict(tasks=[dict(index=0)], fixture=True)
            r.atomic(path/'identity.json', identity)
            r.atomic(path/'units/000000.json', unit)
            with patch.object(r, '_unit_valid', return_value=None), patch.object(r, 'compute', side_effect=AssertionError('repeat')):
                units, _ = r._prefix(path, identity)
                self.assertEqual(units, [])
                original = path/'units/000000.json'
                before = original.stat().st_mtime_ns
                r._append(path, identity, units, r.read(original))
                self.assertEqual(original.stat().st_mtime_ns, before)
                self.assertEqual(r._prefix(path, identity)[0], [r._thin(unit)])

    def test_final_union_refuses_missing_high_footprint(self):
        identity = dict(tasks=[1], fixture=True, pool_sha256='pool', snapshot_sha256='snap',
                        plan_sha256='plan', source_inputs={}, review_sources={})
        unit = dict(index=0, root_index=0, quotients=[], events=0, high_owned_events=0, low_shared_events=0,
                    upper_owned_events=0, archive_ref={},)
        snap = dict(roots=[dict(index=0, quotients=['010002000103050101010007030000'])])
        with self.assertRaisesRegex(Exception, 'full root footprint union'):
            r._report(identity, [unit], snap)

    def test_catalog_only_republication_preserves_proofs_and_report(self):
        parent = r.s.FOLDER/'pure-audit-tests'
        parent.mkdir(parents=True, exist_ok=True)
        for boundary in ('before_catalog', 'after_catalog'):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory(dir=parent) as temp:
                path = Path(temp)
                identity = {'fixture': True}
                report = {'passed': True, 'query_count': 1}
                r.atomic(path/'identity.json', identity)
                r.atomic(path/'units/000000.json', {'saved': 'exact raw proof'})
                def pause(point):
                    if point == boundary:
                        raise InterruptedError(point)
                with patch.object(r, '_report', return_value=report), \
                     patch.object(r, 'compute', side_effect=AssertionError('repeated review')):
                    with self.assertRaises(InterruptedError):
                        r._finish(path, identity, [], {}, pause)
                    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.rglob('*.json')}
                    r._finish(path, identity, [], {})
                    self.assertTrue((path/'catalog.json').exists())
                    for p, value in before.items():
                        self.assertEqual((p.read_bytes(), p.stat().st_mtime_ns), value)


if __name__ == '__main__':
    unittest.main()
