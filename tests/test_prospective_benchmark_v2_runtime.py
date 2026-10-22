"""Pure resource-boundary tests; no worker, model or native query is launched."""
import copy
import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from congklak import config, prospective_benchmark_v2_runtime as r


class ResourceContract(unittest.TestCase):
    def lease(self, fixture=False, operation='arena'):
        output = (r.stage.FOLDER/'fixtures/native-recovery/before_handoff_commit' if fixture
                  else r.stage.FOLDER/'jobs'/operation)
        limit = r._purpose_limit(fixture, operation)
        return dict(worker_entry=str(r.WORKER), worker_entry_sha256='0'*64,
            plan_sha256='0'*64, protocol=r.PROTOCOL, fixture=fixture,
            worker_output=str(output), cache_path=str(r.stage.DATA/'cache'),
            extra_args=['--operation', operation], purpose='recovery' if fixture else operation,
            purpose_limit_ns=limit, reserved_ns=limit)

    def test_distinct_purpose_limits(self):
        self.assertEqual(r._purpose_limit(False, 'engineering'), 120_000_000_000)
        self.assertEqual(r._purpose_limit(True, 'arena'), 120_000_000_000)
        self.assertEqual(r._purpose_limit(False, 'arena'), 1_800_000_000_000)
        self.assertEqual(r._purpose_limit(False, 'endgame'), 1_800_000_000_000)
        for fixture, operation in ((True, 'engineering'), (True, 'endgame'), (False, 'training')):
            with self.assertRaises(ValueError):
                r._purpose_limit(fixture, operation)

    def test_fixture_cannot_take_production_reservation(self):
        lease = self.lease(fixture=True)
        with patch.object(r, 'sha', return_value='0'*64):
            r._validate_lease_shape(lease, dict(protocol=r.PROTOCOL))
            lease['reserved_ns'] = r.PROTOCOL['lease_ns']
            with self.assertRaises(ValueError):
                r._validate_lease_shape(lease, dict(protocol=r.PROTOCOL))

    def test_historical_option_tampering_is_rejected(self):
        for args in (['--operation'], ['--operation', 'arena', '--operation', 'arena'],
                     ['--operation', 'arena', '--max-nodes', '1'], ['--operation', 'training']):
            lease = self.lease(); lease['extra_args'] = args
            with patch.object(r, 'sha', return_value='0'*64), self.assertRaises(ValueError):
                r._validate_lease_shape(lease, dict(protocol=r.PROTOCOL))

    def test_historical_scope_tampering_is_rejected(self):
        for field, value in (('worker_output', str(r.stage.FOLDER/'jobs/not-registered')),
                             ('cache_path', str(r.stage.DATA/'other-cache')),
                             ('purpose', 'engineering')):
            lease = self.lease(); lease[field] = value
            with patch.object(r, 'sha', return_value='0'*64), self.assertRaises(ValueError):
                r._validate_lease_shape(lease, dict(protocol=r.PROTOCOL))

    def ledger(self, partial_unclean=False):
        folder = r.stage.FOLDER/'fixtures/resource-pure'/uuid.uuid4().hex
        folder.mkdir(parents=True)
        identity = dict(schema=r.SCHEMA+'-identity', inherited_charged_ns=11_254_650_000_000,
                        protocol=r.PROTOCOL, fixture=False, old_ledger={'physical_worker_leases':20})
        events, previous = [], None
        for index in range(2):
            token = f'pure-{index}'
            lease = self.lease(fixture=True)
            lease['token'] = token
            lease_folder = folder/'leases'/token; lease_folder.mkdir(parents=True)
            (lease_folder/'lease.json').write_text(json.dumps(lease), encoding='utf-8')
            reserve = dict(kind='reserve', lease=lease)
            charge = (5 if partial_unclean else 120) if index == 0 else 40
            finish = dict(kind='finish', token=token, clean=index == 1,
                          charged_ns=charge*10**9, elapsed_ns=40*10**9, unknown_overrun=False)
            for body in (reserve, finish):
                event = dict(body, index=len(events), previous=previous)
                events.append(event); previous = config.digest(event)
        (folder/'events').mkdir()
        for event in events:
            (folder/'events'/f"{event['index']:06d}.json").write_text(json.dumps(event), encoding='utf-8')
        (folder/'identity.json').write_text(json.dumps(identity), encoding='utf-8')
        (folder/'head.json').write_text(json.dumps(dict(schema=r.SCHEMA+'-head', count=4,
                                                      last_sha256=previous)), encoding='utf-8')
        return folder, identity

    def test_settled_carry_and_full_unclean_charge(self):
        folder, identity = self.ledger()
        with patch.object(r, '_path', return_value=folder), patch.object(r, '_identity', return_value=identity), \
             patch.object(r, 'sha', return_value='0'*64):
            result = r.status(folder)
        self.assertEqual(result['charged_ns'], 11_414_650_000_000)
        self.assertEqual(result['remaining_ns'], 6_585_350_000_000)
        self.assertEqual(result['total_lease_count'], 22)
        self.assertIsNone(result['pending'])

    def test_unclean_partial_refund_is_rejected(self):
        folder, identity = self.ledger(partial_unclean=True)
        with patch.object(r, '_path', return_value=folder), patch.object(r, '_identity', return_value=identity), \
             patch.object(r, 'sha', return_value='0'*64), self.assertRaises(ValueError):
            r.status(folder)

    def scenario_ledger(self, rows):
        """Publish a new internally hashed scenario; never edit prior fixtures."""
        folder = r.stage.FOLDER/'fixtures/resource-pure'/uuid.uuid4().hex
        folder.mkdir(parents=True)
        identity = dict(schema=r.SCHEMA+'-identity', inherited_charged_ns=11_254_650_000_000,
                        protocol=r.PROTOCOL, fixture=False, old_ledger={'physical_worker_leases':20})
        events, previous = [], None
        for row in rows:
            lease = self.lease(fixture=row.get('fixture', True))
            lease['token'] = row['token']
            lease_folder = folder/'leases'/row['token']; lease_folder.mkdir(parents=True, exist_ok=True)
            lease_path = lease_folder/'lease.json'
            payload = json.dumps(lease)
            if lease_path.exists():
                self.assertEqual(lease_path.read_text(encoding='utf-8'), payload)
            else:
                lease_path.write_text(payload, encoding='utf-8')
            finish = dict(kind='finish', token=row['token'], clean=row['clean'],
                          charged_ns=row['charged_ns'], elapsed_ns=row.get('elapsed_ns', row['charged_ns']),
                          unknown_overrun=row.get('unknown_overrun', False))
            for body in (dict(kind='reserve', lease=lease), finish):
                event = dict(body, index=len(events), previous=previous)
                events.append(event); previous = config.digest(event)
        (folder/'events').mkdir()
        for event in events:
            (folder/'events'/f"{event['index']:06d}.json").write_text(json.dumps(event), encoding='utf-8')
        (folder/'identity.json').write_text(json.dumps(identity), encoding='utf-8')
        (folder/'head.json').write_text(json.dumps(dict(schema=r.SCHEMA+'-head', count=len(events),
                                                      last_sha256=previous)), encoding='utf-8')
        return folder, identity

    def test_unknown_overrun_cannot_be_clean_or_refunded(self):
        for clean, charged_ns in ((True, 40*10**9), (True, 120*10**9), (False, 40*10**9)):
            with self.subTest(clean=clean, charged_ns=charged_ns):
                folder, identity = self.scenario_ledger([dict(token='unknown-overrun', clean=clean,
                    charged_ns=charged_ns, elapsed_ns=40*10**9, unknown_overrun=True)])
                with patch.object(r, '_path', return_value=folder), patch.object(r, '_identity', return_value=identity), \
                     patch.object(r, 'sha', return_value='0'*64), self.assertRaises(ValueError):
                    r.status(folder)

    def test_reserve_cap_breach_cannot_be_hidden_by_later_refund(self):
        rows = [dict(token=f'settled-{i}', fixture=False, clean=True,
                     charged_ns=1_800*10**9) for i in range(3)]
        # The settled total is16654.65s. Reserving1800s breaches18000s;
        # its final zero charge would make the terminal arithmetic look safe.
        rows.append(dict(token='over-cap-then-refunded', fixture=False, clean=True,
                         charged_ns=0, elapsed_ns=0))
        folder, identity = self.scenario_ledger(rows)
        with patch.object(r, '_path', return_value=folder), patch.object(r, '_identity', return_value=identity), \
             patch.object(r, 'sha', return_value='0'*64), self.assertRaises(ValueError):
            r.status(folder)

    def test_token_cannot_be_reused_after_settlement(self):
        folder, identity = self.scenario_ledger([
            dict(token='same-durable-token', clean=True, charged_ns=40*10**9),
            dict(token='same-durable-token', clean=True, charged_ns=40*10**9)])
        with patch.object(r, '_path', return_value=folder), patch.object(r, '_identity', return_value=identity), \
             patch.object(r, 'sha', return_value='0'*64), self.assertRaises(ValueError):
            r.status(folder)


if __name__ == '__main__':
    unittest.main()
