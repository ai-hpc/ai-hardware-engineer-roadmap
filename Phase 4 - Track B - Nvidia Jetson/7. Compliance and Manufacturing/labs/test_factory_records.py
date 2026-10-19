import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from factory_records import Ledger, assess, example, strict_json


class FactoryRecords(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='course-factory-test-')
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'ledger.sqlite'
        self.ledger = Ledger(self.path)
        self.addCleanup(self.ledger.close)

    def test_reference_pass_and_inclusive_limits(self):
        for value in (4.9, 5.0, 5.1):
            self.assertTrue(assess(example(voltage=value))['passed'])
        self.assertEqual(assess(example(voltage=4.89))['failed_measurements'], ['supply'])

    def test_fixture_reference_failure_is_unqualified_and_preserves_dut_yield(self):
        record = example()
        record['measurements']['fixture_reference']['value'] = 2.6
        with self.assertRaisesRegex(ValueError, 'unqualified station'):
            assess(record)
        with self.assertRaises(ValueError):
            self.ledger.record(record)
        self.assertEqual(self.ledger.summary()['attempts'], 0)
        self.assertIsNone(self.ledger.summary()['first_pass_yield'])
        self.assertEqual(self.ledger.record(example()), 'recorded')
        self.assertEqual(self.ledger.summary()['first_pass_yield'], 1.0)

    def test_units_missing_and_nonfinite_rejected(self):
        for value in (True, float('nan'), float('inf'), 10**1000, '5.0'):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                assess(example(voltage=value))
        record = example(); record['measurements']['supply']['unit'] = 'mV'
        with self.assertRaises(ValueError): assess(record)
        record = example(); del record['measurements']['fixture_reference']
        with self.assertRaises(ValueError): assess(record)

    def test_wrong_board_image_limits_and_schema_rejected(self):
        for key in ('product', 'board', 'limit_revision', 'image_sha256'):
            record = example(); record[key] = 'wrong'
            with self.subTest(key=key), self.assertRaises(ValueError): assess(record)
        record = example(); record['passed'] = True
        with self.assertRaises(ValueError): assess(record)

    def test_calibration_interval_and_identifier_validation(self):
        for key, value in [('tested_at', 1_801_000_000), ('tested_at', 1_798_999_999),
                           ('calibration_due', float('nan')), ('serial', ''), ('serial', 'x'*65),
                           ('attempt', True), ('attempt', 0), ('attempt', 1001)]:
            record = example(); record[key] = value
            with self.subTest(key=key, value=str(value)), self.assertRaises(ValueError): assess(record)

    def test_duplicate_json_keys_and_nonfinite_json_rejected(self):
        for text in ('{"serial":"a","serial":"b"}', '{"x":{"a":1,"a":2}}', '{"x":NaN}'):
            with self.assertRaises(ValueError): strict_json(text)
        self.assertEqual(strict_json(json.dumps(example())), example())

    def test_replay_does_not_inflate_yield_or_attempts(self):
        record = example()
        self.assertEqual(self.ledger.record(record), 'recorded')
        self.assertEqual(self.ledger.record(copy.deepcopy(record)), 'replayed')
        self.assertEqual(self.ledger.summary()['attempts'], 1)

    def test_conflicting_attempt_rejected_without_overwriting(self):
        self.ledger.record(example(voltage=4.8))
        with self.assertRaises(ValueError): self.ledger.record(example())
        self.assertEqual(self.ledger.summary()['first_pass_units'], 0)
        self.ledger.record(example(attempt=2))
        self.assertEqual(self.ledger.summary()['latest_pass_units'], 1)

    def test_attempt_order_and_empty_yield(self):
        self.assertIsNone(self.ledger.summary()['first_pass_yield'])
        with self.assertRaises(ValueError): self.ledger.record(example(attempt=2))
        self.ledger.record(example())
        with self.assertRaises(ValueError): self.ledger.record(example(attempt=3))
        self.assertEqual(self.ledger.summary()['attempts'], 1)

    def test_first_pass_and_latest_outcomes_use_same_unit_denominator(self):
        self.ledger.record(example('A', voltage=4.8))
        self.ledger.record(example('A', attempt=2))
        self.ledger.record(example('B'))
        self.assertEqual(self.ledger.summary(), {'units_tested': 2, 'attempts': 3,
                         'first_pass_units': 1, 'latest_pass_units': 2,
                         'first_pass_yield': .5, 'latest_pass_yield': 1.0})
        self.ledger.record(example('B', attempt=2, voltage=4.8))
        self.assertEqual(self.ledger.summary()['latest_pass_yield'], .5)

    def test_real_process_exit_before_commit_does_not_leave_half_record(self):
        self.ledger.record(example('A'))
        script = '''import sqlite3,sys,os
c=sqlite3.connect(sys.argv[1])
c.execute("BEGIN IMMEDIATE")
c.execute("INSERT INTO tests VALUES ('CRASH',1,1,'uncommitted','uncommitted')")
os._exit(23)
'''
        result = subprocess.run([sys.executable, '-c', script, str(self.path)], timeout=10)
        self.assertEqual(result.returncode, 23)
        self.assertEqual(self.ledger.summary()['units_tested'], 1)
        connection = sqlite3.connect(self.path)
        try: self.assertEqual(connection.execute('PRAGMA integrity_check').fetchone(), ('ok',))
        finally: connection.close()
        self.ledger.record(example('B'))
        self.assertEqual(self.ledger.summary()['units_tested'], 2)


if __name__ == '__main__': unittest.main()
