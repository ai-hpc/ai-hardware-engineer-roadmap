"""A local teaching station: explicit limits, transactional records and honest yield."""
from dataclasses import dataclass
import hashlib
import json
import math
import re
import sqlite3


@dataclass(frozen=True)
class Limit:
    unit: str
    lower: float
    upper: float


# Fictional product/fixture limits: not Jetson electrical specifications.
LIMITS = {'supply': Limit('V', 4.9, 5.1),
          'loopback_latency': Limit('ms', 0.0, 50.0),
          'fixture_reference': Limit('V', 2.49, 2.51)}
PRODUCT = 'course-observer'
BOARD = 'teaching-rev-a'
LIMIT_REVISION = 'example-1'
IMAGE_SHA256 = hashlib.sha256(b'fictional course image').hexdigest()


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f'{name}: finite number required')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f'{name}: finite number required')
    return value


def assess(record):
    """Return a failed or passed test; reject malformed/unqualified records."""
    expected = {'serial', 'attempt', 'product', 'board', 'limit_revision', 'image_sha256',
                'station', 'fixture', 'tested_at', 'calibrated_at', 'calibration_due', 'measurements'}
    if not isinstance(record, dict) or set(record) != expected:
        raise ValueError('record fields must match the schema')
    for name in ('serial', 'station', 'fixture'):
        value = record[name]
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value):
            raise ValueError(f'{name}: invalid identifier')
    if type(record['attempt']) is not int or not 1 <= record['attempt'] <= 1000:
        raise ValueError('attempt must be an integer in [1,1000]')
    for name, expected_value in [('product', PRODUCT), ('board', BOARD),
                                 ('limit_revision', LIMIT_REVISION), ('image_sha256', IMAGE_SHA256)]:
        if record[name] != expected_value:
            raise ValueError(f'{name}: wrong test configuration')
    tested, calibrated, due = (number(record[n], n) for n in ('tested_at', 'calibrated_at', 'calibration_due'))
    if not 0 <= calibrated <= tested < due:
        raise ValueError('fixture calibration does not cover test time')
    values = record['measurements']
    if not isinstance(values, dict) or set(values) != set(LIMITS):
        raise ValueError('all required measurements must appear exactly once')
    failures = []
    for name, limit in LIMITS.items():
        measurement = values[name]
        if not isinstance(measurement, dict) or set(measurement) != {'value', 'unit'}:
            raise ValueError(f'{name}: value/unit required')
        if measurement['unit'] != limit.unit:
            raise ValueError(f'{name}: wrong unit')
        value = number(measurement['value'], name)
        if not limit.lower <= value <= limit.upper:
            if name == 'fixture_reference':
                # A bad reference invalidates this station measurement, not the DUT.
                # A production station must retain a separate station-fault event.
                raise ValueError('fixture reference failed; unqualified station result')
            failures.append(name)
    return {'passed': not failures, 'failed_measurements': failures}


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'nonfinite JSON: {value}')))


class Ledger:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('''CREATE TABLE IF NOT EXISTS tests (
            serial TEXT NOT NULL, attempt INTEGER NOT NULL, passed INTEGER NOT NULL,
            payload TEXT NOT NULL, digest TEXT NOT NULL,
            PRIMARY KEY(serial, attempt))''')
        self.connection.commit()

    def close(self):
        self.connection.close()

    def record(self, record):
        result = assess(record)
        payload = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        key = (record['serial'], record['attempt'])
        # Serialize check+insert; repeated upload is idempotent, changed data conflicts.
        self.connection.execute('BEGIN IMMEDIATE')
        try:
            previous = self.connection.execute('SELECT payload FROM tests WHERE serial=? AND attempt=?', key).fetchone()
            if previous is not None:
                if previous[0] != payload:
                    raise ValueError('attempt identity already has different evidence')
                self.connection.commit()
                return 'replayed'
            largest = self.connection.execute('SELECT MAX(attempt) FROM tests WHERE serial=?', (key[0],)).fetchone()[0]
            expected = 1 if largest is None else largest + 1
            if key[1] != expected:
                raise ValueError('attempt sequence must begin at 1 and be contiguous')
            self.connection.execute('INSERT INTO tests VALUES (?,?,?,?,?)',
                                    (*key, int(result['passed']), payload, digest))
            self.connection.commit()
            return 'recorded'
        except Exception:
            self.connection.rollback()
            raise

    def summary(self):
        rows = self.connection.execute('SELECT serial, attempt, passed FROM tests ORDER BY serial, attempt').fetchall()
        units = {}
        for serial, attempt, passed in rows:
            units.setdefault(serial, []).append(bool(passed))
        count = len(units)
        return {'units_tested': count, 'attempts': len(rows),
                'first_pass_units': sum(values[0] for values in units.values()),
                'latest_pass_units': sum(values[-1] for values in units.values()),
                'first_pass_yield': sum(values[0] for values in units.values())/count if count else None,
                'latest_pass_yield': sum(values[-1] for values in units.values())/count if count else None}


def example(serial='COURSE-001', attempt=1, voltage=5.0):
    return {'serial': serial, 'attempt': attempt, 'product': PRODUCT, 'board': BOARD,
            'limit_revision': LIMIT_REVISION, 'image_sha256': IMAGE_SHA256,
            'station': 'station-a', 'fixture': 'fixture-a',
            'tested_at': 1_800_000_000, 'calibrated_at': 1_799_000_000, 'calibration_due': 1_801_000_000,
            'measurements': {'supply': {'value': voltage, 'unit': 'V'},
                             'loopback_latency': {'value': 12.0, 'unit': 'ms'},
                             'fixture_reference': {'value': 2.5, 'unit': 'V'}}}


if __name__ == '__main__':
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory(prefix='course-factory-') as directory:
        ledger = Ledger(Path(directory)/'station.sqlite')
        try:
            ledger.record(example('COURSE-001', voltage=4.8))
            ledger.record(example('COURSE-001', attempt=2))
            ledger.record(example('COURSE-002'))
            print(json.dumps(ledger.summary(), indent=2))
        finally:
            ledger.close()
