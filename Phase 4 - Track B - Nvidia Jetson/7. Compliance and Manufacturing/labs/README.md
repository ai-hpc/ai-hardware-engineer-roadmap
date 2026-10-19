# CPU factory lab: measurements, retries and a real SQLite transaction

**Time:** 60–90 minutes. **Requirements:** Python 3 with its standard-library SQLite module. No Jetson, database server, instrument or paid service is required. All values describe a fictional teaching product.

You will distinguish a failed device from an unqualified station, preserve retest history, calculate two different yield measures and observe what happens when a process exits before committing a database transaction.

## 1. Run and predict the result

From the repository root:

```bash
python3 "Phase 4 - Track B - Nvidia Jetson/7. Compliance and Manufacturing/labs/factory_records.py"
python3 -m unittest discover -s "Phase 4 - Track B - Nvidia Jetson/7. Compliance and Manufacturing/labs" -p "test_*.py" -v
```

The program creates a temporary SQLite database and removes it afterward. [factory_records.py](factory_records.py) is the actual implementation; [test_factory_records.py](test_factory_records.py) drives it.

Unit `COURSE-001` first measures 4.8 V, below the fictional 4.9–5.1 V supply limits, and then measures 5.0 V on attempt 2. Unit `COURSE-002` passes on attempt 1. Expected summary:

```json
{
  "units_tested": 2,
  "attempts": 3,
  "first_pass_units": 1,
  "latest_pass_units": 2,
  "first_pass_yield": 0.5,
  "latest_pass_yield": 1.0
}
```

The two fractions divide by the same two units and use different outcomes. Neither is the fraction of attempts that passed (`2/3`). A later failing attempt can reduce the latest-result fraction; the code does not call a unit permanently good merely because it once passed.

## 2. Inspect the qualification contract

A record identifies product, board revision, limit revision, image digest, serial, attempt, station, fixture and calibration interval. Each required measurement includes an explicit unit. The known image digest comes from the bytes `fictional course image`; it is not a real Jetson image.

The three example measurements are supply voltage, loopback latency and a fixture reference. Their limits are teaching choices. Replace them only with a documented product-specific specification and decision rule in a real station.

The assessor rejects missing/unknown fields, wrong board/image/limits, duplicate JSON keys, wrong units, booleans masquerading as numbers, NaNs/infinities and a calibration interval that does not cover the recorded test time. The interval is start-inclusive and end-exclusive. The timestamps are deterministic fictional values; the lab does not consult a live calibration registry or verify a station's clock.

A failed fixture reference makes the station result **unqualified**. It raises an error before recording a DUT attempt, so a bad instrument does not incorrectly lower DUT first-pass yield. A production service should preserve that rejected station event in a separate audit channel; this small lab reports the rejection and leaves the qualified-DUT ledger unchanged.

**Exercise:** change supply units to `mV` without changing 5.0. Predict rejection. Then change the supply value to 5.2 while retaining valid units and fixture state. Predict a qualified recorded failure. Explain why those outcomes differ.

## 3. Follow the transaction

`Ledger.record` validates the record, serializes a stable JSON payload and begins a SQLite immediate transaction. It checks `(serial, attempt)` before inserting:

- An identical existing payload is a replay and adds no row.
- A different payload for the same attempt is a conflict and is rejected.
- A new unit must start at attempt 1.
- Later attempts must be contiguous.

The uniqueness constraint and transaction protect the check/insert operation. Parameter binding keeps data separate from SQL syntax. A SHA-256 digest records payload identity; it is not a signature or an access-control mechanism.

A separate four-process review submitted the same valid record concurrently. Exactly one process recorded it and three recognized a replay; the ledger contained one attempt. This checks that execution on the review host, not arbitrary distributed databases or every concurrency failure mode. The [validation record](validation.json) keeps the scope explicit.

## 4. Interrupt a real process before commit

The test starts a child process, opens the actual SQLite file, begins a transaction, inserts one row, then calls `os._exit(23)` before committing. The parent verifies:

1. The child exited with the expected code.
2. The uncommitted row is absent.
3. SQLite's integrity check reports `ok`.
4. A subsequent valid insertion succeeds.

The operating system and storage remain powered. This is process-crash evidence, not a power-loss experiment or a proof about SSD write caches. The teaching ledger requests SQLite `synchronous=FULL`; physical durability still depends on the complete storage stack and its documented behavior.

## 5. Try meaningful variations

Predict and then test these cases in a copy of the exercise:

| Change | Expected behavior |
| --- | --- |
| Upload an unchanged attempt twice | One row, second call reports replay |
| Correct a failed measurement under the same attempt ID | Conflict; use a new qualified attempt after actual retest |
| Start a unit at attempt 2 | Rejected because attempt 1 is missing |
| Test exactly at calibration expiry | Unqualified record rejected |
| Fail the fixture reference, then repair it | First result excluded; valid test may still use attempt 1 |
| Omit the loopback measurement | Incomplete evidence rejected |

Do not make tests pass by dropping the fixture reference or changing an acceptance limit after observing the result. Version any deliberate limit-policy change and retain the reasoning.

## 6. Submit your evidence

Submit the three-record history, separate unit/attempt counts, yield calculations, one qualified failure, one unqualified station case, one conflict/replay case and the process-crash test output. Explain the remaining boundaries: no actual electrical measurements, no hostile-station authentication, no production retention/backup system and no physical power-loss validation.

Return to the [manufacturing lesson](../Guide.md) to connect the ledger to fixture qualification, applicable conformity evidence and field traceability.
