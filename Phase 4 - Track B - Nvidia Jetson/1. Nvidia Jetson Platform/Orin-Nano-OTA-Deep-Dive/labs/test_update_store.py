import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from update_store import CUTS, HEALTH_KEYS, UpdateStore, digest


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="course-ab-test-")
        self.addCleanup(self.temporary.cleanup)
        self.store = UpdateStore(Path(self.temporary.name))
        self.store.initialize()
        self.payload = b"synthetic simulator image version two\n"
        self.healthy = dict.fromkeys(HEALTH_KEYS, True)

    def install(self, version=2):
        self.store.install(self.payload, version=version, expected_sha256=digest(self.payload))

    def test_success_survives_reopening(self):
        self.install(); token = self.store.boot(); self.store.confirm(token, self.healthy)
        reboot = UpdateStore(self.store.directory).boot()
        self.assertEqual((reboot["slot"], reboot["version"], reboot["trial"]), ("B", 2, False))

    def test_unconfirmed_boots_exhaust_and_fall_back(self):
        self.install()
        boots = [self.store.boot() for _ in range(3)]
        self.assertEqual([b["version"] for b in boots], [2, 2, 1])
        self.assertEqual([b["trial"] for b in boots], [True, True, False])
        self.assertEqual(self.store.read()["pending"], None)

    def test_process_interruption_at_every_install_boundary(self):
        script = str(Path(__file__).with_name("update_store.py"))
        for cut in CUTS:
            with self.subTest(cut=cut), tempfile.TemporaryDirectory(prefix="course-ab-cut-") as directory:
                store = UpdateStore(Path(directory)); store.initialize()
                before = (store.directory / "slot-A.bin").read_bytes()
                process = subprocess.run([sys.executable, script, "--crash-worker", directory,
                                          "--at", cut], capture_output=True, timeout=10)
                self.assertEqual(process.returncode, 95, process.stderr.decode())
                state = store.read()  # Must remain parseable, with the confirmed copy intact.
                self.assertEqual((store.directory / "slot-A.bin").read_bytes(), before)
                self.assertTrue(store.valid_slot(state, state["good"]))
                boots = [store.boot() for _ in range(3)]
                self.assertEqual(boots[-1]["version"], 1)
                self.assertEqual(boots[0]["version"], 2 if cut == "selected_trial" else 1)

    def test_corrupt_trial_is_not_executed(self):
        self.install()
        (self.store.directory / "slot-B.bin").write_bytes(b"damaged")
        boot = self.store.boot()
        self.assertEqual(boot["version"], 1)
        self.assertFalse(boot["trial"])

    def test_no_valid_copy_requires_recovery(self):
        (self.store.directory / "slot-A.bin").write_bytes(b"damaged")
        with self.assertRaises(ValueError):
            self.store.boot()

    def test_stale_boot_health_cannot_accept_new_trial(self):
        self.install(); old = self.store.boot(); current = self.store.boot()
        with self.assertRaises(ValueError):
            self.store.confirm(old, self.healthy)
        self.store.confirm(current, self.healthy)

    def test_health_requires_every_defined_check(self):
        self.install(); token = self.store.boot()
        for health in ({}, {**self.healthy, "sensor_fresh": False},
                       {**self.healthy, "model_loaded": 1}):
            with self.subTest(health=health), self.assertRaises(ValueError):
                self.store.confirm(token, health)
        self.assertEqual(self.store.read()["good"], "A")

    def test_confirm_rechecks_artifact(self):
        self.install(); token = self.store.boot()
        (self.store.directory / "slot-B.bin").write_bytes(b"changed after boot")
        with self.assertRaises(ValueError):
            self.store.confirm(token, self.healthy)

    def test_hash_and_old_version_rejected_without_change(self):
        before = self.store.read()
        with self.assertRaises(ValueError):
            self.store.install(self.payload, version=2, expected_sha256="0"*64)
        with self.assertRaises(ValueError):
            self.install(version=1)
        self.assertEqual(before, self.store.read())

    def test_parallel_update_and_bad_attempt_count_rejected(self):
        for attempts in (0, 11, True):
            with self.subTest(attempts=attempts), self.assertRaises(ValueError):
                self.store.install(self.payload, version=2, expected_sha256=digest(self.payload),
                                   trial_attempts=attempts)
        self.install()
        with self.assertRaises(ValueError):
            self.install(version=3)

    def test_floor_waits_for_compatible_recovery_copy(self):
        self.install(); self.store.confirm(self.store.boot(), self.healthy)
        with self.assertRaises(ValueError):
            self.store.advance_floor(2)  # A still contains version 1.
        self.install(version=3); self.store.confirm(self.store.boot(), self.healthy)
        self.store.advance_floor(2)  # A=3, B=2; both remain eligible.
        self.assertEqual(self.store.read()["minimum_version"], 2)
        with self.assertRaises(ValueError):
            self.store.advance_floor(1)

    def test_reinitialization_rejected(self):
        with self.assertRaises(ValueError):
            self.store.initialize()


if __name__ == "__main__":
    unittest.main()
