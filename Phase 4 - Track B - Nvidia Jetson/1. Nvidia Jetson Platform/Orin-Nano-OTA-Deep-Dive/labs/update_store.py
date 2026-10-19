#!/usr/bin/env python3
"""A two-slot file-backed teaching updater; never accesses partitions or firmware.

Assumptions: one writer, trusted metadata, POSIX local filesystem, and no malicious
symlink changes. Process exits exercise ordering, not physical power-loss guarantees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


CUTS = ("half_image", "full_image", "published_image", "metadata_temp", "selected_trial")
HEALTH_KEYS = {"model_loaded", "sensor_fresh", "inference_valid", "watchdog_checked"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class UpdateStore:
    def __init__(self, directory: Path, crash_at: str | None = None):
        self.directory = Path(directory)
        self.crash_at = crash_at
        if crash_at is not None and crash_at not in CUTS:
            raise ValueError("unknown interruption point")

    def checkpoint(self, name: str) -> None:
        if self.crash_at == name:
            os._exit(95)  # Deliberately bypass Python cleanup in the child process.

    def sync_directory(self) -> None:
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def read(self) -> dict:
        return json.loads((self.directory / "state.json").read_text())

    def save(self, state: dict, *, installation: bool = False) -> None:
        temporary = self.directory / "state.tmp"
        with temporary.open("w") as stream:
            json.dump(state, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        if installation:
            self.checkpoint("metadata_temp")
        os.replace(temporary, self.directory / "state.json")
        self.sync_directory()
        if installation:
            self.checkpoint("selected_trial")

    def initialize(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise ValueError("initialize requires an empty directory")
        data = b"known-good simulator image version one\n"
        with (self.directory / "slot-A.bin").open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self.sync_directory()
        self.save({"schema": 1, "good": "A", "pending": None, "remaining": 0,
                   "boot_counter": 0, "running": None, "minimum_version": 1,
                   "slots": {"A": {"version": 1, "sha256": digest(data)}, "B": None}})

    def valid_slot(self, state: dict, slot: str) -> bool:
        record = state["slots"].get(slot)
        if record is None or record["version"] < state["minimum_version"]:
            return False
        try:
            data = (self.directory / f"slot-{slot}.bin").read_bytes()
        except OSError:
            return False
        return digest(data) == record["sha256"]

    def install(self, data: bytes, *, version: int, expected_sha256: str,
                trial_attempts: int = 2) -> None:
        state = self.read()
        if state["pending"] is not None:
            raise ValueError("finish or reject the existing trial first")
        if type(version) is not int or version <= state["slots"][state["good"]]["version"]:
            raise ValueError("this exercise accepts strictly newer integer versions")
        if type(trial_attempts) is not int or not 1 <= trial_attempts <= 10:
            raise ValueError("trial attempts must be an integer in [1, 10]")
        if not isinstance(data, bytes) or not 2 <= len(data) <= 1024*1024:
            raise ValueError("exercise image must contain 2 bytes to 1 MiB")
        if digest(data) != expected_sha256 or version < state["minimum_version"]:
            raise ValueError("artifact failed its trusted hash/version policy")
        if not self.valid_slot(state, state["good"]):
            raise ValueError("no valid known-good image to preserve")
        slot = "B" if state["good"] == "A" else "A"
        temporary = self.directory / "image.tmp"
        middle = len(data) // 2
        with temporary.open("wb") as stream:
            stream.write(data[:middle]); stream.flush(); os.fsync(stream.fileno())
            self.checkpoint("half_image")
            stream.write(data[middle:]); stream.flush(); os.fsync(stream.fileno())
            self.checkpoint("full_image")
        os.replace(temporary, self.directory / f"slot-{slot}.bin")
        self.sync_directory()
        self.checkpoint("published_image")
        # Re-read the published bytes before selecting them for a trial.
        if digest((self.directory / f"slot-{slot}.bin").read_bytes()) != expected_sha256:
            raise ValueError("read-back digest failed")
        state["slots"][slot] = {"version": version, "sha256": expected_sha256}
        state.update(pending=slot, remaining=trial_attempts, running=None)
        self.save(state, installation=True)

    def boot(self) -> dict:
        state = self.read()
        slot = state["pending"]
        trial = slot is not None and state["remaining"] > 0 and self.valid_slot(state, slot)
        if trial:
            state["remaining"] -= 1  # Persist BEFORE handing control to the trial.
        else:
            slot = state["good"]
            state.update(pending=None, remaining=0)
        if not self.valid_slot(state, slot):
            raise ValueError("no eligible image; explicit recovery required")
        state["boot_counter"] += 1
        record = state["slots"][slot]
        token = {"boot_id": state["boot_counter"], "slot": slot,
                 "version": record["version"], "sha256": record["sha256"], "trial": trial}
        state["running"] = token
        self.save(state)
        return token

    def confirm(self, token: dict, health: dict) -> None:
        state = self.read()
        if token != state["running"] or not token.get("trial") or state["pending"] != token["slot"]:
            raise ValueError("health result is not for the current trial boot")
        if set(health) != HEALTH_KEYS or any(value is not True for value in health.values()):
            raise ValueError("all defined application health checks must pass")
        if not self.valid_slot(state, token["slot"]):
            raise ValueError("running image no longer matches its recorded digest")
        state.update(good=token["slot"], pending=None, remaining=0)
        self.save(state)

    def advance_floor(self, version: int) -> None:
        state = self.read()
        if type(version) is not int or version < state["minimum_version"]:
            raise ValueError("version floor cannot decrease")
        # Deliberately conservative policy: preserve TWO acceptable copies.
        if state["pending"] is not None or any(not self.valid_slot(state, s) or
              state["slots"][s]["version"] < version for s in ("A", "B")):
            raise ValueError("floor would exclude the trial or a recovery copy")
        state["minimum_version"] = version
        self.save(state)


def demonstration() -> dict:
    with tempfile.TemporaryDirectory(prefix="course-ab-demo-") as directory:
        store = UpdateStore(Path(directory)); store.initialize()
        data = b"synthetic simulator image version two\n"
        store.install(data, version=2, expected_sha256=digest(data))
        failed_trials = [store.boot(), store.boot()]
        fallback = store.boot()
        store.install(data, version=2, expected_sha256=digest(data))
        successful_trial = store.boot()
        store.confirm(successful_trial, dict.fromkeys(HEALTH_KEYS, True))
        final_boot = store.boot()
        return {"evidence_kind": "temporary-file teaching protocol, not Jetson boot execution",
                "failed_trials": failed_trials, "fallback": fallback,
                "successful_trial": successful_trial, "final_boot": final_boot}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crash-worker", type=Path)
    parser.add_argument("--at", choices=CUTS)
    args = parser.parse_args()
    if args.crash_worker is not None:
        if args.at is None:
            parser.error("--crash-worker requires --at")
        payload = b"synthetic simulator image version two\n"
        UpdateStore(args.crash_worker, args.at).install(payload, version=2,
                                                       expected_sha256=digest(payload))
    else:
        if args.at is not None:
            parser.error("--at requires --crash-worker")
        print(json.dumps(demonstration(), indent=2))
