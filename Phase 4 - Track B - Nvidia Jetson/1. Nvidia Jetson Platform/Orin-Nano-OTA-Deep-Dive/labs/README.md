# Interrupt an update and recover

**Time:** 45–60 minutes. **Requirements:** Python 3.10+ on Linux; no Jetson, administrator access, network, or large download. Learn the [signed-artifact exercise](../../Orin-Nano-Security/labs/README.md) first.

This lab implements a small, inspectable update protocol using ordinary files in temporary directories. It performs real writes, digest checks, directory synchronization, process exits, reopen operations, and trial-boot decisions. It does **not** emulate NVIDIA firmware, simulate a filesystem losing power, or establish storage-device durability.

## 1. Run the example and tests

Download [update_store.py](update_store.py) and [test_update_store.py](test_update_store.py) into one folder, or enter this `labs` directory:

```bash
python3 update_store.py
python3 -m unittest discover -s . -p 'test_*.py' -v
```

The example creates and removes its own temporary directory. The tests also use temporary directories; they never write a disk partition. The twelve tests include five child-process interruption cases. See the [saved example](example-result.json) and [execution record](validation.json).

## 2. Understand the state

The store has two image files, `slot-A.bin` and `slot-B.bin`, plus `state.json`:

| Field | Meaning |
| --- | --- |
| `good` | Last image accepted by the application checks |
| `pending` | Image selected for a limited trial, or null |
| `remaining` | Number of trial starts still permitted |
| `slots` | Version and expected digest of each published image |
| `running` | Identity of the most recently selected simulated boot |
| `boot_counter` | Monotonic identifier within this local experiment |
| `minimum_version` | Teaching policy for eligible images |

Initially A contains version 1 and is confirmed; B is empty. The code installs version 2 into B while retaining A. The trusted expected hash is supplied by the caller. It must come from authenticated policy in a real updater; a hash downloaded alongside untrusted bytes provides no publisher authentication.

The experiment assumes one writer and trusted metadata in a local POSIX filesystem. Its JSON counter is editable and is **not** a hardware anti-rollback mechanism. Its files and paths are not designed to withstand a malicious local process.

## 3. Follow the ordering

The installation sequence is:

1. Validate version, image length, expected digest, trial count, and the confirmed fallback.
2. Write the first half of a temporary image and synchronize it.
3. Write and synchronize the remaining bytes.
4. Publish the inactive image with a same-directory replacement, then synchronize the directory.
5. Read back and hash the published image.
6. Write and synchronize new temporary metadata selecting a trial.
7. Replace the metadata file, then synchronize the directory.

The old confirmed image is never overwritten during this installation. Selecting a trial happens only after the complete candidate has been published and checked. Writing metadata to a separate temporary file avoids exposing a half-written JSON document to a subsequent reader under the lab's filesystem assumptions.

A rename's atomic visibility is different from durability after actual power loss. The explicit synchronization calls express the intended ordering; their end-to-end guarantees depend on the operating system, filesystem, storage device, and failure. The tests interrupt a process while the operating system remains running.

## 4. Inspect the interrupted cases

The test launches a child process at each named cut point. The child calls `os._exit(95)`, bypassing normal Python cleanup. The parent opens the store again and checks the confirmed image and metadata.

| Interruption point | First simulated boot after reopen |
| --- | --- |
| `half_image` | Confirmed version 1 |
| `full_image` | Confirmed version 1 |
| `published_image` | Confirmed version 1; the candidate has not been selected |
| `metadata_temp` | Confirmed version 1; published metadata is still old |
| `selected_trial` | Trial version 2; confirmed version 1 remains available |

For every row, three unconfirmed simulated boots end at version 1. The five positions test the boundaries supplied in the implementation; they are not exhaustive fault injection into every write, filesystem operation, or device behavior.

**Try:** move the metadata publication before the image write in a copy of the program. Add a cut immediately afterward. Explain which invariant you violated and which check prevents an incomplete candidate from being treated as valid. Preserve the original test and add a regression for the changed ordering.

## 5. Fail health checks, then accept a healthy trial

The teaching policy allows two trial starts. Before each trial starts, `boot()` persists the decremented count. Without confirmation, its sequence is version **2, 2, 1**. An application crash cannot reset that stored retry count in this model.

Acceptance requires all four named checks to be exactly true:

- `model_loaded`: the intended model and runtime initialized.
- `sensor_fresh`: the required input stream supplied a sufficiently recent observation.
- `inference_valid`: a known input produced an acceptable output under the model contract.
- `watchdog_checked`: the application-specific watchdog check passed.

In the demo these are supplied booleans. Implementing real checks is a separate device task. The lesson is their placement and association with one trial, not a claim that these four booleans prove robot safety.

The confirmation token includes boot ID, slot, version, and digest. A healthy result from an earlier trial cannot confirm a later boot. Missing checks, false checks, integer `1` instead of boolean `True`, and an image changed after boot all fail. The successful part of the demo confirms B/version 2 and then reopens it as a nontrial boot.

## 6. See how anti-rollback can remove recovery

After accepting B/version 2, A still contains version 1. Advancing the minimum to version 2 would make A ineligible. This lab adopts a conservative policy that refuses a new floor until **both** copies meet it and no trial is pending.

Install and accept version 3 into A, leaving version 2 in B. Now a floor of 2 permits both. The test checks this sequence and rejects lowering the floor afterward. A real design may use a separate recovery image or another policy, but it must explain what remains bootable after the floor advances.

## 7. Finish with an evidence table

Submit the five cut-point results, the `2, 2, 1` failed-health sequence, one rejected stale confirmation, and the accepted update. Explain these unmodeled failures: full filesystem, corrupted metadata, concurrent updaters, shared-data migration, hostile file changes, physical power loss, and boot firmware incompatibility.

**Next:** map each abstract state to the exact mechanism in the [OTA guide](../Guide.md) and [rootfs/A-B guide](../../Orin-Nano-Rootfs-and-AB-Redundancy/Guide.md). Do not translate the teaching field names into invented `nvbootctrl` commands.
