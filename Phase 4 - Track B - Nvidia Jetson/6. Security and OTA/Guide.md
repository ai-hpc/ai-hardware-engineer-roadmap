# Security and OTA

<div class="course-identity security-ota" markdown="1">
<div class="course-identity__icon">SEC</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track B6 · Security & OTA</p>
<p class="course-identity__title">Harden Jetson products with secure boot, signed artifacts, update strategy, and rollback behavior.</p>
<p class="course-identity__meta">Artifact: secure update plan · Measure: rollback time, integrity checks, failure recovery</p>
</div>
</div>


**Phase 4 — Track B — Nvidia Jetson** · Module 6 of 7

> **Focus:** Harden a **Jetson Orin Nano 8GB** product for field deployment: **secure boot** (PKC/SBK fuse programming), **OP-TEE** trusted execution, **disk encryption**, **A/B rootfs redundancy**, and a complete **OTA update pipeline** with rollback, monitoring, and reliability testing.
>
> **Primary hardware:** Jetson Orin Nano 8GB on custom or dev-kit carrier

**Previous:** [5. Application Development](../5.%20Application%20Development/Guide.md) · **Next:** [7. Compliance and Manufacturing](../7.%20Compliance%20and%20Manufacturing/Guide.md)

---


## 1. Why security and OTA are a single module

Security and OTA are tightly coupled in production Jetson products:

- **Secure boot** ensures only authorized firmware and kernel run on the device
- **OTA channels** must be signed so an attacker cannot push malicious updates
- **Disk encryption** protects data at rest if the device is physically stolen
- **A/B redundancy** makes OTA safe — a failed update rolls back to the last known-good slot
- **Fuse programming** is irreversible and must be planned before production flashing

This module brings together material that was previously split across deep-dive sub-guides in Module 1 and organizes it as a **production decision flow**: what to do, in what order, and why.

> **Deep-dive companions (Module 1):** The sub-guides below contain implementation-level detail that this module references but does not duplicate:
> - [Orin Nano Security](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Security/Guide.md) — threat model, attack surfaces, detailed PKC/SBK walkthrough
> - [Orin Nano OTA Deep Dive](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-OTA-Deep-Dive/Guide.md) — NVIDIA `nv_update_engine`, slot management internals
> - [Orin Nano Rootfs and A/B Redundancy](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Rootfs-and-AB-Redundancy/Guide.md) — partition layout, `nvbootctrl`, slot switching

---

## 2. Threat model for deployed Jetson products

Before hardening, define what you're protecting against:

| Threat | Example | Mitigation |
|--------|---------|------------|
| **Firmware tampering** | Attacker replaces bootloader via physical access | Secure boot (PKC) — hardware root of trust |
| **Data theft** | Device stolen, NVMe removed and read | Disk encryption (LUKS + OP-TEE key) |
| **Unauthorized updates** | Malicious OTA pushed to device | Signed update images + TLS channel |
| **Network intrusion** | Attacker exploits open port on device | Firewall, minimal services, SSH key-only |
| **Privilege escalation** | Application container escapes to host | AppArmor/SELinux, minimal rootfs, no root SSH |
| **Supply chain** | Counterfeit module or tampered firmware image | Fuse-based device identity, signed factory images |

---

## 3. Secure boot — PKC and SBK

### Secure boot chain

Jetson Orin Nano (T234) supports a **hardware root of trust** using fuses:

```
BootROM (silicon)
  │  Verifies MB1 signature using PKC hash in fuses
  │
MB1 (QSPI)
  │  Verifies MB2 using PKC chain
  │
MB2 (QSPI)
  │  Verifies UEFI, OP-TEE, kernel
  │
UEFI → Linux kernel → userspace
```

### PKC (Public Key Cryptography)

- Generate an RSA-2048 or RSA-3072 key pair
- Burn the **public key hash** into OTP fuses
- All firmware images are signed with the private key
- BootROM verifies the signature chain at every stage

```bash
# Generate PKC key pair
openssl genrsa -out pkc.pem 3072

# Extract public key
openssl rsa -in pkc.pem -pubout -out pkc_pub.pem
```

### SBK (Symmetric Binding Key)

- 128-bit AES key burned into OTP fuses
- Encrypts firmware images in addition to signing them
- Prevents reading firmware even with physical access to QSPI
- Combined with PKC: images are encrypted AND signed

---

## 4. Fuse programming workflow

**Fuse programming is irreversible.** Once fuses are burned, the device permanently requires signed/encrypted firmware.

### Lab workflow (test keys)

1. Generate **test** PKC and SBK keys (clearly labeled, stored separately from production keys)
2. Flash and validate the device works with signed images
3. Burn fuses on a **test unit only**
4. Verify the fused device boots with signed images and rejects unsigned images
5. **Do not** fuse production units until the entire boot/OTA chain is validated

### Production workflow

1. Generate **production** PKC and SBK keys
2. Store private key in an HSM (Hardware Security Module) or at minimum an encrypted, access-controlled vault
3. Integrate key signing into the CI/CD build pipeline
4. Program fuses as part of factory flashing (see [Module 8 — Compliance and Manufacturing](../7.%20Compliance%20and%20Manufacturing/Guide.md))

### Fuse burning command

```bash
sudo ./odmfuse.sh -i <board_id> -k pkc.pem -S sbk.key <board_config>
```

### Verifying fuse state

```bash
# On target
cat /sys/devices/platform/tegra-fuse/odm_production_mode
# 1 = fused (production mode)
```

---

## 5. OP-TEE and trusted execution

OP-TEE provides a **Trusted Execution Environment (TEE)** on the Cortex-A78AE cores:

- **Secure world** runs OP-TEE OS alongside **normal world** Linux
- **Trusted Applications (TAs)** execute in the secure world
- Use cases: key storage, attestation, DRM, secure sensor processing

### Jetson OP-TEE integration

NVIDIA ships OP-TEE as part of the L4T BSP. It is loaded during the boot chain (MB2 loads OP-TEE before UEFI).

| Component | Location |
|-----------|----------|
| OP-TEE OS image | `tos-optee_t234.img` in flash partition |
| Trusted Applications | Built into the OP-TEE image or loaded at runtime |
| TEE client library | `libteec.so` (in rootfs) |
| TEE supplicant | `tee-supplicant` daemon |

### Key storage with OP-TEE

Store disk encryption keys in OP-TEE secure storage rather than in plaintext on the filesystem:

1. OP-TEE generates or receives the LUKS key
2. Key is stored in OP-TEE secure storage (RPMB or encrypted file)
3. At boot, OP-TEE retrieves the key and passes it to the kernel for LUKS unlock
4. The key never appears in normal-world memory

---

## 6. Disk encryption (LUKS + OP-TEE key storage)

### Why encrypt

Devices deployed in the field can be physically accessed. Without disk encryption, an attacker can:

- Remove the NVMe SSD and read all data on a different machine
- Extract ML models, credentials, customer data
- Clone the device by copying the filesystem

### LUKS setup on Jetson

```bash
# Create encrypted partition
sudo cryptsetup luksFormat /dev/nvme0n1p1

# Open (unlock) the partition
sudo cryptsetup open /dev/nvme0n1p1 crypt_root

# Format and mount
sudo mkfs.ext4 /dev/mapper/crypt_root
sudo mount /dev/mapper/crypt_root /mnt
```

### Auto-unlock with OP-TEE

For headless devices, manual passphrase entry is not practical. Use OP-TEE to auto-unlock at boot:

1. Store the LUKS key in OP-TEE secure storage during factory provisioning
2. At boot, an initrd script calls the OP-TEE client to retrieve the key
3. The key unlocks the LUKS partition
4. The key is not accessible from normal-world after boot

### Performance impact

Disk encryption adds CPU overhead for I/O:

| Workload | Without encryption | With LUKS (AES-256-XTS) | Overhead |
|----------|-------------------|------------------------|----------|
| Sequential read | ~3.2 GB/s | ~2.4 GB/s | ~25% |
| Sequential write | ~2.8 GB/s | ~2.1 GB/s | ~25% |
| Random 4K read | ~180K IOPS | ~140K IOPS | ~22% |

The Cortex-A78AE has ARMv8 crypto extensions, so AES is hardware-accelerated. The overhead is acceptable for most edge AI workloads where the bottleneck is GPU inference, not storage I/O.

---

## 7. Runtime hardening

### Service minimization

```bash
# List all enabled services
systemctl list-unit-files --state=enabled

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable avahi-daemon
sudo systemctl disable cups
sudo systemctl mask <service>  # prevent re-enabling
```

### SSH hardening

```
# /etc/ssh/sshd_config
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AllowUsers deploy
MaxAuthTries 3
```

### Firewall

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 443/tcp   # OTA update channel
sudo ufw enable
```

### AppArmor

JetPack ships with AppArmor support. Create profiles for your application containers and services to restrict file access, network access, and capabilities.

### Debug UART in production

**Disable or restrict** the debug UART on production devices. An exposed UART gives root shell access. Options:

- Remove the debug UART header from the production carrier PCB (recommended)
- Disable UART console in the kernel command line (`console=` set to `ttyS0` with no getty)
- Protect with password (less secure, still exposes boot log)

---

## 8. A/B rootfs redundancy architecture

A/B rootfs allows safe updates: write the new image to the inactive slot, validate, then switch.

### Partition layout

```
QSPI NOR (boot):
  ├─ mb1_a / mb1_b     (A/B bootloader stage 1)
  ├─ mb2_a / mb2_b     (A/B bootloader stage 2)
  ├─ uefi_a / uefi_b   (A/B UEFI)
  ├─ tos_a / tos_b     (A/B OP-TEE)
  └─ bpmp_a / bpmp_b   (A/B BPMP firmware)

NVMe:
  ├─ APP_a              (rootfs slot A)
  ├─ APP_b              (rootfs slot B)
  └─ DATA               (persistent user data, not duplicated)
```

### Slot management

```bash
# Check active slot
sudo nvbootctrl get-current-slot

# Set next boot slot
sudo nvbootctrl set-active-boot-slot <0|1>

# Mark current slot as successful (after validation)
sudo nvbootctrl set-slot-as-successful <0|1>
```

### Sizing trade-offs

A/B doubles the rootfs storage requirement. For a 4 GB rootfs:

- Single slot: 4 GB
- A/B: 8 GB + ~1 GB overhead = ~9 GB
- With persistent data partition: 9 GB + DATA size

Plan NVMe capacity accordingly. A 128 GB NVMe gives ample room; a 32 GB eMMC may be tight.

> **Deep dive:** [Orin Nano Rootfs and A/B Redundancy](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Rootfs-and-AB-Redundancy/Guide.md)

---

## 9. OTA pipeline design

### End-to-end architecture

```
Build server (CI/CD)
  │  Build rootfs image, sign with PKC/OTA key
  │
Staging server
  │  Host signed images, manage release channels
  │  (stable, beta, canary)
  │
  ├─ TLS ─────────────────────────────────────────┐
  │                                                │
Device agent                                       │
  │  Poll for updates or receive push notification │
  │  Download image, verify signature              │
  │  Write to inactive slot                        │
  │  Reboot into new slot                          │
  │  Run validation tests                          │
  │  Mark slot as successful (or rollback)         │
  │  Report status back to server                  │
  └────────────────────────────────────────────────┘
```

### Release channel strategy

| Channel | Purpose | Rollout |
|---------|---------|---------|
| **canary** | Internal testing, nightly builds | 1–5 devices |
| **beta** | Early adopter or staging fleet | 5–10% of fleet |
| **stable** | Production | 100% of fleet (phased rollout) |

---

## 10. OTA frameworks on Jetson (SWUpdate, Mender, RAUC)

| Framework | Strengths | Jetson integration |
|-----------|-----------|-------------------|
| **SWUpdate** | Flexible, scriptable, double-copy and single-copy modes, signed images, Lua handlers | Good — integrates with `nvbootctrl` for slot management |
| **Mender** | Managed cloud service, device dashboard, fleet management | Good — Mender Hub has Jetson examples |
| **RAUC** | D-Bus API, slot status tracking, bundle signing | Good — RAUC bundles can be configured for Jetson A/B |
| **NVIDIA nv_update_engine** | Native Jetson tool, partition-level updates | Built-in but limited fleet management |

### Recommended approach

Use **SWUpdate** or **Mender** as the device-side agent, with **NVIDIA's partition tools** (`nvbootctrl`, `nv_update_engine`) for bootloader and slot operations. This gives you both ecosystem-standard OTA features and Jetson-specific boot integration.

---

## 11. Bootloader and firmware OTA

Updating QSPI-resident firmware (MB1, MB2, UEFI, OP-TEE, BPMP, SPE) in the field:

- Use A/B slot support in QSPI — write new firmware to the inactive slot
- NVIDIA's `nv_update_engine` handles partition-level writes
- **Risk:** A failed bootloader update can brick the device if not properly A/B managed
- **Mitigation:** Always update bootloader before rootfs; validate bootloader boots before switching rootfs slot

---

## 12. Container-layer OTA

For applications running in Docker containers (as set up in Module 2 — L4T Customization):

- Update container images independently of the rootfs
- Use a private container registry (Harbor, AWS ECR, or local)
- Pull new images, stop old container, start new container
- **Rollback:** Keep the previous image tag; `docker rollback` or restart with old tag

### Layer caching

Docker layer caching reduces download size — only changed layers are pulled. Structure your Dockerfile so that:

1. Base L4T image (large, changes rarely) is the bottom layer
2. System dependencies (changes occasionally) in middle layers
3. Application code (changes frequently) in the top layer

---

## 13. Delta updates and bandwidth engineering

For devices on cellular or bandwidth-constrained networks:

| Technique | Savings | Complexity |
|-----------|---------|------------|
| **Full image** | 0% (baseline) | Low |
| **bsdiff / zstd delta** | 60–90% | Medium — requires generating deltas per version pair |
| **casync** (content-addressable chunks) | 70–95% | Medium — chunk store on server, only new chunks downloaded |
| **Container layer diff** | 80–95% (for app updates) | Low — built into Docker |

### Practical recommendation

- Use **full rootfs images** for the initial fleet (<100 devices) — simpler, less infrastructure
- Add **delta updates** when bandwidth costs become significant
- Use **container-layer updates** for application changes (most common update type)

---

## 14. Rollback and power-fail safety

### Power-fail during OTA

If power fails during an update:

| Phase | Power-fail result | Recovery |
|-------|-------------------|----------|
| Downloading image | Incomplete download | Resume or re-download on next boot |
| Writing to inactive slot | Partially written slot | Inactive slot is corrupt but **active slot is untouched** — device boots normally |
| Rebooting into new slot | Boot interrupted | Watchdog or retry counter triggers rollback to previous slot |
| Validation running | Validation incomplete | Slot not marked as successful — next reboot falls back |

### Watchdog timer

Configure a hardware watchdog to trigger reboot if the system hangs during post-update validation:

```bash
# Enable watchdog
echo 1 > /dev/watchdog

# Application must pet the watchdog periodically
# If it fails to pet within timeout → hardware reset → rollback
```

### Retry counter

Use a boot counter in `nvbootctrl` or a custom persistent flag:

1. Before switching to new slot, set retry counter = 3
2. Each boot attempt decrements the counter
3. If the counter reaches 0 without the slot being marked successful → revert to old slot

---

## 15. OTA signing and verification chain

```
Build server:
  ├─ Build rootfs image
  ├─ Compute SHA-256 hash of image
  ├─ Sign hash with OTA signing key (RSA-3072 or Ed25519)
  └─ Bundle: image + signature + metadata (version, channel, timestamp)

Device:
  ├─ Download bundle
  ├─ Verify signature using embedded OTA public key
  ├─ Verify hash matches image
  ├─ Check version > current (prevent downgrade)
  └─ Apply update
```

### Key management

- **OTA signing key** is separate from the **secure boot PKC key**
- Store OTA private key in CI/CD secrets or HSM
- Embed OTA public key in the rootfs (or in a read-only partition)
- Rotate OTA keys by shipping the new public key in a signed update before switching

---

## 16. Fleet monitoring and telemetry

| Metric | Source | Purpose |
|--------|--------|---------|
| **Boot count** | Persistent counter | Detect reboot loops |
| **Update status** | OTA agent | Track rollout success/failure rate |
| **Slot status** | `nvbootctrl` | Detect devices stuck on old slot |
| **Temperature** | `tegrastats` | Detect thermal issues in the field |
| **Disk usage** | `df` | Prevent devices from running out of space |
| **Uptime** | `/proc/uptime` | Detect unexpected reboots |
| **Application health** | Health check endpoint | Detect application-level failures |

### Telemetry stack

- **Device side:** Lightweight agent (custom, or Telegraf/Fluent Bit) reports metrics via MQTT or HTTPS
- **Server side:** Time-series database (InfluxDB, Prometheus) + dashboard (Grafana)
- **Alerts:** Set thresholds for reboot loops, update failures, temperature anomalies

---

## 17. Reliability testing (HALT/HASS, soak, power-cycle)

Before shipping, validate that the device survives real-world conditions:

### Power-cycle endurance

```bash
#!/bin/bash
# Automated power-cycle test (requires controllable power supply or relay)
for i in $(seq 1 1000); do
    power_off
    sleep 5
    power_on
    wait_for_boot   # timeout → fail
    run_smoke_test  # basic peripheral check
    log_result $i
done
```

**Target:** 1000 power cycles with zero boot failures.

### Thermal cycling

| Parameter | Consumer | Industrial |
|-----------|----------|------------|
| Temperature range | 0 to +45 C | -20 to +70 C |
| Ramp rate | 5 C/min | 10 C/min |
| Soak time at extreme | 15 min | 30 min |
| Cycles | 100 | 500 |

### Soak testing

- Run the production workload (AI inference + OTA checks + telemetry) continuously for **7–14 days**
- Monitor for memory leaks, file descriptor leaks, disk space growth, temperature drift
- Log all metrics and compare day 1 vs day 14

### HALT (Highly Accelerated Life Test)

- Push beyond spec limits (temperature, vibration, voltage) to find design margins
- Not a pass/fail test — it reveals **where** the design breaks
- Performed on 5–10 units early in the design cycle

---

## 18. Projects

- **Secure boot lab:** Generate test PKC/SBK keys, sign a full JetPack image, flash a dev kit with secure boot enabled, then verify that unsigned images are rejected.
- **End-to-end OTA pipeline:** Set up SWUpdate on a Jetson Orin Nano with A/B rootfs, a signing server, and a staging server. Push an update, verify rollback on intentional failure.
- **Power-cycle soak:** Build a test jig with a relay-controlled power supply, run 1000 power-cycle boot tests, log results, and analyze any failures.
- **Fleet dashboard:** Deploy 3+ Jetson devices, set up MQTT telemetry + Grafana dashboard, monitor boot count, temperature, and OTA status across the fleet.

---

## 19. Resources

| Resource | Description |
|----------|-------------|
| **NVIDIA Jetson Security Guide** | Official secure boot, fuse programming, encryption documentation |
| [Orin Nano Security deep dive](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Security/Guide.md) | Detailed PKC/SBK walkthrough, threat model (Module 1 companion) |
| [Orin Nano OTA deep dive](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-OTA-Deep-Dive/Guide.md) | `nv_update_engine`, slot management internals (Module 1 companion) |
| [Orin Nano Rootfs and A/B Redundancy](../1.%20Nvidia%20Jetson%20Platform/Orin-Nano-Rootfs-and-AB-Redundancy/Guide.md) | Partition layout, `nvbootctrl` (Module 1 companion) |
| **SWUpdate** (sbabic.github.io/swupdate) | Open-source OTA framework |
| **Mender** (mender.io) | Managed OTA platform with Jetson support |
| **RAUC** (rauc.io) | Robust Auto-Update Controller |
| **OP-TEE** (optee.org) | Open-source Trusted Execution Environment |
