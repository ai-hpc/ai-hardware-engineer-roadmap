# Compliance and Manufacturing

<div class="course-identity compliance" markdown="1">
<div class="course-identity__icon">MFG</div>
<div markdown="1">
<p class="course-identity__eyebrow">Track B7 · Compliance & Manufacturing</p>
<p class="course-identity__title">Prepare embedded AI hardware for certification, test fixtures, production, and field reliability.</p>
<p class="course-identity__meta">Artifact: manufacturing readiness pack · Measure: test coverage, yield risks, compliance gaps</p>
</div>
</div>


**Phase 4 — Track B — Nvidia Jetson** · Module 7 of 7

> **Focus:** Take a validated **Jetson Orin Nano 8GB** product from engineering prototype through **regulatory certification** (FCC/CE/IC), **DFM review**, **production flashing infrastructure**, **supply-chain management**, and **fleet operations** — the last mile before volume shipping and field sustaining.
>
> **Primary hardware:** Jetson Orin Nano 8GB on custom carrier (from Module 2)

**Previous:** [6. Security and OTA](../6.%20Security%20and%20OTA/Guide.md)

---


## 1. The last mile — why this module exists

Engineering prototypes work on the bench. Shipping products must also be:

- **Legal** — pass regulatory testing (FCC, CE) before you can sell
- **Manufacturable** — a contract manufacturer can build them repeatably at scale
- **Supportable** — you can update, monitor, and repair devices in the field

This module covers everything between "board works" and "product in customer hands."

---

## 2. Regulatory landscape for Jetson products

### What applies to your product

| Regulation | Region | Applies when |
|-----------|--------|-------------|
| **FCC Part 15 Subpart B** | USA | Any digital device marketed in the US |
| **CE (EMC Directive)** | EU | Any electronic product sold in the EU |
| **IC (ISED)** | Canada | Any digital device marketed in Canada |
| **UKCA** | UK | Post-Brexit equivalent of CE |
| **RCM** | Australia/NZ | Any electrical product sold in AU/NZ |
| **MIC / VCCI** | Japan | Any digital device marketed in Japan |

### Modular vs system-level certification

The Jetson Orin Nano **module** may have its own EMC characterization data from NVIDIA, but your **system** (module + custom carrier + enclosure + cables) requires **system-level testing**. The enclosure, cable routing, and carrier board design all affect emissions.

**You almost always need system-level FCC/CE testing** for your complete product.

---

## 3. FCC certification (unintentional radiator)

Most Jetson products without built-in wireless are **unintentional radiators** under FCC Part 15 Subpart B.

### Testing categories

| Test | Standard | What it measures |
|------|----------|-----------------|
| **Radiated emissions** | ANSI C63.4 | RF energy radiated from the device and cables |
| **Conducted emissions** | CISPR 32 / FCC Part 15 | RF noise conducted back onto the power line |

### Common emission sources on Jetson carriers

| Source | Frequency range | Typical fix |
|--------|----------------|-------------|
| **USB 3.0 super-speed** | 2.4–5 GHz spread spectrum | Shielded cables, common-mode chokes, spread-spectrum clocking |
| **HDMI/DP clock** | Harmonics of pixel clock | Shielded connector, ferrite on cable |
| **Switching regulators** | 100 kHz–30 MHz | Proper layout, input/output filtering, shielding |
| **High-speed PCIe** | 2.5–16 GHz (Gen 1–4) | Proper termination, short traces, grounded enclosure |
| **Ethernet** | 100 MHz–1 GHz | Correct magnetics, proper PHY layout |

### Process timeline

| Step | Duration | Notes |
|------|----------|-------|
| Pre-compliance scan (in-house) | 1–2 weeks | Identify issues before paying for lab time |
| Fix identified issues | 1–4 weeks | Layout changes, filtering, shielding |
| Formal lab testing | 1–2 weeks | At an accredited test lab (A2LA, NVLAP) |
| Report and FCC filing | 2–4 weeks | SDoC (Supplier's Declaration of Conformity) for unintentional radiators |
| **Total** | **6–12 weeks** | Budget for at least one re-test cycle |

### Cost estimate

| Item | Cost range |
|------|-----------|
| Pre-compliance scan (rental or lab) | $500–$2,000 |
| Formal FCC Part 15B test + report | $3,000–$8,000 |
| Re-test (if fail + fix) | $2,000–$5,000 |

---

## 4. CE marking (EU)

CE marking requires compliance with multiple directives:

| Directive | Standard | Applies to |
|-----------|----------|-----------|
| **EMC Directive (2014/30/EU)** | EN 55032 (emissions), EN 55035 (immunity) | All electronic products |
| **LVD (2014/35/EU)** | EN 62368-1 | Products operating 50–1000 VAC or 75–1500 VDC |
| **RED (2014/53/EU)** | EN 300 328, etc. | Products with intentional radio (WiFi, BT, cellular) |
| **RoHS (2011/65/EU)** | — | Restriction of hazardous substances in electronics |

### Key differences from FCC

- CE includes **immunity testing** (ESD, surge, conducted immunity, radiated immunity) — FCC does not
- CE requires a **Declaration of Conformity (DoC)** — you self-declare
- If you add WiFi/BT/cellular → RED applies, which adds radio-specific testing

### Immunity tests (commonly failed)

| Test | Standard | Common failure mode |
|------|----------|-------------------|
| **ESD** (±8 kV contact, ±15 kV air) | EN 61000-4-2 | USB ports, exposed connectors, metal enclosure |
| **Radiated immunity** (3 V/m) | EN 61000-4-3 | Unshielded cables act as antennas |
| **EFT (Electrical Fast Transient)** | EN 61000-4-4 | Power supply input |
| **Surge** | EN 61000-4-5 | Power input, Ethernet |

---

## 5. Other markets (IC, MIC, UKCA, RCM)

### Prioritization strategy

- **Launch with FCC + CE** — covers USA + EU, the two largest markets
- **Add IC (Canada)** — often bundled with FCC testing (same lab, same trip)
- **Add others** as market demand requires

| Certification | Effort beyond FCC+CE | Notes |
|--------------|---------------------|-------|
| **IC (ISED)** | Minimal — same tests, different filing | Often done simultaneously with FCC |
| **UKCA** | Similar to CE, separate declaration | Post-Brexit requirement |
| **RCM** | Based on CISPR 32 (similar to EN 55032) | Register with ACMA |
| **MIC/VCCI** | VCCI is voluntary self-declaration | Based on CISPR 32 |

---

## 6. Pre-compliance testing

Catch emissions problems **before** paying for formal lab time.

### Minimum equipment

| Equipment | Cost | Purpose |
|-----------|------|---------|
| **Near-field probe set** | $200–$500 | Localize emission sources on the PCB |
| **Spectrum analyzer** (or SDR like HackRF) | $300–$2,000 | View spectrum of emissions |
| **Current probe** (RF) | $100–$300 | Measure conducted emissions on cables |

### Pre-scan workflow

1. Set up the device in its intended configuration (all cables, enclosure)
2. Run worst-case workload (AI inference + USB + Ethernet active)
3. Scan with near-field probes to identify hot spots on the PCB
4. Use spectrum analyzer to measure approximate field strength
5. Compare against FCC/CISPR limits with margin
6. Iterate: add filtering, shielding, or layout fixes

---

## 7. DFM review and design-for-assembly

### DFM checklist for Jetson carrier boards

| Item | Check |
|------|-------|
| **Panelization** | Board fits standard panel sizes for your CM's pick-and-place |
| **Fiducials** | Global and local fiducials placed per IPC-7351 |
| **Solder paste stencil** | Aperture reductions for fine-pitch parts (SoM connector 0.5 mm pitch) |
| **Component orientation** | All polarized components oriented consistently for inspection |
| **Pick-and-place coordinates** | Centroid file generated and verified |
| **Reflow profile** | Validated for the SoM connector and all components (lead-free SAC305) |
| **Test access** | ICT test points on bottom side, bed-of-nails accessible |
| **Conformal coating** | If required (industrial/outdoor), mask keep-out for connectors |
| **Assembly sequence** | SoM connector → SMT → through-hole → SoM insertion → mechanical |

### Design-for-assembly (DFA)

- Minimize the number of unique screw types
- Use snap-fit or tool-free assembly where possible
- Design the enclosure so the PCB drops in from one direction
- Label test points and debug headers on the silkscreen

---

## 8. Production flashing infrastructure

### Flash station design

```
Flash station:
  ├─ Host PC (Ubuntu 22.04, 32 GB RAM, NVMe SSD)
  │     └─ Linux_for_Tegra + signed images
  │
  ├─ USB hub (powered, USB 3.0)
  │     └─ 1–4 USB recovery cables to Jetson devices
  │
  ├─ Power supply (one per device, or multi-output)
  │
  └─ Flash script (automated: flash + validate + log serial number)
```

### Batch flashing with `l4t_initrd_flash.sh`

```bash
# Flash external NVMe on multiple devices
sudo ./tools/kernel_flash/l4t_initrd_flash.sh \
    --massflash 4 \
    <board_config> \
    external
```

`--massflash N` creates N flash images that can be applied in parallel to N devices simultaneously.

### Flash time optimization

| Approach | Flash time (per device) |
|----------|----------------------|
| Standard `flash.sh` (USB 2.0) | 15–25 min |
| `l4t_initrd_flash.sh` (USB 3.0) | 8–15 min |
| Mass-cloned image (write raw image to NVMe) | 3–5 min |
| Pre-flashed NVMe (SSD pre-loaded by supplier) | 0 min (on-line) |

### Factory image management

- **Golden master:** A signed, tested image tagged with version and build date
- **Image signing:** Part of CI/CD (see [Module 7 — Security and OTA](../6.%20Security%20and%20OTA/Guide.md))
- **Version tracking:** Flash script logs `{serial_number, image_version, timestamp, pass/fail}` to a database

---

## 9. Factory test fixtures and procedures

### Test jig design

For carrier boards with custom connectors, build a **bed-of-nails** test jig:

- **Pogo pins** contact test points on the bottom of the PCB
- **Pneumatic or manual press** holds the board in contact
- **Test controller** (Raspberry Pi, STM32, or host PC) runs automated test sequence

### Automated test sequence

```
1. Apply power → verify rails (pass/fail per rail)
2. Insert SoM module (or pre-inserted)
3. Boot → wait for UART "login:" prompt (timeout = 60s)
4. Run test script on target:
   a. USB: enumerate test device
   b. Ethernet: ping gateway, iperf short burst
   c. NVMe: read/write test
   d. CSI: capture one frame (if camera connected)
   e. I2C: scan for expected device addresses
   f. CAN: send/receive loopback frame
   g. GPIO: toggle test pins, read back
   h. Temperature: read thermal zone (sanity check)
5. Program serial number and MAC address to EEPROM
6. Log results to database
7. Print pass/fail label

Total time target: < 60 seconds per unit
```

### Pass/fail criteria

Define **quantitative thresholds** for every test:

| Test | Pass criteria |
|------|-------------|
| 5V rail | 4.85–5.15 V |
| Ethernet throughput | > 900 Mbps |
| NVMe sequential read | > 2 GB/s |
| Boot time | < 30 s to login prompt |
| GPU temperature at idle | < 50 C |

---

## 10. Supply chain management

### Jetson module procurement

| Source | Lead time | MOQ | Notes |
|--------|-----------|-----|-------|
| **NVIDIA direct** | 12–16 weeks | Varies | For large volumes, requires NVIDIA account |
| **Arrow / Avnet** | 8–16 weeks | 1–100+ | Authorized distributors |
| **Mouser / Digikey** | Stock or 8–12 weeks | 1+ | For prototyping and small runs |

### Critical component tracking

Maintain a **risk register** for components:

| Risk level | Component | Mitigation |
|-----------|-----------|------------|
| **High** | SoM connector (Molex, single source) | Buffer stock (3+ months) |
| **High** | Jetson Orin Nano module | Long lead time — order early |
| **Medium** | Ethernet PHY | Second-source qualified alternate |
| **Low** | Passive components | Multiple manufacturers, standard values |

### Managing SoM revision changes

NVIDIA periodically releases updated module revisions. Your carrier and BSP must be validated against each new revision:

1. Monitor NVIDIA **Product Change Notifications (PCN)**
2. When a new revision is announced, order samples
3. Re-run board bring-up validation (Module 4) and BSP tests
4. Update BSP if device tree or driver changes are required
5. Qualify and release updated firmware before accepting new-revision modules in production

---

## 11. Configuration management and traceability

### Serial number scheme

```
JET-<year><week>-<sequence>
Example: JET-2626-00142
         │    │     │
         │    │     └─ Unit 142
         │    └─ Year 2026, week 26
         └─ Product prefix (example)
```

### What to track per device

| Field | Source | Stored in |
|-------|--------|-----------|
| Serial number | Assigned during factory test | EEPROM + database |
| MAC address(es) | Assigned from allocated range | EEPROM + database |
| SoM serial number | Read from module EEPROM | Database |
| SoM revision | Read from module | Database |
| Firmware version | Flashed image version tag | Database + device |
| Factory test result | Test script output | Database |
| Ship date | Fulfillment system | Database |

### Build artifact management

- Tag every release image in git: `v1.0.0-rc1`, `v1.0.0`
- Store signed images in a versioned artifact repository (S3, Artifactory, or local NAS)
- Never overwrite — append new versions
- Keep build logs and test reports alongside the image

---

## 12. Fleet management and field support

### Remote access

| Method | Use case | Security |
|--------|----------|----------|
| **SSH over VPN** (WireGuard, Tailscale) | Debug, log collection, manual intervention | Strong (encrypted tunnel, key-based) |
| **Reverse SSH tunnel** | Devices behind NAT, no VPN infrastructure | Medium (requires jump server) |
| **MQTT telemetry** | Heartbeat, metrics, light commands | Medium (TLS + auth) |
| **OTA agent** | Firmware and configuration updates | Strong (signed images, TLS) |

### Fleet dashboard

Connect to the telemetry stack from [Module 7 — Security and OTA](../6.%20Security%20and%20OTA/Guide.md):

- **Device inventory:** Online/offline status, firmware version, last check-in
- **Health metrics:** CPU/GPU temperature, disk usage, memory usage, uptime
- **OTA status:** Current version per device, pending updates, rollback events
- **Alerts:** Offline devices, reboot loops, temperature alarms, disk full

### Firmware update cadence

| Type | Frequency | Trigger |
|------|-----------|---------|
| **Security patches** | As needed (ASAP) | CVE in kernel, L4T, or application dependency |
| **Feature updates** | Monthly or quarterly | Product roadmap |
| **Emergency hotfix** | Immediate | Critical bug affecting field devices |

---

## 13. RMA and failure analysis

### RMA process

```
Customer reports issue
  │
  ├─ Remote triage (logs, telemetry, OTA status check)
  │     └─ Software fix? → Push OTA update, close
  │
  ├─ Hardware suspected → Issue RMA authorization
  │     └─ Customer ships device back
  │
  ├─ Incoming inspection
  │     ├─ Visual inspection (physical damage, corrosion, burnt components)
  │     ├─ Re-run factory test sequence
  │     └─ Compare with original factory test log
  │
  ├─ Fault isolation
  │     ├─ Swap SoM → carrier issue
  │     ├─ Swap carrier → SoM issue
  │     └─ Neither → environmental or software root cause
  │
  └─ Resolution
        ├─ Repair and return
        ├─ Replace with new unit
        └─ Update design if systemic issue
```

### Failure tracking

Track all failures in a database with categories:

| Category | Example | Action |
|----------|---------|--------|
| **DOA (Dead on Arrival)** | Unit never booted at customer site | Improve factory test coverage |
| **Infant mortality** | Fails within first 30 days | Possible solder defect — review reflow profile |
| **Wear-out** | Fan failure after 18 months | Spec higher-MTBF fan, or add fan monitoring |
| **Environmental** | Corrosion on exposed connector | Add conformal coating or sealed connector |
| **Software** | OTA bricked device | Improve rollback mechanism (Module 7) |

---

## 14. Projects

- **Production flash station:** Build a flash station that can flash 4 Jetson Orin Nano units in parallel using `--massflash`. Time the process and optimize.
- **Factory test script:** Write an automated test script that validates all peripherals on your custom carrier in under 60 seconds. Output a structured JSON pass/fail report.
- **Pre-compliance scan:** Using a near-field probe and spectrum analyzer (or SDR), scan your carrier board running a worst-case AI workload. Document the top 3 emission sources and proposed mitigations.
- **Fleet dashboard:** Deploy 3+ units with unique serial numbers, set up a Grafana dashboard showing device health, OTA version, and uptime. Simulate an OTA rollout to all devices.

---

## 15. Resources

| Resource | Description |
|----------|-------------|
| **FCC Equipment Authorization** | FCC.gov guide to Part 15 certification process |
| **CISPR 32 (EN 55032)** | International standard for multimedia equipment emissions |
| **EN 62368-1** | Safety standard for audio/video, IT, and communication equipment |
| **IPC-A-610** | Acceptability of electronic assemblies (workmanship standard) |
| **IPC-7711/7721** | Rework, modification, and repair of electronic assemblies |
| **NVIDIA Jetson Partner Hardware Design** | OEM Design Guide, reference design files, certification notes |
| [2. Custom Carrier Board Design](../2.%20Custom%20Carrier%20Board%20Design%20and%20Bring-Up/Guide.md) | Hardware design (this module assumes carrier is built and validated) |
| [7. Security and OTA](../6.%20Security%20and%20OTA/Guide.md) | Signed images, fleet telemetry (feeds into factory flashing and fleet ops) |
