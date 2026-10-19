# Manufacturing and compliance: make each result traceable

**Outcome:** produce a manufacturing-readiness pack that links product configuration, applicable requirements, fixture qualification, per-unit measurements, software identity and field support. **Pace:** five 60–90-minute sessions. Start with the [SQLite factory-record lab](labs/README.md), which runs on a CPU with Python's standard library.

A board that boots is an engineering observation. A shippable product also needs a repeatable build, defined acceptance limits, applicable conformity evidence and a way to identify and support the exact units shipped. This module teaches how those records fit together.

**Regulatory snapshot reviewed 18 October 2026.** Applicability depends on intended use, markets, power, radios and product category. Use the official sources below to build a product-specific requirements register with the responsible compliance specialist/laboratory. This chapter is an engineering planning exercise, not a certification decision for an unspecified product.

## 1. Define the configuration being qualified

Freeze a configuration identity for each qualification run: module SKU/revision, carrier revision, BOM variants, enclosure, cables, power supply, radios/antennas, firmware, kernel, application/model and relevant operating modes. A test report for a bare board with one cable does not automatically cover a different enclosure and cable harness.

Separate design qualification from production screening. Qualification investigates the product's specified operating range and failure behavior. Production screening checks each manufactured unit for defined defects using an efficient fixture. A one-frame camera test can detect a disconnected cable; it does not establish long-term frame integrity, EMC robustness or perception quality.

For a robot, include sensor calibration identity and controller/actuator configuration. A valid AI model paired with the wrong camera transform or action units is a different system from the evaluated one.

## 2. Build a requirements register, not a universal badge list

Use one row per requirement with its source/revision, applicability rationale, evidence owner, tested configuration and status. “Not applicable” needs a reason; an empty cell is not evidence.

| Topic | Questions for the product-specific review |
| --- | --- |
| US equipment authorization | Which Part 2/15 route applies, including SDoC/certification and any intentional-radio conditions? |
| EU conformity/CE | Which product legislation applies, what assessment route is required, and which technical documentation/declarations are needed? |
| Radio integration | Are antenna, enclosure, simultaneous-transmission and host integration conditions satisfied? |
| Electrical/mechanical safety | Which use environment, supply, accessible energy and product-category requirements apply? |
| Materials/end-of-life | Which substance restrictions, labeling and producer obligations apply in each market? |
| Product cybersecurity | Which lifecycle, vulnerability handling, reporting and update duties apply? |
| Robot application | Does the intended machinery, industrial, vehicle, medical or other use introduce additional requirements? |

FCC equipment authorization includes SDoC and certification procedures; they are not interchangeable names for one “FCC certificate.” A transmitter module's grant and integration conditions also do not automatically authorize every completed host configuration. Use the [FCC equipment-authorization guidance database](https://apps.fcc.gov/oetcf/kdb/index.cfm) and [module-integration guidance](https://apps.fcc.gov/oetcf/kdb/forms/FTSSearchResultPage.cfm?id=44637&switch=P).

CE marking expresses conformity with applicable EU legislation; it is not a single test purchased from a laboratory. Not every electronic item has the same assessment route or set of directives. Retain the applicable technical documentation and declaration for the actual product. [European Commission CE guidance](https://single-market-economy.ec.europa.eu/single-market/goods/ce-marking_en).

Do not state that UKCA universally replaced CE in Great Britain. Current UK guidance continues to recognize CE for covered product regulations, with sector-specific rules; Northern Ireland has a different framework. Record the exact market and applicable product regime. [UK government guidance](https://www.gov.uk/guidance/placing-ukca-or-ce-marked-products-on-the-market-in-great-britain).

## 3. Include current cybersecurity obligations

The EU Cyber Resilience Act's reporting obligations began on **11 September 2026**; its main obligations apply from **11 December 2027**. The Commission published implementation guidance in July 2026. These dates matter to a course reviewed in October 2026: reporting readiness is not solely a future consideration. Determine scope and responsibilities for the particular product and organization. [European Commission CRA overview](https://digital-strategy.ec.europa.eu/en/policies/cyber-resilience-act).

For relevant radio equipment, the RED cybersecurity delegated requirements began applying in August 2025. Review the current transition/interaction with the CRA and the applicable consolidated sources rather than assuming all wireless devices have one identical obligation. [Commission RED guidance](https://single-market-economy.ec.europa.eu/sectors/electrical-and-electronic-engineering-industries-eei/radio-equipment-directive-red_en).

An engineering readiness exercise should identify the shipped dependency/artifact inventory, vulnerability intake owner, supported update mechanism, support-period assumptions and a way to locate affected units. The [security/OTA module](../6.%20Security%20and%20OTA/Guide.md) supplies the artifact, trial and recovery exercises. A successful signature check is one useful control, not a complete regulatory assessment.

## 4. Plan EMC work around a reproducible configuration

Begin with representative workload modes and cable arrangements. Identify potential emissions paths, return-current discontinuities, switching nodes and external interfaces. Change one factor at a time and record the effect.

A near-field probe is useful for locating a noisy region. Its reading is not automatically a calibrated far-field measurement that can be compared directly to a regulatory limit. Likewise, an inexpensive SDR can help observe a signal without replacing the detector, bandwidth, calibration, site and procedure required by an applicable test.

Select test levels, detectors, bandwidths and pass criteria from the actual applicable standards/assessment plan. Avoid universal “±8 kV ESD” or “3 V/m immunity” pass statements. Product category, ports, environment and performance criteria affect the plan.

Retain both electrical measurements and functional observations. For a robot, define what acceptable behavior means during and after a disturbance: lost frames, stale commands, reboot, recovery, or uncontrolled output have different implications. Do not silently count an automatic restart as uninterrupted service.

Obtain current laboratory quotations and schedules for the stated configuration. This course does not assign a universal certification cost, six-week schedule or fixed number of retests. Board changes and failed tests can alter both scope and timing.

## 5. Design for assembly and test access

Review component footprints, assembly clearances, polarity, connector seating, mounting loads, thermal interfaces and test-point access with the selected manufacturer. Export fabrication/assembly data from the same released design revision and verify the manufacturer can interpret them. An EDA import or a clean ERC/DRC is useful but does not prove electrical correctness or manufacturability.

Create a test-access drawing showing measurement reference points and fixture connections. Distinguish ground references, power injection, communication loopbacks and debug access. Keep the module insertion/removal procedure consistent with the hardware specification; do not insert it into an energized carrier as an ordinary production-test step.

**Coverage exercise:** a local Ethernet loopback may verify part of a PHY path while bypassing the cable/connector fault you intended to detect. For each test, write “fault detected,” “path exercised,” and “faults not covered.” That prevents a long list of passing tests from becoming an unjustified coverage claim.

## 6. Qualify the fixture before judging the device

A measurement is meaningful only when the fixture and instrument are functioning under the stated calibration/configuration. Include a reference measurement or other appropriate fixture check. Separate these outcomes:

| Outcome | Meaning | Treatment |
| --- | --- | --- |
| Qualified pass | Valid measurements meet the selected limits | Eligible for the next acceptance step |
| Qualified DUT failure | Valid measurements show an out-of-limit device result | Retain failure and route to investigation/rework |
| Unqualified station result | Wrong units, expired calibration, failed fixture reference or malformed evidence | Quarantine/retest; do not call it a valid DUT pass or failure |
| Missing test | Required evidence absent | Acceptance incomplete |

The [CPU lab](labs/README.md) enforces that separation. Its voltage limits are explicitly fictional teaching values, not Jetson electrical specifications. It rejects a bad fixture reference before inserting a DUT result, and tests that a corrected station can submit the same still-unused attempt identity.

Choose guard bands using the measurement process and required decision risk. For an illustrative specification of 5.00 ±0.10 V with a simplified ±0.02 V uncertainty bound, an acceptance interval of 4.92–5.08 V keeps that bound inside 4.90–5.10 V. This is a worked bounded-error example; a real uncertainty budget and decision rule need the appropriate method, distributions and agreement.

## 7. Preserve attempts and calculate honest yield

Do not overwrite a failed attempt when the same unit passes after rework. Store serial, attempt identity, fixture/station, limit revision, image digest, measurement values/units, calibration context and timestamp.

**Worked example:** unit A fails and then passes; unit B passes on its first attempt. There are two units and three qualified attempts. First-pass yield is `1 / 2 = 50%`; latest-result pass fraction is `2 / 2 = 100%`. They share the unit denominator and answer different questions. The fraction of passing attempts, `2 / 3`, is a third metric. Label the metric you report.

Repeated upload of the same attempt should not create another manufactured unit or inflate yield. A changed payload under the same attempt identity should be rejected or handled through an explicit correction process with retained history. The lab uses a real SQLite transaction and uniqueness constraint for this behavior, and interrupts a child process before commit to test recovery.

A local database digest identifies the stored payload; it does not authenticate a hostile station or prevent an administrator changing the database. A deployed traceability service needs identity, authorization, retention, backup and an appropriate audit design. State those remaining boundaries rather than presenting the teaching ledger as a complete factory system.

## 8. Separate flashing from acceptance

Select the host OS, BSP, board configuration, storage layout and secure-state procedure supported by the exact release. NVIDIA's initrd mass-flash workflow separates preparation of a package for a maximum device count from actual concurrent flashing; `--massflash N` is not simply a request to create N independent image copies. Use the documented generation and flash-only stages for the matching BSP. [NVIDIA flashing support](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/FlashingSupport.html).

A preloaded NVMe does not prove that QSPI firmware, board configuration, device identity or encryption/provisioning state is correct. Avoid a “zero-minute flashing” acceptance claim that ignores the rest of the system.

Identify each device explicitly at the station, then record the immutable artifact identity and observed post-flash state. Keep secrets out of ordinary logs. Serial numbers and MAC addresses must follow the product's approved allocation/provisioning process; do not rewrite a module EEPROM or invent globally administered MAC addresses as a generic test step.

Define boot/application checks from the product requirements, not fixed universal thresholds such as “NVMe over 2 GB/s,” “idle below 50°C,” or “all tests under 60 seconds.” A storage SKU, power mode, ambient temperature or interface width can change a valid range. Verify the intended path with representative workloads and bounded failure handling.

## 9. Plan capacity, change control and support

**Station-capacity example:** at 90 seconds of occupied station time, one fixture has a theoretical 40 units/hour. At 80% planned availability/utilization, plan 32 units/hour before rework and bottlenecks. Four fixtures give a theoretical 128 units/hour under those assumptions only if loading, power, network, flashing host and staffing support it. Measure the complete cycle rather than using the image-write interval alone.

Track module/component revisions and product-change notices. A pin-compatible alternate may differ in firmware, timing, temperature behavior or radio authorization conditions. Qualify the affected requirements and release the revised BOM/software combination together.

For field support, link the returned unit to its original configuration and all qualified attempts. Capture the failure mode, reproduction conditions, software/model identity and corrective action. Distinguish no-fault-found, fixture-related and confirmed product failures. Use that information to improve screening without changing limits merely to improve the dashboard's yield.

## 10. Submit a manufacturing-readiness pack

Your submission contains a configuration manifest, requirements register with current primary sources, test-coverage table, fixture qualification procedure, example per-unit record, yield calculation, flashing/acceptance flow and change-control plan. Include the CPU ledger test log and the failure-before-commit result.

**Self-checks:**

- A fixture reference fails but the DUT voltage looks normal: quarantine the station result.
- A failed unit later passes: retain both attempts and preserve first-pass yield.
- The same record arrives four times: record one attempt and recognize identical replays.
- A near-field scan looks quiet: use it as debugging evidence, not a formal conformity verdict.
- A radio module is approved: still evaluate its integration conditions and complete host obligations.
- A model update changes sensor preprocessing: re-evaluate the deployed system's relevant behavior and trace the new artifact identity.
